"""Subprocess-isolation infrastructure for the file parser.

Each file is parsed in a **fully isolated** ``multiprocessing.Process``
(one process per file, one private ``Queue`` per process).  A
``ThreadPoolExecutor`` in the parent controls concurrency so that at most
*N* files parse simultaneously.

This design gives true per-file crash isolation: if one child is
OOM-killed, only that file's result is an error — all other concurrent
files continue unaffected.  The previous ``ProcessPoolExecutor`` approach
shared an internal IPC channel between all workers; a single worker crash
(``SIGKILL`` from the OOM killer) broke that channel, cascading
``BrokenProcessPool`` to every pending future in the same batch.

Forkserver context
    On Linux, ``forkserver`` is used so that child processes inherit
    pre-imported Docling modules through copy-on-write, making startup
    nearly instant even though each child handles exactly one file.

Memory limits
    ``RLIMIT_AS`` is intentionally **not** applied.  Docling loads ONNX
    model weights via ``mmap(2)`` which inflates virtual address space
    well beyond actual physical usage; a VAS cap causes spurious
    ``ENOMEM`` failures on files that would parse successfully with the
    available physical RAM.  Per-file process isolation already contains
    genuine OOM events — if a child is killed by the kernel OOM reaper,
    only that file's result is an error.

All worker-side functions are **top-level** (not methods) so they remain
picklable for ``multiprocessing.Process(target=...)``.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from unity.file_manager.file_parsers.types.contracts import (
        FileParseRequest,
        FileParseResult,
    )
    from unity.file_manager.types.config import ParseConfig

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

# ---------------------------------------------------------------------------
# Forkserver pre-import
# ---------------------------------------------------------------------------

_FORKSERVER_PRELOADED = False


def ensure_forkserver_preload() -> None:
    """Configure the forkserver to pre-import Docling and its heavy deps.

    Must be called **before** the first ``ProcessPoolExecutor`` using
    the ``forkserver`` context is created.  Subsequent calls are no-ops.

    On Windows the forkserver start method does not exist, so this
    function is a silent no-op.  ``RuntimeError`` (raised when the
    forkserver is already running on Unix) is also caught gracefully.
    """
    global _FORKSERVER_PRELOADED
    if _FORKSERVER_PRELOADED or _IS_WINDOWS:
        return
    try:
        multiprocessing.set_forkserver_preload(
            [
                "docling",
                "docling.document_converter",
                "docling.datamodel.pipeline_options",
                "docling.datamodel.base_models",
                "polars",
            ],
        )
    except (RuntimeError, ValueError):
        pass
    _FORKSERVER_PRELOADED = True


# ---------------------------------------------------------------------------
# Top-level subprocess entry point (must be picklable)
# ---------------------------------------------------------------------------


def subprocess_parse_single(
    request: "FileParseRequest",
    parse_config: Optional["ParseConfig"],
) -> "FileParseResult":
    """Parse a single file inside an isolated child process.

    Creates a fresh ``FileParser`` so that all Docling / openpyxl memory
    is fully reclaimed when the child exits.

    ``flush_writer()`` is called explicitly before returning because
    ``ProcessPoolExecutor`` workers may not run ``atexit`` handlers
    reliably when ``max_tasks_per_child=1`` causes the child to exit.
    """
    from pathlib import Path

    from unity.file_manager.file_parsers.file_parser import FileParser
    from unity.file_manager.file_parsers.registry import BackendRegistry
    from unity.file_manager.file_parsers.types.formats import (
        FileFormat,
        extension_to_format,
    )
    from unity.file_manager.file_parsers.utils.diagnostics import flush_writer

    parser = FileParser()
    registry = (
        BackendRegistry.from_config(
            backend_class_paths_by_format=getattr(
                parse_config,
                "backend_class_paths_by_format",
                None,
            ),
        )
        if parse_config is not None
        else parser._default_registry
    )

    source_path = Path(request.source_local_path).expanduser().resolve()
    fmt = request.file_format or extension_to_format(source_path.suffix)
    if fmt == FileFormat.UNKNOWN:
        fmt = FileFormat.TXT

    parser._pick_backend(fmt, registry=registry)
    result = parser._parse_single(request, registry=registry)
    flush_writer()
    return result


# ---------------------------------------------------------------------------
# Isolated per-file process runner
# ---------------------------------------------------------------------------


def _mp_context() -> multiprocessing.context.BaseContext:
    """Return the multiprocessing context (forkserver on Unix, spawn on Windows)."""
    if _IS_WINDOWS:
        return multiprocessing.get_context("spawn")
    ensure_forkserver_preload()
    return multiprocessing.get_context("forkserver")


def _child_worker(
    request: "FileParseRequest",
    parse_config: Optional["ParseConfig"],
    result_queue: multiprocessing.Queue,  # type: ignore[type-arg]
) -> None:
    """Entry point for an isolated child process.

    Runs ``subprocess_parse_single`` and puts the result (or an error
    sentinel) into *result_queue*.  This function **must** be top-level
    (not a closure or lambda) so it is picklable by the forkserver.

    Each child handles exactly one file.  ``subprocess_parse_single``
    lazily imports only the backend required for the file's format via
    ``_pick_backend``.  On Linux the forkserver preload gives CoW
    copies of heavy modules (Docling, Polars) to every child.

    If ``Queue.put()`` itself fails (e.g. ``RuntimeError: can't start
    new thread`` under resource exhaustion), the child exits with code 2
    so the parent can detect the crash via ``proc.exitcode``.
    """
    try:
        result = subprocess_parse_single(request, parse_config)
        result_queue.put(result)
    except Exception as exc:
        try:
            result_queue.put(exc)
        except Exception:
            sys.exit(2)


def run_isolated(
    request: "FileParseRequest",
    parse_config: Optional["ParseConfig"],
    timeout: float,
) -> "FileParseResult":
    """Parse a single file in a fully isolated process.

    Spawns one ``multiprocessing.Process`` with its own ``Queue``.  If the
    child crashes (OOM, segfault) or exceeds *timeout*, only this file is
    affected — no shared pool state is corrupted.

    Returns a ``FileParseResult`` in all cases (success, crash, or timeout).

    Implementation note: the queue **must** be drained before calling
    ``proc.join()``.  ``Queue.put()`` writes to an OS pipe; if the
    serialized result exceeds the pipe buffer (~64 KB on Linux), ``put``
    blocks until the reader drains the pipe.  Calling ``join`` first
    would deadlock: the parent waits for the child to exit while the
    child waits for the parent to read.
    """
    from queue import Empty

    from unity.file_manager.file_parsers.types.contracts import FileParseResult

    ctx = _mp_context()
    result_queue: multiprocessing.Queue = ctx.Queue()  # type: ignore[type-arg]
    proc = ctx.Process(
        target=_child_worker,
        args=(request, parse_config, result_queue),
    )
    proc.start()

    # Drain the queue FIRST (with timeout), then join the process.
    result = None
    try:
        result = result_queue.get(timeout=timeout)
    except Empty:
        pass

    # Give the child a few seconds to exit after producing its result.
    proc.join(timeout=5)

    if proc.is_alive():
        logger.error(
            "Subprocess parse timed out after %.0fs for %s — killing child",
            timeout,
            request.logical_path,
        )
        _kill_process(proc)
        return FileParseResult(
            logical_path=request.logical_path,
            status="error",
            error=f"subprocess_parse_timeout: exceeded {timeout}s",
        )

    if result is None:
        if proc.exitcode != 0:
            logger.error(
                "Subprocess crashed (exit %s) while parsing %s",
                proc.exitcode,
                request.logical_path,
            )
            return FileParseResult(
                logical_path=request.logical_path,
                status="error",
                error=(f"subprocess_crash: child exited with code {proc.exitcode}"),
            )
        return FileParseResult(
            logical_path=request.logical_path,
            status="error",
            error="subprocess_no_result: child exited cleanly but produced no result",
        )

    if isinstance(result, Exception):
        proc.join(timeout=5)
        return FileParseResult(
            logical_path=request.logical_path,
            status="error",
            error=f"subprocess_parse_exception: {result}",
        )

    proc.join(timeout=5)
    return result


def _kill_process(proc: multiprocessing.Process) -> None:
    """Force-kill a process and wait for it to exit."""
    kill_sig = getattr(signal, "SIGKILL", signal.SIGTERM)
    try:
        if proc.pid is not None:
            os.kill(proc.pid, kill_sig)
    except (OSError, ProcessLookupError):
        pass
    proc.join(timeout=5)
