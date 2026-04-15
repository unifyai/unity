"""Subprocess-isolation infrastructure for the file parser.

This module manages the warm ``ProcessPoolExecutor`` that FileParser uses
when ``subprocess_isolation`` is enabled.  Key design decisions:

Forkserver context
    On Linux, ``forkserver`` is used so that the server process pre-imports
    heavy Docling dependencies via ``set_forkserver_preload``.  Workers
    forked from it inherit those imports through copy-on-write, making
    worker startup nearly instant even though each worker is short-lived.

max_tasks_per_child = 1
    Every worker handles exactly **one** file then exits.  This guarantees
    that all memory — including Python arena fragments that ``gc.collect()``
    cannot reclaim — is returned to the OS.  New workers are forked cheaply
    from the warm forkserver.

RLIMIT_AS (belt-and-suspenders)
    An optional per-process virtual-address-space cap (expressed as a
    percentage of total system RAM) is applied in each child.  If a parse
    exceeds the cap the child dies with ``MemoryError`` instead of
    triggering the system OOM killer.

All functions in this module are **top-level** (not methods) because
``ProcessPoolExecutor`` requires submitted callables to be picklable.
"""

from __future__ import annotations

import logging
import multiprocessing
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Optional

import psutil

if TYPE_CHECKING:
    from unity.file_manager.file_parsers.types.contracts import (
        FileParseRequest,
        FileParseResult,
    )
    from unity.file_manager.types.config import ParseConfig

logger = logging.getLogger(__name__)

# ``resource`` is a Unix-only stdlib module (Linux + macOS).  On Windows it
# does not exist and importing it raises ``ModuleNotFoundError``.  We guard
# the import so that the rest of the module (pool management, subprocess
# entry point, warm imports) remains fully usable on all platforms.
_resource = None
try:
    import resource as _resource  # type: ignore[no-redef]
except ModuleNotFoundError:
    pass

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


def warm_imports() -> None:
    """Pool-worker initializer that ensures Docling is imported.

    When the forkserver has already pre-loaded these modules, the
    ``import`` statements below are no-ops (the modules are already in
    ``sys.modules``).  This function acts as a safety net for platforms
    where ``forkserver`` is unavailable and ``spawn`` is used instead.
    """
    import docling  # noqa: F401
    import docling.document_converter  # noqa: F401
    import docling.datamodel.pipeline_options  # noqa: F401
    import docling.datamodel.base_models  # noqa: F401
    import polars  # noqa: F401


# ---------------------------------------------------------------------------
# Per-subprocess memory limit
# ---------------------------------------------------------------------------


def apply_memory_limit(parse_config: Optional["ParseConfig"]) -> None:
    """Set ``RLIMIT_AS`` in the current (child) process.

    The limit is computed as ``max_subprocess_memory_pct * total_system_RAM``.
    Only applied on Linux (where ``RLIMIT_AS`` is available); silently
    skipped on Windows and macOS.
    """
    if parse_config is None:
        return
    pct = getattr(parse_config, "max_subprocess_memory_pct", None)
    if pct is None:
        return
    if platform.system() != "Linux" or _resource is None:
        return
    try:
        total_memory = psutil.virtual_memory().total
        limit = max(int(total_memory * pct), 1 << 30)
        current_soft, current_hard = _resource.getrlimit(_resource.RLIMIT_AS)
        hard_limit = current_hard
        if hard_limit not in (-1, _resource.RLIM_INFINITY):
            limit = min(limit, int(hard_limit))
        _resource.setrlimit(_resource.RLIMIT_AS, (limit, limit))
    except Exception as exc:
        logger.debug("Could not set RLIMIT_AS: %s", exc)


# ---------------------------------------------------------------------------
# Top-level subprocess entry point (must be picklable)
# ---------------------------------------------------------------------------


def subprocess_parse_single(
    request: "FileParseRequest",
    parse_config: Optional["ParseConfig"],
) -> "FileParseResult":
    """Parse a single file inside an isolated child process.

    Creates a fresh ``FileParser`` so that all Docling / openpyxl memory
    is fully reclaimed when the child exits.  Also applies RLIMIT_AS and
    saves results to disk when ``SAVE_PARSED_RESULTS`` is enabled.

    RLIMIT_AS is applied **after** the ``FileParser`` is constructed so
    that all heavy imports (Docling, openpyxl, ONNX, etc.) are complete
    before the virtual-address-space cap takes effect.

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

    # Warm the selected backend import before RLIMIT_AS is installed. Backends
    # are loaded lazily, so applying the memory cap first can make even the
    # import itself fail (for example when importing polars or Docling).
    parser._pick_backend(fmt, registry=registry)
    apply_memory_limit(parse_config)
    # _parse_single() already calls save_result_to_disk() internally.
    result = parser._parse_single(request, registry=registry)
    flush_writer()
    return result


# ---------------------------------------------------------------------------
# Warm pool lifecycle
# ---------------------------------------------------------------------------

_pool: Optional[ProcessPoolExecutor] = None
_pool_max_workers: int = 0


def get_or_create_pool(workers: int) -> ProcessPoolExecutor:
    """Return (or lazily create) a persistent warm ``ProcessPoolExecutor``.

    On Unix (Linux / macOS) the pool uses the ``forkserver`` context so
    that workers inherit pre-imported Docling modules via CoW.  On
    Windows, ``forkserver`` is unavailable; the pool falls back to the
    default ``spawn`` context (``warm_imports`` still runs as the
    initializer so each child imports Docling once before accepting
    work).

    ``max_tasks_per_child=1`` ensures every worker handles exactly one
    file then exits, returning all memory to the OS.  See the module
    docstring for the full rationale.
    """
    global _pool, _pool_max_workers

    if _pool is not None and not getattr(_pool, "_broken", False):
        if _pool_max_workers >= workers:
            return _pool
        _pool.shutdown(wait=False)

    if _IS_WINDOWS:
        # forkserver is not available on Windows; use the default (spawn).
        ctx = None
    else:
        ensure_forkserver_preload()
        ctx = multiprocessing.get_context("forkserver")

    _pool = ProcessPoolExecutor(
        max_workers=workers,
        **({"mp_context": ctx} if ctx is not None else {}),
        initializer=warm_imports,
        max_tasks_per_child=1,
    )
    _pool_max_workers = workers
    return _pool


def shutdown_pool() -> None:
    """Shut down the persistent subprocess pool (if any)."""
    global _pool, _pool_max_workers
    if _pool is not None:
        try:
            _pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _pool = None
        _pool_max_workers = 0


def force_reset_pool(workers: int) -> "ProcessPoolExecutor":
    """Kill stuck child processes and create a fresh pool.

    Called when a subprocess hangs (e.g., ``MemoryError`` leaves it wedged)
    or the pool is broken (e.g., OOM killer terminated a child).  The
    function:

    1. Extracts PIDs of any live child workers from the pool's internals.
    2. Shuts down the pool (non-blocking, cancels pending futures).
    3. Force-kills any children still alive (``SIGKILL`` on Unix,
       ``SIGTERM`` on Windows).
    4. Creates and returns a fresh pool via ``get_or_create_pool``.

    This is intentionally aggressive: the goal is to guarantee forward
    progress in the parent process, even if individual files fail.
    """
    import os
    import signal

    global _pool, _pool_max_workers

    if _pool is not None:
        child_pids: list[int] = []
        try:
            processes = getattr(_pool, "_processes", None)
            if processes:
                child_pids = list(processes.keys())
        except Exception:
            pass

        try:
            _pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        kill_sig = getattr(signal, "SIGKILL", signal.SIGTERM)
        for pid in child_pids:
            try:
                os.kill(pid, kill_sig)
            except (OSError, ProcessLookupError):
                pass

        _pool = None
        _pool_max_workers = 0
        logger.warning(
            "Force-reset subprocess pool (killed %d children); "
            "creating fresh pool with %d workers",
            len(child_pids),
            workers,
        )

    return get_or_create_pool(workers)
