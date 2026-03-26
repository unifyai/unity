"""Debug diagnostics for the file parser.

When the ``SAVE_PARSED_RESULTS`` environment variable (or settings field) is
truthy, every ``FileParseResult`` produced by the parser — whether in the
main process or in a subprocess worker — is serialised to JSON on disk.

The output directory defaults to ``logs/parsed_results/`` and can be
overridden via the ``PARSED_RESULTS_DIR`` setting.  Files are named
``{sanitised_logical_path}_{UTC_timestamp}.json``.

All disk I/O (including the potentially expensive ``model_dump_json``
serialisation) is performed **asynchronously** on a dedicated daemon
thread so it never blocks the main parse pipeline.

This is intended as a zero-code-change debugging tool: set the env var,
re-run ingestion, inspect the JSON output.
"""

from __future__ import annotations

import atexit
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING

from unity.file_manager.file_parsers.settings import FILE_PARSER_SETTINGS

if TYPE_CHECKING:
    from unity.file_manager.file_parsers.types.contracts import FileParseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Background writer thread
# ---------------------------------------------------------------------------

# A sentinel value pushed to the queue to signal the writer to exit.
_SHUTDOWN = object()

# The queue and thread are lazily initialised on first use so there is zero
# overhead when SAVE_PARSED_RESULTS is disabled (the common case).
_queue: SimpleQueue | None = None
_writer_thread: threading.Thread | None = None
_lock = threading.Lock()


def _writer_loop(q: SimpleQueue) -> None:  # type: ignore[type-arg]
    """Drain the queue and write results until the shutdown sentinel arrives."""
    while True:
        item = q.get()
        if item is _SHUTDOWN:
            return
        result, out_dir_str, safe_name, ts = item
        try:
            out_dir = Path(out_dir_str)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{safe_name}_{ts}.json"
            # model_dump_json + write_text happen entirely on this thread.
            out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            logger.info("Saved parsed result to %s", out_path)
        except Exception as exc:
            logger.warning("Failed to save parsed result: %s", exc)


def _ensure_writer() -> SimpleQueue:  # type: ignore[type-arg]
    """Start the background writer thread if it isn't running yet."""
    global _queue, _writer_thread
    if _queue is not None:
        return _queue
    with _lock:
        # Double-checked locking.
        if _queue is not None:
            return _queue
        q: SimpleQueue = SimpleQueue()  # type: ignore[type-arg]
        t = threading.Thread(target=_writer_loop, args=(q,), daemon=True)
        t.start()
        _queue = q
        _writer_thread = t
        atexit.register(_shutdown_writer)
        return _queue


def _shutdown_writer() -> None:
    """Flush the queue and join the writer thread on interpreter shutdown."""
    flush_writer()


def flush_writer() -> None:
    """Block until all enqueued results have been written to disk.

    Safe to call from any thread.  After flushing the writer thread is
    torn down so a fresh one will be lazily created on the next
    ``save_result_to_disk`` call.

    In the main process this happens automatically via an ``atexit``
    handler.  In subprocess workers (``ProcessPoolExecutor`` with
    ``max_tasks_per_child=1``), ``atexit`` may not fire reliably, so
    callers should invoke ``flush_writer()`` explicitly before the
    subprocess returns.
    """
    global _queue, _writer_thread
    with _lock:
        if _queue is None:
            return
        _queue.put(_SHUTDOWN)
        if _writer_thread is not None:
            _writer_thread.join(timeout=30)
        _writer_thread = None
        _queue = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_result_to_disk(result: "FileParseResult") -> None:
    """Enqueue a ``FileParseResult`` for async serialisation and disk write.

    Only enqueues when ``FILE_PARSER_SETTINGS.SAVE_PARSED_RESULTS`` is
    ``True``.  The actual JSON serialisation and file write happen on a
    background daemon thread so this call returns immediately.

    Failures during the write are logged as warnings but never propagated.
    """
    settings = FILE_PARSER_SETTINGS
    if not settings.SAVE_PARSED_RESULTS:
        return
    safe_name = re.sub(r"[^\w.\-]", "_", str(result.logical_path))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    q = _ensure_writer()
    q.put((result, settings.PARSED_RESULTS_DIR, safe_name, ts))
