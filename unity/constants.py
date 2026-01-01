"""
unity/constants.py
==================

Runtime constants that cannot be configured via environment.

For configurable settings, import SETTINGS from unity.settings.
"""

import logging
from datetime import datetime, timezone

from unity.settings import SETTINGS

# ─────────────────────────────────────────────────────────────────────────────
# True Runtime Constants (not configurable via environment)
# ─────────────────────────────────────────────────────────────────────────────
SESSION_ID = datetime.now(timezone.utc).isoformat()
LOGGER = logging.getLogger("unity")

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup for Verbose Asyncio Debug Mode
# ─────────────────────────────────────────────────────────────────────────────

if SETTINGS.ASYNCIO_DEBUG_VERBOSE:
    import asyncio
    import sys
    import threading

    class _TaskFilter(logging.Filter):
        """Attach asyncio task/thread names to log records."""

        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            task = asyncio.current_task()
            record.task = task.get_name() if task else "-"
            record.thread = threading.current_thread().name
            return True

    _FMT = "%(asctime)s %(levelname)7s [%(thread)s|%(task)s] %(message)s"

    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(_FMT))

    _root = logging.getLogger()

    # Avoid adding duplicates if constants.py is re-imported.
    _already_configured = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "_asyncio_debug", False)
        for h in _root.handlers
    )

    if not _already_configured:
        _root.setLevel(logging.INFO)
        _root.addFilter(_TaskFilter())
        _handler._asyncio_debug = True  # type: ignore[attr-defined]
        _root.addHandler(_handler)

# ─────────────────────────────────────────────────────────────────────────────
# Defensive Record Factory
# Ensures optional fields exist to avoid KeyError in formatters.
# ─────────────────────────────────────────────────────────────────────────────

_orig_factory = logging.getLogRecordFactory()


def _safe_record_factory(*args, **kwargs):  # pragma: no cover - trivial shim
    rec = _orig_factory(*args, **kwargs)
    if not hasattr(rec, "task"):
        rec.task = "-"
    if not hasattr(rec, "thread"):
        try:
            import threading as _th

            rec.thread = _th.current_thread().name
        except Exception:
            rec.thread = "-"
    return rec


logging.setLogRecordFactory(_safe_record_factory)
