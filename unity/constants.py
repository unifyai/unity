"""
unity/constants.py
==================

Runtime constants and backward-compatible accessors for settings.

This module provides:
1. True runtime constants (SESSION_ID, LOGGER) that cannot be configured via env
2. Backward-compatible module-level accessors for settings (now in unity/settings.py)
3. Logging setup for verbose asyncio debug mode
"""

import logging
import os
from datetime import datetime, timezone

from unity.settings import SETTINGS

# ─────────────────────────────────────────────────────────────────────────────
# True Runtime Constants (not configurable via environment)
# ─────────────────────────────────────────────────────────────────────────────
SESSION_ID = datetime.now(timezone.utc).isoformat()
LOGGER = logging.getLogger("unity")

# External service key (not managed by settings since it's a secret)
ANTICAPTCHA_KEY = os.getenv("ANTICAPTCHA_KEY")

# ─────────────────────────────────────────────────────────────────────────────
# Backward-Compatible Accessors
# These provide module-level access to settings for existing code.
# New code should import SETTINGS directly from unity.settings.
# ─────────────────────────────────────────────────────────────────────────────
ASYNCIO_DEBUG = SETTINGS.ASYNCIO_DEBUG
ASYNCIO_VERBOSE_DEBUG = SETTINGS.ASYNCIO_VERBOSE_DEBUG
LLM_IO_DEBUG = SETTINGS.LLM_IO_DEBUG
PYTEST_LOG_TO_FILE = SETTINGS.PYTEST_LOG_TO_FILE


def is_semantic_cache_enabled() -> bool:
    """Check if semantic cache mode is enabled."""
    return SETTINGS.UNITY_SEMANTIC_CACHE


def is_readonly_ask_guard_enabled() -> bool:
    """Check if the read-only ask guard is enabled."""
    return SETTINGS.UNITY_READONLY_ASK_GUARD


# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup for Verbose Asyncio Debug Mode
# ─────────────────────────────────────────────────────────────────────────────

if ASYNCIO_VERBOSE_DEBUG:
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
