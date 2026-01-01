"""
unity/constants.py
==================

Runtime constants that cannot be configured via environment.

For configurable settings, import SETTINGS from unity.settings.

File-based logging:
    When UNITY_LOG_DIR is set (via env var or configure_log_dir()),
    Unity's LOGGER output is written to {UNITY_LOG_DIR}/unity.log.
    This captures async tool loop events, manager operations, etc.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from unity.settings import SETTINGS

# ─────────────────────────────────────────────────────────────────────────────
# True Runtime Constants (not configurable via environment)
# ─────────────────────────────────────────────────────────────────────────────
SESSION_ID = datetime.now(timezone.utc).isoformat()
LOGGER = logging.getLogger("unity")

# File handler state (managed by configure_log_dir)
_FILE_HANDLER: Optional[logging.FileHandler] = None
_LOG_DIR: Optional[Path] = None

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


# ─────────────────────────────────────────────────────────────────────────────
# File-based Logging Configuration
# ─────────────────────────────────────────────────────────────────────────────


def configure_log_dir(log_dir: Optional[str] = None) -> Optional[Path]:
    """Configure or reconfigure the Unity LOGGER file output directory.

    When configured, LOGGER output is written to {log_dir}/unity.log.
    This captures async tool loop events, manager operations, hierarchical
    session logs, and any other code using LOGGER.

    Call this after setting UNITY_LOG_DIR if the env var was set
    after this module was imported.

    Args:
        log_dir: Explicit log directory path. If None, reads from
                 UNITY_LOG_DIR env var (or SETTINGS.UNITY_LOG_DIR).

    Returns:
        The configured log directory Path, or None if disabled.
    """
    global _FILE_HANDLER, _LOG_DIR

    # Remove existing file handler if any
    if _FILE_HANDLER is not None:
        LOGGER.removeHandler(_FILE_HANDLER)
        _FILE_HANDLER.close()
        _FILE_HANDLER = None
        _LOG_DIR = None

    # Determine log directory
    if log_dir is not None:
        os.environ["UNITY_LOG_DIR"] = log_dir
        dir_path = log_dir
    else:
        dir_path = os.environ.get("UNITY_LOG_DIR", "").strip() or SETTINGS.UNITY_LOG_DIR

    if not dir_path:
        return None

    try:
        log_path = Path(dir_path)
        log_path.mkdir(parents=True, exist_ok=True)

        log_file = log_path / "unity.log"
        handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")

        # Use same format as console but with timestamp
        fmt = "%(asctime)s %(levelname)7s %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        handler.setLevel(logging.DEBUG)

        # Mark handler for identification
        handler._unity_file_handler = True  # type: ignore[attr-defined]

        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.DEBUG)  # Ensure logger level allows debug

        _FILE_HANDLER = handler
        _LOG_DIR = log_path

        LOGGER.debug(f"Unity file logging enabled: {log_file}")
        return log_path

    except Exception as e:
        # Best-effort: log to console if file logging fails
        logging.warning(f"Failed to configure Unity log directory {dir_path}: {e}")
        return None


def get_log_dir() -> Optional[Path]:
    """Get the current Unity log directory, if configured."""
    return _LOG_DIR


# Auto-configure from settings on module load (if UNITY_LOG_DIR is set)
if SETTINGS.UNITY_LOG_DIR:
    configure_log_dir(SETTINGS.UNITY_LOG_DIR)
