import logging
from pathlib import Path
from datetime import datetime, timezone
import os

SESSION_ID = datetime.now(timezone.utc).isoformat()
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".unity"
LOGGER = logging.getLogger("unity")
ANTICAPTCHA_KEY = os.getenv("ANTICAPTCHA_KEY")

# Global asyncio debug flag loaded from environment variable. Set ASYNCIO_DEBUG=1 (or true/yes/on) to enable.
ASYNCIO_DEBUG = os.getenv("ASYNCIO_DEBUG", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Optional verbose debug logging flag. When enabled, structured logging with
# task/thread breadcrumbs is added.
ASYNCIO_VERBOSE_DEBUG = os.getenv("ASYNCIO_VERBOSE_DEBUG", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def is_semantic_cache_enabled() -> bool:
    """
    Check if semantic cache mode is enabled via the UNITY_SEMANTIC_CACHE environment variable.

    Semantic cache mode is OFF by default. Set UNITY_SEMANTIC_CACHE=true (or 1/yes/on)
    to enable it.
    """
    return os.getenv("UNITY_SEMANTIC_CACHE", "false").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


# --------------------------------------------------------------------------- #
#  Logging setup for verbose asyncio debug mode                               #
# --------------------------------------------------------------------------- #

if ASYNCIO_VERBOSE_DEBUG:
    import asyncio
    import threading
    import sys

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
        _handler._asyncio_debug = True  # Mark to detect duplication
        _root.addHandler(_handler)

        # (Verbose mode uses a logger filter instead of a global record factory.)

# --------------------------------------------------------------------------- #
#  Defensive record factory: ensure optional fields exist to avoid KeyError     #
#  in formatters that reference %(task)s or %(thread)s even when verbose mode   #
#  is off or filters are not attached.                                         #
# --------------------------------------------------------------------------- #

_orig_factory = logging.getLogRecordFactory()


def _safe_record_factory(*args, **kwargs):  # pragma: no cover - trivial shim
    rec = _orig_factory(*args, **kwargs)
    # Provide defaults if absent so formatters never raise KeyError.
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
