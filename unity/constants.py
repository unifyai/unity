import logging
from pathlib import Path
from datetime import datetime, timezone
import os  # NEW: for reading Anti-Captcha key

SESSION_ID = datetime.now(timezone.utc).isoformat()
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".unity"
LOGGER = logging.getLogger("unity")
ANTICAPTCHA_KEY = os.getenv("ANTICAPTCHA_KEY")  # NEW: Anti-Captcha API key

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

# --------------------------------------------------------------------------- #
#  Logging setup for verbose asyncio debug mode                               #
# --------------------------------------------------------------------------- #

if ASYNCIO_VERBOSE_DEBUG:
    import asyncio
    import threading
    import sys

    class _TaskFilter(logging.Filter):
        """Add current asyncio task & thread names to every log record."""

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
