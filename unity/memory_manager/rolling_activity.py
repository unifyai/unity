from __future__ import annotations

"""Thread-safe global cache for the *rolling activity* Markdown snippet.

Instead of querying the backend every time a prompt is built we keep the
latest **time-based** rolling-activity summary in memory:

1. The first call to :func:`get_rolling_activity` lazily initialises the
   cache by instantiating a ``MemoryManager`` and calling
   :pymeth:`~unity.memory_manager.memory_manager.MemoryManager.get_rolling_activity`.
   This deferred import avoids circular-dependency issues during start-up.
2. Subsequent reads are lock-free – they just return the cached string.
3. Whenever the :class:`~unity.memory_manager.memory_manager.MemoryManager``
   persists a *new* snapshot it MUST call :func:`set_rolling_activity` so the
   cache is atomically updated for **all** modules in the current interpreter.
"""

import threading
from typing import Optional

__all__ = [
    "get_rolling_activity",
    "set_rolling_activity",
]

# ---------------------------------------------------------------------------
# Internal state guarded by a re-entrant lock so that nested acquisitions
# (unlikely but possible) do not dead-lock.
# ---------------------------------------------------------------------------

_LOCK = threading.RLock()
# ``None`` = not initialised yet
_ROLLING_ACTIVITY: Optional[str] = None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_rolling_activity() -> str:  # noqa: D401 – imperative helper name
    """Return the cached time-based rolling-activity Markdown.

    The first call lazily populates the cache via
    ``MemoryManager().get_rolling_activity()`` so import order does not matter.
    In case the MemoryManager cannot be instantiated (e.g. during isolated
    unit tests that monkey-patch heavy dependencies) we silently fall back to
    an *empty* string – prompts simply omit the Historic Activity block.
    """

    global _ROLLING_ACTIVITY  # noqa: PLW0603 – module-level mutation intended

    if _ROLLING_ACTIVITY is not None:
        return _ROLLING_ACTIVITY

    with _LOCK:
        if _ROLLING_ACTIVITY is not None:
            return _ROLLING_ACTIVITY

        try:
            # Lazy import to avoid circular dependencies
            from .memory_manager import MemoryManager  # noqa: WPS433

            _ROLLING_ACTIVITY = MemoryManager().get_rolling_activity()
        except Exception:
            _ROLLING_ACTIVITY = ""
        return _ROLLING_ACTIVITY


def set_rolling_activity(value: str) -> None:  # noqa: D401 – imperative helper name
    """Atomically replace the cached rolling-activity snippet with *value*."""

    global _ROLLING_ACTIVITY  # noqa: PLW0603 – intentional global write
    with _LOCK:
        _ROLLING_ACTIVITY = value
