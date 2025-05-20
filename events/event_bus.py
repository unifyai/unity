"""In‑process, asyncio‑friendly event stream with **per‑type sliding windows**.

Changes from the previous version
---------------------------------
* Each *event type* now keeps its own bounded deque capped at a *window size*.
  • Default window is **50** events.
  • Window length for any type can be changed at runtime via :py:meth:`EventBus.set_window`.
* Oldest events for that type are dropped once the limit is exceeded (FIFO).
* Public helpers :py:meth:`EventBus.get_window` and :py:meth:`EventBus.set_window`.

The public surface (``Event``, ``EventBus.subscribe()``, ``Subscription``) is
unchanged, so existing user‑code and unit tests need only adjust their
expectations around history trimming.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional

__all__ = ["Event", "EventBus", "Subscription"]

# ───────────────────────────   Event model   ────────────────────────────────


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable message shared across all managers."""

    type: str
    ts: dt.datetime = field(default_factory=dt.datetime.utcnow)
    payload: Optional[dict] = None


# ───────────────────────────   EventBus   ───────────────────────────────────


class EventBus:
    """Singleton that stores & dispatches events with per‑type sliding windows."""

    _instance: "EventBus | None" = None

    # Default number of events to keep *per type*
    _DEFAULT_WINDOW = 50

    # ---------------------------------------------------------------------
    # Construction / singleton machinery
    # ---------------------------------------------------------------------
    def __new__(cls):  # noqa: D401 – explicit
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_internal_state()
        return cls._instance

    # ------------------------------------------------------------------
    def _init_internal_state(self) -> None:  # noqa: D401 – internal helper
        self._log_by_type: Dict[str, Deque[Event]] = {}
        self._window_sizes: Dict[str, int] = {}
        self._subs: List["Subscription"] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def publish(self, event: Event) -> None:
        """Record *event* and fan‑out to live subscribers.

        Drops the oldest events *of the same type* once the configured window
        length is exceeded, ensuring memory remains bounded regardless of how
        many different event types exist.
        """
        async with self._lock:
            window = self._window_sizes.get(event.type, self._DEFAULT_WINDOW)

            dq = self._log_by_type.setdefault(event.type, deque())
            dq.append(event)
            # Trim to window size (manual so we can change window sizes later).
            while len(dq) > window:
                dq.popleft()

            # Fan‑out (outside *while* loop so latency ≈ O(#subs)).
            for sub in self._subs:
                if sub.matches(event):
                    # *put_nowait* because the queue belongs to the subscriber;
                    # they decide whether to use bounded queues for back‑pressure.
                    sub._queue.put_nowait(event)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def subscribe(
        self,
        *,
        match: Callable[[Event], bool] | None = None,
        event_types: Iterable[str] | None = None,
    ) -> "Subscription":
        """Return a *live* filtered stream of events.

        Exactly the same semantics as the previous version – either supply
        *event_types* (set of strings) or an arbitrary predicate.
        """
        predicate: Callable[[Event], bool]
        if event_types is not None:
            wanted: set[str] = set(event_types)
            predicate = lambda e, w=wanted: e.type in w  # noqa: E731 – λ is fine here
        else:
            predicate = match or (lambda _e: True)

        sub = Subscription(bus=self, predicate=predicate)
        self._subs.append(sub)
        return sub

    # ------------------------------------------------------------------
    def get_history(self, predicate: Callable[[Event], bool]) -> List[Event]:
        """Return *all* retained events matching *predicate*, ordered by timestamp."""
        # Merge all deques, then filter & sort. Deques are individually ordered
        # by insertion time, but we need a global ordering across types.
        events: List[Event] = []
        for dq in self._log_by_type.values():
            events.extend(dq)
        return sorted((e for e in events if predicate(e)), key=lambda e: e.ts)

    # ------------------------------------------------------------------
    # Window‑size management
    # ------------------------------------------------------------------
    def set_window(self, event_type: str, size: int) -> None:
        """Change the *window size* for *event_type* **in‑place**.

        If *size* < current length, the oldest events are trimmed immediately.
        """
        if size <= 0:
            raise ValueError("Window size must be positive")

        self._window_sizes[event_type] = size

        dq = self._log_by_type.get(event_type)
        if dq is not None:
            while len(dq) > size:
                dq.popleft()

    def get_window(self, event_type: str) -> int:
        """Return the current window size for *event_type* (fallback to default)."""
        return self._window_sizes.get(event_type, self._DEFAULT_WINDOW)


# ───────────────────────   Subscription helpers   ───────────────────────────


class Subscription:
    """Handle returned by :py:meth:`EventBus.subscribe`. Async‑iterable."""

    def __init__(self, bus: EventBus, predicate: Callable[[Event], bool]):
        self._bus = bus
        self._pred = predicate
        self._queue: "asyncio.Queue[Event]" = asyncio.Queue()

    # Realtime stream ------------------------------------------------------
    async def __aiter__(self):
        while True:
            yield await self._queue.get()

    # Historical snapshot --------------------------------------------------
    def replay(self) -> List[Event]:
        """Return *past* events visible to this subscription, in timestamp order."""
        return self._bus.get_history(self._pred)

    # Runtime filter tweaks -----------------------------------------------
    def set_types(self, *types: str):
        wanted: set[str] = set(types)
        self._pred = lambda e, w=wanted: e.type in w  # noqa: E731

    def matches(self, event: Event) -> bool:  # internal use
        return self._pred(event)
