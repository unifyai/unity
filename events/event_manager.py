from __future__ import annotations
import asyncio, datetime as dt
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Iterable, List

# ─── 1. Event model ─────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Event:
    type: str
    ts: dt.datetime = field(default_factory=dt.datetime.now(dt.UTC))
    payload: dict | None = None

# ─── 2. EventBus ────────────────────────────────────────────────────────────────
class EventBus:
    _instance: "EventBus" | None = None

    def __new__(cls, maxlen: int = 10_000):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._log = deque[Event](maxlen=maxlen)
            cls._instance._subs = list["Subscription"]()
            cls._instance._lock = asyncio.Lock()
        return cls._instance

    async def publish(self, event: Event) -> None:
        async with self._lock:
            self._log.append(event)          # 1️⃣ record in global timeline
            for sub in self._subs:           # 2️⃣ push to live subscribers
                if sub.matches(event):
                    sub._queue.put_nowait(event)

    def subscribe(
        self,
        *,
        match: Callable[[Event], bool] | None = None,
        event_types: Iterable[str] | None = None,
    ) -> "Subscription":
        predicate = (
            (lambda e: e.type in set(event_types))
            if event_types is not None
            else (match or (lambda _e: True))
        )
        sub = Subscription(bus=self, predicate=predicate)
        self._subs.append(sub)
        return sub

    # Convenience for managers that occasionally need to "widen the window"
    def get_history(self, predicate: Callable[[Event], bool]) -> List[Event]:
        return [e for e in self._log if predicate(e)]

# ─── 3. Subscription handle (per-manager) ───────────────────────────────────────
class Subscription:
    def __init__(self, bus: EventBus, predicate: Callable[[Event], bool]):
        self._bus, self._pred, self._queue = bus, predicate, asyncio.Queue()

    # — streaming interface
    async def __aiter__(self):
        while True:
            yield await self._queue.get()

    # — manual pull of past events
    def replay(self) -> List[Event]:
        return self._bus.get_history(self._pred)

    # — runtime filter tweaks
    def set_types(self, *types: str):
        self._pred = lambda e, st=set(types): e.type in st

    def matches(self, event: Event) -> bool:
        return self._pred(event)

# ─── 4. Helper to create a shared bus (optional) ────────────────────────────────
bus = EventBus(maxlen=50_000)   # keep the last 50 k events
