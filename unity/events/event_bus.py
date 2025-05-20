"""In‑process, asyncio‑friendly event stream **prefilled from Unify logs** and
restricted to Pydantic payload types declared in *events/types/*.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import importlib
import pkgutil
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional, Type

import unify
from pydantic import BaseModel

__all__ = ["Event", "EventBus", "Subscription"]


# ───────────────────────────   Dynamic type registry   ───────────────────────

def _discover_event_types() -> Dict[str, Type[BaseModel]]:
    """Return mapping *event_type → BaseModel subclass* discovered in package."""
    types_pkg = importlib.import_module("unity.events.types")
    registry: Dict[str, Type[BaseModel]] = {}
    for _finder, mod_name, ispkg in pkgutil.iter_modules(types_pkg.__path__):  # type: ignore[arg-type]
        if ispkg:
            continue
        module = importlib.import_module(f"{types_pkg.__name__}.{mod_name}")
        for attr in module.__dict__.values():
            if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel:
                registry[mod_name] = attr
                break
    return registry


_EVENT_TYPES: Dict[str, Type[BaseModel]] = _discover_event_types()

# ───────────────────────────   Event envelope   ─────────────────────────────

@dataclass(frozen=True, slots=True)
class Event:
    type: str
    ts: dt.datetime = field(default_factory=dt.datetime.utcnow)
    payload: Optional[BaseModel] = None


# ───────────────────────────   EventBus singleton   ─────────────────────────

class EventBus:
    _instance: "EventBus | None" = None
    _DEFAULT_WINDOW = 50

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_internal_state()
        return cls._instance

    # ------------------------------------------------------------------
    def _init_internal_state(self) -> None:
        self._log_by_type: Dict[str, Deque[Event]] = {}
        self._window_sizes: Dict[str, int] = {}
        self._subs: List["Subscription"] = []
        self._lock = asyncio.Lock()

        # ── Unify setup ────────────────────────────────────────────────
        unify.initialize_async_logger()
        ctx_info = unify.get_active_context()
        base_ctx = ctx_info["read"] or "Events"
        self._global_ctx = f"{base_ctx}/Events" if base_ctx else "Events"
        self._ctx_by_type = {
            etype: f"{self._global_ctx}/{etype}" for etype in _EVENT_TYPES
        }
        self._logger = unify.AsyncLoggerManager()

        # ── Hydrate in‑memory windows from persisted logs ─────────────
        self._prefill_from_unify()

    # ------------------------------------------------------------------
    def _prefill_from_unify(self):
        """Populate each per‑type deque with newest logs from Unify."""
        for etype, model_cls in _EVENT_TYPES.items():
            window = self._DEFAULT_WINDOW
            try:
                raw_logs = unify.get_logs(context=self._ctx_by_type[etype], limit=window)
            except Exception:  # pragma: no cover – network issues, etc.
                raw_logs = []
            # unify returns most‑recent‑first – reverse for chronological order
            dq: Deque[Event] = deque(maxlen=window)
            for log in reversed(raw_logs):
                entry = getattr(log, "entries", None)
                if entry is None:
                    continue
                if isinstance(entry, Event):
                    evt = entry
                elif isinstance(entry, model_cls):
                    ts = getattr(log, "ts", dt.datetime.utcnow())
                    evt = Event(type=etype, ts=ts, payload=entry)
                else:  # ignore malformed rows
                    continue
                dq.append(evt)
            if dq:
                self._log_by_type[etype] = dq

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def publish(self, event: Event) -> None:
        self._validate(event)

        async with self._lock:
            dq = self._log_by_type.setdefault(event.type, deque())
            dq.append(event)
            window = self._window_sizes.get(event.type, self._DEFAULT_WINDOW)
            while len(dq) > window:
                dq.popleft()
            for sub in self._subs:
                if sub.matches(event):
                    sub._queue.put_nowait(event)  # type: ignore[attr-defined]

        # Non‑blocking telemetry (outside lock)
        self._logger.log_create(
            project=unify.active_project(),
            context=self._global_ctx,
            params={},
            entries=event,
        )
        if event.payload is not None:
            self._logger.log_create(
                project=unify.active_project(),
                context=self._ctx_by_type[event.type],
                params={},
                entries=event.payload,
            )

    def subscribe(
        self,
        *,
        match: Callable[[Event], bool] | None = None,
        event_types: Iterable[str] | None = None,
    ) -> "Subscription":
        predicate: Callable[[Event], bool]
        if event_types is not None:
            wanted = set(event_types)
            predicate = lambda e, w=wanted: e.type in w  # noqa: E731
        else:
            predicate = match or (lambda _e: True)
        sub = Subscription(self, predicate)
        self._subs.append(sub)
        return sub

    def get_history(self, predicate: Callable[[Event], bool]) -> List[Event]:
        events: List[Event] = []
        for dq in self._log_by_type.values():
            events.extend(dq)
        return sorted((e for e in events if predicate(e)), key=lambda e: e.ts)

    def set_window(self, event_type: str, size: int) -> None:
        if event_type not in _EVENT_TYPES:
            raise ValueError(f"Unknown event type '{event_type}'")
        if size <= 0:
            raise ValueError("Window size must be positive")
        self._window_sizes[event_type] = size
        dq = self._log_by_type.get(event_type)
        if dq is not None:
            while len(dq) > size:
                dq.popleft()

    def get_window(self, event_type: str) -> int:
        return self._window_sizes.get(event_type, self._DEFAULT_WINDOW)

    # ------------------------------------------------------------------
    def _validate(self, event: Event) -> None:
        if event.type not in _EVENT_TYPES:
            raise ValueError(f"Event type '{event.type}' is not registered")
        model_cls = _EVENT_TYPES[event.type]
        if event.payload is not None and not isinstance(event.payload, model_cls):
            raise TypeError(
                f"Payload must be {model_cls.__name__} for event type '{event.type}'"
            )


# ───────────────────────────   Subscription   ───────────────────────────────

class Subscription:
    def __init__(self, bus: EventBus, predicate: Callable[[Event], bool]):
        self._bus = bus
        self._pred = predicate
        self._queue: "asyncio.Queue[Event]" = asyncio.Queue()

    async def __aiter__(self):
        while True:
            yield await self._queue.get()

    def replay(self) -> List[Event]:
        return self._bus.get_history(self._pred)

    def set_types(self, *types: str):
        wanted = set(types)
        self._pred = lambda e, w=wanted: e.type in w  # noqa: E731

    def matches(self, event: Event) -> bool:
        return self._pred(event)
