"""In‑process, asyncio‑friendly event stream restricted to **typed payloads** that
live in *events/types/*. Each model is a **Pydantic** `BaseModel`.

Key points
----------
* **Automatic type discovery** – any new `*.py` dropped into `events/types/` that
  exposes at least one `BaseModel` subclass becomes a valid event type.
* **Per‑type sliding windows** – default length = 50, tunable at runtime.
* **Strict validation** – `publish()` refuses
  * unknown `event.type`s; or
  * payloads whose class doesn’t match the registered model for that type.
* **Unify logging** – every event is recorded once in a *global* context and
  once in a *type‑specific* context.

Public API (`Event`, `EventBus`, `Subscription`) is unchanged except that
`Event.payload` **must** be a registered Pydantic model (or `None`).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import pkgutil
from collections import deque
from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable, Deque, Dict, Iterable, List, Optional, Type

import unify
from pydantic import BaseModel

__all__ = ["Event", "EventBus", "Subscription"]

# ───────────────────────────   Dynamic type registry   ───────────────────────

def _discover_event_types() -> Dict[str, Type[BaseModel]]:  # noqa: D401
    """Scan *events.types* package for `BaseModel` subclasses.

    A file ``events/types/foo.py`` containing a class ``BarModel(BaseModel)``
    registers the *filename* (``foo``) as the *event type* and ``BarModel`` as
    the expected payload class. If the file defines multiple models, the first
    subclass encountered wins. This keeps naming predictable while letting the
    user add new event types without touching the bus code.
    """
    from importlib import import_module
    from pathlib import Path

    types_pkg = import_module("unity.events.types")
    registry: Dict[str, Type[BaseModel]] = {}

    for finder, name, ispkg in pkgutil.iter_modules(types_pkg.__path__):  # type: ignore[arg-type]
        if ispkg:
            continue
        module: ModuleType = import_module(f"{types_pkg.__name__}.{name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel:
                registry[name] = attr
                break  # first subclass is enough
    return registry


_EVENT_TYPES: Dict[str, Type[BaseModel]] = _discover_event_types()

# ───────────────────────────   Event dataclass   ────────────────────────────

@dataclass(frozen=True, slots=True)
class Event:  # noqa: D401 – simple container
    """Immutable envelope used inside the EventBus."""

    type: str
    ts: dt.datetime = field(default_factory=dt.datetime.utcnow)
    payload: Optional[BaseModel] = None


# ───────────────────────────   EventBus   ───────────────────────────────────

class EventBus:  # noqa: D401 – singleton façade
    """Process‑wide singleton that stores, validates and dispatches events."""

    _instance: "EventBus | None" = None
    _DEFAULT_WINDOW = 50  # events kept per type

    # ------------------------------------------------------------------
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_internal_state()
        return cls._instance

    # ------------------------------------------------------------------
    def _init_internal_state(self) -> None:  # noqa: D401 – helper
        self._log_by_type: Dict[str, Deque[Event]] = {}
        self._window_sizes: Dict[str, int] = {}
        self._subs: List["Subscription"] = []
        self._lock = asyncio.Lock()

        # ── unify setup ────────────────────────────────────────────────
        unify.initialize_async_logger()
        ctx = unify.get_active_context()
        base_ctx = ctx["write"] or "Events"
        self._global_ctx = f"{base_ctx}/Events" if base_ctx else "Events"
        self._ctx_by_type = {
            etype: f"{self._global_ctx}/{etype}" for etype in _EVENT_TYPES
        }
        self._logger = unify.AsyncLoggerManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def publish(self, event: Event) -> None:
        """Validate, record, log, and fan‑out an *event*."""
        self._validate(event)

        # 1️⃣ Persist in memory
        async with self._lock:
            dq = self._log_by_type.setdefault(event.type, deque())
            dq.append(event)
            window = self._window_sizes.get(event.type, self._DEFAULT_WINDOW)
            while len(dq) > window:
                dq.popleft()

            for sub in self._subs:
                if sub.matches(event):
                    sub._queue.put_nowait(event)  # type: ignore[attr-defined]

        # 2️⃣ Non‑blocking event logging (outside lock)
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

    # ------------------------------------------------------------------
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

        sub = Subscription(bus=self, predicate=predicate)
        self._subs.append(sub)
        return sub

    # ------------------------------------------------------------------
    def get_history(self, predicate: Callable[[Event], bool]) -> List[Event]:
        events: List[Event] = []
        for dq in self._log_by_type.values():
            events.extend(dq)
        return sorted((e for e in events if predicate(e)), key=lambda e: e.ts)

    # ------------------------------------------------------------------
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
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate(self, event: Event) -> None:
        if event.type not in _EVENT_TYPES:
            raise ValueError(
                f"Event type '{event.type}' is not registered. Valid types: {list(_EVENT_TYPES)}"
            )
        model_cls = _EVENT_TYPES[event.type]
        if event.payload is not None and not isinstance(event.payload, model_cls):
            raise TypeError(
                f"Payload must be instance of {model_cls.__name__} for event type '{event.type}'"
            )


# ───────────────────────────   Subscription   ───────────────────────────────

class Subscription:  # noqa: D401
    """Async iterable returned by :py:meth:`EventBus.subscribe`."""

    def __init__(self, bus: EventBus, predicate: Callable[[Event], bool]):
        self._bus = bus
        self._pred = predicate
        self._queue: "asyncio.Queue[Event]" = asyncio.Queue()

    # Realtime -----------------------------------------------------------
    async def __aiter__(self):
        while True:
            yield await self._queue.get()

    # Historical snapshot -----------------------------------------------
    def replay(self) -> List[Event]:
        return self._bus.get_history(self._pred)

    # Runtime filter tweaks --------------------------------------------
    def set_types(self, *types: str):
        wanted = set(types)
        self._pred = lambda e, w=wanted: e.type in w  # noqa: E731

    def matches(self, event: Event) -> bool:  # internal use only
        return self._pred(event)
