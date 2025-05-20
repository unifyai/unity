"""In‑process, asyncio‑friendly event stream **prefilled from Unify logs** and
restricted to Pydantic payload types declared in *events/types/*.
"""

from __future__ import annotations

import unify
import asyncio
import datetime as dt
from collections import deque
from dataclasses import field
from typing import Deque, Dict, Iterable, Optional, Type
from pydantic import BaseModel, ValidationError

from .types.message import Message
from .types.message_exchange_summary import MessageExchangeSummary

__all__ = ["Event", "EventBus", "Subscription"]


_EVENT_TYPES: Dict[str, Type[BaseModel]] = {"message": Message, "message_exchange_summary": MessageExchangeSummary}
_DEFAULT_WINDOW = 50

# ───────────────────────────   Event envelope   ─────────────────────────────

class Event(BaseModel):
    type: str
    ts: dt.datetime = field(default_factory=dt.datetime.now(dt.UTC))
    payload: BaseModel


# ───────────────────────────   EventBus singleton   ─────────────────────────

class EventBus:

    def __init__(self, windows_sizes: Optional[Dict[str, int]] = None):

        # private attributes
        self._deques: Dict[str, Deque[Event]] = {}
        self._window_sizes: Dict[str, int] = {k: windows_sizes.get(k, _DEFAULT_WINDOW) for k in _EVENT_TYPES.keys()}
        self._lock = asyncio.Lock()

        # ── Unify setup ────────────────────────────────────────────────
        unify.initialize_async_logger()
        active_ctx = unify.get_active_context()
        base_ctx = active_ctx["write"] or "Events"
        self._global_ctx = f"{base_ctx}/Events" if base_ctx else "Events"
        self._ctxs = {
            etype: f"{self._global_ctx}/{etype}" for etype in _EVENT_TYPES
        }
        self._logger = unify.AsyncLoggerManager()

        # ── Hydrate in‑memory windows from persisted logs ─────────────
        self._prefill_from_unify()

    # ------------------------------------------------------------------
    def _prefill_from_unify(self):
        """Populate each per‑type deque with newest logs from Unify."""
        for etype, model_cls in _EVENT_TYPES.items():
            window_size = self._window_sizes[etype]
            raw_logs = unify.get_logs(context=self._ctxs[etype], limit=window_size)
            # unify returns most‑recent‑first – reverse for chronological order
            dq: Deque[Event] = deque(maxlen=window_size)
            for log in reversed(raw_logs):
                entries = log.entries
                if entries is None:
                    continue
                try:
                    evt = Event.model_validate(entries)
                    if not isinstance(evt.payload, model_cls):
                        continue
                except ValidationError:
                    if isinstance(entries, model_cls):
                        ts = getattr(log, "ts", dt.datetime.now(dt.UTC))
                        evt = Event(type=etype, ts=ts, payload=entries)
                    else:
                        raise Exception("")
                dq.append(evt)
            self._deques[etype] = dq

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def publish(self, event: Event) -> None:
        window = self._window_sizes[event.type]

        async with self._lock:
            dq = self._deques[event.type]
            dq.append(event)
            while len(dq) > window:
                dq.popleft()

        # Log to global event table
        self._logger.log_create(
            project=unify.active_project(),
            context=self._global_ctx,
            params={},
            entries=event,
        )

        # Log to specific event table
        self._logger.log_create(
            project=unify.active_project(),
            context=self._ctxs[event.type],
            params={},
            entries=event.payload,
        )

    async def get_latest(
        self,
        types: Iterable[str] | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """
        Return up to *limit* events drawn from the specified *types*
        (or from *all* types if None), ordered **newest-first**.

        Always works with the in-memory deques; does not mutate them.
        """
        async with self._lock:
            wanted = set(types) if types is not None else self._deques.keys()

            # 1. collect (usually small) piles of events
            bucket: list[Event] = []
            for t in wanted:
                dq = self._deques.get(t)
                if dq:
                    bucket.extend(dq)          # each dq is already window-bounded

            # 2. sort newest→oldest and slice
            bucket.sort(key=lambda e: e.ts, reverse=True)
            return bucket[:limit]
