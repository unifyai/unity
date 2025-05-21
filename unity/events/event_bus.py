"""In‑process, asyncio‑friendly event stream **prefilled from Unify logs** and
restricted to Pydantic payload types declared in *events/types/*.
"""

from __future__ import annotations

import unify
import asyncio
import datetime as dt
from collections import deque
from typing import List, Deque, Dict, Iterable, Union
from pydantic import BaseModel, ValidationError

__all__ = ["Event", "EventBus", "Subscription"]


_DEFAULT_WINDOW = 50

# ───────────────────────────   Event envelope   ─────────────────────────────


class Event(BaseModel):
    type: str
    timestamp: str
    payload: BaseModel


# ───────────────────────────   EventBus singleton   ─────────────────────────


class EventBus:

    def __init__(self):

        # private attributes
        self._deques: Dict[str, Deque[Event]] = {}
        self._lock = asyncio.Lock()

        # ── Unify setup ────────────────────────────────────────────────
        active_ctx = unify.get_active_context()
        base_ctx = active_ctx["write"]
        self._global_ctx = f"{base_ctx}/Events" if base_ctx else "Events"
        upstream_ctxs = unify.get_contexts()
        if self._global_ctx not in upstream_ctxs:
            unify.create_context(self._global_ctx)
        ctxs = unify.get_contexts(prefix=self._global_ctx)
        self._window_sizes: Dict[str, int] = {
            ctx.split("/")[-1]: _DEFAULT_WINDOW for ctx in ctxs
        }
        self._ctxs = {ctx.split("/")[-1]: ctx for ctx in ctxs}
        self._logger = unify.AsyncLoggerManager()

        # ── Hydrate in‑memory windows from persisted logs ─────────────
        self._prefill_from_unify()

    # ------------------------------------------------------------------
    def _prefill_from_unify(self):
        """Populate each per‑type deque with newest logs from Unify."""
        for etype, context in self._ctxs.items():
            window_size = self._window_sizes.setdefault(etype, _DEFAULT_WINDOW)
            raw_logs = unify.get_logs(context=context, limit=window_size)
            # unify returns most‑recent‑first – reverse for chronological order
            dq: Deque[Event] = deque(maxlen=window_size)
            for log in reversed(raw_logs):
                entries = log.entries
                if entries is None:
                    continue
                ts = getattr(log, "ts", dt.datetime.now(dt.UTC)).isoformat()
                evt = Event(type=etype, timestamp=ts, payload=entries)
                dq.append(evt)
            self._deques[etype] = dq

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_event_types(self, event_types: Union[str, List[str]]) -> None:
        if isinstance(event_types, str):
            event_types = [event_types]
        for event_type in event_types:
            if event_type not in self._ctxs:
                full_ctx = f"{self._global_ctx}/{event_type}"
                self._ctxs[event_type] = full_ctx
                if full_ctx not in unify.get_contexts():
                    unify.create_context(full_ctx)
            if event_type not in self._window_sizes:
                self._window_sizes[event_type] = _DEFAULT_WINDOW

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

    def join_published(self):
        """Ensures all published events have been uploaded"""
        self._logger.join()

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
                    bucket.extend(dq)  # each dq is already window-bounded

            # 2. sort newest→oldest and slice
            bucket.sort(key=lambda e: e.timestamp, reverse=True)
            return bucket[:limit]

    @property
    def ctxs(self):
        return self._ctxs
