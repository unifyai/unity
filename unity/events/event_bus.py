"""In‑process, asyncio‑friendly event stream **prefilled from Unify logs** and
restricted to Pydantic payload types declared in *events/types/*.
"""

from __future__ import annotations

import unify
import asyncio
import datetime as dt
from collections import deque
from typing import List, Deque, Dict, Iterable, Union
from pydantic import BaseModel, Field, ValidationError, field_validator
from uuid import uuid4

__all__ = ["Event", "EventBus", "Subscription"]


_DEFAULT_WINDOW = 50

# ───────────────────────────   Event envelope   ─────────────────────────────


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    calling_id: str = ""
    type: str
    timestamp: str = Field(
        default_factory=lambda: dt.datetime.now(dt.UTC).isoformat()
    )
    payload: BaseModel

    @field_validator("timestamp", mode="before")
    def _ensure_iso(cls, v):
        if isinstance(v, dt.datetime):
            return v.isoformat()
        return v


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
                evt = Event(
                    event_id=entries["event_id"],
                    calling_id=entries["calling_id"],
                    type=etype,
                    timestamp=entries["timestamp"],
                    payload=entries,
                )
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
        self.register_event_types(event.type)
        window = self._window_sizes[event.type]
        if event.type not in self._ctxs:
            if event.type not in unify.get_contexts():
                unify.create_context()

        async with self._lock:
            dq = self._deques.setdefault(event.type, deque())
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

    def update_window_size(self, event_type: str, new_size: int) -> None:
        """
        Change the *in-memory* history window for ``event_type`` to
        ``new_size`` events.

        • Creates the event-type on-the-fly if not registered yet
          (mirrors :pymeth:`register_event_types` behaviour).
        • Rebuilds the internal :class:`collections.deque` so the new
          ``maxlen`` takes effect immediately, keeping **the most recent**
          messages up to *new_size*.
        """
        if new_size <= 0:
            raise ValueError("new_size must be a positive integer")

        # Ensure bookkeeping structures exist
        if event_type not in self._ctxs:
            self.register_event_types(event_type)

        self._window_sizes[event_type] = new_size

        old_dq: Deque[Event] = self._deques.get(event_type, deque())
        # Re-hydrate deque with new maxlen (keeps newest → oldest order intact)
        new_dq: Deque[Event] = deque(old_dq, maxlen=new_size)
        self._deques[event_type] = new_dq

    async def get_event_call_stack(self, event_id: str | int) -> list[Event]:
        """
        Return all parent-events of *event_id* **top-down**.

        The first element is the *root* event whose ``calling_id`` is an empty
        string, the last element is the event that matches the supplied
        *event_id*.  If any link in the chain cannot be found the traversal
        stops and the partial stack is returned.

        Lookup strategy
        ----------------
        1. Search the in-memory deques (cheap).
        2. If a required event is missing, query Unify's *global* Events
           context with ``filter_expr='event_id == "<id>"'`` to hydrate it.
        """

        def _find_local(eid: str) -> Event | None:
            """Scan all in-memory windows for *eid* (non-blocking)."""
            for dq in self._deques.values():
                for ev in dq:
                    if ev.event_id == eid:
                        return ev
            return None

        def _fetch_from_unify(eid: str) -> Event | None:
            """Slow path – pull single row from the persisted global log."""
            try:
                logs = unify.get_logs(
                    context=self._global_ctx,
                    limit=1,
                    filter_expr=f"'event_id == \"{eid}\"'",
                )
            except Exception:
                return None

            if not logs:
                return None

            row = logs[0]
            entries = getattr(row, "entries", None)
            if not entries:
                return None

            try:
                # Entries might already be a full Event-like dict
                return Event(**entries)
            except ValidationError:
                # Fall back to a best-effort reconstruction
                ts = getattr(row, "ts", dt.datetime.now(dt.UTC)).isoformat()
                return Event(
                    event_id=str(eid),
                    calling_id=entries.get("calling_id", ""),
                    type=entries.get("type", "Unknown"),
                    timestamp=entries.get("timestamp", ts),
                    payload=entries,
                )

        eid = str(event_id)
        stack_rev: list[Event] = []
        seen: set[str] = set()

        while eid and eid not in seen:
            seen.add(eid)

            evt = _find_local(eid)
            if evt is None:  # not in cache → ask Unify
                evt = _fetch_from_unify(eid)
                if evt is None:
                    break  # gap – abort traversal

                # Cache it inside the appropriate deque for future calls
                self._deques.setdefault(evt.type, deque(maxlen=_DEFAULT_WINDOW)).append(
                    evt,
                )

            stack_rev.append(evt)
            eid = evt.calling_id

        # Return *root → leaf*
        return list(reversed(stack_rev))

    @property
    def ctxs(self):
        return self._ctxs
