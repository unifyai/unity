"""In‑process, asyncio‑friendly event stream **prefilled from Unify logs** and
restricted to Pydantic payload types declared in *events/types/*.
"""

from __future__ import annotations

import unify
import asyncio
import datetime as dt
from collections import deque
from datetime import datetime
from typing import (
    List,
    Deque,
    Dict,
    Iterable,
    Union,
    Mapping,
    Any,
    Optional,
    Callable,
    Awaitable,
)

from importlib import import_module
from pydantic import (
    BaseModel,
    Field,
    SerializeAsAny,
    ValidationError,
    field_validator,
    model_validator,
    field_serializer,
    ConfigDict,
)
from pydantic.alias_generators import to_snake
from uuid import uuid4

__all__ = ["Event", "EventBus", "Subscription", "EVENT_BUS"]


# ───────────────────────────   Event envelope   ─────────────────────────────

UNASSIGNED = -1  # shared sentinel – keep it in one place if you already have it


class Event(BaseModel):
    # ────────────────────────────────────────────────
    # primary / synthetic keys
    # ────────────────────────────────────────────────
    row_id: int = Field(
        default=UNASSIGNED,
        ge=UNASSIGNED,
        description="Auto-incrementing database row id (-1 until the DB assigns it)",
    )
    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Stable UUID for this event (unique across DBs)",
    )

    # ────────────────────────────────────────────────
    # metadata
    # ────────────────────────────────────────────────
    calling_id: str = Field(
        default="",
        description="Identifier of the process/machine that produced the event",
    )
    type: str = Field(
        description="Domain-level event type or ‘topic’",
    )
    timestamp: datetime = Field(
        default_factory=lambda: dt.datetime.now(dt.UTC),
        description="ISO-8601 timestamp (UTC)",
    )

    # ────────────────────────────────────────────────
    # polymorphic payload
    # ────────────────────────────────────────────────
    payload: SerializeAsAny[Any]
    payload_cls: str = ""  # dotted Python path (filled automatically)

    # ────────────────────────────────────────────────
    # validators
    # ────────────────────────────────────────────────
    @model_validator(mode="before")
    @classmethod
    def _inject_sentinel(cls, data: dict) -> dict:
        """
        Ensure `row_id` is always present so downstream code never needs to
        handle `None` vs. int.
        """
        data.setdefault("row_id", UNASSIGNED)
        return data

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_iso(cls, v: str | dt.datetime) -> str:
        if isinstance(v, dt.datetime):
            return v.isoformat()
        return v

    @model_validator(mode="after")
    def _auto_payload_cls(self):
        if not self.payload_cls and isinstance(self.payload, BaseModel):
            object.__setattr__(
                self,
                "payload_cls",
                f"{self.payload.__class__.__module__}.{self.payload.__class__.__name__}",
            )
        return self

    # ────────────────────────────────────────────────
    # serialiser helpers
    # ────────────────────────────────────────────────
    @field_serializer("payload", when_used="json")
    def _serialise_payload(self, value: Any, _info):
        """Recursively convert nested BaseModels → plain Python objects."""
        return self._to_python(value)

    def to_post_join(self) -> dict:
        """
        Dump a JSON-serialisable dict suitable for an *insert-and-join* REST
        endpoint.
        If `row_id` is still the sentinel (-1) we *omit* it so the server can
        allocate the next sequence value just like it does for `Contact` and
        `Message`.
        """
        exclude = {"row_id"} if self.row_id == UNASSIGNED else {}
        return self.model_dump(mode="json", exclude=exclude)

    # ────────────────────────────────────────────────
    # config
    # ────────────────────────────────────────────────
    model_config = ConfigDict(
        extra="forbid",  # keep the existing strictness
        arbitrary_types_allowed=True,  # payload can be literally anything
        alias_generator=to_snake,  # optional: stay in sync with your other models
    )

    # ────────────────────────────────────────────────
    # helpers
    # ────────────────────────────────────────────────
    @classmethod
    def _to_python(cls, v: Any) -> Any:  # noqa: PLR0911 – simple, explicit recursion
        if isinstance(v, BaseModel):
            return v.model_dump(mode="python")

        if isinstance(v, Mapping):
            return {k: cls._to_python(sub) for k, sub in v.items()}

        if isinstance(v, (list, tuple, set)):
            it: Iterable[Any] = [_ for _ in v]  # help mypy/pyright
            return [cls._to_python(sub) for sub in it]

        return v


# ───────────────────────────   Subscription   ─────────────────────────────


class Subscription(BaseModel):
    """
    Declarative description of a callback triggered either
    • every *count_step* matching events, **or**
    • every *time_step* seconds since the last trigger.

    Pure-data attributes are persisted to a dedicated Unify context so that
    progress survives interpreter restarts.  The in-memory ``callback`` is
    (re-)attached by client code at runtime.
    """

    subscription_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    filter: Optional[str] = None

    # Trigger rules  ────────────────────────────────────────────────────
    count_step: Optional[int] = None  # e.g. "every 50"
    time_step: Optional[int] = None  # seconds

    # Progress bookkeeping  ────────────────────────────────────────────
    last_row_id: int = 0
    last_timestamp: Optional[datetime] = None

    # in-memory only
    callback: Optional[Callable[[List["Event"]], Union[Awaitable[None], None]]] = Field(
        default=None,
        exclude=True,
    )
    local_count: int = Field(default=0, exclude=True)  # row_id-less fallback

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # ------------------------------------------------------------------
    def matches(self, evt: "Event") -> bool:
        if self.event_type != evt.type:
            return False
        if not self.filter:
            return True
        ns = {
            "evt": evt,
            "event_type": evt.type,
            "type": evt.type,
            **evt.model_dump(mode="python"),
        }
        return bool(eval(self.filter, {"__builtins__": {}}, ns))

    # ------------------------------------------------------------------
    def should_trigger(self, evt: "Event") -> bool:
        """Return *True* if *evt* moves us past the next threshold."""

        # count-based ---------------------------------------------------
        if self.count_step is not None:
            if evt.row_id != UNASSIGNED:
                if evt.row_id - self.last_row_id >= self.count_step:
                    return True
            else:  # freshly published – no row_id yet
                self.local_count += 1
                if self.local_count >= self.count_step:
                    return True

        # time-based ----------------------------------------------------
        if self.time_step is not None:
            if not self.last_timestamp:
                return True
            prev = self.last_timestamp
            now = evt.timestamp
            if (now - prev).total_seconds() >= self.time_step:
                return True

        return False

    # ------------------------------------------------------------------
    def update_progress(self, evt: "Event") -> None:
        if evt.row_id != UNASSIGNED:
            self.last_row_id = evt.row_id
        self.last_timestamp = evt.timestamp
        self.local_count = 0

    def to_post_join(self) -> dict:
        return self.model_dump(mode="json")


# ───────────────────────────   EventBus singleton   ─────────────────────────


class EventBus:
    def __init__(self):

        # private attributes
        self._deques: Dict[str, Deque[Event]] = {}
        self._lock = asyncio.Lock()
        self._default_window = 50

        # ── Unify setup ────────────────────────────────────────────────
        active_ctx = unify.get_active_context()
        base_ctx = active_ctx["write"]
        self._global_ctx = f"{base_ctx}/Events" if base_ctx else "Events"
        upstream_ctxs = unify.get_contexts()
        if self._global_ctx not in upstream_ctxs:
            unify.create_context(self._global_ctx)

        # Persisted subscription metadata lives here
        self._callbacks_ctx = f"{self._global_ctx}/_callbacks"
        if self._callbacks_ctx not in upstream_ctxs:
            unify.create_context(
                self._callbacks_ctx,
                unique_column_ids="row_id",
            )
        ctxs = unify.get_contexts(prefix=f"{self._global_ctx}/")
        self._window_sizes: Dict[str, int] = {
            ctx.split("/")[-1]: self._default_window for ctx in ctxs
        }
        self._specific_ctxs = {
            ctx.split("/")[-1]: ctx for ctx in ctxs if ctx != self._callbacks_ctx
        }
        self._logger = unify.AsyncLoggerManager()

        # runtime subscriptions (id → Subscription)
        self._subscriptions: Dict[str, Subscription] = {}

        # ── Hydrate in the *background* rather than blocking import time ───
        # The original synchronous pre-fill was executed right here,
        # effectively stalling every process that imported the module.
        #
        # We now:
        #   1.  spin up a task (if an event-loop already exists) **or**
        #   2.  postpone scheduling until the first coroutine touches
        #       the bus (common during CLI / test startup).
        #
        self._prefill_done: asyncio.Event = asyncio.Event()
        self._prefill_task: Optional["asyncio.Task[None]"] = None

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop yet (import time in sync context) – we’ll launch lazily.
            pass
        else:
            self._prefill_task = loop.create_task(self._async_initial_hydration())

    # ------------------------------------------------------------------
    # New *non-blocking* hydration helpers
    # ------------------------------------------------------------------
    async def _async_initial_hydration(self) -> None:
        """
        Concurrently hydrate deques *and* persisted subscriptions.
        Sets `self._prefill_done` when complete so other coroutines can
        await bus readiness.
        """
        await asyncio.gather(
            self._async_prefill_from_unify(),
            self._async_load_subscriptions(),
        )
        self._prefill_done.set()

    async def _async_prefill_from_unify(self) -> None:
        """Populate per-type deques without blocking the event-loop."""

        async def _prefill_one(etype: str, context: str, window_size: int):
            raw_logs = await asyncio.to_thread(
                unify.get_logs,
                context=context,
                limit=window_size,
                sorting={"timestamp": "descending"},
            )
            dq: Deque[Event] = deque(maxlen=window_size)
            for log in reversed(raw_logs):
                entries = log.entries
                if not entries:
                    continue

                row_id = entries.pop("row_id", UNASSIGNED)
                event_id = entries.pop("event_id")
                calling_id = entries.pop("calling_id")
                timestamp = entries.pop("event_timestamp")
                cls_path = entries.pop("payload_cls")

                Model: type[BaseModel] | None = None
                if cls_path:
                    try:
                        mod, name = cls_path.rsplit(".", 1)
                        Model = getattr(import_module(mod), name)
                    except (ModuleNotFoundError, AttributeError, ValueError):
                        Model = None

                if Model is not None:
                    try:
                        payload_obj = Model.model_validate(entries)
                    except ValidationError:
                        payload_obj = entries
                else:
                    payload_obj = entries

                dq.append(
                    Event(
                        event_id=event_id,
                        row_id=row_id,
                        calling_id=calling_id,
                        type=etype,
                        timestamp=timestamp,
                        payload=payload_obj,
                        payload_cls=cls_path or "",
                    ),
                )
            self._deques[etype] = dq

        tasks = [
            _prefill_one(
                et,
                ctx,
                self._window_sizes.setdefault(et, self._default_window),
            )
            for et, ctx in self._specific_ctxs.items()
        ]
        if tasks:
            await asyncio.gather(*tasks)

    async def _async_load_subscriptions(self) -> None:
        """Async wrapper around the former blocking `_load_subscriptions`."""
        rows = await asyncio.to_thread(
            unify.get_logs,
            context=self._callbacks_ctx,
            sorting={"row_id": "ascending"},
        )
        latest: Dict[str, dict] = {}
        for lg in rows:
            data = lg.entries.copy()
            latest[data["subscription_id"]] = data

        for sdata in latest.values():
            self._subscriptions[sdata["subscription_id"]] = Subscription(
                subscription_id=sdata["subscription_id"],
                event_type=sdata["event_type"],
                filter=sdata.get("filter"),
                count_step=sdata.get("count_step"),
                time_step=sdata.get("time_step"),
                last_row_id=sdata.get("last_row_id", 0),
                last_timestamp=sdata.get("last_timestamp", ""),
            )

    # ------------------------------------------------------------------
    async def _ensure_ready(self) -> None:
        """
        Await background hydration (lazy-started if not running yet).
        Call this at the top of any *public* coroutine that needs the
        internal state to be fully initialised.
        """
        if self._prefill_done.is_set():
            return

        if self._prefill_task is None:
            self._prefill_task = asyncio.create_task(self._async_initial_hydration())

        await self._prefill_done.wait()

    # ------------------------------------------------------------------
    def _load_subscriptions(self) -> None:
        """
        Recreate the in-memory ``_subscriptions`` map from the metadata
        persisted in *self._callbacks_ctx*.
        """
        rows = unify.get_logs(
            context=self._callbacks_ctx,
            sorting={"row_id": "ascending"},
        )
        latest: Dict[str, dict] = {}
        for lg in rows:
            data = lg.entries.copy()
            latest[data["subscription_id"]] = data

        for sdata in latest.values():
            self._subscriptions[sdata["subscription_id"]] = Subscription(
                subscription_id=sdata["subscription_id"],
                event_type=sdata["event_type"],
                filter=sdata.get("filter"),
                count_step=sdata.get("count_step"),
                time_step=sdata.get("time_step"),
                last_row_id=sdata.get("last_row_id", 0),
                last_timestamp=sdata.get("last_timestamp", ""),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_event_types(self, event_types: Union[str, List[str]]) -> None:
        if isinstance(event_types, str):
            event_types = [event_types]
        for event_type in event_types:
            if event_type not in self._specific_ctxs:
                full_ctx = f"{self._global_ctx}/{event_type}"
                self._specific_ctxs[event_type] = full_ctx
                if full_ctx not in unify.get_contexts():
                    unify.create_context(full_ctx, unique_column_ids="row_id")
            if event_type not in self._window_sizes:
                self._window_sizes[event_type] = self._default_window

    async def publish(self, event: Event) -> None:
        await self._ensure_ready()
        if event.type not in self._specific_ctxs:
            self.register_event_types(event.type)
        window = self._window_sizes[event.type]
        async with self._lock:
            dq = self._deques.setdefault(event.type, deque())
            dq.append(event)
            while len(dq) > window:
                dq.popleft()

        if hasattr(event, "to_post_json") and callable(getattr(event, "to_post_json")):
            entries = event.to_post_json()
        else:
            entries = event.model_dump(mode="json")

        # Log to global event table
        self._logger.log_create(
            project=unify.active_project(),
            context=self._global_ctx,
            params={},
            entries=entries,
        )

        # Log to specific event table
        if isinstance(event.payload, BaseModel):
            payload_dict = event.payload.model_dump(mode="python")
        else:
            payload_dict = dict(event.payload)

        self._logger.log_create(
            project=unify.active_project(),
            context=self._specific_ctxs[event.type],
            params={},
            entries={
                **{
                    "event_id": event.event_id,
                    "calling_id": event.calling_id,
                    "event_timestamp": event.timestamp,
                    "payload_cls": event.payload_cls,
                },
                **payload_dict,
            },
        )

        # ── Evaluate subscriptions *after* persistence ──────────────────────
        self._process_event(event)

    def join_published(self):
        """Ensures all published events have been uploaded"""
        self._logger.join()

    async def search(
        self,
        *,
        filter: Optional[str] = None,
        offset: Union[int, Dict[str, int]] = 0,
        limit: Union[int, Dict[str, int]] = 100,
        grouped_by_type: bool = False,
    ) -> Union[List[Event], Dict[str, List[Event]]]:
        await self._ensure_ready()
        """
        Return events that satisfy *filter*, applying *offset*/**limit** rules as
        follows

        ``offset`` & ``limit`` can **each** be either

        * ``int``               – apply the same value to **all** event-types
        * ``{event_type: int}`` – independent per-type value

        The *interaction* of the two parameters is important:

        ┌──────────────┬──────────────┬────────────────────────────────────────────┐
        │ ``offset``   │ ``limit``    │ Interpretation                             │
        ├──────────────┼──────────────┼────────────────────────────────────────────┤
        │ *dict*       │ *dict*       │ Per-type window (dict values respected)    │
        │ *dict*       │ *int*        │ Per-type window – reuse the *int* for      │
        │              │              │ every missing key in *offset*              │
        │ *int*        │ *dict*       │ Per-type window – reuse the *int* for      │
        │              │              │ every missing key in *offset*              │
        │ *int*        │ *int*        │ **Global** window – *offset*/*limit* are   │
        │              │              │ applied **after combining & interweaving** │
        │              │              │ all matching event-types                   │
        └──────────────┴──────────────┴────────────────────────────────────────────┘

        When *both* parameters are simple ``int`` s, the method behaves like a
        traditional "single table" query: imagine all relevant event-types
        merged into one time-ordered list, then drop the first *offset* entries
        and return up to *limit* that follow.
        """
        # 0. Work out which semantics we're in ---------------------------------
        combined_window = isinstance(offset, int) and isinstance(limit, int)

        # ----- per-type helpers ----------------------------------------------
        if combined_window:
            # grab *enough* from every queue (offset + limit) so the global
            # pass later has material to slice from
            per_type_limit = {t: offset + limit for t in self._deques}
            per_type_offset = {t: 0 for t in self._deques}  # skip globally later
        else:
            if isinstance(limit, int):
                per_type_limit = {t: limit for t in self._deques}
            else:
                per_type_limit = {t: limit.get(t, 0) for t in self._deques}

            if isinstance(offset, int):
                per_type_offset = {t: offset for t in self._deques}
            else:
                per_type_offset = {t: offset.get(t, 0) for t in self._deques}

        # ----------------------------------------------------------------------
        # 1. scan the deque -----------------------------------------------------
        def _matches(evt: Event) -> bool:
            if not filter:
                return True
            # VERY small, controlled eval-sandbox
            ns = {
                "evt": evt,
                "event_type": evt.type,
                "type": evt.type,
                **evt.model_dump(mode="python"),
            }
            return bool(eval(filter, {"__builtins__": {}}, ns))

        in_memory: Dict[str, List[Event]] = {}
        deque_meta: Dict[str, tuple[int, int]] = {}  # etype -> (skipped, collected)

        async with self._lock:
            for etype, dq in self._deques.items():
                lim = per_type_limit[etype]
                if lim == 0:
                    continue

                skipped = collected = 0
                keep: list[Event] = []

                for evt in reversed(dq):  # newest → oldest
                    if not _matches(evt):
                        continue

                    if skipped < per_type_offset[etype]:  # still burning offset
                        skipped += 1
                        continue

                    keep.append(evt)
                    collected += 1
                    if collected >= lim:
                        break

                in_memory[etype] = keep
                deque_meta[etype] = (skipped, collected)

        # ----------------------------------------------------------------------
        # 2. decide what and where to fetch ------------------------------------
        need_backend: dict[str, int] = {}
        backend_offsets: dict[str, int] = {}

        for etype, lim in per_type_limit.items():
            skipped, collected = deque_meta.get(etype, (0, 0))
            still_needed = lim - collected
            if still_needed <= 0:
                continue

            # offset still missing from deque + duplicates we already collected
            backend_offsets[etype] = per_type_offset[etype] + collected
            need_backend[etype] = still_needed

        # ----------------------------------------------------------------------
        # 3a. FAST-PATH: one backend call when we are in "global window" mode
        #     *and* have no in-memory events yet (cold start). This avoids N
        #     serial/parallel round-trips while staying trivial to reason about.
        # ----------------------------------------------------------------------
        if combined_window and all(len(dq) == 0 for dq in self._deques.values()):

            if need_backend:  # only when something is actually missing
                types_in = ", ".join(f'"{et}"' for et in need_backend)
                global_limit = offset + limit
                full_filter = f"type in ({types_in})"
                if filter:
                    full_filter += f" and ({filter})"

                logs = await asyncio.to_thread(
                    unify.get_logs,
                    context=self._global_ctx,
                    filter=full_filter,
                    sorting={"timestamp": "descending"},
                    offset=0,
                    limit=global_limit,
                )

                for lg in logs:
                    e = lg.entries.copy()
                    evt = Event(
                        event_id=e.pop("event_id"),
                        row_id=e.pop("row_id", None),
                        calling_id=e.pop("calling_id"),
                        type=e.pop("type"),
                        timestamp=e.pop("timestamp"),
                        payload_cls=e.pop("payload_cls", ""),
                        payload=e.pop("payload"),
                    )
                    in_memory.setdefault(evt.type, []).append(evt)

                # We've satisfied the need; skip the per-type fetch branch
                need_backend.clear()

        # ----------------------------------------------------------------------
        # 3b. Per-type backend fetches – concurrently (as before) --------------
        async def _fetch_one(etype: str, want: int) -> tuple[str, list[Event]]:
            """
            Run the blocking ``unify.get_logs`` call in a worker thread and
            re-wrap the raw log rows as :class:`Event` objects.
            """
            full_filter = f'type == "{etype}"' + (f" and ({filter})" if filter else "")

            logs = await asyncio.to_thread(
                unify.get_logs,
                context=self._global_ctx,
                filter=full_filter,
                sorting={"timestamp": "descending"},
                offset=backend_offsets[etype],
                limit=want,
            )

            evts: list[Event] = []
            for lg in logs:
                e = lg.entries.copy()
                evts.append(
                    Event(
                        event_id=e.pop("event_id"),
                        row_id=e.pop("row_id", None),
                        calling_id=e.pop("calling_id"),
                        type=e.pop("type"),
                        timestamp=e.pop("timestamp"),
                        payload_cls=e.pop("payload_cls", ""),
                        payload=e.pop("payload"),
                    ),
                )
            return etype, evts

        # Kick off all remaining I/O in parallel (if any)
        backend_tasks = [
            _fetch_one(et, want) for et, want in need_backend.items() if want > 0
        ]
        if backend_tasks:
            results = await asyncio.gather(*backend_tasks, return_exceptions=False)
            for etype, fetched in results:
                in_memory.setdefault(etype, []).extend(fetched)

        # 4. shape the result --------------------------------------------
        if grouped_by_type:
            # guarantee each list is *exactly* per_type_limit long
            return {
                et: evts[: per_type_limit[et]] for et, evts in in_memory.items() if evts
            }

        # ── Build the final flat list ────────────────────────────────────────
        flat: List[Event] = []
        for evts in in_memory.values():
            flat.extend(evts)

        # Global ordering (newest-first)
        flat.sort(key=lambda e: e.timestamp, reverse=True)

        if combined_window:
            # apply global windowing now
            return flat[offset : offset + limit]

        # classic per-type limits (already enforced), but we may still need
        # to truncate if the caller passed a *dict* for limit *and* wants
        # fewer rows overall – honour only the per-type caps here.
        if isinstance(limit, int):
            flat = flat[:limit]
        return flat

    def set_window(self, event_type: str, new_size: int) -> None:
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
        if event_type not in self._specific_ctxs:
            self.register_event_types(event_type)

        self._window_sizes[event_type] = new_size

        old_dq: Deque[Event] = self._deques.get(event_type, deque())
        # Re-hydrate deque with new maxlen (keeps newest → oldest order intact)
        new_dq: Deque[Event] = deque(old_dq, maxlen=new_size)
        self._deques[event_type] = new_dq

    # ------------------------------------------------------------------
    async def register_callback(
        self,
        *,
        event_type: str,
        callback: Callable[[List[Event]], Union[Awaitable[None], None]],
        filter: Optional[str] = None,
        every_n: Optional[int] = None,
        every_seconds: Optional[int] = None,
    ) -> str:
        await self._ensure_ready()
        """
        Register *callback* to be fired either every **N** matching events
        or after **X** seconds have elapsed since the previous trigger.
        """
        if every_n is None and every_seconds is None:
            raise ValueError("either `every_n` or `every_seconds` must be supplied")

        # Ensure context exists
        self.register_event_types(event_type)

        # Existing identical subscription? Just attach runtime callback
        for sub in self._subscriptions.values():
            if (
                sub.event_type == event_type
                and sub.filter == filter
                and sub.count_step == every_n
                and sub.time_step == every_seconds
            ):
                sub.callback = callback
                return sub.subscription_id

        # Snapshot current baseline -----------------------------------------
        latest = await asyncio.to_thread(
            unify.get_logs,
            context=self._specific_ctxs[event_type],
            filter=filter,
            sorting={"row_id": "descending"},
            limit=1,
        )
        if latest:
            row = latest[0].entries
            last_row_id = row.get("row_id", 0)
            last_ts = row.get("event_timestamp") or row.get("timestamp", "")
        else:
            last_row_id, last_ts = 0, ""

        sub = Subscription(
            event_type=event_type,
            filter=filter,
            count_step=every_n,
            time_step=every_seconds,
            last_row_id=last_row_id,
            last_timestamp=last_ts,
            callback=callback,
        )
        self._subscriptions[sub.subscription_id] = sub
        self._persist_subscription_state(sub)
        return sub.subscription_id

    # ------------------------------------------------------------------
    def _persist_subscription_state(self, sub: Subscription) -> None:
        """Append current state to the callbacks context for durability."""
        self._logger.log_create(
            project=unify.active_project(),
            context=self._callbacks_ctx,
            params={},
            entries={
                "subscription_id": sub.subscription_id,
                "event_type": sub.event_type,
                "filter": sub.filter,
                "count_step": sub.count_step,
                "time_step": sub.time_step,
                "last_row_id": sub.last_row_id,
                "last_timestamp": sub.last_timestamp,
            },
        )

    # ------------------------------------------------------------------
    def _process_event(self, evt: Event) -> None:
        """Evaluate all subscriptions against *evt* and fire callbacks."""
        loop = asyncio.get_event_loop()
        for sub in list(self._subscriptions.values()):
            if not sub.callback or not sub.matches(evt):
                continue
            if not sub.should_trigger(evt):
                continue

            sub.update_progress(evt)
            self._persist_subscription_state(sub)

            cb = sub.callback
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb([evt]))
                else:
                    loop.run_in_executor(None, cb, [evt])
            except RuntimeError:
                # No running loop (shutdown) – last-ditch synchronous call
                cb([evt])

    def set_default_window(self, new_size: int) -> None:
        """
        Change the *in-memory* history window for ``event_type`` to
        ``new_size`` events.

        • Creates the event-type on-the-fly if not registered yet
          (mirrors :pymeth:`register_event_types` behaviour).
        • Rebuilds the internal :class:`collections.deque` so the new
          ``maxlen`` takes effect immediately, keeping **the most recent**
          messages up to *new_size*.
        """
        self._default_window = new_size

    @property
    def ctxs(self):
        return self._specific_ctxs


# ─────────────────────────   Global singleton   ──────────────────────────
EVENT_BUS: "EventBus" = EventBus()
