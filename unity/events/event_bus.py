"""In‑process, asyncio‑friendly event stream **prefilled from Unify logs** and
restricted to Pydantic payload types declared in *events/types/*.
"""

from __future__ import annotations

import unify
import asyncio
import datetime as dt
from collections import deque
from typing import List, Deque, Dict, Iterable, Union, Mapping, Any, Optional
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
from uuid import uuid4

__all__ = ["Event", "EventBus", "Subscription"]


# ───────────────────────────   Event envelope   ─────────────────────────────


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    calling_id: str = ""
    type: str
    timestamp: str = Field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())

    # Accept *anything* here; we'll convert it on the way out.
    payload: SerializeAsAny[Any]

    # dotted Python path to the payload model – filled in automatically
    payload_cls: str = ""

    # -------------------------------------------------
    #  validators
    # -------------------------------------------------
    @field_validator("timestamp", mode="before")
    def _ensure_iso(cls, v):
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

    # -------------------------------------------------
    #  serializer
    # -------------------------------------------------
    @field_serializer("payload", when_used="json")
    def _serialise_payload(self, value: Any, _info):
        """
        Convert every nested BaseModel inside `payload` to plain Python objects
        so that `event.model_dump()` never hits Pydantic’s functional serializer
        pathway again.
        """
        return self._to_python(value)

    # -------------------------------------------------
    #  model config
    # -------------------------------------------------
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # let `payload` contain anything
        extra="forbid",
    )

    # -------------------------------------------------
    #  helpers
    # -------------------------------------------------
    @classmethod
    def _to_python(cls, v: Any) -> Any:
        """
        Recursively turn BaseModels → dicts, leaving primitive types unchanged.
        """
        if isinstance(v, BaseModel):
            return v.model_dump(mode="python")

        if isinstance(v, Mapping):
            # Reuse the classmethod to walk nested structures
            return {k: cls._to_python(sub) for k, sub in v.items()}

        if isinstance(v, (list, tuple, set)):
            it: Iterable[Any] = [_ for _ in v]  # satisfy mypy/pyright
            return [cls._to_python(sub) for sub in it]

        return v


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
        ctxs = unify.get_contexts(prefix=f"{self._global_ctx}/")
        self._window_sizes: Dict[str, int] = {
            ctx.split("/")[-1]: self._default_window for ctx in ctxs
        }
        self._specific_ctxs = {ctx.split("/")[-1]: ctx for ctx in ctxs}
        self._logger = unify.AsyncLoggerManager()

        # ── Hydrate in‑memory windows from persisted logs ─────────────
        self._prefill_from_unify()

    # ------------------------------------------------------------------
    def _prefill_from_unify(self):
        """Populate each per‑type deque with newest logs from Unify."""
        for etype, context in self._specific_ctxs.items():
            window_size = self._window_sizes.setdefault(etype, self._default_window)
            raw_logs = unify.get_logs(
                context=context,
                limit=window_size,
                sorting={"timestamp": "descending"},
            )
            # unify returns most‑recent‑first – reverse for chronological order
            dq: Deque[Event] = deque(maxlen=window_size)
            for log in reversed(raw_logs):
                entries = log.entries
                if entries is None:
                    continue

                # Extract the event metadata fields
                event_id = entries.pop("event_id")
                calling_id = entries.pop("calling_id")
                timestamp = entries.pop("event_timestamp")
                cls_path = entries.pop("payload_cls")

                # ── 1. recover the payload class (if recorded) ──────────
                Model: type[BaseModel] | None = None
                if cls_path:
                    try:
                        mod, name = cls_path.rsplit(".", 1)
                        Model = getattr(import_module(mod), name)
                    except (ModuleNotFoundError, AttributeError, ValueError):
                        Model = None

                # ── 2. rebuild the payload instance (fallback: dict) ────
                if Model is not None:
                    try:
                        payload_obj = Model.model_validate(entries)
                    except ValidationError:  # corrupted row → keep dict
                        payload_obj = entries
                else:
                    payload_obj = entries

                evt = Event(
                    event_id=event_id,
                    calling_id=calling_id,
                    type=etype,
                    timestamp=timestamp,
                    payload=payload_obj,
                    payload_cls=cls_path or "",
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
            if event_type not in self._specific_ctxs:
                full_ctx = f"{self._global_ctx}/{event_type}"
                self._specific_ctxs[event_type] = full_ctx
                if full_ctx not in unify.get_contexts():
                    unify.create_context(full_ctx)
            if event_type not in self._window_sizes:
                self._window_sizes[event_type] = self._default_window

    async def publish(self, event: Event) -> None:
        self.register_event_types(event.type)
        window = self._window_sizes[event.type]
        if event.type not in self._specific_ctxs:
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
            entries=event.model_dump(),
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

    def join_published(self):
        """Ensures all published events have been uploaded"""
        self._logger.join()

    async def search(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: Union[int, Dict[str, int]] = 100,
        grouped_by_type: bool = False,
    ) -> Union[List[Event], Dict[str, List[Event]]]:
        """
        Return the most-recent *limit* events that satisfy *filter*.

        Parameters
        ----------
        filter
            A Python expression evaluated against every event.
            • Use ``event_type`` or ``type`` for the event-type
            • The whole ``Event`` object is available as ``evt``
              (e.g. ``evt.payload.user_id == 7``)
        offset
            Number of matching events to **skip first** (newest→oldest
            chronology – identical to Unify's *offset* semantics).
        limit
            Either a single integer applied to *every* event-type or a
            ``{event_type: int}`` dict for per-type limits.
        grouped_by_type
            ``False``  → flat list (each element *includes* its ``type``)
            ``True``   → ``{etype: [events...]}``
        """
        # 0. figure the per-type limits we should respect -----------------
        if isinstance(limit, int):
            per_type_limit = {t: limit for t in self._deques}
        else:
            per_type_limit = {t: limit.get(t, 0) for t in self._deques}

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

                    if skipped < offset:  # still burning offset
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
            backend_offsets[etype] = offset + collected
            need_backend[etype] = still_needed

        # ----------------------------------------------------------------------
        # 3. fetch from *global* Events table ----------------------------------
        for etype, want in need_backend.items():
            full_filter = f'type == "{etype}"' + (f" and ({filter})" if filter else "")

            logs = unify.get_logs(
                context=self._global_ctx,
                filter=full_filter,
                sorting={"timestamp": "descending"},
                offset=backend_offsets[etype],
                limit=want,
            )

            fetched = []
            for lg in logs:  # lg.entries uses Event schema
                e = lg.entries.copy()
                fetched.append(
                    Event(
                        event_id=e.pop("event_id"),
                        calling_id=e.pop("calling_id"),
                        type=e.pop("type"),
                        timestamp=e.pop("timestamp"),
                        payload_cls=e.pop("payload_cls", ""),
                        payload=e.pop("payload"),
                    ),
                )

            in_memory.setdefault(etype, []).extend(fetched)

        # 4. shape the result --------------------------------------------
        if grouped_by_type:
            # guarantee each list is *exactly* per_type_limit long
            return {
                et: evts[: per_type_limit[et]] for et, evts in in_memory.items() if evts
            }

        # flatten + truncate globally (to respect int-style limit)
        flat: List[Event] = []
        for et, evts in in_memory.items():
            flat.extend(evts[: per_type_limit[et]])
        # newest-first global ordering
        flat.sort(key=lambda e: e.timestamp, reverse=True)
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
