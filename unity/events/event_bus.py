"""In‑process, asyncio‑friendly event stream **prefilled from Unify logs** and
restricted to Pydantic payload types declared in *events/types/*.
"""

from __future__ import annotations

import unify
import json
import asyncio
import datetime as dt
from collections import defaultdict, deque
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
    Set,
)

# Context propagation helper for callback cascades
import contextvars

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
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.log_utils import _derive_all_contexts, _inject_private_fields
from ..common.model_to_fields import model_to_fields
from ..logger import LOGGER
from .stream_filters import is_streaming_noise

# ---------------------------------------------------------------------------
# Context-variable to track the *root* sequence number of a callback cascade.
# Every time EventBus schedules a callback it checks whether we are currently
# inside another callback; if yes, the descendant inherits the same root-seq
# so that join_callbacks() can await the complete cascade while
# still ignoring unrelated new activity.
# ---------------------------------------------------------------------------

_CURRENT_ROOT_SEQ: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_CURRENT_ROOT_SEQ",
    default=None,
)


# ───────────────────────────   Event envelope   ─────────────────────────────

# The backend no longer auto-assigns `row_id`.
# A value of `None` indicates that the client-side `EventBus` has not yet
# attached a sequence number.


class Event(BaseModel):
    # ────────────────────────────────────────────────
    # primary / synthetic keys
    # ────────────────────────────────────────────────
    row_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="Monotonically increasing client-managed sequence number (set by EventBus)",
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
        description="Domain-level event type or 'topic'",
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
    def _ensure_row_id_key(cls, data: dict) -> dict:
        # Ensure the key exists so downstream code can safely assume presence.
        data.setdefault("row_id", None)
        return data

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_iso(cls, v: str | dt.datetime) -> str:
        if isinstance(v, dt.datetime):
            return v.isoformat()
        return v

    @model_validator(mode="after")
    def _validate_and_coerce_payload(self):
        """Validate payload against known type schema and convert to dict.

        After validation, payloads are always stored as plain dicts. This ensures
        a consistent interface for consumers (always use .get() or ["key"]).

        The Pydantic models serve two purposes:
        1. Validation at publish time (correctness enforced)
        2. Schema generation for eager field creation (type-safe DB fields)
        """
        from .types import PAYLOAD_REGISTRY

        # Enforce known event types only
        if self.type not in PAYLOAD_REGISTRY:
            raise ValueError(
                f"Unknown event type '{self.type}'. "
                f"Known types: {list(PAYLOAD_REGISTRY.keys())}. "
                f"Define a Pydantic payload model in unity/events/types/.",
            )

        expected_model = PAYLOAD_REGISTRY[self.type]
        validated_model: BaseModel | None = None

        # Validate and coerce to model first
        if isinstance(self.payload, expected_model):
            validated_model = self.payload
        elif isinstance(self.payload, BaseModel):
            # Already a Pydantic model (possibly from a different registry entry
            # like Message which is re-exported as MessagePayload)
            validated_model = self.payload
        elif isinstance(self.payload, dict):
            # Try to validate dict against expected model
            try:
                validated_model = expected_model.model_validate(self.payload)
            except ValidationError:
                # Validation failed (e.g. rehydrating incomplete data) - keep dict
                validated_model = None
        else:
            raise ValueError(
                f"Payload for event type '{self.type}' must be a dict or "
                f"{expected_model.__name__}, got {type(self.payload).__name__}",
            )

        # Auto-set payload_cls from the validated model (before converting to dict)
        if validated_model is not None and not self.payload_cls:
            object.__setattr__(
                self,
                "payload_cls",
                f"{validated_model.__class__.__module__}.{validated_model.__class__.__name__}",
            )

        # Convert validated model to dict for consistent consumer interface
        if validated_model is not None:
            object.__setattr__(
                self,
                "payload",
                validated_model.model_dump(mode="python"),
            )
        # If validation failed and payload was already a dict, it stays a dict

        return self

    # ────────────────────────────────────────────────
    # serialiser helpers
    # ────────────────────────────────────────────────
    @field_serializer("payload", when_used="json")
    def _serialise_payload(self, value: Any, _info):
        """Recursively convert nested BaseModels → plain Python objects."""
        return self._to_python(value)

    def to_post_json(self) -> dict:
        """
        Dump a JSON-serialisable dict suitable for an *insert-and-join* REST
        endpoint.
        If `row_id` has not yet been set (``None``) we omit it so the
        caller can still rely on the server to allocate a value if needed –
        though the normal path is that the :class:`EventBus` sets it before
        persistence.
        """
        exclude = {"row_id"} if self.row_id is None else {}
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
        # ── 1. datetime family → ISO-8601 string ───────────────────────
        if isinstance(v, (dt.datetime, dt.date, dt.time)):
            return v.isoformat()

        # ── 2. pydantic model → dict (JSON-mode guarantees strings) ────
        if isinstance(v, BaseModel):
            return cls._to_python(v.model_dump(mode="json"))

        # ── 3. containers – depth-first recursion ──────────────────────
        if isinstance(v, Mapping):
            return {k: cls._to_python(sub) for k, sub in v.items()}

        if isinstance(v, (list, tuple, set)):
            it: Iterable[Any] = list(v)  # help type checkers
            return [cls._to_python(sub) for sub in it]

        # ── 4. primitives stay unchanged ───────────────────────────────
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
    last_row_id: int = -1
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
        # Reuse shared helper for evaluating optional filter expressions
        return EventBus._match_filter(evt, self.filter)

    # ------------------------------------------------------------------
    def should_trigger(self, evt: "Event") -> bool:
        """Return *True* if *evt* moves us past the next threshold."""

        # count-based ---------------------------------------------------
        if self.count_step is not None:
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
        if evt.row_id is not None:
            self.last_row_id = evt.row_id
        self.last_timestamp = evt.timestamp
        self.local_count = 0

    def to_post_json(self) -> dict:
        return self.model_dump(mode="json")


# ───────────────────────────   EventBus singleton   ─────────────────────────


class EventBus:
    _LOGGER = unify.AsyncLoggerManager(name="EventBus", num_consumers=16)

    # Class-level flag to control event publishing. Initialized from SETTINGS on
    # first EventBus instantiation. Can be overridden (e.g., tests use markers).
    _publishing_enabled: bool | None = None

    # Monotonic timestamp of the most recent publish() call. Used by
    # ConversationManager's inactivity check to detect that internal work
    # (LLM calls, tool-loop turns, manager methods, …) is still happening
    # even when no external pubsub messages are arriving.
    last_publish_monotonic: float = 0.0

    # ── Pub/Sub streaming for Live Actions ────────────────────────────────
    _GCP_PROJECT = "responsive-city-458413-a2"
    _ACTION_EVENT_TYPES = frozenset({"ManagerMethod", "ToolLoop"})
    _pubsub_publisher = None
    _pubsub_streaming_enabled: bool | None = None

    @classmethod
    def _init_publishing_enabled(cls) -> None:
        """Initialize _publishing_enabled from settings if not already set."""
        if cls._publishing_enabled is None:
            try:
                from ..settings import SETTINGS

                cls._publishing_enabled = SETTINGS.EVENTBUS_PUBLISHING_ENABLED
            except Exception:
                # Fallback to disabled if settings can't be loaded
                cls._publishing_enabled = False
            LOGGER.info(
                "EventBus publishing %s",
                "enabled" if cls._publishing_enabled else "disabled",
            )

    @classmethod
    def _init_pubsub_streaming(cls) -> None:
        """Initialize _pubsub_streaming_enabled from settings if not already set."""
        if cls._pubsub_streaming_enabled is None:
            try:
                from ..settings import SETTINGS

                cls._pubsub_streaming_enabled = SETTINGS.EVENTBUS_PUBSUB_STREAMING
            except Exception:
                cls._pubsub_streaming_enabled = False
            LOGGER.info(
                "Pub/Sub action streaming %s",
                "enabled" if cls._pubsub_streaming_enabled else "disabled",
            )

    @classmethod
    def _get_pubsub_publisher(cls):
        """Lazily initialize the GCP Pub/Sub publisher client.

        Message ordering is enabled so that messages published with the same
        ``ordering_key`` are delivered to subscribers in publish order.  Each
        assistant's action events share a single ordering key (the assistant
        ID), guaranteeing the console receives ManagerMethod and ToolLoop
        events in the exact sequence they occurred.
        """
        if cls._pubsub_publisher is None:
            from google.cloud import pubsub_v1
            from google.cloud.pubsub_v1.types import PublisherOptions

            cls._pubsub_publisher = pubsub_v1.PublisherClient(
                publisher_options=PublisherOptions(
                    enable_message_ordering=True,
                ),
            )
        return cls._pubsub_publisher

    def __init__(self):
        # Initialize publishing flag from settings (once, on first instantiation)
        EventBus._init_publishing_enabled()

        # private attributes
        self._deques: Dict[str, Deque[Event]] = {}
        self._lock = asyncio.Lock()
        self._default_window = 50

        # ── Unify setup ────────────────────────────────────────────────
        active_ctx = unify.get_active_context()
        base_ctx = active_ctx["write"]
        if not base_ctx:
            # Ensure the global assistant/context is selected before we derive our sub-context
            try:
                from .. import (
                    ensure_initialised as _ensure_initialised,
                )  # local to avoid cycles

                _ensure_initialised()
                active_ctx = unify.get_active_context()
                base_ctx = active_ctx["write"]
            except Exception:
                # If ensure fails (e.g. offline tests), proceed; downstream will fall back safely
                pass
        self._global_ctx = f"{base_ctx}/Events" if base_ctx else "Events"
        unify.create_context(self._global_ctx)

        # Persisted subscription metadata lives here
        self._callbacks_ctx = f"{self._global_ctx}/_callbacks"
        unify.create_context(
            self._callbacks_ctx,
            unique_keys={"row_id": "int"},
            auto_counting={"row_id": None},
        )
        ctxs = unify.get_contexts(prefix=f"{self._global_ctx}/")
        self._window_sizes: Dict[str, int] = {
            ctx.split("/")[-1]: self._default_window for ctx in ctxs
        }
        self._specific_ctxs = {
            ctx.split("/")[-1]: ctx for ctx in ctxs if ctx != self._callbacks_ctx
        }
        # Manual per-event-type row_id counters (initialised during hydration)
        self._next_row_ids: Dict[str, int] = {}

        # Eagerly create contexts with pre-defined field schemas for all known types
        self._ensure_known_contexts()

        # ---------------- Pinning support ----------------
        # Call-IDs (typically tool handles) that are currently **open** and whose
        # related events must stay resident regardless of the window size.
        self._pinned_call_ids: Set[str] = set()
        # Declarative auto-pin/unpin rules (see `register_auto_pin`).
        # Each entry is a dict with keys: event_type, open_pred, close_pred, key_fn
        self._auto_pin_rules: list[dict[str, Any]] = []

        # Track pending callback futures so we can await their completion
        self._callback_futures: Set[asyncio.Future] = set()
        # Monotonically increasing sequence number for callback tasks –
        # allows join_callbacks to distinguish tasks scheduled *before* its
        # invocation from those scheduled afterwards (see implementation
        # below).
        self._callback_seq: int = 0

        # runtime subscriptions (id → Subscription)
        self._subscriptions: Dict[str, Subscription] = {}

        # Deferred persistence buffer: (entries_dict, context_name) pairs
        # accumulated during publish() and flushed in batch via flush()/clear().
        self._pending_writes: list[tuple[dict, str]] = []

        # Periodic flush: drains _pending_writes every few seconds so that
        # Orchestra stays current for console REST queries while still
        # batching writes for efficiency.
        self._FLUSH_INTERVAL_S = 5.0
        self._periodic_flush_task: Optional["asyncio.Task[None]"] = None

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
        self._prefill_exc: Optional[Exception] = None

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop yet (import time in sync context) – we'll launch lazily.
            pass
        else:
            self._prefill_task = loop.create_task(self._async_initial_hydration())

    # ------------------------------------------------------------------
    # Known event type context creation
    # ------------------------------------------------------------------

    # Common fields present on all events (from Event envelope)
    _COMMON_EVENT_FIELDS: Dict[str, Dict[str, Any]] = {
        "row_id": {"type": "int", "mutable": False},
        "event_id": {"type": "str", "mutable": False},
        "calling_id": {"type": "str", "mutable": False},
        "event_timestamp": {"type": "datetime", "mutable": False},
        "payload_cls": {"type": "str", "mutable": False},
    }

    def _ensure_known_contexts(self) -> None:
        """Create type-specific contexts with pre-defined field schemas.

        For each known event type in PAYLOAD_REGISTRY, this method:
        1. Creates the context if it doesn't exist
        2. Creates fields from the Pydantic payload model using model_to_fields
        3. Registers the context in _specific_ctxs
        4. Creates aggregation contexts for multi-assistant/multi-user views

        This ensures fields exist before any logs are written, preventing
        type inference issues from the first log value.
        """
        from .types import PAYLOAD_REGISTRY

        # Create aggregation contexts for the global Events context
        self._ensure_aggregation_contexts(self._global_ctx)

        for event_type, payload_model in PAYLOAD_REGISTRY.items():
            ctx_name = f"{self._global_ctx}/{event_type}"

            # Skip if already registered
            if event_type in self._specific_ctxs:
                continue

            # Create context
            unify.create_context(ctx_name)

            # Create fields from Pydantic model + common event fields
            try:
                payload_fields = model_to_fields(payload_model)
                all_fields = {**self._COMMON_EVENT_FIELDS, **payload_fields}
                unify.create_fields(all_fields, context=ctx_name)
            except Exception:
                # Fields may already exist or context may have issues; proceed
                pass

            # Register in our tracking dicts
            self._specific_ctxs[event_type] = ctx_name
            self._window_sizes.setdefault(event_type, self._default_window)

            # Create aggregation contexts for this event type
            self._ensure_aggregation_contexts(ctx_name)

    def _ensure_aggregation_contexts(self, context: str) -> None:
        """Create aggregation contexts for multi-assistant and multi-user views.

        For a context like {User}/{Assistant}/Events or {User}/{Assistant}/Events/LLM,
        this creates:
        - {User}/All/Events (or {User}/All/Events/LLM) for user-level aggregation
        - All/Events (or All/Events/LLM) for global aggregation

        These contexts store references to the same logs (not copies), enabling
        cross-assistant and cross-user queries.
        """
        all_ctxs = _derive_all_contexts(context)
        for all_ctx in all_ctxs:
            try:
                unify.create_context(all_ctx)
            except Exception:
                pass  # Context may already exist; proceed

    # ------------------------------------------------------------------
    # Public readonly state helpers
    # ------------------------------------------------------------------

    @property
    def initialized(self) -> bool:
        """Return *True* once the background hydration launched from the
        constructor has finished (successfully **or** with an error).

        This provides a lightweight, synchronous way for callers to check
        whether the EventBus is ready without having to `await` the private
        `_ensure_ready()` coroutine.  It simply reflects the completion state
        of the internal `_prefill_done` event.
        """

        return self._prefill_done.is_set()

    @classmethod
    def _get_logger(cls) -> unify.AsyncLoggerManager:
        return cls._LOGGER

    # ------------------------------------------------------------------
    # New *non-blocking* hydration helpers
    # ------------------------------------------------------------------
    async def _async_initial_hydration(self) -> None:
        """
        Concurrently hydrate deques *and* persisted subscriptions.
        Sets `self._prefill_done` when complete so other coroutines can
        await bus readiness.  Also starts the periodic flush loop.
        """
        try:
            await asyncio.gather(
                self._async_prefill_from_unify(),
                self._async_load_subscriptions(),
            )
        except Exception as exc:  # pragma: no cover – defensive
            # Never leave waiters hanging – remember the error and continue.
            self._prefill_exc = exc
            try:
                self._get_logger().error("EventBus – initial hydration failed: %r", exc)
            except Exception:
                # Logger might not be fully ready; ignore.
                pass
        finally:
            self._prefill_done.set()

        if self._periodic_flush_task is None:
            self._periodic_flush_task = asyncio.create_task(self._periodic_flush_loop())

    async def _periodic_flush_loop(self) -> None:
        """Background task that drains _pending_writes at a fixed cadence."""
        while True:
            await asyncio.sleep(self._FLUSH_INTERVAL_S)
            if self._pending_writes:
                try:
                    await asyncio.to_thread(self.flush)
                except Exception as exc:
                    LOGGER.debug("Periodic flush error: %s", exc)

    # Only these event types are prefilled into in-memory deques on startup.
    # Comms is the only type with production search() consumers
    # (hydrate_global_thread, get_last_store_chat_history).  All other types
    # are either write-only from the deque perspective (LLM, ToolLoop) or
    # consumed exclusively via publish-driven callbacks (ManagerMethod,
    # Message).  search() falls back to the backend for any type not in the
    # deque, so non-prefilled types still work — they just make an API call
    # instead of reading from memory.
    _PREFILL_TYPES: frozenset[str] = frozenset({"Comms"})

    async def _async_prefill_from_unify(self) -> None:
        """Prefill Comms deques and seed row_id counters for all types."""

        async def _prefill_deque(etype: str, context: str, window_size: int):
            """Fetch recent events into the in-memory deque."""
            raw_logs = await asyncio.to_thread(
                unify.get_logs,
                context=context,
                limit=window_size,
                sorting={"timestamp": "descending"},
            )
            dq: Deque[Event] = deque()
            for log in reversed(raw_logs):
                if not log.entries:
                    continue
                dq.append(self._row_to_event(log.entries, default_type=etype))
            self._deques[etype] = dq
            async with self._lock:
                self._trim_window(etype)

        async def _seed_row_id(etype: str, context: str):
            """Fetch only the latest row_id so the counter stays monotonic."""
            raw_logs = await asyncio.to_thread(
                unify.get_logs,
                context=context,
                limit=1,
                sorting={"row_id": "descending"},
            )
            if raw_logs and raw_logs[0].entries:
                row_id = raw_logs[0].entries.get("row_id")
                if row_id is not None:
                    self._next_row_ids[etype] = int(row_id) + 1

        tasks = []
        for et, ctx in self._specific_ctxs.items():
            if et in self._PREFILL_TYPES:
                tasks.append(
                    _prefill_deque(
                        et,
                        ctx,
                        self._window_sizes.setdefault(et, self._default_window),
                    ),
                )
            else:
                tasks.append(_seed_row_id(et, ctx))
        if tasks:
            await asyncio.gather(*tasks)

        # Seed row_id counters for prefilled types from their deques
        for etype in self._PREFILL_TYPES:
            dq = self._deques.get(etype)
            if not dq:
                continue
            max_id = max(
                (evt.row_id for evt in dq if evt.row_id is not None),
                default=-1,
            )
            self._next_row_ids[etype] = max_id + 1

    async def _async_load_subscriptions(self) -> None:
        """Async wrapper around the former blocking `_load_subscriptions`."""
        try:
            rows = await asyncio.to_thread(
                unify.get_logs,
                context=self._callbacks_ctx,
                sorting={"row_id": "ascending"},
            )
        except Exception as e:
            # Handle 404 errors gracefully - the context may not exist yet due to
            # eventual consistency after create_context(). For a newly created
            # context, having no subscriptions is the expected initial state.
            if "404" in str(e) or "not found" in str(e).lower():
                rows = []
            else:
                raise
        self._subscriptions = self._rows_to_subscriptions(rows)

    # ------------------------------------------------------------------
    async def join_initialization(self) -> None:
        """
        Await background hydration (lazy-started if not running yet).
        Call this at the top of any *public* coroutine that needs the
        internal state to be fully initialised.
        """
        if self._prefill_done.is_set():
            if self._prefill_exc:
                raise self._prefill_exc
            return

        if self._prefill_task is None:
            self._prefill_task = asyncio.create_task(self._async_initial_hydration())

        await self._prefill_done.wait()

        # Hydration finished; bubble-up any error so callers fail fast
        if self._prefill_exc:
            raise self._prefill_exc

    def _lazy_start_hydration_if_needed(self) -> None:
        """
        If the background hydration task wasn't started during import (because
        no event loop was running), this method starts it. It does not wait
        for the task to complete.
        """
        if self._prefill_task is not None or self._prefill_done.is_set():
            return
        try:
            loop = asyncio.get_running_loop()
            if self._prefill_task is None:
                self._prefill_task = loop.create_task(self._async_initial_hydration())
        except RuntimeError:
            pass

        # ------------------------------------------------------------------
        #  Pinning helpers
        # ------------------------------------------------------------------

    def _trim_window(self, event_type: str) -> None:
        """Internal: trim *unpinned* events to fit the configured window for *event_type*.  Must be called with `self._lock` held."""
        dq = self._deques.get(event_type)
        if not dq:
            return
        window = self._window_sizes.get(event_type, self._default_window)
        pinned = self._pinned_call_ids
        # Quick exit when total unpinned already within the limit
        unpinned = sum(1 for ev in dq if ev.calling_id not in pinned)
        if unpinned <= window:
            return
        # Remove oldest unpinned events until within window
        while unpinned > window:
            for ev in dq:
                if ev.calling_id in pinned:
                    continue
                dq.remove(ev)
                unpinned -= 1
                break
            else:
                # All remaining events are pinned
                break

    # ------------------------------------------------------------------
    #  Public pin/unpin API
    # ------------------------------------------------------------------

    def pin_call_id(self, call_id: str) -> None:
        """Pin all events whose `calling_id` equals *call_id* until `unpin_call_id` is invoked."""
        self._pinned_call_ids.add(call_id)

    def unpin_call_id(self, call_id: str) -> None:
        """Remove previously set pin for *call_id* and run window-trimming immediately."""
        if call_id in self._pinned_call_ids:
            self._pinned_call_ids.discard(call_id)
            # We do not synchronously trim here to avoid deadlocks with running
            # event-loops.  The very next call to `publish` or `set_window` will
            # invoke `_trim_window` and enforce the window guarantees.

    def register_auto_pin(
        self,
        *,
        event_type: str | None,
        open_predicate: Callable[["Event"], bool],
        close_predicate: Callable[["Event"], bool],
        key_fn: Callable[["Event"], str] | None = None,
    ) -> None:
        """Register *open*/*close* predicates which automatically manage pins.

        Parameters
        ----------
        event_type : str | None
            Restrict the rule to a single event-type.  ``None`` means it applies to *all* types.
        open_predicate : Callable[[Event], bool]
            Evaluated on every published event.  If *True*, the pin represented by ``key_fn(evt)`` is set.
        close_predicate : Callable[[Event], bool]
            If *True*, the pin is removed.
        key_fn : Callable[[Event], str], optional
            Function that extracts the *pin key* from the event. Defaults to the ``calling_id`` field.
        """
        if key_fn is None:
            key_fn = lambda e: e.calling_id  # noqa: E731
        self._auto_pin_rules.append(
            {
                "event_type": event_type,
                "open_pred": open_predicate,
                "close_pred": close_predicate,
                "key_fn": key_fn,
            },
        )

    # ------------------------------------------------------------------
    def _load_subscriptions(self) -> None:
        """Synchronously rebuild the in-memory subscription map."""
        rows = unify.get_logs(
            context=self._callbacks_ctx,
            sorting={"row_id": "ascending"},
        )
        self._subscriptions = self._rows_to_subscriptions(rows)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_event_types(self, event_types: Union[str, List[str]]) -> None:
        """Validate that event types are known and ensure row_id counters exist.

        Known event types have their contexts created eagerly in __init__ via
        _ensure_known_contexts(). This method validates that the requested types
        are in the registry and initializes row_id counters.
        """
        from .types import PAYLOAD_REGISTRY

        if isinstance(event_types, str):
            event_types = [event_types]

        for event_type in event_types:
            if event_type not in PAYLOAD_REGISTRY:
                raise ValueError(
                    f"Unknown event type '{event_type}'. "
                    f"Known types: {list(PAYLOAD_REGISTRY.keys())}. "
                    f"Define a Pydantic payload model in unity/events/types/.",
                )

            # Ensure a local counter exists for this event-type
            self._next_row_ids.setdefault(event_type, 0)

    async def publish(self, event: Event, *, blocking: bool = False) -> None:
        import time as _time

        now = _time.monotonic()
        prev = EventBus.last_publish_monotonic
        EventBus.last_publish_monotonic = now
        gap = now - prev
        if prev > 0 and gap > 20:
            import traceback as _tb

            LOGGER.debug(
                "EventBus.publish after %.1fs gap: type=%s event_id=%s caller:\n%s",
                gap,
                event.type,
                event.event_id,
                "".join(_tb.format_stack()[-6:-1]),
            )

        # Initialize publishing flag from settings if not already done
        if EventBus._publishing_enabled is None:
            EventBus._init_publishing_enabled()
        # Skip publishing if disabled (e.g., during local dev or tests)
        if not EventBus._publishing_enabled:
            return

        self._lazy_start_hydration_if_needed()
        # Best-effort wait for hydration so row_id counters are initialised.
        # If prefill failed (e.g. Orchestra 500s), degrade gracefully: assign
        # row_ids from zero and keep routing events in-process.  Persistence
        # may produce duplicate row_ids in that edge case, but the alternative
        # — permanently blocking publish() for the pod's lifetime — is far
        # worse (it kills the @log_manager_call decorator path and prevents
        # the CodeActActor from ever executing).
        try:
            await self.join_initialization()
        except Exception:
            if not getattr(self, "_prefill_warning_logged", False):
                LOGGER.warning(
                    "EventBus hydration failed — operating with degraded "
                    "persistence (row_id counters not seeded from backend). "
                    "In-process event routing continues normally. Error: %r",
                    self._prefill_exc,
                )
                self._prefill_warning_logged = True
        # --- Auto pin/unpin evaluation *before* we acquire the deque lock ---
        for _rule in self._auto_pin_rules:
            if _rule["event_type"] is None or _rule["event_type"] == event.type:
                key = _rule["key_fn"](event)
                if _rule["close_pred"](event):
                    self.unpin_call_id(key)
                elif _rule["open_pred"](event):
                    self.pin_call_id(key)

        # Known event types are registered eagerly in __init__ via _ensure_known_contexts.
        # Unknown types are rejected by Event validation, so this should always exist.
        window = self._window_sizes[event.type]
        async with self._lock:
            # ── Assign and increment the manual row_id counter ───────────────
            current = self._next_row_ids.get(event.type, 0)
            if event.row_id is None:
                event.row_id = current
            # Advance the counter for the next event
            self._next_row_ids[event.type] = event.row_id + 1

            dq = self._deques.setdefault(event.type, deque())
            dq.append(event)
            # Honour pinning – only trim *unpinned* events
            self._trim_window(event.type)

        # Uniform serialisation – reuse the robust helper already implemented
        # on the Event model to avoid maintaining a second custom walker.
        payload_dict = (
            event.payload.model_dump(mode="json")
            if isinstance(event.payload, BaseModel)
            else Event._to_python(event.payload)
        )

        # Base entries for both log calls (before private field injection)
        base_entries = {
            "row_id": event.row_id,
            "event_id": event.event_id,
            "calling_id": event.calling_id,
            "event_timestamp": event.timestamp.isoformat(),
            "payload_cls": event.payload_cls,
        }

        # Buffer entries for deferred batch upload (flushed in flush()/clear())
        global_entries = _inject_private_fields(
            {
                **base_entries,
                "type": event.type,
                "payload_json": json.dumps(payload_dict),
            },
        )
        self._pending_writes.append((global_entries, self._global_ctx))

        specific_entries = _inject_private_fields(
            {
                **base_entries,
                **payload_dict,
            },
        )
        self._pending_writes.append((specific_entries, self._specific_ctxs[event.type]))

        # ── Stream action events to Pub/Sub for real-time frontend rendering ─
        if event.type in self._ACTION_EVENT_TYPES:
            if not is_streaming_noise(event.type, payload_dict):
                self._stream_action_to_pubsub(event, base_entries, payload_dict)

        # ── Evaluate subscriptions ────────────────────────────────────────
        self._process_event(event)

        if blocking:
            self.flush()

    # ------------------------------------------------------------------
    # Pub/Sub streaming (Live Actions)
    # ------------------------------------------------------------------

    def _stream_action_to_pubsub(
        self,
        event: Event,
        base_entries: dict,
        payload_dict: dict,
    ) -> None:
        """Fire-and-forget publish of a ManagerMethod/ToolLoop event to Pub/Sub.

        The message lands on the assistant's existing Pub/Sub topic with
        ``thread="action_event"`` as a message attribute.  A dedicated
        subscription filtered on that attribute delivers these events to the
        console's SSE endpoint for real-time rendering without Orchestra polling.

        Errors are logged at DEBUG level and never propagate — the Orchestra
        dual-write is the authoritative persistence path.
        """
        if EventBus._pubsub_streaming_enabled is None:
            EventBus._init_pubsub_streaming()
        if not EventBus._pubsub_streaming_enabled:
            return

        topic_name = None
        agent_id = None
        try:
            from ..session_details import SESSION_DETAILS
            from ..settings import SETTINGS

            agent_id = str(SESSION_DETAILS.assistant.agent_id)
            env_suffix = SETTINGS.ENV_SUFFIX if agent_id is not None else ""
            topic_name = f"unity-{agent_id}{env_suffix}"

            publisher = self._get_pubsub_publisher()
            topic_path = publisher.topic_path(self._GCP_PROJECT, topic_name)

            message_data = {
                "thread": "action_event",
                "event": {
                    **base_entries,
                    "type": event.type,
                    **payload_dict,
                },
            }

            future = publisher.publish(
                topic_path,
                json.dumps(message_data, default=str).encode("utf-8"),
                ordering_key=agent_id,
                thread="action_event",
            )
            LOGGER.debug(
                "Pub/Sub publish fired: topic=%s event_type=%s row_id=%s",
                topic_name,
                event.type,
                event.row_id,
            )
            future.add_done_callback(
                self._make_publish_done_callback(topic_path, agent_id),
            )

        except Exception as exc:
            LOGGER.warning(
                "Pub/Sub action streaming failed: topic=%s agent_id=%s "
                "event_type=%s row_id=%s error=%s",
                topic_name,
                agent_id,
                event.type,
                event.row_id,
                exc,
                exc_info=True,
            )

    @classmethod
    def _make_publish_done_callback(cls, topic_path: str, ordering_key: str):
        """Return a callback that handles publish success/failure.

        On failure, calls ``resume_publish`` so that subsequent messages with the
        same ordering key are not permanently blocked by a single transient error.
        """

        def _on_done(future) -> None:
            try:
                message_id = future.result()
                LOGGER.debug(
                    "Pub/Sub publish confirmed: topic=%s ordering_key=%s message_id=%s",
                    topic_path,
                    ordering_key,
                    message_id,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Pub/Sub publish failed: topic=%s ordering_key=%s error=%s",
                    topic_path,
                    ordering_key,
                    exc,
                    exc_info=True,
                )
                try:
                    if cls._pubsub_publisher and ordering_key:
                        cls._pubsub_publisher.resume_publish(topic_path, ordering_key)
                        LOGGER.info(
                            "Resumed Pub/Sub publishing: topic=%s ordering_key=%s",
                            topic_path,
                            ordering_key,
                        )
                except Exception as resume_exc:
                    LOGGER.warning(
                        "Failed to resume Pub/Sub publishing: topic=%s "
                        "ordering_key=%s error=%s",
                        topic_path,
                        ordering_key,
                        resume_exc,
                        exc_info=True,
                    )

        return _on_done

    def flush(self) -> None:
        """Batch-upload all buffered event writes, grouped by context.

        Uses ``unify.create_logs`` (single HTTP POST per context) rather
        than N individual ``log_create`` calls.  Aggregation mirrors are
        attached in bulk after each batch completes.

        Thread-safe: atomically swaps the buffer so concurrent publish()
        calls append to a fresh list while we process the snapshot.
        """
        if not self._pending_writes:
            return

        # Atomic swap: grab current buffer and replace with empty list.
        snapshot = self._pending_writes
        self._pending_writes = []

        project = unify.active_project()

        batches: dict[str, list[dict]] = defaultdict(list)
        for entries, context in snapshot:
            batches[context].append(entries)
        total_events = len(snapshot)

        LOGGER.debug(
            "EventBus flush: %d events across %d contexts",
            total_events,
            len(batches),
        )

        for context, entries_list in batches.items():
            try:
                logs = unify.create_logs(
                    project=project,
                    context=context,
                    entries=entries_list,
                )
                LOGGER.debug(
                    "EventBus flush: wrote %d logs to context=%s",
                    len(entries_list),
                    context,
                )
            except Exception as exc:
                LOGGER.warning(
                    "EventBus flush failed: context=%s events=%d error=%s",
                    context,
                    len(entries_list),
                    exc,
                    exc_info=True,
                )
                continue

            # Mirror to aggregation contexts in bulk
            all_ctxs = _derive_all_contexts(context)
            if all_ctxs and logs:
                log_ids = [lg.id for lg in logs if lg.id is not None]
                if log_ids:
                    for all_ctx in all_ctxs:
                        try:
                            unify.add_logs_to_context(
                                log_ids,
                                context=all_ctx,
                                project=project,
                            )
                        except Exception as exc:
                            LOGGER.debug(
                                "EventBus flush: aggregation mirror failed: "
                                "context=%s target=%s error=%s",
                                context,
                                all_ctx,
                                exc,
                            )

    def join_published(self):
        """Ensures all published events have been uploaded."""
        self.flush()

    async def search(
        self,
        *,
        filter: Optional[str] = None,
        offset: Union[int, Dict[str, int]] = 0,
        limit: Union[int, Dict[str, int]] = 100,
        grouped_by_type: bool = False,
    ) -> Union[List[Event], Dict[str, List[Event]]]:
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

        # Use all known event types (not just those with populated deques)
        # so search works before hydration completes.
        all_types = set(self._deques) | set(self._specific_ctxs)

        # ----- per-type helpers ----------------------------------------------
        if combined_window:
            # grab *enough* from every queue (offset + limit) so the global
            # pass later has material to slice from
            per_type_limit = {t: offset + limit for t in all_types}
            per_type_offset = {t: 0 for t in all_types}  # skip globally later
        else:
            if isinstance(limit, int):
                per_type_limit = {t: limit for t in all_types}
            else:
                per_type_limit = {t: limit.get(t, 0) for t in all_types}

            if isinstance(offset, int):
                per_type_offset = {t: offset for t in all_types}
            else:
                per_type_offset = {t: offset.get(t, 0) for t in all_types}

        # ----------------------------------------------------------------------
        # 1. scan the deque -----------------------------------------------------
        _matches = lambda evt: self._match_filter(evt, filter)

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
                    evt = self._row_to_event(lg.entries)
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

            evts = [self._row_to_event(lg.entries, default_type=etype) for lg in logs]
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
        flat.sort(
            key=lambda e: (
                e.timestamp
                if e.timestamp.tzinfo
                else e.timestamp.replace(tzinfo=dt.UTC)
            ),
            reverse=True,
        )

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

        • Validates that event_type is a known registered type.
        • Rebuilds the internal :class:`collections.deque` so the new
          ``maxlen`` takes effect immediately, keeping **the most recent**
          messages up to *new_size*.
        """
        if new_size <= 0:
            raise ValueError("new_size must be a positive integer")

        # Validate known event type (raises ValueError if unknown)
        self.register_event_types(event_type)

        self._window_sizes[event_type] = new_size

        old_dq: Deque[Event] = self._deques.get(event_type, deque())
        # Re-hydrate deque (no automatic maxlen – manual trimming honours pins)
        new_dq: Deque[Event] = deque(old_dq)
        self._deques[event_type] = new_dq
        self._trim_window(event_type)

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
        await self.join_initialization()
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
                # (re-)attach the runtime callback
                sub.callback = callback

                # make sure we have a baseline that survived round-trip
                if every_seconds is not None and sub.last_timestamp is None:
                    # same helper used for brand new subscriptions
                    sub.last_row_id, sub.last_timestamp = await self._compute_baseline(
                        event_type,
                        filter,
                    )
                    self._persist_subscription_state(sub)

                return sub.subscription_id

        # ------------------------------------------------------------------
        # no existing subscription – create a fresh one **immediately** so
        # events published concurrently are not missed.  We then compute the
        # baseline *asynchronously* and update the persisted state.
        # ------------------------------------------------------------------
        sub = Subscription(
            event_type=event_type,
            filter=filter,
            count_step=every_n,
            time_step=every_seconds,
            callback=callback,
        )
        self._subscriptions[sub.subscription_id] = sub  # <-- race-free registration

        # Compute the baseline *after* registering so we don't miss events
        # that may arrive during the potentially slow I/O below.
        sub.last_row_id, sub.last_timestamp = await self._compute_baseline(
            event_type,
            filter,
        )

        self._persist_subscription_state(sub)

        return sub.subscription_id

    # ------------------------------------------------------------------
    def _persist_subscription_state(self, sub: Subscription) -> None:
        """Buffer subscription state for deferred batch upload."""
        self._pending_writes.append(
            (
                {
                    "subscription_id": sub.subscription_id,
                    "event_type": sub.event_type,
                    "filter": sub.filter,
                    "count_step": sub.count_step,
                    "time_step": sub.time_step,
                    "last_row_id": sub.last_row_id,
                    "last_timestamp": (
                        sub.last_timestamp.isoformat()
                        if isinstance(sub.last_timestamp, dt.datetime)
                        else sub.last_timestamp
                    ),
                },
                self._callbacks_ctx,
            ),
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
                    # Assign a sequence number and decide the *root* seq
                    self._callback_seq += 1
                    seq = self._callback_seq
                    root_seq = _CURRENT_ROOT_SEQ.get() or seq

                    # Propagate root-seq to descendants via context-var
                    token = _CURRENT_ROOT_SEQ.set(root_seq)
                    try:
                        task = asyncio.create_task(cb([evt]))
                    finally:
                        _CURRENT_ROOT_SEQ.reset(token)

                    # Attach sequencing metadata
                    setattr(task, "_eb_seq", seq)
                    setattr(task, "_eb_root_seq", root_seq)

                    # Track the new task and remove it when done
                    self._callback_futures.add(task)
                    task.add_done_callback(self._callback_futures.discard)
                else:
                    # For executor jobs we still assign a seq for filtering
                    self._callback_seq += 1
                    seq = self._callback_seq
                    root_seq = _CURRENT_ROOT_SEQ.get() or seq

                    fut = loop.run_in_executor(None, cb, [evt])
                    setattr(fut, "_eb_seq", seq)  # type: ignore[attr-defined]
                    setattr(fut, "_eb_root_seq", root_seq)  # type: ignore[attr-defined]

                    self._callback_futures.add(fut)  # type: ignore[arg-type]
                    fut.add_done_callback(self._callback_futures.discard)  # type: ignore[attr-defined]
            except RuntimeError:
                # No running loop (shutdown) – last-ditch synchronous call
                cb([evt])

    def set_default_window(self, new_size: int) -> None:
        """
        Change the default *in-memory* history window size for all event types.
        """
        self._default_window = new_size

    @property
    def ctxs(self):
        return self._specific_ctxs

    # ------------------------------------------------------------------
    def clear(self, delete_contexts: bool = True) -> None:

        # 1. Stop background tasks (pre-fill and periodic flush)
        try:
            if getattr(self, "_prefill_task", None) and not self._prefill_task.done():
                self._prefill_task.cancel()
        except Exception:  # pragma: no cover – defensive
            pass
        try:
            if (
                getattr(self, "_periodic_flush_task", None)
                and not self._periodic_flush_task.done()
            ):
                self._periodic_flush_task.cancel()
                self._periodic_flush_task = None
        except Exception:
            pass

        # 2. Flush all buffered writes before cleanup
        self.flush()

        # 3. Delete all Unify contexts owned by this EventBus instance
        if delete_contexts:
            # First remove children (…/Events/<TYPE>, …/Events/_callbacks, …)
            upstream_ctxs = list(unify.get_contexts(prefix=self._global_ctx) or [])
            for ctx in upstream_ctxs:
                unify.delete_context(ctx)

            # Finally remove the global Events context itself
            unify.delete_context(self._global_ctx)

        # 4. Re-initialise this *same* instance
        self._get_logger().clear_queue()
        self._get_logger().join()
        type(self).__init__(self)

    # ------------------------------------------------------------------
    async def ajoin_callbacks(
        self,
        *,
        cascade: bool = True,
    ) -> None:
        """Async version of join_callbacks for use in async contexts.

        Use this instead of join_callbacks when calling from async code to avoid
        deadlocks with nest_asyncio.
        """
        cutoff = self._callback_seq

        while True:
            to_await: list[asyncio.Future] = [
                t
                for t in list(self._callback_futures)
                if getattr(
                    t,
                    "_eb_root_seq" if cascade else "_eb_seq",
                    0,
                )
                <= cutoff
            ]

            if not to_await:
                return

            await asyncio.gather(*to_await, return_exceptions=True)
            # Yield to allow done_callbacks (scheduled via call_soon) to execute
            # and remove completed futures from _callback_futures
            await asyncio.sleep(0)

            if not cascade:
                return

    # ------------------------------------------------------------------
    def join_callbacks(
        self,
        *,
        cascade: bool = True,
    ) -> None:  # noqa: D401 – imperative name
        """Block until callback tasks have finished.

        Parameters
        ----------
        cascade : bool, default False
            If *False* only tasks that were already pending at the time
            of invocation are awaited – tasks spawned later are ignored.

            If *True* (default) the method also waits for **all** tasks that
            share the *root sequence* of those initial callbacks, i.e. any
            descendants spawned *indirectly* by them.  Unrelated new activity
            triggered by fresh, external events (and thus carrying a higher
            root-seq) is **not** awaited, preventing deadlocks under high
            throughput while still guaranteeing that entire cascades (e.g. the
            rolling summary hierarchy) have settled before returning."""

        async def _helper():  # inner coroutine – former implementation
            # Snapshot the highest sequence number currently assigned so that
            # we can identify all callbacks that belong to the *same* cascade
            # (root-seq ≤ cutoff).  New, unrelated work gets a fresh root-seq
            # > cutoff and is therefore ignored even in cascade-mode.

            cutoff = self._callback_seq

            while True:
                to_await: list[asyncio.Future] = [
                    t
                    for t in list(self._callback_futures)
                    if getattr(
                        t,
                        "_eb_root_seq" if cascade else "_eb_seq",
                        0,
                    )
                    <= cutoff
                ]

                if not to_await:
                    return

                await asyncio.gather(*to_await, return_exceptions=True)

                # If not cascading, we only wait once (historic behaviour)
                if not cascade:
                    return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No active event-loop in *this* thread.  However, the callback
            # tasks we want to await are bound to *some* loop (typically the
            # main thread's loop).  Use that loop to run the helper
            # coroutine via `run_coroutine_threadsafe` instead of creating a
            # fresh, incompatible loop.

            # Determine the target loop from one of the pending tasks (if any)
            pending = [t for t in list(self._callback_futures)]
            if pending:
                tgt_loop = pending[0].get_loop()

                # Schedule the helper coroutine onto the target loop and
                # block until it completes.
                fut = asyncio.run_coroutine_threadsafe(_helper(), tgt_loop)
                fut.result()
                return

            # Fallback – nothing to wait for: no pending tasks
            return
        else:
            # If we're already inside an event-loop, attempt a re-entrant run
            # using `nest_asyncio`; otherwise delegate to a background thread.

            try:
                import nest_asyncio  # type: ignore

                nest_asyncio.apply(loop)  # type: ignore[arg-type]
                loop.run_until_complete(_helper())
            except ModuleNotFoundError:
                # Fallback: run the coroutine in a background thread
                import threading

                exc: list[BaseException] | None = []

                # Instead of creating a **new** event-loop (which cannot await
                # tasks bound to the **original** loop), schedule the helper
                # coroutine *onto the existing running loop* in a
                # thread-safe manner and wait for its completion.

                def _runner():  # noqa: D401 – imperative helper
                    try:
                        fut = asyncio.run_coroutine_threadsafe(_helper(), loop)
                        # Wait for the coroutine to finish (propagates errors)
                        fut.result()
                    except BaseException as e:  # noqa: BLE001
                        exc.append(e)

                t = threading.Thread(target=_runner, daemon=True)
                t.start()
                t.join()
                if exc:
                    raise exc[0]

    # helper extracted from the old inline code
    async def _compute_baseline(
        self,
        event_type: str,
        filter: Optional[str],
    ) -> tuple[int, Optional[dt.datetime]]:
        """
        Return (last_row_id, last_timestamp) of the most-recent event of
        *event_type* that matches *filter* (or ``(-1, None)`` if none exist).
        """
        recent_logs = await asyncio.to_thread(
            unify.get_logs,
            context=self._specific_ctxs[event_type],
            sorting={"row_id": "descending"},
            limit=100,
        )

        for lg in recent_logs:  # newest → oldest
            evt = self._row_to_event(lg.entries, default_type=event_type)
            if self._match_filter(evt, filter):
                return evt.row_id, evt.timestamp
        return -1, None

    # ────────────────────────────  Static helpers  ────────────────────────────
    @staticmethod
    def _match_filter(evt: "Event", filter_expr: Optional[str]) -> bool:
        """Return True if *evt* satisfies the provided *filter_expr* (or if the
        expression is None/empty). The eval sandbox mirrors the original
        implementation but is now centralised for reuse across the class."""
        if not filter_expr:
            return True
        ns: dict[str, Any] = {
            "evt": evt,
            "event_type": evt.type,
            "type": evt.type,  # legacy alias
            **evt.model_dump(mode="python"),
        }
        return bool(eval(filter_expr, {"__builtins__": {}}, ns))

    @staticmethod
    def _row_to_event(row: dict, default_type: Optional[str] | None = None) -> "Event":
        """Convert a *flattened* Unify log row back into an :class:`Event`.

        The logic was previously duplicated in several places (prefill, search
        fetch, baseline computation). Centralising it greatly reduces code
        repetition and ensures consistent behaviour.

        Payloads are always returned as dicts - the Event validator will attempt
        to validate against the expected Pydantic model but stores as dict.
        """
        entries = row.copy()
        row_id = entries.pop("row_id", None)
        event_id = entries.pop("event_id", str(uuid4()))
        calling_id = entries.pop("calling_id", "")
        timestamp = entries.pop("event_timestamp", None) or entries.pop(
            "timestamp",
            None,
        )
        cls_path = entries.pop("payload_cls", "")
        etype = entries.pop("type", default_type)
        # Prefer non-flattened payload when available (newer schema)
        payload_json_str = entries.pop("payload_json", None)

        # Determine the payload dict: prefer JSON blob when present
        if payload_json_str is not None:
            try:
                payload_dict = json.loads(payload_json_str)
            except Exception:
                payload_dict = entries
        else:
            payload_dict = entries

        # Pass dict payload - Event validator handles validation and keeps as dict
        return Event(
            event_id=event_id,
            row_id=row_id,
            calling_id=calling_id,
            type=etype,
            timestamp=timestamp,
            payload=payload_dict,
            payload_cls=cls_path or "",
        )

    # ------------------------------------------------------------------
    #  Static subscription helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rows_to_subscriptions(rows: Iterable[Any]) -> Dict[str, Subscription]:
        """Convert raw Unify log *rows* into a mapping of Subscription objects.

        Multiple log entries may exist for the *same* ``subscription_id`` – we
        keep only the newest per ID (matching previous behaviour) before
        instantiating the Subscription models.
        """

        latest: Dict[str, dict] = {}
        for lg in rows:
            data = lg.entries.copy()
            # Skip any log rows that are not actual subscription snapshots.
            # These could include context metadata entries (e.g., __columns__)
            # or other housekeeping logs which do not carry a `subscription_id`
            # field.  Treating them as subscriptions would raise a KeyError and
            # break EventBus initialisation.
            sid = data.get("subscription_id")
            if not sid:
                # Safely ignore non-subscription rows to make hydration
                # tolerant of mixed-purpose contexts.
                continue

            latest[sid] = data

        return {
            sid: Subscription(
                subscription_id=sid,
                event_type=sdata["event_type"],
                filter=sdata.get("filter"),
                count_step=sdata.get("count_step"),
                time_step=sdata.get("time_step"),
                last_row_id=sdata.get("last_row_id", -1),
                last_timestamp=sdata.get("last_timestamp", ""),
            )
            for sid, sdata in latest.items()
        }


# ─────────────────────────   Global singleton (lazy)   ────────────────────


class _EventBusProxy:
    """Proxy that defers creation of the real :class:`EventBus` instance
    until :pyfunc:`unity.init` is invoked. Attempting to use the bus before
    initialisation raises a helpful :class:`RuntimeError`."""

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        self._inner: EventBus | None = None

    # internal – called by unity.init()
    def _set(self, bus: "EventBus") -> None:
        if self._inner is not None:
            raise RuntimeError("EVENT_BUS has already been initialised.")
        self._inner = bus

    # transparent proxy behaviour -----------------------------------
    def __getattr__(self, item):
        if self._inner is None:
            raise RuntimeError(
                "EVENT_BUS has not been initialised yet – call unity.init() first.",
            )
        return getattr(self._inner, item)

    def __bool__(self):
        return self._inner is not None


# Module-level placeholder – becomes the real EventBus once unity.init() runs
EVENT_BUS: "EventBus" = _EventBusProxy()  # type: ignore[assignment]


def _initialize_event_bus() -> "EventBus":
    """Internal helper used by :pyfunc:`unity.init` to instantiate the real
    :class:`EventBus` exactly once and wire it up to the module-level proxy.
    """
    if isinstance(EVENT_BUS, _EventBusProxy):
        bus = EventBus()
        EVENT_BUS._set(bus)  # type: ignore[attr-defined]
        return bus  # type: ignore[return-value]
    return EVENT_BUS  # type: ignore[return-value]


# Attach centralised docstring
EventBus.clear.__doc__ = CLEAR_METHOD_DOCSTRING
