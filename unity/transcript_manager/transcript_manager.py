from __future__ import annotations

import asyncio
import functools
import threading
from collections import Counter
from statistics import mean, median, pvariance, pstdev
from typing import List, Dict, Optional, Type, Union, Any, Callable, Literal

import unify
from pydantic import BaseModel
from ..common.embed_utils import ensure_vector_column
from ..common.log_utils import log as unity_log, _inject_private_fields, _add_to_all
from ..contact_manager.base import BaseContactManager
from ..manager_registry import ManagerRegistry
from .types.message import Message, UNASSIGNED
from .types.exchange import Exchange

# New: allow Contact objects to appear in messages
from ..contact_manager.types.contact import Contact
from ..common.llm_helpers import (
    methods_to_tool_dict,
    make_request_clarification_tool,
)
from ..common.llm_client import new_llm_client
from ..events.event_bus import EVENT_BUS, Event
from ..common.llm_policies import require_first
from ..common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..common.filter_utils import normalize_filter_expr
from ..events.manager_event_logging import (
    log_manager_call,
)
from .prompt_builders import build_ask_prompt
from .base import BaseTranscriptManager
from ..common.tool_spec import read_only, manager_tool, ToolSpec
from ..settings import SETTINGS
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from .storage import (
    provision_storage as _storage_provision,
    get_columns as _storage_get_columns,
    list_columns as _storage_list_columns,
    num_messages as _storage_num_messages,
    clear as _storage_clear,
    ensure_exchanges_records as _storage_ensure_exchanges,
)
from .search import (
    search_messages as _search_messages_impl,
    filter_messages as _filter_messages_impl,
    format_contacts_and_messages as _format_contacts_and_messages_impl,
)
from .images import (
    ensure_image_manager as _ensure_image_manager,
    get_images_for_message as _get_images_for_message_impl,
    ask_image as _ask_image_impl,
    attach_image_to_context as _attach_image_to_context_impl,
    attach_message_images_to_context as _attach_message_images_to_context_impl,
)
from ..common.context_registry import (
    ContextRegistry,
    SPACE_CONTEXT_PREFIX,
    TableContext,
)
from ..common.model_to_fields import model_to_fields
from ..common.metrics_utils import reduce_logs
from ..common.tool_outcome import ToolErrorException, ToolOutcome

TRANSCRIPTS_TABLE = "Transcripts"
EXCHANGES_TABLE = "Exchanges"


class TranscriptManager(BaseTranscriptManager):
    # ──────────────────────────────────────────────────────────────────────
    #  Class-level constants / configuration
    # ──────────────────────────────────────────────────────────────────────
    _LOGGER = unify.AsyncLoggerManager(name="TranscriptManager", num_consumers=16)

    # Vector embedding column names
    _MSG_EMB = "_content_emb"

    class Config:
        required_contexts = [
            TableContext(
                name=TRANSCRIPTS_TABLE,
                description="List of all timestamped messages sent between all contacts across all mediums.",
                fields=model_to_fields(Message),
                unique_keys={"message_id": "int"},
                auto_counting={"message_id": None},
                foreign_keys=[
                    {
                        "name": "sender_id",
                        "references": "Contacts.contact_id",
                        "on_delete": "SET NULL",
                        "on_update": "CASCADE",
                    },
                    {
                        "name": "receiver_ids[*]",
                        "references": "Contacts.contact_id",
                        "on_delete": "SET NULL",
                        "on_update": "CASCADE",
                    },
                    {
                        "name": "exchange_id",
                        "references": "Exchanges.exchange_id",
                        "on_delete": "CASCADE",
                        "on_update": "CASCADE",
                    },
                    {
                        "name": "images[*].raw_image_ref.image_id",
                        "references": "Images.image_id",
                        "on_delete": "SET NULL",
                        "on_update": "CASCADE",
                    },
                ],
            ),
            TableContext(
                name=EXCHANGES_TABLE,
                description="One row per conversation exchange/thread with optional metadata.",
                fields=model_to_fields(Exchange),
                unique_keys={"exchange_id": "int"},
                auto_counting={"exchange_id": None},
            ),
        ]

    # ──────────────────────────────────────────────────────────────────────
    #  Construction & tool registration
    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        *,
        contact_manager: Optional[BaseContactManager] = None,
        rolling_summary_in_prompts: bool = True,
    ) -> None:
        """
        Responsible for *searching through* the full transcripts across all communcation channels exposed to the assistant.
        """
        super().__init__()
        self.include_in_multi_assistant_table = True

        if contact_manager is not None:
            self._contact_manager = contact_manager
        else:
            self._contact_manager = ManagerRegistry.get_contact_manager()

        ask_tools = {
            **methods_to_tool_dict(
                ToolSpec(
                    fn=self._contact_manager.ask,
                    display_label="Looking up contact details",
                ),
                include_class_name=True,
            ),
            **methods_to_tool_dict(
                ToolSpec(
                    fn=self._filter_messages,
                    display_label="Filtering conversation messages",
                ),
                ToolSpec(
                    fn=self._search_messages,
                    display_label="Searching conversation messages",
                ),
                ToolSpec(fn=self._reduce, display_label="Summarising conversations"),
                include_class_name=False,
            ),
        }

        self._transcripts_ctx = ContextRegistry.get_context(self, TRANSCRIPTS_TABLE)
        self._exchanges_ctx = ContextRegistry.get_context(self, EXCHANGES_TABLE)
        self._image_destinations_by_id: dict[int, str] = {}

        # Image support: lazy-safe image manager and image-aware tools
        _ensure_image_manager(self)
        ask_tools.update(
            methods_to_tool_dict(
                ToolSpec(
                    fn=self._get_images_for_message,
                    display_label="Retrieving message images",
                ),
                ToolSpec(fn=self._ask_image, display_label="Analysing an image"),
                ToolSpec(
                    fn=self._attach_image_to_context,
                    display_label="Attaching image to context",
                ),
                ToolSpec(
                    fn=self._attach_message_images_to_context,
                    display_label="Attaching message images",
                ),
                include_class_name=False,
            ),
        )
        self.add_tools("ask", ask_tools)

        # ── Async logging (mirrors EventBus) ────────────────────────────────
        # Using a dedicated logger means log_create() returns immediately,
        # leaving the actual network I/O to an internal worker thread.
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._pending_async_log_fallbacks: list[dict[str, Any]] = []

        # Provision storage (contexts, fields, columns)
        self._provision_storage()

    def _context_for_root(self, root_context: str, table_name: str) -> str:
        """Return one concrete transcript-manager table context under a root."""

        return f"{root_context.strip('/')}/{table_name}"

    def _transcripts_context_for_destination(self, destination: str | None) -> str:
        """Resolve a public destination into one concrete Transcripts context."""

        root_context = ContextRegistry.write_root(
            self,
            TRANSCRIPTS_TABLE,
            destination=destination,
        )
        return self._context_for_root(root_context, TRANSCRIPTS_TABLE)

    def _exchanges_context_for_destination(self, destination: str | None) -> str:
        """Resolve a public destination into one concrete Exchanges context."""

        root_context = ContextRegistry.write_root(
            self,
            EXCHANGES_TABLE,
            destination=destination,
        )
        return self._context_for_root(root_context, EXCHANGES_TABLE)

    def _read_transcript_contexts(self) -> list[str]:
        """Return ordered concrete Transcripts contexts visible to this assistant."""

        return list(
            dict.fromkeys(
                self._context_for_root(root, TRANSCRIPTS_TABLE)
                for root in ContextRegistry.read_roots(self, TRANSCRIPTS_TABLE)
            ),
        )

    def _read_exchange_contexts(self) -> list[str]:
        """Return ordered concrete Exchanges contexts visible to this assistant."""

        return list(
            dict.fromkeys(
                self._context_for_root(root, EXCHANGES_TABLE)
                for root in ContextRegistry.read_roots(self, EXCHANGES_TABLE)
            ),
        )

    def _root_context_for_move(self, table_name: str, from_root: str) -> str:
        """Resolve a move source root label into a concrete root context."""

        if from_root == "personal":
            return ContextRegistry.write_root(self, table_name, destination=None)
        if from_root.startswith("space:"):
            return ContextRegistry.write_root(self, table_name, destination=from_root)
        if from_root.startswith(SPACE_CONTEXT_PREFIX):
            destination = f"space:{from_root.split('/')[1]}"
            return ContextRegistry.write_root(self, table_name, destination=destination)
        return from_root.rstrip("/")

    def _should_add_to_all_context(self, context: str) -> bool:
        """Return whether writes to this context should mirror into All/* contexts."""

        return self.include_in_multi_assistant_table and not context.startswith(
            SPACE_CONTEXT_PREFIX,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Public API (English-only entrypoints for the LLM)
    # ──────────────────────────────────────────────────────────────────────
    # English-Text Question
    @functools.wraps(BaseTranscriptManager.ask, updated=())
    @manager_tool
    @log_manager_call(
        "TranscriptManager",
        "ask",
        payload_key="question",
        display_label="Reviewing conversations",
    )
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        tool_policy: Union[
            Literal["default"],
            Callable[[int, Dict[str, Any]], tuple[str, Dict[str, Any]]],
            None,
        ] = "default",
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        # ── 0.  Build the *live* tools-dict (may include clarification helper) ──
        tools = dict(self.get_tools("ask"))
        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

            async def _on_clar_req(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "TranscriptManager",
                                "method": "ask",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_clar_ans(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "TranscriptManager",
                                "method": "ask",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

        # ── 1.  Build LLM client & inject dynamic system-prompt ───────────
        client = new_llm_client()
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_ask_prompt(
                tools,
                num_messages=_storage_num_messages(self),
                transcript_columns=_storage_list_columns(self),
                contact_columns=self._contact_manager._list_columns(),
                include_activity=include_activity,
            ).to_list(),
        )

        # Decide effective tool policy (default requires search_messages first).
        if tool_policy == "default":
            effective_tool_policy = require_first("search_messages")
        else:
            effective_tool_policy = tool_policy

        # ── 2.  Launch the interactive tool-use loop ───────────────────────
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=effective_tool_policy,
            handle_cls=(
                ReadOnlyAskGuardHandle if SETTINGS.UNITY_READONLY_ASK_GUARD else None
            ),
            response_format=response_format,
            clarification_queues=_clar_queues,
            on_clarification_request=_on_clar_req,
            on_clarification_answer=_on_clar_ans,
        )

        # ── 4.  Optional reasoning exposure  ───────────────────────────────
        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    @functools.wraps(BaseTranscriptManager.clear, updated=())
    def clear(self) -> None:

        _storage_clear(self)

    # (Optional) Public programmatic helpers (non-LLM)
    def log_messages(
        self,
        messages: Union[
            Union[Dict[str, Any], Message],
            List[Union[Dict[str, Any], Message]],
        ],
        synchronous: bool = False,
        _skip_event_bus: bool = False,
        destination: str | None = None,
    ) -> List[Message]:
        """
        Insert one or more messages into the backing store.

        This enhanced variant additionally accepts **Contact** objects in place
        of numeric ``sender_id`` / ``receiver_ids``.  When such a Contact has
        its ``contact_id`` set to the sentinel ``-1`` (meaning *not yet
        persisted*) the contact is **created on-the-fly** via
        :pyfunc:`ContactManager._create_contact` before the message is logged.

        Parameters
        ----------
        messages : dict | Message | list[dict | Message]
            One or more messages to log. Each message can be either:
            - A dictionary following the
              :class:`unity.transcript_manager.types.message.Message` schema –
              where ``sender_id`` / ``receiver_ids`` may contain ``Contact``
              objects instead of ints.
            - A :class:`~unity.transcript_manager.types.message.Message` instance
              whose *id* fields may likewise contain ``Contact`` objects.
            - A list with any combination of the above.
        synchronous : bool, default=False
            If True, messages will be logged in order synchronously. If False,
            messages may be logged asynchronously in any order.
        destination : str | None, default None
            Internal routing destination for the transcript rows. Omitted means
            personal. Session synthesis owns this value for implicit writes.

        Notes
        -----
        This method requires an explicit ``exchange_id`` on every message. To create
        a brand‑new exchange (i.e. when no id exists yet), call
        :pyfunc:`log_first_message_in_new_exchange` instead.

        Returns
        -------
        list[Message]
            The created messages as validated ``Message`` models, populated
            with assigned identifiers (e.g., ``message_id`` and
            ``exchange_id``) exactly as returned by the storage backend. The
            shape mirrors ``_filter_messages`` (list of ``Message``).
        """

        # ── 0. Early-exit on empty input ────────────────────────────────────
        if not messages:
            return []

        if not isinstance(messages, list):
            messages = [messages]

        try:
            transcripts_context = self._transcripts_context_for_destination(destination)
            exchanges_context = self._exchanges_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]

        # ── 1. Helper to ensure we have a numeric contact-id ───────────────
        # Derive the built-in (canonical) Contact fields *dynamically* from the
        # `Contact` model itself – this avoids hard-coding and ensures there is
        # exactly one source of truth across the code-base.
        built_in_fields = set(Contact.model_fields.keys())

        contact_cache: Dict[int, int] = {}

        def _ensure_contact_id(c: Union[int, Contact]) -> int:
            """Return an existing or newly-created **contact_id** for *c*."""

            # Fast-path: already an int → nothing to do
            if not isinstance(c, Contact):
                if c is None:
                    raise ValueError(
                        "sender_id / receiver_ids cannot be None – either provide an int or a Contact instance.",
                    )
                return int(c)

            # If the Contact already had a valid id – reuse it
            if c.contact_id is not None and c.contact_id != -1:
                return int(c.contact_id)

            # Deduplicate identical Contact objects within the same call
            obj_key = id(c)
            if obj_key in contact_cache:
                return contact_cache[obj_key]

            # Build kwargs for _create_contact using *non-None* built-in fields
            # detected directly from the `Contact` schema instead of hard-coding
            # the field names.  This ensures any future additions to the
            # Contact model automatically propagate here.

            full_data = c.model_dump(exclude_none=True)  # include only defined fields

            # Separate canonical Contact fields from any custom extras
            create_kwargs: Dict[str, Any] = {
                k: v
                for k, v in full_data.items()
                if k in built_in_fields and k != "contact_id"
            }

            # Merge any extra / custom fields directly into the creation kwargs
            for k, v in c.model_dump().items():
                if k not in built_in_fields and v is not None:
                    create_kwargs[k] = v

            # Synchronously create the new contact
            outcome = self._contact_manager._create_contact(
                **create_kwargs,
                destination=destination,
            )
            try:
                new_cid = int(outcome["details"]["contact_id"])
            except Exception:
                # Fall back to best-effort id extraction / raise
                raise RuntimeError(
                    "Failed to extract contact_id from ContactManager outcome: "
                    f"{outcome}",
                )

            # Update cache and the original Contact instance for consistency
            contact_cache[obj_key] = new_cid
            try:
                c.contact_id = new_cid  # type: ignore[attr-defined]
            except Exception:
                pass  # read-only / frozen instance – safe to ignore

            return new_cid

        # ── 2. Normalise each input payload into Message objects ───────────
        normalised_messages: List[Message] = []
        for raw in messages:
            # Convert to dict early so we can mutate fields easily
            if isinstance(raw, Message):
                payload: Dict[str, Any] = raw.model_dump(mode="python")
            else:  # assume mapping
                payload = dict(raw)

            # Enforce explicit exchange_id on all messages unless explicitly allowed (internal use only)
            exid_val = payload.get("exchange_id", None)
            try:
                # Treat UNASSIGNED (-1) and None as missing
                if exid_val is None or int(exid_val) < 0:
                    raise ValueError(
                        "exchange_id is required when calling TranscriptManager.log_messages. "
                        "To start a brand-new exchange, use TranscriptManager.log_first_message_in_new_exchange(message, exchange_initial_metadata=...).",
                    )
            except (TypeError, ValueError):
                # Non-int or unparsable also counts as missing/invalid
                raise ValueError(
                    "exchange_id must be an integer when calling TranscriptManager.log_messages. "
                    "To start a brand-new exchange, use TranscriptManager.log_first_message_in_new_exchange(message, exchange_initial_metadata=...).",
                )

            # Ensure required keys exist
            if "receiver_ids" not in payload:
                raise ValueError("Each message must include 'receiver_ids'.")

            # Replace any Contact objects with their numeric ids
            payload["sender_id"] = _ensure_contact_id(payload.get("sender_id"))
            payload["receiver_ids"] = [
                _ensure_contact_id(r) for r in payload.get("receiver_ids", [])
            ]

            # Re-instantiate Message model for validation
            normalised_messages.append(Message(**payload))

        initial_exchange_ids: set[int] = set()
        initial_eid_to_medium: Dict[int, str] = {}
        for message in normalised_messages:
            exchange_id = int(message.exchange_id)
            initial_exchange_ids.add(exchange_id)
            initial_eid_to_medium.setdefault(exchange_id, str(message.medium))
        if initial_exchange_ids:
            self._ensure_exchanges_records(
                initial_exchange_ids,
                eid_to_medium=initial_eid_to_medium,
                context=exchanges_context,
            )

        # ── 3. Dump POST-ready JSON for each message ──────────────────────
        msg_entries = [m.to_post_json() for m in normalised_messages]

        # ── 4. Persist messages and publish EventBus notifications ───────
        from ..events.event_bus import EVENT_BUS, Event  # local import to avoid cycles

        async def _publish_message(msg: Message) -> None:
            try:
                await EVENT_BUS.publish(
                    Event(
                        type="Message",
                        timestamp=msg.timestamp,
                        payload=msg,
                    ),
                    blocking=synchronous,
                )
            except Exception:
                # Defensive – never propagate EventBus issues to caller
                pass

        created_messages: List[Message] = []

        for entries, _orig_msg in zip(msg_entries, normalised_messages):
            # Ensure correct creation order by performing contact creation *before*
            # the logger call (already satisfied above).  Now we can log safely.
            if synchronous:
                # Sync path: block until backend responds, get assigned IDs
                log = unity_log(
                    context=transcripts_context,
                    **entries,
                    new=True,
                    mutable=True,
                    add_to_all_context=self._should_add_to_all_context(
                        transcripts_context,
                    ),
                )

                # Build a Message directly from the POST response
                persisted_payload = {
                    k: log.entries.get(k) for k in Message.model_fields.keys()
                }
                # Remove any None values for id fields so the validator can apply sentinel if needed
                if persisted_payload.get("message_id") is None:
                    persisted_payload.pop("message_id", None)
                if persisted_payload.get("exchange_id") is None:
                    persisted_payload.pop("exchange_id", None)

                created_msg = Message(**persisted_payload)
            else:
                # Async path: fire-and-forget, don't block on network I/O
                # Inject private fields (same as sync path via unity_log)
                entries_with_private = _inject_private_fields(entries)
                entries_with_private["explicit_types"] = {
                    key: {"mutable": True}
                    for key in entries_with_private
                    if key not in {"explicit_types", "infer_untyped_fields"}
                }
                future = self._get_logger().log_create(
                    project=unify.active_project(),
                    context={"name": transcripts_context},
                    entries=entries_with_private,
                )
                # Add callback to mirror to aggregation contexts when log is created
                # and to preserve the write if the async worker fails after enqueue.
                if future is not None:
                    ctx = transcripts_context
                    should_add_to_all = self._should_add_to_all_context(ctx)
                    fallback_entries = dict(entries)
                    fallback_state: dict[str, Any] = {
                        "future": future,
                        "context": ctx,
                        "entries": fallback_entries,
                        "add_to_all": should_add_to_all,
                        "handled": False,
                        "fallback_lock": threading.Lock(),
                    }
                    self._pending_async_log_fallbacks.append(fallback_state)

                    def _on_log_created(
                        fut,
                        state=fallback_state,
                    ):
                        try:
                            log_id = fut.result()
                            if log_id:
                                if state["add_to_all"]:
                                    _add_to_all([log_id], state["context"])
                                state["handled"] = True
                        except Exception:
                            self._fallback_async_log_create(state)

                    future.add_done_callback(_on_log_created)
                # In async mode, we don't wait for the response, so use the
                # original message (IDs may not be assigned yet)
                created_msg = _orig_msg

            created_messages.append(created_msg)

            if not _skip_event_bus:
                try:
                    # If we're inside an event-loop schedule the coroutine there …
                    loop = asyncio.get_running_loop()
                    loop.create_task(_publish_message(created_msg))
                except RuntimeError:
                    # … otherwise create a *temporary* loop so the event isn't lost.
                    asyncio.run(_publish_message(created_msg))

        # ── 5. Inactivity-followup activity sync (best-effort, async) ──────
        # Tell orchestra that the assistant just exchanged a message so its
        # inactivity-followup routine sees fresh ``last_correspondence_at``
        # and clears any pending ``last_followup_sent_at``. Skipped for
        # internal/system-bus-only writes. Dispatched to a daemon thread so
        # the network round-trip never blocks message logging; failures are
        # swallowed inside ``touch_assistant_activity``.
        if not _skip_event_bus and created_messages:
            try:
                from .activity_sync import touch_assistant_activity

                agent_id = getattr(SESSION_DETAILS.assistant, "agent_id", None)
                if agent_id is not None:
                    threading.Thread(
                        target=touch_assistant_activity,
                        args=(agent_id,),
                        daemon=True,
                        name="touch_assistant_activity",
                    ).start()
            except Exception:
                pass

        return created_messages

    def join_published(self):
        self._get_logger().join()
        for state in list(self._pending_async_log_fallbacks):
            future = state["future"]
            if state.get("handled") or not future.done():
                continue
            try:
                future.result()
            except Exception:
                self._fallback_async_log_create(state)
        self._pending_async_log_fallbacks = [
            state
            for state in self._pending_async_log_fallbacks
            if not state.get("handled")
        ]

    @staticmethod
    def _async_entry_values_match(expected: Any, actual: Any) -> bool:
        """Return whether an async fallback candidate already exists."""

        if expected is None:
            return actual is None
        if isinstance(expected, list):
            return list(actual or []) == expected
        if str(expected) == str(actual):
            return True
        expected_text = str(expected).replace("+00:00", "Z")
        actual_text = str(actual).replace("+00:00", "Z")
        return expected_text == actual_text

    def _find_existing_async_log_id(self, state: dict[str, Any]) -> Optional[int]:
        """Find a row written by the async logger before fallback persistence runs."""

        entries = state["entries"]
        exchange_id = entries.get("exchange_id")
        if exchange_id is None:
            return None
        try:
            rows = unify.get_logs(
                context=state["context"],
                filter=f"exchange_id == {int(exchange_id)}",
                limit=20,
            )
        except Exception:
            return None

        match_fields = (
            "medium",
            "sender_id",
            "receiver_ids",
            "timestamp",
            "content",
            "exchange_id",
        )
        for row in rows:
            row_entries = getattr(row, "entries", {}) or {}
            if all(
                self._async_entry_values_match(
                    entries.get(field),
                    row_entries.get(field),
                )
                for field in match_fields
                if field in entries
            ):
                try:
                    return int(row.id)
                except Exception:
                    return None
        return None

    def _fallback_async_log_create(self, state: dict[str, Any]) -> None:
        """Synchronously persist a row if the async logger dropped it."""

        fallback_lock = state.get("fallback_lock")
        if fallback_lock is not None:
            with fallback_lock:
                self._fallback_async_log_create_unlocked(state)
            return
        self._fallback_async_log_create_unlocked(state)

    def _fallback_async_log_create_unlocked(self, state: dict[str, Any]) -> None:
        """Persist one async fallback while the caller holds the state lock."""

        if state.get("handled"):
            return
        existing_log_id = None
        for delay_seconds in (0.0, 0.25, 0.75, 1.5):
            if delay_seconds:
                try:
                    import time

                    time.sleep(delay_seconds)
                except Exception:
                    pass
            existing_log_id = self._find_existing_async_log_id(state)
            if existing_log_id is not None:
                break
        if existing_log_id is not None:
            if state["add_to_all"]:
                try:
                    _add_to_all([existing_log_id], state["context"])
                except Exception:
                    pass
            state["handled"] = True
            return
        try:
            unity_log(
                context=state["context"],
                **state["entries"],
                new=True,
                mutable=True,
                add_to_all_context=state["add_to_all"],
            )
            state["handled"] = True
        except Exception:
            pass

    @staticmethod
    def build_plain_transcript(
        messages: list[dict],
        *,
        contact_manager: Optional["ContactManager"] = None,
    ) -> str:
        """Return a plain-text transcript (``Full Name: content``) for ``messages``.

        Accepts two input shapes:

        1. Raw EventBus events::

               {"kind": "message", "data": {"sender_id": 3, "content": "Hi"}}

        2. Simplified sandbox dicts::

               {"sender": "Jane Smith", "content": "Hi"}

        An optional ``contact_manager`` can be supplied; otherwise a fresh
        ``ContactManager`` is constructed lazily. Numeric ``sender_id`` values are
        resolved to full names (first + surname when available).

        Parameters
        ----------
        messages : list[dict]
            The list of message-like dictionaries to convert.
        contact_manager : ContactManager | None, optional
            Manager used to resolve numeric ``sender_id`` values to names. If not
            provided, a new ``ContactManager`` instance is constructed lazily.

        Returns
        -------
        str
            The plain-text transcript with one line per message in the format
            ``Full Name: content``.
        """

        # Local import avoids widening module dependencies at import-time
        from unity.contact_manager.contact_manager import (
            ContactManager as _CM,
        )  # noqa: WPS433

        cm = contact_manager or _CM()

        name_cache: dict[int, str] = {}

        def _name_for_cid(cid: int) -> str:  # noqa: D401 – helper
            if cid in name_cache:
                return name_cache[cid]
            try:
                recs = cm.filter_contacts(filter=f"contact_id == {cid}", limit=1)
                if recs:
                    rec = recs[0]
                    full = " ".join(
                        p for p in [rec.first_name, rec.surname] if p
                    ).strip()
                    if not full:
                        full = (rec.first_name or "").strip()
                    if full:
                        name_cache[cid] = full
                        return full
            except Exception:
                pass
            name_cache[cid] = str(cid)
            return name_cache[cid]

        lines: list[str] = []
        for itm in messages:
            # Shape 1 – live EventBus message
            if "kind" in itm:
                if itm.get("kind") != "message":
                    continue
                data = itm.get("data", {})
                sender_val = data.get("sender_id")
                content_val = data.get("content", "")
                if sender_val is None:
                    continue
                sender_name = _name_for_cid(int(sender_val))
            else:  # Shape 2 – sandbox simplified dict
                sender_name = str(itm.get("sender"))
                content_val = str(itm.get("content", ""))

            lines.append(f"{sender_name}: {content_val}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    #  Private tools (LLM-exposed to tool loops)
    #    – these are the underscore-prefixed methods you pass into add_tools
    # ──────────────────────────────────────────────────────────────────────
    @read_only
    def _search_messages(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> Dict[str, Any]:
        """
        Semantic search across transcript messages (two-table aware).

        Parameters
        ----------
        references : Dict[str, str] | None, default None
            Mapping of source expressions to reference text for semantic search.
        k : int, default 10
            Maximum number of results to return. Must be <= 1000.

        Returns
        -------
        Dict[str, Any]
            Search results with contact information.
        """
        return _search_messages_impl(self, references=references, k=k)

    @read_only
    def _reduce(
        self,
        *,
        metric: str,
        keys: str | list[str],
        filter: Optional[str | dict[str, str]] = None,
        group_by: Optional[str | list[str]] = None,
    ) -> Any:
        """
        Compute reduction metrics over the primary transcripts/messages table.

        Parameters
        ----------
        metric : str
            Reduction metric to compute. Supported values (case-insensitive) are
            ``\"sum\"``, ``\"mean\"``, ``\"var\"``, ``\"std\"``, ``\"min\"``,
            ``\"max\"``, ``\"median\"``, ``\"mode\"``, and ``\"count\"``.
        keys : str | list[str]
            One or more numeric message fields to aggregate (for example
            ``\"message_id\"`` or duration/length columns). A single column name
            returns a scalar; a list of column names computes the metric
            independently per key and returns a ``{key -> value}`` mapping.
        filter : str | dict[str, str] | None, default None
            Optional row-level filter expression(s) in the same Python syntax as
            :py:meth:`_filter_messages`. When a string, the expression is applied
            uniformly; when a dict, each key maps to its own filter expression.
        group_by : str | list[str] | None, default None
            Optional message field(s) to group by, for example ``\"medium\"`` or
            ``\"sender_id\"``. Use a single column name for one grouping level,
            or a list such as ``[\"medium\", \"sender_id\"]`` to group
            hierarchically in that order. When provided, the result becomes a
            nested mapping keyed by group values, mirroring
            :func:`unify.get_logs_metric`.

        Returns
        -------
        Any
            Metric value(s) computed over the transcripts context:

            * Single key, no grouping  → scalar (float/int/str/bool).
            * Multiple keys, no grouping → ``dict[key -> scalar]``.
            * With grouping             → nested ``dict`` keyed by group values.
        """
        metric_name = metric.strip().lower()
        key_names = [keys] if isinstance(keys, str) else list(keys)
        group_fields = (
            []
            if group_by is None
            else ([group_by] if isinstance(group_by, str) else list(group_by))
        )

        def _metric_value(rows: list[dict[str, Any]], key: str) -> Any:
            values = [row.get(key) for row in rows if row.get(key) is not None]
            if metric_name == "count":
                return len(values)
            if not values:
                return None
            if metric_name == "mode":
                return Counter(values).most_common(1)[0][0]

            numeric_values = [float(value) for value in values]
            if metric_name == "sum":
                return sum(numeric_values)
            if metric_name == "mean":
                return mean(numeric_values)
            if metric_name == "var":
                return pvariance(numeric_values)
            if metric_name == "std":
                return pstdev(numeric_values)
            if metric_name == "min":
                return min(numeric_values)
            if metric_name == "max":
                return max(numeric_values)
            if metric_name == "median":
                return median(numeric_values)
            return reduce_logs(
                context=self._transcripts_ctx,
                metric=metric,
                keys=key,
                filter=filter,
                group_by=group_by,
            )

        def _nest_grouped_values(grouped_values: dict[tuple[Any, ...], Any]) -> dict:
            nested: dict[Any, Any] = {}
            for group_key, value in grouped_values.items():
                cursor = nested
                for part in group_key[:-1]:
                    cursor = cursor.setdefault(part, {})
                cursor[group_key[-1]] = value
            return nested

        def _rows_for_key(key: str) -> list[dict[str, Any]]:
            if isinstance(filter, dict):
                filter_expr = filter.get(key)
            else:
                filter_expr = filter
            normalized_filter = normalize_filter_expr(filter_expr)
            fields = list(dict.fromkeys([key, *group_fields]))
            rows: list[dict[str, Any]] = []
            for context in self._read_transcript_contexts():
                offset = 0
                batch_size = 1000
                while True:
                    logs = unify.get_logs(
                        context=context,
                        filter=normalized_filter,
                        offset=offset,
                        limit=batch_size,
                        from_fields=fields,
                    )
                    if not logs:
                        break
                    rows.extend(dict(log.entries) for log in logs)
                    if len(logs) < batch_size:
                        break
                    offset += batch_size
            return rows

        def _reduce_key(key: str) -> Any:
            rows = _rows_for_key(key)
            if not group_fields:
                return _metric_value(rows, key)

            buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            for row in rows:
                group_key = tuple(row.get(field) for field in group_fields)
                buckets.setdefault(group_key, []).append(row)
            return _nest_grouped_values(
                {
                    group_key: _metric_value(bucket_rows, key)
                    for group_key, bucket_rows in buckets.items()
                },
            )

        result_by_key = {key: _reduce_key(key) for key in key_names}
        return result_by_key[keys] if isinstance(keys, str) else result_by_key

    @read_only
    def _filter_messages(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int | None = 100,
    ) -> Dict[str, Any]:
        """
        Filter transcript messages using an exact column-wise boolean expression.
        The expression must be expressed in valid python syntax.

        Parameters
        ----------
        filter : str | None, default None
            A Python boolean expression evaluated with column names in scope.
            When None, returns all messages.
        offset : int, default 0
            Zero-based index of the first result to include.
        limit : int | None, default 100
            Maximum number of records to return. Must be <= 1000.

        Returns
        -------
        Dict[str, Any]
            Filtered messages with contact information.
        """
        return _filter_messages_impl(self, filter=filter, offset=offset, limit=limit)

    def update_contact_id(
        self,
        *,
        original_contact_id: int,
        new_contact_id: int,
    ) -> Dict[str, Any]:
        """Replace **all** occurrences of *original_contact_id* with *new_contact_id*
        across every transcript message.

        The substitution is applied to both the ``sender_id`` **and** every entry
        inside ``receiver_ids``.  The update is *in-place* – no new rows are
        created.

        Parameters
        ----------
        original_contact_id : int
            The contact identifier to be replaced.
        new_contact_id : int
            The replacement contact identifier.

        Returns
        -------
        dict
            ToolOutcome-style payload summarising how many messages were
            updated.
        """
        if original_contact_id == new_contact_id:
            raise ValueError("original_contact_id and new_contact_id must differ.")

        total_updates = 0

        # ── 1.  Bulk update all *sender_id* occurrences ────────────────────
        for context in self._read_transcript_contexts():
            sender_log_ids = unify.get_logs(
                context=context,
                filter=f"sender_id is not None and sender_id == {original_contact_id}",
                return_ids_only=True,
            )
            if sender_log_ids:
                unify.update_logs(
                    logs=sender_log_ids,
                    context=context,
                    entries={"sender_id": new_contact_id},
                    overwrite=True,
                )
                total_updates += len(sender_log_ids)

            # ── 2.  Update all *receiver_ids* lists containing the old id ──────
            receiver_logs = unify.get_logs(
                context=context,
                filter=f"{original_contact_id} in receiver_ids",
                return_ids_only=False,
            )
            for lg in receiver_logs:
                rids = lg.entries.get("receiver_ids", [])
                if not isinstance(rids, list):  # defensive – should always be list
                    continue

                updated_rids = [
                    (new_contact_id if rid == original_contact_id else rid)
                    for rid in rids
                ]
                # Optional: remove duplicates while preserving order
                seen: set[int] = set()
                deduped_rids: list[int] = []
                for rid in updated_rids:
                    if rid not in seen:
                        seen.add(rid)
                        deduped_rids.append(rid)

                # Only write when the list actually changed
                if deduped_rids != rids:
                    unify.update_logs(
                        logs=lg.id if hasattr(lg, "id") else lg,
                        context=context,
                        entries={"receiver_ids": deduped_rids},
                        overwrite=True,
                    )
                    total_updates += 1

        return {
            "outcome": "contact ids updated",
            "details": {
                "old_contact_id": original_contact_id,
                "new_contact_id": new_contact_id,
                "updated_messages": total_updates,
            },
        }

    def update_message_images(
        self,
        message_id: int,
        images: list[dict],
        *,
        destination: str | None = None,
    ) -> None:
        """Attach or replace images on an already-logged transcript message."""
        try:
            context = self._transcripts_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]
        log_ids = unify.get_logs(
            context=context,
            filter=f"message_id == {message_id}",
            return_ids_only=True,
        )
        if log_ids:
            unify.update_logs(
                logs=log_ids,
                context=context,
                entries={"images": images},
                overwrite=True,
            )

    # ──────────────────────────────────────────────────────────────────────
    #  Image tools
    # ──────────────────────────────────────────────────────────────────────
    @read_only
    def _get_images_for_message(self, *, message_id: int) -> List[Dict[str, Any]]:
        """Return image metadata (no raw data) for images referenced by a message.

        Output schema (list of objects):
        - image_id: int
        - caption: str | None
        - timestamp: str (ISO8601)
        - annotation: str  → freeform explanation of how the image relates to the text
        """
        return _get_images_for_message_impl(self, message_id=message_id)

    @read_only
    async def _ask_image(
        self,
        *,
        image_id: int,
        question: str,
        destination: str | None = None,
    ) -> str:
        """Ask a one‑off question about a specific stored image.

        This helper mirrors the behaviour of :pyfunc:`ImageHandle.ask` but is
        exposed as a TranscriptManager tool that requires an explicit
        ``image_id``. It sends the underlying image to a vision‑capable model as
        an image block and returns a textual answer only.

        Parameters
        ----------
        image_id : int
            Identifier of the image to analyse. If the image's ``data`` is a
            Google Cloud Storage URL, a short‑lived signed URL is generated to
            grant the model access; otherwise the stored base64 is converted to
            a ``data:image/...;base64,`` URL.
        question : str
            Natural‑language question to ask about the image.
        destination : str | None
            Root containing the image metadata. When omitted, the manager uses
            the most recent message-image lookup for the image id, falling back
            to personal memory.

        Returns
        -------
        str
            Text answer from the vision model.

        Notes
        -----
            This method does not persist the visual context for follow‑up turns.
        """
        return await _ask_image_impl(
            self,
            image_id=image_id,
            question=question,
            destination=destination,
        )

    def _attach_image_to_context(
        self,
        *,
        image_id: int,
        note: Optional[str] = None,
        destination: str | None = None,
    ) -> Dict[str, Any]:
        """Attach a single image (by id) as raw base64 for persistent context.

        Loads the image bytes for ``image_id`` and returns a payload suitable
        for inclusion as an image block in the current tool‑use loop. Behaviour
        aligns with :pyfunc:`ImageHandle.raw` for source resolution:
        - If ``data`` is a GCS URL (``gs://`` or
          ``https://storage.googleapis.com/...``), the bytes are downloaded
          (raising if not accessible).
        - Otherwise, ``data`` is expected to be base64 and is decoded to bytes.

        Parameters
        ----------
        image_id : int
            Identifier of the image to attach.
        note : str | None
            Optional human‑readable note describing why the image is attached.
        destination : str | None
            Root containing the image metadata. When omitted, the manager uses
            the most recent message-image lookup for the image id, falling back
            to personal memory.

        Returns
        -------
        dict
            A payload of the form:
            {"note": str, "image": base64_string}
            where ``image`` is the raw bytes of the image encoded as base64
            (PNG or JPEG). Downstream should render this as an image block.
        """
        return _attach_image_to_context_impl(
            self,
            image_id=image_id,
            note=note,
            destination=destination,
        )

    def _attach_message_images_to_context(
        self,
        *,
        message_id: int,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """Attach multiple images referenced by a message to the loop context.

        Characteristics
        ---------------
        - Batches attachment of several images linked via the message's image references.
        - Returns metadata (including optional annotations) alongside the base64 for each image.
        - Useful for multi‑image tasks where the loop should retain visual context.

        Parameters
        ----------
        limit : int
            Cap on how many images are attached (order preserved by reference order).

        Returns
        -------
        dict
            { "attached_count": int, "images": [ { "meta": {...}, "image": base64 }, ... ] }
            Each ``meta`` includes ``image_id``, ``caption``, ``timestamp``, and optional ``annotation``.
        """
        return _attach_message_images_to_context_impl(
            self,
            message_id=message_id,
            limit=limit,
        )

    # (Span substring helper removed – images now aligned via freeform annotations)

    # ──────────────────────────────────────────────────────────────────────
    #  Internal helpers (not exposed as tools)
    # ──────────────────────────────────────────────────────────────────────
    # Column and metrics helpers (paralleling ContactManager)
    def _get_columns(self) -> Dict[str, str]:
        """
        Return {column_name: column_type} for the transcripts table.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping column names to their types.
        """
        return _storage_get_columns(self)

    def _list_columns(
        self,
        *,
        include_types: bool = True,
        include_private: bool = False,
    ) -> Dict[str, str] | list[str]:
        """
        Return the list of available columns in the transcripts table, optionally with types.

        Parameters
        ----------
        include_types : bool, default True
            Controls the shape of the returned value:
            - When True: returns a mapping {column_name: column_type}.
            - When False: returns a list of column names.
        include_private : bool, default False
            When False, private/internal columns (those whose names start with "_")
            are omitted from the result to reduce payload size and avoid exposing
            vector/derived fields. Set to True to return all columns.
        """
        cols = _storage_list_columns(
            self,
            include_types=include_types,
            include_private=include_private,
        )
        return cols

    def _num_messages(self) -> int:
        """Return the total number of messages in transcripts."""
        return _storage_num_messages(self)

    # Internal provisioning helper
    def warm_embeddings(self) -> None:
        for context in self._read_transcript_contexts():
            try:
                ensure_vector_column(
                    context,
                    embed_column="_content_emb",
                    source_column="content",
                )
            except Exception:
                pass

    def _provision_storage(self) -> None:
        _storage_provision(self)

    # Exchanges helper
    def _ensure_exchanges_records(
        self,
        exchange_ids: set[int],
        *,
        eid_to_medium: Optional[Dict[int, str]] = None,
        context: str | None = None,
    ) -> None:
        _storage_ensure_exchanges(
            self,
            exchange_ids,
            eid_to_medium=eid_to_medium,
            context=context,
        )

    def get_exchange_metadata(
        self,
        exchange_id: int,
        *,
        destination: str | None = None,
    ) -> Exchange:
        """Fetch the Exchanges row for ``exchange_id`` as an Exchange model."""
        if destination is None:
            contexts = self._read_exchange_contexts()
        else:
            contexts = [self._exchanges_context_for_destination(destination)]
        rows = []
        for context in contexts:
            rows = unify.get_logs(
                context=context,
                filter=f"exchange_id == {int(exchange_id)}",
                limit=1,
            )
            if rows:
                break
        if not rows:
            raise ValueError(f"No exchange found for exchange_id={exchange_id}.")

        rec = rows[0].entries
        try:
            return Exchange(
                exchange_id=int(rec.get("exchange_id")),
                metadata=dict(rec.get("metadata") or {}),
                medium=str(rec.get("medium") or ""),
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to reconstruct Exchange for exchange_id={exchange_id}.",
            ) from exc

    def update_exchange_metadata(
        self,
        exchange_id: int,
        metadata: Dict[str, Any],
        *,
        destination: str | None = None,
    ) -> Exchange:
        """Update or create exchange metadata in one routed root."""
        try:
            context = self._exchanges_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]
        # Try update first
        row_ids = unify.get_logs(
            context=context,
            filter=f"exchange_id == {int(exchange_id)}",
            return_ids_only=True,
        )
        if row_ids:
            unify.update_logs(
                logs=row_ids,
                context=context,
                entries={"metadata": dict(metadata or {})},
                overwrite=True,
            )
        else:
            # Upsert behaviour – create a new row with empty medium if missing
            unity_log(
                context=context,
                exchange_id=int(exchange_id),
                metadata=dict(metadata or {}),
                medium="",
                new=True,
                mutable=True,
                add_to_all_context=self._should_add_to_all_context(context),
            )

        # Read back and return canonical shape
        return self.get_exchange_metadata(exchange_id, destination=destination)

    @functools.wraps(BaseTranscriptManager.filter_exchanges, updated=())
    def filter_exchanges(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int | None = 100,
    ) -> Dict[str, Any]:
        normalized = normalize_filter_expr(filter)
        logs = []
        fetch_limit = (offset + limit) if limit is not None else 1000
        for context in self._read_exchange_contexts():
            logs.extend(
                unify.get_logs(
                    context=context,
                    filter=normalized,
                    offset=0,
                    limit=fetch_limit,
                    from_fields=list(Exchange.model_fields.keys()),
                ),
            )
        exchanges: list[Exchange] = []
        for lg in logs[offset : (offset + limit) if limit is not None else None]:
            try:
                exchanges.append(Exchange(**lg.entries))
            except Exception:
                continue
        return {"exchanges": exchanges}

    def log_first_message_in_new_exchange(
        self,
        message: Union[Dict[str, Any], Message],
        *,
        exchange_initial_metadata: Optional[Dict[str, Any]] = None,
        destination: str | None = None,
    ) -> tuple[int, int]:
        """Log the first message of a brand-new exchange and set initial metadata.

        Returns (exchange_id, message_id) for the newly created exchange and message.
        The destination is managed internally by session synthesis for implicit writes.
        """
        try:
            transcripts_context = self._transcripts_context_for_destination(destination)
            exchanges_context = self._exchanges_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]

        # 1) Validate no exchange_id is provided by the caller
        if isinstance(message, dict):
            if "exchange_id" in message:
                raise ValueError(
                    "exchange_id must NOT be provided when starting a new exchange; use TranscriptManager.log_messages(...) if you already have an existing exchange id.",
                )
        else:  # Message instance
            try:
                if getattr(message, "exchange_id", UNASSIGNED) not in (
                    None,
                    UNASSIGNED,
                ):
                    raise ValueError(
                        "Message.exchange_id must NOT be set when starting a new exchange; use TranscriptManager.log_messages(...) if you already have an existing exchange id.",
                    )
            except Exception:
                # If attribute missing, treat as acceptable (will be injected downstream)
                pass

        # 2) Normalise payload and persist directly to obtain an assigned exchange_id
        def _ensure_contact_id_local(c: Union[int, Contact]) -> int:
            if not isinstance(c, Contact):
                if c is None:
                    raise ValueError(
                        "sender_id / receiver_ids cannot be None – either provide an int or a Contact instance.",
                    )
                return int(c)
            if c.contact_id is not None and c.contact_id != -1:
                return int(c.contact_id)
            # Create via ContactManager and return id
            full_data = c.model_dump(exclude_none=True)
            create_kwargs = {k: v for k, v in full_data.items() if k != "contact_id"}
            outcome = self._contact_manager._create_contact(
                **create_kwargs,
                destination=destination,
            )
            return int(outcome["details"]["contact_id"])  # type: ignore[index]

        if isinstance(message, Message):
            payload: Dict[str, Any] = message.model_dump(mode="python")
        else:
            payload = dict(message)

        if "receiver_ids" not in payload:
            raise ValueError("Each message must include 'receiver_ids'.")

        payload["sender_id"] = _ensure_contact_id_local(payload.get("sender_id"))
        payload["receiver_ids"] = [
            _ensure_contact_id_local(r) for r in payload.get("receiver_ids", [])
        ]

        # Ensure no explicit exchange id provided
        if payload.get("exchange_id") is not None:
            raise ValueError(
                "exchange_id must NOT be provided when starting a new exchange; use TranscriptManager.log_messages(...) if you already have an existing exchange id.",
            )

        # 3) Create Exchange row FIRST to satisfy FK constraint
        exchange_log = unity_log(
            context=exchanges_context,
            metadata=dict(exchange_initial_metadata or {}),
            medium=str(payload.get("medium", "")),
            new=True,
            mutable=True,
            add_to_all_context=self._should_add_to_all_context(exchanges_context),
        )

        # Extract the assigned exchange_id
        try:
            exid = int(exchange_log.entries["exchange_id"])
        except Exception as exc:  # noqa: BLE001 – precise error context
            raise RuntimeError(
                "Created exchange lacks an assigned exchange_id.",
            ) from exc
        if exid < 0:
            raise RuntimeError("Created exchange has an unassigned exchange_id.")

        # 4) Add exchange_id to payload and create message SECOND
        payload["exchange_id"] = exid

        created_model = Message(**payload)
        entries = created_model.to_post_json()

        log = unity_log(
            context=transcripts_context,
            **entries,
            new=True,
            mutable=True,
            add_to_all_context=self._should_add_to_all_context(transcripts_context),
        )

        tm_message_id = int(log.entries.get("message_id", -1))

        # ── Inactivity-followup activity sync (best-effort, async) ─────────
        # Mirrors the hook at the end of log_messages so first-message writes
        # — which use unity_log directly and bypass log_messages — also bump
        # last_correspondence_at on the assistant row. Dispatched to a daemon
        # thread so the network round-trip never blocks the caller; failures
        # are swallowed inside ``touch_assistant_activity``.
        try:
            import threading

            from .activity_sync import touch_assistant_activity

            agent_id = getattr(SESSION_DETAILS.assistant, "agent_id", None)
            if agent_id is not None:
                threading.Thread(
                    target=touch_assistant_activity,
                    args=(agent_id,),
                    daemon=True,
                    name="touch_assistant_activity",
                ).start()
        except Exception:
            pass

        return exid, tm_message_id

    def _move_row(
        self,
        *,
        table_name: str,
        id_field: str,
        row_id: int,
        from_root: str,
        to_destination: str | None,
        model: type[BaseModel],
    ) -> ToolOutcome:
        source_root = self._root_context_for_move(table_name, from_root)
        target_root = ContextRegistry.write_root(
            self,
            table_name,
            destination=to_destination,
        )
        source_context = self._context_for_root(source_root, table_name)
        target_context = self._context_for_root(target_root, table_name)

        rows = unify.get_logs(
            context=source_context,
            filter=f"{id_field} == {int(row_id)}",
            limit=2,
        )
        if not rows:
            raise ValueError(
                f"No {table_name} row found with {id_field}={int(row_id)} in {source_context}.",
            )
        if len(rows) > 1:
            raise RuntimeError(
                f"Multiple {table_name} rows found with {id_field}={int(row_id)} in {source_context}.",
            )

        record = model(**rows[0].entries)
        if hasattr(record, "to_post_json"):
            payload = record.to_post_json()
        else:
            payload = record.model_dump(mode="json")
        target_ids = unify.get_logs(
            context=target_context,
            filter=f"{id_field} == {int(row_id)}",
            return_ids_only=True,
            limit=2,
        )
        if len(target_ids) > 1:
            raise RuntimeError(
                f"Multiple {table_name} rows found with {id_field}={int(row_id)} in {target_context}.",
            )
        if target_ids:
            unify.update_logs(
                context=target_context,
                logs=[target_ids[0]],
                entries=payload,
                overwrite=True,
            )
        else:
            unity_log(
                context=target_context,
                **payload,
                new=True,
                mutable=True,
                add_to_all_context=self._should_add_to_all_context(target_context),
            )

        unify.delete_logs(context=source_context, logs=rows[0].id)
        return {
            "outcome": f"{table_name} row moved",
            "details": {
                id_field: int(row_id),
                "from_context": source_context,
                "to_context": target_context,
            },
        }

    def move_message(
        self,
        message_id: int,
        *,
        from_root: str,
        to_destination: str | None,
    ) -> ToolOutcome:
        """Move one transcript message row between personal/shared roots."""

        try:
            return self._move_row(
                table_name=TRANSCRIPTS_TABLE,
                id_field="message_id",
                row_id=message_id,
                from_root=from_root,
                to_destination=to_destination,
                model=Message,
            )
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]

    def move_exchange(
        self,
        exchange_id: int,
        *,
        from_root: str,
        to_destination: str | None,
    ) -> ToolOutcome:
        """Move one exchange row between personal/shared roots."""

        try:
            return self._move_row(
                table_name=EXCHANGES_TABLE,
                id_field="exchange_id",
                row_id=exchange_id,
                from_root=from_root,
                to_destination=to_destination,
                model=Exchange,
            )
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]

    # Formatting helper: single contacts table + messages
    def _format_contacts_and_messages(self, messages: List[Message]) -> Dict[str, Any]:
        return _format_contacts_and_messages_impl(self, messages)

    # Misc small utilities (kept last)
    @classmethod
    def _get_logger(cls) -> unify.AsyncLoggerManager:
        return cls._LOGGER

    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        # Deprecated: use common.llm_policies.require_first("search_messages") instead.
        return require_first("search_messages")(step_index, current_tools)
