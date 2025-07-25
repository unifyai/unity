# memory_manager/memory_manager.py
from __future__ import annotations

import asyncio
import json
import os
import functools
from typing import Optional, Callable, Dict, Any

import unify

from ..contact_manager.contact_manager import ContactManager
from ..transcript_manager.transcript_manager import TranscriptManager
from ..knowledge_manager.knowledge_manager import KnowledgeManager
from ..task_scheduler.task_scheduler import TaskScheduler
from ..common.llm_helpers import methods_to_tool_dict, start_async_tool_use_loop
from .prompt_builders import (
    build_contact_update_prompt,
    build_bio_prompt,
    build_rolling_prompt,
    build_knowledge_prompt,
    build_activity_events_summary_prompt,
)
from .base import BaseMemoryManager
from ..events.event_bus import EVENT_BUS, Event
from .broader_context import set_broader_context


# ---------------------------------------------------------------------------
#  Environment toggle helpers
# ---------------------------------------------------------------------------


def _env_flag(
    var_name: str,
    default: bool = True,
) -> bool:  # noqa: D401 – imperative helper name
    """Return *True* if the environment variable *var_name* is set to a truthy
    value (case-insensitive ``true, 1, yes, y``).  Missing variables fall back
    to *default* so existing behaviour remains unchanged when the variable is
    absent.
    """

    val = os.getenv(var_name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y"}


class MemoryManager(BaseMemoryManager):
    """
    Offline helper invoked by a scheduler every ~50 messages.
    """

    _MANAGERS = {
        "ContactManager": "contact_manager",
        "TranscriptManager": "transcript_manager",
        "KnowledgeManager": "knowledge_manager",
        "TaskScheduler": "task_scheduler",
        "Conductor": "conductor",
    }

    _TIME_WINDOWS = {  # seconds
        "past_day": 60 * 60 * 24,
        "past_week": 60 * 60 * 24 * 7,
        "past_4_weeks": 60 * 60 * 24 * 7 * 4,
        "past_12_weeks": 60 * 60 * 24 * 7 * 12,
        "past_52_weeks": 60 * 60 * 24 * 7 * 52,
    }
    _COUNT_WINDOWS = {
        "past_interaction": 1,
        "past_10_interactions": 10,
        "past_40_interactions": 40,
        "past_120_interactions": 120,
        "past_520_interactions": 520,
    }

    # ────────────────────────────────────────────────────────────────────
    #   hierarchy helpers so higher-level windows summarise the lower
    #       level rather than the raw ManagerMethod events
    # ────────────────────────────────────────────────────────────────────
    _TIME_ORDER = [
        "past_day",
        "past_week",
        "past_4_weeks",
        "past_12_weeks",
        "past_52_weeks",
    ]
    _COUNT_ORDER = [
        "past_interaction",
        "past_10_interactions",
        "past_40_interactions",
        "past_120_interactions",
        "past_520_interactions",
    ]

    # child_window → (immediate_lower_window, how_many_lower_summaries)
    _TIME_PARENT: dict[str, tuple[str, int]] = {}
    for i in range(1, len(_TIME_ORDER)):
        child, parent = _TIME_ORDER[i], _TIME_ORDER[i - 1]
        _TIME_PARENT[child] = (
            parent,
            _TIME_WINDOWS[child] // _TIME_WINDOWS[parent],
        )

    _COUNT_PARENT: dict[str, tuple[str, int]] = {}
    for i in range(1, len(_COUNT_ORDER)):
        child, parent = _COUNT_ORDER[i], _COUNT_ORDER[i - 1]
        _COUNT_PARENT[child] = (
            parent,
            _COUNT_WINDOWS[child] // _COUNT_WINDOWS[parent],
        )

    _tmp_cols = []
    for nick in _MANAGERS.values():
        for window in list(_TIME_WINDOWS) + list(_COUNT_WINDOWS):
            _tmp_cols.append(f"{nick}/{window}")

    # ───────────────────────────  SUMMARY COLS  ───────────────────────────
    _SUMMARY_TIME_COL = "time_based_activity"
    _SUMMARY_COUNT_COL = "count_based_activity"
    _tmp_cols.extend([_SUMMARY_TIME_COL, _SUMMARY_COUNT_COL])

    _ROLLING_COLUMNS = tuple(_tmp_cols)
    del _tmp_cols

    # ------------------------------------------------------------------ #
    #  Shared helper: convert Message / event dicts to plain-text transcript
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_plain_transcript(
        messages: list[dict],
        contact_manager: Optional["ContactManager"] = None,
    ) -> str:
        """Delegate to `TranscriptManager.build_plain_transcript` to avoid duplication."""

        from unity.transcript_manager.transcript_manager import (
            TranscriptManager as _TM,
        )

        return _TM.build_plain_transcript(messages, contact_manager=contact_manager)

    # ---------------------------------------------------------------------- #
    def __init__(
        self,
        *,
        contact_manager: Optional[ContactManager] = None,
        transcript_manager: Optional[TranscriptManager] = None,
        knowledge_manager: Optional[KnowledgeManager] = None,
        task_scheduler: Optional[TaskScheduler] = None,
    ):
        self._contact_manager = contact_manager or ContactManager()
        self._transcript_manager = transcript_manager or TranscriptManager(
            contact_manager=self._contact_manager,
        )
        self._knowledge_manager = knowledge_manager or KnowledgeManager()
        self._task_scheduler = task_scheduler or TaskScheduler()

        # ── Environment-controlled callback registration -------------------------

        #  Determine which groups of callbacks should be active based on
        #  environment variables (defaults = enabled for full backward compatibility).
        self._register_summary_callbacks: bool = _env_flag(
            "REGISTER_SUMMARY_CALLBACKS",
            True,
        )
        self._register_update_callbacks: bool = _env_flag(
            "REGISTER_UPDATE_CALLBACKS",
            True,
        )

        # ── Rolling-Activity context & subscriptions (summaries) ────────────────
        self._rolling_ctx = self._ensure_rolling_context()
        self._rolling_lock = asyncio.Lock()

        # Expose a readiness Event so external callers/tests can await summary callbacks
        self._callbacks_ready = asyncio.Event()

        if self._register_summary_callbacks:
            # Fire-and-forget registration of summary callbacks
            asyncio.create_task(self._setup_rolling_callbacks())
        else:
            # No summary callbacks -> consider the system *ready* immediately
            self._callbacks_ready.set()

        # ── real-time 50-message trigger (update callbacks) --------------------
        self._CHUNK_SIZE: int = 50
        self._recent_messages: list[dict] = []
        self._messages_since_update: int = 0

        self._chunk_lock = asyncio.Lock()

        if self._register_update_callbacks:
            # Fire-and-forget setup of message counter and explicit-call callbacks
            asyncio.create_task(self._setup_message_callbacks())

            # 1.  Automatically pin the *calling_id* for the lifetime of any explicit
            #     tool invocation coming from ConversationManager so the related
            #     ManagerMethod events are never trimmed before completion.
            EVENT_BUS.register_auto_pin(
                event_type="ManagerMethod",
                open_predicate=lambda e: (
                    e.payload.get("source") == "ConversationManager"
                    and e.payload.get("phase") == "incoming"
                ),
                close_predicate=lambda e: (
                    e.payload.get("source") == "ConversationManager"
                    and e.payload.get("phase") == "outgoing"
                ),
                key_fn=lambda e: e.calling_id,
            )

            # 2.  Listen to those explicit ManagerMethod events so they are included
            #     in the 50-message rolling window given to the LLM.
            asyncio.create_task(self._setup_explicit_call_callbacks())

        # If update callbacks are disabled  no-op for message processing

    # ------------------------------------------------------------------ #
    # 1  update_contacts                                                 #
    # ------------------------------------------------------------------ #
    async def update_contacts(
        self,
        transcript: str,
        guidance: Optional[str] = None,
        *,
        update_bios: bool = True,
        update_rolling_summaries: bool = True,
    ) -> str:
        """
        Scan the transcript, identify *new* contacts or modified details,
        and persist them.  Returns a short description of what changed.
        """

        # ─ 1.  Build **restricted** live tool-set  ──────────────────────────
        # Guardrails:
        #   • Disallow creation of *new* columns via custom_fields
        #   • Disallow any modification of the `bio` or `rolling_summary` fields
        #
        # We therefore expose *thin* wrappers around the low-level synchronous
        # helpers (`_create_contact` / `_update_contact`) that validate the
        # supplied keyword arguments **before** delegating to the real
        # implementation in a background thread.  The *ask* helpers are still
        # exposed unmodified so the LLM can read the current state.

        # Track ids of contacts that are *newly* created during this tool loop so
        # we can optionally follow up with bio / rolling-summary updates.
        new_contact_ids: list[int] = []

        @functools.wraps(self._contact_manager._create_contact, updated=())
        async def _safe_create_contact(**kwargs):
            # Reject attempts to touch forbidden fields
            for forbidden in ("bio", "rolling_summary"):
                if kwargs.get(forbidden) is not None:
                    raise ValueError(
                        "MemoryManager.update_contacts – creation of contacts must not set 'bio' or 'rolling_summary'.",
                    )
            # Reject custom_fields entirely to avoid implicit column creation
            if kwargs.get("custom_fields"):
                raise ValueError(
                    "MemoryManager.update_contacts – creation of custom columns is not allowed.",
                )

            # ── Strip *internal* helper parameters that the underlying ContactManager
            #    implementation is unaware of (e.g. parent_chat_context, clarification queues…)
            import inspect  # local import to avoid polluting module namespace

            allowed = set(
                inspect.signature(self._contact_manager._create_contact).parameters,
            )
            cleaned_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

            outcome = await asyncio.to_thread(
                self._contact_manager._create_contact,
                **cleaned_kwargs,
            )

            # -------------------  capture newly assigned id -------------------
            try:
                contact_id = (
                    outcome.get("details", {}).get("contact_id")  # type: ignore[assignment]
                    if isinstance(outcome, dict)
                    else None
                )
                if isinstance(contact_id, int):
                    new_contact_ids.append(contact_id)
            except Exception:
                # Defensive – never let a parsing error break the caller.
                pass

            return outcome

        @functools.wraps(self._contact_manager._update_contact, updated=())
        async def _safe_update_contact(**kwargs):
            # Reject forbidden field modifications
            for forbidden in ("bio", "rolling_summary"):
                if kwargs.get(forbidden) is not None:
                    raise ValueError(
                        "MemoryManager.update_contacts – modification of 'bio' or 'rolling_summary' is not allowed.",
                    )
            # Reject custom_fields to prevent new columns
            if kwargs.get("custom_fields"):
                raise ValueError(
                    "MemoryManager.update_contacts – modification involving custom columns is not allowed.",
                )

            # Same hidden-arg stripping logic as for _safe_create_contact
            import inspect  # local import

            allowed = set(
                inspect.signature(self._contact_manager._update_contact).parameters,
            )
            cleaned_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

            return await asyncio.to_thread(
                self._contact_manager._update_contact,
                **cleaned_kwargs,
            )

        # ────────────────────────────────────────────────────────────────
        #   Patch wrapper *signatures* & *docstrings* so the LLM sees a
        #   cleaned-up schema that hides now-forbidden parameters.
        # ----------------------------------------------------------------
        from inspect import Signature, Parameter, signature as _sig  # type: ignore
        from unity.common.llm_helpers import _strip_hidden_params_from_doc

        _FORBIDDEN = {"bio", "rolling_summary", "custom_fields"}

        def _prune_wrapper(_wrapper, _original):
            """Reuse *original* metadata but remove forbidden params."""
            orig_sig = _sig(_original)
            # Keep parameter **order** but drop any forbidden names
            new_params = [
                p.replace()  # shallow copy
                for p in orig_sig.parameters.values()
                if p.name not in _FORBIDDEN
            ]
            _wrapper.__signature__ = Signature(parameters=new_params)

            orig_doc = _original.__doc__ or ""
            _wrapper.__doc__ = (
                _strip_hidden_params_from_doc(orig_doc, _FORBIDDEN)
                + "\n\nNOTE: Disregard any mention of the 'bio', "
                "'rolling_summary', or 'custom_fields' arguments, which have all been removed."
            )

        _prune_wrapper(_safe_create_contact, self._contact_manager._create_contact)
        _prune_wrapper(_safe_update_contact, self._contact_manager._update_contact)

        # Base read-only helpers ------------------------------------------------
        tools: Dict[str, Callable[..., Any]] = {
            "contact_ask": self._contact_manager.ask,
            "transcript_ask": self._transcript_manager.ask,
            # Restricted mutation helpers
            "create_contact": _safe_create_contact,
            "update_contact": _safe_update_contact,
        }

        # ─ 2.  LLM client
        llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
        )
        llm.set_system_message(build_contact_update_prompt(tools, guidance))

        # ─ 3.  Kick off *single* tool-use loop
        handle = start_async_tool_use_loop(
            llm,
            transcript,
            tools,
            loop_id="MemoryManager.update_contacts",
            tool_policy=lambda i, _: (
                ("required", {"contact_ask": self._contact_manager.ask})
                if i < 1
                else ("auto", _)
            ),
        )

        result_str = await handle.result()  # plain str returned by LLM

        # ------------------------------------------------------------------
        # Follow-up: for every *new* contact created during this call, refresh
        # their bio and/or rolling summary using the same transcript chunk.
        # ------------------------------------------------------------------
        if new_contact_ids and (update_bios or update_rolling_summaries):
            follow_up_tasks: list[asyncio.Task] = []
            for _cid in new_contact_ids:
                if update_bios:
                    follow_up_tasks.append(
                        asyncio.create_task(
                            self.update_contact_bio(
                                transcript,
                                contact_id=_cid,
                            ),
                        ),
                    )
                if update_rolling_summaries:
                    follow_up_tasks.append(
                        asyncio.create_task(
                            self.update_contact_rolling_summary(
                                transcript,
                                contact_id=_cid,
                            ),
                        ),
                    )

            if follow_up_tasks:
                # Run concurrently – ignore individual failures so one contact
                # does not block others or propagate errors back to callers.
                await asyncio.gather(*follow_up_tasks, return_exceptions=True)

        return result_str

    # ------------------------------------------------------------------ #
    # 2  update_contact_bio                                              #
    # ------------------------------------------------------------------ #
    async def update_contact_bio(
        self,
        transcript: str,
        *,
        contact_id: int,
        guidance: Optional[str] = None,
    ) -> str:
        """Refresh the *bio* column for the given contact."""
        # Ensure the assistant writes about itself in **second person**.
        assistant_extra = (
            "IMPORTANT: This bio belongs to the assistant itself (contact_id 0). Always use **second-person** pronouns ('you') when describing the assistant. Never refer to the assistant in the third person."  # noqa: E501
            if contact_id == 0
            else None
        )

        # Merge any caller-supplied guidance with our assistant-specific rule.
        combined_guidance: Optional[str] = (
            "\n".join(g for g in (guidance, assistant_extra) if g) or None
        )
        target_id = contact_id  # capture for closure

        async def set_bio(contact_id: int, bio: str) -> str:
            """Update only the bio column for the supplied contact id."""
            final_id = contact_id or target_id
            if final_id is None:
                raise ValueError(
                    "contact_id must be supplied either via the method argument or the tool call.",
                )
            await asyncio.to_thread(
                self._contact_manager._update_contact,
                contact_id=final_id,
                custom_fields={"bio": bio},
            )
            return f"Bio for contact with id {final_id} successfully updated"

        tools: Dict[str, Callable[..., Any]] = {
            "set_bio": set_bio,
        }

        llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
        )
        contacts = await asyncio.to_thread(
            self._contact_manager._search_contacts,
            filter=f"contact_id == {contact_id}",
            limit=1,
        )
        c0 = contacts[0]
        latest_bio_val = c0.bio
        contact_name_val = (
            " ".join(p for p in [c0.first_name, c0.surname] if p).strip() or None
        )
        llm.set_system_message(
            build_bio_prompt(
                f"{contact_name_val} (id {contact_id})",
                tools,
                guidance=combined_guidance,
            ),
        )

        # ------------------------------------------------------------------
        # Retrieve the *current* bio from the backend so the LLM always sees
        # the latest state without relying on the caller to supply it.
        # We perform the lookup in a background thread because the underlying
        # implementation is synchronous and may hit the network.
        try:
            contacts = await asyncio.to_thread(
                self._contact_manager._search_contacts,
                filter=f"contact_id == {contact_id}",
                limit=1,
            )
            latest_bio_val = contacts[0].bio if contacts else None
        except Exception:
            latest_bio_val = None  # best-effort fallback

        user_payload = {
            "contact_id": contact_id,
            "latest_bio (to maybe update)": latest_bio_val,
            "transcript": transcript,
        }
        if contact_name_val is not None:
            user_payload["contact_name"] = contact_name_val

        user_blob = json.dumps(user_payload, indent=2)

        handle = start_async_tool_use_loop(
            llm,
            user_blob,
            tools,
            loop_id="MemoryManager.update_contact_bio",
        )

        return await handle.result()

    # ------------------------------------------------------------------ #
    # 3  update_contact_rolling_summary                                  #
    # ------------------------------------------------------------------ #
    async def update_contact_rolling_summary(
        self,
        transcript: str,
        *,
        contact_id: int,
        guidance: Optional[str] = None,
    ) -> str:
        """Refresh the *rolling_summary* column for the given contact."""

        # Ensure the assistant's rolling summary uses **second person**.
        assistant_extra = (
            "IMPORTANT: This rolling summary belongs to the assistant itself (contact_id 0). Always use **second-person** pronouns ('you') when describing the assistant. Never refer to the assistant in the third person."  # noqa: E501
            if contact_id == 0
            else None
        )

        combined_guidance: Optional[str] = (
            "\n".join(g for g in (guidance, assistant_extra) if g) or None
        )
        target_id = contact_id  # capture for closure

        async def set_rolling_summary(contact_id: int, rolling_summary: str) -> str:
            final_id = contact_id or target_id
            if final_id is None:
                raise ValueError(
                    "contact_id must be supplied either via the method argument or the tool call.",
                )
            await asyncio.to_thread(
                self._contact_manager._update_contact,
                contact_id=final_id,
                custom_fields={"rolling_summary": rolling_summary},
            )
            return (
                f"Rolling summary for contact with id {final_id} successfully updated"
            )

        tools: Dict[str, Callable[..., Any]] = {
            "transcript_ask": self._transcript_manager.ask,
            "contact_ask": self._contact_manager.ask,
            "set_rolling_summary": set_rolling_summary,
        }

        llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
        )
        llm.set_system_message(build_rolling_prompt(tools, combined_guidance))

        # ------------------------------------------------------------------
        # Retrieve the *current* rolling summary from the backend so the LLM
        # always has the freshest context and callers do not need to supply it.
        try:
            contacts = await asyncio.to_thread(
                self._contact_manager._search_contacts,
                filter=f"contact_id == {contact_id}",
                limit=1,
            )
            latest_summary_val = contacts[0].rolling_summary if contacts else None
        except Exception:
            latest_summary_val = None  # best-effort fallback

        user_blob = json.dumps(
            {
                "contact_id": contact_id,
                "latest_rolling_summary (to maybe update)": latest_summary_val,
                "transcript": transcript,
            },
            indent=2,
        )

        handle = start_async_tool_use_loop(
            llm,
            user_blob,
            tools,
            loop_id="MemoryManager.update_contact_rolling_summary",
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )

        return await handle.result()

    # ------------------------------------------------------------------ #
    # 4  update_knowledge                                                #
    # ------------------------------------------------------------------ #
    async def update_knowledge(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Mine reusable information and persist to the long-term knowledge base.
        """

        tools: Dict[str, Callable[..., Any]] = methods_to_tool_dict(
            self._contact_manager.ask,
            self._transcript_manager.ask,
            self._knowledge_manager.ask,
            self._knowledge_manager.refactor,
            self._knowledge_manager.update,
            include_class_name=True,
        )

        llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
        )
        llm.set_system_message(build_knowledge_prompt(tools, guidance))

        handle = start_async_tool_use_loop(
            llm,
            transcript,
            tools,
            loop_id="MemoryManager.update_knowledge",
            tool_policy=lambda i, _: ("required", _) if i < 2 else ("auto", _),
        )

        return await handle.result()

    # ------------------------------------------------------------------ #
    # 5  update_tasks                                                    #
    # ------------------------------------------------------------------ #
    async def update_tasks(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Analyse the latest transcript chunk and update the task list using
        the TaskScheduler's public API (ask / update).  Returns a concise
        description of what was changed or 'no-op' when no updates were
        necessary.
        """

        tools: Dict[str, Callable[..., Any]] = methods_to_tool_dict(
            self._task_scheduler.ask,
            self._task_scheduler.update,
            include_class_name=True,
        )

        llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
        )

        from .prompt_builders import build_task_prompt  # local import to avoid cycles

        llm.set_system_message(build_task_prompt(tools, guidance))

        handle = start_async_tool_use_loop(
            llm,
            transcript,
            tools,
            loop_id="MemoryManager.update_tasks",
            tool_policy=lambda i, _: ("required", _) if i < 2 else ("auto", _),
        )

        return await handle.result()

    # ------------------------------------------------------------------ #
    # 5  reset – new blocking helper                                     #
    # ------------------------------------------------------------------ #
    async def reset(self) -> None:  # noqa: D401 – imperative name
        """Completely reset the MemoryManager's rolling-activity state.

        1. Delegates to ``EVENT_BUS.reset()`` to wipe all event history and
           callback registrations.
        2. Re-creates and waits for a fresh set of rolling-activity callback
           subscriptions so callers can immediately publish new events without
           manually re-registering helpers.
        """

        # 1. Reset the global EventBus singleton (clears callbacks & logs)
        EVENT_BUS.reset()

        # Re-create readiness Event and schedule callback registrations according
        # to the current environment flags so callers can rely on them right after
        # this coroutine returns.

        self._callbacks_ready = asyncio.Event()

        tasks: list[asyncio.Task] = []

        if self._register_summary_callbacks:
            tasks.append(asyncio.create_task(self._setup_rolling_callbacks()))
        else:
            # Summary callbacks disabled – consider ready instantly
            self._callbacks_ready.set()

        if self._register_update_callbacks:
            tasks.extend(
                [
                    asyncio.create_task(self._setup_message_callbacks()),
                    asyncio.create_task(self._setup_explicit_call_callbacks()),
                ],
            )

        if tasks:
            await asyncio.gather(*tasks)

        # Wait for summary callbacks (if any) to signal readiness – timeout safeguards
        await asyncio.wait_for(self._callbacks_ready.wait(), timeout=5)

    # ───────────────────────────  MESSAGE-BASED CALLBACKS  ───────────────────────────

    async def _setup_message_callbacks(self) -> None:
        """Register a callback that fires *every* incoming `message` event.

        The helper relies on the EventBus singleton being fully initialised – we therefore
        make sure to await `_ensure_ready` via `register_callback` internally.
        """

        async def _cb(events):  # noqa: ANN001 – signature imposed by EventBus
            await self._on_new_message(events[0])

        try:
            await EVENT_BUS.register_callback(
                event_type="Message",
                callback=_cb,
                every_n=1,  # every single message
            )
        except Exception:  # pragma: no cover – defensive
            # We do *not* propagate registration failures – the MemoryManager still
            # works via manual scheduling even when the callback cannot be installed.
            pass

    # ------------------------------------------------------------------
    # NEW: capture *explicit* ManagerMethod invocations coming from the
    #       ConversationManager so the passive 50-message chunk has full
    #       context.
    # ------------------------------------------------------------------

    async def _setup_explicit_call_callbacks(self) -> None:
        """Register a callback for ManagerMethod events tagged with
        `source == "ConversationManager"` (incoming & outgoing)."""

        async def _cb(events):  # noqa: ANN001 – imposed by EventBus
            await self._on_new_explicit_call(events[0])

        try:
            await EVENT_BUS.register_callback(
                event_type="ManagerMethod",
                callback=_cb,
                filter='evt.payload.get("source") == "ConversationManager"',
                every_n=1,
            )
        except Exception:  # pragma: no cover – defensive
            pass

    # ------------------------------------------------------------------
    async def _on_new_explicit_call(self, evt: Event) -> None:
        """Append explicit ManagerMethod events to the rolling window."""

        # Keep the data lightweight & JSON-serialisable
        self._recent_messages.append(
            {
                "kind": "manager_method",
                "data": {
                    **(
                        evt.payload.model_dump(mode="json")
                        if hasattr(evt.payload, "model_dump")
                        else evt.payload
                    ),
                    "timestamp": evt.timestamp.isoformat(),
                    "calling_id": evt.calling_id,
                },
            },
        )

        self._messages_since_update += 1
        if self._messages_since_update >= self._CHUNK_SIZE:
            await self._flush_recent_items()

    # ------------------------------------------------------------------
    async def _flush_recent_items(self) -> None:
        """Helper that triggers chunk processing & resets local counters."""

        self._messages_since_update = 0
        items = self._recent_messages.copy()
        self._recent_messages.clear()

        await self._process_message_chunk(items)

    # ------------------------------------------------------------------
    # Override the original message-handler so it stores the unified format
    # ------------------------------------------------------------------

    async def _on_new_message(self, evt: Event) -> None:
        """Collect messages and trigger memory updates every *CHUNK_SIZE* messages."""

        # Payload is guaranteed to be a `Message` instance thanks to the
        # instrumentation in TranscriptManager.log_message.
        from unity.transcript_manager.types.message import Message  # local import

        if not isinstance(evt.payload, Message):  # ignore unexpected payloads
            return

        msg = evt.payload

        # Append a typed record so downstream code can distinguish items
        self._recent_messages.append(
            {
                "kind": "message",
                "data": {
                    "sender_id": msg.sender_id,
                    "receiver_ids": msg.receiver_ids,
                    "medium": msg.medium,
                    "timestamp": msg.timestamp.isoformat(),
                    "content": msg.content,
                },
            },
        )

        self._messages_since_update += 1

        if self._messages_since_update >= self._CHUNK_SIZE:
            await self._flush_recent_items()

    async def _process_message_chunk(self, messages: list[dict]) -> None:
        """Run the full suite of memory updates for one 50-message chunk."""

        # Serialise – prevent concurrent chunks from interleaving updates
        async with self._chunk_lock:
            try:
                plain_transcript = self.build_plain_transcript(
                    messages,
                    contact_manager=self._contact_manager,
                )

                # ── 1. Global, transcript-level updates (run *concurrently*) ────────
                global_tasks = [
                    self.update_contacts(plain_transcript),
                    self.update_knowledge(plain_transcript),
                    self.update_tasks(plain_transcript),
                ]

                # ── 2. Per-contact updates (bio & rolling summary) ──────────────────
                contact_ids: set[int] = set()

                # Attempt to exclude the assistant contact (id provided via env-var or 0)
                try:
                    assistant_id = int(os.getenv("ASSISTANT_CONTACT_ID", "0"))
                except ValueError:
                    assistant_id = 0

                for item in messages:
                    if item.get("kind") != "message":
                        continue  # ignore manager-method items for contact updates

                    md = item.get("data", {})

                    # 1) sender -----------------------------------------------------
                    sid = md.get("sender_id")
                    if isinstance(sid, int) and sid != assistant_id:
                        contact_ids.add(sid)

                    # 2) receiver(s) ----------------------------------------------
                    rids = md.get("receiver_ids")

                    if rids is None:
                        continue

                    if isinstance(rids, int):
                        if rids != assistant_id:
                            contact_ids.add(rids)
                    elif isinstance(rids, (list, tuple, set)):
                        for rid in rids:
                            if isinstance(rid, int) and rid != assistant_id:
                                contact_ids.add(rid)

                # Build per-contact tasks
                for _cid in contact_ids:
                    global_tasks.extend(
                        [
                            self.update_contact_bio(
                                plain_transcript,
                                contact_id=_cid,
                            ),
                            self.update_contact_rolling_summary(
                                plain_transcript,
                                contact_id=_cid,
                            ),
                        ],
                    )

                # Run *all* updates concurrently – failures are captured but do not
                # cancel the remaining updates so one misbehaving method doesn’t stall
                # the entire batch.
                await asyncio.gather(*global_tasks, return_exceptions=True)
            except Exception:  # pragma: no cover – defensive
                # Never propagate errors back to the EventBus – log and swallow.
                import traceback

                traceback.print_exc()

    # ───────────────────────────  HELPERS  ────────────────────────────
    # 1. Context & schema ---------------------------------------------------
    def _ensure_rolling_context(self) -> str:
        """Create the `RollingActivity` context (idempotent) and return its name."""
        active_ctx = unify.get_active_context()["write"] or ""
        ctx = f"{active_ctx}/RollingActivity" if active_ctx else "RollingActivity"
        if ctx not in unify.get_contexts():
            unify.create_context(ctx, unique_column_ids="row_id")
            fields = {
                col: {"type": "str", "mutable": True} for col in self._ROLLING_COLUMNS
            }
            unify.create_fields(fields, context=ctx)
        return ctx

    # 2. Callback registration ---------------------------------------------
    async def _setup_rolling_callbacks(self) -> None:
        """
        Register callbacks that build the Rolling-Activity hierarchy.

        • Base-level snapshots
            – past_interaction (count based)
            – past_day         (time based)
          are still triggered directly from *ManagerMethod* events.

        • All higher-level windows are now triggered from the new
          `_hierarchical_summaries` event that each completed snapshot emits.
          The callback fires once *ratio* (= how many lower-level
          summaries constitute this window) such events have arrived.
        """

        async def _register_for_manager(mgr_cls: str, nick: str):
            # Register callbacks for one manager

            def _mk_cb(col_name: str):
                async def _cb(events, _col=col_name):
                    await self._record_rolling_activity(_col, events)

                    # schedule recording (silent now)

                return _cb

            mm_filter = (
                f'evt.payload["manager"] == "{mgr_cls}" '
                f'and evt.payload.get("phase") == "outgoing"'
            )

            async def _reg(cb_col: str, *, event_type: str, **kw):
                try:
                    await EVENT_BUS.register_callback(
                        event_type=event_type,
                        callback=_mk_cb(cb_col),
                        **kw,
                    )
                except Exception as e:
                    # propagate any registration error
                    raise

            # base-level
            await _reg(
                f"{nick}/past_interaction",
                event_type="ManagerMethod",
                filter=mm_filter,
                every_n=self._COUNT_WINDOWS["past_interaction"],
            )

            await _reg(
                f"{nick}/past_day",
                event_type="ManagerMethod",
                filter=mm_filter,
                every_seconds=self._TIME_WINDOWS["past_day"],
            )

            # count hierarchy
            for child, (parent, ratio) in self._COUNT_PARENT.items():
                await _reg(
                    f"{nick}/{child}",
                    event_type="_hierarchical_summaries",
                    filter=(
                        f'evt.payload["manager"] == "{nick}" '
                        f'and evt.payload["window"] == "{parent}"'
                    ),
                    every_n=ratio,
                )

            # time hierarchy
            for child, (parent, ratio) in self._TIME_PARENT.items():
                if child == "past_day":
                    continue
                await _reg(
                    f"{nick}/{child}",
                    event_type="_hierarchical_summaries",
                    filter=(
                        f'evt.payload["manager"] == "{nick}" '
                        f'and evt.payload["window"] == "{parent}"'
                    ),
                    every_n=ratio,
                )

        # launch all manager registrations concurrently
        await asyncio.gather(
            *[
                _register_for_manager(mgr_cls, nick)
                for mgr_cls, nick in self._MANAGERS.items()
            ],
        )

        # indicate readiness (useful for tests)
        try:
            self._callbacks_ready.set()  # type: ignore[attr-defined]
        except AttributeError:
            self._callbacks_ready = asyncio.Event()
            self._callbacks_ready.set()

    # 3. Persisting the new snapshot ---------------------------------------
    async def _record_rolling_activity_body(
        self,
        column: str,
        events: list[Event],
    ) -> None:
        """
        Append a **new** row to RollingActivity, copying the previous one and
        updating *column* with a fresh summary.

        • Base-level windows (``past_day`` / ``past_interaction``) are generated
          from the **raw** ManagerMethod events (unchanged behaviour).
        • Higher-level windows are created **only** from the summaries of the
          immediate lower-level window – thus forming a cascade:
              raw events → day → week → 4 weeks → 12 weeks → 52 weeks
              raw events → 1 interaction → 10 → 40 → 120 → 520
        """

        # ---- 0. previous snapshot ----------------------------------------
        prev = unify.get_logs(
            context=self._rolling_ctx,
            sorting={"row_id": "descending"},
            limit=1,
        )
        base_payload = prev[0].entries.copy() if prev else {}

        # ---- helper: concise LLM summary ---------------------------------
        async def _summarise(items: list[str | dict]) -> str:
            if not items:
                return ""
            llm = unify.AsyncUnify(
                "o4-mini@openai",
                cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
                traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            )
            llm.set_system_message(build_activity_events_summary_prompt())
            return (await llm.generate(json.dumps(items, indent=2))).strip()

        # ------------------------------------------------------------------
        mgr_nick, window = column.split("/", 1)

        # ── 1.  Decide data-source (raw events vs. lower-level summaries) ──
        if window in {"past_day", "past_interaction"}:
            # base-level – summarise RAW events
            relevant = [
                {
                    "manager": ev.payload.get("manager"),
                    "method": ev.payload.get("method"),
                    "details": {
                        k: v
                        for k, v in ev.payload.items()
                        if k not in {"manager", "method"}
                    },
                }
                for ev in events
            ]
            summary = await _summarise(relevant)

        else:
            # higher-level – derive from lower-level summaries
            if window in self._TIME_PARENT:
                lower_window, need = self._TIME_PARENT[window]
            elif window in self._COUNT_PARENT:
                lower_window, need = self._COUNT_PARENT[window]
            else:  # unexpected – fall back to raw events
                lower_window, need = None, 0

            if lower_window is None:
                relevant = [
                    {
                        "manager": ev.payload.get("manager"),
                        "method": ev.payload.get("method"),
                        "details": {
                            k: v
                            for k, v in ev.payload.items()
                            if k not in {"manager", "method"}
                        },
                    }
                    for ev in events
                ]
                summary = await _summarise(relevant)
            else:
                # -----------------------  deterministic lineage  -----------------------
                # Every *_hierarchical_summaries* event now carries the **row_id** of the
                # lower-level RollingActivity snapshot it represents.  Using these ids removes
                # the race where the child window used to look at “the most recent N” rows and
                # could therefore pick up *newer* summaries written while it was waiting for the
                # write-lock.

                lower_col = f"{mgr_nick}/{lower_window}"

                # 1.  Collect explicit row_ids from the triggering events (if present).
                row_ids: list[int] = []
                for ev in events:
                    try:
                        rid = int(ev.payload.get("row_id"))  # type: ignore[arg-type]
                    except (ValueError, TypeError, AttributeError):
                        rid = None
                    if rid is not None:
                        row_ids.append(rid)

                collected: list[str] = []

                # 2a. Fetch the *exact* rows referenced by the events.
                if row_ids:
                    for rid in row_ids:
                        rows = unify.get_logs(
                            context=self._rolling_ctx,
                            filter=f"row_id == {rid}",
                            limit=1,
                        )
                        if rows:
                            txt = rows[0].entries.get(lower_col)
                            if txt:
                                collected.append(txt)

                # 2b. If we still need more to reach the required `need` count
                #     (because the callback only delivered the *latest* event),
                #     back-fill by walking backwards from the current row_id.
                if len(collected) < need:
                    # Determine the highest row_id we already included so we
                    # only look at *earlier* snapshots and therefore avoid the
                    # original race condition.
                    max_seen = max(row_ids) if row_ids else None

                    extra_rows = unify.get_logs(
                        context=self._rolling_ctx,
                        sorting={"row_id": "descending"},
                        limit=need * 5,  # generous buffer; we'll filter below
                    )
                    for lg in extra_rows:
                        rid = lg.entries.get("row_id")
                        if max_seen is not None and rid is not None and rid > max_seen:
                            # newer than the latest row we already have → skip
                            continue
                        txt = lg.entries.get(lower_col)
                        if not txt:
                            continue
                        if rid in row_ids:
                            continue  # already have it
                        collected.append(txt)
                        if len(collected) >= need:
                            break

                summary = await _summarise(collected)

        # ---- 2.  persist --------------------------------------------------
        # Ensure **new** row creation: remove any inherited `row_id` so Unify
        # allocates a fresh sequence number instead of silently updating the
        # previous snapshot (which would make successive interactions appear
        # as a single row and break tests expecting one row per call).
        base_payload.pop("row_id", None)

        base_payload[column] = summary

        # ──────────────────────────  Pre-compute summaries  ────────────────
        base_payload[self._SUMMARY_TIME_COL] = self._build_activity_summary(
            base_payload,
            "time",
        )
        base_payload[self._SUMMARY_COUNT_COL] = self._build_activity_summary(
            base_payload,
            "interaction",
        )

        # ------------------------------------------------------------------
        # -----------------------  write new snapshot -----------------------
        unify.log(
            context=self._rolling_ctx,
            new=True,
            mutable=True,
            **base_payload,
        )

        # Retrieve the *row_id* of the snapshot just written so that dependant
        # windows know exactly which lower-level rows to aggregate.
        try:
            _last_row = unify.get_logs(
                context=self._rolling_ctx,
                sorting={"row_id": "descending"},
                limit=1,
            )[0]
            new_row_id = _last_row.entries.get("row_id")
        except Exception:
            new_row_id = None

        # ---- 2b.  update global cache --------------------------------------
        # Keep the in-process snapshot in sync so prompt builders never have
        # to query the backend after the initial bootstrap.
        try:
            set_broader_context(base_payload[self._SUMMARY_TIME_COL])
        except Exception:
            # Defensive guard – updating the cache must never break the caller.
            pass

        # ---- 3.  notify dependants ----------------------------------------
        # Emit a *_hierarchical_summaries* event so higher-level windows trigger only
        # after the lower-level snapshot is fully written.
        await EVENT_BUS.publish(
            Event(
                type="_hierarchical_summaries",
                payload={
                    "manager": mgr_nick,  # e.g. "contact_manager"
                    "window": window,  # e.g. "past_10_interactions"
                    "row_id": new_row_id,  # lineage id for deterministic roll-ups
                },
            ),
        )

    # ------------------------------------------------------------------ #
    #  Wrapper: guarantees single writer for RollingActivity             #
    # ------------------------------------------------------------------ #
    async def _record_rolling_activity(
        self,
        column: str,
        events: list[Event],
    ) -> None:
        """
        Thread-safe wrapper around :py:meth:`_record_rolling_activity_body` that
        ensures only *one* coroutine at a time can append a new snapshot to the
        ``RollingActivity`` context.  This prevents scenarios where two
        concurrent callbacks would both read the same *latest* row, apply their
        individual update, and then write out diverging successors derived from
        an inconsistent base state.
        """

        async with self._rolling_lock:
            await self._record_rolling_activity_body(column, events)

    # ------------------------------------------------------------------ #
    #  helper: build human-readable activity summary                     #
    # ------------------------------------------------------------------ #
    def _build_activity_summary(
        self,
        entries: dict[str, str],
        mode: str = "time",
    ) -> str:
        """Return the rolling activity summary Markdown for *entries*."""
        mode = mode.lower()
        if mode not in {"time", "interaction"}:
            raise ValueError("mode must be either 'time' or 'interaction'")

        windows: list[str] = (
            list(self._TIME_ORDER) if mode == "time" else list(self._COUNT_ORDER)
        )

        def _pretty(w: str) -> str:
            parts = w.split("_")
            return "Past " + " ".join(
                p.capitalize() if not p.isdigit() else p for p in parts[1:]
            )

        _TITLE_DESC = {
            "task_scheduler": (
                "Tasks",
                "Overview of the tasks scheduled, updated, and performed.",
            ),
            "knowledge_manager": (
                "Knowledge",
                "Overview of the long-term memory (knowledge) added, updated, restructured, removed etc.",
            ),
            "contact_manager": (
                "Contacts",
                "Overview of contacts created or updated and related actions.",
            ),
            "transcript_manager": (
                "Transcripts",
                "Overview of messages and transcript summaries.",
            ),
            "conductor": (
                "Orchestration",
                "High-level orchestration and planning actions.",
            ),
        }

        lines: list[str] = []
        for mgr_cls, nick in self._MANAGERS.items():
            title, desc = _TITLE_DESC.get(
                nick,
                (mgr_cls.replace("Manager", ""), ""),
            )

            available: list[tuple[str, str]] = []
            for w in windows:
                col = f"{nick}/{w}"
                summary = entries.get(col)
                if summary:
                    available.append((w, summary))

            if not available:
                continue

            lines.append(f"# {title}")
            if desc:
                lines.append(desc)
            lines.append("")

            for w, summary in available:
                lines.append(f"## {_pretty(w)}")
                lines.append(summary)
                lines.append("")

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------ #
    # 5  get_broader_context                                            #
    # ------------------------------------------------------------------ #
    def get_rolling_activity(self, mode: str = "time") -> str:
        """
        Return the **latest** Rolling-Activity snapshot as a human-readable
        Markdown string.
        """
        mode = mode.lower()
        if mode not in {"time", "interaction"}:
            raise ValueError("mode must be either 'time' or 'interaction'")

        rows = unify.get_logs(
            context=self._rolling_ctx,
            sorting={"row_id": "descending"},
            limit=1,
        )
        # If there is no stored rolling activity yet, return an *empty* string so
        # callers can completely omit the Historic Activity block.  This avoids
        # polluting system prompts with a verbose placeholder that carries no
        # useful information.
        if not rows:
            return ""

        latest = rows[0].entries
        key = self._SUMMARY_TIME_COL if mode == "time" else self._SUMMARY_COUNT_COL
        stored = latest.get(key)
        if stored:
            return stored

        # Fallback – build on the fly if snapshot predates summary columns
        return self._build_activity_summary(latest, mode)
