# memory_manager/memory_manager.py
from __future__ import annotations

import asyncio
import json
import os
import functools
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

import unify

from ..contact_manager.contact_manager import ContactManager
from ..transcript_manager.transcript_manager import TranscriptManager
from ..knowledge_manager.knowledge_manager import KnowledgeManager
from ..task_scheduler.task_scheduler import TaskScheduler
from ..common.llm_helpers import methods_to_tool_dict
from ..common.async_tool_loop import start_async_tool_loop
from .prompt_builders import (
    build_contact_update_prompt,
    build_bio_prompt,
    build_rolling_prompt,
    build_knowledge_prompt,
    build_response_policy_prompt,
)
from .base import BaseMemoryManager
from ..events.event_bus import EVENT_BUS, Event


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
    Offline helper that processes transcripts in chunks (~50 messages by default).
    """

    # Rolling activity logic removed

    # ------------------------------------------------------------------ #
    #  Shared helper: convert Message / event dicts to plain-text transcript
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_plain_transcript(
        messages: list[dict],
        contact_manager: Optional["ContactManager"] = None,
    ) -> str:
        """Return *plain-text* view of the message/event list.

        • Chat messages are rendered exactly like `TranscriptManager.build_plain_transcript`.
        • *ManagerMethod* events (kind == "manager_method") are preserved as **raw JSON** so
          downstream helpers/tests can still inspect all attributes (the tests assert the
          literal substring `"kind": "manager_method"` is present).
        """

        # Delegate chat messages to the central helper so naming etc. stays consistent
        from unity.transcript_manager.transcript_manager import (
            TranscriptManager as _TM,
        )

        plain_chat = _TM.build_plain_transcript(
            messages,
            contact_manager=contact_manager,
        )

        # Append serialised manager-method events (if any) *in the order they appear*
        extra_lines: list[str] = []
        for itm in messages:
            if itm.get("kind") == "manager_method":
                try:
                    import json  # local import to avoid polluting module namespace

                    dat = itm.get("data", {})
                    # Omit only EventBus metadata keys; include everything else so
                    # summaries retain meaningful content (question/request/answer/etc.).
                    keys_to_omit = {
                        "row_id",
                        "event_id",
                        "calling_id",
                        "type",
                        "timestamp",
                        "event_timestamp",
                        "payload_cls",
                    }
                    concise = {
                        "kind": "manager_method",
                        **{k: v for k, v in dat.items() if k not in keys_to_omit},
                    }
                    extra_lines.append(json.dumps(concise))
                except Exception:
                    # Fallback – best-effort string conversion
                    extra_lines.append(str(itm))

        if extra_lines:
            if plain_chat:
                return "\n".join([plain_chat] + extra_lines)
            return "\n".join(extra_lines)

        return plain_chat

    # ---------------------------------------------------------------------- #
    @dataclass(frozen=True)
    class MemoryConfig:
        """Configuration flags that control orchestration/runtime behaviour.

        enable_callbacks: When True, register EventBus callbacks for message
            ingestion and manager-method tracking. When False, skip callback
            registration.
        """

        enable_callbacks: bool = True

    def __init__(
        self,
        *,
        contact_manager: Optional[ContactManager] = None,
        transcript_manager: Optional[TranscriptManager] = None,
        knowledge_manager: Optional[KnowledgeManager] = None,
        task_scheduler: Optional[TaskScheduler] = None,
        config: Optional["MemoryManager.MemoryConfig"] = None,
    ):

        self._contact_manager = contact_manager or ContactManager()
        self._transcript_manager = transcript_manager or TranscriptManager(
            contact_manager=self._contact_manager,
        )
        self._knowledge_manager = knowledge_manager or KnowledgeManager()
        self._task_scheduler = task_scheduler or TaskScheduler()

        # ── Config & environment-controlled callback registration ----------------
        self._cfg: MemoryManager.MemoryConfig = (
            config if config is not None else MemoryManager.MemoryConfig()
        )
        self._register_update_callbacks: bool = (
            self._cfg.enable_callbacks and _env_flag("REGISTER_UPDATE_CALLBACKS", True)
        )
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

            # 2.  Listen to explicit ManagerMethod events so they can be included
            #     in the transcript chunk payloads passed to the LLM.
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
        update_response_policies: bool = True,
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

            # Strictly filter to parameters accepted by the underlying
            # _create_contact implementation to drop any tool-loop control
            # kwargs (e.g. pause_event, interject_queue, ...).
            import inspect  # local import

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

        #  merge_contacts – expose full-contact merge to the LLM
        #
        #  Unlike the create/update wrappers above, *all* columns are allowed
        #  during a merge, therefore we only strip internal helper parameters
        #  (e.g. parent_chat_context) that the underlying implementation does
        #  not understand – no additional field restrictions are applied.
        @functools.wraps(self._contact_manager._merge_contacts, updated=())
        async def _safe_merge_contacts(**kwargs):
            # Remove any kwargs that _merge_contacts is not expecting (helps
            # avoid accidental leaks of hidden parameters coming from the
            # tool-use loop itself).
            import inspect  # local import to avoid polluting module namespace

            allowed = set(
                inspect.signature(self._contact_manager._merge_contacts).parameters,
            )
            cleaned_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

            return await asyncio.to_thread(
                self._contact_manager._merge_contacts,
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

        # Base helpers -----------------------------------------------------------
        # Always expose the same tools across real and simulated environments.
        tools: Dict[str, Callable[..., Any]] = {
            "contact_ask": self._contact_manager.ask,
            "create_contact": _safe_create_contact,
            "update_contact": _safe_update_contact,
            # Full-contact merge helper (no field restrictions)
            "merge_contacts": _safe_merge_contacts,
        }

        # ─ 2.  LLM client
        llm = unify.AsyncUnify(
            "gpt-5@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )
        llm.set_system_message(build_contact_update_prompt(tools, guidance))

        # ─ 3.  Kick off *single* tool-use loop
        handle = start_async_tool_loop(
            llm,
            transcript,
            tools,
            loop_id="MemoryManager.update_contacts",
            # Align with simulated policy: require tool use for the first two steps
            tool_policy=lambda i, t: ("required", t) if i < 2 else ("auto", t),
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
                if update_response_policies:
                    follow_up_tasks.append(
                        asyncio.create_task(
                            self.update_contact_response_policy(
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

        async def set_bio(bio: str) -> str:
            """Update only the bio column for the target contact id captured in the closure."""
            final_id = target_id
            if final_id is None:
                raise ValueError(
                    "contact_id is required but was not provided by the caller.",
                )
            await asyncio.to_thread(
                self._contact_manager._update_contact,
                contact_id=final_id,
                bio=bio,
            )
            return f"Bio for contact with id {final_id} successfully updated"

        tools: Dict[str, Callable[..., Any]] = {
            "set_bio": set_bio,
        }

        llm = unify.AsyncUnify(
            "gpt-5@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )
        contacts = await asyncio.to_thread(
            self._contact_manager._filter_contacts,
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
                self._contact_manager._filter_contacts,
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

        handle = start_async_tool_loop(
            llm,
            user_blob,
            tools,
            loop_id="MemoryManager.update_contact_bio",
            tool_policy=lambda i, t: ("required", t) if i < 1 else ("auto", t),
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

        async def set_rolling_summary(rolling_summary: str) -> str:
            final_id = target_id
            if final_id is None:
                raise ValueError(
                    "contact_id is required but was not provided by the caller.",
                )
            await asyncio.to_thread(
                self._contact_manager._update_contact,
                contact_id=final_id,
                rolling_summary=rolling_summary,
            )
            return (
                f"Rolling summary for contact with id {final_id} successfully updated"
            )

        tools: Dict[str, Callable[..., Any]] = {
            "set_rolling_summary": set_rolling_summary,
        }

        # ────────────────────────────────────────────────────────────────
        #   Retrieve contact details once (name + current rolling summary)
        #   so we can build the prompt and seed the user payload without
        #   duplicating backend look-ups later on.
        # ----------------------------------------------------------------
        try:
            contacts = await asyncio.to_thread(
                self._contact_manager._filter_contacts,
                filter=f"contact_id == {contact_id}",
                limit=1,
            )
            c0 = contacts[0] if contacts else None
            contact_name_val = (
                " ".join(p for p in [c0.first_name, c0.surname] if p).strip()
                if c0
                else None
            )
            latest_summary_val = c0.rolling_summary if c0 else None
        except Exception:
            contact_name_val = None
            latest_summary_val = None  # best-effort fallback

        # ------------------------------------------------------------------
        llm = unify.AsyncUnify(
            "gpt-5@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )
        contact_label = (
            f"{contact_name_val} (id {contact_id})"
            if contact_name_val
            else f"id {contact_id}"
        )
        llm.set_system_message(
            build_rolling_prompt(
                contact_label,
                tools,
                guidance=combined_guidance,
            ),
        )

        # ------------------------------------------------------------------
        # Build the LLM *user* payload using the already-retrieved summary.
        # ------------------------------------------------------------------
        user_blob = json.dumps(
            {
                "contact_id": contact_id,
                "latest_rolling_summary (to maybe update)": latest_summary_val,
                "transcript": transcript,
            },
            indent=2,
        )

        handle = start_async_tool_loop(
            llm,
            user_blob,
            tools,
            loop_id="MemoryManager.update_contact_rolling_summary",
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )

        return await handle.result()

    # ------------------------------------------------------------------ #
    # 4  update_contact_response_policy                                  #
    # ------------------------------------------------------------------ #
    async def update_contact_response_policy(
        self,
        transcript: str,
        *,
        contact_id: int,
        guidance: Optional[str] = None,
    ) -> str:
        """Refresh the *response_policy* column for the given contact."""

        # Ensure the assistant's response policy uses **second person**.
        assistant_extra = (
            "IMPORTANT: This response policy belongs to the assistant itself (contact_id 0). Always use **second-person** pronouns ('you') when describing the assistant. Never refer to the assistant in the third person."  # noqa: E501
            if contact_id == 0
            else None
        )

        combined_guidance: Optional[str] = (
            "\n".join(g for g in (guidance, assistant_extra) if g) or None
        )
        target_id = contact_id  # capture for closure

        async def set_response_policy(response_policy: str) -> str:
            final_id = target_id
            if final_id is None:
                raise ValueError(
                    "contact_id is required but was not provided by the caller.",
                )
            await asyncio.to_thread(
                self._contact_manager._update_contact,
                contact_id=final_id,
                response_policy=response_policy,
            )
            return (
                f"Response policy for contact with id {final_id} successfully updated"
            )

        tools: Dict[str, Callable[..., Any]] = {
            "set_response_policy": set_response_policy,
        }

        # ────────────────────────────────────────────────────────────────
        #   Retrieve contact details once (name + current response policy)
        #   so we can build the prompt and seed the user payload without
        #   duplicating backend look-ups later on.
        # ----------------------------------------------------------------
        try:
            contacts = await asyncio.to_thread(
                self._contact_manager._filter_contacts,
                filter=f"contact_id == {contact_id}",
                limit=1,
            )
            c0 = contacts[0] if contacts else None
            contact_name_val = (
                " ".join(p for p in [c0.first_name, c0.surname] if p).strip()
                if c0
                else None
            )
            latest_policy_val = c0.response_policy if c0 else None
        except Exception:
            contact_name_val = None
            latest_policy_val = None  # best-effort fallback

        # ------------------------------------------------------------------
        llm = unify.AsyncUnify(
            "gpt-5@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )
        contact_label = (
            f"{contact_name_val} (id {contact_id})"
            if contact_name_val
            else f"id {contact_id}"
        )
        llm.set_system_message(
            build_response_policy_prompt(
                contact_label,
                tools,
                guidance=combined_guidance,
            ),
        )

        # ------------------------------------------------------------------
        # Build the LLM *user* payload using the already-retrieved policy.
        # ------------------------------------------------------------------
        user_blob = json.dumps(
            {
                "contact_id": contact_id,
                "latest_response_policy (to maybe update)": latest_policy_val,
                "transcript": transcript,
            },
            indent=2,
        )

        handle = start_async_tool_loop(
            llm,
            user_blob,
            tools,
            loop_id="MemoryManager.update_contact_response_policy",
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

        # Use the instance-provided KnowledgeManager (real or simulated)
        _km = self._knowledge_manager

        tools: Dict[str, Callable[..., Any]] = methods_to_tool_dict(
            self._contact_manager.ask,
            _km.ask,
            _km.refactor,
            _km.update,
            include_class_name=True,
        )

        llm = unify.AsyncUnify(
            "gpt-5@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )
        llm.set_system_message(build_knowledge_prompt(tools, guidance))

        handle = start_async_tool_loop(
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
            "gpt-5@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )

        from .prompt_builders import build_task_prompt  # local import to avoid cycles

        llm.set_system_message(build_task_prompt(tools, guidance))

        handle = start_async_tool_loop(
            llm,
            transcript,
            tools,
            loop_id="MemoryManager.update_tasks",
            tool_policy=lambda i, _: ("required", _) if i < 2 else ("auto", _),
        )

        return await handle.result()

    # ------------------------------------------------------------------ #
    # 5  reset – simplified helper                                       #
    # ------------------------------------------------------------------ #
    async def reset(self) -> None:  # noqa: D401 – imperative name
        """Reset the event bus and re-register message-related callbacks."""

        EVENT_BUS.reset()

        if self._register_update_callbacks:
            await asyncio.gather(
                self._setup_message_callbacks(),
                self._setup_explicit_call_callbacks(),
            )

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
    #       Capture *explicit* ManagerMethod invocations coming from the
    #       ConversationManager so the passive 30-message chunk has full
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
        """Append explicit ManagerMethod events to the current buffer.

        Note: These events do NOT advance the 50-message counter; only
        actual chat `Message` events count towards flushing the chunk.
        """

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

        # Advance the counter for manager-method events as well so that
        # passive chunks include explicit tool invocations without waiting
        # for additional chat messages.
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
                            self.update_contact_response_policy(
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
    @classmethod
    def get_rolling_activity(cls, mode: str = "time") -> str:
        """Rolling activity has been removed; return an empty string."""
        return ""
