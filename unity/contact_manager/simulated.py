# unity/contact_manager/simulated_contact_manager.py
from __future__ import annotations

import asyncio
import json
import functools
import threading
from typing import List, Dict, Any, Optional, Type, TYPE_CHECKING

import unillm
from pydantic import BaseModel, Field
from .base import BaseContactManager
from .types.contact import Contact
from .prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
    build_simulated_method_prompt,
)
from ..common.async_tool_loop import SteerableToolHandle
from ..common.simulated import (
    mirror_contact_manager_tools,
    SimulatedLineage,
    SimulatedLog,
    simulated_llm_roundtrip,
    SimulatedHandleMixin,
    build_followup_prompt,
    maybe_tool_log_scheduled,
    maybe_tool_log_completed,
)
from ..logger import LOGGER
from ..common.hierarchical_logger import ICONS

# ─────────────────────────────────────────────────────────────────────────────
# Structured response models for simulated private methods
# ─────────────────────────────────────────────────────────────────────────────


class _ContactRecord(Contact):
    # Inherit all fields from the canonical Contact model to avoid drift and
    # ensure a single source of truth. Tighten schema for OpenAI structured
    # output by forbidding extras at this wrapper level.
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


_ContactRecord.model_rebuild()


class _ContactsListResponse(BaseModel):
    contacts: List[_ContactRecord]

    # OpenAI structured output requires top-level additionalProperties=false
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


_ContactsListResponse.model_rebuild()


class _UpdateDetails(BaseModel):
    contact_id: int

    # Disallow extra keys; keeps schema strict for OpenAI parser
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


class _UpdateOutcome(BaseModel):
    outcome: str
    details: _UpdateDetails

    # Enforce top-level additionalProperties=false for structured output
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


_UpdateOutcome.model_rebuild()


class _CreateDetails(BaseModel):
    contact_id: int

    # Disallow extra keys; keeps schema strict for OpenAI parser
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


class _CreateOutcome(BaseModel):
    outcome: str
    details: _CreateDetails

    # Enforce top-level additionalProperties=false for structured output
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


_CreateOutcome.model_rebuild()


class _DeleteDetails(BaseModel):
    contact_id: int

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


class _DeleteOutcome(BaseModel):
    outcome: str
    details: _DeleteDetails

    # Enforce top-level additionalProperties=false for structured output
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


_DeleteOutcome.model_rebuild()


class _MergeDetails(BaseModel):
    kept_contact_id: int
    deleted_contact_id: int
    # Represent overrides as an empty object with additionalProperties: false
    overrides: dict = Field(
        default_factory=dict,
        json_schema_extra={"additionalProperties": False},
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


class _MergeOutcome(BaseModel):
    outcome: str
    details: _MergeDetails

    # Enforce top-level additionalProperties=false for structured output
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


_MergeOutcome.model_rebuild()


class _MergeDetailsStrict(BaseModel):
    kept_contact_id: int
    deleted_contact_id: int

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


class _MergeOutcomeStrict(BaseModel):
    outcome: str
    details: _MergeDetailsStrict

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


_MergeOutcomeStrict.model_rebuild()


# ─────────────────────────────────────────────────────────────────────────────
# Internal handle
# ─────────────────────────────────────────────────────────────────────────────
class _SimulatedContactHandle(SimulatedHandleMixin, SteerableToolHandle):
    """
    Minimal LLM-backed handle used by SimulatedContactManager.ask / update.
    """

    def __init__(
        self,
        llm: unillm.Unify,
        initial_text: str,
        *,
        mode: str,
        _return_reasoning_steps: bool,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
        response_format: Optional[Type[BaseModel]] = None,
        hold_completion: bool = False,
    ):
        self._llm = llm
        self._initial = initial_text
        self._want_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        self._mode = str(mode or "ask")
        self._response_format = response_format
        if _requests_clarification and (
            not clarification_up_q or not clarification_down_q
        ):
            raise ValueError(
                "Clarification queues must be provided when _requests_clarification is True",
            )
        self._needs_clar = _requests_clarification

        # Human-friendly log label derived from current lineage, mirroring other simulated managers:
        # "<outer...>->SimulatedContactManager.<mode>(abcd)"
        self._log_label = SimulatedLineage.make_label(
            f"SimulatedContactManager.{self._mode}",
        )

        self._init_completion_gate(hold_completion)

        if self._needs_clar:
            try:
                q_text = "Could you clarify your request about contacts?"
                self._clar_up_q.put_nowait(q_text)
                try:
                    SimulatedLog.log_clarification_request(self._log_label, q_text)
                except Exception:
                    pass
                try:
                    LOGGER.info(
                        f"{ICONS['clarification']} [{self._log_label}] Clarification requested",
                    )
                except Exception:
                    pass
            except asyncio.QueueFull:
                pass

        self._extra_msgs: List[str] = []

        self._done = threading.Event()
        self._cancelled = False
        self._answer: str | None = None
        self._messages: List[Dict[str, Any]] = []
        self._paused = False
        # Async cancellation signal to break out of awaited clarification
        self._cancel_event: asyncio.Event = asyncio.Event()

    # --------------------------------------------------------------------- #
    # SteerableToolHandle implementation
    # --------------------------------------------------------------------- #
    async def result(self):
        if self._cancelled:
            return "processed stopped early, no result"

        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done.is_set():
            if self._needs_clar:
                try:
                    LOGGER.info(
                        f"{ICONS['pending']} [{self._log_label}] Waiting for clarification answer…",
                    )
                except Exception:
                    pass
                # Wait for either a clarification answer or cancellation
                clar: str | None = None
                get_task = asyncio.create_task(self._clar_down_q.get())
                cancel_task = asyncio.create_task(self._cancel_event.wait())
                done, pending = await asyncio.wait(
                    {get_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                if cancel_task in done:
                    # Cancelled: finish immediately
                    self._done.set()
                    return "processed stopped early, no result"
                try:
                    clar = get_task.result()
                except Exception:
                    clar = None
                if clar is None:
                    self._done.set()
                    return "processed stopped early, no result"
                self._extra_msgs.append(f"Clarification: {clar}")
                try:
                    SimulatedLog.log_clarification_answer(self._log_label, clar)
                except Exception:
                    pass
                try:
                    LOGGER.info(
                        f"{ICONS['interjection']} [{self._log_label}] Clarification answer received",
                    )
                except Exception:
                    pass

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_msgs)

            # Unified simulated LLM roundtrip with lineage-aware logging and gated response preview
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            answer = await simulated_llm_roundtrip(
                self._llm,
                label=self._log_label,
                prompt=prompt,
                response_format=self._response_format,
            )
            self._answer = answer
            self._messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            # Await the completion gate (no-op when hold_completion=False).
            await self._await_completion_gate()
            self._done.set()

        # If cancellation happened after the coroutine started, return a stable post-cancel value.
        if self._cancelled:
            return "processed stopped early, no result"
        if self._want_steps:
            return self._answer, self._messages
        return self._answer

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> None:
        """Interject a message into the in-flight handle.

        Args:
            message: The interjection message to inject.
            _parent_chat_context_cont: Optional continuation of parent chat context.
                Accepted for API parity with real handles but not currently used.
        """
        if self._cancelled:
            return
        self._log_interject(message)
        self._extra_msgs.append(message)

    async def stop(
        self,
        reason: str | None = None,
        **kwargs,
    ) -> None:
        """Stop the in-flight handle.

        Args:
            reason: Optional reason for stopping.
            _parent_chat_context_cont: Optional continuation of parent chat context.
                Accepted for API parity with real handles but not currently used.
        """
        self._log_stop(reason)
        self._cancelled = True
        self._open_completion_gate()
        try:
            self._cancel_event.set()
        except Exception:
            pass
        self._done.set()

    async def pause(self) -> str:
        if self._paused:
            return "Already paused."
        self._log_pause()
        self._paused = True
        return "Paused."

    async def resume(self) -> str:
        if not self._paused:
            return "Already running."
        self._log_resume()
        self._paused = False
        return "Resumed."

    def done(self) -> bool:  # type: ignore[override]
        return self._done.is_set() and self._gate_open

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
    ) -> "SteerableToolHandle":
        """Ask a follow-up question about the current operation.

        Args:
            question: The question to ask.
            parent_chat_context: Optional parent chat context for the inspection loop.
                Accepted for API parity with real handles but not currently used.
        """
        follow_up_prompt = build_followup_prompt(
            question=question,
            initial_instruction=self._initial,
            extra_messages=list(self._extra_msgs),
        )
        # Create a nested helper handle so we can log using its stable label, mirroring TaskScheduler/Actor
        handle = _SimulatedContactHandle(
            self._llm,
            follow_up_prompt,
            mode=self._mode,
            _return_reasoning_steps=self._want_steps,
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )
        # Align with other simulated components: concise "Question(<parent>)" label
        try:
            handle._log_label = SimulatedLineage.question_label(  # type: ignore[attr-defined]
                self._log_label,
            )
        except Exception:
            pass
        try:
            SimulatedLog.log_request("ask", getattr(handle, "_log_label", ""), question)  # type: ignore[arg-type]
        except Exception:
            pass
        return handle

    # --- event APIs required by SteerableToolHandle ---------------------
    async def next_clarification(self) -> dict:
        """Block until a clarification arrives, or forever if not requested."""
        if not getattr(self, "_needs_clar", False):
            return await super().next_clarification()
        try:
            if self._clar_up_q is not None:
                msg = await self._clar_up_q.get()
                return {
                    "type": "clarification",
                    "call_id": "unknown",
                    "tool_name": "unknown",
                    "question": msg,
                }
        except Exception:
            pass
        return await super().next_clarification()

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        try:
            if self._clar_down_q is not None:
                await self._clar_down_q.put(answer)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Public simulated manager
# ─────────────────────────────────────────────────────────────────────────────
# Adding rolling summary flag


class SimulatedContactManager(BaseContactManager):
    """
    Drop-in replacement for ContactManager with imaginary data and
    stateful LLM memory.

    Maintains an internal contact store for deterministic state tracking.
    All programmatic methods (`get_contact_info`, `filter_contacts`, `_create_contact`,
    etc.) operate on the internal store deterministically.

    The `ask` and `update` methods use LLM for natural language responses:
    - In **deterministic mode** (default): LLM sees the actual store contents,
      producing responses grounded in real data.
    - In **freeform mode**: LLM uses the description to imagine contacts,
      with no awareness of the internal store (original behavior).
    """

    # System contact IDs (cannot be deleted)
    ASSISTANT_CONTACT_ID = 0
    USER_CONTACT_ID = 1

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
        *,
        deterministic: bool = True,
        log_events: bool = False,
        rolling_summary_in_prompts: bool = True,
        simulation_guidance: Optional[str] = None,
        hold_completion: bool = False,
        # Accept but ignore extra parameters for compatibility
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._description = description
        self._deterministic = deterministic
        self._log_events = log_events
        self._simulation_guidance = simulation_guidance
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._hold_completion = hold_completion

        # ─────────────────────────────────────────────────────────────────────
        # Internal contact store for deterministic state tracking
        # ─────────────────────────────────────────────────────────────────────
        # Pre-populate system contacts (assistant=0, user=1)
        self._contacts: Dict[int, Dict[str, Any]] = {
            self.ASSISTANT_CONTACT_ID: {
                "contact_id": self.ASSISTANT_CONTACT_ID,
                "first_name": "Default",
                "surname": "Assistant",
                "email_address": "assistant@example.com",
                "phone_number": "+15555550000",
                "bio": "The AI assistant",
                "should_respond": True,
                "is_system": True,
            },
            self.USER_CONTACT_ID: {
                "contact_id": self.USER_CONTACT_ID,
                "first_name": "Default",
                "surname": "User",
                "email_address": "user@example.com",
                "phone_number": "+15555550001",
                "bio": "The primary user/boss",
                "should_respond": True,
                "is_system": True,
            },
        }
        # Counter for assigning new contact_ids (starts at 2, 0 and 1 are system)
        self._next_contact_id: int = 2

        # Shared, *stateful* **asynchronous** LLM
        from ..common.llm_client import (
            new_llm_client as _new_llm_client,
        )  # local import to avoid cycles

        self._llm = _new_llm_client(
            stateful=True,
            origin="SimulatedContactManager",
        )
        # Mirror the real manager's tool exposure programmatically
        # and build the *exact* same prompts via the shared builders.
        ask_tools = mirror_contact_manager_tools("ask")
        upd_tools = mirror_contact_manager_tools("update")

        ask_msg = build_ask_prompt(
            ask_tools,
            10,
            [{k: str(v.annotation)} for k, v in Contact.model_fields.items()],
            include_activity=self._rolling_summary_in_prompts,
        ).flatten()
        upd_msg = build_update_prompt(
            upd_tools,
            10,
            [{k: str(v.annotation)} for k, v in Contact.model_fields.items()],
            include_activity=self._rolling_summary_in_prompts,
        ).flatten()

        # Set system message based on mode
        if self._deterministic:
            # Deterministic mode: LLM will see actual store contents in each query
            self._llm.set_system_message(
                "You are a contact-manager assistant with access to a real contact store. "
                "When answering questions or processing updates, base your responses on the "
                "actual contact data provided in each query. Do not invent contacts.\n\n"
                "As a reference, the system messages for the *real* contact-manager 'ask' and 'update' methods are as follows.\n"
                "You do not have access to any real tools, so you should just create a final answer to the question/request.\n\n"
                f"'ask' system message:\n{ask_msg}\n\n"
                f"'update' system message:\n{upd_msg}\n\n",
            )
        else:
            # Freeform mode: LLM imagines contacts based on description
            self._llm.set_system_message(
                "You are a *simulated* contact-manager assistant. "
                "There is no real database; invent plausible contact records and "
                "keep your story consistent across turns.\n\n"
                "As a reference, the system messages for the *real* contact-manager 'ask' and 'update' methods are as follows."
                "You do not have access to any real tools, so you should just create a final answer to the question/request.\n\n"
                f"'ask' system message:\n{ask_msg}\n\n"
                f"'update' system message:\n{upd_msg}\n\n"
                f"Back-story: {self._description}",
            )

    def _get_store_summary(self) -> str:
        """Generate a summary of the current contact store for LLM context."""
        if not self._contacts:
            return "The contact store is currently empty."

        lines = [f"Current contact store ({len(self._contacts)} contacts):"]
        for cid, contact in sorted(self._contacts.items()):
            name_parts = []
            if contact.get("first_name"):
                name_parts.append(contact["first_name"])
            if contact.get("surname"):
                name_parts.append(contact["surname"])
            name = " ".join(name_parts) or "(unnamed)"

            details = []
            if contact.get("email_address"):
                details.append(f"email: {contact['email_address']}")
            if contact.get("phone_number"):
                details.append(f"phone: {contact['phone_number']}")
            if contact.get("is_system"):
                details.append("(system)")

            detail_str = f" [{', '.join(details)}]" if details else ""
            lines.append(f"  - contact_id={cid}: {name}{detail_str}")

        return "\n".join(lines)

    def reduce(
        self,
        *,
        metric: str,
        keys: str | list[str],
        filter: Optional[str | dict[str, str]] = None,
        group_by: Optional[str | list[str]] = None,
    ) -> Any:
        """
        Simulated counterpart of the ContactManager.reduce tool.

        This method does **not** talk to a real datastore. It returns
        deterministic, shape-correct placeholder values so tests and demos can
        rely on the same return shapes as the concrete manager:

        * Single key, no grouping  → scalar.
        * Multiple keys, no grouping → ``dict[key -> scalar]``.
        * With grouping             → nested ``dict[group -> value or dict]``.
        """

        def _scalar(k: str) -> float:
            # Simple deterministic value derived from the key name
            return float(len(str(k)) or 1)

        key_list: list[str] = [keys] if isinstance(keys, str) else list(keys)

        # No grouping → scalar or {key -> scalar}
        if group_by is None:
            if isinstance(keys, str):
                return _scalar(keys)
            return {k: _scalar(k) for k in key_list}

        # With grouping → {group_value -> scalar or {key -> scalar}}
        groups: list[str] = (
            [group_by] if isinstance(group_by, str) else [str(g) for g in group_by]
        )
        if isinstance(keys, str):
            return {g: _scalar(keys) for g in groups}
        return {g: {k: _scalar(k) for k in key_list} for g in groups}

    # --------------------------------------------------------------------- #
    # get_contact_info - deterministic lookup from internal store           #
    # --------------------------------------------------------------------- #
    def get_contact_info(
        self,
        contact_id: int | List[int],
        fields: Optional[str | List[str]] = None,
        search_local_storage: bool = True,  # ignored, kept for API compat
    ) -> Dict[int, Dict[str, Any]]:
        """
        Return a mapping of requested fields for one or many contacts.

        This method queries the internal contact store deterministically.
        Unlike the LLM-backed methods, this returns actual stored state.

        Parameters
        ----------
        contact_id : int | list[int]
            Single contact ID or list of contact IDs to retrieve.
        fields : str | list[str] | None
            Specific fields to include. If None or "all", returns all fields.
        search_local_storage : bool
            Ignored (kept for API compatibility with ContactManager).

        Returns
        -------
        dict[int, dict]
            Mapping of contact_id → field→value. Missing IDs are omitted.
        """
        # Normalise to list
        ids = [contact_id] if isinstance(contact_id, int) else list(contact_id)

        # Normalise fields
        if fields is None or (isinstance(fields, str) and fields.lower() == "all"):
            requested_fields: Optional[List[str]] = None  # all fields
        elif isinstance(fields, str):
            requested_fields = [fields]
        else:
            requested_fields = list(fields)

        result: Dict[int, Dict[str, Any]] = {}
        for cid in ids:
            contact = self._contacts.get(cid)
            if contact is not None:
                if requested_fields is None:
                    result[cid] = dict(contact)
                else:
                    result[cid] = {
                        k: v for k, v in contact.items() if k in requested_fields
                    }
        return result

    # --------------------------------------------------------------------- #
    # ask                                                                   #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseContactManager.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        # Build instruction with store data in deterministic mode
        if self._deterministic:
            store_summary = self._get_store_summary()
            instruction = (
                f"{store_summary}\n\n"
                f"User question: {text}\n\n"
                "Please answer the question based on the contact data above."
            )
        else:
            instruction = build_simulated_method_prompt(
                "ask",
                text,
                parent_chat_context=_parent_chat_context,
            )
        handle = _SimulatedContactHandle(
            self._llm,
            instruction,
            mode="ask",
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            response_format=response_format,
            hold_completion=self._hold_completion,
        )

        # Tool-style scheduled log (only when no parent lineage)
        maybe_tool_log_scheduled(
            "SimulatedContactManager.ask",
            "ask",
            {"text": text, "requests_clarification": _requests_clarification},
        )

        # No EventBus publishing for simulated managers

        return handle

    # --------------------------------------------------------------------- #
    # update                                                                #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseContactManager.update, updated=())
    async def update(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        # In deterministic mode, update() would be misleading - the LLM would
        # describe changes but the store wouldn't actually be modified.
        # Use update_contact() or _create_contact() instead.
        if self._deterministic:
            raise RuntimeError(
                "SimulatedContactManager.update() is not available in deterministic mode. "
                "Use update_contact() or _create_contact() to modify the contact store directly.",
            )

        should_log = self._log_events or log_events
        call_id = None

        instruction = build_simulated_method_prompt(
            "update",
            text,
            parent_chat_context=_parent_chat_context,
        )
        handle = _SimulatedContactHandle(
            self._llm,
            instruction,
            mode="update",
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            response_format=response_format,
            hold_completion=self._hold_completion,
        )

        # Tool-style scheduled log (only when no parent lineage)
        maybe_tool_log_scheduled(
            "SimulatedContactManager.update",
            "update",
            {"text": text, "requests_clarification": _requests_clarification},
        )

        # No EventBus publishing for simulated managers

        return handle

    def filter_contacts(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Filter contacts from the internal store using a Python boolean expression.

        This method searches the internal contact store deterministically.
        Unlike the original LLM-backed implementation, this evaluates the
        filter expression against actual stored contacts.

        Parameters
        ----------
        filter : str | None
            A Python boolean expression evaluated per contact. Column names
            are available in scope. Examples:
            - ``"first_name == 'John'"``
            - ``"email_address.endswith('@company.com')"``
            - ``"contact_id > 5"``
            When None or "True", returns all contacts.
        offset : int
            Zero-based index of the first result.
        limit : int
            Maximum number of records to return.

        Returns
        -------
        dict
            ``{"contacts": [Contact, ...]}`` matching the filter.
        """
        sched = maybe_tool_log_scheduled(
            "SimulatedContactManager.filter_contacts",
            "filter_contacts",
            {"filter": filter, "offset": offset, "limit": limit},
        )

        # Collect matching contacts
        matching: List[Contact] = []
        for contact_dict in self._contacts.values():
            # If no filter or filter is "True", include all
            if filter is None or filter.strip().lower() == "true":
                # Convert dict to Contact - validation should always succeed
                # since we control the data we store
                matching.append(Contact.model_validate(contact_dict))
            else:
                # Evaluate the filter expression with contact fields in scope
                try:
                    # Create a safe evaluation context with contact fields
                    eval_context = dict(contact_dict)
                    # Handle None values gracefully in string operations
                    for key, val in eval_context.items():
                        if val is None:
                            eval_context[key] = ""
                    if eval(filter, {"__builtins__": {}}, eval_context):
                        matching.append(Contact.model_validate(contact_dict))
                except Exception:
                    # If eval fails (e.g., filter references non-existent field),
                    # skip this contact
                    pass

        # Sort by contact_id for deterministic ordering
        matching.sort(key=lambda c: c.contact_id)

        # Apply offset and limit
        result = matching[offset : offset + limit] if limit else matching[offset:]

        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "filter_contacts",
                {"count": len(result), "offset": offset, "limit": limit},
                t0,
            )

        return {"contacts": result}

    # ------------------------------------------------------------------ #
    #  Simulated _create_contact - deterministic with internal store     #
    # ------------------------------------------------------------------ #
    def _create_contact(
        self,
        *,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        bio: Optional[str] = None,
        timezone: Optional[str] = None,
        rolling_summary: Optional[str] = None,
        should_respond: bool = True,
        response_policy: Optional[str] = None,
        custom_fields: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "ToolOutcome":
        """
        Create a new contact in the internal store with a deterministic contact_id.

        This method does NOT use the LLM - it directly creates the contact
        in the internal store and returns immediately. The contact_id is
        assigned using a simple incrementing counter.

        Parameters match the real ContactManager._create_contact for compatibility.
        """
        # Assign a new contact_id using the counter
        contact_id = self._next_contact_id
        self._next_contact_id += 1

        # Build the contact dict
        contact: Dict[str, Any] = {
            "contact_id": contact_id,
            "first_name": first_name,
            "surname": surname,
            "email_address": email_address,
            "phone_number": phone_number,
            "bio": bio,
            "timezone": timezone,
            "rolling_summary": rolling_summary,
            "should_respond": should_respond,
            "response_policy": response_policy,
        }

        # Add any custom fields
        if custom_fields:
            contact.update(custom_fields)

        # Add any extra kwargs
        if kwargs:
            contact.update(kwargs)

        # Store in internal contacts dict
        self._contacts[contact_id] = contact

        return {
            "outcome": "contact created",
            "details": {"contact_id": contact_id},
        }

    # ------------------------------------------------------------------ #
    #  Simulated update_contact - deterministic with internal store      #
    # ------------------------------------------------------------------ #
    def update_contact(
        self,
        *,
        contact_id: int,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        whatsapp_number: Optional[str] = None,
        description: Optional[str] = None,
        bio: Optional[str] = None,
        timezone: Optional[str] = None,
        rolling_summary: Optional[str] = None,
        should_respond: Optional[bool] = None,
        response_policy: Optional[str] = None,
        custom_fields: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "ToolOutcome":
        """
        Update one or more fields of an existing contact in the internal store.

        This method does NOT use the LLM - it directly updates the contact
        in the internal store and returns immediately. If the contact doesn't
        exist, it will be created with the given contact_id.

        Parameters match the real ContactManager.update_contact for compatibility.
        """
        sched = maybe_tool_log_scheduled(
            "SimulatedContactManager.update_contact",
            "update_contact",
            {
                "contact_id": contact_id,
                "fields": sorted(
                    [
                        k
                        for k, v in {
                            "first_name": first_name,
                            "surname": surname,
                            "email_address": email_address,
                            "phone_number": phone_number,
                            "whatsapp_number": whatsapp_number,
                            "description": description,
                            "bio": bio,
                            "timezone": timezone,
                            "rolling_summary": rolling_summary,
                            "should_respond": should_respond,
                            "response_policy": response_policy,
                            **(custom_fields or {}),
                            **kwargs,
                        }.items()
                        if v is not None
                    ],
                ),
            },
        )

        # Get existing contact or create new one
        if contact_id not in self._contacts:
            # Create a new contact entry for this ID
            self._contacts[contact_id] = {"contact_id": contact_id}
            # Update counter if needed to avoid future collisions
            if contact_id >= self._next_contact_id:
                self._next_contact_id = contact_id + 1

        contact = self._contacts[contact_id]

        # Apply updates (only non-None values)
        updates = {
            k: v
            for k, v in {
                "first_name": first_name,
                "surname": surname,
                "email_address": email_address,
                "phone_number": phone_number,
                "whatsapp_number": whatsapp_number,
                "description": description,
                "bio": bio,
                "timezone": timezone,
                "rolling_summary": rolling_summary,
                "should_respond": should_respond,
                "response_policy": response_policy,
                **(custom_fields or {}),
                **kwargs,
            }.items()
            if v is not None
        }

        contact.update(updates)

        out = {
            "outcome": "contact updated",
            "details": {"contact_id": contact_id},
        }

        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(label, cid, "update_contact", out, t0)

        return out

    # ------------------------------------------------------------------ #
    #  Simulated _delete_contact - deterministic with internal store     #
    # ------------------------------------------------------------------ #
    def _delete_contact(
        self,
        *,
        contact_id: int,
    ) -> "ToolOutcome":
        """
        Delete a contact from the internal store.

        This method does NOT use the LLM - it directly removes the contact
        from the internal store and returns immediately.

        Parameters
        ----------
        contact_id : int
            The identifier of the contact to remove. Must not be a system contact.

        Returns
        -------
        ToolOutcome
            ``{"outcome": "contact deleted", "details": {"contact_id": <int>}}``.

        Raises
        ------
        RuntimeError
            If attempting to delete system contacts (0=assistant, 1=user).
        ValueError
            If the contact does not exist.
        """
        # Reject deletion of system contacts
        if contact_id in (self.ASSISTANT_CONTACT_ID, self.USER_CONTACT_ID):
            raise RuntimeError(
                f"Cannot delete system contact (contact_id={contact_id}). "
                f"System contacts (0=assistant, 1=user) are protected.",
            )

        # Check if contact exists
        if contact_id not in self._contacts:
            raise ValueError(f"Contact with id {contact_id} does not exist.")

        # Remove from internal store
        del self._contacts[contact_id]

        return {
            "outcome": "contact deleted",
            "details": {"contact_id": contact_id},
        }

    def _sync_required_contacts(self) -> None:
        """Ensure system contacts (assistant=0, user=1) exist.

        For SimulatedContactManager, this is a no-op since system contacts
        are pre-populated in __init__. This method exists for API compatibility
        with the real ContactManager.
        """
        # System contacts are already pre-populated in __init__, nothing to do

    def _merge_contacts(
        self,
        *,
        contact_id_1: int,
        contact_id_2: int,
        overrides: dict,
    ) -> "ToolOutcome":  # noqa: D401 – imperative helper
        """
        Simulate a merge through the shared stateful LLM and return a structured
        confirmation validated against ``_MergeOutcome``.
        """

        payload = {
            "contact_id_1": contact_id_1,
            "contact_id_2": contact_id_2,
            "overrides": overrides or {},
        }

        instruction = (
            "You are simulating the private helper `_merge_contacts` of a CRM. "
            "There is no real database – maintain consistency with the ongoing conversation and your prior outputs.\n\n"
            "Merge the two contacts described below using the overrides to pick winners per field. "
            "Respond ONLY with a JSON object that conforms to the provided response schema."
        )

        async def _call_llm() -> str:
            self._llm.set_response_format(_MergeOutcomeStrict)
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            label = SimulatedLineage.make_label(
                "SimulatedContactManager._merge_contacts",
            )
            return await simulated_llm_roundtrip(
                self._llm,
                label=label,
                prompt=f"{instruction}\n\n{json.dumps(payload, indent=2)}",
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                raise RuntimeError(
                    "SimulatedContactManager._merge_contacts cannot be invoked from within an active event loop.",
                )
            raw = asyncio.run(_call_llm())
        finally:
            try:
                self._llm.reset_response_format()
            except Exception:
                pass

        parsed = _MergeOutcomeStrict.model_validate_json(raw)
        data = parsed.model_dump()
        # Echo overrides back for the test while keeping strict parsing
        try:
            data.setdefault("details", {})["overrides"] = overrides or {}
        except Exception:
            pass
        return data

    @functools.wraps(BaseContactManager.clear, updated=())
    def clear(self) -> None:
        sched = maybe_tool_log_scheduled(
            "SimulatedContactManager.clear",
            "clear",
            {},
        )
        type(self).__init__(
            self,
            description=getattr(
                self,
                "_description",
                "nothing fixed, make up some imaginary scenario",
            ),
            log_events=getattr(self, "_log_events", False),
            rolling_summary_in_prompts=getattr(
                self,
                "_rolling_summary_in_prompts",
                True,
            ),
            simulation_guidance=getattr(self, "_simulation_guidance", None),
        )
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(label, cid, "clear", {"outcome": "reset"}, t0)


# --- TYPE CHECKING SUPPORT --------------------------------------------------

if TYPE_CHECKING:
    from ..common.tool_outcome import ToolOutcome
