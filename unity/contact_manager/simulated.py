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
from ..constants import LOGGER

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
class _SimulatedContactHandle(SteerableToolHandle, SimulatedHandleMixin):
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

        if self._needs_clar:
            try:
                q_text = "Could you clarify your request about contacts?"
                self._clar_up_q.put_nowait(q_text)
                try:
                    SimulatedLog.log_clarification_request(self._log_label, q_text)
                except Exception:
                    pass
                try:
                    LOGGER.info(f"❓ [{self._log_label}] Clarification requested")
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
                        f"⏳ [{self._log_label}] Waiting for clarification answer…",
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
                    LOGGER.info(f"💬 [{self._log_label}] Clarification answer received")
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
            self._done.set()

        # If cancellation happened after the coroutine started, return a stable post-cancel value.
        if self._cancelled:
            return "processed stopped early, no result"
        if self._want_steps:
            return self._answer, self._messages
        return self._answer

    def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> str:
        """Interject a message into the in-flight handle.

        Args:
            message: The interjection message to inject.
            parent_chat_context_cont: Optional continuation of parent chat context.
                Accepted for API parity with real handles but not currently used.
            images: Optional image references. Accepted for API parity with real handles
                but not currently used.
        """
        if self._cancelled:
            return "Interaction stopped."
        self._log_interject(message)
        self._extra_msgs.append(message)
        return "Acknowledged."

    def stop(
        self,
        reason: str | None = None,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> str:
        """Stop the in-flight handle.

        Args:
            reason: Optional reason for stopping.
            parent_chat_context_cont: Optional continuation of parent chat context.
                Accepted for API parity with real handles but not currently used.
        """
        self._log_stop(reason)
        self._cancelled = True
        try:
            self._cancel_event.set()
        except Exception:
            pass
        self._done.set()
        return "Stopped." if reason is None else f"Stopped: {reason}"

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
        return self._done.is_set()

    async def ask(
        self,
        question: str,
        *,
        parent_chat_context: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> "SteerableToolHandle":
        """Ask a follow-up question about the current operation.

        Args:
            question: The question to ask.
            parent_chat_context: Optional parent chat context for the inspection loop.
                Accepted for API parity with real handles but not currently used.
            images: Optional image references. Accepted for API parity with real handles
                but not currently used.
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
        """Retrieve the next clarification request, if any.

        Only surfaces clarification events when this handle explicitly requested
        clarification. This prevents cross-handle consumption of shared clarification
        queues that may be injected by external processes.
        """
        if not getattr(self, "_needs_clar", False):
            return {}
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
        return {}

    async def next_notification(self) -> dict:
        return {}

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
    """

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
        *,
        log_events: bool = False,
        rolling_summary_in_prompts: bool = True,
        simulation_guidance: Optional[str] = None,
        # Accept but ignore extra parameters for compatibility
        **kwargs: Any,
    ) -> None:
        self._description = description
        self._log_events = log_events
        self._simulation_guidance = simulation_guidance

        # Shared, *stateful* **asynchronous** LLM
        from ..common.llm_client import (
            new_llm_client as _new_llm_client,
        )  # local import to avoid cycles

        self._llm = _new_llm_client(stateful=True)
        # Mirror the real manager's tool exposure programmatically
        # and build the *exact* same prompts via the shared builders.
        ask_tools = mirror_contact_manager_tools("ask")
        upd_tools = mirror_contact_manager_tools("update")
        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        ask_msg = build_ask_prompt(
            ask_tools,
            10,
            [{k: str(v.annotation)} for k, v in Contact.model_fields.items()],
            include_activity=self._rolling_summary_in_prompts,
        )
        upd_msg = build_update_prompt(
            upd_tools,
            10,
            [{k: str(v.annotation)} for k, v in Contact.model_fields.items()],
            include_activity=self._rolling_summary_in_prompts,
        )

        self._llm.set_system_message(
            "You are a *simulated* contact-manager assistant. "
            "There is no real database; invent plausible contact records and "
            "keep your story consistent across turns.\n\n"
            "As a reference, the system messages for the *real* contact-manager 'ask' and 'update' methods are as follows."
            "You do not have access to any real tools, so you should just create a final answer to the question/request . "
            f"\n\n'ask' system message:\n{ask_msg}\n\n"
            f"\n\n'update' system message:\n{upd_msg}\n\n"
            f"Back-story: {self._description}",
        )

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
        images: object | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

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
        images: object | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
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
    ) -> List[Contact]:
        """
        Simulated variant of :pyfunc:`ContactManager.filter_contacts`.

        The same stateful LLM used by the public methods generates a JSON
        payload that strictly conforms to ``_ContactsListResponse`` via
        ``response_format`` enforcement, ensuring scenario consistency.
        """

        sched = maybe_tool_log_scheduled(
            "SimulatedContactManager.filter_contacts",
            "filter_contacts",
            {"filter": filter, "offset": offset, "limit": limit},
        )

        schema_json = json.dumps(Contact.model_json_schema(), indent=2)
        filter_clause = (
            f"Filter expression: `{filter}`."
            if filter
            else "No filter (return any plausible contacts)."
        )

        prompt = (
            "You are simulating the private helper `filter_contacts` of a CRM. "
            "There is no real database – maintain consistency with the ongoing conversation and your prior outputs.\n\n"
            f"{filter_clause} Offset: {offset}. Limit: {limit}.\n\n"
            "Respond ONLY with a JSON object that conforms to the provided response schema.\n\n"
            f"Contact JSON schema (for each item in `contacts`):\n{schema_json}"
        )

        # Use unified simulated roundtrip so standard logs ("LLM simulating…") are emitted
        try:
            sys_msg = getattr(self._llm, "system_message", None)
        except Exception:
            sys_msg = None
        # Prefer the scheduled label when available so logs correlate
        label = (
            sched[0]
            if sched is not None and isinstance(sched, tuple) and len(sched) >= 1
            else SimulatedLineage.make_label("SimulatedContactManager.filter_contacts")
        )

        async def _call_llm() -> str:
            self._llm.set_response_format(_ContactsListResponse)
            return await simulated_llm_roundtrip(
                self._llm,
                label=label,
                prompt=prompt,
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                raise RuntimeError(
                    "SimulatedContactManager.filter_contacts cannot be invoked from within an active event loop.",
                )
            raw = asyncio.run(_call_llm())
        finally:
            # Best-effort reset; ignore if unsupported on the client
            try:
                self._llm.reset_response_format()
            except Exception:
                pass

        # Validate using the Pydantic model (enforced shape)
        model = _ContactsListResponse.model_validate_json(raw)
        # _ContactRecord is a subclass of Contact, so callers can treat these
        # as Contact instances. Apply slicing locally as a safety net.
        contacts = (
            model.contacts[offset : offset + limit]
            if limit is not None
            else model.contacts[offset:]
        )
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "filter_contacts",
                {"count": len(contacts), "offset": offset, "limit": limit},
                t0,
            )
        return contacts

    # ------------------------------------------------------------------ #
    #  Simulated _create_contact                                         #
    # ------------------------------------------------------------------ #
    def _create_contact(
        self,
        *,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        bio: Optional[str] = None,
        rolling_summary: Optional[str] = None,
        should_respond: bool = True,
        response_policy: Optional[str] = None,
        custom_fields: Optional[Dict[str, Any]] = None,
    ) -> "ToolOutcome":
        """
        Simulated variant of :pyfunc:`ContactManager._create_contact` with strict
        structured output enforced via ``response_format``. The shared stateful
        LLM generates a JSON payload that we validate against ``_CreateOutcome``.
        """

        # Only include fields that are actually being provided
        payload_fields: Dict[str, Any] = {
            k: v
            for k, v in {
                "first_name": first_name,
                "surname": surname,
                "email_address": email_address,
                "phone_number": phone_number,
                "bio": bio,
                "rolling_summary": rolling_summary,
                "should_respond": should_respond,
                "response_policy": response_policy,
                **(custom_fields or {}),
            }.items()
            if v is not None and v != {}
        }

        # Always include should_respond explicitly (boolean) for clarity
        if "should_respond" not in payload_fields:
            payload_fields["should_respond"] = should_respond

        instruction = (
            "You are simulating the private helper `_create_contact` of a CRM. "
            "There is no real database – maintain consistency with the ongoing conversation and your prior outputs.\n\n"
            "Create a new contact using the provided fields. "
            "Respond ONLY with a JSON object that conforms to the provided response schema."
        )

        user_payload = json.dumps({"contact": payload_fields}, indent=2)

        async def _call_llm() -> str:
            self._llm.set_response_format(_CreateOutcome)
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            label = SimulatedLineage.make_label(
                "SimulatedContactManager._create_contact",
            )
            return await simulated_llm_roundtrip(
                self._llm,
                label=label,
                prompt=f"{instruction}\n\n{user_payload}",
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                raise RuntimeError(
                    "SimulatedContactManager._create_contact cannot be invoked from within an active event loop.",
                )
            raw = asyncio.run(_call_llm())
        finally:
            try:
                self._llm.reset_response_format()
            except Exception:
                pass

        model = _CreateOutcome.model_validate_json(raw)
        return model.model_dump()

    # ------------------------------------------------------------------ #
    #  Simulated update_contact                                          #
    # ------------------------------------------------------------------ #
    def update_contact(
        self,
        *,
        contact_id: int,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        description: Optional[str] = None,
        bio: Optional[str] = None,
        rolling_summary: Optional[str] = None,
        custom_fields: Optional[Dict[str, Any]] = None,
    ) -> "ToolOutcome":
        """
        Simulated variant of :pyfunc:`ContactManager.update_contact` with strict
        structured output enforced via ``response_format``. The shared stateful
        LLM generates a JSON payload that we validate against ``_UpdateOutcome``.
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
                            "description": description,
                            "bio": bio,
                            "rolling_summary": rolling_summary,
                            **(custom_fields or {}),
                        }.items()
                        if v is not None
                    ],
                ),
            },
        )

        # Only include fields that are actually being modified
        updates = {
            k: v
            for k, v in {
                "first_name": first_name,
                "surname": surname,
                "email_address": email_address,
                "phone_number": phone_number,
                "description": description,
                "bio": bio,
                "rolling_summary": rolling_summary,
                **(custom_fields or {}),
            }.items()
            if v is not None
        }

        if not updates:
            updates = {"note": "no-op update requested – acknowledge anyway"}

        instruction = (
            "You are simulating the private helper `update_contact` of a CRM. "
            "There is no real database – maintain consistency with the ongoing conversation and your prior outputs.\n\n"
            f"Update the contact with id {contact_id} using the following fields (treat them as the diff to apply).\n"
            "Respond ONLY with a JSON object that conforms to the provided response schema."
        )

        user_payload = json.dumps(
            {"contact_id": contact_id, "updates": updates},
            indent=2,
        )

        # Use unified simulated roundtrip so standard logs ("LLM simulating…") are emitted
        try:
            sys_msg = getattr(self._llm, "system_message", None)
        except Exception:
            sys_msg = None
        label = (
            sched[0]
            if sched is not None and isinstance(sched, tuple) and len(sched) >= 1
            else SimulatedLineage.make_label("SimulatedContactManager.update_contact")
        )

        async def _call_llm() -> str:
            self._llm.set_response_format(_UpdateOutcome)
            return await simulated_llm_roundtrip(
                self._llm,
                label=label,
                prompt=f"{instruction}\n\n{user_payload}",
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                raise RuntimeError(
                    "SimulatedContactManager.update_contact cannot be invoked from within an active event loop.",
                )
            raw = asyncio.run(_call_llm())
        finally:
            try:
                self._llm.reset_response_format()
            except Exception:
                pass

        model = _UpdateOutcome.model_validate_json(raw)
        out = model.model_dump()
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(label, cid, "update_contact", out, t0)
        return out

    # ------------------------------------------------------------------ #
    #  Simulated _delete_contact                                          #
    # ------------------------------------------------------------------ #
    def _delete_contact(
        self,
        *,
        contact_id: int,
    ) -> "ToolOutcome":
        """
        Simulate deletion through the shared stateful LLM and enforce a structured
        confirmation via ``response_format``.
        """

        instruction = (
            "You are simulating the private helper `_delete_contact` of a CRM. "
            "There is no real database – maintain consistency with the ongoing conversation and your prior outputs.\n\n"
            f"Delete the contact with id {contact_id}. "
            "Respond ONLY with a JSON object that conforms to the provided response schema."
        )

        async def _call_llm() -> str:
            self._llm.set_response_format(_DeleteOutcome)
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            label = SimulatedLineage.make_label(
                "SimulatedContactManager._delete_contact",
            )
            return await simulated_llm_roundtrip(
                self._llm,
                label=label,
                prompt=instruction,
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                raise RuntimeError(
                    "SimulatedContactManager._delete_contact cannot be invoked from within an active event loop.",
                )
            raw = asyncio.run(_call_llm())
        finally:
            try:
                self._llm.reset_response_format()
            except Exception:
                pass

        model = _DeleteOutcome.model_validate_json(raw)
        return model.model_dump()

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
