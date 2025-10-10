# unity/contact_manager/simulated_contact_manager.py
from __future__ import annotations

import asyncio
import json
import os
import functools
import threading
from typing import List, Dict, Any, Optional, TYPE_CHECKING

import unify
from pydantic import BaseModel, Field
from .base import BaseContactManager
from .types.contact import Contact
from .prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
    build_simulated_method_prompt,
)
from ..common.async_tool_loop import SteerableToolHandle
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
from ..common.simulated import mirror_contact_manager_tools


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
class _SimulatedContactHandle(SteerableToolHandle):
    """
    Minimal LLM-backed handle used by SimulatedContactManager.ask / update.
    """

    def __init__(
        self,
        llm: unify.Unify,
        initial_text: str,
        *,
        _return_reasoning_steps: bool,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
    ):
        self._llm = llm
        self._initial = initial_text
        self._want_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        if _requests_clarification and (
            not clarification_up_q or not clarification_down_q
        ):
            raise ValueError(
                "Clarification queues must be provided when _requests_clarification is True",
            )
        self._needs_clar = _requests_clarification

        if self._needs_clar:
            try:
                self._clar_up_q.put_nowait(
                    "Could you clarify your request about contacts?",
                )
            except asyncio.QueueFull:
                pass

        self._extra_msgs: List[str] = []

        self._done = threading.Event()
        self._cancelled = False
        self._answer: str | None = None
        self._messages: List[Dict[str, Any]] = []
        self._paused = False

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
                clar = await self._clar_down_q.get()
                self._extra_msgs.append(f"Clarification: {clar}")

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_msgs)

            answer = await self._llm.generate(prompt)
            self._answer = answer
            self._messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            self._done.set()

        if self._want_steps:
            return self._answer, self._messages
        return self._answer

    def interject(self, message: str) -> str:
        if self._cancelled:
            return "Interaction stopped."
        self._extra_msgs.append(message)
        return "Acknowledged."

    def stop(self, reason: str | None = None) -> str:
        self._cancelled = True
        self._done.set()
        return "Stopped." if reason is None else f"Stopped: {reason}"

    def pause(self) -> str:
        if self._paused:
            return "Already paused."
        self._paused = True
        return "Paused."

    def resume(self) -> str:
        if not self._paused:
            return "Already running."
        self._paused = False
        return "Resumed."

    def done(self) -> bool:  # type: ignore[override]
        return self._done.is_set()

    async def ask(self, question: str) -> "SteerableToolHandle":
        q_msg = (
            f"Your only task is to simulate an answer to the following question: {question}\n\n"
            "However, there is a also ongoing simulated process which had the instructions given below. "
            "Please make your answer realastic and conceivable given the provided context of the simulated taks."
        )
        follow_up_prompt = "\n\n---\n\n".join(
            [q_msg]
            + [self._initial]
            + self._extra_msgs
            + [f"Question to answer (as a reminder!): {question}"],
        )
        return _SimulatedContactHandle(
            self._llm,
            follow_up_prompt,
            _return_reasoning_steps=self._want_steps,
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )

    # --- event APIs required by SteerableToolHandle ---------------------
    async def next_clarification(self) -> dict:
        try:
            if self._clar_up_q is not None:
                msg = await self._clar_up_q.get()
                return {"message": msg}
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
    ) -> None:
        self._description = description
        self._log_events = log_events
        self._simulation_guidance = simulation_guidance

        # Shared, *stateful* **asynchronous** LLM
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
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

    # --------------------------------------------------------------------- #
    # ask                                                                   #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseContactManager.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "ContactManager",
                "ask",
                phase="incoming",
                question=text,
            )

        instruction = build_simulated_method_prompt(
            "ask",
            text,
            parent_chat_context=parent_chat_context,
        )
        handle = _SimulatedContactHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "ContactManager",
                "ask",
            )

        return handle

    # Append guidance to influence outer orchestrators via tool descriptions
    ask.__doc__ = (ask.__doc__ or "") + (
        "\n\nOuter-orchestrator guidance: Avoid invoking this tool repeatedly with the same "
        "arguments within the same conversation. Prefer reusing prior results and "
        "compose the final answer once sufficient information has been gathered."
    )

    # --------------------------------------------------------------------- #
    # update                                                                #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseContactManager.update, updated=())
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "ContactManager",
                "update",
                phase="incoming",
                request=text,
            )

        instruction = build_simulated_method_prompt(
            "update",
            text,
            parent_chat_context=parent_chat_context,
        )
        handle = _SimulatedContactHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "ContactManager",
                "update",
            )

        return handle

    # Provide guidance for outer orchestrators via tool description on mutation methods
    update.__doc__ = (update.__doc__ or "") + (
        "\n\nOuter-orchestrator guidance: Avoid invoking this mutation with the same arguments multiple times in the same "
        "conversation. Treat this operation as idempotent; if confirmation is needed, perform a single read to verify the outcome."
    )

    def _filter_contacts(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Contact]:
        """
        Simulated variant of :pyfunc:`ContactManager._filter_contacts`.

        The same stateful LLM used by the public methods generates a JSON
        payload that strictly conforms to ``_ContactsListResponse`` via
        ``response_format`` enforcement, ensuring scenario consistency.
        """

        schema_json = json.dumps(Contact.model_json_schema(), indent=2)
        filter_clause = (
            f"Filter expression: `{filter}`."
            if filter
            else "No filter (return any plausible contacts)."
        )

        prompt = (
            "You are simulating the private helper `_filter_contacts` of a CRM. "
            "There is no real database – maintain consistency with the ongoing conversation and your prior outputs.\n\n"
            f"{filter_clause} Offset: {offset}. Limit: {limit}.\n\n"
            "Respond ONLY with a JSON object that conforms to the provided response schema.\n\n"
            f"Contact JSON schema (for each item in `contacts`):\n{schema_json}"
        )

        async def _call_llm() -> str:
            self._llm.set_response_format(_ContactsListResponse)
            return await self._llm.generate(prompt)

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                raise RuntimeError(
                    "SimulatedContactManager._filter_contacts cannot be invoked from within an active event loop.",
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
        return (
            model.contacts[offset : offset + limit]
            if limit is not None
            else model.contacts[offset:]
        )

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
        whatsapp_number: Optional[str] = None,
        bio: Optional[str] = None,
        rolling_summary: Optional[str] = None,
        respond_to: bool = False,
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
                "whatsapp_number": whatsapp_number,
                "bio": bio,
                "rolling_summary": rolling_summary,
                "respond_to": respond_to,
                "response_policy": response_policy,
                **(custom_fields or {}),
            }.items()
            if v is not None and v != {}
        }

        # Always include respond_to explicitly (boolean) for clarity
        if "respond_to" not in payload_fields:
            payload_fields["respond_to"] = respond_to

        instruction = (
            "You are simulating the private helper `_create_contact` of a CRM. "
            "There is no real database – maintain consistency with the ongoing conversation and your prior outputs.\n\n"
            "Create a new contact using the provided fields. "
            "Respond ONLY with a JSON object that conforms to the provided response schema."
        )

        user_payload = json.dumps({"contact": payload_fields}, indent=2)

        async def _call_llm() -> str:
            self._llm.set_response_format(_CreateOutcome)
            return await self._llm.generate(f"{instruction}\n\n{user_payload}")

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
    #  Simulated _update_contact                                          #
    # ------------------------------------------------------------------ #
    def _update_contact(
        self,
        *,
        contact_id: int,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        whatsapp_number: Optional[str] = None,
        bio: Optional[str] = None,
        rolling_summary: Optional[str] = None,
        respond_to: Optional[bool] = None,
        response_policy: Optional[str] = None,
        custom_fields: Optional[Dict[str, Any]] = None,
    ) -> "ToolOutcome":
        """
        Simulated variant of :pyfunc:`ContactManager._update_contact` with strict
        structured output enforced via ``response_format``. The shared stateful
        LLM generates a JSON payload that we validate against ``_UpdateOutcome``.
        """

        # Only include fields that are actually being modified
        updates = {
            k: v
            for k, v in {
                "first_name": first_name,
                "surname": surname,
                "email_address": email_address,
                "phone_number": phone_number,
                "whatsapp_number": whatsapp_number,
                "bio": bio,
                "rolling_summary": rolling_summary,
                "respond_to": respond_to,
                "response_policy": response_policy,
                **(custom_fields or {}),
            }.items()
            if v is not None
        }

        if not updates:
            updates = {"note": "no-op update requested – acknowledge anyway"}

        instruction = (
            "You are simulating the private helper `_update_contact` of a CRM. "
            "There is no real database – maintain consistency with the ongoing conversation and your prior outputs.\n\n"
            f"Update the contact with id {contact_id} using the following fields (treat them as the diff to apply).\n"
            "Respond ONLY with a JSON object that conforms to the provided response schema."
        )

        user_payload = json.dumps(
            {"contact_id": contact_id, "updates": updates},
            indent=2,
        )

        async def _call_llm() -> str:
            self._llm.set_response_format(_UpdateOutcome)
            return await self._llm.generate(f"{instruction}\n\n{user_payload}")

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                raise RuntimeError(
                    "SimulatedContactManager._update_contact cannot be invoked from within an active event loop.",
                )
            raw = asyncio.run(_call_llm())
        finally:
            try:
                self._llm.reset_response_format()
            except Exception:
                pass

        model = _UpdateOutcome.model_validate_json(raw)
        return model.model_dump()

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
            return await self._llm.generate(instruction)

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
            return await self._llm.generate(
                f"{instruction}\n\n{json.dumps(payload, indent=2)}",
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

    def clear(self) -> None:
        """
        Reset simulated state for contacts.

        There is no persistent backend context to delete. The simplest and
        most reliable way to reset the simulated manager is to re-run its
        constructor in-place, which creates a fresh stateful LLM and rebuilds
        tool exposure and prompts.
        """
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


# --- TYPE CHECKING SUPPORT --------------------------------------------------

if TYPE_CHECKING:
    from ..common.tool_outcome import ToolOutcome
