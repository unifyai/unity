# unity/contact_manager/simulated_contact_manager.py
from __future__ import annotations

import asyncio
import json
import os
import functools
import threading
from typing import List, Dict, Any, Optional, TYPE_CHECKING

import unify
from .base import BaseContactManager
from .types.contact import Contact
from .contact_manager import ContactManager
from .prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
    build_simulated_method_prompt,
)
from ..common.llm_helpers import SteerableToolHandle, methods_to_tool_dict
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)


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
            raise asyncio.CancelledError()

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

    def stop(self) -> str:
        self._cancelled = True
        self._done.set()
        return "Stopped."

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

    @property
    def valid_tools(self):
        tools = {
            self.interject.__name__: self.interject,
            self.stop.__name__: self.stop,
        }
        if self._paused:
            tools[self.resume.__name__] = self.resume
        else:
            tools[self.pause.__name__] = self.pause
        return tools

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
    ) -> None:
        self._description = description
        self._log_events = log_events

        # Shared, *stateful* **asynchronous** LLM
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        # Re-create the same tool-dicts the real manager uses, then
        # build the *exact* same prompts via the shared builders.
        ask_tools = methods_to_tool_dict(
            ContactManager._search_contacts,
            include_class_name=False,
        )
        upd_tools = methods_to_tool_dict(
            ContactManager._create_contact,
            ContactManager._update_contact,
            ContactManager._delete_contact,
            ContactManager._search_contacts,
            include_class_name=False,
        )
        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        ask_msg = build_ask_prompt(
            ask_tools,
            10,
            [{k: str(v.annotation)} for k, v in Contact.model_fields.items()],
            include_activity=self._rolling_summary_in_prompts,
        )
        upd_msg = build_update_prompt(
            upd_tools,
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

    def _search_contacts(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Contact]:
        """
        Simulated variant of :pyfunc:`ContactManager._search_contacts`.

        Delegates the heavy lifting to the *stateful* LLM backing this
        simulated manager.  We instruct the model to respond **only** with a
        JSON array of contact records that match the requested *filter* so the
        result can be parsed straight into :class:`Contact` objects.

        The method guarantees a *non-empty* return value to satisfy downstream
        components (e.g. TranscriptManager) that expect at least one contact.
        """
        # Craft an instruction that re-uses the manager's description so the
        # LLM keeps its narrative consistent across turns.
        import json

        schema_json = json.dumps(Contact.model_json_schema(), indent=2)
        filter_clause = f"Filter: `{filter}`." if filter else "No filter."

        prompt = (
            "The user has called _search_contacts with the following arguments. Please simulate the response. "
            f"{filter_clause} Return ONLY a JSON array (no markdown) of up to {limit} contacts starting at index {offset}.\n\n"
            f"Here is the Contact JSON schema for reference:\n{schema_json}"
        )

        # Because this helper is synchronous while the underlying LLM is async,
        # we spin up a temporary event-loop when needed.
        async def _call_llm() -> str:
            return await self._llm.generate(prompt)

        try:
            # Attempt to use an existing running loop if present (rare for sync callers)
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop and loop.is_running():
                # Synchronous function called from an active event loop: this is a misuse
                # of the simulated contact manager – surface a clear error.
                raise RuntimeError(
                    "SimulatedContactManager._search_contacts cannot be invoked from within an active event loop.",
                )
            else:
                raw = asyncio.run(_call_llm())

            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("Model did not return a JSON array.")
        except Exception as exc:
            # Propagate parsing / generation errors to the caller – no silent fallbacks.
            raise exc

        # Apply offset/limit and convert to Contact objects
        sliced = data[offset : offset + limit] if limit is not None else data[offset:]
        return [Contact(**c) for c in sliced]

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
        custom_fields: Optional[Dict[str, Any]] = None,
    ) -> "ToolOutcome":
        """
        Simulated variant of :pyfunc:`ContactManager._update_contact`.

        The method formulates a short instruction to the **stateful** LLM
        backing this simulated manager asking it to *pretend* that the given
        contact has been updated.  The LLM must respond with a JSON object
        matching the :class:`~unity.common.tool_outcome.ToolOutcome` schema so
        downstream callers can parse the result.
        """

        import json

        # Build a concise instruction that lists only the fields that are actually being modified
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
                **(custom_fields or {}),
            }.items()
            if v is not None
        }

        if not updates:
            updates = {"note": "no-op update requested – acknowledge anyway"}

        prompt = (
            "You are simulating the private helper `_update_contact` of a CRM. "
            f"Pretend that the contact with id {contact_id} has just been updated with the fields below. "
            "Reply with a JSON object **without** markdown that contains keys 'outcome' and 'details'. "
            "The 'details' object must include the 'contact_id' and the changed fields."
        )

        user_payload = json.dumps(
            {"contact_id": contact_id, "updates": updates},
            indent=2,
        )

        async def _call_llm() -> str:
            return await self._llm.generate(f"{prompt}\n\n{user_payload}")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop and loop.is_running():
                raise RuntimeError(
                    "SimulatedContactManager._update_contact cannot be invoked from within an active event loop.",
                )
            else:
                raw = asyncio.run(_call_llm())

            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Model did not return a JSON object.")
            if "outcome" not in data or "details" not in data:
                raise ValueError("Returned JSON missing required keys.")
        except Exception:
            data = {
                "outcome": "contact updated (simulated)",
                "details": {"contact_id": contact_id, **updates},
            }

        return data

    # ------------------------------------------------------------------ #
    #  Simulated _delete_contact                                          #
    # ------------------------------------------------------------------ #
    def _delete_contact(
        self,
        *,
        contact_id: int,
    ) -> "ToolOutcome":
        """Simulate deletion of a contact and return a confirmation payload."""
        # Compose a confirmation JSON manually – no actual storage.
        return {
            "outcome": "contact deleted (simulated)",
            "details": {"contact_id": contact_id},
        }


# --- TYPE CHECKING SUPPORT --------------------------------------------------

if TYPE_CHECKING:
    from ..common.tool_outcome import ToolOutcome
