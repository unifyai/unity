# unity/transcript_manager/simulated_transcript_manager.py
from __future__ import annotations

import asyncio
import json
import os
import threading
import functools
from typing import List, Optional, Dict, Any

import unify

from ..common.async_tool_loop import SteerableToolHandle
from .base import BaseTranscriptManager
from .prompt_builders import (
    build_ask_prompt,
    build_simulated_method_prompt,
)
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
from ..common.simulated import mirror_transcript_manager_tools
from .types.message import Message


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────
class _SimulatedTranscriptHandle(SteerableToolHandle):
    """
    A very small, LLM-backed handle used by SimulatedTranscriptManager.ask.
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

        # fire clarification immediately if queues supplied
        if self._needs_clar:
            try:
                self._clar_up_q.put_nowait(
                    "Could you clarify your information-need around the transcripts?",
                )
            except asyncio.QueueFull:
                pass

        self._extra_user_msgs: List[str] = []

        # completion primitives
        self._done = threading.Event()
        self._cancelled = False
        self._answer: Optional[str] = None
        self._msgs: List[Dict[str, Any]] = []
        self._paused = False

    # ──  API expected by SteerableToolHandle  ──────────────────────────────
    async def result(self):
        if self._cancelled:
            return "processed stopped early, no result"

        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done.is_set():
            # wait for clarification reply if requested
            if self._needs_clar:
                clar_reply = await self._clar_down_q.get()
                self._extra_user_msgs.append(f"Clarification: {clar_reply}")

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_user_msgs)
            answer = await self._llm.generate(prompt)
            self._answer = answer
            self._msgs = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            self._done.set()

        if self._want_steps:
            return self._answer, self._msgs
        return self._answer

    def interject(self, message: str) -> str:
        if self._cancelled:
            return "Interaction has been stopped."
        self._extra_user_msgs.append(message)
        return "Acknowledged."

    def stop(self, reason: Optional[str] = None) -> str:
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

    def done(self) -> bool:
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
            + self._extra_user_msgs
            + [f"Question to answer (as a reminder!): {question}"],
        )

        return _SimulatedTranscriptHandle(
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
# Public Simulated Manager
# ─────────────────────────────────────────────────────────────────────────────
class SimulatedTranscriptManager(BaseTranscriptManager):
    """
    Lightweight, fake implementation of TranscriptManager that only uses an
    LLM to invent plausible answers.  Suitable for offline demos and tests
    where the real storage layer is unnecessary.
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
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance

        # Shared, *stateful* **asynchronous** LLM
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        # Use shared helper to mirror the real TranscriptManager's tools
        tools_for_prompt = mirror_transcript_manager_tools()
        # Provide placeholder counts/columns for the simulated environment
        fake_columns = [{k: str(v.annotation)} for k, v in Message.model_fields.items()]
        # Include sender contact columns for clarity
        from ..contact_manager.types.contact import Contact as _Contact

        fake_contact_columns = [
            {k: str(v.annotation)} for k, v in _Contact.model_fields.items()
        ]
        ask_sys = build_ask_prompt(
            tools_for_prompt,
            num_messages=10,
            transcript_columns=fake_columns,
            contact_columns=fake_contact_columns,
            include_activity=self._rolling_summary_in_prompts,
        )

        self._llm.set_system_message(
            "You are a *simulated* transcript assistant. "
            "There is **no** backing datastore – create plausible yet "
            "self-consistent answers.\n\n"
            "For reference, here are the *real* system messages used by the "
            "production implementation:\n"
            f"\n\n'ask' system message:\n{ask_sys}\n\n"
            f"Back-story: {self._description}",
        )

    # --------------------------------------------------------------------- #
    # ask                                                                   #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseTranscriptManager.ask, updated=())
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
                "TranscriptManager",
                "ask",
                phase="incoming",
                question=text,
            )

        instruction = build_simulated_method_prompt(
            "ask",
            text,
            parent_chat_context=parent_chat_context,
        )
        handle = _SimulatedTranscriptHandle(
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
                "TranscriptManager",
                "ask",
            )

        return handle

    @functools.wraps(BaseTranscriptManager.clear, updated=())
    def clear(self) -> None:
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
