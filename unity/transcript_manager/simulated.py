# unity/transcript_manager/simulated_transcript_manager.py
from __future__ import annotations

import asyncio
import json
import os
import threading
import functools
from typing import List, Optional, Union, Dict, Any

import unify

from ..common.llm_helpers import SteerableToolHandle
from .base import BaseTranscriptManager
from .prompt_builders import build_ask_prompt, build_summarize_prompt


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
            raise asyncio.CancelledError()

        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done.is_set():
            # wait for clarification reply if requested
            if self._needs_clar:
                clar_reply = await self._clar_down_q.get()
                self._extra_user_msgs.append(f"Clarification: {clar_reply}")

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_user_msgs)
            answer = await asyncio.to_thread(self._llm.generate, prompt)
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

    def done(self) -> bool:
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
    ) -> None:
        self._description = description

        # one shared, *stateful* LLM instance – preserves chat history
        self._llm = unify.Unify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        ask_sys = build_ask_prompt({})
        sum_sys = build_summarize_prompt()

        self._llm.set_system_message(
            "You are a *simulated* transcript assistant. "
            "There is **no** backing datastore – create plausible yet "
            "self-consistent answers.\n\n"
            "For reference, here are the *real* system messages used by the "
            "production implementation:\n"
            f"\n\n'ask' system message:\n{ask_sys}\n\n"
            f"\n\n'summarize' system message:\n{sum_sys}\n\n"
            f"Back-story: {self._description}",
        )

    # --------------------------------------------------------------------- #
    # ask                                                                   #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseTranscriptManager.ask, updated=())
    def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        instruction = (
            "On this turn you are simulating the 'ask' method.\n"
            f"The user question is:\n{text}"
        )
        if parent_chat_context:
            instruction += (
                f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}"
            )
        return _SimulatedTranscriptHandle(
            self._llm,
            text,
            _return_reasoning_steps=_return_reasoning_steps,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    # --------------------------------------------------------------------- #
    # summarize                                                             #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseTranscriptManager.summarize, updated=())
    async def summarize(
        self,
        *,
        exchange_ids: Union[int, List[int]],
        guidance: Optional[str] = None,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> str:
        instruction = "On this turn you are simulating the 'summarize' method.\n"
        if parent_chat_context:
            instruction += (
                f"Calling chat context:\n{json.dumps(parent_chat_context, indent=4)}"
            )
        if not isinstance(exchange_ids, list):
            exchange_ids = [exchange_ids]

        # Clarification flow if required
        if clarification_up_q is not None and clarification_down_q is not None:
            try:
                clarification_up_q.put_nowait(
                    "Any special focus for this summary?",
                )
            except asyncio.QueueFull:
                pass
            try:
                clar = await asyncio.wait_for(clarification_down_q.get(), timeout=30)
            except asyncio.TimeoutError:
                clar = None
        else:
            clar = None

        prompt_parts = [instruction] + [
            f"\nSummarise imaginary exchange(s) with id(s): {exchange_ids}.",
        ]
        if guidance:
            prompt_parts.append(f"Guidance: {guidance}")
        if clar:
            prompt_parts.append(f"User clarification: {clar}")

        summary = await asyncio.to_thread(
            self._llm.generate,
            "\n\n".join(prompt_parts),
        )
        return summary
