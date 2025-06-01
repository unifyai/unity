# unity/transcript_manager/simulated_transcript_manager.py
from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import List, Optional, Union, Dict, Any

import unify

from ..common.llm_helpers import SteerableToolHandle


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
        return_reasoning_steps: bool,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
    ):
        self._llm = llm
        self._initial = initial_text
        self._want_steps = return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        self._needs_clar = self._clar_up_q is not None and self._clar_down_q is not None

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

    # ──  API expected by SteerableToolHandle  ──────────────────────────────
    async def result(self):
        if self._cancelled:
            raise asyncio.CancelledError()

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

    # property expected by orchestrator
    def done(self) -> bool:  # type: ignore[override]
        return self._done.is_set()

    @property
    def valid_tools(self):
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Public Simulated Manager
# ─────────────────────────────────────────────────────────────────────────────
class SimulatedTranscriptManager:
    """
    Lightweight, fake implementation of TranscriptManager that only uses an
    LLM to invent plausible answers.  Suitable for offline demos and tests
    where the real storage layer is unnecessary.
    """

    def __init__(
        self,
        description: str = "Imaginary multi-channel transcript database.",
    ) -> None:
        self._description = description

    # --------------------------------------------------------------------- #
    # ask                                                                   #
    # --------------------------------------------------------------------- #
    def ask(
        self,
        text: str,
        *,
        return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        sys_msg = (
            "You are a *simulated* transcript assistant. "
            "There is NO real database – feel free to fabricate convincing yet "
            "consistent answers about emails, chats, calls, etc.\n\n"
            f"Back-story: {self._description}"
        )
        llm = unify.Unify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        llm.set_system_message(sys_msg)

        return _SimulatedTranscriptHandle(
            llm,
            text,
            return_reasoning_steps=return_reasoning_steps,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    # --------------------------------------------------------------------- #
    # summarize                                                             #
    # --------------------------------------------------------------------- #
    async def summarize(
        self,
        *,
        exchange_ids: Union[int, List[int]],
        guidance: Optional[str] = None,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> str:
        """
        Synthetically summarise the given exchange IDs.  All content is
        invented; we simply echo back a plausible summary paragraph.
        """
        if not isinstance(exchange_ids, list):
            exchange_ids = [exchange_ids]

        sys_msg = (
            "You are a *simulated* summariser of message exchanges. "
            "No real raw messages exist – just invent a plausible, concise "
            "summary that a human would find believable."
        )
        llm = unify.Unify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        llm.set_system_message(sys_msg)

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

        prompt_parts = [
            f"Summarise imaginary exchange(s) with id(s): {exchange_ids}.",
        ]
        if guidance:
            prompt_parts.append(f"Guidance: {guidance}")
        if clar:
            prompt_parts.append(f"User clarification: {clar}")

        summary = await asyncio.to_thread(
            llm.generate,
            "\n\n".join(prompt_parts),
        )
        return summary
