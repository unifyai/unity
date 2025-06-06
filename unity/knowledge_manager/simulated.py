from __future__ import annotations

import asyncio
import json
import os
import functools
import threading
from typing import List, Dict, Any

import unify
from .base import BaseKnowledgeManager
from ..common.llm_helpers import SteerableToolHandle
from .sys_msgs import STORE, RETRIEVE


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────
class _SimulatedKnowledgeHandle(SteerableToolHandle):
    """
    Handle returned by SimulatedKnowledgeManager.store / retrieve.
    """

    def __init__(
        self,
        llm: unify.Unify,
        initial_text: str,
        *,
        _return_reasoning_steps: bool,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
    ):
        self._llm = llm
        self._initial = initial_text
        self._want_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        self._needs_clar = self._clar_up_q is not None and self._clar_down_q is not None

        # fire clarification question immediately if queues supplied
        if self._needs_clar:
            try:
                self._clar_up_q.put_nowait(
                    "Could you clarify your knowledge request?",
                )
            except asyncio.QueueFull:
                pass

        self._extra_msgs: List[str] = []

        self._done_event = threading.Event()
        self._cancelled = False
        self._answer: str | None = None
        self._messages: List[Dict[str, Any]] = []
        self._paused = False

    # --------------------------------------------------------------------- #
    # SteerableToolHandle API
    # --------------------------------------------------------------------- #
    async def result(self):
        if self._cancelled:
            raise asyncio.CancelledError()

        # honour pauses injected by an outer loop
        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done_event.is_set():
            if self._needs_clar:
                clar = await self._clar_down_q.get()
                self._extra_msgs.append(f"Clarification: {clar}")

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_msgs)
            answer = await asyncio.to_thread(self._llm.generate, prompt)
            self._answer = answer
            self._messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            self._done_event.set()

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
        self._done_event.set()
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
        return self._done_event.is_set()

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
# Public simulated KnowledgeManager
# ─────────────────────────────────────────────────────────────────────────────
class SimulatedKnowledgeManager(BaseKnowledgeManager):
    """
    A drop-in, side-effect-free replacement for KnowledgeManager that uses a
    single stateful LLM to invent and recall knowledge in-chat.
    """

    def __init__(self, description: str = "Imaginary knowledge base.") -> None:
        self._description = description

        # One shared, memory-retaining LLM
        self._llm = unify.Unify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        self._llm.set_system_message(
            "You are a *simulated* knowledge-base manager. "
            "No real database exists – you should fabricate plausible tables, "
            "columns and rows and maintain a consistent story across turns.\n\n"
            "As a reference, the system messages for the *real* knowledge-manager 'store' and 'retrieve' methods are as follows."
            "You do not have access to any real tools, so you should just create a final answer to the storage/retireval request. "
            f"\n\n'store' system message:\n{STORE}\n\n"
            f"\n\n'retrieve' system message:\n{RETRIEVE}\n\n"
            f"Back-story: {self._description}",
        )

    # ------------------------------------------------------------------ #
    #  store                                                             #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseKnowledgeManager.store, updated=())
    def store(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,  # unused – we keep state
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        instruction = (
            "On this turn you are simulating the 'store' method.\n"
            f"The user stoage request is:\n{text}"
        )
        if parent_chat_context:
            instruction += (
                f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}"
            )
        return _SimulatedKnowledgeHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    # ------------------------------------------------------------------ #
    #  retrieve                                                          #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseKnowledgeManager.retrieve, updated=())
    def retrieve(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        instruction = (
            "On this turn you are simulating the 'retrieve' method.\n"
            f"The user retrieve request is:\n{text}"
        )
        if parent_chat_context:
            instruction += (
                f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}"
            )
        return _SimulatedKnowledgeHandle(
            self._llm,
            text,
            _return_reasoning_steps=_return_reasoning_steps,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
