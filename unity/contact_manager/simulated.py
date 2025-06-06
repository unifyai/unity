# unity/contact_manager/simulated_contact_manager.py
from __future__ import annotations

import asyncio
import json
import os
import functools
import threading
from typing import List, Dict, Any

import unify
from .base import BaseContactManager
from .sys_msgs import ASK_CONTACTS, UPDATE_CONTACTS
from ..common.llm_helpers import SteerableToolHandle


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
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
    ):
        self._llm = llm
        self._initial = initial_text
        self._want_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        self._needs_clar = self._clar_up_q is not None and self._clar_down_q is not None

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
            answer = await asyncio.to_thread(self._llm.generate, prompt)
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


# ─────────────────────────────────────────────────────────────────────────────
# Public simulated manager
# ─────────────────────────────────────────────────────────────────────────────
class SimulatedContactManager(BaseContactManager):
    """
    Drop-in replacement for ContactManager with imaginary data and
    stateful LLM memory.
    """

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
    ) -> None:
        self._description = description

        # Shared, stateful LLM
        self._llm = unify.Unify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        self._llm.set_system_message(
            "You are a *simulated* contact-manager assistant. "
            "There is no real database; invent plausible contact records and "
            "keep your story consistent across turns.\n\n"
            "As a reference, the system messages for the *real* contact-manager 'ask' and 'update' methods are as follows."
            "You do not have access to any real tools, so you should just create a final answer to the question/request . "
            f"\n\n'ask' system message:\n{ASK_CONTACTS}\n\n"
            f"\n\n'update' system message:\n{UPDATE_CONTACTS}\n\n"
            f"Back-story: {self._description}",
        )

    # --------------------------------------------------------------------- #
    # ask                                                                   #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseContactManager.ask, updated=())
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
        return _SimulatedContactHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    # --------------------------------------------------------------------- #
    # update                                                                #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseContactManager.update, updated=())
    def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        instruction = (
            "On this turn you are simulating the 'update' method.\n"
            f"The user update request is:\n{text}"
        )
        if parent_chat_context:
            instruction += (
                f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}"
            )
        return _SimulatedContactHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
