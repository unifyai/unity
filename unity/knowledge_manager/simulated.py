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
from .prompt_builders import (
    build_refactor_prompt,
    build_update_prompt,
    build_ask_prompt,
)


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
            answer = await self._llm.generate(prompt)
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

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
    ) -> None:
        self._description = description

        # One shared, memory-retaining LLM
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        # Build *empty* reference prompts (no tools, empty schema) purely for flavour.
        refactor_ref = build_refactor_prompt({}, table_schemas_json="{}")
        store_ref = build_update_prompt({}, table_schemas_json="{}")
        retrieve_ref = build_ask_prompt({}, table_schemas_json="{}")

        self._llm.set_system_message(
            "You are a *simulated* knowledge-base manager. "
            "There is no real database; invent plausible tables, columns and rows "
            "and keep your story consistent across turns.\n\n"
            "As a reference, the (tool-enabled) system messages for the *real* "
            "knowledge-manager are pasted below. **You do not actually have access "
            "to any tools – just produce the final answer.**\n\n"
            f"'refactor' system message:\n{refactor_ref}\n\n"
            f"'store' system message:\n{store_ref}\n\n"
            f"'retrieve' system message:\n{retrieve_ref}\n\n"
            f"Back-story: {self._description}",
        )

    # ------------------------------------------------------------------ #
    #  refactor                                                          #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseKnowledgeManager.refactor, updated=())
    async def refactor(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        """
        Simulated version of KnowledgeManager.refactor – no real DDL is run.
        The LLM simply invents a plausible migration plan and returns it.
        """
        instruction = (
            "On this turn you are simulating the 'refactor' method.\n"
            "Pretend you have analysed the entire schema and the contacts "
            "table; respond with a short migration plan that removes any "
            "duplication, introduces surrogate keys where useful, and generally "
            "moves the schema toward third-normal-form.  **Do not** execute any "
            "actual tool calls – just describe what you *would* do.\n"
            f"The user's refactor request is:\n{text}"
        )
        if parent_chat_context:
            instruction += (
                f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}"
            )
        return _SimulatedKnowledgeHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=False,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    # ------------------------------------------------------------------ #
    #  store                                                             #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseKnowledgeManager.update, updated=())
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        instruction = (
            "On this turn you are simulating the 'store' method.\n"
            f"The user stoage request is:\n{text}\n"
            "If the user refers to creating *tasks*, then you should **not** store any tasks.\n"
            "Tasks should exclusively be sotred by a separate task manager, this is **not your responsibility**.\n"
            "Please explain this to the user in your response, if this is part of the their request."
        )
        if parent_chat_context:
            instruction += (
                f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}"
            )
        return _SimulatedKnowledgeHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    # ------------------------------------------------------------------ #
    #  retrieve                                                          #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseKnowledgeManager.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        instruction = (
            "On this turn you are simulating the 'retrieve' method.\n"
            "Please always return imaginery information (making up the response), "
            "do not ask for clarifications, or only state *how* you will get the information.\n"
            "Just respond immediately with imaginery information.\n"
            f"The user retrieve request is:\n{text}"
        )
        if parent_chat_context:
            instruction += (
                f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}"
            )
        return _SimulatedKnowledgeHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
