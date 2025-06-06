# unity/task_scheduler/simulated_task_scheduler.py
import asyncio
import json
import os
import threading
import functools
from typing import List, Optional

import unify

from ..common.llm_helpers import SteerableToolHandle
from .base import BaseTaskScheduler
from .sys_msgs import ASK, UPDATE
from ..planner.simulated import SimulatedPlanner


class _SimulatedTaskScheduleHandle(SteerableToolHandle):
    """A minimal, LLM-backed handle for ask/update interactions."""

    def __init__(
        self,
        llm: unify.Unify,
        initial_text: str,
        *,
        mode: str,
        _return_reasoning_steps: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> None:
        self._llm = llm
        self._initial_text = initial_text
        self._mode = mode  # "ask" | "update"
        self._ret_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        self._needs_clar = self._clar_up_q is not None and self._clar_down_q is not None

        # ── fire the clarification request right away ──────────────────
        self._clar_requested = False
        if self._needs_clar:
            try:
                self._clar_up_q.put_nowait(
                    "Could you please clarify exactly what you want?",
                )
                self._clar_requested = True
            except asyncio.QueueFull:
                pass

        self._interjections: List[str] = []

        self._done_event = threading.Event()
        self._cancelled = False
        self._answer: Optional[str] = None
        self._messages: List[dict] = []
        self._paused = False

    # ──────────────────────────────────────────────────────────────────────
    # Public API required by SteerableToolHandle
    # ──────────────────────────────────────────────────────────────────────
    async def result(self):
        """Return the LLM answer (or raise if stopped)."""
        if self._cancelled:
            raise asyncio.CancelledError()

        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done_event.is_set():
            # Wait for clarification answer if required
            if self._needs_clar:
                clar_reply = await self._clar_down_q.get()
                self._interjections.append(f"Clarification: {clar_reply}")

            prompt_parts = [self._initial_text] + self._interjections
            user_block = "\n\n---\n\n".join(prompt_parts)

            answer = await asyncio.to_thread(
                self._llm.generate,
                user_block,
            )

            self._answer = answer
            # very small, synthetic trace of “reasoning”
            self._messages = [
                {"role": "user", "content": user_block},
                {"role": "assistant", "content": answer},
            ]
            self._done_event.set()

        if self._ret_steps:
            return self._answer, self._messages
        return self._answer

    def interject(self, message: str) -> str:
        """Append a follow-up message that will be folded into the prompt."""
        if self._cancelled:
            return "Interaction already stopped."
        self._interjections.append(message)
        return "Noted."

    def stop(self) -> str:
        """Cancel further processing so `.result()` raises."""
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


class SimulatedTaskScheduler(BaseTaskScheduler):
    """
    Drop-in replacement for TaskScheduler where the underlying data is
    entirely imaginary – useful for offline demos or unit tests that only
    need the conversational surface.
    """

    def __init__(self, description: str = "You manage an imaginary task list.") -> None:
        self._description = description

        # One shared, *stateful* LLM for *everything*
        self._llm = unify.Unify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        self._llm.set_system_message(
            "You are a *simulated* task-list manager. "
            "No real database exists; invent plausible tasks but stay internally "
            "consistent across turns.\n\n"
            "As reference, here are the *real* TaskScheduler prompts:\n\n"
            f"ASK system message:\n{ASK}\n\n"
            f"UPDATE system message:\n{UPDATE}\n\n"
            f"Back-story: {self._description}",
        )

        # Re-use a single simulated planner for every `start_task`
        self._planner = SimulatedPlanner(steps=2)

    # ------------------------------------------------------------------ #
    #  ask                                                               #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTaskScheduler.ask, updated=())
    def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,  # Ignored – we do not expose tools
        parent_chat_context: list[dict] | None = None,  # Unused – synthetic
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
        return _SimulatedTaskScheduleHandle(
            self._llm,
            instruction,
            mode="ask",
            _return_reasoning_steps=_return_reasoning_steps,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    # ------------------------------------------------------------------ #
    #  update                                                            #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTaskScheduler.update, updated=())
    def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,  # Ignored – no tools here
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
        return _SimulatedTaskScheduleHandle(
            self._llm,
            instruction,
            mode="update",
            _return_reasoning_steps=_return_reasoning_steps,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    # ------------------------------------------------------------------ #
    #  start_task – delegate to SimulatedPlanner.plan                     #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTaskScheduler.start_task, updated=())
    def start_task(
        self,
        task_id: int,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        """
        In the simulated world we don't have a real DB of tasks, so we
        fabricate a description from the *task_id* and spin up a **real**
        `SimulatedPlan` by calling the shared `SimulatedPlanner.plan`.
        """
        task_description = f"Simulated task #{task_id}"
        return self._planner.plan(
            task_description,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
