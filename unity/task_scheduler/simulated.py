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
from .prompt_builders import build_ask_prompt, build_update_prompt
from ..common.llm_helpers import methods_to_tool_dict
from .task_scheduler import TaskScheduler
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
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> None:
        self._llm = llm
        self._initial_text = initial_text
        self._mode = mode  # "ask" | "update"
        self._ret_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        if _requests_clarification and (
            not clarification_up_q or not clarification_down_q
        ):
            raise ValueError(
                "Clarification queues must be provided when _requests_clarification is True",
            )
        self._needs_clar = _requests_clarification

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

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
    ) -> None:
        self._description = description

        # One shared, *stateful* LLM for *everything*
        self._llm = unify.Unify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        # Re-create the real TaskScheduler prompts *dynamically* so the
        # simulated assistant can use them for grounding.
        ask_tools = methods_to_tool_dict(
            TaskScheduler._search_tasks,
            TaskScheduler._nearest_tasks,
            TaskScheduler._get_task_queue,
            include_class_name=False,
        )
        update_tools = methods_to_tool_dict(
            TaskScheduler._create_task,
            TaskScheduler._delete_task,
            TaskScheduler._cancel_tasks,
            TaskScheduler._update_task_queue,
            TaskScheduler._update_task_name,
            TaskScheduler._update_task_description,
            TaskScheduler._update_task_status,
            TaskScheduler._update_task_start_at,
            TaskScheduler._update_task_deadline,
            TaskScheduler._update_task_repetition,
            TaskScheduler._update_task_priority,
            TaskScheduler._search_tasks,
            TaskScheduler._nearest_tasks,
            TaskScheduler._get_task_queue,
            include_class_name=False,
        )
        ask_msg = build_ask_prompt(ask_tools)
        update_msg = build_update_prompt(update_tools)

        self._llm.set_system_message(
            "You are a *simulated* task-list manager. "
            "No real database exists; invent plausible tasks but remain internally "
            "consistent across turns.\n\n"
            "As reference, here are the *real* TaskScheduler prompts:\n\n"
            f"ASK system message:\n{ask_msg}\n\n"
            f"UPDATE system message:\n{update_msg}\n\n"
            f"Back-story: {self._description}",
        )

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
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        instruction = (
            "On this turn you are simulating the 'ask' method.\n"
            "Please always *answer* the question (making up the response), "
            "do not ask for clarifications, or only state *how* you will answer the question.\n"
            "Just answer the question with an imaginery response.\n"
            "Please *always* mention the relevant task id(s) in your response.\n"
            "The user will almost certainly require the task ids in order to do anything meaningful with your answer.\n"
            "If they ask if a task already exists in the task list, always respond 'No', "
            "stating that the task does *not* already exist."
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
            _requests_clarification=_requests_clarification,
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
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        instruction = (
            "On this turn you are simulating the 'update' method.\n"
            "Please always act as though the task has been completed "
            "(making up an imaginery response to adress any specific details if necessary), "
            "do not ask for clarifications, explain how you *would* proceed.\n"
            "Just respond as though the user request has been handled without error.\n"
            "If a any tasks were created or updated in the imagined process,"
            "then please *always* include these task id(s) in your final response."
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
            _requests_clarification=_requests_clarification,
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
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        """
        In the simulated world we don't have a real DB of tasks, so we
        fabricate a description from the *task_id* and spin up a **real**
        `SimulatedPlan` by calling the shared `SimulatedPlanner.plan`.
        """
        task_description = f"Simulated task #{task_id}"
        return SimulatedPlanner(
            timeout=10,
            _requests_clarification=_requests_clarification,
        ).plan(
            task_description,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
