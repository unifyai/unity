# unity/task_scheduler/simulated_task_scheduler.py
import asyncio
import time
import json
import os
import threading
import functools
from typing import List, Optional

import unify

from ..common.llm_helpers import SteerableToolHandle
from .base import BaseActiveTask, BaseTaskScheduler
from .prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
    build_simulated_method_prompt,
)
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
from ..common.simulated import mirror_task_scheduler_tools


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

            answer = await self._llm.generate(user_block)

            self._answer = answer
            # very small, synthetic trace of "reasoning"
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

    def stop(self, reason: Optional[str] = None) -> str:
        """Cancel further processing so `.result()` raises."""
        self._cancelled = True
        self._done_event.set()
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

    async def ask(self, question: str) -> "SteerableToolHandle":
        q_msg = (
            f"Your only task is to simulate an answer to the following question: {question}\n\n"
            "However, there is a also ongoing simulated process which had the instructions given below. "
            "Please make your answer realastic and conceivable given the provided context of the simulated taks."
        )
        follow_up_prompt = "\n\n---\n\n".join(
            [q_msg]
            + [self._initial_text]
            + self._interjections
            + [f"Question to answer (as a reminder!): {question}"],
        )

        return _SimulatedTaskScheduleHandle(
            self._llm,
            follow_up_prompt,
            mode=self._mode,
            _return_reasoning_steps=self._ret_steps,
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )


class SimulatedActiveTask(BaseActiveTask):
    """
    A dummy active task class that simulates task execution and question answering.
    Public API surface (stop, ask, interject, pause, resume) is determined dynamically
    based on whether a task is running and whether it is paused.
    """

    def __init__(
        self,
        llm: unify.AsyncUnify,
        task: str,
        steps: int | None,
        timeout: float | None = None,
        parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> None:
        self._llm = llm
        self._task = task
        self._steps = steps
        self._timeout = timeout
        self._parent_chat_context = parent_chat_context
        self._clarification_up_q = clarification_up_q
        self._clarification_down_q = clarification_down_q
        self._requests_clarification = _requests_clarification

        self._steps_taken = 0
        self._step_lock = threading.Lock()
        self._start_time: float | None = None

        self._done_event = threading.Event()
        self._result_str: str | None = None
        self._paused = None
        self._task_thread: threading.Thread | None = None
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()

        self._start()

    @property
    def clarification_up_q(self) -> Optional[asyncio.Queue[str]]:
        return self._clarification_up_q

    @property
    def clarification_down_q(self) -> Optional[asyncio.Queue[str]]:
        return self._clarification_down_q

    def _run_task(self, task: str) -> None:
        try:
            while True:
                if self._requests_clarification:
                    try:
                        self._clarification_up_q.put_nowait(
                            "Can you please clarify what exactly you'd like me to do?",
                        )
                    except asyncio.QueueFull:
                        pass
                    while True:
                        try:
                            answer: str = self._clarification_down_q.get_nowait()
                            break
                        except asyncio.QueueEmpty:
                            time.sleep(0.05)
                    self._complete(f"Clarification received: {answer}")
                    return
                if self._stop_event.is_set():
                    return
                if (
                    self._timeout is not None
                    and self._start_time is not None
                    and (time.monotonic() - self._start_time) >= self._timeout
                ):
                    self._complete(
                        f"Completed task '{task}' after {self._timeout}\u2009s timeout.",
                    )
                    return
                if self._steps is not None and self._steps_taken >= (self._steps or 0):
                    self._complete(
                        f"Completed task '{task}' in {self._steps} steps.",
                    )
                    return
                self._pause_event.wait()
                time.sleep(0.1)
        finally:
            self._task = None
            self._paused = None
            self._task_thread = None
            self._pause_event.set()
            self._stop_event.clear()

    def _start(self):
        self._paused = False
        self._pause_event.set()
        self._stop_event.clear()
        self._start_time = time.monotonic()
        self._task_thread = threading.Thread(
            target=self._run_task,
            args=(self._task,),
            daemon=True,
        )
        self._task_thread.start()

    def _complete(self, message: str) -> None:
        if not self._done_event.is_set():
            self._stop_event.set()
            self._result_str = message
            self._done_event.set()
            import threading as _th

            if (
                self._task_thread
                and self._task_thread.is_alive()
                and _th.current_thread() is not self._task_thread
            ):
                self._task_thread.join(timeout=1)

    def simulate_step(self):
        if not self._done_event.is_set():
            with self._step_lock:
                self._steps_taken += 1

    @functools.wraps(BaseActiveTask.result, updated=())
    async def result(self) -> str:
        await asyncio.to_thread(self._done_event.wait)
        return self._result_str  # type: ignore

    @functools.wraps(BaseActiveTask.stop, updated=())
    def stop(self, reason: Optional[str] = None) -> str:
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        msg = f"Stopped task '{self._task}' for reason: {reason}"
        self._complete(msg)
        return msg

    @functools.wraps(BaseActiveTask.interject, updated=())
    async def interject(self, instruction: str) -> None:
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        self.simulate_step()
        prompt = (
            f"Current simulated task:\n{self._task}\n\n"
            f"User instruction to adjust the plan:\n{instruction}"
        )
        await self._llm.generate(prompt)

    @functools.wraps(BaseActiveTask.pause, updated=())
    def pause(self) -> str:
        if not self._task:
            raise Exception("No task is running, so nothing to pause.")
        if self._paused:
            return "Task is already paused."
        self._paused = True
        self._pause_event.clear()
        self.simulate_step()
        return f"Paused task '{self._task}'."

    @functools.wraps(BaseActiveTask.resume, updated=())
    def resume(self) -> str:
        if not self._task:
            raise Exception("No task is running, so nothing to resume.")
        if not self._paused:
            return "Task is already running."
        self._paused = False
        self._pause_event.set()
        self.simulate_step()
        return f"Resumed task '{self._task}'."

    @functools.wraps(BaseActiveTask.ask, updated=())
    async def ask(self, question: str) -> str:
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        self.simulate_step()
        prompt = (
            f"You are working on the simulated task:\n{self._task}\n\n"
            f"User asks: {question}"
        )
        return await self._llm.generate(prompt)

    @functools.wraps(BaseActiveTask.done, updated=())
    def done(self) -> bool:
        return self._done_event.is_set()

    @property
    @functools.wraps(BaseActiveTask.valid_tools, updated=())
    def valid_tools(self):
        if self._task is None:
            return {}
        available = {
            self.stop.__name__: self.stop,
            self.interject.__name__: self.interject,
            self.ask.__name__: self.ask,
        }
        if self._paused:
            available[self.resume.__name__] = self.resume
        else:
            available[self.pause.__name__] = self.pause
        return available


class SimulatedTaskScheduler(BaseTaskScheduler):
    """
    Drop-in replacement for TaskScheduler where the underlying data is
    entirely imaginary – useful for offline demos or unit tests that only
    need the conversational surface.
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
        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # One shared, *stateful* LLM for *everything*
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        # Mirror the real TaskScheduler tool exposure programmatically
        # so prompts always reflect the current surface.
        ask_tools = mirror_task_scheduler_tools("ask")
        update_tools = mirror_task_scheduler_tools("update")

        # Provide placeholder counts/columns for the simulated environment
        from .types.task import Task as _Task

        fake_task_columns = [
            {k: str(v.annotation)} for k, v in _Task.model_fields.items()
        ]

        ask_msg = build_ask_prompt(
            ask_tools,
            num_tasks=10,
            columns=fake_task_columns,
            include_activity=self._rolling_summary_in_prompts,
        )
        update_msg = build_update_prompt(
            update_tools,
            num_tasks=10,
            columns=fake_task_columns,
            include_activity=self._rolling_summary_in_prompts,
        )

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
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,  # Ignored – we do not expose tools
        parent_chat_context: list[dict] | None = None,  # Unused – synthetic
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
                "TaskScheduler",
                "ask",
                phase="incoming",
                question=text,
            )

        instruction = build_simulated_method_prompt(
            "ask",
            text,
            parent_chat_context=parent_chat_context,
        )
        instruction += (
            "\n\nPlease *always* mention the relevant task id(s) in your response. "
            "If the user asks whether a task already exists in the list, reply 'No' and state it does *not* exist."
        )
        handle = _SimulatedTaskScheduleHandle(
            self._llm,
            instruction,
            mode="ask",
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "TaskScheduler",
                "ask",
            )

        return handle

    # Provide guidance for outer orchestrators via tool description (read-only ask)
    ask.__doc__ = (ask.__doc__ or "") + (
        "\n\nOuter-orchestrator guidance: Avoid invoking this tool repeatedly with the same "
        "arguments within the same conversation. Prefer reusing prior results and "
        "compose the final answer once sufficient information has been gathered."
    )

    # ------------------------------------------------------------------ #
    #  update                                                            #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTaskScheduler.update, updated=())
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,  # Ignored – no tools here
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
                "TaskScheduler",
                "update",
                phase="incoming",
                request=text,
            )

        instruction = build_simulated_method_prompt(
            "update",
            text,
            parent_chat_context=parent_chat_context,
        )
        instruction += "\n\nIf any tasks were created or updated during the imagined process, include their id(s) in your reply."
        handle = _SimulatedTaskScheduleHandle(
            self._llm,
            instruction,
            mode="update",
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "TaskScheduler",
                "update",
            )

        return handle

    # Provide guidance for outer orchestrators (mutation idempotence)
    update.__doc__ = (update.__doc__ or "") + (
        "\n\nOuter-orchestrator guidance: Avoid invoking this mutation with the same arguments multiple times in the same "
        "conversation. Treat this operation as idempotent; if confirmation is needed, perform a single read to verify the outcome."
    )

    # ------------------------------------------------------------------ #
    #  execite_task – delegate to SimulatedActor.act                     #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseTaskScheduler.execute_task, updated=())
    async def execute_task(
        self,
        text: str,
        *,
        parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        """
        Execute a *simulated* task from **free-form** text.

        The implementation pretends that the supplied *text* uniquely
        identifies the task – no attempt is made to reconcile with a real data
        store.  A new :class:`unity.actor.simulated.SimulatedPlan` is spun up
        and its handle returned.
        """
        should_log = self._log_events or log_events
        call_id = None

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "TaskScheduler",
                "execute_task",
                phase="incoming",
                request=text,
            )

        task_description = f"{text} (simulated)"
        # Local import to avoid circular import with actor.simulated which re-exports SimulatedActiveTask
        from ..actor.simulated import SimulatedActor

        actor = SimulatedActor(
            timeout=10,
            _requests_clarification=_requests_clarification,
        )
        handle = await actor.act(
            task_description,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "TaskScheduler",
                "execute_task",
            )

        return handle
