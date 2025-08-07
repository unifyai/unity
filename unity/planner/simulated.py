import time
import functools
import asyncio
import threading
import os
import json

import unify
from .base import BasePlanner, BaseActiveTask
from typing import Optional


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
        steps: int,
        timeout: float | None = None,
        parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> None:
        """
        Initialize a simulated active task.

        Args:
            task:       The task description to simulate.
            steps:      *(Optional)* Number of tool-invocation steps before this
                        plan automatically completes.
            timeout:    *(Optional)* Absolute timeout (in **seconds**) after
                        which the plan completes, irrespective of the number of
                        steps performed.
        """
        self._llm = llm
        self._task = task
        self._steps = steps
        self._timeout = timeout
        self._parent_chat_context = parent_chat_context
        self._clarification_up_q = clarification_up_q
        self._clarification_down_q = clarification_down_q
        self._requests_clarification = _requests_clarification

        # step-counting
        self._steps_taken = 0
        self._step_lock = threading.Lock()
        # wall-clock timeout
        self._start_time: float | None = None

        # task-control primitives
        self._done_event = threading.Event()
        self._result_str: str | None = None
        self._paused = None
        self._task_thread: threading.Thread | None = None
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()

        self._start()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def clarification_up_q(self) -> Optional[asyncio.Queue[str]]:
        return self._clarification_up_q

    @property
    def clarification_down_q(self) -> Optional[asyncio.Queue[str]]:
        return self._clarification_down_q

    def _run_task(self, task: str) -> None:
        """
        Run the simulated task in a background thread.

        Args:
            task: The task description to simulate
        """
        try:
            while True:

                if self._requests_clarification:
                    # send the question up
                    try:
                        self._clarification_up_q.put_nowait(
                            "Can you please clarify what exactly you'd like me to do?",
                        )
                    except asyncio.QueueFull:
                        pass

                    # wait (non-blocking) for the answer to come back down
                    while True:
                        try:
                            answer: str = self._clarification_down_q.get_nowait()
                            break
                        except asyncio.QueueEmpty:
                            time.sleep(0.05)

                    # finish immediately once we have the clarification
                    self._complete(f"Clarification received: {answer}")
                    return

                # normal execution path (only reached when no clarification needed)

                # honour explicit stop requests --------------------------------
                if self._stop_event.is_set():
                    return

                # wall-clock timeout -------------------------------------------
                if (
                    self._timeout is not None
                    and self._start_time is not None
                    and (time.monotonic() - self._start_time) >= self._timeout
                ):
                    self._complete(
                        f"Completed task '{task}' after {self._timeout} s timeout.",
                    )
                    return

                # tool-step budget ---------------------------------------------
                if self._steps is not None and self._steps_taken >= self._steps:
                    self._complete(
                        f"Completed task '{task}' in {self._steps} steps.",
                    )
                    return

                self._pause_event.wait()
                time.sleep(0.1)
        finally:
            # reset internal state
            self._task = None
            self._paused = None
            self._task_thread = None
            self._pause_event.set()
            self._stop_event.clear()

    def _start(self):
        """Initialize and start the background task thread."""
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
        """
        Internal: finish the plan once step target reached or stopped early.

        Args:
            message: The completion message to store as the result
        """
        if not self._done_event.is_set():
            # stop background thread
            self._stop_event.set()
            # store result and signal completion
            self._result_str = message
            self._done_event.set()
            # kill task thread
            # Avoid self-join which would raise RuntimeError when _complete is
            # called *inside* the task thread.  Only join when invoked from a
            # different thread.
            import threading

            if (
                self._task_thread
                and self._task_thread.is_alive()
                and threading.current_thread() is not self._task_thread
            ):
                self._task_thread.join(timeout=1)

    # Pubic

    def simulate_step(self):
        if not self._done_event.is_set():
            with self._step_lock:
                self._steps_taken += 1

    @functools.wraps(BaseActiveTask.result, updated=())
    async def result(self) -> str:
        await asyncio.to_thread(self._done_event.wait)
        return self._result_str  # type: ignore

    # Dynamic Methods (Public vs Private Depending on State)

    @functools.wraps(BaseActiveTask.stop, updated=())
    def stop(self) -> str:
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        msg = f"Stopped task '{self._task}'"
        # complete with stop message
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
        # When paused we want the user to be able to resume, not call start again.
        if self._paused:
            available[self.resume.__name__] = self.resume
        else:
            available[self.pause.__name__] = self.pause
        return available


class SimulatedPlanner(BasePlanner):
    def __init__(
        self,
        *,
        steps: int | None = None,
        timeout: float | None = None,
        _requests_clarification: bool = False,
    ) -> None:
        """
        Initialize a simulated planner.

        Args:
            steps:      *(Optional)* Maximum tool steps each plan should run
                        before auto-completion.
            timeout:    *(Optional)* Maximum wall-clock seconds before plans
                        auto-complete.
        """
        super().__init__()
        self._steps = steps
        self._timeout = timeout
        self._requests_clarification = _requests_clarification

        # One shared, memory-retaining LLM for *all* plans
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            stateful=True,
        )
        self._llm.set_system_message(
            "You are a *simulated* planner and executor. "
            "Invent plausible task progress and remain internally consistent "
            "across multiple plans and calls.",
        )

    async def _execute_task_and_return_handle(
        self,
        task_description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SimulatedActiveTask:
        return SimulatedActiveTask(
            self._llm,
            task_description,
            self._steps,
            timeout=self._timeout,
            parent_chat_context=parent_chat_context,
            _requests_clarification=self._requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
