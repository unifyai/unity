import time
import functools
import asyncio
import threading
import os
import json

import unify
from .base import BasePlanner, BasePlan
from typing import Optional


class SimulatedPlan(BasePlan):
    """
    A dummy plan class that simulates task execution and question answering.
    Public API surface (stop, ask, interject, pause, resume) is determined dynamically
    based on whether a task is running and whether it is paused.
    """

    def __init__(
        self,
        task: str,
        steps: int,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        request_clarification: bool = False,
    ) -> None:
        """
        Initialize a simulated plan.

        Args:
            task: The task description to simulate
            steps: Number of steps before the plan completes
        """
        self._task = task
        self._steps = steps
        self._clarification_up_q = clarification_up_q
        self._clarification_down_q = clarification_down_q
        self._request_clarification = request_clarification

        # step-counting
        self._step_count = 0
        self._step_lock = threading.Lock()

        # task-control primitives
        self._done_event = threading.Event()
        self._result_str: str | None = None
        self._paused = None
        self._task_thread: threading.Thread | None = None
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()

        self._ask_simulator = unify.Unify(
            "gpt-4o@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            stateful=True,
        )
        self._interject_simulator = unify.Unify(
            "gpt-4o@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            stateful=True,
        )
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
            self._ask_simulator.set_system_message(
                f"You should pretend you are completing the following task:\n{task}\n"
                "Come up with imaginary answers to the user questions about the task",
            )
            self._interject_simulator.set_system_message(
                f"You should pretend you are completing the following task:\n{task}\n"
                "Come up with imaginary responses to the user requests to steer the task behaviour.",
            )

            while True:

                if self._request_clarification:
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

                if self._stop_event.is_set():
                    return

                if self._step_count >= self._steps:
                    self._complete(f"Completed task '{task}' in {self._steps} steps.")
                    return

                self._pause_event.wait()
                time.sleep(0.1)
        finally:
            self._ask_simulator.reset_messages()
            self._ask_simulator.reset_system_message()
            self._interject_simulator.reset_messages()
            self._interject_simulator.reset_system_message()

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
            if self._task_thread and self._task_thread.is_alive():
                self._task_thread.join(timeout=1)

    def _count_step(self):
        if not self._done_event.is_set():
            with self._step_lock:
                self._step_count += 1

    # Pubic

    @functools.wraps(BasePlan.result, updated=())
    async def result(self) -> str:
        await asyncio.to_thread(self._done_event.wait)
        return self._result_str  # type: ignore

    # Dynamic Methods (Public vs Private Depending on State)

    @functools.wraps(BasePlan.stop, updated=())
    def stop(self) -> str:
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        msg = f"Stopped task '{self._task}'"
        # complete with stop message
        self._complete(msg)
        return msg

    @functools.wraps(BasePlan.interject, updated=())
    async def interject(self, instruction: str) -> None:
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        self._count_step()
        await asyncio.to_thread(self._interject_simulator.generate, instruction)

    @functools.wraps(BasePlan.pause, updated=())
    def pause(self) -> str:
        if not self._task:
            raise Exception("No task is running, so nothing to pause.")
        if self._paused:
            return "Task is already paused."
        self._paused = True
        self._pause_event.clear()
        self._count_step()
        return f"Paused task '{self._task}'."

    @functools.wraps(BasePlan.resume, updated=())
    def resume(self) -> str:
        if not self._task:
            raise Exception("No task is running, so nothing to resume.")
        if not self._paused:
            return "Task is already running."
        self._paused = False
        self._pause_event.set()
        self._count_step()
        return f"Resumed task '{self._task}'."

    @functools.wraps(BasePlan.ask, updated=())
    async def ask(self, question: str) -> str:
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        self._count_step()
        return await asyncio.to_thread(self._ask_simulator.generate, question)

    @functools.wraps(BasePlan.done, updated=())
    def done(self) -> bool:
        return self._done_event.is_set()

    @property
    @functools.wraps(BasePlan.valid_tools, updated=())
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


class SimulatedPlanner(BasePlanner[SimulatedPlan]):

    def __init__(self, steps, request_clarification: bool = False) -> None:
        """
        Initialize a simulated planner.

        Args:
            steps: Number of steps before plans complete
        """
        super().__init__()
        self._steps = steps
        self._request_clarification = request_clarification

    def _make_plan(
        self,
        task_description: str,
        *,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SimulatedPlan:
        return SimulatedPlan(
            task_description,
            self._steps,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
            request_clarification=self._request_clarification,
        )
