import time
import asyncio
import threading

import unify
from unity.common.llm_helpers import SteerableToolHandle


class SimulatedPlan(SteerableToolHandle):
    """
    A dummy plan class that simulates task execution and question answering.
    Public API surface (stop, ask, interject, pause, resume) is determined dynamically
    based on whether a task is running and whether it is paused.
    """

    def __init__(self, task: str, steps: int) -> None:
        """
        Initialize a simulated plan.

        Args:
            task: The task description to simulate
            steps: Number of steps before the plan completes
        """
        self._task = task
        self._steps = steps

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
            cache=True,
            traced=True,
            stateful=True,
        )
        self._interject_simulator = unify.Unify(
            "gpt-4o@openai",
            cache=True,
            traced=True,
            stateful=True,
        )
        self._start()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

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

    async def result(self) -> str:
        """
        Wait until the specified number of public method calls have completed.

        Returns:
            The final result message from the completed plan
        """
        # block in threadpool until we call _complete
        await asyncio.to_thread(self._done_event.wait)
        return self._result_str  # type: ignore

    # Dynamic Methods (Public vs Private Depending on State)

    def stop(self) -> str:
        """
        Stop the currently running task.

        Returns:
            A message confirming the task was stopped

        Raises:
            Exception: If no task is currently running
        """
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        msg = f"Stopped task '{self._task}'"
        # complete with stop message
        self._complete(msg)
        return msg

    def interject(self, instruction: str) -> str:
        """
        Send an instruction to influence the running task.

        Args:
            instruction: The instruction to send

        Returns:
            A simulated response to the instruction

        Raises:
            Exception: If no task is currently running
        """
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        self._count_step()
        return self._interject_simulator.generate(instruction)

    def pause(self) -> str:
        """
        Pause the currently running task.

        Returns:
            A message confirming the task was paused

        Raises:
            Exception: If no task is running
        """
        if not self._task:
            raise Exception("No task is running, so nothing to pause.")
        if self._paused:
            return "Task is already paused."
        self._paused = True
        self._pause_event.clear()
        self._count_step()
        return f"Paused task '{self._task}'."

    def resume(self) -> str:
        """
        Resume a paused task.

        Returns:
            A message confirming the task was resumed

        Raises:
            Exception: If no task is running
        """
        if not self._task:
            raise Exception("No task is running, so nothing to resume.")
        if not self._paused:
            return "Task is already running."
        self._paused = False
        self._pause_event.set()
        self._count_step()
        return f"Resumed task '{self._task}'."

    def ask(self, question: str) -> str:
        """
        Ask a question about the progress of the ongoing plan.

        Args:
            question: The question to ask

        Returns:
            A simulated answer to the question

        Raises:
            Exception: If no task is currently running
        """
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        self._count_step()
        return self._ask_simulator.generate(question)

    @property
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


class SimulatedPlanner:

    def __init__(self, steps) -> None:
        """
        Initialize a simulated planner.

        Args:
            steps: Number of steps before plans complete
        """
        self._steps = steps
        self._plans = list()

    def start(self, task: str):
        """
        Start a new simulated plan.

        Args:
            task: The task description to simulate

        Returns:
            A new SimulatedPlan instance
        """
        plan = SimulatedPlan(task, self._steps)
        self._plans.append(plan)
        return plan
