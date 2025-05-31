import time
import asyncio
import inspect
import threading
import functools

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
        # count how many public calls have been made
        self._step_count = 0
        # event to signal completion and storage for result
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
            args=(self._task),
            daemon=True,
        )
        self._task_thread.start()

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

    def _complete(self, message: str) -> None:
        """
        Internal: finish the plan once step target reached or stopped early.

        Args:
            message: The completion message to store as the result
        """
        if not self._done_event.is_set():
            # stop background thread
            self._stop_event.set()
            if self._task_thread and self._task_thread.is_alive():
                self._task_thread.join(timeout=1)
            # store result and signal completion
            self._result_str = message
            self._done_event.set()

    # Dynamic Methods (Public vs Private Depending on State)

    def _stop(self, reason: str) -> str:
        """
        Stop the currently running task.

        Args:
            reason: The reason for stopping the task

        Returns:
            A message confirming the task was stopped

        Raises:
            Exception: If no task is currently running
        """
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        task = self._task
        msg = f"Stopped task '{self._task}' for reason: {reason}"
        # complete with stop message
        self._complete(msg)
        return msg

    def _interject(self, instruction: str) -> str:
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
        return self._interject_simulator.generate(instruction)

    def _pause(self) -> str:
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
        return f"Paused task '{self._task}'."

    def _resume(self) -> str:
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
        return f"Resumed task '{self._task}'."

    def _ask(self, question: str) -> str:
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
        return self._ask_simulator.generate(question)

    # Dynamic exposure of only the valid methods

    def _can_stop(self) -> bool:
        """Check if stop operation is currently valid."""
        return self._task is not None

    def _can_interject(self) -> bool:
        """Check if interject operation is currently valid."""
        return self._task is not None

    def _can_ask(self) -> bool:
        """Check if ask operation is currently valid."""
        return self._task is not None

    def _can_pause(self) -> bool:
        """Check if pause operation is currently valid."""
        return (self._task is not None) and (not self._paused)

    def _can_resume(self) -> bool:
        """Check if resume operation is currently valid."""
        return (self._task is not None) and (self._paused is True)

    def __getattr__(self, name: str):
        """
        Dynamic attribute lookup that exposes public methods only when valid.

        Args:
            name: The attribute name being looked up

        Returns:
            The wrapped method if available

        Raises:
            AttributeError: If the method is not available in current state
        """
        # any public API call counts as a step
        public = ("stop", "interject", "ask", "pause", "resume")
        if name in public:
            can_method = getattr(self, f"_can_{name}")
            if not can_method():
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'",
                )
            fn = getattr(self, f"_{name}")

            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                # increment step counter
                if not self._done_event.is_set():
                    self._step_count += 1
                    if self._step_count >= self._steps:
                        # complete with success message
                        self._complete(
                            f"Task '{self._task}' completed after {self._steps} steps.",
                        )
                return fn(*args, **kwargs)

            sig = inspect.signature(fn)
            wrapped.__signature__ = sig  # so introspection works
            wrapped.__annotations__ = getattr(fn, "__annotations__", {}).copy()

            return wrapped
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'",
        )

    def __dir__(self):
        """
        Return list of valid attributes, including only currently available public methods.

        Returns:
            List of valid attribute names
        """
        base = set(super().__dir__())
        # add only those public names whose predicate is true
        for name in ("stop", "interject", "ask", "pause", "resume"):
            if getattr(self, f"_can_{name}")():
                base.add(name)
        return sorted(base)


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
