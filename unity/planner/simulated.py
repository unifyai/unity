import time
import random
import threading

import unify
from unity.common.llm_helpers import SteerableToolHandle


class SimulatedPlan(SteerableToolHandle):
    """
    A dummy plan class that simulates task execution and question answering.
    Public API surface (stop, ask, interject, pause, resume) is determined dynamically
    based on whether a task is running and whether it is paused.
    """

    def __init__(self, task: str) -> None:
        self._task = task
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

    def _run_task(self, task: str, duration: float) -> None:
        try:
            self._ask_simulator.set_system_message(
                f"You should pretend you are completing the following task:\n{task}\n"
                "Come up with imaginary answers to the user questions about the task",
            )
            self._interject_simulator.set_system_message(
                f"You should pretend you are completing the following task:\n{task}\n"
                "Come up with imaginary responses to the user requests to steer the task behaviour.",
            )

            start_time = time.perf_counter()
            while time.perf_counter() - start_time < duration:
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
        self._paused = False
        self._pause_event.set()
        self._stop_event.clear()
        duration = random.uniform(5, 30)
        self._task_thread = threading.Thread(
            target=self._run_task,
            args=(self._task, duration),
            daemon=True,
        )
        self._task_thread.start()

    # ──────────────────────────────────────────────────────────────────────────
    # Rename public API methods to private…
    # ──────────────────────────────────────────────────────────────────────────

    def _stop(self, reason: str) -> str:
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        task = self._task
        self._stop_event.set()
        if self._task_thread and self._task_thread.is_alive():
            self._task_thread.join(timeout=1)
        return f"Stopped task '{task}' for reason: {reason}"

    def _interject(self, instruction: str) -> str:
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        return self._interject_simulator.generate(instruction)

    def _pause(self) -> str:
        if not self._task:
            raise Exception("No task is running, so nothing to pause.")
        if self._paused:
            return "Task is already paused."
        self._paused = True
        self._pause_event.clear()
        return f"Paused task '{self._task}'."

    def _resume(self) -> str:
        if not self._task:
            raise Exception("No task is running, so nothing to resume.")
        if not self._paused:
            return "Task is already running."
        self._paused = False
        self._pause_event.set()
        return f"Resumed task '{self._task}'."

    def _ask(self, question: str) -> str:
        if not self._task:
            raise Exception("No tasks are currently being performed.")
        return self._ask_simulator.generate(question)

    # ──────────────────────────────────────────────────────────────────────────
    # Dynamic exposure of only the valid methods
    # ──────────────────────────────────────────────────────────────────────────

    def _can_stop(self) -> bool:
        return self._task is not None

    def _can_interject(self) -> bool:
        return self._task is not None

    def _can_ask(self) -> bool:
        return self._task is not None

    def _can_pause(self) -> bool:
        return (self._task is not None) and (not self._paused)

    def _can_resume(self) -> bool:
        return (self._task is not None) and (self._paused is True)

    def __getattr__(self, name: str):
        # if it's one of our public APIs, check its predicate
        if name in ("stop", "interject", "ask", "pause", "resume"):
            can_method = getattr(self, f"_can_{name}")
            if can_method():
                return getattr(self, f"_{name}")
            # else: not currently valid
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'",
            )
        # otherwise normal behavior
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'",
        )

    def __dir__(self):
        base = set(super().__dir__())
        # add only those public names whose predicate is true
        for name in ("stop", "interject", "ask", "pause", "resume"):
            if getattr(self, f"_can_{name}")():
                base.add(name)
        return sorted(base)


class SimulatedPlanner:

    def __init__(self) -> None:
        self._plans = list()

    def start(self, task: str):
        plan = SimulatedPlan(task)
        self._plans.append(plan)
        return plan
