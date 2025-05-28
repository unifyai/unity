import time
import random
import threading

import unify


class SimulatedPlanner:
    """
    A dummy planner class that simulates task execution and question answering.
    The simulated execution timer can now be paused and resumed.
    """

    def __init__(self) -> None:
        self._active_task = None
        self._paused = None
        self._task_thread: threading.Thread | None = None
        self._pause_event = threading.Event()  # set ⇒ running, clear ⇒ paused
        self._stop_event = threading.Event()  # set ⇒ abort execution

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

    # ──────────────────────────────────────────────────────────────────────────
    # Core helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _run_task(self, task: str, duration: float) -> None:
        """Internal worker loop executed inside a daemon thread."""
        try:
            self._ask_simulator.set_system_message(
                f"You should pretend you are completing the following task:\n{task}\n"
                f"Come up with imaginary answers to the user questions about the task",
            )
            self._interject_simulator.set_system_message(
                f"You should pretend you are completing the following task:\n{task}\n"
                f"Come up with imaginary responses to the user requests to steer the task behaviour, "
                f"stating that you either can or cannot steer the ongoing task as requested.",
            )

            start_time = time.perf_counter()
            while (elapsed := time.perf_counter() - start_time) < duration:
                # Block here while paused; immediately return if we were asked to stop.
                if self._stop_event.is_set():
                    return
                self._pause_event.wait()  # waits if pause_event is cleared
                # Sleep in small increments so pause can be felt quickly.
                time.sleep(0.1)

        finally:
            # Clean-up happens regardless of normal completion or external stop.
            self._ask_simulator.reset_messages()
            self._ask_simulator.reset_system_message()
            self._interject_simulator.reset_messages()
            self._interject_simulator.reset_system_message()
            self._active_task = None
            self._paused = None
            self._task_thread = None
            self._pause_event.set()  # ensure future tasks start unpaused
            self._stop_event.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Public planner API
    # ──────────────────────────────────────────────────────────────────────────
    def start(self, task: str):
        """
        Begin a simulated task asynchronously.
        The method returns immediately; the task can be paused/resumed.
        """
        if self._active_task is not None:
            raise Exception("Another task is already running.")
        self._active_task = task
        self._paused = False

        # Prepare control events
        self._pause_event.set()
        self._stop_event.clear()

        duration = random.uniform(5, 30)
        self._task_thread = threading.Thread(
            target=self._run_task,
            args=(task, duration),
            daemon=True,
        )
        self._task_thread.start()

    def stop(self, reason: str) -> str:
        """
        Stops the currently active task (even if paused).
        """
        if not self._active_task:
            raise Exception(
                "No tasks are currently being performed, so there is nothing to stop.",
            )

        task = self._active_task
        # Signal the thread to exit and wait for it to join.
        self._stop_event.set()
        if self._task_thread and self._task_thread.is_alive():
            self._task_thread.join(timeout=1)

        # Clean-up handled by worker's finally block; just return message.
        return f"Stopped task '{task}' for reason: {reason}"

    def interject(self, instruction: str) -> str:
        if not self._active_task:
            raise Exception(
                "No tasks are currently being performed, so I have nothing to steer.",
            )
        return self._interject_simulator.generate(instruction)

    def pause(self) -> str:
        """
        Pause the simulated execution timer.
        """
        if not self._active_task:
            raise Exception("No task is running, so nothing to pause.")
        if self._paused:
            return "Task is already paused."

        self._paused = True
        self._pause_event.clear()
        return f"Paused task '{self._active_task}'."

    def resume(self) -> str:
        """
        Resume the simulated execution timer.
        """
        if not self._active_task:
            raise Exception("No task is running, so nothing to resume.")
        if not self._paused:
            return "Task is already running."

        self._paused = False
        self._pause_event.set()
        return f"Resumed task '{self._active_task}'."

    def ask(self, question: str) -> str:
        if not self._active_task:
            raise Exception(
                "No tasks are currently being performed, so I cannot answer your question.",
            )
        return self._ask_simulator.generate(question)

    @property
    def valid_tools(self):
        if self._active_task is None:
            return {f"Planner.{self.start}": self.start}

        available = {
            f"Planner.{self.stop}": self.stop,
            f"Planner.{self.interject}": self.interject,
            f"Planner.{self.ask}": self.ask,
        }
        # When paused we want the user to be able to resume, not call start again.
        if self._paused:
            available[f"Planner.{self.resume}"] = self.resume
        else:
            available[f"Planner.{self.pause}"] = self.pause
        return available
