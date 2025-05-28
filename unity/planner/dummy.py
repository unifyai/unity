import time
import random

import unify


class DummyPlanner:
    """
    A dummy planner class that simulates task execution and question answering.
    """

    def __init__(self) -> None:
        """
        Initialize the dummy planner with no active task and a GPT-4 client.
        """
        self._active_task = None
        self._ask_simulator = unify.Unify("o4-mini@openai", stateful=True)
        self._steer_simulator = unify.Unify("o4-mini@openai", stateful=True)

    def start(self, task: str):
        """
        Simulate executing a task by setting it as active, waiting, then clearing it.

        Args:
            task (str): The task description to simulate executing.
        """
        self._active_task = task
        self._ask_simulator.set_system_message(
            f"You should pretend you are completing the following task:\n{task}\nCome up with imaginary answers to the user questions about the task",
        )
        self._steer_simulator.set_system_message(
            f"You should pretend you are completing the following task:\n{task}\nCome up with imaginary responses to the user requests to steer the task behaviour, stating that you either can or cannot steer the ongoing task as requested.",
        )
        time.sleep(random.uniform(5, 30))
        self._ask_simulator.set_messages([])
        self._ask_simulator.set_system_message("")
        self._steer_simulator.set_messages([])
        self._steer_simulator.set_system_message("")
        self._active_task = None

    def steer(self, instruction: str) -> str:
        """
        Steer the behaviour of the currently active task.

        Args:
            instruction (str): The instruction for the planner to follow wrt the current active task.

        Returns:
            str: The result of the attempt to steer behaviour, whether this was doable or not.
        """
        if not self._active_task:
            return "No tasks are currently being performed, so I have nothing to steer."
        return self._steer_simulator.generate(instruction)

    def ask(self, question: str) -> str:
        """
        Answer questions about the currently active task.

        Args:
            question (str): The question to answer about the active task.

        Returns:
            str: The answer to the question, or a message indicating no active task.
        """
        if not self._active_task:
            return "No tasks are currently being performed, so I cannot answer your question."
        return self._ask_simulator.generate(question)

    def stop(self, reason: str) -> str:
        """
        Stops the currently active task.

        Args:
            reason (str): The reason for stopping the task.

        Returns:
            str: A message indicating whether the task was stopped or if there was no active task.
        """
        if not self._active_task:
            return (
                "No tasks are currently being performed, so there is nothing to stop."
            )

        task = self._active_task
        self._active_task = None
        self._ask_simulator.set_messages([])
        self._ask_simulator.set_system_message("")
        self._steer_simulator.set_messages([])
        self._steer_simulator.set_system_message("")

        return f"Stopped task '{task}' for reason: {reason}"
