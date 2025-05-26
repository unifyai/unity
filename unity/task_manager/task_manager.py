import time
import random
import threading
from typing import Dict

import unify

from ..common.llm_helpers import start_async_tool_use_loop
from ..task_list_manager.task_list_manager import TaskListManager
from .sys_msgs import REQUEST


class _DummyPlanner:
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


class TaskManager(threading.Thread):

    def __init__(self, *, daemon: bool = True) -> None:
        """
        Responsible for managing the set of tasks to complete by updating, scheduling, executing and steering tasks, both scheduled and active.

        Args:
            daemon (bool): Whether the thread should be a daemon thread.
        """
        super().__init__(daemon=daemon)
        self._tlm = TaskListManager()
        self._planner = _DummyPlanner()

        self._task_list_tools = {
            self._ask_about_task_list.__name__: self._ask_about_task_list,
            self._update_task_list.__name__: self._update_task_list,
        }
        self._active_task_tools = {
            self._start_task.__name__: self._start_task,
            self._ask_about_active_task.__name__: self._ask_about_active_task,
            self._steer_active_task.__name__: self._steer_active_task,
            self._stop_active_task.__name__: self._stop_active_task,
        }

        self._tools = {
            **self._task_list_tools,
            **self._active_task_tools,
        }

    # Public #
    # -------#

    # English-Text requests

    async def request(
        self,
        text: str,
        *,
        return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,
    ) -> Dict[str, str]:
        """
        Handle any plain-text english task-related request, which can refer to inactive (cancelled, scheduled, failed) or active tasks in the task list, and can also involve updates to any of those tasks and/or simply answering questions about them. It can also involve changes to the currently active task, and involve multi-step reasoning which includes questions and actions for both inactive and active tasks.

        For example: text="if this {some task description} is marked as high or lower, then mark it as urgent and start it right now (pausing any other task that might be active right now))"

        Args:
            text (str): The task request, which can involve questions, actions, or both interleaved.
            return_reasoning_steps (bool): Whether to return the reasoning steps for the ask request.
            log_tool_steps (bool): Whether to log the steps taken by the tool.

        Returns:
            Dict[str, str]: Answers to the question(s), and updates on any action(s) performed.
        """
        client = unify.AsyncUnify("o4-mini@openai", cache=True, traced=True)
        client.set_system_message(REQUEST)
        ans = await start_async_tool_use_loop(
            client,
            text,
            self._tools,
            log_steps=log_tool_steps,
        ).result()
        if return_reasoning_steps:
            return ans, client.messages
        return ans

    # Tools #
    # ------#

    # Task List

    def _ask_about_task_list(self, question: str) -> str:
        f"""
        Ask any question about the list of tasks (including scheduled, cancelled, failed, and the active task) based on a natural language question.

        This function *cannot* answer questions about the *live state* of the active task.
        It can answer questions about the schedule, priority, title, description, queue ordering etc.

        Args:
            question (str): The question to ask about the task list.

        Returns:
            str: The answer to the question about the task list.
        """
        return self._tlm.ask(question)

    def _update_task_list(self, update: str) -> str:
        f"""
        Update the list of tasks (including scheduled, cancelled, failed, and the active task) based on a natural language question.

        Args:
            update (str): The update instruction in natural language.

        Returns:
            str: Whether the update was applied successfully or not.
        """
        return self._tlm.update(update)

    # Active Task

    def _start_task(self, description: str) -> str:
        """
        Start a new task, making it the active task. If there is already an active task,
        it will be paused.

        Args:
            description (str): Description of the task to start.

        Returns:
            str: A message confirming the task was started, or explaining why it couldn't be started.
        """
        return self._planner.start(description)

    def _ask_about_active_task(self, question: str) -> str:
        """
        Ask a question about the currently active task, including its live state.

        Args:
            question (str): The question to ask about the active task.

        Returns:
            str: The answer about the active task's current state, or a message indicating no active task.
        """
        return self._planner.ask(question)

    def _steer_active_task(self, instruction: str) -> str:
        """
        Provide steering instructions to modify the behavior of the currently active task.

        Args:
            instruction (str): The steering instruction in natural language.

        Returns:
            str: A message confirming the steering instruction was applied, or explaining why it couldn't be applied.
        """
        return self._planner.steer(instruction)

    def _stop_active_task(self, reason: str) -> str:
        """
        Stop the currently active task.

        Args:
            reason (str): The reason for stopping the task, which will be recorded.

        Returns:
            str: A message confirming the task was stopped, or explaining why it couldn't be stopped.
        """
        self._planner.stop(reason)
