import time
import random
import threading
from typing import Dict, Type

import unify

from common.llm_helpers import tool_use_loop
from task_list_manager.task_list_manager import TaskListManager
from task_manager.sys_msgs import REQUEST


class _DummyPlanner:
    """
    A dummy planner class that simulates task execution and question answering.
    """

    def __init__(self) -> None:
        """
        Initialize the dummy planner with no active task and a GPT-4 client.
        """
        self._active_task = None
        self._client = unify.Unify("o4-mini@openai", stateful=True)

    def ask(self, question: str, response_type: Type = str) -> str:
        """
        Answer questions about the currently active task.

        Args:
            question (str): The question to answer about the active task.
            response_type (Type): The expected response type, defaults to str.

        Returns:
            str: The answer to the question, or a message indicating no active task.
        """
        if not self._active_task:
            return "No tasks are currently being performed, so I cannot answer your question."
        if response_type is not str:
            return self._client.generate(question, response_format=response_type)
        return self._client.generate(question)

    def act(self, task: str):
        """
        Simulate executing a task by setting it as active, waiting, then clearing it.

        Args:
            task (str): The task description to simulate executing.
        """
        self._active_task = task
        self._client.set_system_message(
            f"You should pretend you are completing the following task:\n{task}\nCome up with imaginary answers to the user questions about the task",
        )
        time.sleep(random.uniform(5, 30))
        self._client.set_messages([])
        self._client.set_system_message("")
        self._active_task = None


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

        self._tools = {
            self._tlm.ask.__name__: self._tlm.ask,
            self._planner.ask.__name__: self._planner.ask,
            self._tlm.update__name__: self._tlm.update,
            self._planner.act.__name__: self._planner.act,
        }

    # Public #
    # -------#

    # English-Text requests

    def request(
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
        client = unify.Unify("o4-mini@openai", cache=True, traced=True)
        client.set_system_message(REQUEST)
        ans = tool_use_loop(client, text, self._tools, log_steps=log_tool_steps)
        if return_reasoning_steps:
            return ans, client.messages
        return ans

    # Tools #
    # ------#

    # Task List

    def _ask_about_task_list(self):
        raise NotImplementedError

    def _modify_task_list(self):
        raise NotImplementedError

    # Active Task

    def _start_task(self):
        raise NotImplementedError

    def _ask_about_active_task(self):
        raise NotImplementedError

    def _steer_active_task(self):
        raise NotImplementedError

    def _stop_active_task(self):
        raise NotImplementedError
