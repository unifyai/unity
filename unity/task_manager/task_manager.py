import threading
from typing import Dict

import unify
import os
from ..common.llm_helpers import start_async_tool_use_loop
from ..planner.dummy import DummyPlanner
from ..task_list_manager.task_list_manager import TaskListManager
from .sys_msgs import REQUEST, ASK
import json


class TaskManager(threading.Thread):

    def __init__(self, *, daemon: bool = True) -> None:
        """
        Responsible for managing the set of tasks to complete by updating, scheduling, executing and steering tasks, both scheduled and active.

        Args:
            daemon (bool): Whether the thread should be a daemon thread.
        """
        super().__init__(daemon=daemon)
        self._tlm = TaskListManager()
        self._planner = DummyPlanner()

        self._passive_tools = {
            # task list
            self._ask_about_task_list.__name__: self._ask_about_task_list,
            # planner
            self._ask_about_active_task.__name__: self._ask_about_active_task,
        }
        self._active_tools = {
            # task list
            self._update_task_list.__name__: self._update_task_list,
            # planner
            self._start_task.__name__: self._start_task,
            self._steer_active_task.__name__: self._steer_active_task,
            self._stop_active_task.__name__: self._stop_active_task,
        }

        self._tools = {
            **self._passive_tools,
            **self._active_tools,
        }

    # Public #
    # -------#

    # English-Text requests

    async def ask(
        self,
        question: str,
        *,
        return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,
    ) -> str:
        """
        Asks a question about any task, whether it be scheduled, active, cancelled, complete.
        The question can be about the fine-grained details of execution, the broader context, or anything in between.

        Args:
            question (str): The question to ask about the task.
            return_reasoning_steps (bool): Whether to return the reasoning steps for the ask request.
            log_tool_steps (bool): Whether to log the steps taken by the tool.

        Returns:
            str: The answer to the question about the task.
        """
        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(ASK)
        ans = await start_async_tool_use_loop(
            client,
            question,
            self._passive_tools,
            log_steps=log_tool_steps,
        ).result()
        if return_reasoning_steps:
            return ans, client.messages
        return ans

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
        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
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
