from typing import Dict

import unify
import os
from ..common.llm_helpers import start_async_tool_use_loop
from ..planner.simulated import SimulatedPlanner
from ..communication.transcript_manager.transcript_manager import TranscriptManager
from ..task_list_manager.task_list_manager import TaskListManager
from ..knowledge_manager.knowledge_manager import KnowledgeManager

# from ..contact_manager.contact_manager import ContactManager
from ..events.event_bus import EventBus
from .sys_msgs import REQUEST, ASK
import json


class TaskManager:

    def __init__(self) -> None:
        """
        Responsible for managing the set of tasks to complete by updating, scheduling, executing and steering tasks, both scheduled and active.
        """
        self._event_bus = EventBus()
        # self._contact_manager = ContactManager
        self._transcript_manager = TranscriptManager(self._event_bus)
        self._knowledge_manager = KnowledgeManager()
        self._task_list_manager = TaskListManager()
        self._planner = SimulatedPlanner()

        self._passive_tools = {
            # contact
            # f"ContactManager.{self._contact_manager.ask.__name__}": self._contact_manager.ask,
            # transcript
            f"TranscriptManager.{self._transcript_manager.ask.__name__}": self._transcript_manager.ask,
            # knowledge
            f"KnowledgeManager.{self._knowledge_manager.retrieve}": self._knowledge_manager.retrieve,
            # task list
            f"TaskListManager.{self._task_list_manager.ask}": self._task_list_manager.ask,
            # planner
            f"Planner.{self._planner.ask}": self._planner.ask,
        }
        self._active_tools = {
            # contact
            # f"ContactManager.{self._contact_manager.update.__name__}": self._contact_manager.update,
            # transcript
            f"TranscriptManager.{self._transcript_manager.summarize}": self._transcript_manager.summarize,
            # knowledge
            f"KnowledgeManager.{self._knowledge_manager.store}": self._knowledge_manager.store,
            # task list
            f"TaskListManager.{self._task_list_manager.update}": self._task_list_manager.update,
            # planner
            f"Planner.{self._planner.start}": self._planner.start,
            f"Planner.{self._planner.steer}": self._planner.steer,
            f"Planner.{self._planner.stop}": self._planner.stop,
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
