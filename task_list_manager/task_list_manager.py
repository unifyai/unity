import threading
from datetime import datetime
from typing import Dict, List, Optional

import unify
from llm_helpers import tool_use_loop
from task_list_manager.types.priority import Priority


class TaskListManager(threading.Thread):

    def __init__(self, *, daemon: bool = True) -> None:
        """
        Responsible managing the list of tasks, updating the names, descriptions, schedles, repeating pattern and status of all tasks.
        """
        super().__init__(daemon=daemon)
        # ToDo: implement the tools
        self._tools = {}

    # Public #
    # -------#

    # English-Text update request

    def update(self, text: str, return_reasoning_steps: bool = False) -> Dict[str, str]:
        """
        Handle any plain-text english command to update the list of tasks in some manner.

        Args:
            text (str): The text-based request to update the task list.

            return_reasoning_steps (bool): Whether to return the reasoning steps for the update request.

        Returns:
            Dict[str, str]: Whether the task list was updated or not.
        """
        from task_list_manager.sys_msgs import UPDATE

        client = unify.Unify("o4-mini@openai")
        client.set_system_message(UPDATE)
        ans = tool_use_loop(client, text, self._tools)
        if return_reasoning_steps:
            return ans, client.messages
        return ans

    # Private #
    # --------#

    def create_task(
        self,
        *,
        name: str,
        description: str,
        start_at: Optional[datetime] = None,
        deadline: Optional[datetime] = None,
        repeat: Optional[List[str]] = None,
        priority: Optional[Priority] = None,
    ) -> str:
        """
        Create a new task in the task list.

        Args:
            name (str): The name of the task.

            description (str): The description of the task.

            start_at (Optional[datetime]): The start date of the task.

            deadline (Optional[datetime]): The deadline of the task.

            repeat (Optional[List[str]]): The repeat pattern of the task.

            priority (Optional[Priority]): The priority of the task.

        Returns:
            str: The id of the new task.
        """

        # Prune None values
        task_details = {
            "name": name,
            "description": description,
            "start_at": start_at,
            "deadline": deadline,
            "repeat": repeat,
            "priority": priority,
        }

        # If it's the fist task, create immediately
        if "Tasks" not in unify.get_contexts():
            return unify.log(
                context="Tasks",
                **task_details,
                task_id=0,
                new=True,
            )

        # Verify uniqueness
        for key, value in task_details.items():
            if key not in ["name", "description"]:
                continue
            logs = unify.get_logs(
                context="Tasks",
                filter=f"{key} == '{value}'",
            )
            assert len(logs) == 0, f"Invalid, task with {key} {value} already exists."

        # ToDo: filter only for task_id once supported in the Python utility function
        logs = unify.get_logs(
            context="Tasks",
        )
        largest_id = max([lg.entries["task_id"] for lg in logs])
        this_id = largest_id + 1

        # Create the new task
        return unify.log(
            context="Tasks",
            **task_details,
            task_id=this_id,
            new=True,
        )
