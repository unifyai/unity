import threading
from typing import Dict

import unify
from llm_helpers import tool_use_loop


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
