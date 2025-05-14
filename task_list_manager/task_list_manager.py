import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import unify
from llm_helpers import tool_use_loop
from task_list_manager.types.status import Status
from task_list_manager.types.priority import Priority
from task_list_manager.types.schedule import Schedule
from task_list_manager.types.repetition import RepeatPattern


class TaskListManager(threading.Thread):

    def __init__(self, *, daemon: bool = True) -> None:
        """
        Responsible for managing the list of tasks, updating the names, descriptions, schedules, repeating pattern and status of all tasks.

        Args:
            daemon (bool): Whether the thread should be a daemon thread.
        """
        super().__init__(daemon=daemon)
        # ToDo: implement the tools
        self._tools = {}

    # Public #
    # -------#

    # English-Text update request

    def update(
        self,
        *,
        text: str,
        return_reasoning_steps: bool = False,
    ) -> Dict[str, str]:
        """
        Handle any plain-text english command to update the list of tasks in some manner.

        Args:
            text (str): The text-based request to update the task list.
            return_reasoning_steps (bool): Whether to return the reasoning steps for the update request.

        Returns:
            Dict[str, str]: Whether the task list was updated or not.
        """
        from task_list_manager.sys_msgs import UPDATE

        client = unify.Unify("o4-mini@openai", cache=True, traced=True)
        client.set_system_message(UPDATE)
        ans = tool_use_loop(client, text, self._tools)
        if return_reasoning_steps:
            return ans, client.messages
        return ans

    def _get_logs_by_task_ids(
        self,
        *,
        task_ids: Union[int, List[int]],
    ) -> List[unify.Log]:
        """
        Get the log for the specified task id.

        Args:
            task_ids (Union[int, List[int]]): The id or ids of the tasks to get the logs for.

        Returns:
            List[unify.Log]: The logs for the specified task ids.
        """
        singular = False
        if isinstance(task_ids, int):
            singular = True
            task_ids = [task_ids]
        log_ids = unify.get_logs(
            context="Tasks",
            filter=f"task_id in {task_ids}",
            return_ids_only=True,
        )
        assert not singular or len(log_ids) == 1
        return log_ids

    # Private #
    # --------#

    # Create

    def _create_task(
        self,
        *,
        name: str,
        description: str,
        status: Status = Status.queued,
        schedule: Optional[Schedule] = None,
        deadline: Optional[str] = None,
        repeat: Optional[List[RepeatPattern]] = None,
        priority: Priority = Priority.normal,
    ) -> int:
        """
        Create a new task in the task list.

        Args:
            name (str): The name of the task.
            description (str): The description of the task.
            status (Status): The status of the task (default: queued).
            schedule (Optional[Schedule]): The schedule information for the task.
            deadline (Optional[str]): The deadline of the task in ISO-8601 format.
            repeat (Optional[List[RepeatPattern]]): The repeat pattern of the task.
            priority (Priority): The priority of the task (default: normal).

        Returns:
            int: The id of the new task.
        """

        # Prune None values
        task_details = {
            "name": name,
            "description": description,
            "status": status,
            "schedule": schedule.model_dump() if schedule else None,
            "deadline": deadline,
            "repeat": [r.model_dump() for r in repeat] if repeat else None,
            "priority": priority,
        }

        # If it's the fist task, create immediately
        if "Tasks" not in unify.get_contexts():
            return unify.log(
                context="Tasks",
                **task_details,
                task_id=0,
                new=True,
            ).id

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
        ).id

    # Delete

    def _delete_task(
        self,
        *,
        task_id: int,
    ) -> Dict[str, str]:
        """
        Deletes the specified task from the task list.

        Args:
            task_id (int): The id of the task to delete.

        Returns:
            Dict[str, str]: Whether the task was deleted or not.
        """
        # ToDo: replace with single API call once this task [https://app.clickup.com/t/86c3c1awp] is done
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        unify.delete_logs(
            context="Tasks",
            logs=log_id,
        )

    # Pause / Continue Active Task

    def _get_paused_task(self) -> Optional[int]:
        """
        Get the currently paused task, if any.

        Returns:
            Optional[int]: The task_id of the paused task, or None if no task is paused.
        """
        paused_tasks = self._search(filter="status == 'paused'")
        assert paused_tasks <= 1, f"More than one paused task found {paused_tasks}"
        if not paused_tasks:
            return
        return paused_tasks[0]

    def _get_active_task(self) -> Optional[int]:
        """
        Get the currently active task, if any.

        Returns:
            Optional[int]: The task_id of the active task, or None if no task is active.
        """
        active_tasks = self._search(filter="status == 'active'")
        assert active_tasks <= 1, f"More than one active task found {active_tasks}"
        if not active_tasks:
            return
        return active_tasks[0]

    def _pause(self) -> Optional[Dict[str, str]]:
        """
        Pause the currently active task, if any.

        Returns:
            Optional[Dict[str, str]]: The result of updating the task status, or None if no active task.
        """
        active_task = self._get_active_task()
        if not active_task:
            return
        return self._update_task_status(task_id=active_task, new_status="paused")

    def _continue(self) -> Optional[Dict[str, str]]:
        """
        Continue the currently paused task, if any.

        Returns:
            Optional[Dict[str, str]]: The result of updating the task status, or None if no paused task.
        """
        paused_task = self._get_paused_task()
        if not paused_task:
            return
        return self._update_task_status(task_id=paused_task, new_status="active")

    # Cancel Task(s)

    def _cancel_tasks(self, task_ids: List[int]) -> None:
        """
        Cancel the specified tasks.

        Args:
            task_ids (List[int]): The ids of the tasks to cancel.
        """
        completed_tasks = self._search(filter="status == 'completed'")
        completed_task_ids = [lg["task_id"] for lg in completed_tasks]
        assert not set(task_ids).intersection(
            set(completed_task_ids),
        ), "Cannot cancel completed tasks"
        self._update_task_status(task_ids=task_ids, new_status="cancelled")

    # Update Task Queue

    def _get_task_queue():
        pass

    def _reorder_task_queue():
        pass

    def _add_task_to_queue():
        pass

    # Update Name / Description

    def _update_task_name(
        self,
        *,
        task_id: int,
        new_name: str,
    ) -> Dict[str, str]:
        """
        Update the name of the specified task.

        Args:
            task_id (int): The id of the task to update.
            new_name (str): The new name of the task.

        Returns:
            Dict[str, str]: Whether the task was updated or not.
        """
        # ToDo: replace with single API call once this task [https://app.clickup.com/t/86c3c1y63] is done
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        return unify.update_logs(
            logs=log_id,
            context="Tasks",
            entries={"name": new_name},
            overwrite=True,
        )

    def _update_task_description(
        self,
        *,
        task_id: int,
        new_description: str,
    ) -> Dict[str, str]:
        """
        Update the description for the specified task.

        Args:
            task_id (int): The id of the task to update.
            new_description (str): The new description for the task.

        Returns:
            Dict[str, str]: Whether the task was updated or not.
        """
        # ToDo: replace with single API call once this task [https://app.clickup.com/t/86c3c1y63] is done
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        return unify.update_logs(
            logs=log_id,
            context="Tasks",
            entries={"description": new_description},
            overwrite=True,
        )

    # Update Task(s) Status / Schedule / Deadline / Repetition / Priority

    def _update_task_status(
        self,
        *,
        task_ids: Union[int, List[int]],
        new_status: str,
    ) -> Dict[str, str]:
        """
        Update the status for the specified task(s).

        Args:
            task_ids (Union[int, List[int]]): The id or ids of the tasks to update.
            new_status (str): The new status for the task(s).

        Returns:
            Dict[str, str]: Whether the task(s) were updated or not.
        """
        # ToDo: replace with single API call once this task [https://app.clickup.com/t/86c3c1y63] is done
        log_ids = self._get_logs_by_task_ids(task_ids=task_ids)
        return unify.update_logs(
            logs=log_ids,
            context="Tasks",
            entries={"status": new_status},
            overwrite=True,
        )

    def _update_task_start_at(
        self,
        *,
        task_id: int,
        new_start_at: datetime,
    ) -> Dict[str, str]:
        """
        Update the start date for the specified task.

        Args:
            task_id (int): The id of the task to update.
            new_start_at (datetime): The new start date for the task.

        Returns:
            Dict[str, str]: Whether the task was updated or not.
        """
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        return unify.update_logs(
            logs=log_id,
            context="Tasks",
            entries={"start_at": new_start_at},
            overwrite=True,
        )

    def _update_task_deadline(
        self,
        *,
        task_id: int,
        new_deadline: datetime,
    ) -> Dict[str, str]:
        """
        Update the deadline for the specified task.

        Args:
            task_id (int): The id of the task to update.
            new_deadline (datetime): The new deadline for the task.

        Returns:
            Dict[str, str]: Whether the task was updated or not.
        """
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        return unify.update_logs(
            logs=log_id,
            context="Tasks",
            entries={"deadline": new_deadline},
            overwrite=True,
        )

    def _update_task_repetition(
        self,
        *,
        task_id: int,
        new_repeat: List[RepeatPattern],
    ) -> Dict[str, str]:
        """
        Update the repeat pattern for the specified task.

        Args:
            task_id (int): The id of the task to update.
            new_repeat (List[RepeatPattern]): The new repeat patterns for the task.

        Returns:
            Dict[str, str]: Whether the task was updated or not.
        """
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        return unify.update_logs(
            logs=log_id,
            context="Tasks",
            entries={"repeat": [r.model_dump() for r in new_repeat]},
            overwrite=True,
        )

    def _update_task_priority(
        self,
        *,
        task_id: int,
        new_priority: Priority,
    ) -> Dict[str, str]:
        """
        Update the priority for the specified task.

        Args:
            task_id (int): The id of the task to update.
            new_priority (Priority): The new priority for the task.

        Returns:
            Dict[str, str]: Whether the task was updated or not.
        """
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        return unify.update_logs(
            logs=log_id,
            context="Tasks",
            entries={"priority": new_priority},
            overwrite=True,
        )

    # Search Across Tasks

    def _search(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Apply the filter to the the list of tasks, and return the results following the filter.

        Args:
            filter (Optional[str]): Arbitrary Python logical expressions which evaluate to `bool`, with column names expressed as standard variables. For example, a filter expression of "'email'in description and priority == 'normal'" would be a valid. The expression just needs to be valid Python with the column names as variables.
            offset (int): The offset to start the search from, in the paginated result.
            limit (int): The number of rows to return, in the paginated result.

        Returns:
            List[Dict[str, Any]]: A list where each item in the list is a dict representing a row in the table.
        """
        return [
            log.entries
            for log in unify.get_logs(
                context="Tasks",
                filter=filter,
                offset=offset,
                limit=limit,
            )
        ]
