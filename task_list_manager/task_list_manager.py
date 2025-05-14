import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import unify

from common.embed_utils import EMBED_MODEL, ensure_vector_column
from common.llm_helpers import tool_use_loop
from task_list_manager.types.status import Status
from task_list_manager.types.priority import Priority
from task_list_manager.types.schedule import Schedule
from task_list_manager.types.repetition import RepeatPattern
from task_list_manager.types.schedule import Schedule
from task_list_manager.types.status import Status
from task_list_manager.types.task import Task


class TaskListManager(threading.Thread):

    _VEC_TASK = "task_emb"

    def __init__(self, *, daemon: bool = True) -> None:
        """
        Responsible for managing the list of tasks, updating the names, descriptions, schedules, repeating pattern and status of all tasks.

        Args:
            daemon (bool): Whether the thread should be a daemon thread.
        """
        super().__init__(daemon=daemon)
        # ToDo: implement the tools
        self._tools = {"search_similar": self._search_similar}

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
        assert (
            not singular or len(log_ids) == 1
        ), f"Expected 1 log for singular task_id, but got {len(log_ids)}"
        return log_ids

    # Private #
    # --------#

    # Create

    def _create_task(
        self,
        *,
        name: str,
        description: str,
        status: Optional[Status] = None,
        schedule: Optional[Schedule] = None,
        deadline: Optional[str] = None,
        repeat: Optional[List[RepeatPattern]] = None,
        priority: Priority = Priority.normal,
    ) -> int:
        """
        Create a new task and – if appropriate – insert it into the
        runnable queue.

        Behaviour
        ---------
        • If *status* is **omitted** the method decides:
            – **active**     when no active task exists *and* either
                             no schedule is given or its start_time ≤ now.
            – **queued**     when an active task already exists.
            – **scheduled**  when a schedule.start_time > now.

        • If the caller supplies an explicit *status* we validate that it
          does not conflict with the current state (e.g. no 2nd active).

        • New queued / active tasks are appended (tail) or prepended (head)
          by re-using `_update_task_queue`.

        • Tasks whose `start_time` is in the future are **not** placed
          in the active queue.

        Raises
        ------
        ValueError for invalid combinations or duplicate name/description.
        """
        # ----------------  helper: iso-8601 → datetime  ---------------- #
        from datetime import datetime, timezone

        def _parse_iso(ts: str) -> datetime:
            dt = datetime.fromisoformat(ts)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        # ----------------  initial validation & dedup  ---------------- #
        if not name or not description:
            raise ValueError("Both 'name' and 'description' are required")

        # uniqueness (name / description)
        for key, value in {"name": name, "description": description}.items():
            clashes = unify.get_logs(
                context="Tasks",
                filter=f"{key} == '{value}'",
                limit=1,
            )
            if clashes:
                raise ValueError(f"A task with {key!r} = {value!r} already exists")

        # ----------------------------------- #
        #  derive status when caller omitted   #
        # ----------------------------------- #
        if status is not None and isinstance(status, str):
            status = Status(status)

        active_task = self._get_active_task()

        # figure out if schedule is "future"
        future_start = False
        if schedule and schedule.start_time:
            future_start = _parse_iso(schedule.start_time) > datetime.now(timezone.utc)

        if status is None:
            if future_start:
                status = Status.scheduled
            elif active_task is None:
                status = Status.active
            else:
                status = Status.queued

        # ------------------  conflict checks  ------------------ #
        if status == Status.active and active_task is not None:
            raise ValueError("An active task already exists")

        if status == Status.active and future_start:
            raise ValueError("Cannot mark task as active with a future start_time")

        if status == Status.scheduled and not future_start:
            raise ValueError("Scheduled tasks require a future start_time")

        # ------------------  generate new task_id  ------------------ #
        if "Tasks" not in unify.get_contexts():
            next_id = 0
        else:
            all_ids = [lg.entries["task_id"] for lg in unify.get_logs(context="Tasks")]
            next_id = (max(all_ids) + 1) if all_ids else 0

        # ------------------  assemble payload  ------------------ #
        task_details = {
            "name": name,
            "description": description,
            "status": status,
            "schedule": schedule.model_dump() if schedule else None,
            "deadline": deadline,
            "repeat": [r.model_dump() for r in repeat] if repeat else None,
            "priority": priority,
        }

        # ------------------  write log immediately  ------------------ #
        unify.log(
            context="Tasks",
            **task_details,
            task_id=next_id,
            new=True,
        )

        # ------------------  queue insertion (if relevant)  ------------------ #
        if status in (Status.active, Status.queued):
            original_q = [t.task_id for t in self._get_task_queue()]
            new_q = (
                [next_id] + original_q
                if status == Status.active
                else original_q + [next_id]
            )
            # updates prev/next pointers for every node (incl. new one)
            self._update_task_queue(original=original_q, new=new_q)

        return next_id

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

    def _get_paused_task(self) -> Optional[Task]:
        """
        Get the currently paused task, if any.

        Returns:
            Optional[Task]: The complete Task object of the paused task, or None if no task is paused.
        """
        paused_tasks = self._search(filter="status == 'paused'")
        assert (
            len(paused_tasks) <= 1
        ), f"More than one paused task found: {paused_tasks}"
        if not paused_tasks:
            return
        return paused_tasks[0]

    def _get_active_task(self) -> Optional[Task]:
        """
        Get the currently active task, if any.

        Returns:
            Optional[Task]: The complete Task object of the active task, or None if no task is active.
        """
        active_tasks = self._search(filter="status == 'active'")
        assert (
            len(active_tasks) <= 1
        ), f"More than one active task found: {active_tasks}"
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
        return self._update_task_status(
            task_ids=active_task["task_id"],
            new_status="paused",
        )

    def _continue(self) -> Optional[Dict[str, str]]:
        """
        Continue the currently paused task, if any.

        Returns:
            Optional[Dict[str, str]]: The result of updating the task status, or None if no paused task.
        """
        paused_task = self._get_paused_task()
        if not paused_task:
            return
        return self._update_task_status(
            task_ids=paused_task["task_id"],
            new_status="active",
        )

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
        ), f"Cannot cancel completed tasks. Attempted to cancel: {set(task_ids).intersection(set(completed_task_ids))}"
        self._update_task_status(task_ids=task_ids, new_status="cancelled")

    # Update Task Queue

    # --------------------  small helpers  -------------------- #
    @staticmethod
    def _sched_prev(sched):
        """Return *prev_task* from a Schedule *dict* / *model* / *None*."""
        if sched is None:
            return None
        if isinstance(sched, dict):
            return sched.get("prev_task")
        # assume pydantic Schedule
        return getattr(sched, "prev_task", None)

    @staticmethod
    def _sched_next(sched):
        """Return *next_task* (mirrors _sched_prev)."""
        if sched is None:
            return None
        if isinstance(sched, dict):
            return sched.get("next_task")
        return getattr(sched, "next_task", None)

    _TERMINAL_STATUSES = {"completed", "cancelled", "failed"}

    def _get_task_queue(
        self,
        task_id: Optional[int] = None,
    ) -> List[Task]:
        """
        Return the runnable task queue (head → tail).

        • If *task_id* is *None* we begin with **the single active task**
          (falling back to the queue head if there is no active task).
        • Tasks whose status is completed / cancelled / failed are *ignored*.
        • Only the nodes actually traversed are loaded from storage; we never
          materialise the entire task table in memory.
        """

        # ----------------  helpers  ---------------- #
        def _get_task_row(tid: int) -> Optional[dict]:
            """Fetch exactly one task row by id or return None."""
            rows = self._search(filter=f"task_id == {tid}", limit=1)
            return rows[0] if rows else None

        # ----------------  starting node  ---------------- #
        start_row: Optional[dict] = None

        if task_id is None:
            active = self._get_active_task()
            if active:
                start_row = active
                task_id = active["task_id"]

        if start_row is None and task_id is not None:
            start_row = _get_task_row(task_id)

        if start_row is None:
            # fall back to queue head: node with no prev_task and non-terminal status
            head_candidates = self._search(
                filter=(
                    "schedule is not None and "
                    "status not in ('completed','cancelled','failed') and "
                    "schedule.get('prev_task') is None"
                ),
                limit=2,
            )
            assert head_candidates, f"Queue is malformed – no head found"
            assert (
                len(head_candidates) == 1
            ), f"Multiple heads detected: {head_candidates}"
            start_row = head_candidates[0]

        # ----------  not in queue yet? return empty list  ---------- #
        if start_row is not None and start_row["schedule"] is None:
            # Task exists but has no schedule pointers; therefore the
            # queue is currently empty.
            return []

        # ----------------  walk backwards to head  ---------------- #
        cur = start_row
        while True:
            prev_id = self._sched_prev(cur["schedule"])
            if prev_id is None:
                break
            prev_row = _get_task_row(prev_id)
            if prev_row is None:
                break  # broken link – treat cur as head
            cur = prev_row  # keep walking

        head_row = cur

        # ----------------  walk forwards collecting list  ---------------- #
        ordered: List[Task] = []
        cur = head_row
        while cur:
            if cur["status"] not in self._TERMINAL_STATUSES:
                ordered.append(Task(**cur))

            nxt_id = self._sched_next(cur["schedule"])
            if nxt_id is None:
                break

            # fetch the next node lazily
            cur = _get_task_row(nxt_id)
            # guard against broken links (missing row)
            if cur is None:
                break

        return ordered

    def _update_task_queue(
        self,
        *,
        original: List[int],
        new: List[int],
    ) -> None:
        """
        Re-write the queue so that its order matches the new order.

        Args:
            original (List[int]): The current queue order, used for validation.
            new (List[int]): The new queue order, which may include extra task IDs.

        * `original` must describe the *current* queue order; we use it
          only for validation.
        * `new` may be a pure re-ordering or may also include **extra**
          task-ids (inserting new tasks). Removing tasks is *not*
          allowed here – cancel them instead.

        For every task we update its ``schedule`` field so that the linked
        list stays consistent, using `None` for the head's `prev_task`
        and the tail's `next_task`.
        """
        # -------  sanity checks  -------
        assert len(set(original)) == len(
            original,
        ), f"'original' contains duplicates: {original}"
        assert len(set(new)) == len(new), f"'new' contains duplicates: {new}"
        assert set(original).issubset(
            set(new),
        ), f"update cannot remove existing tasks; cancel them first. Missing tasks: {set(original) - set(new)}"

        # -------  gather existing logs  -------
        existing_logs = {
            t["task_id"]: t for t in self._search() if t["schedule"] is not None
        }

        updates_per_log: Dict[int, Dict[str, Any]] = {}
        for idx, tid in enumerate(new):
            prev_tid = None if idx == 0 else new[idx - 1]
            next_tid = None if idx == len(new) - 1 else new[idx + 1]

            # keep an existing start_time; otherwise leave it unset
            start_ts = None
            if tid in existing_logs and existing_logs[tid]["schedule"]:
                start_ts = existing_logs[tid]["schedule"].get("start_time")

            sched_payload = {
                "prev_task": prev_tid,
                "next_task": next_tid,
            }
            if start_ts is not None:
                sched_payload["start_time"] = start_ts

            updates_per_log[tid] = {"schedule": sched_payload}

        # Persist
        for tid, payload in updates_per_log.items():
            log_ids = self._get_logs_by_task_ids(task_ids=tid)
            unify.update_logs(
                logs=log_ids,
                context="Tasks",
                entries=payload,
                overwrite=True,
            )

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

    def _bootstrap_embeddings(self) -> None:
        """
        Ensure that the vector embedding column exists for task search.
        Creates a derived column combining name and description for embedding.
        """
        expr = "str({name}) + ' || ' + str({description})"
        ensure_vector_column(
            "Tasks",
            self._VEC_TASK,
            derived_column="name_plus_desc",
            derived_expr=expr,
        )

    def _search_similar(
        self,
        *,
        text: str,
        k: int = 5,
    ) -> List[Task]:
        """
        Find tasks semantically similar to the provided text.

        Args:
            text (str): The text to find similar tasks to.
            k (int): The number of similar tasks to return.

        Returns:
            List[Task]: A list of Task objects similar to the provided text.
        """
        self._bootstrap_embeddings()
        return [
            log.entries
            for log in unify.get_logs(
                context="Tasks",
                sorting={
                    f"cosine({self._VEC_TASK}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
                },
                limit=k,
            )
        ]

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
