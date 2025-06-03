import os
import unify
import asyncio
import functools
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union

from ..common.embed_utils import EMBED_MODEL, ensure_vector_column
from ..common.llm_helpers import start_async_tool_use_loop, SteerableToolHandle
from .types.status import Status
from .types.priority import Priority
from .types.schedule import Schedule
from .types.repetition import RepeatPattern
from .types.schedule import Schedule
from .types.status import Status
from .types.task import Task
from .sys_msgs import ASK
from .base import BaseTaskListManager
import json


class TaskListManager(BaseTaskListManager):

    _VEC_TASK = "task_emb"

    def __init__(self, *, traced: bool = True) -> None:
        """
        Responsible for managing the list of tasks, updating the names, descriptions, schedules, repeating pattern and status of all tasks.

        Args:
            daemon (bool): Whether the thread should be a daemon thread.
        """

        self._ask_tools = {
            # Query-only helpers – safe, read-only operations
            self._search.__name__: self._search,
            self._search_similar.__name__: self._search_similar,
            self._get_task_queue.__name__: self._get_task_queue,
            self._get_active_task.__name__: self._get_active_task,
            self._get_paused_task.__name__: self._get_paused_task,
        }

        # Write-capable helpers – every mutating operation as well as the read-only ones.
        self._update_tools = {
            **self._ask_tools,
            # Creation / deletion
            self._create_task.__name__: self._create_task,
            self._delete_task.__name__: self._delete_task,
            # Status transitions
            self._pause.__name__: self._pause,
            self._continue.__name__: self._continue,
            self._cancel_tasks.__name__: self._cancel_tasks,
            # Queue manipulation
            self._update_task_queue.__name__: self._update_task_queue,
            # Attribute mutations
            self._update_task_name.__name__: self._update_task_name,
            self._update_task_description.__name__: self._update_task_description,
            self._update_task_status.__name__: self._update_task_status,
            self._update_task_start_at.__name__: self._update_task_start_at,
            self._update_task_deadline.__name__: self._update_task_deadline,
            self._update_task_repetition.__name__: self._update_task_repetition,
            self._update_task_priority.__name__: self._update_task_priority,
        }

        # Internal monotonically-increasing task-id counter.  We keep it local
        # to the manager to avoid an expensive scan across *all* logs every
        # time we create a task.  Initialised lazily on first use.
        self._next_id: Optional[int] = None

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a TaskListManager."
        self._ctx = f"{read_ctx}/Tasks" if read_ctx else "Tasks"

        if self._ctx not in unify.get_contexts():
            unify.create_context(self._ctx)
        # Add tracing
        if traced:
            self = unify.traced(self)

    # Public #
    # -------#

    # English-Text question

    @functools.wraps(BaseTaskListManager.ask, updated=())
    def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(
            ASK.replace(
                "{datetime}",
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            ),
        )
        # ── 0.  Optionally expose a request_clarification helper ───────────
        tools = dict(self._ask_tools)

        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                """Bubble *question* up, then wait for the answer."""
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing.")
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        # ── 1.  Kick off the tool-use loop ─────────────────────────────────
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            parent_chat_context=parent_chat_context,
            log_steps=log_tool_steps,
        )
        if _return_reasoning_steps:
            # Wrap the handle.result() to return both answer and reasoning steps
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result

        return handle

    # English-Text update request

    @functools.wraps(BaseTaskListManager.update, updated=())
    def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        from .sys_msgs import UPDATE

        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(
            UPDATE.replace(
                "{datetime}",
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            ),
        )
        # ── 0.  Offer a request_clarification helper if queues supplied ──
        tools = dict(self._update_tools)

        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                """Bubble *question* up and wait for the reply."""
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing.")
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        # ── 1.  Kick off interactive loop ─────────────────────────────────
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            parent_chat_context=parent_chat_context,
            log_steps=log_tool_steps,
        )
        if _return_reasoning_steps:
            # Wrap the handle.result() to return both answer and reasoning steps
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result

        return handle

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
            context=self._ctx,
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
                context=self._ctx,
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
        # We avoid fetching *all* logs just to know the next id.  Instead we
        # maintain a simple counter that is initialised the first time we
        # create a task in this process by looking at the *largest* existing
        # id (if any) through a single, cheap query.

        if self._next_id is None:
            # First use – find the current maximum task_id (if any) with a
            # limited query.  The stubbed SDK doesn't expose sorting, so we
            # fall back to scanning just once during initialisation which is
            # acceptable in practise.
            existing = [lg.entries.get("task_id") for lg in unify.get_logs(context=self._ctx)]  # type: ignore[arg-type]
            existing = [i for i in existing if i is not None]
            self._next_id = (max(existing) + 1) if existing else 0

        next_id = self._next_id
        self._next_id += 1

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
            context=self._ctx,
            **task_details,
            task_id=next_id,
            new=True,
        )

        # ------------------  queue insertion (if relevant)  ---------- #
        if status in (Status.active, Status.queued):
            original_q = [t.task_id for t in self._get_task_queue()]

            # Only insert if the new task isn't already in that list
            if next_id not in original_q:
                new_q = (
                    [next_id] + original_q  # prepend for active
                    if status == Status.active
                    else original_q + [next_id]  # append for queued
                )
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
            context=self._ctx,
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
                    "schedule is not None and \n                    status not in ('completed','cancelled','failed', 'scheduled') and \n                    schedule.get('prev_task') is None"
                ),
                limit=2,
            )
            if not head_candidates:
                return []
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
            if (
                cur["status"] not in self._TERMINAL_STATUSES
                and cur["status"] != "scheduled"
            ):
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

            # Only include *start_time* when we actually know one (i.e. when
            # the task was explicitly scheduled by the user).  For plain queue
            # insertions `start_ts` will be *None* and we leave the field
            # absent.
            if start_ts is not None:
                sched_payload["start_time"] = start_ts

            updates_per_log[tid] = {"schedule": sched_payload}

        # Persist
        for tid, payload in updates_per_log.items():
            log_ids = self._get_logs_by_task_ids(task_ids=tid)
            unify.update_logs(
                logs=log_ids,
                context=self._ctx,
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
            context=self._ctx,
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
            context=self._ctx,
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
            context=self._ctx,
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
        Update the scheduled **start_time** for the specified task.

        This sets / overwrites the ``schedule['start_time']`` field while
        preserving any existing ``prev_task`` / ``next_task`` linkage.
        If the task did not have a schedule previously we create one with
        ``prev_task`` / ``next_task`` set to ``None`` so that the task is
        *not* implicitly inserted into the runnable queue.
        """
        log_id = self._get_logs_by_task_ids(task_ids=task_id)

        # Coerce to ISO-8601 string (Unify stores plain serialisable values)
        if isinstance(new_start_at, datetime):
            new_start_at = new_start_at.isoformat()

        # Fetch the current task row to preserve linkage information if present
        current_rows = self._search(filter=f"task_id == {task_id}", limit=1)
        current_sched = current_rows[0].get("schedule") if current_rows else None
        if current_sched is None:
            current_sched = {}

        # Preserve queue linkage if it exists, otherwise default to None
        sched_payload = {
            "prev_task": self._sched_prev(current_sched),
            "next_task": self._sched_next(current_sched),
            "start_time": new_start_at,
        }

        return unify.update_logs(
            logs=log_id,
            context=self._ctx,
            entries={"schedule": sched_payload},
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
            context=self._ctx,
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
            context=self._ctx,
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
            context=self._ctx,
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
            context=self._ctx,
            embed_column=self._VEC_TASK,
            source_column="name_plus_desc",
            derived_expr=expr,
        )

    def _search_similar(
        self,
        *,
        text: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find tasks semantically similar to the provided text.

        Args:
            text (str): The text to find similar tasks to.
            k (int): The number of similar tasks to return.

        Returns:
            List[Dict[str, Any]]: A list where each item in the list is a dict representing a row in the table.
        """
        self._bootstrap_embeddings()
        return [
            log.entries
            for log in unify.get_logs(
                context=self._ctx,
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
                context=self._ctx,
                filter=filter,
                offset=offset,
                limit=limit,
            )
        ]
