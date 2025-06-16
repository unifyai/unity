import os
import unify
import asyncio
import functools
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from ..common.embed_utils import EMBED_MODEL, ensure_vector_column
from ..common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    methods_to_tool_dict,
)
from .types.status import Status
from .types.priority import Priority
from .types.schedule import Schedule
from .types.repetition import RepeatPattern
from .types.status import Status
from .types.task import Task
from .prompt_builders import build_ask_prompt, build_update_prompt
from .base import BaseTaskScheduler
from ..planner.base import BasePlanner
from ..planner.simulated import SimulatedPlanner
from .active_task import ActiveTask
import json


class TaskScheduler(BaseTaskScheduler):

    _VEC_TASK = "task_emb"

    def __init__(
        self,
        *,
        planner: Optional[BasePlanner] = None,
    ) -> None:
        """
        Responsible for managing the list of tasks, updating the names, descriptions, schedules, repeating pattern and status of all tasks.

        Args:
            daemon (bool): Whether the thread should be a daemon thread.
        """

        # Query-only helpers – safe, read-only operations
        self._ask_tools = methods_to_tool_dict(
            self._search_tasks,
            self._nearest_tasks,
            self._get_task_queue,
            include_class_name=False,  # redundant, all same class (this one)
        )

        # Write-capable helpers – every mutating operation as well as the read-only ones.
        self._update_tools = {
            **self._ask_tools,
            **methods_to_tool_dict(
                # Creation / deletion / cancellation
                self._create_task,
                self._delete_task,
                self._cancel_tasks,
                # Queue manipulation
                self._update_task_queue,
                # Attribute mutations
                self._update_task_name,
                self._update_task_description,
                self._update_task_status,
                self._update_task_start_at,
                self._update_task_deadline,
                self._update_task_repetition,
                self._update_task_priority,
                include_class_name=False,  # redundant, all same class (this one)
            ),
        }

        # active task
        if planner is None:
            self._planner = SimulatedPlanner(timeout=20)
        else:
            self._planner = planner

        # Internal monotonically-increasing task-id counter.  We keep it local
        # to the manager to avoid an expensive scan across *all* logs every
        # time we create a task.  Initialised lazily on first use.
        self._next_id: Optional[int] = None

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a TaskScheduler."
        self._ctx = f"{read_ctx}/Tasks" if read_ctx else "Tasks"

        if self._ctx not in unify.get_contexts():
            unify.create_context(self._ctx)

        # ID of the *single* task that is allowed to be in the **active**
        # state at any moment.  This will be maintained by a forthcoming
        # tool; until then it may legitimately stay as ``None``.
        self._active_task: Optional[Dict[str, Any]] = None
        primed_tasks = self._search_tasks(filter="status == 'primed'")
        if primed_tasks:
            assert (
                len(primed_tasks) == 1
            ), f"More than one primed task found:\n{primed_tasks}"
            self._primed_task: Optional[Dict[str, Any]] = primed_tasks[0]
        else:
            self._primed_task: Optional[Dict[str, Any]] = None

    # Public #
    # -------#

    # English-Text Question

    @functools.wraps(BaseTaskScheduler.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        # ── 0.  Build a *live* tools-dict so the prompt reflects reality ───
        tools = dict(self._ask_tools)

        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                """Bubble *question* up, then wait for the answer."""
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing.")
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        # ── 1.  Inject the dynamic system-prompt ───────────────────────────
        client.set_system_message(build_ask_prompt(tools))

        # ── 2.  Kick off the tool-use loop ────────────────────────────────
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_chat_context=parent_chat_context,
            log_steps=_log_tool_steps,
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )
        if _return_reasoning_steps:
            # Wrap the handle.result() to return both answer and reasoning steps
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result

        return handle

    # English-Text Update Request

    @functools.wraps(BaseTaskScheduler.update, updated=())
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        # ── 0.  Build a *live* tools-dict first (prompt needs it) ─────────
        tools = dict(self._update_tools)

        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                """Bubble *question* up and wait for the reply."""
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing.")
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        # ── 1.  Inject the dynamic system-prompt ──────────────────────────
        client.set_system_message(build_update_prompt(tools))

        # ── 2.  Kick off interactive loop ─────────────────────────────────
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_chat_context=parent_chat_context,
            log_steps=_log_tool_steps,
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )
        if _return_reasoning_steps:
            # Wrap the handle.result() to return both answer and reasoning steps
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result

        return handle

    # Start Task

    @functools.wraps(BaseTaskScheduler.start_task, updated=())
    async def start_task(
        self,
        task_id: int,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        # 0. sanity
        if self._active_task is not None:
            raise RuntimeError("Another task is already running – stop it first.")

        rows = self._search_tasks(filter=f"task_id == {task_id}", limit=1)
        if not rows:
            raise ValueError(f"No task found with id={task_id}")

        task_row = rows[0]
        if task_row["status"] in ("completed", "cancelled", "failed", "active"):
            raise ValueError(f"Task {task_id} is already {task_row['status']!r}.")

        # 1. build & store the ActiveTask with scheduler-awareness
        handle = ActiveTask(
            task_row["description"],
            self._planner,
            task_id=task_id,
            scheduler=self,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
        self._active_task = {"task_id": task_id, "handle": handle}

        # 2. Promote status → active and clear primed pointer if needed
        self._update_task_status(
            task_ids=task_id,
            new_status="active",
            allow_active=True,
        )
        if self._primed_task and self._primed_task["task_id"] == task_id:
            self._primed_task = None

        return handle

    # Private Helpers #
    # ----------------#

    def _validate_scheduled_invariants(
        self,
        *,
        status: Status | str,
        schedule: Optional[Union[Schedule, Dict[str, Any]]],
        err_prefix: str = "Invalid task state:",
    ) -> None:
        """
        Enforce that **Status.scheduled** is *only* legal when the task is
        (a) somewhere inside the runnable queue (`prev_task` ≠ None) **or**
        (b) has an explicit `start_at` / `start_time` timestamp.

        Args
        ----
        status
            The prospective status **after** the change.
        schedule
            The prospective schedule **after** the change (may be None).

        Raises
        ------
        ValueError
            If the rule is violated.
        """
        # normalise
        status = Status(status)

        if status != Status.scheduled:
            return

        prev_ptr = self._sched_prev(schedule)
        # model uses `.start_at`; dicts (and our existing code) use `"start_time"`
        if schedule is None:
            start_ts = None
        elif isinstance(schedule, Schedule):
            start_ts = schedule.start_at
        else:  # dict
            start_ts = schedule.get("start_at") or schedule.get("start_time")

        if prev_ptr is None and start_ts is None:
            raise ValueError(
                f"{err_prefix} a task with status 'scheduled' must have either "
                "`prev_task` (it sits behind another task in the queue) or a "
                "`start_at`/`start_time` timestamp.",
            )

    def _ensure_not_active_task(self, task_ids: Union[int, List[int]]) -> None:
        """
        Raise **RuntimeError** if *task_ids* contains the current
        ``self._active_task``.  When ``self._active_task`` is *None* the
        check is a cheap no-op.
        """
        if self._active_task is None:
            return

        if isinstance(task_ids, int):
            ids = [task_ids]
        else:
            ids = list(task_ids)

        active_task_id = self._active_task["task_id"]
        if active_task_id in ids:
            raise RuntimeError(
                f"Operation not permitted on the active task (task_id={active_task_id})",
            )

    def _get_logs_by_task_ids(
        self,
        *,
        task_ids: Union[int, List[int]],
        return_ids_only: bool = True,
    ) -> List[Union[int, unify.Log]]:
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
            return_ids_only=return_ids_only,
        )
        assert (
            not singular or len(log_ids) == 1
        ), f"Expected 1 log for singular task_id, but got {len(log_ids)}"
        return log_ids

    # Private Tools #
    # --------------#

    # Create

    def _create_task(
        self,
        *,
        name: str,
        description: str,
        status: Optional[Status] = None,
        schedule: Optional[Union[Schedule, Dict[str, Any]]] = None,
        deadline: Optional[str] = None,
        repeat: Optional[List[Union[RepeatPattern, Dict[str, Any]]]] = None,
        priority: Priority = Priority.normal,
    ) -> int:
        """
        Create a **brand-new task** and, depending on its attributes, place it
        into the appropriate queue or scheduled slot.

        Parameters
        ----------
        name : str
            Short, human-friendly label (unique across all tasks).
        description : str
            Detailed free-text explanation of what should be done.
        status : Status | None, default ``None``
            Desired initial lifecycle state.  When omitted the method infers
            one based on *schedule* and current queue status.
        schedule : Schedule | dict | None, default ``None``
            Optional explicit schedule (start-time plus linkage pointers).
            Can be either a Schedule object or a dictionary that will be converted to Schedule.
        deadline : str | None, default ``None``
            ISO-8601 timestamp (UTC) by which the task *must* be finished.
        repeat : list[RepeatPattern | dict] | None
            Zero or more recurrence rules for automatically re-instantiating
            the task. Can be either RepeatPattern objects or dictionaries that will be converted to RepeatPattern.
        priority : Priority, default :pyattr:`Priority.normal`
            Relative importance used for queue ordering.

        Returns
        -------
        int
            The **integer** ``task_id`` assigned to the new task.

        Raises
        ------
        ValueError
            On invalid field combinations or uniqueness violations.
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

        # Convert schedule dict to Schedule model if needed
        if schedule is not None and isinstance(schedule, dict):
            schedule = Schedule(**schedule)

        # Convert repeat dicts to RepeatPattern models if needed
        if repeat is not None:
            repeat = [RepeatPattern(**r) if isinstance(r, dict) else r for r in repeat]

        # figure out if schedule is "future"
        future_start = False
        if schedule and schedule.start_at:
            future_start = _parse_iso(schedule.start_at) > datetime.now(timezone.utc)

        if status is None:
            # New tasks can only ever begin their life as **scheduled** or
            # **queued**. Promotion to *active* is handled by a dedicated
            # tool that is not part of this commit.
            if future_start:
                status = Status.scheduled
            elif self._active_task is None and self._primed_task is None:
                # this goes to the top of the queue, in "primed" state
                status = Status.primed
            else:
                # this goes to be back of the queue
                status = Status.queued

        # ------------------  conflict checks  ------------------ #
        self._validate_scheduled_invariants(
            status=status,
            schedule=schedule,
            err_prefix="While creating a task:",
        )

        if status == Status.active:
            raise ValueError(
                "Tasks cannot be created directly in the 'active' state; "
                "create them as 'primed', 'queued', 'scheduled' and use the "
                "activation tool later.",
            )

        if status == Status.primed and self._active_task is not None:
            raise ValueError(
                "Tasks cannot be created in the 'primed' state when there is an 'active' task "
                "create them as 'queued' or 'scheduled', or stop the active task before setting "
                "this one as 'primed'.",
            )

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
            "task_id": next_id,
        }

        if status == Status.primed:
            self._primed_task = task_details

        # ------------------  write log immediately  ------------------ #
        unify.log(
            context=self._ctx,
            **task_details,
            new=True,
        )

        # ------------------  queue insertion (if relevant)  ---------- #
        if status == Status.queued:
            original_q = [t.task_id for t in self._get_task_queue()]

            # Only insert if the new task isn't already in that list
            if next_id not in original_q:
                new_q = original_q + [next_id]
                self._update_task_queue(original=original_q, new=new_q)

        return next_id

    # Delete

    def _delete_task(self, *, task_id: int) -> None:
        """
        Permanently **remove** a task from storage.

        Parameters
        ----------
        task_id : int
            Identifier of the task to delete.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the task is currently *active* (active tasks cannot be deleted).
        """
        self._ensure_not_active_task(task_id)
        # ToDo: replace with single API call once this task [https://app.clickup.com/t/86c3c1awp] is done
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        unify.delete_logs(
            context=self._ctx,
            logs=log_id,
        )

    # Cancel Task(s)

    def _cancel_tasks(self, task_ids: List[int]) -> None:
        """
        Mark one or many tasks as **cancelled** (non-recoverable terminal
        state).

        Parameters
        ----------
        task_ids : list[int]
            Identifiers of the tasks to cancel.

        Raises
        ------
        AssertionError
            If any referenced task is already *completed*.
        RuntimeError
            When trying to cancel the currently *active* task.
        """
        self._ensure_not_active_task(task_ids)
        completed_tasks = self._search_tasks(filter="status == 'completed'")
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

        • If *task_id* is *None* we begin with **the single active/primed task**
        • Tasks whose status is completed / cancelled / failed are *ignored*.
        • Only the nodes actually traversed are loaded from storage; we never
          materialise the entire task table in memory.
        """

        # ----------------  helpers  ---------------- #
        def _get_task_by_task_id(tid: int) -> Optional[dict]:
            """Fetch exactly one task row by id or return None."""
            rows = self._search_tasks(filter=f"task_id == {tid}", limit=1)
            return rows[0] if rows else None

        # ----------------  starting node  ---------------- #
        start_task: Optional[dict] = None

        if task_id is None:
            if self._primed_task:
                start_task = self._primed_task
                task_id = start_task["task_id"]
            else:
                raise Exception("task_id must be specified if there is no primed task.")

        if start_task is None and task_id is not None:
            start_task = _get_task_by_task_id(task_id)

        if start_task is None:
            # fall back to queue head: node with no prev_task and non-terminal status
            head_candidates = self._search_tasks(
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
            start_task = head_candidates[0]

        # not in queue yet? return list with only start task
        if start_task is not None and start_task["schedule"] is None:
            # Task exists but has no schedule pointers; therefore the
            # queue only has one item (the start task).
            return [Task(**start_task)]

        # ----------------  walk backwards to head  ---------------- #
        cur = start_task
        while True:
            prev_id = self._sched_prev(cur["schedule"])
            if prev_id is None:
                break
            prev_row = _get_task_by_task_id(prev_id)
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
            cur = _get_task_by_task_id(nxt_id)
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
        **Re-link** the runnable queue so its order matches *new*.

        Parameters
        ----------
        original : list[int]
            Snapshot of the *current* queue order.  Used for sanity checks.
        new : list[int]
            Desired queue order (may include *additional* task-ids to be
            inserted; removal is **not** permitted – cancel tasks first).

        Behaviour
        ---------
        Updates every affected task's ``schedule`` field so that the queue
        remains a well-formed doubly-linked list.  The head stores
        ``prev_task=None`` and the tail ``next_task=None``.

        Raises
        ------
        AssertionError
            On duplicates, attempted removals or other invariants breaches.
        RuntimeError
            If the active task appears in either *original* or *new*.
        """
        # The active task may **never** be reordered or touched here.
        self._ensure_not_active_task(original)
        self._ensure_not_active_task(new)
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
            t["task_id"]: t for t in self._search_tasks() if t["schedule"] is not None
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

        # ── Invariant check across the whole queue relink ────────────────────
        for tid, payload in updates_per_log.items():
            status_here = existing_logs.get(tid, {}).get("status", Status.queued)
            self._validate_scheduled_invariants(
                status=status_here,
                schedule=payload["schedule"],
                err_prefix=f"While re-ordering the queue (task {tid}):",
            )

        # Re-primed
        prime_swap_needed = False
        if self._primed_task is not None:
            orig_primed_tid = self._primed_task["task_id"]
            if orig_primed_tid in original:
                assert (
                    orig_primed_tid == original[0]
                ), "Primed task should be at the front of the queue."
                prime_swap_needed = new[0] != orig_primed_tid
        else:
            orig_primed_tid = None

        # Persist
        _task_id_to_task = dict()
        for i, (tid, payload) in enumerate(updates_per_log.items()):
            if prime_swap_needed:
                if i == 0:
                    payload = {**payload, "status": Status.primed}
                elif tid == orig_primed_tid:
                    payload = {**payload, "status": Status.queued}
            if tid == orig_primed_tid:
                self._primed_task = {**self._primed_task, **payload}
            logs = self._get_logs_by_task_ids(task_ids=tid, return_ids_only=False)
            assert len(logs) == 1, "Task IDs should be unique"
            log = logs[0]
            _task_id_to_task[tid] = log
            unify.update_logs(
                logs=log.id,
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
        Change the **name** (title) of an existing task.

        Parameters
        ----------
        task_id : int
            Identifier of the task to rename.
        new_name : str
            New unique name.

        Returns
        -------
        dict[str, str]
            Confirmation payload from :pyfunc:`unify.update_logs`.
        """
        self._ensure_not_active_task(task_id)
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
        Replace the **description** of an existing task.

        Parameters
        ----------
        task_id : int
            Identifier of the task to modify.
        new_description : str
            Fresh free-text description (no length limit, Markdown allowed).

        Returns
        -------
        dict[str, str]
            Confirmation payload as returned by :pyfunc:`unify.update_logs`.

        Raises
        ------
        RuntimeError
            If the referenced task is currently *active* – active tasks are
            immutable from the scheduler's perspective.
        """
        self._ensure_not_active_task(task_id)
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
        allow_active: bool = False,
    ) -> Dict[str, str]:
        """
        Change the **lifecycle status** of one or many tasks.

        Parameters
        ----------
        task_ids : int | list[int]
            One or multiple task identifiers to update.
        new_status : str
            Target status value.  Must be a valid member of
            :class:`~task_scheduler.types.status.Status`.
        allow_active : bool, default ``False``
            Guard-rail – when *False* the method refuses to set the status to
            ``'active'`` or to touch the *currently* active task.  Internal
            helpers (e.g. *start_task*) pass *True* when they *really* need to.

        Returns
        -------
        dict[str, str]
            Confirmation object from :pyfunc:`unify.update_logs`.

        Raises
        ------
        ValueError
            If *new_status* is ``'active'`` while *allow_active* is ``False``.
        RuntimeError
            When trying to edit the live active task without permission.
        """
        # 1. Forbid making anything *active* (unless explicitly allowed)
        if str(new_status) == Status.active.value and not allow_active:
            raise ValueError(
                "Direct status changes to 'active' are not allowed; "
                "use the dedicated activation tool.",
            )

        # 2. Forbid touching the existing active task
        if not allow_active:
            self._ensure_not_active_task(task_ids)

        # ── Invariant check *per task* if new_status becomes 'scheduled' ─────
        if str(new_status) == Status.scheduled.value:
            rows = self._search_tasks(filter=f"task_id in {task_ids}")
            for row in rows:
                self._validate_scheduled_invariants(
                    status=new_status,
                    schedule=row.get("schedule"),
                    err_prefix=f"While changing status of task {row['task_id']}:",
                )

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
        Set or change a task's **scheduled start-time** (UTC).

        Parameters
        ----------
        task_id : int
            Identifier of the task to reschedule.
        new_start_at : datetime
            Exact moment the task becomes *eligible* for activation.  A naive
            datetime is assumed to be UTC; otherwise the value is preserved
            verbatim.

        Returns
        -------
        dict[str, str]
            Confirmation payload from :pyfunc:`unify.update_logs`.

        Notes
        -----
        * The method **preserves** any existing queue linkage
          (``prev_task`` / ``next_task``).
        * When the task previously had *no* schedule, a minimal one is
          created with linkage fields set to ``None`` (task is *not*
          inserted into the runnable queue automatically).
        """
        self._ensure_not_active_task(task_id)
        log_id = self._get_logs_by_task_ids(task_ids=task_id)

        # Coerce to ISO-8601 string (Unify stores plain serialisable values)
        if isinstance(new_start_at, datetime):
            new_start_at = new_start_at.isoformat()

        # Fetch the current task row to preserve linkage information if present
        current_rows = self._search_tasks(filter=f"task_id == {task_id}", limit=1)
        current_sched = current_rows[0].get("schedule") if current_rows else None
        if current_sched is None:
            current_sched = {}

        # Preserve queue linkage if it exists, otherwise default to None
        sched_payload = {
            "prev_task": self._sched_prev(current_sched),
            "next_task": self._sched_next(current_sched),
            "start_time": new_start_at,
        }

        # ensure the new schedule does not violate the invariant
        self._validate_scheduled_invariants(
            status=current_rows[0]["status"],
            schedule=sched_payload,
            err_prefix=f"While updating start_time for task {task_id}:",
        )

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
        Adjust a task's **hard deadline** (UTC ISO-8601 timestamp).

        Parameters
        ----------
        task_id : int
            Task identifier.
        new_deadline : datetime
            Absolute "must-finish-by" moment.  Naive datetimes are coerced to
            UTC; timezone-aware values are stored unchanged.

        Returns
        -------
        dict[str, str]
            Confirmation from :pyfunc:`unify.update_logs`.
        """
        self._ensure_not_active_task(task_id)
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
        Replace the **recurrence rules** associated with a task.

        Parameters
        ----------
        task_id : int
            Identifier of the task to modify.
        new_repeat : list[RepeatPattern]
            Complete list of replacement recurrence definitions.  Pass an
            empty list to *disable* repetition.

        Returns
        -------
        dict[str, str]
            Confirmation payload from :pyfunc:`unify.update_logs`.
        """
        self._ensure_not_active_task(task_id)
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
        Set a task's **priority** (relative importance cue for queueing).

        Parameters
        ----------
        task_id : int
            Task identifier.
        new_priority : Priority
            One of the enumeration values from
            :class:`~task_scheduler.types.priority.Priority`.

        Returns
        -------
        dict[str, str]
            Confirmation payload from :pyfunc:`unify.update_logs`.
        """
        self._ensure_not_active_task(task_id)
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

    def _nearest_tasks(
        self,
        *,
        text: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return the **k** tasks whose *name + description* embeddings are
        *closest* (cosine distance) to the supplied *text*.

        Parameters
        ----------
        text : str
            Query text from which to derive the embedding vector.
        k : int, default ``5``
            Number of neighbours to return.

        Returns
        -------
        list[dict]
            Log-entry dictionaries of the closest tasks (ascending distance).
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

    def _search_tasks(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Run a **column-wise Python expression** (`filter`) against every task
        and return the matching rows.

        Parameters
        ----------
        filter : str | None, default ``None``
            Any valid Python boolean expression referencing column names as
            variables, e.g. ``"status == 'queued' and priority == 'high'"``.
            *None* selects **all** tasks.
        offset : int, default ``0``
            Zero-based row offset for pagination.
        limit : int, default ``100``
            Maximum number of rows to return.

        Returns
        -------
        list[dict]
            Entries for each matching task (raw JSON-serialisable dictionaries).
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
