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
from ..common.tool_outcome import ToolOutcome
from .types.status import Status
from .types.priority import Priority
from .types.schedule import Schedule
from .types.trigger import Trigger
from .types.repetition import RepeatPattern
from .types.task import Task

# Contact manager import (lazy at module level to avoid cycles in other modules)
from ..contact_manager.contact_manager import ContactManager
from ..common.model_to_fields import model_to_fields
from .prompt_builders import build_ask_prompt, build_update_prompt
from .base import BaseTaskScheduler
from ..planner.base import BasePlanner
from ..planner.simulated import SimulatedPlanner
from .active_task import ActiveTask
import json

from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)


class TaskScheduler(BaseTaskScheduler):

    _VEC_TASK = "_task_emb"

    _HEAD_FILTER = (
        "schedule is not None and "
        "status not in ('completed','cancelled','failed') and "
        "schedule.get('prev_task') is None"
    )

    def __init__(
        self,
        *,
        planner: Optional[BasePlanner] = None,
        rolling_summary_in_prompts: bool = True,
    ) -> None:
        """
        Responsible for managing the list of tasks, updating the names, descriptions, schedules, repeating pattern and status of all tasks.

        Args:
            daemon (bool): Whether the thread should be a daemon thread.
        """

        # Instantiate a ContactManager once so its bound methods can act as tools
        self._contact_manager = ContactManager()

        # Query-only helpers – safe, read-only operations.  Include the *external* contact lookup
        self._ask_tools = {
            **methods_to_tool_dict(
                self._search_tasks,
                self._nearest_tasks,
                self._get_task_queue,
                include_class_name=False,  # redundant, all same class (this one)
            ),
            **methods_to_tool_dict(
                self._contact_manager.ask,
                include_class_name=True,  # Retain originating class so name is ContactManager.ask
            ),
        }

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
                self._update_task_trigger,
                include_class_name=False,  # redundant, all same class (this one)
            ),
        }

        # active task
        if planner is None:
            self._planner = SimulatedPlanner(timeout=20)
        else:
            self._planner = planner

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a TaskScheduler."
        self._ctx = f"{read_ctx}/Tasks" if read_ctx else "Tasks"

        if self._ctx not in unify.get_contexts():
            unify.create_context(
                self._ctx,
                unique_column_ids=["task_id", "instance_id"],
                description=(
                    "List of all tasks with their name, description, status, "
                    "schedule, deadline, repeat pattern, priority **and** "
                    "`instance_id` which tracks multiple executions of the "
                    "same logical task."
                ),
            )
            fields = model_to_fields(Task)
            unify.create_fields(
                fields,
                context=self._ctx,
            )

        # ID of the *single* task that is allowed to be in the **active**
        # state at any moment.  This will be maintained by a forthcoming
        # tool; until then it may legitimately stay as ``None``.
        # {'task_id': int, 'instance_id': int, 'handle': ActiveTask}
        self._active_task: Optional[Dict[str, Any]] = None
        primed_tasks = self._search_tasks(filter="status == 'primed'")
        if primed_tasks:
            assert (
                len(primed_tasks) == 1
            ), f"More than one primed task found:\n{primed_tasks}"
            self._primed_task: Optional[Dict[str, Any]] = primed_tasks[0]
        else:
            self._primed_task: Optional[Dict[str, Any]] = None

        self._rolling_summary_in_prompts = rolling_summary_in_prompts

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
        rolling_summary_in_prompts: Optional[bool] = None,
    ) -> SteerableToolHandle:
        call_id = new_call_id()
        await publish_manager_method_event(
            call_id,
            "TaskScheduler",
            "ask",
            phase="incoming",
            question=text,
        )

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
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        client.set_system_message(
            build_ask_prompt(tools, include_activity=include_activity),
        )

        # ── 2.  Kick off the tool-use loop ────────────────────────────────
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_chat_context=parent_chat_context,
            log_steps=_log_tool_steps,
            preprocess_msgs=self._inject_broader_context,
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )
        # ── 3a.  Add logging wrapper ──────────────────────────────────────
        handle = wrap_handle_with_logging(
            handle,
            call_id,
            "TaskScheduler",
            "ask",
        )

        # ── 3b.  Optional reasoning exposure ─────────────────────────────
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
        rolling_summary_in_prompts: Optional[bool] = None,
    ) -> SteerableToolHandle:
        call_id = new_call_id()
        await publish_manager_method_event(
            call_id,
            "TaskScheduler",
            "update",
            phase="incoming",
            request=text,
        )

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
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_update_prompt(tools, include_activity=include_activity),
        )

        # ── 2.  Kick off interactive loop ─────────────────────────────────
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_chat_context=parent_chat_context,
            log_steps=_log_tool_steps,
            preprocess_msgs=self._inject_broader_context,
            tool_policy=lambda i, _: (
                ("required", self._ask_tools) if i < 1 else ("auto", _)
            ),
        )
        # ── 3a.  Add logging wrapper ──────────────────────────────────────
        handle = wrap_handle_with_logging(
            handle,
            call_id,
            "TaskScheduler",
            "update",
        )

        # ── 3b.  Optional reasoning exposure ─────────────────────────────
        if _return_reasoning_steps:
            # Wrap the handle.result() to return both answer and reasoning steps
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result

        return handle

    # Start Task

    @functools.wraps(BaseTaskScheduler.execute_task, updated=())
    async def execute_task(
        self,
        text: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        """Execute a task from **free-form** textual input.

        The method launches a *private* async tool-use loop with two tools:

        • ``ask`` – leverage the normal question-answer helper to identify the
          numeric `task_id` (when not explicitly present).
        • ``execute_task_by_id`` – thin wrapper around the internal
          :py:meth:`_execute_task_internal` helper which returns the real
          :class:`~unify.common.llm_helpers.SteerableToolHandle` **and sets the
          pass-through flag** so the outer handle upgrades seamlessly.
        """

        freeform_text: str = text

        call_id = new_call_id()
        await publish_manager_method_event(
            call_id,
            "TaskScheduler",
            "execute_task",
            phase="incoming",
            request=freeform_text,
        )

        import json, inspect

        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        # ── tool definitions ────────────────────────────────────────────────
        async def _execute_task_by_id(*, task_id: int) -> SteerableToolHandle:  # type: ignore[valid-type]
            """Start the task with *task_id* and bubble up its handle (passthrough)."""

            handle = await self._execute_task_internal(
                task_id=task_id,
                parent_chat_context=parent_chat_context,
                clarification_up_q=clarification_up_q,
                clarification_down_q=clarification_down_q,
            )
            # 💡 signal pass-through so the outer loop adopts this handle
            setattr(handle, "__passthrough__", True)
            return handle

        async def request_clarification(question: str) -> str:  # type: ignore[valid-type]
            """Bubble *question* up to the caller and await the answer.

            When *clarification_up_q* or *clarification_down_q* are *None* the
            tool raises **RuntimeError** so the LLM avoids using it in
            non-interactive contexts.
            """

            if clarification_up_q is None or clarification_down_q is None:
                raise RuntimeError(
                    "Clarification queues not supplied – cannot request clarification in this context.",
                )

            await clarification_up_q.put(question)
            return await clarification_down_q.get()

        tools = {
            "ask": self.ask,  # determine an existing/created task id
            "update": self.update,  # create a brand-new task or tweak an existing one
            "request_clarification": request_clarification,  # human clarification channel
            "execute_task_by_id": _execute_task_by_id,  # finally start the task
        }

        # ── dynamic system prompt ───────────────────────────────────────────
        sig_json = json.dumps(
            {n: str(inspect.signature(fn)) for n, fn in tools.items()},
            indent=4,
        )

        prompt = "\n".join(
            [
                "You are an assistant that starts tasks on demand.",
                "1. If the user instruction already contains the numeric task id, call `execute_task_by_id` directly.",
                "2. Otherwise, call `ask` to figure out the task id from the description, then call `execute_task_by_id` with that id.",
                "Respond *only* with tool calls until `execute_task_by_id` has been invoked. After it returns you may reply DONE.",
                "",
                "Tools (name → argspec):",
                sig_json,
                "",
            ],
        )

        client.set_system_message(prompt)

        outer_handle = start_async_tool_use_loop(
            client,
            freeform_text,
            tools,
            loop_id=f"{self.__class__.__name__}.execute_task_resolver",
            parent_chat_context=parent_chat_context,
            log_steps=True,
            preprocess_msgs=self._inject_broader_context,
        )

        # Wire up event logging wrapper – outer_handle is an AsyncToolUseLoopHandle
        outer_handle = wrap_handle_with_logging(
            outer_handle,
            call_id,
            "TaskScheduler",
            "execute_task",
        )

        return outer_handle

    # ------------------------------------------------------------------ #
    #  Internal helper – run existing *by-id* logic without event logging   #
    # ------------------------------------------------------------------ #

    async def _execute_task_internal(
        self,
        *,
        task_id: int,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        """The original *execute_task* implementation (minus event logging).

        Separated so that both the public ``execute_task`` method **and** the
        internally exposed ``execute_task_by_id`` tool can delegate to the same
        core logic **without** duplicating ManagerMethod events.
        """

        # 0. sanity checks
        if self._active_task is not None:
            raise RuntimeError("Another task is already running – stop it first.")

        candidate_rows = self._search_tasks(
            filter=(
                f"task_id == {task_id} and status not in "
                "('completed','cancelled','failed','active')"
            ),
        )
        if not candidate_rows:
            raise ValueError(f"No runnable task found with id={task_id}")

        # Pick the *oldest* runnable instance (lowest instance_id)
        task_row = sorted(
            candidate_rows,
            key=lambda r: r.get("instance_id", 0),
        )[0]
        if task_row["status"] in ("completed", "cancelled", "failed", "active"):
            raise ValueError(f"Task {task_id} is already {task_row['status']!r}.")

        # 1. build the *real* active plan
        plan_handle = await self._planner.execute(
            task_row["description"],
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        # 2. wrap it so we can keep the task-table in sync
        handle = ActiveTask(
            plan_handle,
            task_id=task_id,
            instance_id=task_row["instance_id"],
            scheduler=self,
        )

        self._active_task = {
            "task_id": task_id,
            "instance_id": task_row["instance_id"],
            "handle": handle,
        }

        # ── clone if this is a triggerable or recurring task ──────────────
        if task_row["status"] == Status.triggerable.value or task_row.get("repeat"):
            self._clone_task_instance(task_row)

        # 3. Promote status → active and clear primed pointer if needed
        self._update_task_status_instance(
            task_id=task_id,
            instance_id=task_row["instance_id"],
            new_status="active",
        )
        if self._primed_task and self._primed_task["task_id"] == task_id:
            self._primed_task = None

        return handle

    #  Per-instance helpers

    def _update_task_status_instance(
        self,
        *,
        task_id: int,
        instance_id: int,
        new_status: str,
    ) -> Dict[str, str]:
        """
        Same semantics as `_update_task_status` but scoped to a single
        **(task_id, instance_id)** pair.
        """
        log_objs = unify.get_logs(
            context=self._ctx,
            filter=f"task_id == {task_id} and instance_id == {instance_id}",
            return_ids_only=False,
        )
        if not log_objs:
            raise ValueError(
                f"No task instance ({task_id}.{instance_id}) found.",
            )
        assert len(log_objs) == 1, "Composite primary key must be unique."
        return unify.update_logs(
            logs=log_objs[0].id if hasattr(log_objs[0], "id") else log_objs[0],
            context=self._ctx,
            entries={"status": new_status},
            overwrite=True,
        )

    def _clone_task_instance(self, task_row: Dict[str, Any]) -> None:
        """
        Create a *fresh* row for the **next** instance of a triggerable or
        recurring task.  We copy every user-facing field, keep the *same*
        `task_id`, intentionally **omit** `instance_id` (so the backend
        auto-increments it) and leave the status unchanged (*triggerable*
        or *scheduled*).
        """
        allowed = set(Task.model_json_schema()["properties"].keys())
        clone_payload = {
            k: v for k, v in task_row.items() if k in allowed and k != "instance_id"
        }
        # Drop any internal bookkeeping injected by Unify (_id, _log_id …)
        unify.log(
            context=self._ctx,
            new=True,
            **clone_payload,
        )

    # Private Helpers #
    # ----------------#

    def _validate_scheduled_invariants(
        self,
        *,
        status: Status | str,
        schedule: Optional[Union[Schedule, Dict[str, Any]]],
        trigger: Optional[Union["Trigger", Dict[str, Any]]] = None,
        err_prefix: str = "Invalid task state:",
    ) -> None:
        """
        Enforce that **Status.scheduled** is *only* legal when the task is
        (a) somewhere inside the runnable queue (`prev_task` ≠ None) **or**
        (b) has an explicit `start_at` timestamp.

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
        # ── Trigger-based tasks are **not** subject to the schedule rules ──
        if trigger is not None:
            return

        # normalise
        status = Status(status)

        prev_ptr = self._sched_prev(schedule)
        if schedule is None:
            start_ts = None
        elif isinstance(schedule, Schedule):
            start_ts = schedule.start_at
        else:  # dict
            start_ts = schedule.get("start_at")

        # ── Invariant – queue-head tasks with an explicit start_at must be 'scheduled' ──
        if status == Status.queued and prev_ptr is None and start_ts is not None:
            raise ValueError(
                f"{err_prefix} tasks at the head of the queue that define 'start_at' must have status 'scheduled', not 'queued'.",
            )

        if prev_ptr is not None and start_ts is not None:
            raise ValueError(
                f"{err_prefix} a task cannot define both 'prev_task' and "
                "'start_at' – the timestamp belongs on the queue head only.",
            )

        # ── Invariant #2 – 'primed' must always be the queue head ───────────
        if Status(status) == Status.primed and prev_ptr is not None:
            raise ValueError(
                f"{err_prefix} a task in 'primed' state must be at the head of the queue (prev_task must be None).",
            )

        if status != Status.scheduled:
            return

        if prev_ptr is None and start_ts is None:
            raise ValueError(
                f"{err_prefix} a task with status 'scheduled' must have either "
                "`prev_task` (it sits behind another task in the queue) or a "
                "`start_at` timestamp.",
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
        trigger: Optional[Union[Trigger, Dict[str, Any]]] = None,
        deadline: Optional[str] = None,
        repeat: Optional[List[Union[RepeatPattern, Dict[str, Any]]]] = None,
        priority: Priority = Priority.normal,
    ) -> ToolOutcome:
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
        ToolOutcome
            Tool outcome with any extra relevant details.

        Raises
        ------
        ValueError
            On invalid field combinations or uniqueness violations.
        """
        # ----------------  helper: iso-8601 → datetime  ---------------- #
        from datetime import datetime, timezone

        def _parse_maybe_iso(ts: str) -> datetime:
            dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(ts)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        # ----------------  initial validation & dedup  ---------------- #
        if not name or not description:
            raise ValueError("Both 'name' and 'description' are required")

        # uniqueness (name / description)
        # Escape *value* via ``repr`` so that any internal quotes (like apostrophes)
        # do **not** break the filter expression.  Using ``!r`` ensures that we
        # always generate a *valid* Python string literal regardless of the
        # characters contained in *value*.
        for key, value in {"name": name, "description": description}.items():
            clashes = unify.get_logs(
                context=self._ctx,
                filter=f"{key} == {value!r}",
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

        # Convert trigger / repeat dicts to strong models if needed
        if trigger is not None and isinstance(trigger, dict):
            trigger = Trigger(**trigger)

        if repeat is not None:
            repeat = [RepeatPattern(**r) if isinstance(r, dict) else r for r in repeat]

        #  Trigger / schedule exclusivity
        if schedule is not None and trigger is not None:
            raise ValueError("`schedule` and `trigger` are mutually exclusive.")

        # figure out if schedule is "future"
        future_start = False  # ← only meaningful for *schedule*-tasks
        if schedule and schedule.start_at:
            future_start = _parse_maybe_iso(schedule.start_at) > datetime.now(
                timezone.utc,
            )

        #  If the task is explicitly linked **behind**  another task (prev_task ≠ None)
        # and that task is not terminal, we NEVER mark the newcomer as *primed*.
        prev_ptr = self._sched_prev(schedule)

        if trigger is not None:
            # --------  event-driven task  -------- #
            if status is None:
                status = Status.triggerable
            elif Status(status) != Status.triggerable:
                raise ValueError(
                    "Tasks with a *trigger* must start in the 'triggerable' state.",
                )

        elif status is None:
            if prev_ptr is not None:
                # Already queued behind another runnable task → never primed
                status = Status.scheduled if future_start else Status.queued
            else:
                # No predecessor pointer – use the old heuristic
                if future_start:
                    status = Status.scheduled
                elif self._active_task is None and self._primed_task is None:
                    status = Status.primed
                else:
                    status = Status.queued

        # ------------------  conflict checks  ------------------ #
        self._validate_scheduled_invariants(
            status=status,
            schedule=schedule,
            trigger=trigger,
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
            raise ValueError("Scheduled tasks require a future start_at")

        # ------------------  assemble payload  ------------------ #
        task_details = Task(
            name=name,
            description=description,
            status=status,
            schedule=schedule,
            trigger=trigger,
            deadline=deadline,
            repeat=repeat,
            priority=priority,
        ).to_post_json()

        # ------------------  write log immediately  ------------------ #
        log = unify.log(
            context=self._ctx,
            **task_details,
            new=True,
        )
        task_id = log.entries["task_id"]
        task_details["task_id"] = task_id

        # Keep linkage symmetric right after creation
        self._sync_adjacent_links(task_id=task_id, schedule=schedule)

        # ── Ensure the in-memory cache reflects any linkage tweaks ──
        if status == Status.primed:
            self._refresh_primed_cache(task_id)

        # ------------------  queue insertion (if relevant)  ---------- #
        if status == Status.queued:
            # Only *auto-append* when the caller did **not** supply an
            # explicit linkage (prev/next).  If linkage was given we assume
            # the user knows where the task belongs.
            explicit_linkage = schedule is not None and (
                self._sched_prev(schedule) is not None
                or self._sched_next(schedule) is not None
            )

            if explicit_linkage:
                return {
                    "outcome": "task created successfully",
                    "details": {"task_id": task_id},
                }
            original_q = [t.task_id for t in self._get_task_queue()]

            # Only insert if the new task isn't already in that list
            if task_id not in original_q:
                new_q = original_q + [task_id]
                self._update_task_queue(original=original_q, new=new_q)

        return {
            "outcome": "task created successfully",
            "details": {"task_id": task_id},
        }

    # Delete

    def _delete_task(self, *, task_id: int) -> ToolOutcome:
        """
        Permanently **remove** a task from storage.

        Parameters
        ----------
        task_id : int
            Identifier of the task to delete.

        Returns
        -------
        ToolOutcome
            Tool outcome with any extra relevant details.

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
        return {
            "outcome": "task deleted",
            "details": {"task_id": task_id},
        }

    # Cancel Task(s)

    def _cancel_tasks(self, task_ids: List[int]) -> ToolOutcome:
        """
        Mark one or many tasks as **cancelled** (non-recoverable terminal
        state).

        Parameters
        ----------
        task_ids : list[int]
            Identifiers of the tasks to cancel.

        Returns
        -------
        ToolOutcome
            Tool outcome with any extra relevant details.

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
        return {
            "outcome": "tasks cancelled",
            "details": {"task_ids": task_ids},
        }

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

    def _sync_adjacent_links(
        self,
        *,
        task_id: int,
        schedule: Optional[Union[Schedule, dict]],
    ) -> None:
        """
        Guarantee **link symmetry**:

        * If *schedule.prev_task* → *P*, then *P.schedule.next_task* → *task_id*
        * If *schedule.next_task* → *N*, then *N.schedule.prev_task* → *task_id*
        """
        if schedule is None:
            return

        if isinstance(schedule, Schedule):
            schedule = schedule.model_dump()

        neighbours: list[tuple[str, str, int]] = []
        if schedule.get("prev_task") is not None:
            neighbours.append(("next_task", "prev_task", schedule["prev_task"]))
        if schedule.get("next_task") is not None:
            neighbours.append(("prev_task", "next_task", schedule["next_task"]))

        for field_to_set, _, neighbour_id in neighbours:
            rows = self._search_tasks(filter=f"task_id == {neighbour_id}", limit=1)
            if not rows:
                raise ValueError(
                    f"Broken queue linkage: referenced task_id {neighbour_id} not found.",
                )

            row = rows[0]
            n_sched = {**(row.get("schedule") or {})}
            if n_sched.get(field_to_set) == task_id:
                continue  # already correct

            # Strip start_at if the neighbour ceases to be queue head
            if field_to_set == "prev_task":
                n_sched.pop("start_at", None)

            n_sched[field_to_set] = task_id
            log_id = self._get_logs_by_task_ids(task_ids=row["task_id"])
            unify.update_logs(
                logs=log_id,
                context=self._ctx,
                entries={"schedule": n_sched},
                overwrite=True,
            )

            # Was the neighbour the *primed* task?  Keep cache in lock-step.
            if (
                self._primed_task is not None
                and self._primed_task["task_id"] == neighbour_id
            ):
                self._refresh_primed_cache(neighbour_id)

    _TERMINAL_STATUSES = {"completed", "cancelled", "failed"}

    def _refresh_primed_cache(self, task_id: Optional[int] = None) -> None:
        """
        Reload the *primed* task from storage so that the in-memory copy
        always mirrors the authoritative log row.

        When *task_id* is *None*, the method refreshes the **currently
        cached** primed task (if there is one).  Otherwise the referenced
        row is fetched and promoted to ``self._primed_task``.
        """
        if task_id is None and self._primed_task is not None:
            task_id = self._primed_task["task_id"]
        if task_id is None:
            return

        rows = self._search_tasks(filter=f"task_id == {task_id}", limit=1)
        self._primed_task = rows[0] if rows else None

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
        execute_task: Optional[dict] = None

        # ── 0.  Pick a starting node ─────────────────────────────────────
        if task_id is None:
            if self._primed_task:
                execute_task = self._primed_task
                task_id = execute_task["task_id"]
            else:
                # Derive the head: the runnable task whose `prev_task` is None
                head_candidates = self._search_tasks(
                    filter=self._HEAD_FILTER,
                    limit=2,
                )
                if not head_candidates:
                    return []
                assert (
                    len(head_candidates) == 1
                ), f"Multiple heads detected: {head_candidates}"
                execute_task = head_candidates[0]
                task_id = execute_task["task_id"]

        if execute_task is None and task_id is not None:
            execute_task = _get_task_by_task_id(task_id)

        if execute_task is None:
            # fall back to queue head: node with no prev_task and non-terminal status
            head_candidates = self._search_tasks(
                filter=self._HEAD_FILTER,
                limit=2,
            )
            if not head_candidates:
                return []
            assert (
                len(head_candidates) == 1
            ), f"Multiple heads detected: {head_candidates}"
            execute_task = head_candidates[0]

        # not in queue yet? return list with only start task
        if execute_task is not None and execute_task["schedule"] is None:
            # Task exists but has no schedule pointers; therefore the
            # queue only has one item (the start task).
            return [Task(**execute_task)]

        # ----------------  walk backwards to head  ---------------- #
        cur = execute_task
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
            if cur["status"] not in self._TERMINAL_STATUSES:
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
    ) -> ToolOutcome:
        """
        **Re-link** the runnable queue so its order matches *new* **and**
        make sure that exactly one task – the **head** – carries the queue-
        level ``start_at`` field.

        Rationale
        ---------
        The timestamp denotes the *earliest* moment **any** work in the queue
        may begin. Logically that information belongs to the first task.
        Whenever we promote another task to the front we therefore have to
        transfer the timestamp alongside it and strip it from every other
        node.

        Parameters
        ----------
        original : list[int]
            Snapshot of the queue before the change. Used to locate the
            authoritative timestamp (if present) on the *former* head.
        new : list[int]
            Desired queue order (may include inserts; never removals).

        Returns
        -------
        ToolOutcome
            Tool outcome with any extra relevant details.

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
        for tid in new:
            row = self._search_tasks(filter=f"task_id == {tid}", limit=1)[0]
            if row.get("trigger") is not None:
                raise ValueError(
                    f"Task {tid} is trigger-based and cannot be placed in the queue.",
                )
        # Collect every task that already has a schedule entry – we need its
        # linkage pointers *and* any existing start_at value.
        existing_logs = {
            t["task_id"]: t
            for t in self._search_tasks()
            if t.get("schedule") is not None
        }

        # ── 1.  Extract the queue-level timestamp from the old head ──────────
        queue_start_ts: Optional[str] = None
        if original:
            _old_head = existing_logs.get(original[0])
            if _old_head:
                queue_start_ts = (_old_head.get("schedule") or {}).get("start_at")

        updates_per_log: Dict[int, Dict[str, Any]] = {}
        for idx, tid in enumerate(new):
            prev_tid = None if idx == 0 else new[idx - 1]
            next_tid = None if idx == len(new) - 1 else new[idx + 1]

            # ── Decide who owns the timestamp & status after re-order ──────
            if idx == 0:  # ↤ HEAD
                # Prefer the queue-level ts taken from the old head; fall back
                # to a ts that was already on the new head (rare but legal).
                start_ts = queue_start_ts
                if start_ts is None:
                    start_ts = (existing_logs.get(tid, {}).get("schedule") or {}).get(
                        "start_at",
                    )
            else:  # ↤ not head ⇒ must not have ts
                start_ts = None

            sched_payload = {
                "prev_task": prev_tid,
                "next_task": next_tid,
            }

            # Only include *start_at* when we actually know one (i.e. when
            # the task was explicitly scheduled by the user).  For plain queue
            # insertions `start_ts` will be *None* and we leave the field
            # absent.
            if start_ts is not None:
                sched_payload["start_at"] = start_ts

            # ----------------  derive *new* status  ---------------- #
            existing_status = Status(
                existing_logs.get(tid, {}).get("status", Status.queued),
            )

            # ── Determine the desired status after re-ordering ─────────────
            if start_ts is not None:  # head carries explicit timestamp
                desired_status = Status.scheduled
            else:
                desired_status = (
                    existing_status
                    if existing_status != Status.scheduled
                    else Status.queued
                )

            # Non-head tasks can *never* remain 'primed' – downgrade to queued
            if idx != 0 and desired_status == Status.primed:
                desired_status = Status.queued

            payload: Dict[str, Any] = {"schedule": sched_payload}
            if desired_status != existing_status:
                payload["status"] = desired_status

            updates_per_log[tid] = payload

        # ── Invariant check across the whole queue relink ────────────────────
        for tid, payload in updates_per_log.items():
            status_here = payload.get(
                "status",
                existing_logs.get(tid, {}).get("status", Status.queued),
            )
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
        return {
            "outcome": "queue reordered",
            "details": {"new_order": new},
        }

    def _update_task_trigger(
        self,
        *,
        task_id: int,
        new_trigger: Optional[Union[Trigger, Dict[str, Any]]],
    ) -> ToolOutcome:
        """
        Set, replace **or clear** a task's *trigger*.

        • Disallowed when the task already has a *schedule*.<br>
        • When a *trigger* is introduced the status becomes **triggerable**.<br>
        • When a *trigger* is removed and the task was *triggerable* it falls
          back to **queued** (idle, waiting for manual start or queue insert).
        """

        self._ensure_not_active_task(task_id)

        current_rows = self._search_tasks(filter=f"task_id == {task_id}", limit=1)
        if not current_rows:
            raise ValueError(f"No task found with id={task_id}")

        current = current_rows[0]

        if current.get("schedule") is not None and new_trigger is not None:
            raise ValueError(
                "Cannot add a *trigger* while a *schedule* exists. "
                "Remove the schedule first.",
            )

        if isinstance(new_trigger, dict):
            new_trigger = Trigger(**new_trigger)

        entries: Dict[str, Any] = {"trigger": new_trigger}

        # ── status transitions ───────────────────────────────────────────
        cur_status = Status(current["status"])
        if new_trigger is not None and cur_status != Status.triggerable:
            entries["status"] = Status.triggerable
        elif new_trigger is None and cur_status == Status.triggerable:
            entries["status"] = Status.queued

        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        unify.update_logs(
            logs=log_id,
            context=self._ctx,
            entries=entries,
            overwrite=True,
        )

        return {
            "outcome": "trigger updated",
            "details": {"task_id": task_id},
        }

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
            helpers (e.g. *execute_task*) pass *True* when they *really* need to.

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
        # ── Invariant check when transitioning to 'queued' ───────────────
        if str(new_status) == Status.queued.value:
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

        # Fetch current row (needed for invariants & trigger check)
        current_rows = self._search_tasks(filter=f"task_id == {task_id}", limit=1)

        if current_rows and current_rows[0].get("trigger") is not None:
            raise ValueError(
                "Cannot add/update *start_at* – the task is trigger-based.",
            )
        current_sched = current_rows[0].get("schedule") if current_rows else None

        # Guard-rail: tasks inside a queue can't own a start_at
        if self._sched_prev(current_sched) is not None:
            raise ValueError(
                "Cannot set 'start_at' when the task has 'prev_task'. "
                "Move it to the queue head first.",
            )

        if current_sched is None:
            current_sched = {}

        # Preserve queue linkage if it exists, otherwise default to None
        sched_payload = {
            "prev_task": self._sched_prev(current_sched),
            "next_task": self._sched_next(current_sched),
            "start_at": new_start_at,
        }

        # ensure the new schedule does not violate the invariant
        self._validate_scheduled_invariants(
            status=current_rows[0]["status"],
            schedule=sched_payload,
            err_prefix=f"While updating start_at for task {task_id}:",
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
            source_column="_name_plus_desc",
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
                exclude_fields=[self._VEC_TASK],
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
        if isinstance(filter, str):
            filter = filter.replace(".start_at", "['start_at']")
        return [
            log.entries
            for log in unify.get_logs(
                context=self._ctx,
                filter=filter,
                offset=offset,
                limit=limit,
                exclude_fields=[self._VEC_TASK],
            )
        ]

    # ────────────────────────────────────────────────────────────────────
    # Broader context helper
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _inject_broader_context(msgs: list[dict]) -> list[dict]:
        """Replace `{broader_context}` placeholders inside *system* messages with
        the latest summary from `MemoryManager` right before sending the prompt."""

        import copy

        from unity.memory_manager.memory_manager import (
            MemoryManager,
        )  # local import to avoid cycles

        patched = copy.deepcopy(msgs)

        try:
            broader_ctx = MemoryManager.get_rolling_activity()
        except Exception:
            broader_ctx = ""

        for m in patched:
            if m.get("role") == "system" and "{broader_context}" in (
                m.get("content") or ""
            ):
                m["content"] = m["content"].replace("{broader_context}", broader_ctx)

        return patched
