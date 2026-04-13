"""Task Scheduler: create, schedule, update, and execute tasks with queues.

This module provides the concrete TaskScheduler which:
- exposes read-only ask and mutating update methods;
- manages runnable queues (head→tail) with invariant-preserving operations;
- executes tasks individually or as a chain and tracks a single active task;
- records reintegration plans to restore deferred tasks precisely.
"""

from __future__ import annotations

import unify
import unillm
import asyncio
import functools
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Type, Union, Callable, Tuple
from typing import Literal, overload
from pydantic import BaseModel
from dataclasses import dataclass
from functools import cached_property

from ..settings import SETTINGS
from ..common.embed_utils import ensure_vector_column
from ..common.tool_spec import read_only, ToolSpec
from ..common.llm_helpers import (
    methods_to_tool_dict,
    short_id,
)
from ..common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..common.tool_outcome import ToolOutcome
from .types.status import Status, to_status
from .types.priority import Priority
from .types.schedule import Schedule
from .types.trigger import Trigger
from .types.repetition import (
    RepeatPattern,
    Frequency,
    Weekday,
    next_repeated_start_at,
)
from .types.task import TaskBase, Task
from .types.activated_by import ActivatedBy
from .types.queue_plan import QueuePlan
from ..common.metrics_utils import reduce_logs

# ------------------------------------------------------------------ #
#  Local type aliases                                                 #
# ------------------------------------------------------------------ #
# These aliases improve readability and keep signatures concise.
ScheduleLike = Optional[Union[Schedule, Dict[str, Any]]]
TriggerLike = Optional[Union[Trigger, Dict[str, Any]]]
RepeatLike = Optional[List[Union[RepeatPattern, Dict[str, Any]]]]
ToolsDict = Dict[str, Callable]

# Contact manager obtained via ManagerRegistry to respect IMPL settings
from ..manager_registry import ManagerRegistry
from ..common.model_to_fields import model_to_fields
from .prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
)
from .base import BaseTaskScheduler
from ..actor.base import BaseActor
from ..actor.simulated import SimulatedActor
from .active_task import ActiveTask
from .active_queue import ActiveQueue
from dataclasses import dataclass

from ..events.manager_event_logging import log_manager_call
from ..common.search_utils import table_search_top_k
from .types.schedule import (
    sched_prev,
    sched_next,
)
from .queue_utils import sync_adjacent_links as _q_sync_adjacent_links
from .activation_ops import detach_from_queue_for_activation
from .reintegration import ReintegrationManager
from ..common.filter_utils import normalize_filter_expr
from .queue_engine import plan_reorder_queue, derive_status_after_queue_edit
from ..common.llm_client import new_llm_client
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from ..common.sentinels import _UnsetSentinel
from ..common.context_registry import ContextRegistry, TableContext

# Sentinel for optional-argument presence detection
_UNSET = _UnsetSentinel()

# ------------------------------------------------------------------ #
#  Typed reintegration plan                                          #
# ------------------------------------------------------------------ #
from .types.reintegration_plan import ReintegrationPlan
from .storage import TasksStore, LocalTaskView


class TaskScheduler(BaseTaskScheduler):
    @dataclass
    class ActivePointer:
        task_id: int
        instance_id: int
        handle: "ActiveTask"

    _HEAD_FILTER = (
        "schedule is not None and "
        "status not in ('completed','cancelled','failed') and "
        "schedule.get('prev_task') is None"
    )

    class Config:
        required_contexts = [
            TableContext(
                name="Tasks",
                description=(
                    "List of all tasks with their name, description, status, "
                    "schedule, deadline, repeat pattern, priority **and** "
                    "`instance_id` which tracks multiple executions of the "
                    "same logical task."
                ),
                fields=model_to_fields(Task),
                unique_keys={"task_id": "int", "instance_id": "int"},
                auto_counting={
                    "task_id": None,
                    "instance_id": "task_id",
                },
                foreign_keys=[
                    {
                        "name": "entrypoint",
                        "references": "Functions/Compositional.function_id",
                        "on_delete": "SET NULL",
                        "on_update": "CASCADE",
                    },
                ],
            ),
        ]

    def __init__(
        self,
        *,
        actor: Optional[BaseActor] = None,
        rolling_summary_in_prompts: bool = True,
    ) -> None:
        """
        Create a scheduler responsible for creating, searching, updating and executing tasks in the current Unify context.

        Parameters
        ----------
        actor : BaseActor | None, default ``None``
            Actor used to execute the steps of an active task. When ``None``, a
            ``SimulatedActor(duration=20)`` is used.
        rolling_summary_in_prompts : bool, default ``True``
            Whether to inject the rolling activity summary into system prompts sent to the LLM.

        Notes
        -----
        - Ensures a ``"<active_context>/Tasks"`` context exists with fields derived from the ``Task`` model.
        - Exposes read/write tools and mirrors selected ``ContactManager`` tools for cross‑domain workflows.
        - Maintains in‑memory pointers to the single primed task and the current active task handle (if any).
        """
        super().__init__()
        self.include_in_multi_assistant_table = True

        # Get ContactManager via registry so its bound methods can act as tools
        self._contact_manager = ManagerRegistry.get_contact_manager()

        # Query-only helpers – safe, read-only operations.  Include the *external* contact lookup
        ask_tools = {
            **methods_to_tool_dict(
                ToolSpec(fn=self._filter_tasks, display_label="Filtering tasks"),
                ToolSpec(fn=self._search_tasks, display_label="Searching tasks"),
                ToolSpec(fn=self._get_queue, display_label="Viewing a task queue"),
                ToolSpec(
                    fn=self._get_queue_for_task,
                    display_label="Finding queue for a task",
                ),
                ToolSpec(fn=self._reduce, display_label="Summarising tasks"),
                include_class_name=False,
            ),
            **methods_to_tool_dict(
                ToolSpec(
                    fn=self._contact_manager.ask,
                    display_label="Looking up contact details",
                ),
                include_class_name=True,
            ),
        }
        # Persist a stable copy for simulated mirroring / prompt-build stability.
        # (Used by unity.common.simulated.mirror_task_scheduler_tools via AST reflection.)
        self._ask_tools = dict(ask_tools)
        self.add_tools("ask", ask_tools)

        # Write-capable helpers – every mutating operation as well as the read-only ones.
        update_tools = {
            **methods_to_tool_dict(
                # Ask
                ToolSpec(fn=self.ask, display_label="Querying tasks"),
                # Read-only task discovery (useful for update flows too)
                ToolSpec(fn=self._filter_tasks, display_label="Filtering tasks"),
                ToolSpec(fn=self._search_tasks, display_label="Searching tasks"),
                # Creation / deletion / cancellation
                ToolSpec(
                    fn=self._create_tasks,
                    display_label="Creating multiple tasks",
                ),
                ToolSpec(fn=self._create_task, display_label="Creating a new task"),
                ToolSpec(fn=self._delete_task, display_label="Deleting a task"),
                ToolSpec(fn=self._cancel_tasks, display_label="Cancelling tasks"),
                # Queue manipulation
                ToolSpec(fn=self._list_queues, display_label="Listing all task queues"),
                ToolSpec(fn=self._get_queue, display_label="Viewing a task queue"),
                ToolSpec(
                    fn=self._get_queue_for_task,
                    display_label="Finding queue for a task",
                ),
                ToolSpec(
                    fn=self._reorder_queue,
                    display_label="Reordering a task queue",
                ),
                ToolSpec(
                    fn=self._move_tasks_to_queue,
                    display_label="Moving tasks between queues",
                ),
                ToolSpec(
                    fn=self._partition_queue,
                    display_label="Splitting a task queue",
                ),
                # Reintegration
                ToolSpec(
                    fn=self._reinstate_task_to_previous_queue,
                    display_label="Reinstating a task",
                ),
                # Attribute mutations (single general-purpose updater)
                ToolSpec(fn=self._update_task, display_label="Updating a task"),
                include_class_name=False,
            ),
            **methods_to_tool_dict(
                ToolSpec(
                    fn=self._contact_manager.ask,
                    display_label="Looking up contact details",
                ),
                include_class_name=True,
            ),
        }
        # Persist a stable copy for simulated mirroring / prompt-build stability.
        self._update_tools = dict(update_tools)
        self.add_tools("update", update_tools)

        # active task
        self.__actor = actor
        self._ctx = ContextRegistry.get_context(self, "Tasks")

        # Install storage adapter and ensure context/fields exist
        self._provision_storage()

        # `_num_tasks()` will lazily populate and maintain the cached count.

        # In-memory checkpoints for reversible multi-queue edits within a session
        # Keyed by opaque checkpoint ids; values contain a minimal snapshot of all queues
        # (queue_id, head_id, order list, and queue-level start_at).
        self._queue_checkpoints: Dict[str, Dict[str, Any]] = {}

        # Pointer to the single currently active task handle (or None).
        # Exactly one task can be active at a time.
        self._active_task: Optional[TaskScheduler.ActivePointer] = None
        self._primed_task: Optional[Task] = None

        primed_tasks = self._filter_tasks(filter="status == 'primed'")
        if primed_tasks:
            assert (
                len(primed_tasks) == 1
            ), f"More than one primed task found:\n{primed_tasks}"
            self._primed_task = primed_tasks[0]

        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # Registry of corrective plans per active task so we can restore their
        # original position on defer stop. Map: task_id -> ReintegrationPlan
        self._reintegration_plans: Dict[Tuple[int, int], "ReintegrationPlan"] = {}
        self._reintegration_manager = ReintegrationManager(self)
        # Queue index, log-id memoization and id allocator are centralized in
        # LocalTaskView.

        # Lightweight cached count of tasks within the current Tasks context.
        # - Populated lazily on first use by _num_tasks()
        # - Kept in sync by create/clone/delete flows
        # Because this scheduler is a singleton and all mutations flow through it,
        # this cache remains coherent without extra backend reads between tool calls.
        self._num_tasks_cached: Optional[int] = None

    @cached_property
    def _actor(self) -> BaseActor:
        if self.__actor is None:
            self.__actor = SimulatedActor(duration=SETTINGS.task.SIM_ACTOR_DURATION)
        return self.__actor

    # ------------------------------ Provisioning ----------------------------- #
    def warm_embeddings(self) -> None:
        for col in ("name", "description"):
            try:
                ensure_vector_column(
                    self._ctx,
                    embed_column=f"_{col}_emb",
                    source_column=col,
                )
            except Exception:
                pass

    def _provision_storage(self) -> None:
        """Ensure Tasks context, schema and local view exist (idempotent)."""
        # Install storage adapter and ensure context/fields exist
        self._store = TasksStore(
            self._ctx,
            add_to_all_context=self.include_in_multi_assistant_table,
        )

        # Centralised local view for queue membership, allocator and light caching.
        self._view = LocalTaskView(self._store)

    @functools.wraps(BaseTaskScheduler.clear, updated=())
    def clear(self) -> None:
        unify.delete_context(self._ctx)

        # Reset local/cached state so subsequent operations see a clean slate
        try:
            self._queue_checkpoints.clear()
        except Exception:
            pass
        self._active_task = None
        self._primed_task = None
        self._reintegration_plans = {}
        self._num_tasks_cached = None

        # Clear the cached context from ContextRegistry so it will be re-created
        ContextRegistry.forget(self, "Tasks")

        # Re-create the context with proper schema and fields
        self._ctx = ContextRegistry.get_context(self, "Tasks")

        # Re-provision storage (schema, local view)
        self._provision_storage()

    # ------------------------------ Small helpers ------------------------------ #
    def _task_id_to_log_id_map(self, task_ids: List[int]) -> Dict[int, int]:
        """Resolve a mapping of task_id → log_id in one call (best-effort)."""
        try:
            log_objs = self._get_logs_by_task_ids(
                task_ids=task_ids,
                return_ids_only=False,
            )
        except Exception:
            log_objs = []

        if not log_objs:
            return {}

        id_map: Dict[int, int] = {}
        for lg in log_objs:
            id_map[lg.entries["task_id"]] = lg.id
        return id_map

    def _write_entries_batched(
        self,
        *,
        entries_by_tid: Dict[int, Dict[str, Any]],
    ) -> None:
        """Best-effort batched write using LocalTaskView; fall back to per-task."""
        if not entries_by_tid:
            return
        try:
            self._view.write_entries_by_task_ids(entries_by_tid=entries_by_tid)
            return
        except Exception:
            pass
        # Fallback: per-task validated writes
        for tid, write_entries in entries_by_tid.items():
            task = self._get_task_or_raise(tid)
            self._validated_write(
                task_id=tid,
                entries=write_entries,
                err_prefix=f"While batched-writing entries (task {tid}):",
                current_row=task,
                skip_cross_queue_guard=True,
                skip_sync=True,
            )

    def _head_start_at_from_rows(self, rows: List[Dict[str, Any]]) -> Optional[str]:
        """Return the queue-level start_at from the head row among provided rows.

        Expects rows as simple dicts with a 'schedule' key; ignores non-dict entries.
        """
        for r in rows:
            sched = r.get("schedule", {})
            if sched.get("prev_task") is None:
                return sched.get("start_at")
        return None

    # Public #
    # -------#

    # English-Text Question

    @functools.wraps(BaseTaskScheduler.ask, updated=())
    @log_manager_call(
        "TaskScheduler",
        "ask",
        payload_key="question",
        display_label="Checking tasks",
    )
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        tool_policy: Union[
            Literal["default"],
            Callable[[int, Dict[str, Any]], tuple[str, Dict[str, Any]]],
            None,
        ] = "default",
    ) -> SteerableToolHandle:
        client = new_llm_client()

        # Build a live tools dictionary so the prompt reflects reality
        tools = dict(self.get_tools("ask"))

        _clar_queues = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            from ..common.llm_helpers import make_request_clarification_tool

            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

        # Inject the dynamic system prompt
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        client.set_system_message(
            build_ask_prompt(
                tools,
                num_tasks=self._num_tasks(),
                columns=self._list_columns(),
                include_activity=include_activity,
            ).to_list(),
        )

        # Prepare effective tool_policy
        if tool_policy == "default":
            effective_tool_policy = self._default_ask_tool_policy
        else:
            # pass through callable or None
            effective_tool_policy = tool_policy

        # Start the tool-use loop
        handle = self._start_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_chat_context=_parent_chat_context,
            log_steps=_log_tool_steps,
            tool_policy=effective_tool_policy,
            handle_cls=(
                ReadOnlyAskGuardHandle if SETTINGS.UNITY_READONLY_ASK_GUARD else None
            ),
            response_format=response_format,
            clarification_queues=_clar_queues,
        )
        # Logging wrapper applied by decorator

        # Optional reasoning exposure
        if _return_reasoning_steps:
            handle = self._wrap_result_with_messages(handle, client)

        return handle

    # English-Text Update Request

    @functools.wraps(BaseTaskScheduler.update, updated=())
    @log_manager_call(
        "TaskScheduler",
        "update",
        payload_key="request",
        display_label="Updating tasks",
    )
    async def update(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        tool_policy: Union[
            Literal["default"],
            Callable[[int, Dict[str, Any]], tuple[str, Dict[str, Any]]],
            None,
        ] = "default",
    ) -> SteerableToolHandle:
        client = new_llm_client()

        # Build a live tools dictionary first (prompt needs it)
        tools = dict(self.get_tools("update"))

        # Merge these helpers into the update toolset
        tools.update(
            methods_to_tool_dict(
                ToolSpec(
                    fn=self.validate_queue_plan,
                    display_label="Validating queue changes",
                ),
                ToolSpec(
                    fn=self.apply_queue_plan,
                    display_label="Applying queue changes",
                ),
                ToolSpec(
                    fn=self.checkpoint_queue_state,
                    display_label="Saving a queue checkpoint",
                ),
                ToolSpec(
                    fn=self.revert_to_checkpoint,
                    display_label="Reverting to a checkpoint",
                ),
                ToolSpec(
                    fn=self.get_latest_checkpoint,
                    display_label="Retrieving latest checkpoint",
                ),
                ToolSpec(
                    fn=self._set_queue,
                    display_label="Setting task queue directly",
                ),
                ToolSpec(
                    fn=self._set_schedules_atomic,
                    display_label="Updating task schedules",
                ),
                include_class_name=False,
            ),
        )

        _clar_queues = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            from ..common.llm_helpers import make_request_clarification_tool

            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

        # Inject the dynamic system prompt
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_update_prompt(
                tools,
                num_tasks=self._num_tasks(),
                columns=self._list_columns(),
                include_activity=include_activity,
            ).to_list(),
        )

        # Prepare effective tool_policy
        if tool_policy == "default":
            effective_tool_policy = self._default_update_tool_policy
        else:
            # pass through callable or None
            effective_tool_policy = tool_policy

        # Start the interactive loop
        handle = self._start_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_chat_context=_parent_chat_context,
            log_steps=_log_tool_steps,
            tool_policy=effective_tool_policy,
            response_format=response_format,
            clarification_queues=_clar_queues,
        )
        # Logging wrapper applied by decorator

        # Optional reasoning exposure
        if _return_reasoning_steps:
            handle = self._wrap_result_with_messages(handle, client)

        return handle

    # Execute

    @functools.wraps(BaseTaskScheduler.execute, updated=())
    @log_manager_call(
        "TaskScheduler",
        "execute",
        payload_key="request",
        display_label="Working on task",
    )
    async def execute(
        self,
        task_id: int,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        isolated: Optional[bool] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        # Refuse execution when a task is already active.
        if self._active_task is not None:
            raise RuntimeError("Another task is already running – stop it first.")

        # Also guard against orphan 'active' rows (e.g., after crash) even if pointer is None.
        try:
            any_active = any(
                r.status == Status.active
                for r in self._filter_tasks(filter="status == 'active'", limit=1)
            )
        except Exception:
            any_active = False
        if any_active:
            raise RuntimeError(
                "A task is marked as active, but no active handle is present – reconcile state before starting another task.",
            )

        # Execute strictly by id; choose isolation semantics based on flag
        handle = await self._execute_queue_internal(
            task_id=task_id,
            parent_chat_context=_parent_chat_context,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            detach=bool(isolated),
        )
        return handle

    # ------------------------------------------------------------------ #
    #  Internal helper – run existing *by-id* logic without event logging   #
    # ------------------------------------------------------------------ #

    async def _execute_internal(
        self,
        *,
        task_id: int,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        activated_by: Optional[ActivatedBy] = None,
        detach: bool = True,
        unlink_from_prev: bool = False,
    ) -> SteerableToolHandle:
        """
        Start the execution of a runnable task by its identifier.

        Parameters
        ----------
        task_id : int
            Identifier of the task to start. Must resolve to a single, non‑terminal,
            non‑active instance.
        parent_chat_context : list[dict] | None, default ``None``
            Prior messages to seed the conversation used for the actor execution.
        clarification_up_q : asyncio.Queue[str] | None, default ``None``
            Queue used to bubble clarification questions to the caller. Must be provided
            together with ``clarification_down_q`` for interactive sessions.
        clarification_down_q : asyncio.Queue[str] | None, default ``None``
            Queue on which answers to clarification questions are received.
        activated_by : ActivatedBy | None, default ``None``
            Activation reason for the task. If not provided, it will be inferred from the task's configuration.

        Returns
        -------
        SteerableToolHandle
            The handle for the active plan tied to ``task_id``.

        Raises
        ------
        RuntimeError
            If another task is already active.
        ValueError
            If ``task_id`` does not exist, refers to a non‑runnable instance, or the
            task is already terminal/active.
        """

        # Sanity checks
        if self._active_task is not None:
            raise RuntimeError("Another task is already running – stop it first.")

        candidate_tasks = self._filter_tasks(
            filter=(
                f"task_id == {task_id} and status not in "
                "('completed','cancelled','failed','active')"
            ),
        )
        if not candidate_tasks:
            raise ValueError(f"No runnable task found with id={task_id}")

        # Pick the *oldest* runnable instance (lowest instance_id)
        task = sorted(
            candidate_tasks,
            key=lambda t: t.instance_id,
        )[0]
        if task.status in (
            Status.completed,
            Status.cancelled,
            Status.failed,
            Status.active,
        ):
            raise ValueError(f"Task {task_id} is already {task.status!r}.")

        # Adjust queue linkages for activation (and record reintegration plan).
        # detach=True → isolation semantics; detach=False → chain semantics.

        detach_from_queue_for_activation(
            self,
            task_id=task_id,
            detach=detach,
            unlink_from_prev=unlink_from_prev,
        )

        # Start task execution (delegated to the current execution environment when available)
        # and wrap the resulting handle for Tasks-table synchronization.

        handle = await ActiveTask.create(
            self._actor,
            task_description=task.description or task.name,
            _parent_chat_context=parent_chat_context,
            _clarification_up_q=clarification_up_q,
            _clarification_down_q=clarification_down_q,
            task_id=task_id,
            instance_id=task.instance_id,
            scheduler=self,
            entrypoint=task.entrypoint,
        )

        self._active_task = TaskScheduler.ActivePointer(
            task_id=task_id,
            instance_id=task.instance_id,
            handle=handle,
        )

        # Clone if this is a triggerable or recurring task
        if task.status == Status.triggerable or (
            task.repeat is not None and task.schedule_start_at is not None
        ):
            self._clone_task_instance(task)

        # Promote status to active (and record the activation reason) and clear the primed pointer if needed

        # Infer activation reason based on provided cause or task configuration
        reason: ActivatedBy
        if activated_by is not None:
            reason = activated_by
        else:
            if task.trigger is not None:
                reason = ActivatedBy.trigger
            elif (task.schedule_prev is None) and (task.schedule_start_at is not None):
                reason = ActivatedBy.schedule
            elif task.schedule_prev is not None:
                reason = ActivatedBy.queue
            else:
                reason = ActivatedBy.explicit

        self._update_task_status_instance(
            task_id=task_id,
            instance_id=task.instance_id,
            new_status=Status.active,
            activated_by=reason,
        )
        if self._primed_task and self._primed_task.task_id == task_id:
            self._primed_task = None

        return handle

    # ------------------------------------------------------------------ #
    #  Chain orchestrator: sequentially executes the follower queue       #
    # ------------------------------------------------------------------ #

    async def _execute_queue_internal(
        self,
        *,
        task_id: int,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        detach: bool = False,
    ) -> SteerableToolHandle:
        """Start queue execution at `task_id` and return a composite queue handle."""
        first = await self._execute_internal(
            task_id=task_id,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
            activated_by=ActivatedBy.explicit,
            # Detach first task when explicitly requested; otherwise keep queue semantics
            detach=detach,
            # Only at creation: if we are starting from a mid-queue task in chained mode,
            # unlink from predecessor once to make this the effective head.
            unlink_from_prev=not detach,
        )
        return ActiveQueue(
            self,
            first_task_id=task_id,
            first_handle=first,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    # ------------------------------------------------------------------ #
    #  Public execution tools (registered via get_tools("execute"))      #
    # ------------------------------------------------------------------ #

    async def _execute_by_id(self, *, task_id: int) -> SteerableToolHandle:
        """
        Start the task identified by task_id using queue semantics and return a steerable handle.

        Behaviour
        ---------
        - Does not mutate scheduling fields (e.g., start_at) purely to run.
        - Preserves existing queue membership and chaining semantics; followers remain attached.
        - Returns a live handle that can be adopted by outer loops.

        Returns
        -------
        SteerableToolHandle
            A live handle for the running task or queue head that supports pause, resume,
            interject, stop and result().
        """
        handle = await self._execute_queue_internal(
            task_id=task_id,
            parent_chat_context=None,
            clarification_up_q=None,
            clarification_down_q=None,
            detach=False,
        )
        return handle

    async def _execute_isolated_by_id(self, *, task_id: int) -> SteerableToolHandle:
        """
        Start ONLY the specified task in isolation by detaching it from any queue, and return a handle.

        Behaviour
        ---------
        - Detaches the task from its queue for this run so followers do not chain automatically.
        - Does not rewrite scheduling fields merely to run.
        - Returns a steerable handle that the caller can use directly.

        Returns
        -------
        SteerableToolHandle
            A live handle for the isolated run of the requested task.
        """
        handle = await self._execute_queue_internal(
            task_id=task_id,
            parent_chat_context=None,
            clarification_up_q=None,
            clarification_down_q=None,
            detach=True,
        )
        return handle

    def create_task(self, *, name: str, description: str) -> ToolOutcome:
        """
        Create a brand‑new task with minimal inputs (name and description only).

        Purpose
        -------
        This narrowly scoped creator exists for the execute flow so the LLM can
        materialize a missing task without manipulating scheduling/status/queue
        fields during execution planning. The scheduler infers lifecycle values
        and enforces invariants; callers should not attempt to set schedule or
        status here. Use update tools for full edits.

        Parameters
        ----------
        name : str
            Human‑readable task name.
        description : str
            A short description of the task.
        """
        return self._create_task(name=name, description=description)

    # (Removed LLM-driven outer loop – execution is strictly by id)

    # ------------------------------------------------------------------ #
    #  Scope classification helper (LLM-routed)                           #
    # ------------------------------------------------------------------ #

    #  Per-instance helpers

    def _update_task_status_instance(
        self,
        *,
        task_id: int,
        instance_id: int,
        new_status: str | Status,
        activated_by: Optional["ActivatedBy"] = None,
    ) -> Dict[str, str]:
        """
        Update the lifecycle ``status`` for a single ``(task_id, instance_id)`` row.

        Parameters
        ----------
        task_id : int
            Task identifier.
        instance_id : int
            Instance identifier within the task.
        new_status : str
            New status value to apply.

        Returns
        -------
        dict[str, str]
            Confirmation payload from ``unify.update_logs``.

        Raises
        ------
        ValueError
            If the specified instance cannot be found.
        AssertionError
            If more than one row matches the composite key.
        """
        log_objs = self._view.get_rows(
            filter=f"task_id == {task_id} and instance_id == {instance_id}",
            return_ids_only=False,
        )
        if not log_objs:
            raise ValueError(
                f"No task instance ({task_id}.{instance_id}) found.",
            )
        assert len(log_objs) == 1, "Composite primary key must be unique."
        # Normalise status to enum for consistent comparisons
        new_status_enum = to_status(new_status)
        entries: Dict[str, Any] = {"status": new_status_enum}
        # Only allow `activated_by` to be set during transition to 'active'.
        # For transitions away from 'active', preserve the existing value for auditability.
        if new_status_enum == Status.active and activated_by is not None:
            # Set only at the moment of activation; never overwrite later
            entries["activated_by"] = str(activated_by)

        result = self._write_log_entries(
            logs=log_objs[0].id,
            entries=entries,
        )
        # Auto-clear reintegration plan on completion/failed to avoid stale replay.
        # Intentionally keep the plan on 'cancelled' so callers can reinstate
        # a cancelled isolated activation back to its prior queue position.
        key = (task_id, instance_id)
        plan = self._reintegration_plans.get(key)
        if (
            plan is not None
            and plan.task_id == task_id
            and plan.instance_id == instance_id
            and new_status_enum in {Status.completed, Status.failed}
        ):
            self._reintegration_plans.pop(key, None)

        return result

    def _clone_task_instance(self, task: Task) -> None:
        """
        Create a fresh row for the next instance of a triggerable or recurring task.

        Parameters
        ----------
        task : Task
            Existing task used as the template. Copies user‑facing fields,
            keeps the same ``task_id``, omits ``instance_id`` so the backend auto‑increments it,
            and preserves the existing status (``triggerable`` or ``scheduled``).
        """
        # Do not carry over activation metadata to a fresh instance
        # Let the backend assign a new instance_id
        clone_payload = task.model_dump(exclude={"instance_id", "activated_by"})
        if task.repeat is not None and task.schedule_start_at is not None:
            next_start_at = next_repeated_start_at(
                previous_start=task.schedule_start_at,
                patterns=task.repeat,
                current_occurrence_index=task.instance_id,
            )
            if next_start_at is None:
                return
            clone_payload["status"] = Status.scheduled
            clone_payload["schedule"] = {
                "prev_task": None,
                "next_task": None,
                "start_at": next_start_at.isoformat(),
            }
        self._view.create_one(entries=clone_payload, new=True)
        # Maintain cached total count (+1 new instance row)
        if self._num_tasks_cached is not None:
            self._num_tasks_cached += 1

    # Private Helpers #
    # ----------------#

    def _validate_scheduled_invariants(
        self,
        *,
        status: Status | str,
        schedule: ScheduleLike,
        trigger: TriggerLike = None,
        err_prefix: str = "Invalid task state:",
    ) -> None:
        """
        Validate invariants related to queue linkage and scheduling.

        Invariants (when the task is not trigger-based):
        - If a task is at the head of the queue (prev_task is None) and defines
          a start_at, then its status must be 'scheduled' (not 'queued').
        - A task must not define both prev_task and start_at simultaneously –
          the queue-level timestamp lives on the head node only.
        - A 'primed' task must be the queue head (prev_task is None).
        - When setting status to 'scheduled', the task must have either a
          prev_task (it sits in the queue) or a start_at timestamp.

        Parameters
        ----------
        status : Status | str
            The prospective status after the change.
        schedule : Schedule | dict | None
            The prospective schedule after the change.
        trigger : Trigger | dict | None, default ``None``
            When provided, schedule invariants do not apply.
        err_prefix : str, default ``"Invalid task state:"``
            Prefix used in raised error messages for context.

        Raises
        ------
        ValueError
            If any of the queue/scheduling invariants are violated.
        """
        # Trigger-driven tasks are not subject to queue/schedule invariants
        if trigger is not None:
            return

        # Normalise status and extract linkage/timestamp
        status = to_status(status)
        prev_task_id = sched_prev(schedule)
        start_at_ts = self._extract_start_at(schedule)

        # Head-of-queue tasks with explicit start_at must be 'scheduled'
        if status == Status.queued and prev_task_id is None and start_at_ts is not None:
            raise ValueError(
                f"{err_prefix} tasks at the head of the queue that define 'start_at' must have status 'scheduled', not 'queued'.",
            )

        # A non-head task may not carry a start_at
        if prev_task_id is not None and start_at_ts is not None:
            raise ValueError(
                f"{err_prefix} a task cannot define both 'prev_task' and "
                "'start_at' – the timestamp belongs on the queue head only.",
            )

        # 'primed' must always be at the head
        if status == Status.primed and prev_task_id is not None:
            raise ValueError(
                f"{err_prefix} a task in 'primed' state must be at the head of the queue (prev_task must be None).",
            )

        if status != Status.scheduled:
            return

        # 'scheduled' requires either a queue position or a start_at
        if prev_task_id is None and start_at_ts is None:
            raise ValueError(
                f"{err_prefix} a task with status 'scheduled' must have either "
                "`prev_task` (it sits behind another task in the queue) or a "
                "`start_at` timestamp.",
            )

    def _ensure_not_active_task(self, task_ids: Union[int, List[int]]) -> None:
        """
        Guard against mutating the currently active task.

        Parameters
        ----------
        task_ids : int | list[int]
            Single id or list of ids that must not include the active task id.

        Raises
        ------
        RuntimeError
            If the active task id is among ``task_ids``.
        """
        if self._active_task is None:
            return

        if isinstance(task_ids, int):
            ids = [task_ids]
        else:
            ids = list(task_ids)

        active_task_id = self._active_task.task_id
        if active_task_id in ids:
            raise RuntimeError(
                f"Operation not permitted on the active task (task_id={active_task_id})",
            )

    @overload
    def _get_logs_by_task_ids(
        self,
        *,
        task_ids: Union[int, List[int]],
        return_ids_only: Literal[True] = True,
    ) -> List[int]: ...

    @overload
    def _get_logs_by_task_ids(
        self,
        *,
        task_ids: Union[int, List[int]],
        return_ids_only: Literal[False],
    ) -> List[unify.Log]: ...

    def _get_logs_by_task_ids(
        self,
        *,
        task_ids: Union[int, List[int]],
        return_ids_only: bool = True,
    ):
        """
        Fetch the Unify log objects (or ids) corresponding to one or many task ids.

        Parameters
        ----------
        task_ids : int | list[int]
            Single id or list of ids to look up.
        return_ids_only : bool, default ``True``
            When ``True``, return underlying log ids instead of full log objects.

        Returns
        -------
        list[int] | list[unify.Log]
            The matching log identifiers or objects.
        """
        return self._view.get_log_ids_by_task_ids(
            task_ids=task_ids,
            return_ids_only=return_ids_only,
        )

    # Private Tools #
    # --------------#

    # Create

    def _create_task(
        self,
        *,
        name: str,
        description: str,
        queue_id: Optional[int] = None,
        status: Optional[Status] = None,
        schedule: ScheduleLike = None,
        trigger: TriggerLike = None,
        deadline: Optional[str] = None,
        repeat: RepeatLike = None,
        priority: Priority = Priority.normal,
        response_policy: Optional[str] = None,
        entrypoint: Optional[int] = None,
        offline: bool = False,
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
        entrypoint : int | None, default ``None``
            Optional function_id from the Functions table that should be invoked to perform this task. When null,
            the task is executed by an Actor interpreting the description on the fly.
        offline : bool, default ``False``
            When true, the task executes in the hidden offline lane and therefore
            must provide a numeric ``entrypoint``.
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
        response_policy : str | None
            Freeform policy dictating how the assistant should interact with relevant contacts during the task.

        Returns
        -------
        ToolOutcome
            Tool outcome with any extra relevant details.

        Raises
        ------
        ValueError
            On invalid field combinations or uniqueness violations.

        Notes
        -----
        Schedule/Queue invariants the model MUST respect when supplying arguments:

        • If you provide a ``schedule`` with ``prev_task is None`` and a non-empty
          ``start_at`` timestamp (i.e., the queue head with a start time), the
          task's ``status`` MUST be ``scheduled``. Do not set it to ``queued``.

        • Non-head tasks (``prev_task`` not ``None``) MUST NOT define ``start_at``.
          The queue-level timestamp lives on the head node only.

        • ``primed`` tasks must be at the head (``prev_task is None``). Do not
          set ``primed`` on tasks that sit behind another task.

        • A task in ``scheduled`` state must have either a queue position
          (``prev_task`` is set) or a ``start_at`` timestamp.

        To avoid mistakes, prefer omitting ``status`` and let the scheduler infer
        the correct lifecycle value from the ``schedule`` you supply.
        """
        # ----------------  helper: iso-8601 → datetime  ---------------- #
        from datetime import datetime, timezone

        def _parse_maybe_iso(ts: str) -> datetime:
            dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(ts)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        # ----------------  initial validation & dedup  ---------------- #
        if not name or not description:
            raise ValueError("Both 'name' and 'description' are required")

        # Uniqueness (name/description) – single read covering both columns within this tool call
        # Escape values via repr to keep a valid filter string regardless of content.
        _dupe_rows = self._find_name_desc_collisions(
            name=name,
            description=description,
            limit=2,
        )
        if _dupe_rows:
            # Identify which field(s) collide for precise errors
            for _r in _dupe_rows:
                if _r.get("name") == name:
                    raise ValueError(
                        f"A task with {'name'!r} = {name!r} already exists",
                    )
                if _r.get("description") == description:
                    raise ValueError(
                        f"A task with {'description'!r} = {description!r} already exists",
                    )

        # ----------------------------------- #
        #  derive status when caller omitted   #
        # ----------------------------------- #
        if status is not None and isinstance(status, str):
            status = to_status(status)

        # Convert schedule dict to Schedule model if needed
        if schedule is not None and isinstance(schedule, dict):
            schedule = Schedule(**schedule)

        # Convert trigger / repeat dicts to strong models if needed
        if trigger is not None and isinstance(trigger, dict):
            trigger = Trigger(**trigger)

        if repeat is not None:
            repeat = [RepeatPattern(**r) if isinstance(r, dict) else r for r in repeat]

        if offline and entrypoint is None:
            raise ValueError("Offline tasks require a numeric entrypoint.")

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
        prev_ptr = sched_prev(schedule)

        if trigger is not None:
            # --------  event-driven task  -------- #
            if status is None:
                status = Status.triggerable
            elif to_status(status) != Status.triggerable:
                raise ValueError(
                    "Tasks with a *trigger* must start in the 'triggerable' state.",
                )

        elif status is None:
            if prev_ptr is not None:
                # Already queued behind another runnable task → never primed
                status = Status.scheduled if future_start else Status.queued
            else:
                # No predecessor pointer – prefer the in-memory primed pointer to avoid an extra read
                if future_start:
                    status = Status.scheduled
                else:
                    primed_exists = (
                        self._primed_task is not None
                        and self._primed_task.status == Status.primed
                    )

                    if self._active_task is None and not primed_exists:
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

        # Allow scheduled tasks with past start_at values; downstream logic may
        # handle these cases (e.g., immediate eligibility) without raising.

        # ------------------  assemble payload  ------------------ #
        # Ensure queue_id presence for queued/scheduled tasks:
        # - If schedule provided, inherit queue_id from predecessor when possible;
        #   otherwise, allocate a fresh queue id (head case).
        derived_qid = None
        prev_task = None
        if queue_id is not None:
            derived_qid = int(queue_id)
        elif schedule is not None:
            prev_tid = sched_prev(schedule)
            if prev_tid is not None:
                try:
                    prev_task = self._get_task_or_raise(int(prev_tid))
                    derived_qid = prev_task.queue_id
                except Exception:
                    derived_qid = None
            if derived_qid is None:
                # Head or standalone scheduled/queued task → allocate new queue id
                derived_qid = self._allocate_new_queue_id()

        task_details = TaskBase(
            name=name,
            description=description,
            status=status,
            schedule=schedule,
            trigger=trigger,
            deadline=deadline,
            repeat=repeat,
            priority=priority,
            response_policy=response_policy,
            queue_id=derived_qid,
            entrypoint=entrypoint,
            offline=offline,
        ).to_post_json()

        # ------------------  write log immediately  ------------------ #
        log = self._view.create_one(entries=task_details, new=True)
        task_id = log.entries["task_id"]
        # Keep the monotonic queue-id allocator in sync with the id we just
        # materialized (if any). This ensures preview calls to
        # _allocate_new_queue_id() do not consume ids and tests that capture a
        # queue id before creation continue to pass.
        if derived_qid is not None:
            self._view.sync_max_queue_id_seen(derived_qid)

        # Maintain cached total count (+1 new row)
        if self._num_tasks_cached is not None:
            self._num_tasks_cached += 1

        # Keep linkage symmetric only when linkage exists; reuse any prefetched
        # neighbour row (e.g., predecessor) to avoid an extra backend read.
        if schedule is not None:
            # from earlier derivation
            prev_tid = sched_prev(schedule)
            if prev_tid is not None and prev_task is not None:
                self._sync_adjacent_links(
                    task_id=task_id,
                    schedule=schedule,
                    prefetched_rows={int(prev_tid): prev_task},
                )

        # ── Ensure the in-memory cache reflects any linkage tweaks ──
        if status == Status.primed:
            # Avoid a backend read: populate primed pointer directly from the created log
            try:
                primed_task = Task(**log.entries)
                self._primed_task = primed_task
            except Exception:
                # Fallback to lazy refresh if direct population fails
                self._refresh_primed_cache(task_id)

        # ------------------  queue insertion (if relevant)  ---------- #
        if status == Status.queued:
            # Only *auto-append* when the caller did **not** supply an
            # explicit linkage (prev/next).  If linkage was given we assume
            # the user knows where the task belongs.
            explicit_linkage = schedule is not None and (
                sched_prev(schedule) is not None or sched_next(schedule) is not None
            )

            if explicit_linkage:
                return {
                    "outcome": "task created successfully",
                    "details": {"task_id": task_id},
                }
            # Creation should not auto-append to any queue.

        return {
            "outcome": "task created successfully",
            "details": {"task_id": task_id},
        }

    def _create_tasks(
        self,
        *,
        tasks: List[Dict[str, Any]],
        queue_ordering: Optional[List[Union[List[int], Dict[str, Any]]]] = None,
    ) -> ToolOutcome:
        """
        Batch‑create tasks with ascending ids and (optionally) materialise one
        or more runnable queues in a single atomic flow.

        Motivation and intended use
        ---------------------------
        Use this tool when the user asks to create a series/chain of new tasks
        (potentially across multiple queues) and to establish their order immediately.
        In one call you get both:
        1) predictable, ascending ``task_id`` assignment matching the provided
           list order; and
        2) explicit single‑queue or multi‑queue ordering for those new tasks.

        Behaviour
        ---------
        - Task identifiers are assigned in ascending order following the order
          of the provided ``tasks`` list (driven by the underlying auto‑increment).
        - When ``queue_ordering`` is supplied, this method creates one or more
          queues that include ONLY the newly created tasks, in the exact
          head→tail order you specify. Fresh backend ``queue_id`` values are
          allocated for each such queue.
        - ``queue_head`` is REQUIRED per queue (see examples below)
        - If any of these tasks are to be added into an *already existing* queue,
          then leave this task out of the ``queue_ordering`` and add to the queue
          later with the dedicated queue manipultation tools.

        Parameters
        ----------
        tasks : list[dict]
            One dict per task mirroring the arguments of ``_create_task``.
            Typical usage is to provide just ``name`` and ``description`` for
            each task and rely on ``queue_ordering`` for ordering.
        queue_ordering : list[dict] | None
            Optional declaration of one or more queues using RELATIVE indices
            into the ``tasks`` list (0‑based; distinct from backend ``queue_id``).
            Each item MUST be a dict of the form:
            - ``{"order": [int, ...], "queue_head": {"start_at": <ISO|datetime>}}``
              → create a queue with the given order and schedule the head.
            - ``{"order": [int, ...], "queue_head": {"primed": true}}``
              → create a queue with a primed head (no timestamp). At most one
                queue in this call may request a primed head, and only when no
                existing task is already primed.

        Returns
        -------
        ToolOutcome
            ``{"outcome": "tasks created", "details": {"task_ids": [...], "queues": [...]}}``
            where ``queues`` lists the allocated ``queue_id`` for each declared
            queue, its ``relative_queue_index`` (0‑based in the input array) and
            the realised ``task_ids`` order.

        Notes
        -----
        - Relative indices in ``queue_ordering`` refer to the position of each
          spec in the ``tasks`` argument. They are not persistent ids (not the
          ``queue_id``).
        - For each declared queue you MUST provide a head policy via
          ``queue_head``: either ``start_at`` (scheduled head) or ``primed``.
          It is not permitted to omit both. Only one queue may request a primed
          head and only when no existing task is already primed.
        - If you need to create tasks without establishing their order yet,
          you may omit ``queue_ordering`` and manipulate queues later via
          dedicated queue tools. When both creation and ordering are requested
          together, prefer this batched tool.
        """

        # Fast path: nothing to do
        if not tasks:
            return {
                "outcome": "tasks created",
                "details": {"task_ids": [], "queues": []},
            }

        # Pre‑validate names/descriptions to avoid partial creation on obvious duplicates
        seen_names: set[str] = set()
        seen_descs: set[str] = set()
        # Defer checking existing primed state until needed (only when queue_ordering is provided).
        for idx, spec in enumerate(tasks):
            name = spec.get("name")
            desc = spec.get("description")
            if not name or not desc:
                raise ValueError(
                    f"Each task spec must include non‑empty 'name' and 'description' (index={idx}).",
                )
            if name in seen_names:
                raise ValueError(
                    f"Duplicate task name in batch: {name!r} (index={idx})",
                )
            if desc in seen_descs:
                raise ValueError(
                    "Duplicate task description in batch – descriptions must be unique: "
                    f"{desc!r} (index={idx})",
                )
            seen_names.add(str(name))
            seen_descs.add(str(desc))

        # Always create tasks sequentially to preserve ascending id assignment
        created_ids: List[int] = []
        for spec in tasks:
            payload: Dict[str, Any] = {}
            for key in (
                "name",
                "description",
                "status",
                "schedule",
                "trigger",
                "deadline",
                "repeat",
                "priority",
                "response_policy",
                "entrypoint",
            ):
                if key in spec:
                    payload[key] = spec[key]

            # When queue_ordering is provided, avoid auto-priming during creation.
            # Defer head-state selection to the explicit queue materialization below.
            if queue_ordering is not None and "status" not in payload:
                payload["status"] = Status.queued

            out = self._create_task(**payload)
            created_ids.append(int(out["details"]["task_id"]))

        queues_report: List[Dict[str, Any]] = []

        if queue_ordering:
            # Single read to determine primed existence for policy validation
            try:
                _primed_existed_before = bool(
                    self._filter_tasks(filter="status == 'primed'", limit=1),
                )
            except Exception:
                _primed_existed_before = self._primed_task is not None
            # Normalise queue_ordering into a list of {order: [...], head_policy: {...}}
            normalised: List[Dict[str, Any]] = []

            def _norm_one(item: Dict[str, Any]) -> Dict[str, Any]:
                if not isinstance(item, dict):
                    raise ValueError(
                        "Each queue specification must be a dict with keys 'order' and 'queue_head'.",
                    )
                order = item.get("order")
                assert isinstance(order, list) and order, (
                    "Each queue spec must contain a non-empty 'order' list of relative indices",
                )
                heads = item.get("queue_head") or {}
                if not isinstance(heads, dict):
                    raise ValueError(
                        "queue_head must be an object with either 'start_at' or 'primed'.",
                    )
                has_start = heads.get("start_at") is not None
                has_primed = bool(heads.get("primed"))
                if has_start == has_primed:
                    raise ValueError(
                        "queue_head must specify exactly one of {'start_at', 'primed'} per queue.",
                    )
                return {
                    "order": list(order),
                    "head_policy": {
                        "start_at": heads.get("start_at"),
                        "primed": has_primed,
                    },
                }

            normalised = [_norm_one(x) for x in queue_ordering]

            # Enforce at most one primed head across all queues and none if one already exists
            primed_requests = sum(
                1 for q in normalised if q["head_policy"].get("primed")
            )
            if primed_requests > 1:
                raise ValueError(
                    "At most one queue may request a primed head in a single call.",
                )
            if _primed_existed_before and primed_requests == 1:
                raise ValueError(
                    "Cannot create a primed head when another task is already primed. Choose start_at or clear the primed task first.",
                )

            # If we plan to have a primed head and there was no primed task prior
            # to this batch, demote any auto-primed rows among the newly created
            # tasks to 'queued' before applying the requested head policy.
            if primed_requests == 1 and not _primed_existed_before:
                try:
                    auto_primed_tasks = self._filter_tasks(
                        filter=f"task_id in {created_ids} and status == 'primed'",
                    )
                except Exception:
                    auto_primed_tasks = []
                for t in auto_primed_tasks:
                    try:
                        self._update_task_status(
                            task_ids=t.task_id,
                            new_status=Status.queued,
                        )
                    except Exception:
                        pass

            used_indices: set[int] = set()
            for rel_qidx, qspec in enumerate(normalised):
                indices: List[int] = [int(i) for i in qspec.get("order", [])]
                # Validate indices
                for i in indices:
                    if i < 0 or i >= len(created_ids):
                        raise ValueError(
                            f"queue_ordering references out‑of‑range task index {i}; valid range is 0..{len(created_ids)-1}",
                        )
                    if i in used_indices:
                        raise ValueError(
                            f"Task at relative index {i} is referenced by more than one queue in queue_ordering.",
                        )
                used_indices.update(indices)

                # Map relative indices to real task ids in the provided order
                order_ids: List[int] = [created_ids[i] for i in indices]

                # Allocate a fresh numeric queue id and materialise the queue
                qid = self._allocate_new_queue_id()
                head_policy = qspec.get("head_policy", {})
                start_at_value = head_policy.get("start_at")
                self._set_queue(
                    queue_id=qid,
                    order=order_ids,
                    queue_start_at=start_at_value,
                )
                # If head should be primed, explicitly set its status to primed
                if head_policy.get("primed"):
                    if not order_ids:
                        raise ValueError(
                            "Queue 'order' must include at least one task to mark head as primed.",
                        )
                    self._update_task_status(
                        task_ids=order_ids[0],
                        new_status=Status.primed,
                    )
                queues_report.append(
                    {
                        "relative_queue_index": rel_qidx,
                        "queue_id": qid,
                        "task_ids": order_ids,
                    },
                )

        return {
            "outcome": "tasks created",
            "details": {"task_ids": created_ids, "queues": queues_report},
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
        # Fast path: if we know the backing log id for this task, delete directly
        # Resolve the log id via a single lookup then delete (LocalTaskView manages memoization)
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        self._view.delete(logs=log_id)
        try:
            removed_count = (
                len(log_id)
                if isinstance(log_id, list)
                else (1 if log_id is not None else 0)
            )
        except Exception:
            removed_count = 0

        # Maintain cached total count (subtract removed rows)
        try:
            if self._num_tasks_cached is not None and removed_count:
                self._num_tasks_cached = max(
                    0,
                    int(self._num_tasks_cached) - int(removed_count),
                )
        except Exception:
            pass
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
        # Guard against touching the active task (fast in‑memory check)
        self._ensure_not_active_task(task_ids)

        # Single targeted read for the referenced tasks only with a minimal field projection
        # (avoid scanning all completed tasks and avoid fetching unused columns within this call).
        logs = self._view.get_minimal_rows_by_task_ids(
            task_ids=task_ids,
            fields=["status"],
        )
        # Validate none of the referenced tasks are already completed
        try:
            completed_ids = {
                int(lg.entries["task_id"])
                for lg in logs
                if str(lg.entries.get("status")) == "completed"
            }
        except Exception:
            completed_ids = set()
        overlap = set(task_ids).intersection(completed_ids)
        assert not overlap, (
            "Cannot cancel completed tasks. Attempted to cancel: " f"{overlap}"
        )

        # Batch update status using the resolved log ids directly (no extra reads)
        self._write_log_entries(
            logs=[lg.id for lg in logs],
            entries={"status": Status.cancelled},
        )
        return {
            "outcome": "tasks cancelled",
            "details": {"task_ids": task_ids},
        }

    # Update Task Queue

    # --------------------  small helpers  -------------------- #

    def _extract_start_at(self, sched):
        """Return the start_at value from a Schedule model or plain dict, or None.

        This helper intentionally performs a light-touch extraction without
        coercion to preserve existing behaviour; the invariant checks only need
        to know whether a timestamp is present.
        """
        if sched is None:
            return None
        if isinstance(sched, Schedule):
            return sched.start_at
        try:
            return sched.get("start_at")
        except Exception:
            return None

    def _sync_adjacent_links(
        self,
        *,
        task_id: int,
        schedule: ScheduleLike,
        prefetched_rows: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> None:
        """Delegate to queue-utils to maintain symmetric neighbour links."""
        _q_sync_adjacent_links(
            self,
            task_id=task_id,
            schedule=schedule,
            prefetched_rows=prefetched_rows,
        )

    # ────────────────────────────────────────────────────────────────────
    # Multi-queue helpers (public tools for the update loop)
    # ────────────────────────────────────────────────────────────────────

    def _allocate_new_queue_id(self) -> int:
        """Return a fresh integer queue identifier via LocalTaskView."""
        return self._view.allocate_new_queue_id()

    def _list_queues(self) -> List[Dict[str, Any]]:
        """
        List all runnable queues. Every queue must have a numeric ``queue_id``.

        Returns
        -------
        list[dict]
            One entry per queue head with keys:
            - ``queue_id`` (int): identifier of the queue.
            - ``queue_label`` (str): human‑readable label ("Q<N>").
            - ``head_id`` (int): task id of the head.
            - ``size`` (int): number of runnable tasks in the queue.
            - ``start_at`` (str | None): ISO timestamp from the head's schedule.

        Notes
        -----
        Queues are explicit by `queue_id`. This method lists queues whose heads
        are non-terminal tasks with `schedule.prev_task is None` and a numeric
        `queue_id`.
        """
        # Fast-path via LocalTaskView cache.
        try:
            summaries = self._view.get_all_queue_summaries()
        except Exception:
            summaries = []
        if summaries:
            out_fast: list[Dict[str, Any]] = []
            for s in summaries:
                out_fast.append(
                    {
                        "queue_id": s.queue_id,
                        "queue_label": f"Q{s.queue_id}",
                        "head_id": s.order[0],
                        "size": len(s.order),
                        "start_at": s.start_at,
                    },
                )
            if out_fast:
                return out_fast

        tasks = [
            t
            for t in self._filter_tasks()
            if t.schedule is not None and t.status not in self._TERMINAL_STATUSES
        ]

        # Single-pass index for constant-time next lookups within this tool call
        tasks_by_ids: Dict[int, Task] = {t.task_id: t for t in tasks}

        # Heads are rows with prev_task == None
        heads: list[Task] = [
            t for t in tasks if t.schedule is not None and t.schedule.prev_task is None
        ]

        out: list[Dict[str, Any]] = []
        # Prepare fresh caches reconstructed from the single read above
        new_queue_index: Dict[int, List[int]] = {}
        new_task_to_queue: Dict[int, int] = {}
        new_head_start_at: Dict[int, Optional[str]] = {}
        for h in heads:
            qid = h.queue_id
            if qid is None:
                continue

            start_at = (
                h.schedule_start_at.isoformat()
                if h.schedule_start_at is not None
                else None
            )
            # Compute chain size purely in-memory to avoid extra backend reads
            size = 0
            seen: set[int] = set()
            cur = h
            order_for_q: list[int] = []
            while cur is not None:
                task_id = cur.task_id
                if task_id is not None:
                    if task_id in seen:
                        break
                    seen.add(task_id)
                    size += 1
                    order_for_q.append(task_id)
                nxt = cur.schedule_next
                if nxt is None:
                    break
                cur = tasks_by_ids.get(nxt)

            # Update reconstructed caches
            if order_for_q:
                new_queue_index[qid] = list(order_for_q)
                for _tid in order_for_q:
                    new_task_to_queue[_tid] = qid
                new_head_start_at[qid] = start_at

            out.append(
                {
                    "queue_id": qid,
                    "queue_label": f"Q{qid}",
                    "head_id": h.task_id,
                    "size": size,
                    "start_at": start_at,
                },
            )

        # Best-effort: refresh the LocalTaskView cache for future fast-paths
        try:
            self._view.refresh_queue_index_from_rows(tasks)
        except Exception:
            pass

        return out

    def _get_queue(
        self,
        *,
        queue_id: Optional[int] = None,
    ) -> List[Task]:
        """
        Return the runnable queue for a given ``queue_id`` (head→tail).

        Parameters
        ----------
        queue_id : int | None, default ``None``
            Identifier of the queue. When ``None``, no implicit default is
            assumed; this returns an empty list.

        Returns
        -------
        list[Task]
            Ordered tasks from head to tail. Returns an empty list when the
            queue does not exist or contains no runnable tasks.

        Notes
        -----
        - This method operates on explicit queues only; pass a numeric `queue_id`.
        """
        if queue_id is None:
            return []

        # Fast-path via LocalTaskView with a single minimal read.
        member_ids = list(self._view.get_member_ids(queue_id) or [])

        if member_ids:
            fields_needed: List[str] = self._queue_member_fields()
            rows_by_id: Dict[int, Dict[str, Any]] = self._read_rows_by_ids(
                ids=member_ids,
                fields=fields_needed,
            )

            ordered: List[Task] = []
            for tid in member_ids:
                row = rows_by_id.get(int(tid))
                if row is None:
                    continue
                st = to_status(row.get("status"))  # type: ignore[arg-type]
                if st in self._TERMINAL_STATUSES:
                    continue
                row = self._sanitize_activation(row)
                ordered.append(Task(**row))
            return ordered

        # Fallback: single filtered read of all runnable rows in this queue
        tasks_in_queue: List[Task] = [
            r
            for r in self._filter_tasks(
                filter=(
                    "schedule is not None and "
                    "status not in ('completed','cancelled','failed') and "
                    f"queue_id == {int(queue_id)}"
                ),
            )
        ]
        if not tasks_in_queue:
            return []

        # Identify head with tolerance for terminal/missing predecessor.
        # A row is a head if: prev_task is None OR prev_task not present among non-terminal members.
        ids_in_q: set[int] = set(t.task_id for t in tasks_in_queue)

        head_candidates: list[Task] = []
        prefer_none_prev: list[Task] = []
        for task in tasks_in_queue:
            prev_id = task.schedule_prev
            if prev_id is None:
                prefer_none_prev.append(task)
                head_candidates.append(task)
            else:
                if prev_id not in ids_in_q:
                    head_candidates.append(task)

        if not head_candidates:
            return []

        # If multiple, prefer a true None-prev head; else choose deterministically by task_id
        head_task: Task
        if len(head_candidates) > 1:
            if prefer_none_prev:
                head_task = prefer_none_prev[0]
            else:
                head_task = sorted(
                    head_candidates,
                    key=lambda x: x.task_id,
                )[0]
        else:
            head_task = head_candidates[0]

        # Build id -> row map for O(1) next lookups without further backend reads
        task_by_id: Dict[int, Task] = {t.task_id: t for t in tasks_in_queue}

        # Walk head→tail using next_task pointers in-memory
        ordered: List[Task] = []
        seen: set[int] = set()
        current_task = head_task
        while current_task is not None:
            if current_task.task_id in seen:
                break

            seen.add(current_task.task_id)

            # Strip stale activation metadata on non-active rows
            ordered.append(self._sanitize_activation(current_task))

            next_task = current_task.schedule_next
            if next_task is None:
                break
            current_task = task_by_id.get(next_task)

        return ordered

    def _walk_queue_from_task(self, *, task_id: int) -> List[Task]:
        """
        Walk the chain that contains `task_id` by following schedule.prev_task to
        the head and then schedule.next_task forward, returning rows as `Task`.

        This helper ignores the top-level queue_id and is used when a task does
        not carry a numeric queue_id but still belongs to a linked chain.
        """
        # Locate the starting row
        try:
            current_task = self._get_task_or_raise(task_id)
        except Exception:
            return []

        # Walk to head using prev_task pointers
        head_task = current_task
        while head_task is not None:
            prev_id = head_task.schedule_prev
            if prev_id is None:
                break
            prev_task = self._filter_tasks(
                filter=f"task_id == {prev_id}",
                limit=1,
            )
            head_task = prev_task[0] if prev_task else None

        if head_task is None:
            return []

        # Walk forward using next_task pointers; include terminal rows for context
        ordered: List[Task] = []
        current_task: Task | None = head_task
        seen: set[int] = set()
        while current_task is not None:
            task_id = current_task.task_id
            if task_id in seen:
                break
            seen.add(task_id)
            # Strip stale activation metadata on non-active rows
            ordered.append(self._sanitize_activation(current_task))
            next_id = current_task.schedule_next
            if next_id is None:
                break
            next_task = self._filter_tasks(filter=f"task_id == {next_id}", limit=1)
            current_task = next_task[0] if next_task else None

        return ordered

    def _get_queue_for_task(self, *, task_id: int) -> List[Task]:
        """
        Return the runnable queue (head→tail) that contains ``task_id``.

        Use this read‑only tool to retrieve the live chain of runnable tasks that
        the given task participates in, preserving the actual execution order.
        It inspects state only and never creates or mutates rows.

        Parameters
        ----------
        task_id : int
            Identifier of a task whose current runnable chain should be returned.

        Returns
        -------
        list[Task]
            Ordered tasks from head to tail. Returns an empty list when the task
            has no runnable queue or the chain cannot be reconstructed.
        """
        # Strategy
        # ---------
        # - Fast-path: when a local queue index is available and not marked stale,
        # resolve `queue_id` and delegate to `_get_queue(queue_id=…)`.
        # - Otherwise, read the single row; if it carries a numeric `queue_id`,
        # delegate to `_get_queue(queue_id=…)`; else fall back to
        # `_walk_queue_from_task` which ignores `queue_id` and follows links.

        # Fast-path via LocalTaskView when membership is known.
        try:
            qid_cached = self._view.get_queue_id_for_task(task_id)
        except Exception:
            qid_cached = None
        if qid_cached is not None:
            members = self._view.get_member_ids(qid_cached)
            if task_id in members:
                return self._get_queue(queue_id=qid_cached)

        # Fallback: resolve via storage
        try:
            task = self._get_task_or_raise(task_id)
        except Exception:
            return []

        queue_id = task.queue_id
        if queue_id is not None:
            return self._get_queue(queue_id=queue_id)

        # No numeric queue_id – follow the linked chain defensively
        return self._walk_queue_from_task(task_id=task_id)

    def _reorder_queue(
        self,
        *,
        queue_id: Optional[int],
        new_order: List[int],
    ) -> ToolOutcome:
        """
        Reorder a single queue to exactly match ``new_order`` (head→tail).

        When to use
        -----------
        - You want to change the order of tasks that are ALREADY members of a
          single queue. This method does not move tasks across queues.

        Parameters
        ----------
        queue_id : int | None
            Target queue identifier.
        new_order : list[int]
            Complete desired order of all runnable tasks within this queue.
            This tool does not add or remove tasks across queues; every task in
            the current queue must appear exactly once in ``new_order``.

        Behaviour
        ---------
        - Maintains neighbour pointers symmetrically.
        - Ensures exactly one head owns ``start_at`` (preserves the queue-level
          timestamp when present) and sets statuses consistently:
            • head with ``start_at`` → ``scheduled``;
            • non-heads → at most ``queued``.
        - The active task (if any) in this queue retains its ``active`` status.

        Notes
        -----
        - Tasks executed in isolation are detached from their queues; do not
          include detached tasks in `new_order` for that queue.
        - This method asserts that `new_order` is an exact permutation of the current queue;
          if you see an assertion error, refresh state and reconstruct `new_order` accordingly.
        """
        # Resolve current membership once (prefer local index; fallback to storage)
        queue_id_exists = queue_id is not None
        member_ids = self._view.get_member_ids(queue_id) if queue_id_exists else []

        in_queue_tasks: list[Task] = []
        if not member_ids:
            # Single filtered read of runnable rows in this queue
            filter_expr = "schedule is not None and status not in ('completed','cancelled','failed') "
            if queue_id_exists:
                filter_expr += f"and queue_id == {queue_id}"
            in_queue_tasks = self._filter_tasks(filter=filter_expr)
            member_ids = [t.task_id for t in in_queue_tasks]

        # Validate permutation
        current_set: set[int] = {t for t in member_ids}
        if current_set != set(new_order):
            raise AssertionError(
                "new_order must be a permutation of the current queue. "
                f"Current members: {sorted(list(current_set))}; "
                f"Provided: {sorted(list(set(new_order)))}. "
                f"Refresh with list_queues() and get_queue(queue_id={queue_id}) "
                "then rebuild new_order accordingly.",
            )

        # Minimal fields required for planning and comparison
        minimal_fields = ["task_id", "status", "schedule"]
        rows_by_id: Dict[int, Dict[str, Any]] = self._read_rows_by_ids(
            ids=member_ids,
            fields=minimal_fields,
        )
        # Trim to ids we actually care about (defensive and deterministic order)
        rows_by_id = {tid: rows_by_id.get(tid, {}) for tid in new_order}

        # Compute invariant-preserving plan
        updates_per_log: Dict[int, Dict[str, Any]] = plan_reorder_queue(
            new_order=new_order,
            rows_by_id=rows_by_id,
        )

        # Build tid→log_id map once
        id_map: Dict[int, int] = self._task_id_to_log_id_map(list(new_order))

        # Filter out no-op writes and batch the rest
        to_write_ids: list[int] = []
        to_write_entries: list[Dict[str, Any]] = []
        for tid in new_order:
            payload = updates_per_log.get(tid) or {}
            cur_row = rows_by_id.get(tid) or {}
            cur_sched = {**(cur_row.get("schedule") or {})}
            desired_sched = {**(payload.get("schedule") or {})}
            need_status = False
            try:
                existing_status = to_status(cur_row.get("status"))
                desired_status = to_status(payload.get("status", existing_status))
                need_status = existing_status != desired_status
            except Exception:
                need_status = "status" in payload
            if (cur_sched == desired_sched) and (not need_status):
                continue
            lid = id_map.get(int(tid))
            if isinstance(lid, int):
                to_write_ids.append(int(lid))
                to_write_entries.append(payload)

        if to_write_ids:
            self._write_log_entries(logs=to_write_ids, entries=to_write_entries)

        # Auto-checkpoint after successful edit (best-effort)
        cid = short_id(8)
        snap = {"label": "auto:_reorder_queue", "queues": []}
        head_start = self._head_start_at_from_rows(list(rows_by_id.values()))
        snap["queues"].append(
            {
                "queue_id": queue_id,
                "head_id": new_order[0] if new_order else None,
                "start_at": head_start,
                "order": list(new_order),
            },
        )
        self._queue_checkpoints[cid] = snap
        last_checkpoint_id = cid  # noqa: F841

        # Best-effort: refresh LocalTaskView
        if queue_id_exists:
            self._view.update_after_reorder(
                queue_id=queue_id,
                new_order=list(new_order),
                head_start_at=head_start,
            )

        return {
            "outcome": "queue reordered",
            "details": {
                "queue_id": queue_id,
                "new_order": new_order,
                "checkpoint_id": last_checkpoint_id,
            },
        }

        # Keep local queue index in sync (best-effort)
        # Intentionally left to call-sites post-persistence when needed.

    def _move_tasks_to_queue(
        self,
        *,
        task_ids: List[int],
        queue_id: Optional[int] = None,
        position: Optional[str] = "back",
    ) -> ToolOutcome:
        """
        Move one or more runnable tasks to a specific queue and position.

        This implementation minimizes backend calls. Within a single tool call, the backend state is assumed stable.

        Returns
        -------
        ToolOutcome
            {"outcome": "tasks moved", "details": {"queue_id": <int>, "task_ids": [...]}}
        """
        # Normalize inputs and guard against moving the active task
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        # Deduplicate while preserving order
        block = list(dict.fromkeys(int(t) for t in task_ids))
        if not block:
            return {
                "outcome": "tasks moved",
                "details": {"queue_id": queue_id, "task_ids": []},
            }
        self._ensure_not_active_task(block)

        # Validate existence, reject terminal/trigger-based; single consolidated read
        tasks = self._filter_tasks(filter=f"task_id in {block}")
        ids_found = {t.task_id for t in tasks}
        missing = [tid for tid in block if tid not in ids_found]
        assert not missing, f"Unknown task ids: {missing}"
        for t in tasks:
            assert (
                t.status not in self._TERMINAL_STATUSES
            ), f"Task {t.task_id} is terminal"
            if t.trigger is not None:
                raise ValueError(
                    f"Task {t.task_id} is trigger-based and cannot be placed in the queue.",
                )

        # Determine each task's current queue (prefer local index; reuse prefetched rows)
        source_qid_by_tid: Dict[int, Optional[int]] = {}
        for _task_id in block:
            _queue_id = self._view.get_queue_id_for_task(_task_id)
            if _queue_id is not None:
                source_qid_by_tid[_task_id] = _queue_id
        # For any remaining ids, reuse the single consolidated read done above
        missing = [t for t in block if t not in source_qid_by_tid]
        if missing:
            tasks_by_ids = {t.task_id: t for t in tasks}
            for t in missing:
                task = tasks_by_ids.get(t)
                if task is not None:
                    source_qid_by_tid[t] = task.queue_id

        # Allocate target queue id when requested (new queue)
        target_qid = queue_id if queue_id is not None else self._allocate_new_queue_id()
        # Keep monotonic allocator in sync when caller specifies a higher id
        if target_qid is not None:
            self._view.sync_max_queue_id_seen(target_qid)

        # Build target queue's existing order once (prefer local index)
        try:
            if target_qid is not None:
                tgt_existing = list(self._view.get_member_ids(target_qid) or [])
            else:
                tgt_existing = []
            if not tgt_existing:
                raise RuntimeError
        except Exception:
            try:
                tgt_existing = [t.task_id for t in self._get_queue(queue_id=target_qid)]
            except Exception:
                tgt_existing = []

        # Compose new target order based on requested position
        tgt_base = [tid for tid in tgt_existing if tid not in block]
        if position == "front":
            tgt_new_order = block + tgt_base
        elif position == "back":
            tgt_new_order = tgt_base + block
        else:
            tgt_new_order = tgt_base + block

        # For each source queue (excluding target), compute reduced order
        source_qids: set[int] = {
            q_id
            for q_id in (source_qid_by_tid.get(t) for t in block)
            if q_id is not None
        }
        if target_qid is not None and target_qid in source_qids:
            source_qids.discard(target_qid)

        def _current_order_for(qid: int) -> List[int]:
            try:
                cached = self._view.get_member_ids(qid)
                if cached is not None and len(cached) > 0:
                    return list(cached)
                raise RuntimeError
            except Exception:
                return [t.task_id for t in self._get_queue(queue_id=qid)]

        source_orders: Dict[int, List[int]] = {}
        for q_id in list(source_qids):
            try:
                cur = _current_order_for(q_id)
            except Exception:
                cur = []
            if not cur:
                continue
            reduced = [tid for tid in cur if tid not in block]
            source_orders[q_id] = reduced

        # Materialize edits: source queues first (detach cleanly), then target queue
        checkpoint_id = None
        for _queue_id, cur_order in source_orders.items():
            # Skip when no member actually moved
            try:
                if cur_order == _current_order_for(q_id):
                    continue
            except Exception:
                pass
            # Use core primitive to preserve head start_at and status semantics reliably
            self._set_queue(queue_id=_queue_id, order=cur_order)

        if tgt_new_order:
            # Apply target queue materialization in one batch using the core primitive
            set_res = self._set_queue(queue_id=target_qid, order=tgt_new_order)
            target_qid = set_res.get("details", {}).get("queue_id", target_qid)
            checkpoint_id = set_res.get("details", {}).get("checkpoint_id")

        return {
            "outcome": "tasks moved",
            "details": {
                "queue_id": target_qid,
                "task_ids": list(block),
                "checkpoint_id": checkpoint_id,
            },
        }

    # ------------------------------------------------------------------ #
    #  Atomic queue materialization                                       #
    # ------------------------------------------------------------------ #

    def _set_queue(
        self,
        *,
        queue_id: Optional[int],
        order: List[int],
        queue_start_at: Optional[str] = None,
    ) -> ToolOutcome:
        """
        Materialize a complete queue ordering in a single atomic step.

        Behaviour
        ---------
        - Moves all ``order`` tasks into the target queue (creating it if needed).
        - Removes any other runnable tasks from this queue.
        - Rewires neighbour links to match ``order`` exactly.
        - Applies ``queue_start_at`` to the head only when provided; otherwise
          preserves existing head start_at if present; non-heads never carry
          start_at.
        - Adjusts statuses: head with start_at → scheduled; others at most queued.

        When to use
        -----------
        - Declare an entire chain in a single call (especially after creation),
          avoiding iterative move/reorder loops. This is the preferred tool for
          building or resetting the exact membership/order of a queue.
        """

        # Normalize and validate ids; ensure tasks exist and are not terminal
        if not isinstance(order, list) or not order:
            return {
                "outcome": "queue set",
                "details": {"queue_id": queue_id, "order": []},
            }

        tasks: List[Task] = self._filter_tasks(filter=f"task_id in {order}")
        ids_found = {t.task_id for t in tasks}
        missing = [tid for tid in order if tid not in ids_found]
        assert not missing, f"Unknown task ids: {missing}"
        for task in tasks:
            assert (
                task.status not in self._TERMINAL_STATUSES
            ), f"Task {task.task_id} is terminal"
        # Reject placing trigger-based tasks into a runnable queue
        for task in tasks:
            if task.trigger is not None:
                raise ValueError(
                    f"Task {task.task_id} is trigger-based and cannot be placed in the queue.",
                )
        # Build a one-shot rows map for reuse throughout this tool call
        tasks_by_ids: Dict[int, Task] = {t.task_id: t for t in tasks}
        # Allow editing a queue that includes the currently active task; preserve its status below
        active_tid: Optional[int] = None
        if self._active_task is not None:
            active_tid = self._active_task.task_id

        # Allocate queue id when needed
        target_qid = queue_id if queue_id is not None else self._allocate_new_queue_id()
        # Keep the allocator in sync when a new queue becomes materialized here
        try:
            if target_qid is not None:
                self._view.sync_max_queue_id_seen(target_qid)
        except Exception:
            pass

        # Capture existing head-level start_at BEFORE any mutations so it can be
        # restored onto the new head reliably (avoids losing it during neutralisation)
        existing_head_start: Optional[str] = None

        # When the caller explicitly supplies queue_start_at, we don't need to read
        # the current head. Similarly, when we can confidently determine that the
        # target queue is a fresh, empty queue, we avoid an unnecessary backend read.
        assume_empty_target_queue = False
        try:
            if isinstance(queue_id, int):
                cached_members = self._view.get_member_ids(int(queue_id))
                assume_empty_target_queue = not bool(cached_members)
        except Exception:
            assume_empty_target_queue = False

        # Remove any other members currently in the target queue (strict by queue_id)
        current_members: List[int] = []
        current_tasks_by_ids: Dict[int, Task] = {}

        # Prefer a single filtered read of the target queue to derive both:
        # - existing head start_at (when queue_start_at not provided), and
        # - current membership to compute removals.
        if queue_id is not None and not assume_empty_target_queue:
            try:
                tasks_in_queue: List[Task] = self._filter_tasks(
                    filter=(
                        "schedule is not None and "
                        "status not in ('completed','cancelled','failed') and "
                        f"queue_id == {target_qid}"
                    ),
                )
            except Exception:
                tasks_in_queue = []

            # Derive current members and by-id map from the same read
            current_members: List[int] = [r.task_id for r in tasks_in_queue]
            current_tasks_by_ids: Dict[int, Task] = {
                t.task_id: t for t in tasks_in_queue
            }

            # Compute existing head start_at locally to avoid an unfiltered scan
            if queue_start_at is None and existing_head_start is None:
                for task in tasks_in_queue:
                    if task.schedule_prev is None:
                        existing_head_start = task.schedule_start_at
                        break

        # Fallback using already-fetched rows when available (only scans 'order')
        if queue_start_at is None and existing_head_start is None:
            for _tid in order:
                _task = tasks_by_ids.get(_tid)
                if not _task:
                    continue
                if _task.schedule_start_at is not None and _task.schedule_prev is None:
                    existing_head_start = _task.schedule_start_at
                    break

        to_remove: List[int] = [tid for tid in current_members if tid not in order]

        if to_remove:
            # Detach removed tasks in a single backend call: neutral schedule, queued status, no queue_id
            try:
                log_ids = self._get_logs_by_task_ids(task_ids=to_remove)
                self._write_log_entries(
                    logs=log_ids,
                    entries={
                        "schedule": {},
                        "status": Status.queued,
                        "queue_id": None,
                    },
                )
            except Exception:
                # Fallback to per-task validated writes if batch update fails for any reason
                for tid in to_remove:
                    task = current_tasks_by_ids.get(tid) or self._get_task_or_raise(
                        tid,
                    )
                    self._validated_write(
                        task_id=tid,
                        entries={
                            "schedule": {},
                            "status": Status.queued,
                            "queue_id": None,
                        },
                        err_prefix=f"While clearing removed task {tid} from queue {target_qid}:",
                        current_row=task,
                    )

        # Rewire links to match order and apply head start_at (single write per member)
        # Accumulate member writes and persist in one batch to minimize I/O
        entries_by_tid: Dict[int, Dict[str, Any]] = {}
        for idx, tid in enumerate(order):
            prev_tid = None if idx == 0 else order[idx - 1]
            next_tid = None if idx == len(order) - 1 else order[idx + 1]
            sched = {
                "prev_task": prev_tid,
                "next_task": next_tid,
            }
            if idx == 0:
                # Prefer provided queue_start_at; else preserve the existing head start
                if queue_start_at is not None:
                    sched["start_at"] = queue_start_at
                elif existing_head_start is not None:
                    sched["start_at"] = existing_head_start

            # Prepare entries; derive status centrally and avoid writing 'active'
            write_entries: Dict[str, Any] = {"schedule": sched, "queue_id": target_qid}

            # Fetch current row once for status derivation and no-op detection
            task = tasks_by_ids.get(tid) or self._get_task_or_raise(tid)
            existing_status = task.status
            is_head = idx == 0
            head_has_start_at = "start_at" in sched

            if not (active_tid is not None and tid == active_tid):
                # Centralized derivation; non-head cannot remain primed
                derived_status = derive_status_after_queue_edit(
                    existing_status=existing_status,
                    is_head=is_head,
                    head_has_start_at=head_has_start_at,
                )
                if (not is_head) and derived_status == Status.primed:
                    derived_status = Status.queued
                # Only include status when it actually changes and is not 'active'
                if existing_status != derived_status:
                    write_entries["status"] = derived_status

            # Skip no-op writes when the current row already matches the desired state
            try:
                same_sched = (
                    task.schedule_prev == sched["prev_task"]
                    and task.schedule_next == sched["next_task"]
                    and (task.schedule_start_at == sched.get("start_at"))
                )
                same_qid = task.queue_id == target_qid
                if "status" in write_entries:
                    same_status = task.status == write_entries["status"]
                else:
                    # When active, we never change status in this call
                    same_status = True
                if same_sched and same_qid and same_status:
                    continue
            except Exception:
                pass
            entries_by_tid[int(tid)] = write_entries

        if entries_by_tid:
            # Prefer a single batched write; fall back internally to per-task
            self._write_entries_batched(entries_by_tid=entries_by_tid)

        # No additional start_at write needed – applied on head above

        # Auto-checkpoint (avoid extra reads by using local state)
        cid = short_id(8)
        snap = {"label": "auto:_set_queue", "queues": []}
        order_now = list(order)
        # Prefer explicit queue_start_at; else preserve the captured head start
        head_start = (
            queue_start_at if queue_start_at is not None else existing_head_start
        )
        snap["queues"].append(
            {
                "queue_id": target_qid,
                "head_id": order_now[0] if order_now else None,
                "start_at": head_start,
                "order": order_now,
            },
        )
        self._queue_checkpoints[cid] = snap
        last_checkpoint_id = cid  # noqa: F841

        # Best-effort: refresh LocalTaskView membership mapping
        if target_qid is not None:
            _head_start_local = (
                queue_start_at if queue_start_at is not None else existing_head_start
            )
            self._view.update_after_queue_materialized(
                queue_id=target_qid,
                order=order,
                head_start_at=_head_start_local,
            )

        return {
            "outcome": "queue set",
            "details": {
                "queue_id": target_qid,
                "order": list(order),
                "checkpoint_id": last_checkpoint_id,
            },
        }

        # Keep local queue index in sync (best-effort)
        # Intentionally left to call-sites post-persistence when needed.

    # ------------------------------------------------------------------ #
    #  Bulk low-level schedule edit (atomic)                              #
    # ------------------------------------------------------------------ #

    def _set_schedules_atomic(
        self,
        *,
        schedules: List[Dict[str, Any]],
    ) -> ToolOutcome:
        """
        Apply multiple schedule edits atomically with graph validation.

        Each item: {"task_id": int, "schedule": {"queue_id": int | None,
        "prev_task": int | None, "next_task": int | None, "start_at": str}}

        Validation:
        - All referenced tasks must exist and be non-terminal/non-active.
        - No cross-queue adjacency (neighbours must share queue_id).
        - Exactly one head per connected chain; no cycles.
        - Only heads may have start_at.
        - Status normalization: head with start_at → scheduled; others ≤ queued.

        When to use
        -----------
        - Advanced scenarios where you need fine-grained control over adjacency
          across multiple tasks at once and can provide a consistent, validated
          graph in one shot. Prefer :pyfunc:`_set_queue` for common materialization
          cases.
        """

        if not schedules:
            return {"outcome": "schedules updated", "details": {"count": 0}}

        # Build a local view and validate cross-refs with MINIMAL backend I/O
        # 1) Normalise input → by_id (tid → schedule dict). Ignore nested queue_id.
        by_id: Dict[int, Dict[str, Any]] = {}
        for item in schedules:
            tid = int(item.get("task_id"))
            sch = dict(item.get("schedule", {}))
            sch.pop("queue_id", None)
            by_id[tid] = sch

        # 2) Single read for all target rows
        tasks = self._filter_tasks(filter=f"task_id in {list(by_id.keys())}")
        ids_found = {t.task_id for t in tasks}
        missing = [tid for tid in by_id.keys() if tid not in ids_found]
        assert not missing, f"Unknown task ids: {missing}"
        for task in tasks:
            assert (
                task.status not in self._TERMINAL_STATUSES
            ), f"Task {task.task_id} is terminal"
        self._ensure_not_active_task(list(by_id.keys()))

        # Local map for quick access to current rows
        tasks_by_ids: Dict[int, Task] = {t.task_id: t for t in tasks}

        # 3) Precompute any top-level queue_id provided in the payload (once)
        provided_qid: Dict[int, Optional[int]] = {}
        for item in schedules:
            try:
                provided_qid[int(item.get("task_id"))] = item.get("queue_id")
            except Exception:
                continue

        # 4) Batch-resolve external neighbours referenced by the new schedules
        external_neighbours: set[int] = set()
        for _tid, _sch in by_id.items():
            for _k in ("prev_task", "next_task"):
                _nbr = _sch.get(_k)
                if _nbr is None:
                    continue
                try:
                    _nbr_int = int(_nbr)
                    if _nbr_int not in by_id:
                        external_neighbours.add(_nbr_int)
                except Exception:
                    continue
        if external_neighbours:
            ext_tasks = self._filter_tasks(
                filter=f"task_id in {list(external_neighbours)}",
            )
            for t in ext_tasks:
                tasks_by_ids[t.task_id] = t

        # 5) Build a queue_id lookup covering targets + any external neighbours
        qid_for_tid: Dict[int, Optional[int]] = {}
        for tid in list(tasks_by_ids.keys()):
            if tid in provided_qid and provided_qid.get(tid) is not None:
                qid_for_tid[tid] = provided_qid.get(tid)
            else:
                task = tasks_by_ids.get(tid)
                qid_for_tid[tid] = task.queue_id if task else None

        # 6) Cross-queue guard using only the prefetched state
        graph: Dict[int, List[int]] = {tid: [] for tid in by_id.keys()}
        for tid, sch in by_id.items():
            cur_qid = qid_for_tid.get(int(tid))
            for nbr_key in ("prev_task", "next_task"):
                nbr = sch.get(nbr_key)
                if nbr is None:
                    continue
                nbr = int(nbr)
                nbr_qid = qid_for_tid.get(nbr)
                if nbr_qid != cur_qid:
                    raise ValueError(
                        f"Cross-queue link rejected: task {tid} (qid={cur_qid}) → {nbr_key}={nbr} (qid={nbr_qid}).",
                    )
                graph[int(tid)].append(nbr)

        # 7) Cycle validation within the provided graph
        visited: Dict[int, int] = {}
        temp: set[int] = set()

        def _dfs(u: int):
            if u in temp:
                raise ValueError("Cycle detected in provided schedules")
            if u in visited:
                return
            temp.add(u)
            for v in graph.get(u, []):
                _dfs(v)
            temp.remove(u)
            visited[u] = 1

        for u in graph.keys():
            _dfs(u)

        # 8) Head/start_at rule: start_at only on heads
        for tid, sch in by_id.items():
            prev_tid = sch.get("prev_task")
            if sch.get("start_at") is not None and prev_tid is not None:
                raise ValueError(f"Only heads may define start_at (task {tid})")

        # 9) Apply atomically: reuse current_row and skip cross-queue guard (already validated)
        for tid, sch in by_id.items():
            is_head = sch.get("prev_task") is None
            head_has_start_at = sch.get("start_at") is not None

            task = tasks_by_ids.get(int(tid))
            existing_status = task.status if task else Status.queued
            desired_status = derive_status_after_queue_edit(
                existing_status=existing_status,
                is_head=is_head,
                head_has_start_at=head_has_start_at,
            )
            # Non-head cannot remain primed
            if (not is_head) and desired_status == Status.primed:
                desired_status = Status.queued

            top_qid = provided_qid.get(int(tid))
            if top_qid is None:
                top_qid = task.queue_id if task else None

            entries = {
                "schedule": sch,
                **({"queue_id": int(top_qid)} if isinstance(top_qid, int) else {}),
            }
            if existing_status != to_status(desired_status):
                entries["status"] = desired_status

            self._validated_write(
                task_id=int(tid),
                entries=entries,
                err_prefix=f"While applying set_schedules_atomic (task {tid}):",
                current_row=task,
                skip_cross_queue_guard=True,
            )

        return {"outcome": "schedules updated", "details": {"count": len(by_id)}}

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    def _partition_queue(
        self,
        *,
        parts: List[Dict[str, Any]],
        strategy: str = "preserve_order",
    ) -> ToolOutcome:
        """
        Split the current runnable queue into multiple smaller queues.

        Parameters
        ----------
        parts : list[dict]
            Each item describes one output queue with keys:
            - ``task_ids`` (list[int], required): tasks that should form this queue.
            - ``queue_start_at`` (str | datetime | None, optional): when set, the
              head of this queue will carry this ``start_at`` and the queue head
              status becomes ``scheduled``; otherwise it remains ``queued``.
            - ``queue_name`` (str | None, optional): unused metadata (accepted for future use).

            The first part is applied to the identified source queue. Subsequent
            parts are materialised as new queues (fresh ``queue_id``s).

        strategy : {"preserve_order", "as_list"}
            - ``preserve_order`` (default): within each part, preserve the relative
              order as found in the original source queue;
            - ``as_list``: use the exact ``task_ids`` order provided for each part.

        Behaviour
        ---------
        - Tasks mentioned in later parts are detached from the source queue and
          moved to newly created queues.
        - The original source queue is reduced to the tasks in the first part.
        - Queue-level ``start_at`` is set from each part's ``queue_start_at`` (if
          provided); otherwise, the original source queue's timestamp is retained
          on the new head of the first part only.

        Notes
        -----
        This tool is designed for readability in the update loop when the user
        asks for sequences like "do [0,2] tomorrow and [1,3] the day after".
        """
        # Determine the source queue id:
        # - Prefer the queue that contains the first task listed in the first part
        # - Fallback to source queue heuristics when unavailable
        source_qid: Optional[int] = None
        try:
            if parts and parts[0].get("task_ids"):
                _head_tid = int(parts[0]["task_ids"][0])
                _head_task = self._get_task_or_raise(_head_tid)
                source_qid = _head_task.queue_id
        except Exception:
            source_qid = None

        # Current queue snapshot (head→tail) and head start_at for the identified source queue.
        # Fetch once to avoid multiple backend calls in this tool invocation.
        tasks_in_q = self._get_queue(queue_id=source_qid)
        original = [t.task_id for t in tasks_in_q]
        if not original:
            return {"outcome": "queue partitioned", "details": {"queues": []}}

        # Normalise per-part order
        def _ordered(ids: List[int]) -> List[int]:
            if strategy == "as_list":
                return list(ids)
            # preserve original relative order
            rank = {tid: i for i, tid in enumerate(original)}
            return sorted(ids, key=lambda x: rank.get(x, 10**9))

        # Remember original queue-level timestamp (head.start_at) without extra reads
        queue_start_ts = None
        try:
            if tasks_in_q:
                _head_sched = getattr(tasks_in_q[0], "schedule", None)
                _val = (
                    getattr(_head_sched, "start_at", None)
                    if _head_sched is not None
                    else None
                )
                if _val is not None:
                    if hasattr(_val, "isoformat"):
                        queue_start_ts = _val.isoformat()
                    else:
                        queue_start_ts = _val
        except Exception:
            queue_start_ts = None

        # 1) Move all tasks not in the first part out first (to avoid complex rewiring)
        first_ids = set(parts[0].get("task_ids", [])) if parts else set()
        rest_ids = [tid for tid in original if tid not in first_ids]

        # For later parts, compute their target membership
        later_parts = parts[1:] if len(parts) > 1 else []
        # Map each tid in rest_ids to the index of its part (relative to later_parts)
        part_map: Dict[int, int] = {}
        for idx, p in enumerate(later_parts):
            for tid in p.get("task_ids", []):
                part_map[int(tid)] = idx

        # Group by part
        groups: Dict[int, List[int]] = {}
        for tid in rest_ids:
            j = part_map.get(tid)
            if j is None:
                continue
            groups.setdefault(j, []).append(tid)

        created: list[Dict[str, Any]] = []

        # Pre-allocate a fresh queue id ONCE, then increment locally for subsequent queues
        next_qid: Optional[int] = None
        if groups:
            next_qid = self._allocate_new_queue_id()

        for j, tids in groups.items():
            ordered = _ordered(tids)
            qid = next_qid if next_qid is not None else self._allocate_new_queue_id()
            # Advance next_qid for the next created queue within this tool call
            if next_qid is not None:
                next_qid = qid + 1
            # Materialize the new queue in one step via core primitive; pass start_at when provided
            qstart = later_parts[j].get("queue_start_at")
            self._set_queue(queue_id=qid, order=ordered, queue_start_at=qstart)
            # Update LocalTaskView for the new queue
            try:
                self._view.update_after_queue_materialized(
                    queue_id=int(qid),
                    order=list(ordered),
                    head_start_at=qstart,
                )
            except Exception:
                pass
            created.append(
                {"queue_id": qid, "task_ids": ordered, "queue_start_at": qstart},
            )

        # 2) Reduce the source queue to the first part (in chosen order)
        first_list = _ordered(list(first_ids))
        # Reorder source queue to include only these tasks: move out everything else already done
        if first_list:
            # apply provided start_at or carry the original one
            fstart = parts[0].get("queue_start_at") if parts else None
            if fstart is None:
                fstart = queue_start_ts
            self._set_queue(
                queue_id=source_qid,
                order=first_list,
                queue_start_at=fstart,
            )
            # Update LocalTaskView for the reduced source queue
            try:
                if source_qid is not None:
                    self._view.update_after_queue_materialized(
                        queue_id=source_qid,
                        order=list(first_list),
                        head_start_at=fstart,
                    )
            except Exception:
                pass
        else:
            fstart = None

        details = {"default_queue": first_list, "new_queues": created}
        # Auto-checkpoint after successful edit (best-effort): capture only touched queues to avoid extra reads
        cid = short_id(8)
        snap = {"label": "auto:_partition_queue", "queues": []}
        # Source queue snapshot (if any)
        if first_list:
            snap["queues"].append(
                {
                    "queue_id": source_qid,
                    "head_id": first_list[0] if first_list else None,
                    "start_at": fstart,
                    "order": list(first_list),
                },
            )
        # Newly created queues
        for created_q in created:
            _order = list(created_q.get("task_ids", []))
            snap["queues"].append(
                {
                    "queue_id": created_q.get("queue_id"),
                    "head_id": _order[0] if _order else None,
                    "start_at": created_q.get("queue_start_at"),
                    "order": _order,
                },
            )
        self._queue_checkpoints[cid] = snap
        last_checkpoint_id = cid  # noqa: F841

        return {
            "outcome": "queue partitioned",
            "details": {
                **details,
                "checkpoint_id": last_checkpoint_id,
            },
        }

    # ------------------------------------------------------------------ #
    #  Centralised schedule/status write with invariant validation        #
    # ------------------------------------------------------------------ #

    def _validated_write(
        self,
        *,
        task_id: int,
        entries: Dict[str, Any],
        err_prefix: str,
        current_row: Optional[TaskBase] = None,
        skip_sync: bool = False,
        skip_cross_queue_guard: bool = False,
    ) -> Dict[str, str]:
        """
        Single funnel for writing schedule/status that enforces invariants and
        keeps neighbour links symmetric.

        When to use
        -----------
        - Internal helper used by all queue/schedule mutations. External callers
          should prefer the public tools (`_set_queue`, `_reorder_queue`,
          `_move_tasks_to_queue`, `_set_schedules_atomic`) instead of calling
          this method directly.
        """
        # Fast-path: when NOT touching lifecycle/queue fields we can update directly
        # without reading the current row or running invariant checks.
        touches_lifecycle = any(
            k in entries for k in ("schedule", "status", "trigger", "queue_id")
        )

        if not touches_lifecycle:
            # Resolve log id via a single lookup, then write entries
            log_id = self._get_logs_by_task_ids(task_ids=task_id)
            return self._write_log_entries(logs=log_id, entries=entries)

        current = current_row or self._get_task_or_raise(task_id)

        prospective_schedule = entries.get("schedule", current.schedule)
        prospective_status = entries.get("status", current.status)
        prospective_trigger = entries.get("trigger", current.trigger)

        # Belt-and-braces: forbid setting status to 'active' via this funnel.
        norm_status = None
        if "status" in entries:
            try:
                norm_status = to_status(entries["status"])  # type: ignore[arg-type]
            except Exception:
                norm_status = None
        if norm_status == Status.active:
            raise ValueError(
                f"{err_prefix} direct writes to 'active' are not allowed; use the execution method instead.",
            )

        self._validate_scheduled_invariants(
            status=prospective_status,
            schedule=prospective_schedule,
            trigger=prospective_trigger,
            err_prefix=err_prefix,
        )

        # Trigger-based tasks must not be members of runnable queues or have start_at.
        # Enforce against both schedule linkage and assigning a numeric queue_id.
        try:
            _sched_dict = (
                prospective_schedule.model_dump()
                if hasattr(prospective_schedule, "model_dump")
                else dict(prospective_schedule or {})
            )
        except Exception:
            _sched_dict = prospective_schedule

        has_membership: bool = False
        try:
            if isinstance(_sched_dict, dict):
                if (
                    _sched_dict.get("prev_task") is not None
                    or _sched_dict.get("next_task") is not None
                    or _sched_dict.get("start_at") is not None
                ):
                    has_membership = True
        except Exception:
            has_membership = False

        assigned_qid = entries.get("queue_id")
        assigns_queue = isinstance(assigned_qid, int)

        if prospective_trigger is not None and (has_membership or assigns_queue):
            raise ValueError(
                f"{err_prefix} trigger-based tasks cannot be placed in the runnable queue or scheduled. "
                "Remove the trigger first, or clear schedule/queue.",
            )

        # Cross-queue adjacency guard: when setting prev/next ensure neighbours share queue_id
        prefetched = None
        if ("schedule" in entries and prospective_schedule is not None) and (
            not skip_cross_queue_guard
        ):
            try:
                _sched = (
                    prospective_schedule.model_dump()
                    if hasattr(prospective_schedule, "model_dump")
                    else dict(prospective_schedule)
                )
            except Exception:
                _sched = prospective_schedule
            try:
                qid = entries.get("queue_id") if isinstance(entries, dict) else None
                prev_tid = (_sched or {}).get("prev_task")
                next_tid = (_sched or {}).get("next_task")

                # Batch-resolve neighbour queue ids in a single read
                neighbour_ids = [int(t) for t in (prev_tid, next_tid) if t is not None]
                tasks_by_ids: Dict[int, Task] = {}
                if neighbour_ids:
                    try:
                        tasks = self._filter_tasks(
                            filter=f"task_id in {neighbour_ids}",
                        )
                    except Exception:
                        tasks = []
                    for t in tasks:
                        tasks_by_ids[t.task_id] = t

                # Only enforce when linkage exists
                for _nbr, _tid in (("prev_task", prev_tid), ("next_task", next_tid)):
                    if _tid is None:
                        continue
                    try:
                        neighbour_task = tasks_by_ids.get(int(_tid))
                        neighbour_qid = (
                            neighbour_task.queue_id if neighbour_task else None
                        )
                    except Exception:
                        neighbour_qid = None
                    if neighbour_qid != qid:
                        raise ValueError(
                            f"{err_prefix} cross-queue link rejected: {_nbr}={_tid} has queue_id={neighbour_qid} "
                            f"but current task would be in queue_id={qid}. Use set_queue() or move_tasks_to_queue() "
                            f"followed by reorder_queue() to materialize chains within a single queue.",
                        )
                # Store for neighbour symmetry reuse below
                prefetched = tasks_by_ids
            except Exception:
                # Defensive: do not block writes on guard failure paths; the invariant validator will still run
                pass

        # If caller supplied a queue_id alongside schedule, it becomes the source of truth
        if (
            "schedule" in entries
            and "queue_id" in entries
            and entries["queue_id"] is not None
        ):
            # ensure type
            try:
                entries["queue_id"] = int(entries["queue_id"])
            except Exception:
                pass

        # Resolve log id via a single lookup
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        result = self._write_log_entries(logs=log_id, entries=entries)

        # Ensure neighbour symmetry whenever schedule changed (unless skipped by caller)
        if ("schedule" in entries) and (not skip_sync):
            self._sync_adjacent_links(
                task_id=task_id,
                schedule=prospective_schedule,
                prefetched_rows=prefetched,
            )

        return result

    # ------------------------------------------------------------------ #
    #  Centralised helpers for queue link manipulation                    #
    # ------------------------------------------------------------------ #

    _TERMINAL_STATUSES = {Status.completed, Status.cancelled, Status.failed}

    # ------------------------------------------------------------------ #
    #  Public lifecycle helpers                                           #
    # ------------------------------------------------------------------ #

    def _reinstate_to_previous_queue(
        self,
        *,
        task_id: int,
        allow_active: bool = False,
    ) -> ToolOutcome:
        # Facade retained for internal callers; the update tool exposes
        # `_reinstate_task_to_previous_queue`, which carries the full docstring.
        return self._reintegration_manager.apply(
            task_id=task_id,
            allow_active=allow_active,
        )

    def _refresh_primed_cache(self, task_id: Optional[int] = None) -> None:
        """
        Reload the primed task pointer from storage.

        Parameters
        ----------
        task_id : int | None, default ``None``
            When ``None``, refresh the currently cached primed task (if any).
            Otherwise load the row for ``task_id`` and promote it to the cache.
        """
        if task_id is None and self._primed_task is not None:
            task_id = self._primed_task.task_id
        if task_id is None:
            return

        tasks = self._filter_tasks(filter=f"task_id == {task_id}", limit=1)
        task = tasks[0] if tasks else None
        # Only cache when the referenced row is actually in 'primed' state
        if task is not None and task.status == Status.primed:
            self._primed_task = task
        else:
            self._primed_task = None

    # (moved) _select_final_neighbours: now defined in reintegration.py

    # Update Task(s) Status / Schedule / Deadline / Repetition / Priority

    def _update_task_status(
        self,
        *,
        task_ids: Union[int, List[int]],
        new_status: Status,
    ) -> Dict[str, str]:
        """
        Change the **lifecycle status** of one or many tasks.

        Notes:
        - Setting status to 'active' directly is forbidden. Activation is performed
          exclusively by the execution path (execute / execute_by_id).
        """
        # Forbid making anything active (unless explicitly allowed)
        if new_status == Status.active:
            raise ValueError(
                "Direct status changes to 'active' are not allowed; use the dedicated activation method.",
            )

        # Forbid touching the existing active task (always) – lifecycle changes
        # to the active task must go through the live ActiveTask handle.
        self._ensure_not_active_task(task_ids)

        # Invariant checks for queue/schedule-sensitive statuses
        if new_status in (Status.scheduled, Status.queued):
            tasks = self._filter_tasks(filter=f"task_id in {task_ids}")
            for task in tasks:
                self._validate_scheduled_invariants(
                    status=new_status,
                    schedule=task.schedule,
                    err_prefix=f"While changing status of task {task.task_id}:",
                )

        log_ids = self._get_logs_by_task_ids(task_ids=task_ids)
        entries: Dict[str, Any] = {"status": new_status}
        return self._write_log_entries(logs=log_ids, entries=entries)

    def _update_task(
        self,
        *,
        task_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[Union["Status", str]] = None,
        start_at: Optional[Union[str, "datetime"]] = None,
        deadline: Optional[Union[str, "datetime"]] = None,
        repeat: Optional[List[Union["RepeatPattern", Dict[str, Any]]]] = None,
        priority: Optional[Union["Priority", str]] = None,
        trigger: Any = _UNSET,
        entrypoint: Any = _UNSET,
        offline: Any = _UNSET,
    ) -> Dict[str, Any]:
        """
        Update one or more fields of an existing task.

        Parameters
        ----------
        task_id : int
            Identifier of the task to modify.
        name : str | None
            New task name.
        description : str | None
            New task description.
        status : Status | str | None
            Lifecycle status. Setting to 'active' is not allowed here.
        start_at : datetime | str | None
            Queue head start timestamp. Only valid when the task is at the head
            (no prev_task). Mutually exclusive with trigger.
        deadline : datetime | str | None
            Hard deadline.
        repeat : list[RepeatPattern | dict] | None
            Replacement repetition rules. Use an empty list to clear.
        priority : Priority | str | None
            Importance level.
        trigger : Trigger | dict | None
            Replacement trigger. Mutually exclusive with any schedule/start_at.
        offline : bool | None
            Toggle whether the task should execute in the hidden offline lane.
            Offline tasks must retain a numeric ``entrypoint``.

        Returns
        -------
        dict
            Confirmation payload from the write operation.
        """

        # Forbid edits on the currently active task via scheduler APIs
        self._ensure_not_active_task(task_id)

        # Was 'trigger' explicitly provided (even if None)?
        _trigger_provided = trigger is not _UNSET
        _offline_provided = offline is not _UNSET

        # Fetch current task for invariants/derivations
        task = self._get_task_or_raise(task_id)

        # No-op guard – allow updates when at least one field is provided OR when
        # the caller explicitly provided 'trigger' (even if None, meaning clear it).
        if (
            name is None
            and description is None
            and status is None
            and start_at is None
            and deadline is None
            and repeat is None
            and priority is None
            and not _trigger_provided
            and entrypoint is _UNSET
            and not _offline_provided
        ):
            raise ValueError("At least one field must be provided for an update.")

        # Mutually exclusive guard: trigger with any schedule/start_at
        if _trigger_provided and trigger is not None:
            # If the update itself adds a start_at or the current schedule is present, reject
            if start_at is not None:
                raise ValueError("Cannot set a trigger alongside a start_at schedule.")
            if task.schedule is not None:
                raise ValueError(
                    "Cannot add a trigger while a schedule exists. Remove schedule first.",
                )

        # Build prospective schedule if start_at is supplied
        schedule_payload: Optional[Dict[str, Any]] = None
        if start_at is not None:
            # Disallow start_at when the task is trigger-based
            # Allow when the update explicitly clears the trigger in the same call
            if task.trigger is not None and not (_trigger_provided and trigger is None):
                raise ValueError(
                    "Cannot add/update *start_at* – the task is trigger-based.",
                )
            # Guard-rail: tasks with a predecessor cannot own start_at
            if task.schedule_prev is not None:
                raise ValueError(
                    "Cannot set 'start_at' when the task has 'prev_task'. Move it to the queue head first.",
                )
            # Coerce datetime to ISO-8601 string if needed
            if isinstance(start_at, datetime):
                start_at = start_at.isoformat()  # type: ignore[assignment]
            schedule_payload = {
                "prev_task": task.schedule_prev,
                "next_task": task.schedule_next,
                "start_at": start_at,
            }

        # Determine desired status
        desired_status: Optional["Status"] = None
        if status is not None:
            # Forbid forcing 'active'
            status_enum = to_status(status)  # type: ignore[arg-type]
            if status_enum == Status.active:
                raise ValueError(
                    "Direct status changes to 'active' are not allowed; use the execution method.",
                )
            desired_status = status_enum
        else:
            # Infer from trigger/start_at when caller didn't specify a status
            if _trigger_provided and trigger is not None:
                desired_status = Status.triggerable
            elif (
                schedule_payload is not None
                and schedule_payload.get("start_at") is not None
            ):
                desired_status = Status.scheduled

        # Validate queue/schedule invariants when status or start_at provided
        if desired_status is not None or schedule_payload is not None:
            self._validate_scheduled_invariants(
                status=(desired_status if desired_status is not None else task.status),
                schedule=(
                    schedule_payload if schedule_payload is not None else task.schedule
                ),
                err_prefix=f"While updating task {task_id}:",
            )

        # Compose entries
        entries: Dict[str, Any] = {}
        if name is not None:
            entries["name"] = name
        if description is not None:
            entries["description"] = description
        if deadline is not None:
            entries["deadline"] = deadline
        if repeat is not None:
            # Normalise RepeatPattern objects to plain dicts
            norm_repeat: List[Dict[str, Any]] = []
            for r in repeat:
                if isinstance(r, RepeatPattern):
                    norm_repeat.append(r.model_dump())  # type: ignore[assignment]
                else:
                    norm_repeat.append(dict(r))  # type: ignore[arg-type]
            entries["repeat"] = norm_repeat
        if priority is not None:
            entries["priority"] = priority
        if _trigger_provided:
            if trigger is None:
                entries["trigger"] = None
            else:
                if isinstance(trigger, dict):
                    trigger = Trigger(**trigger)
                entries["trigger"] = trigger.model_dump()
        if schedule_payload is not None:
            entries["schedule"] = schedule_payload
        if desired_status is not None:
            entries["status"] = desired_status
        # entrypoint set/clear
        if entrypoint is not _UNSET:
            # allow None to clear it
            if entrypoint is None:
                entries["entrypoint"] = None
            else:
                try:
                    entries["entrypoint"] = int(entrypoint)
                except Exception:
                    raise ValueError("entrypoint must be an integer or None")
        if _offline_provided:
            if isinstance(offline, str):
                normalized_offline = offline.strip().lower()
                if normalized_offline in {"true", "1"}:
                    offline = True
                elif normalized_offline in {"false", "0"}:
                    offline = False
                else:
                    raise ValueError("offline must be a boolean value")
            else:
                offline = bool(offline)
            entries["offline"] = offline

        desired_entrypoint = task.entrypoint
        if "entrypoint" in entries:
            desired_entrypoint = entries["entrypoint"]
        desired_offline = task.offline
        if "offline" in entries:
            desired_offline = bool(entries["offline"])
        if desired_offline and desired_entrypoint is None:
            raise ValueError("Offline tasks require a numeric entrypoint.")

        # If clearing a trigger (trigger explicitly None) and current status is triggerable
        if (
            _trigger_provided
            and (trigger is None)
            and (status is None)
            and task.status == Status.triggerable
        ):
            # Downgrade to queued when trigger removed (and not setting start_at)
            if schedule_payload is None:
                entries["status"] = Status.queued

        # Persist via central validated funnel when schedule/status involved; otherwise a direct write is fine
        if ("schedule" in entries) or ("status" in entries):
            # Provide queue_id when we know it to avoid an extra guard read and ensure consistency
            if ("schedule" in entries) and ("queue_id" not in entries):
                qid = task.queue_id
                if qid is not None:
                    entries["queue_id"] = qid
            # When we did not change adjacency (prev/next unchanged), skip neighbour sync and cross-queue guard
            _skip_sync = False
            _skip_cross_guard = False
            try:
                if "schedule" in entries and isinstance(entries.get("schedule"), dict):
                    _new = entries["schedule"]
                    _skip_sync = (
                        _new.get("prev_task") == task.schedule_prev
                        and _new.get("next_task") == task.schedule_next
                    )
                    _skip_cross_guard = _skip_sync
            except Exception:
                _skip_sync = False
                _skip_cross_guard = False
            return self._validated_write(
                task_id=task_id,
                entries=entries,
                err_prefix=f"While updating task {task_id}:",
                # Reuse the row we already fetched in this tool call to avoid a second backend read
                current_row=task,
                skip_sync=_skip_sync,
                skip_cross_queue_guard=_skip_cross_guard,
            )
        else:
            log_id = self._get_logs_by_task_ids(task_ids=task_id)
            return self._write_log_entries(logs=log_id, entries=entries)

    def _update_task_instance(
        self,
        *,
        task_id: int,
        instance_id: int,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Validate and update fields for a specific task instance (task_id + instance_id).
        Supports lifecycle fields such as status and info.

        Raises:
            ValueError: If the instance doesn't exist or if updates violate invariants.
        """
        # Ensure we are not trying to update the *currently* active task pointer directly
        # (though this method is usually called *after* it finishes).
        if (
            self._active_task is not None
            and self._active_task.task_id == task_id
            and self._active_task.instance_id == instance_id
        ):
            return {
                "outcome": "skipped",
                "reason": "Cannot update active task instance directly",
            }

        # Find the specific log for this instance
        log_objs = self._view.get_rows(
            filter=f"task_id == {task_id} and instance_id == {instance_id}",
            limit=1,
            return_ids_only=False,
        )
        if not log_objs:
            raise ValueError(
                f"No task instance found for task_id={task_id}, instance_id={instance_id}",
            )

        log_to_update = log_objs[0]
        current_row = dict(log_to_update.entries)
        entries_to_write = {}
        current_sched = current_row.get("schedule") or {}

        if "status" in kwargs:
            new_status = to_status(kwargs["status"])
            if new_status == Status.active:
                raise ValueError("Direct status changes to 'active' are not allowed.")
            self._validate_scheduled_invariants(
                status=new_status,
                schedule=current_sched,
                trigger=current_row.get("trigger"),
                err_prefix=f"While updating instance {task_id}.{instance_id}:",
            )
            entries_to_write["status"] = new_status

        # If 'info' is being updated
        if "info" in kwargs:
            entries_to_write["info"] = kwargs["info"]

        if not entries_to_write:
            return {
                "outcome": "no changes",
                "details": {"task_id": task_id, "instance_id": instance_id},
            }

        result = self._write_log_entries(
            logs=log_to_update.id,
            entries=entries_to_write,
        )

        if "status" in entries_to_write:
            new_status_written = entries_to_write["status"]
            if (
                self._primed_task
                and self._primed_task.task_id == task_id
                and self._primed_task.instance_id == instance_id
            ):
                if new_status_written != Status.primed:
                    self._primed_task = None
            if new_status_written in (
                Status.completed,
                Status.cancelled,
                Status.failed,
            ):
                self._view.mark_queue_changed()

        return result

    # ────────────────────────────────────────────────────────────────────
    # Small internal helpers
    # ────────────────────────────────────────────────────────────────────

    # ------------------------------------------------------------------ #
    #  Queue plan + checkpoints (shared helpers exposed as tools)        #
    # ------------------------------------------------------------------ #

    def validate_queue_plan(self, *, plan: Dict[str, Any] | str) -> Dict[str, Any]:  # type: ignore[valid-type]
        """Validate a proposed queue plan (dict or JSON string) and return the normalised structure."""

        try:
            parsed = json.loads(plan) if isinstance(plan, str) else plan
        except Exception as _e:  # noqa: N806
            raise ValueError(f"Invalid plan: {_e}")
        model = QueuePlan.model_validate(parsed)
        preview: Dict[str, Any] = {"now": model.now, "later": []}
        for g in model.later_groups:
            preview["later"].append(
                {"task_ids": list(g.task_ids), "queue_start_at": g.queue_start_at},
            )
        return {
            "outcome": "validated",
            "details": {"plan": model.model_dump(), "preview": preview},
        }

    def apply_queue_plan(self, *, plan: Dict[str, Any] | str) -> Dict[str, Any]:  # type: ignore[valid-type]
        """Apply a validated queue plan atomically using invariant-preserving tools and checkpoint."""

        try:
            parsed = json.loads(plan) if isinstance(plan, str) else plan
        except Exception as _e:
            raise ValueError(f"Invalid plan: {_e}")
        model = QueuePlan.model_validate(parsed)
        if model.later_groups:
            parts = [{"task_ids": list(model.now)}] + [
                {"task_ids": list(g.task_ids), "queue_start_at": g.queue_start_at}
                for g in model.later_groups
            ]
            self._partition_queue(parts=parts, strategy="preserve_order")
        else:
            if model.now:
                self._reorder_queue(queue_id=None, new_order=list(model.now))
        cp = self.checkpoint_queue_state(label="post-apply-plan")
        return {
            "outcome": "applied",
            "details": {"checkpoint_id": cp["details"]["checkpoint_id"]},
        }

    def checkpoint_queue_state(self, *, label: Optional[str] = None) -> Dict[str, Any]:
        """Create a session-scoped checkpoint snapshot of all runnable queues."""
        snapshot: Dict[str, Any] = {"label": label, "queues": []}
        try:
            all_queues = self._list_queues()
        except Exception:
            all_queues = []
        for q in all_queues:
            qid = q.get("queue_id")
            start_at = q.get("start_at")
            try:
                order = [t.task_id for t in self._get_queue(queue_id=qid)]
            except Exception:
                order = []
            snapshot["queues"].append(
                {
                    "queue_id": qid,
                    "head_id": q.get("head_id"),
                    "start_at": start_at,
                    "order": order,
                },
            )

        cid = short_id(8)
        self._queue_checkpoints[cid] = snapshot
        return {"outcome": "checkpointed", "details": {"checkpoint_id": cid}}

    def revert_to_checkpoint(self, *, checkpoint_id: str) -> Dict[str, Any]:
        """Revert all queues to a previously created checkpoint."""
        snap = self._queue_checkpoints.get(str(checkpoint_id))
        assert snap is not None, f"Unknown checkpoint_id={checkpoint_id}"
        for q in snap.get("queues", []):
            qid = q.get("queue_id")
            order = list(q.get("order", []) or [])
            if order:
                self._reorder_queue(queue_id=qid, new_order=order)
                start_at = q.get("start_at")
                if start_at is not None:
                    head_tid = int(order[0])
                    try:
                        head_task = self._get_task_or_raise(head_tid)
                        sched = head_task.schedule or Schedule(
                            start_at=start_at,
                        )
                        self._validated_write(
                            task_id=head_tid,
                            entries={"schedule": sched, "status": Status.scheduled},
                            err_prefix="While restoring queue start_at from checkpoint:",
                        )
                    except Exception:
                        pass
        return {"outcome": "reverted", "details": {"checkpoint_id": checkpoint_id}}

    def get_latest_checkpoint(self) -> Dict[str, Any]:
        """Return the most recently created checkpoint id and label (or persisted latest when enabled)."""
        try:
            keys = list(self._queue_checkpoints.keys())
            if keys:
                cid = keys[-1]
                snap = self._queue_checkpoints.get(cid, {})
                return {
                    "outcome": "ok",
                    "details": {"checkpoint_id": cid, "label": snap.get("label")},
                }
            return {
                "outcome": "none",
                "details": {"checkpoint_id": None, "label": None},
            }
        except Exception:
            return {
                "outcome": "none",
                "details": {"checkpoint_id": None, "label": None},
            }

    # Default tool-policy helpers
    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require search_tasks on the first step (if enabled); auto thereafter."""
        from unity.settings import SETTINGS

        if (
            SETTINGS.FIRST_ASK_TOOL_IS_SEARCH
            and step_index < 1
            and "search_tasks" in current_tools
        ):
            return (
                "required",
                {"search_tasks": current_tools["search_tasks"]},
            )
        return ("auto", current_tools)

    @staticmethod
    def _default_update_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require ask on the first step (if enabled); auto thereafter."""
        from unity.settings import SETTINGS

        if (
            SETTINGS.FIRST_MUTATION_TOOL_IS_ASK
            and step_index < 1
            and "ask" in current_tools
        ):
            return ("required", {"ask": current_tools["ask"]})
        return ("auto", current_tools)

    # ------------------------------------------------------------------ #
    #  Small centralised write helper                                     #
    # ------------------------------------------------------------------ #

    def _write_log_entries(
        self,
        *,
        logs: Union[int, "unify.Log", List[Union[int, "unify.Log"]]],
        entries: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Centralised adapter for log writes. Keeps all mutation calls going
        through one place in the scheduler.
        """
        return self._view.write_entries(logs=logs, entries=entries)

    # ------------------------------------------------------------------ #
    #  (removed) checkpoint persistence                                   #
    # ------------------------------------------------------------------ #

    # ────────────────────────────────────────────────────────────────────
    # Small DRY helpers used by ask/update flows
    # ────────────────────────────────────────────────────────────────────

    def _start_loop(
        self,
        client: "unillm.AsyncUnify",
        text: str,
        tools: ToolsDict,
        *,
        loop_id: str,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        log_steps: bool = True,
        tool_policy: Optional[
            Union[
                Literal["default"],
                Callable[[int, Dict[str, Any]], tuple[str, Dict[str, Any]]],
            ]
        ] = None,
        handle_cls: Optional["type[SteerableToolHandle]"] = None,
        response_format: Optional[Type[BaseModel]] = None,
        clarification_queues: Optional[Tuple["asyncio.Queue", "asyncio.Queue"]] = None,
    ) -> SteerableToolHandle:
        """Centralised wrapper around start_async_tool_loop."""
        return start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=loop_id,
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            log_steps=log_steps,
            tool_policy=tool_policy,
            handle_cls=handle_cls,
            response_format=response_format,
            clarification_queues=clarification_queues,
        )

    def _wrap_result_with_messages(
        self,
        handle: SteerableToolHandle,
        client: "unillm.AsyncUnify",
    ) -> SteerableToolHandle:
        """Wrap handle.result() so it returns (answer, client.messages)."""
        original_result = handle.result

        async def wrapped_result():
            answer = await original_result()
            return answer, client.messages

        handle.result = wrapped_result  # type: ignore[assignment]
        return handle

    def _get_task_or_raise(self, task_id: int) -> Task:
        """Fetch exactly one task by id or raise ValueError."""
        tasks = self._filter_tasks(filter=f"task_id == {task_id}", limit=1)
        if not tasks:
            raise ValueError(f"No task found with id={task_id}")
        return tasks[0]

    # Reinstate a previously isolated-and-activated task back to its prior queue position

    def _reinstate_task_to_previous_queue(
        self,
        *,
        task_id: int,
        _allow_active: bool = False,
    ) -> ToolOutcome:
        """
        Reinstate a previously isolated task to its prior queue position.

        Use this write tool when a task was executed in isolation (detached from
        its queue) and should be reattached to the original queue at the stored
        position, preserving neighbour links and queue‑level semantics. This does
        not alter task content; it only restores membership/order.

        Parameters
        ----------
        task_id : int
            Identifier of the task to reinstate into its previous queue.
        _allow_active : bool, default ``False``
            Internal/testing flag that permits reintegration even if the task is
            currently marked ``active``.

        Returns
        -------
        ToolOutcome
            Structured result with outcome and details (e.g., restored queue_id
            and position) suitable for logging or follow‑up edits.
        """
        # Delegate to the reintegration manager; accepts `_allow_active` for tests/callers
        return self._reintegration_manager.apply(
            task_id=task_id,
            allow_active=_allow_active,
        )

    # ------------------------------------------------------------------ #
    #  Public helper – execute by id                                      #
    # ------------------------------------------------------------------ #

    async def execute_by_id(
        self,
        *,
        task_id: int,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        """
        Public entrypoint to start execution at a specific task id using queue semantics.

        Behaviour mirrors the numeric-id fast path in `execute(text)` and returns
        an ActiveQueue handle that will adopt and continue the sequence behind the
        specified task (followers remain attached).
        """
        # Refuse execution when a task is already active.
        if self._active_task is not None:
            raise RuntimeError("Another task is already running – stop it first.")

        # Also guard against orphan 'active' rows (e.g., after crash) even if pointer is None.
        try:
            any_active = any(
                t.status == Status.active
                for t in self._filter_tasks(filter="status == 'active'", limit=1)
            )
        except Exception:
            any_active = False
        if any_active:
            raise RuntimeError(
                "A task is marked as active, but no active handle is present – reconcile state before starting another task.",
            )

        return await self._execute_queue_internal(
            task_id=task_id,
            parent_chat_context=_parent_chat_context,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
        )

    # Search Across Tasks

    def _search_tasks(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Task]:
        """
        Semantic search across tasks using one or more reference texts.

        Parameters
        ----------
        references : dict[str, str]
            Mapping of ``source_expr → reference_text`` terms. Each source expression
            can be a plain column (e.g. ``"name"``) or a derived expression.
        k : int, default ``10``
            Maximum number of results to return. Must be <= 1000.

        Returns
        -------
        list[Task]
            Up to ``k`` matching tasks sorted by ascending cosine distance. If the
            similarity search yields fewer than ``k`` results and there are more
            than ``k`` tasks overall, the remainder is backfilled from
            ``unify.get_logs(limit=k)`` in returned order, skipping duplicates.
        """
        # Use a minimal field projection to reduce backend payload size.
        # Only include fields required to construct a Task model.
        allowed_fields: List[str] = [
            "task_id",
            "instance_id",
            "name",
            "description",
            "status",
            "priority",
            # Include schedule/deadline so callers can answer date-related questions
            # without issuing an additional read per task.
            "schedule",
            "deadline",
        ]

        # Single shared helper: semantic similarity with safe backfill
        filled = table_search_top_k(
            self._ctx,
            references,
            k=k,
            allowed_fields=allowed_fields,
            row_filter=None,
            unique_id_field="task_id",
        )
        return [Task(**lg) for lg in filled]

    def _filter_tasks(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Task]:
        """
        Run a **column-wise Python expression** (`filter`) against every task
        and return the matching rows.

        Use this tool only for exact, equality, inequality, membership checks and
        column-wise filtering (e.g. id or equality checks).

        Parameters
        ----------
        filter : str | None, default ``None``
            Any valid Python boolean expression referencing column names as
            variables, e.g. ``"status == 'queued' and priority == 'high'"``.
            *None* selects **all** tasks.
        offset : int, default ``0``
            Zero-based row offset for pagination.
        limit : int, default ``100``
            Maximum number of rows to return. Must be <= 1000.

        Returns
        -------
        list[dict]
            Entries for each matching task (raw JSON-serialisable dictionaries).
        """
        normalized_filter = normalize_filter_expr(filter)

        include_fields = list(Task.model_fields.keys())

        rows = [
            lg.entries
            for lg in self._view.get_rows(
                filter=normalized_filter,
                offset=offset,
                limit=limit,
                # Avoid an extra backend call here by deriving private fields from the
                # cached schema instead of calling get_fields() again.
                include_fields=include_fields,
            )
        ]

        # Rehydrate Enum values inside repetition patterns so callers see
        # the same structure as produced by `RepeatPattern.model_dump()`.
        # This preserves test expectations and keeps the Python-facing API
        # consistent regardless of how values were serialised at rest.
        def _rehydrate_repeat(item: dict) -> dict:
            if not isinstance(item, dict):
                return item
            out = dict(item)
            # frequency: accept values like "weekly" or names like "WEEKLY"
            freq = out.get("frequency")
            if isinstance(freq, str):
                token = freq
                if "." in token:
                    # e.g. "Frequency.WEEKLY" → "WEEKLY"
                    token = token.split(".")[-1]
                try:
                    out["frequency"] = Frequency[token]  # by name
                except Exception:
                    try:
                        out["frequency"] = Frequency(token)  # by value
                    except Exception:
                        pass

            # weekdays: list of strings like "MO" or "Weekday.MO"
            wds = out.get("weekdays")
            if isinstance(wds, list):
                new_wds = []
                for wd in wds:
                    if isinstance(wd, str):
                        tok = wd
                        if "." in tok:
                            tok = tok.split(".")[-1]
                        try:
                            new_wds.append(Weekday[tok])
                        except Exception:
                            try:
                                new_wds.append(Weekday(tok))
                            except Exception:
                                new_wds.append(wd)
                    else:
                        new_wds.append(wd)
                out["weekdays"] = new_wds
            # Ensure optional keys exist to mirror RepeatPattern.model_dump()
            # (storage normalisation may drop None values; tests expect explicit None keys)
            for _opt in ("count", "until", "time_of_day"):
                if _opt not in out:
                    out[_opt] = None
            return out

        for row in rows:
            rep = row.get("repeat")
            if isinstance(rep, list):
                row["repeat"] = [_rehydrate_repeat(x) for x in rep]

        return [Task(**row) for row in rows]

    # ────────────────────────────────────────────────────────────────────
    # Broader context helper
    # ────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────
    # Column and metrics helpers (paralleling Contact/TranscriptManager)
    # ────────────────────────────────────────────────────────────────────

    def _get_columns(self) -> Dict[str, str]:
        """
        Return {column_name: column_type} for the tasks table.
        """
        return self._view.fields

    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, str] | list[str]:
        """
        Return the list of available columns in the tasks table, optionally with types.
        """
        cols = self._get_columns()
        return cols if include_types else list(cols)

    def _num_tasks(self) -> int:
        """Return the total number of tasks in the Tasks context."""
        if self._num_tasks_cached is None:
            try:
                self._num_tasks_cached = int(
                    self._view.get_metric_count(key="task_id"),
                )
            except Exception:
                # Defensive fallback; a failed metric read should not crash tools
                self._num_tasks_cached = 0
        return int(self._num_tasks_cached)

    @read_only
    def _reduce(
        self,
        *,
        metric: str,
        keys: str | list[str],
        filter: Optional[str | dict[str, str]] = None,
        group_by: Optional[str | list[str]] = None,
    ) -> Any:
        """
        Compute reduction metrics over the Tasks table in the current context.

        Parameters
        ----------
        metric : str
            Reduction metric to compute. Supported values (case-insensitive) are
            ``\"sum\"``, ``\"mean\"``, ``\"var\"``, ``\"std\"``, ``\"min\"``,
            ``\"max\"``, ``\"median\"``, ``\"mode\"``, and ``\"count\"``.
        keys : str | list[str]
            One or more numeric task fields to aggregate (for example
            ``\"task_id\"``, ``\"queue_id\"``, or numeric custom columns). A
            single column name returns a scalar; a list of column names
            computes the metric independently per key and returns a
            ``{key -> value}`` mapping.
        filter : str | dict[str, str] | None, default None
            Optional row-level filter expression(s) in the same Python syntax as
            the ``_filter_tasks`` tool. When a string, the expression is applied
            uniformly; when a dict, each key maps to its own filter expression.
        group_by : str | list[str] | None, default None
            Optional task field(s) to group by (for example ``\"status\"`` or
            ``\"queue_id\"``). Use a single column name for one grouping level,
            or a list such as ``[\"status\", \"queue_id\"]`` to group
            hierarchically in that order. When provided, the result becomes a
            nested mapping keyed by group values, mirroring
            :func:`unify.get_logs_metric` behaviour.

        Returns
        -------
        Any
            Metric value(s) computed over the Tasks context:

            * Single key, no grouping  → scalar (float/int/str/bool).
            * Multiple keys, no grouping → ``dict[key -> scalar]``.
            * With grouping             → nested ``dict`` keyed by group values.
        """
        return reduce_logs(
            context=self._ctx,
            metric=metric,
            keys=keys,
            filter=filter,
            group_by=group_by,
        )

    # ----------------------------- Read helpers ----------------------------- #
    def _read_rows_by_ids(
        self,
        *,
        ids: List[int],
        fields: List[str],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Fetch a minimal projection for the given task ids and return a map
        of task_id -> entries dict. Always includes task_id in the projection.
        """
        try:
            logs = self._view.get_minimal_rows_by_task_ids(
                task_ids=ids,
                fields=fields,
            )
        except Exception:
            logs = []

        rows_by_id: Dict[int, Dict[str, Any]] = {
            lg.entries["task_id"]: lg.entries for lg in logs
        }
        return rows_by_id

    def _find_name_desc_collisions(
        self,
        *,
        name: str,
        description: str,
        limit: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Return existing rows that collide on name or description.
        Uses a single filtered read within the current tool call.
        """
        try:
            return self._view.get_entries(
                filter=f"name == {name!r} or description == {description!r}",
                limit=limit,
            )
        except Exception:
            return []

    # ---------------------------- Field helpers ---------------------------- #
    def _queue_member_fields(self) -> List[str]:
        """Projection for queue member rows used in ordered queue reads."""
        return [
            "task_id",
            "instance_id",
            "name",
            "description",
            "status",
            "schedule",
            "priority",
            "queue_id",
            "activated_by",
            "trigger",
            "deadline",
            "repeat",
            "response_policy",
            "entrypoint",
            "offline",
        ]

    @overload
    def _sanitize_activation(self, task: Dict[str, Any]) -> Dict[str, Any]: ...

    @overload
    def _sanitize_activation(self, task: Task) -> Task: ...

    def _sanitize_activation(
        self,
        task: Union[Dict[str, Any], Task],
    ) -> Union[Dict[str, Any], Task]:
        """
        Drop `activated_by` unless the row is currently active to keep
        payloads clean and Pydantic construction predictable.
        """
        if isinstance(task, Task):
            if task.status != Status.active:
                task.activated_by = None
            return task
        try:
            if to_status(task.get("status")) != Status.active:  # type: ignore[arg-type]
                task.pop("activated_by", None)
        except Exception:
            if str(task.get("status")) != str(Status.active):
                task.pop("activated_by", None)
        return task
