from __future__ import annotations

import os
import unify
import asyncio
import functools
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from typing import Literal
from typing import cast
from dataclasses import dataclass
from pydantic import BaseModel, Field


from ..common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    methods_to_tool_dict,
    inject_broader_context,
    TOOL_LOOP_LINEAGE,
)
from ..common.tool_outcome import ToolOutcome
from .types.status import Status
from .types.priority import Priority
from .types.schedule import Schedule
from .types.trigger import Trigger
from .types.repetition import RepeatPattern, Frequency, Weekday
from .types.task import Task
from .types.activated_by import ActivatedBy

# ------------------------------------------------------------------ #
#  Local type aliases                                                 #
# ------------------------------------------------------------------ #
# These aliases improve readability and keep signatures concise.
ScheduleLike = Optional[Union[Schedule, Dict[str, Any]]]
TriggerLike = Optional[Union[Trigger, Dict[str, Any]]]
RepeatLike = Optional[List[Union[RepeatPattern, Dict[str, Any]]]]
TaskRow = Dict[str, Any]
ToolsDict = Dict[str, Callable]

# Contact manager import (lazy at module level to avoid cycles in other modules)
from ..contact_manager.contact_manager import ContactManager
from ..common.model_to_fields import model_to_fields
from .prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
    build_execute_prompt,
)
from .base import BaseTaskScheduler
from ..actor.base import BaseActor
from ..actor.simulated import SimulatedActor
from .active_task import ActiveTask
from .active_queue import ActiveQueue
import json
from dataclasses import dataclass

from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
from ..common.semantic_search import fetch_top_k_by_references, backfill_rows
from ._queue_utils import (
    sched_prev as _q_prev,
    sched_next as _q_next,
    sync_adjacent_links as _q_sync_adjacent_links,
)
from ._queue_ops import (
    detach_from_queue_for_activation as _ops_detach_for_activation,
    attach_with_links as _ops_attach_with_links,
)
from .reintegration import ReintegrationManager
from .queue_engine import plan_reorder_queue


# Sentinel for optional-argument presence detection
_UNSET = object()

# ------------------------------------------------------------------ #
#  Optional per-tool runtime logging (TaskScheduler)                 #
# ------------------------------------------------------------------ #
from ..events.event_bus import EVENT_BUS, Event
import time


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw_l = str(raw).strip().lower()
    return raw_l in {"1", "true", "yes", "on"}


def _ts_timing_enabled() -> bool:
    # Per-manager override → global fallback
    return _env_truthy(
        "TASK_SCHEDULER_TOOL_TIMING",
        _env_truthy("TOOL_TIMING", False),
    )


def _ts_timing_print_enabled() -> bool:
    return _env_truthy(
        "TASK_SCHEDULER_TOOL_TIMING_PRINT",
        _env_truthy("TOOL_TIMING_PRINT", False),
    )


def _ts_log_tool_runtime(func):
    """Decorator to measure and optionally publish per-tool runtimes.

    Controlled by env flags:
      • TASK_SCHEDULER_TOOL_TIMING (fallback TOOL_TIMING)
      • TASK_SCHEDULER_TOOL_TIMING_PRINT (fallback TOOL_TIMING_PRINT)
    Publishes a lightweight ManagerTool event on EVENT_BUS when enabled.
    """

    @functools.wraps(func, updated=())
    def _wrapper(self: "TaskScheduler", *args, **kwargs):
        start = time.perf_counter()
        res = None
        try:
            # Any explicit returns from the finally block override the
            # return from the try block so we store it here and
            # return it in the finally block if needed
            res = func(self, *args, **kwargs)
            return res
        finally:
            try:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
            except Exception:
                elapsed_ms = -1.0

            if _ts_timing_print_enabled():
                try:
                    print(f"TaskScheduler.{func.__name__} took {elapsed_ms:.2f} ms")
                except Exception:
                    pass

            if not _ts_timing_enabled():
                return res

            # Determine category best-effort at runtime
            try:
                if (
                    isinstance(getattr(self, "_ask_tools", None), dict)
                    and func.__name__ in self._ask_tools
                ):
                    category = "ask"
                elif (
                    isinstance(getattr(self, "_update_tools", None), dict)
                    and func.__name__ in self._update_tools
                ):
                    category = "update"
                else:
                    category = "direct"
            except Exception:
                category = "direct"

            # Publish event if an event loop is running
            try:
                evt = Event(
                    type="ManagerTool",
                    payload={
                        "manager": "TaskScheduler",
                        "tool": func.__name__,
                        "category": category,
                        "duration_ms": float(elapsed_ms),
                    },
                )
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and EVENT_BUS:
                    asyncio.create_task(EVENT_BUS.publish(evt))
            except Exception:
                # Never let timing/logging affect tool behaviour
                pass

    return _wrapper


# ------------------------------------------------------------------ #
#  Typed reintegration plan                                          #
# ------------------------------------------------------------------ #
from .types.reintegration_plan import ReintegrationPlan
from .storage import TasksStore


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

    # ------------------------------------------------------------------ #
    #  Decorator – uniform ManagerMethod logging                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _log_manager_call(method_name: str, payload_key: str):
        """Decorator factory to publish incoming ManagerMethod and wrap handle.

        Ensures a single call_id is used for both the incoming event and the
        logging wrapper around the returned handle. The payload value is taken
        from the positional/keyword argument named 'text' (the first arg after
        self), matching existing method signatures.
        """

        def _decorator(func):
            @functools.wraps(func, updated=())
            async def _wrapper(self, *args, **kwargs):
                # Determine the textual payload (all three methods accept 'text')
                if "text" in kwargs:
                    payload_value = kwargs["text"]
                elif len(args) >= 1:
                    payload_value = args[0]
                else:
                    payload_value = ""

                call_id = new_call_id()
                await publish_manager_method_event(
                    call_id,
                    "TaskScheduler",
                    method_name,
                    phase="incoming",
                    **{payload_key: payload_value},
                )

                handle = await func(self, *args, **kwargs)
                handle = wrap_handle_with_logging(
                    handle,
                    call_id,
                    "TaskScheduler",
                    method_name,
                )
                return handle

            return _wrapper

        return _decorator

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

        # Instantiate a ContactManager once so its bound methods can act as tools
        self._contact_manager = ContactManager()

        # Query-only helpers – safe, read-only operations.  Include the *external* contact lookup
        self._ask_tools = {
            **methods_to_tool_dict(
                self._filter_tasks,
                self._search_tasks,
                self._get_queue,
                self._get_queue_for_task,
                include_class_name=False,  # redundant, all same class (this one)
            ),
            **methods_to_tool_dict(
                self._contact_manager.ask,
                include_class_name=True,  # Retain originating class so name is ContactManager.ask
            ),
        }

        # Write-capable helpers – every mutating operation as well as the read-only ones.
        self._update_tools = {
            **methods_to_tool_dict(
                # Ask
                self.ask,
                # Creation / deletion / cancellation
                self._create_tasks,
                self._create_task,
                self._delete_task,
                self._cancel_tasks,
                # Queue manipulation
                # Multi-queue helpers
                self._list_queues,
                self._get_queue,
                self._get_queue_for_task,
                self._reorder_queue,
                self._move_tasks_to_queue,
                self._partition_queue,
                # Reintegration
                self._reinstate_task_to_previous_queue,
                # Attribute mutations (single general-purpose updater)
                self._update_task,
                include_class_name=False,  # redundant, all same class (this one)
            ),
            **methods_to_tool_dict(
                self._contact_manager.ask,
                include_class_name=True,  # Retain originating class so name is ContactManager.ask
            ),
        }

        # active task
        if actor is None:
            # Allow tests to override default simulated duration via env var
            try:
                _dur_env = os.environ.get("UNITY_SIM_ACTOR_DURATION")
                _duration = float(_dur_env) if _dur_env is not None else 20.0
            except Exception:
                _duration = 20.0
            self._actor = SimulatedActor(duration=_duration)
        else:
            self._actor = actor

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        if not read_ctx:
            # Ensure the global assistant/context is selected before we derive our sub-context
            try:
                from .. import (
                    ensure_initialised as _ensure_initialised,
                )  # local to avoid cycles

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs["read"], ctxs["write"]
            except Exception:
                # If ensure fails (e.g. offline tests), proceed; downstream will fall back safely
                pass
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a TaskScheduler."
        self._ctx = f"{read_ctx}/Tasks" if read_ctx else "Tasks"

        # Install storage adapter and ensure context/fields exist
        self._store = TasksStore(self._ctx)
        self._store.ensure_context(
            unique_keys={"task_id": "int", "instance_id": "int"},
            auto_counting={
                "task_id": None,
                "instance_id": "task_id",
            },
            description=(
                "List of all tasks with their name, description, status, "
                "schedule, deadline, repeat pattern, priority **and** "
                "`instance_id` which tracks multiple executions of the "
                "same logical task."
            ),
            fields=model_to_fields(Task),
        )

        # In-memory checkpoints for reversible multi-queue edits within a session
        # Keyed by opaque checkpoint ids; values contain a minimal snapshot of all queues
        # (queue_id, head_id, order list, and queue-level start_at).
        self._queue_checkpoints: Dict[str, Dict[str, Any]] = {}

        # ID of the *single* task that is allowed to be in the **active**
        # state at any moment.  This will be maintained by a forthcoming
        # tool; until then it may legitimately stay as ``None``.
        # Pointer to the currently active task (if any)
        self._active_task: Optional[TaskScheduler.ActivePointer] = None
        primed_tasks = self._filter_tasks(filter="status == 'primed'")
        if primed_tasks:
            assert (
                len(primed_tasks) == 1
            ), f"More than one primed task found:\n{primed_tasks}"
            self._primed_task: Optional[Dict[str, Any]] = primed_tasks[0]
        else:
            self._primed_task: Optional[Dict[str, Any]] = None

        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # Registry of corrective plans per active task so we can restore their
        # original position on defer stop. Map: task_id -> ReintegrationPlan
        self._reintegration_plans: dict[int, "ReintegrationPlan"] = {}
        self._reintegration_manager = ReintegrationManager(self)

        # Linkage barriers: per-task events set after queue linkage updates
        # complete for a given activation. ActiveQueue can await these to avoid
        # racing reads before symmetric neighbour writes are visible.
        import asyncio as _aio  # local import to avoid top-level cost

        self._linkage_barriers: Dict[int, _aio.Event] = {}

    # Public #
    # -------#

    # English-Text Question

    @functools.wraps(BaseTaskScheduler.ask, updated=())
    @_log_manager_call.__func__("ask", "question")  # type: ignore[attr-defined]
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
        tool_policy: Union[
            Literal["default"],
            Callable[[int, Dict[str, Any]], tuple[str, Dict[str, Any]]],
            None,
        ] = "default",
    ) -> SteerableToolHandle:
        client = self._new_llm_client("gpt-5@openai")

        # Build a live tools dictionary so the prompt reflects reality
        tools = dict(self._ask_tools)

        # Add clarification tool when queues are provided
        self._maybe_add_clarification_tool(
            tools,
            clarification_up_q,
            clarification_down_q,
        )

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
            ),
        )

        # Prepare effective tool_policy – central helper determines requirement
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
            parent_chat_context=parent_chat_context,
            log_steps=_log_tool_steps,
            tool_policy=effective_tool_policy,
        )
        # Logging wrapper applied by decorator

        # Optional reasoning exposure
        if _return_reasoning_steps:
            handle = self._wrap_result_with_messages(handle, client)

        return handle

    # English-Text Update Request

    @functools.wraps(BaseTaskScheduler.update, updated=())
    @_log_manager_call.__func__("update", "request")  # type: ignore[attr-defined]
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
        tool_policy: Union[
            Literal["default"],
            Callable[[int, Dict[str, Any]], tuple[str, Dict[str, Any]]],
            None,
        ] = "default",
    ) -> SteerableToolHandle:
        client = self._new_llm_client("gpt-5@openai")

        # Build a live tools dictionary first (prompt needs it)
        tools = dict(self._update_tools)

        # Add queue planning helpers for atomic modifications, mirroring execute
        # and include checkpoint helpers for reversibility.
        def _later_group_schema():
            class _LaterGroup(BaseModel):
                task_ids: List[int] = Field(min_length=1)
                queue_start_at: Optional[str] = None

            return _LaterGroup

        _LaterGroup = _later_group_schema()

        class _QueuePlan(BaseModel):
            now: List[int] = Field(min_length=1)
            later_groups: List[_LaterGroup] = Field(default_factory=list)  # type: ignore[name-defined]
            notes: Optional[str] = None

        def validate_queue_plan(*, plan: Dict[str, Any] | str) -> Dict[str, Any]:  # type: ignore[valid-type]
            import json as _json

            try:
                parsed = _json.loads(plan) if isinstance(plan, str) else plan
            except Exception as _e:  # noqa: N806
                raise ValueError(f"Invalid plan: {_e}")
            model = _QueuePlan.model_validate(parsed)
            preview: Dict[str, Any] = {"now": model.now, "later": []}
            for g in model.later_groups:
                preview["later"].append(
                    {"task_ids": list(g.task_ids), "queue_start_at": g.queue_start_at},
                )
            return {
                "outcome": "validated",
                "details": {"plan": model.model_dump(), "preview": preview},
            }

        def apply_queue_plan(*, plan: Dict[str, Any] | str) -> Dict[str, Any]:  # type: ignore[valid-type]
            import json as _json

            try:
                parsed = _json.loads(plan) if isinstance(plan, str) else plan
            except Exception as _e:
                raise ValueError(f"Invalid plan: {_e}")
            model = _QueuePlan.model_validate(parsed)
            if model.later_groups:
                parts = [{"task_ids": list(model.now)}] + [
                    {"task_ids": list(g.task_ids), "queue_start_at": g.queue_start_at}
                    for g in model.later_groups
                ]
                self._partition_queue(parts=parts, strategy="preserve_order")
            else:
                if model.now:
                    self._reorder_queue(queue_id=None, new_order=list(model.now))
            cp = checkpoint_queue_state(label="post-apply-plan")
            return {
                "outcome": "applied",
                "details": {"checkpoint_id": cp["details"]["checkpoint_id"]},
            }

        def checkpoint_queue_state(*, label: Optional[str] = None) -> Dict[str, Any]:  # type: ignore[valid-type]
            snapshot: Dict[str, Any] = {"label": label, "queues": []}
            try:
                all_q = self._list_queues()
            except Exception:
                all_q = []
            for q in all_q:
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

            from ..common.llm_helpers import short_id as _short_id  # local import

            cid = _short_id(8)
            self._queue_checkpoints[cid] = snapshot
            try:
                if str(os.getenv("UNITY_TS_PERSIST_CHECKPOINTS", "")).lower() in {
                    "1",
                    "true",
                    "yes",
                }:
                    self._persist_checkpoint(cid, label, snapshot)
            except Exception:
                pass
            return {"outcome": "checkpointed", "details": {"checkpoint_id": cid}}

        def revert_to_checkpoint(*, checkpoint_id: str) -> Dict[str, Any]:  # type: ignore[valid-type]
            snap = self._queue_checkpoints.get(str(checkpoint_id))
            if snap is None:
                try:
                    if str(os.getenv("UNITY_TS_PERSIST_CHECKPOINTS", "")).lower() in {
                        "1",
                        "true",
                        "yes",
                    }:
                        snap = self._load_checkpoint(str(checkpoint_id))
                except Exception:
                    snap = None
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
                            head_row = self._get_single_row_or_raise(head_tid)
                            sched = {**(head_row.get("schedule") or {})}
                            sched["start_at"] = start_at
                            self._validated_write(
                                task_id=head_tid,
                                entries={"schedule": sched, "status": Status.scheduled},
                                err_prefix="While restoring queue start_at from checkpoint:",
                            )
                        except Exception:
                            pass
            return {"outcome": "reverted", "details": {"checkpoint_id": checkpoint_id}}

        def get_latest_checkpoint() -> Dict[str, Any]:  # type: ignore[valid-type]
            try:
                keys = list(self._queue_checkpoints.keys())
                if keys:
                    cid = keys[-1]
                    snap = self._queue_checkpoints.get(cid, {})
                    return {
                        "outcome": "ok",
                        "details": {"checkpoint_id": cid, "label": snap.get("label")},
                    }
                if str(os.getenv("UNITY_TS_PERSIST_CHECKPOINTS", "")).lower() in {
                    "1",
                    "true",
                    "yes",
                }:
                    latest = self._get_latest_persisted_checkpoint()
                    if latest is not None:
                        return {"outcome": "ok", "details": latest}
                return {
                    "outcome": "none",
                    "details": {"checkpoint_id": None, "label": None},
                }
            except Exception:
                return {
                    "outcome": "none",
                    "details": {"checkpoint_id": None, "label": None},
                }

        # Bind to shared scheduler helpers to avoid duplication
        validate_queue_plan = self.validate_queue_plan
        apply_queue_plan = self.apply_queue_plan
        checkpoint_queue_state = self.checkpoint_queue_state
        revert_to_checkpoint = self.revert_to_checkpoint
        get_latest_checkpoint = self.get_latest_checkpoint

        # Merge these helpers into the update toolset
        tools.update(
            methods_to_tool_dict(
                validate_queue_plan,
                apply_queue_plan,
                checkpoint_queue_state,
                revert_to_checkpoint,
                get_latest_checkpoint,
                self._set_queue,
                self._set_schedules_atomic,
                include_class_name=False,
            ),
        )

        # Add clarification tool when queues are provided
        self._maybe_add_clarification_tool(
            tools,
            clarification_up_q,
            clarification_down_q,
        )

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
            ),
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
            parent_chat_context=parent_chat_context,
            log_steps=_log_tool_steps,
            tool_policy=effective_tool_policy,
        )
        # Logging wrapper applied by decorator

        # Optional reasoning exposure
        if _return_reasoning_steps:
            handle = self._wrap_result_with_messages(handle, client)

        return handle

    # Execute

    @functools.wraps(BaseTaskScheduler.execute, updated=())
    @_log_manager_call.__func__("execute", "request")
    async def execute(
        self,
        text: str,
        *,
        isolated: Optional[bool] = None,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        freeform_text: str = text

        # Refuse execution when a task is already active.
        if self._active_task is not None:
            raise RuntimeError("Another task is already running – stop it first.")

        # Also guard against orphan 'active' rows (e.g., after crash) even if pointer is None.
        try:
            any_active = any(
                r.get("status") == str(Status.active)
                for r in self._filter_tasks(filter="status == 'active'", limit=1)
            )
        except Exception:
            any_active = False
        if any_active:
            raise RuntimeError(
                "A task is marked as active, but no active handle is present – reconcile state before starting another task.",
            )

        # Fast path: numeric task_id provided → start at that id using queue semantics
        stripped = freeform_text.strip()
        if stripped.isdigit():
            try:
                # Honor explicit override when provided; default remains chained
                if isolated is True:
                    return await self._execute_internal(
                        task_id=int(stripped),
                        parent_chat_context=parent_chat_context,
                        clarification_up_q=clarification_up_q,
                        clarification_down_q=clarification_down_q,
                        activated_by=ActivatedBy.explicit,
                        detach=True,
                    )
                else:
                    return await self._execute_queue_internal(
                        task_id=int(stripped),
                        parent_chat_context=parent_chat_context,
                        clarification_up_q=clarification_up_q,
                        clarification_down_q=clarification_down_q,
                    )
            except (ValueError, RuntimeError):
                # Fall back to the outer loop (will ask/clarify/create)
                pass

        # Start LLM-driven outer loop which will resolve the task id and adopt the queue handle.
        # When an explicit isolation preference is provided, append a short guiding sentence
        # so the outer loop can route to the appropriate execution tool.
        if isolated is True:
            try:
                freeform_text = f"{freeform_text}\n\nExecution preference: run this task in isolation (detach it from any queue)."
            except Exception:
                pass
        elif isolated is False:
            try:
                freeform_text = f"{freeform_text}\n\nExecution preference: if this task is part of a queue, preserve the task queue and do not detach the task."
            except Exception:
                pass

        return self._start_execute_loop(
            freeform_text=freeform_text,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

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

        candidate_rows = self._filter_tasks(
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

        # Adjust queue linkages for activation (and record reintegration plan).
        # detach=True → isolation semantics; detach=False → chain semantics.
        desired_next: Optional[int] = _q_next(task_row.get("schedule"))
        self._detach_from_queue_for_activation(
            task_id=task_id,
            detach=detach,
        )
        # Yield the event loop until the current row reflects the expected
        # sub-head linkage when chaining (prev=None, next=desired_next).
        # Avoids relying on arbitrary sleeps while remaining low-latency.
        if not detach and desired_next is not None:
            for _ in range(5):  # a handful of yields should suffice
                try:
                    rows_after = self._filter_tasks(
                        filter=f"task_id == {task_id}",
                        limit=1,
                    )
                    if rows_after:
                        sched_after = rows_after[0].get("schedule") or {}
                        if (sched_after.get("prev_task") is None) and (
                            sched_after.get("next_task") == desired_next
                        ):
                            break
                except Exception:
                    pass
                await asyncio.sleep(0)

        # Build the active plan via the actor and wrap it so the task table stays in sync
        _task_desc = task_row.get("description") or task_row.get("name") or ""
        handle = await ActiveTask.create(
            self._actor,
            task_description=_task_desc,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
            task_id=task_id,
            instance_id=task_row["instance_id"],
            scheduler=self,
        )

        self._active_task = TaskScheduler.ActivePointer(
            task_id=task_id,
            instance_id=task_row["instance_id"],
            handle=handle,
        )

        # Clone if this is a triggerable or recurring task
        if self._to_status(task_row["status"]) == Status.triggerable or task_row.get(
            "repeat",
        ):
            self._clone_task_instance(task_row)

        # Promote status to active (and record the activation reason) and clear the primed pointer if needed

        # Infer activation reason based on provided cause or task configuration
        reason: ActivatedBy
        if activated_by is not None:
            reason = activated_by
        else:
            sched = task_row.get("schedule") or {}
            if task_row.get("trigger") is not None:
                reason = ActivatedBy.trigger
            elif (sched.get("prev_task") is None) and (
                sched.get("start_at") is not None
            ):
                reason = ActivatedBy.schedule
            elif sched.get("prev_task") is not None:
                reason = ActivatedBy.queue
            else:
                reason = ActivatedBy.explicit

        self._update_task_status_instance(
            task_id=task_id,
            instance_id=task_row["instance_id"],
            new_status="active",
            activated_by=reason,
        )
        if self._primed_task and self._primed_task["task_id"] == task_id:
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
    ) -> SteerableToolHandle:
        """Start queue execution at `task_id` and return a composite queue handle."""
        first = await self._execute_internal(
            task_id=task_id,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
            activated_by=ActivatedBy.explicit,
            # Always use queue semantics – followers remain attached
            detach=False,
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
    #  Helper – build and start the execute outer tool-use loop      #
    # ------------------------------------------------------------------ #
    def _start_execute_loop(
        self,
        *,
        freeform_text: str,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """Compose tools and prompt, then start the execute reasoning loop."""
        client = self._new_llm_client("gpt-5@openai")

        # ── tool definitions ────────────────────────────────────────────────
        class _LaterGroup(BaseModel):
            task_ids: List[int] = Field(min_length=1)
            queue_start_at: Optional[str] = None

        class _QueuePlan(BaseModel):
            now: List[int] = Field(min_length=1)
            later_groups: List[_LaterGroup] = Field(default_factory=list)
            notes: Optional[str] = None

        def validate_queue_plan(*, plan: Dict[str, Any] | str) -> Dict[str, Any]:  # type: ignore[valid-type]
            """Validate a proposed queue plan (dict or JSON string) and return the normalised structure.

            The plan must include a non-empty `now` list and zero or more
            `later_groups`, each with a non-empty `task_ids` list. The function
            returns the validated plan and previews of the resulting queues.
            """
            import json as _json

            try:
                parsed = _json.loads(plan) if isinstance(plan, str) else plan
            except Exception as _e:  # noqa: N806 (keep local name)
                raise ValueError(f"Invalid plan: {_e}")
            model = _QueuePlan.model_validate(parsed)
            # Preview shapes – we do not mutate here
            preview: Dict[str, Any] = {"now": model.now, "later": []}
            for g in model.later_groups:
                preview["later"].append(
                    {"task_ids": list(g.task_ids), "queue_start_at": g.queue_start_at},
                )
            return {
                "outcome": "validated",
                "details": {"plan": model.model_dump(), "preview": preview},
            }

        def apply_queue_plan(*, plan: Dict[str, Any] | str) -> Dict[str, Any]:  # type: ignore[valid-type]
            """Atomically apply a validated plan (dict or JSON string) using invariant-preserving tools.

            Strategy:
            - Use `_partition_queue` when there are any later groups to split the
              source queue into [now] and later queues with optional dates.
            - Otherwise, reorder the chosen queue so that `now` is the head
              (dropping other runnable tasks from that queue).
            After success, automatically create a checkpoint to allow revert.
            """
            import json as _json

            try:
                parsed = _json.loads(plan) if isinstance(plan, str) else plan
            except Exception as _e:
                raise ValueError(f"Invalid plan: {_e}")
            model = _QueuePlan.model_validate(parsed)
            # If later groups exist, build parts payload
            if model.later_groups:
                parts = [{"task_ids": list(model.now)}] + [
                    {"task_ids": list(g.task_ids), "queue_start_at": g.queue_start_at}
                    for g in model.later_groups
                ]
                self._partition_queue(parts=parts, strategy="preserve_order")
            else:
                # Only reorder the selected queue to contain exactly `now`
                if model.now:
                    self._reorder_queue(queue_id=None, new_order=list(model.now))
            # Auto-checkpoint
            cp = checkpoint_queue_state(label="post-apply-plan")
            return {
                "outcome": "applied",
                "details": {"checkpoint_id": cp["details"]["checkpoint_id"]},
            }

        def checkpoint_queue_state(*, label: Optional[str] = None) -> Dict[str, Any]:  # type: ignore[valid-type]
            """Create a session-scoped checkpoint of ALL runnable queues.

            The checkpoint captures, for every queue, the head id, the full order
            (head→tail), and the queue-level start_at timestamp. This enables the
            execute loop to revert multi-queue edits if the user changes their mind
            before completion.

            Parameters
            ----------
            label : str | None, optional
                Optional human-readable label for diagnostics.

            Returns
            -------
            dict
                {"outcome": "checkpointed", "details": {"checkpoint_id": str}}
            """
            # Build snapshot
            snapshot: Dict[str, Any] = {"label": label, "queues": []}
            try:
                all_q = self._list_queues()
            except Exception:
                all_q = []
            for q in all_q:
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

            from ..common.llm_helpers import short_id as _short_id  # local import

            cid = _short_id(8)
            self._queue_checkpoints[cid] = snapshot
            # Optionally persist to a dedicated context for cross-session safety
            try:
                if str(os.getenv("UNITY_TS_PERSIST_CHECKPOINTS", "")).lower() in {
                    "1",
                    "true",
                    "yes",
                }:
                    self._persist_checkpoint(cid, label, snapshot)
            except Exception:
                pass
            return {"outcome": "checkpointed", "details": {"checkpoint_id": cid}}

        # Create an initial checkpoint at the start of execute to guarantee a known revert point
        try:
            checkpoint_queue_state(label="pre-execute")
        except Exception:
            pass

        def revert_to_checkpoint(*, checkpoint_id: str) -> Dict[str, Any]:  # type: ignore[valid-type]
            """Revert all queues to a previously created checkpoint.

            This operation is deterministic and invariant-preserving. It rewrites
            each queue to the exact order stored in the checkpoint and reapplies
            the queue-level start_at to each restored head.
            """
            snap = self._queue_checkpoints.get(str(checkpoint_id))
            if snap is None:
                # Optionally load from persistence when enabled
                try:
                    if str(os.getenv("UNITY_TS_PERSIST_CHECKPOINTS", "")).lower() in {
                        "1",
                        "true",
                        "yes",
                    }:
                        snap = self._load_checkpoint(str(checkpoint_id))
                except Exception:
                    snap = None
            assert snap is not None, f"Unknown checkpoint_id={checkpoint_id}"
            for q in snap.get("queues", []):
                qid = q.get("queue_id")
                order = list(q.get("order", []) or [])
                if order:
                    self._reorder_queue(queue_id=qid, new_order=order)
                    # Restore queue-level start_at on head if present
                    start_at = q.get("start_at")
                    if start_at is not None:
                        head_tid = int(order[0])
                        try:
                            head_row = self._get_single_row_or_raise(head_tid)
                            sched = {**(head_row.get("schedule") or {})}
                            sched["start_at"] = start_at
                            self._validated_write(
                                task_id=head_tid,
                                entries={"schedule": sched, "status": Status.scheduled},
                                err_prefix="While restoring queue start_at from checkpoint:",
                            )
                        except Exception:
                            # Defensive: if head row cannot be found now, skip silently
                            pass
            return {"outcome": "reverted", "details": {"checkpoint_id": checkpoint_id}}

        def get_latest_checkpoint() -> Dict[str, Any]:  # type: ignore[valid-type]
            """Return the most recently created checkpoint id and label.

            When no checkpoints exist yet, returns {"checkpoint_id": None}.
            """
            try:
                keys = list(self._queue_checkpoints.keys())
                if keys:
                    cid = keys[-1]
                    snap = self._queue_checkpoints.get(cid, {})
                    return {
                        "outcome": "ok",
                        "details": {"checkpoint_id": cid, "label": snap.get("label")},
                    }
                # Optionally consult persisted store
                if str(os.getenv("UNITY_TS_PERSIST_CHECKPOINTS", "")).lower() in {
                    "1",
                    "true",
                    "yes",
                }:
                    latest = self._get_latest_persisted_checkpoint()
                    if latest is not None:
                        return {"outcome": "ok", "details": latest}
                return {
                    "outcome": "none",
                    "details": {"checkpoint_id": None, "label": None},
                }
            except Exception:
                return {
                    "outcome": "none",
                    "details": {"checkpoint_id": None, "label": None},
                }

        def create_task(*, name: str, description: str) -> ToolOutcome:  # type: ignore[valid-type]
            """Create a brand-new task with minimal inputs (name, description).

            Notes
            -----
            - This scoped creator intentionally exposes only name/description to
              prevent schedule/status/queue manipulation from the execute loop.
            - Lifecycle values and invariants are inferred by the scheduler.
            """
            return self._create_task(name=name, description=description)

        async def _execute_by_id(
            *,
            task_id: int,
        ) -> SteerableToolHandle:  # type: ignore[valid-type]
            """Start the task with *task_id* and adopt the queue handle.

            Behavioural rules
            -----------------
            - Never modify scheduling/ordering or `start_at` to begin execution.
            - If the user wants the whole sequence now, reorder the queue explicitly first
              (see `_update_task_queue`) so the desired subset is at the head, then call this.

            Post-conditions (for the outer loop / LLM):
            - Mode: "queue" (followers remain attached; chaining semantics).
            - The selected task stays a member of its current queue.
            - You SHOULD refresh queues after this call using `list_queues()` and `get_queue(queue_id=…)`
              before attempting any further queue edits.
            """

            handle = await self._execute_queue_internal(
                task_id=task_id,
                parent_chat_context=parent_chat_context,
                clarification_up_q=clarification_up_q,
                clarification_down_q=clarification_down_q,
            )
            # 💡 signal pass-through so the outer loop adopts this handle
            setattr(handle, "__passthrough__", True)
            return handle

        async def _execute_isolated_by_id(
            *,
            task_id: int,
        ) -> SteerableToolHandle:  # type: ignore[valid-type]
            """Start ONLY the specified task by first detaching it from the queue.

            Behavioural rules
            -----------------
            - Use this when the user explicitly requests to run a task "in isolation"
              or to "detach it entirely from the queue".
            - Detachment semantics:
              • If the task is the current head, its successor (if any) becomes the new head
                and inherits the queue-level start_at timestamp.
              • The detached task loses its schedule entirely (no prev/next/start_at).
              • Followers of the detached task remain linked to each other.
            - This does NOT move the task into a new queue, and does NOT partition queues.

            Post-conditions (for the outer loop / LLM):
            - Mode: "isolated" (detached execution).
            - The started task is NO LONGER a member of its former queue.
            - You MUST re-query queues using `list_queues()` / `get_queue(queue_id=…)` before
              any subsequent queue edits. Do NOT include the detached task id in `reorder_queue`
              `new_order` arrays for its former queue.
            """

            # Run internal execute with detach=True to enforce isolation semantics
            handle = await self._execute_internal(
                task_id=task_id,
                parent_chat_context=parent_chat_context,
                clarification_up_q=clarification_up_q,
                clarification_down_q=clarification_down_q,
                activated_by=ActivatedBy.explicit,
                detach=True,
            )
            # Signal pass-through so the outer loop adopts this handle
            setattr(handle, "__passthrough__", True)
            return handle

        async def request_clarification(question: str) -> str:  # type: ignore[valid-type]
            """Bubble *question* up to the caller and await the answer."""
            rc = self._make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
            )
            return await rc(question)

        tools = methods_to_tool_dict(
            # Read-only helpers
            self.ask,
            # Multi-queue helpers (coarse, invariant-preserving)
            self._list_queues,
            self._get_queue,
            self._reorder_queue,
            self._move_tasks_to_queue,
            self._partition_queue,
            # Atomic queue materialization
            self._set_queue,
            self._set_schedules_atomic,
            # Plan helpers
            validate_queue_plan,
            apply_queue_plan,
            # Reintegration and safety
            self._reinstate_task_to_previous_queue,
            checkpoint_queue_state,
            revert_to_checkpoint,
            get_latest_checkpoint,
            # Start execution
            _execute_by_id,
            _execute_isolated_by_id,
            # Creation (name + description only)
            create_task,
            include_class_name=False,
        )
        # Only expose clarification tool when both queues are available
        self._maybe_add_clarification_tool(
            tools,
            clarification_up_q,
            clarification_down_q,
        )

        # ── dynamic system prompt ───────────────────────────────────────────
        client.set_system_message(
            build_execute_prompt(tools),
        )

        outer_handle = self._start_loop(
            client,
            freeform_text,
            tools,
            loop_id=f"{self.__class__.__name__}.execute",
            parent_chat_context=parent_chat_context,
            log_steps=True,
        )

        return outer_handle

    # ------------------------------------------------------------------ #
    #  Scope classification helper (LLM-routed)                           #
    # ------------------------------------------------------------------ #

    #  Per-instance helpers

    def _update_task_status_instance(
        self,
        *,
        task_id: int,
        instance_id: int,
        new_status: str,
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
        log_objs = self._store.get_rows(
            filter=f"task_id == {task_id} and instance_id == {instance_id}",
            return_ids_only=False,
        )
        if not log_objs:
            raise ValueError(
                f"No task instance ({task_id}.{instance_id}) found.",
            )
        assert len(log_objs) == 1, "Composite primary key must be unique."
        # Normalise status to enum for consistent comparisons
        new_status_enum = self._to_status(new_status)
        entries: Dict[str, Any] = {"status": new_status_enum}
        # Only allow `activated_by` to be set during transition to 'active'.
        # For transitions away from 'active', preserve the existing value for auditability.
        if new_status_enum == Status.active and activated_by is not None:
            # Set only at the moment of activation; never overwrite later
            entries["activated_by"] = str(activated_by)

        result = self._write_log_entries(
            logs=log_objs[0].id if hasattr(log_objs[0], "id") else log_objs[0],
            entries=entries,
            overwrite=True,
        )
        # Auto-clear reintegration plan on completion/failed to avoid stale replay.
        # Intentionally keep the plan on 'cancelled' so callers can reinstate
        # a cancelled isolated activation back to its prior queue position.
        try:
            key = (task_id, instance_id)
            plan = self._reintegration_plans.get(key)
            if (
                plan is not None
                and plan.task_id == task_id
                and plan.instance_id == instance_id
                and new_status_enum in {Status.completed, Status.failed}
            ):
                self._reintegration_plans.pop(key, None)
        except Exception:
            pass

        return result

    def _clone_task_instance(self, task_row: Dict[str, Any]) -> None:
        """
        Create a fresh row for the next instance of a triggerable or recurring task.

        Parameters
        ----------
        task_row : dict
            Existing task row used as the template. Copies user‑facing fields,
            keeps the same ``task_id``, omits ``instance_id`` so the backend auto‑increments it,
            and preserves the existing status (``triggerable`` or ``scheduled``).
        """
        allowed = set(Task.model_json_schema()["properties"].keys())
        clone_payload = {
            k: v for k, v in task_row.items() if k in allowed and k != "instance_id"
        }
        # Do not carry over activation metadata to a fresh instance
        clone_payload.pop("activated_by", None)
        # Drop any internal bookkeeping injected by Unify (_id, _log_id …)
        self._store.log(entries=clone_payload, new=True)

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
        status = self._to_status(status)
        prev_task_id = self._sched_prev(schedule)
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

    def _get_logs_by_task_ids(
        self,
        *,
        task_ids: Union[int, List[int]],
        return_ids_only: bool = True,
    ) -> List[Union[int, unify.Log]]:
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
        list[int | unify.Log]
            The matching log identifiers or objects.
        """
        return self._store.get_logs_by_task_ids(
            task_ids=task_ids,
            return_ids_only=return_ids_only,
        )

    # Private Tools #
    # --------------#

    # Create

    @_ts_log_tool_runtime
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
        _name_lit = f"{name!r}"
        _desc_lit = f"{description!r}"
        _dupe_rows = self._store.get_entries(
            filter=f"name == {_name_lit} or description == {_desc_lit}",
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
            status = self._to_status(status)

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
            elif self._to_status(status) != Status.triggerable:
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
                    try:
                        primed_exists = (
                            self._primed_task is not None
                            and self._to_status(self._primed_task.get("status"))
                            == Status.primed
                        )
                    except Exception:
                        primed_exists = False

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
        try:
            if queue_id is not None:
                derived_qid = int(queue_id)
            elif schedule is not None:
                prev_tid = self._sched_prev(schedule)
                if prev_tid is not None:
                    try:
                        prev_row = self._get_single_row_or_raise(int(prev_tid))
                        derived_qid = prev_row.get("queue_id")
                    except Exception:
                        derived_qid = None
                if derived_qid is None:
                    # Head or standalone scheduled/queued task → allocate new queue id
                    derived_qid = self._allocate_new_queue_id()
        except Exception:
            derived_qid = None

        task_details = Task(
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
        ).to_post_json()

        # ------------------  write log immediately  ------------------ #
        log = self._store.log(entries=task_details, new=True)
        task_id = log.entries["task_id"]
        task_details["task_id"] = task_id

        # Keep linkage symmetric right after creation
        self._sync_adjacent_links(task_id=task_id, schedule=schedule)

        # ── Ensure the in-memory cache reflects any linkage tweaks ──
        if status == Status.primed:
            # Avoid a backend read: populate primed pointer directly from the created log
            try:
                primed_row = dict(log.entries)
                # Ensure required keys are present on the cached row
                primed_row["task_id"] = task_id
                if "instance_id" not in primed_row:
                    primed_row["instance_id"] = getattr(log, "entries", {}).get(
                        "instance_id",
                        task_details.get("instance_id"),
                    )
                self._primed_task = primed_row
            except Exception:
                # Fallback to lazy refresh if direct population fails
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
            # Creation should not auto-append to any queue.

        return {
            "outcome": "task created successfully",
            "details": {"task_id": task_id},
        }

    @_ts_log_tool_runtime
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
        This avoids multiple calls to the singular ``_create_task`` followed by separate
        queue manipulation calls. In one call you get:
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

        # Batched fast path: simple specs (no schedule/trigger/queue edits) and no explicit ordering
        simple_allowed = queue_ordering is None and all(
            not any(
                k in spec
                for k in (
                    "schedule",
                    "trigger",
                    "deadline",
                    "repeat",
                    "priority",
                    "response_policy",
                    "status",
                    "queue_id",
                )
            )
            for spec in tasks
        )

        created_ids: List[int] = []
        if simple_allowed:
            # Single deduplication read across all names/descriptions
            try:
                # Build a single OR-chain filter across provided names/descriptions
                clauses: list[str] = []
                for spec in tasks:
                    nm = spec.get("name")
                    ds = spec.get("description")
                    if nm:
                        clauses.append(f"name == {nm!r}")
                    if ds:
                        clauses.append(f"description == {ds!r}")
                filter_expr = " or ".join(clauses) if clauses else None
                existing = (
                    self._store.get_entries(filter=filter_expr, limit=len(tasks) + 1)
                    if filter_expr
                    else []
                )
            except Exception:
                existing = []
            # Check collisions precisely
            if existing:
                existing_names = {r.get("name") for r in existing}
                existing_descs = {r.get("description") for r in existing}
                for idx, spec in enumerate(tasks):
                    if spec.get("name") in existing_names:
                        raise ValueError(
                            f"A task with {'name'!r} = {spec.get('name')!r} already exists",
                        )
                    if spec.get("description") in existing_descs:
                        raise ValueError(
                            f"A task with {'description'!r} = {spec.get('description')!r} already exists",
                        )

            # Decide statuses once (no extra reads within this tool call)
            assign_primed = self._active_task is None and (
                self._primed_task is None
                or self._to_status(self._primed_task.get("status")) != Status.primed
            )  # type: ignore[arg-type]

            entries_list: List[Dict[str, Any]] = []
            primed_index: Optional[int] = (
                0 if assign_primed and len(tasks) > 0 else None
            )
            for idx, spec in enumerate(tasks):
                desired_status = (
                    Status.primed
                    if primed_index is not None and idx == primed_index
                    else Status.queued
                )
                task_payload = Task(
                    name=str(spec.get("name")),
                    description=str(spec.get("description")),
                    status=desired_status,
                    schedule=None,
                    trigger=None,
                    deadline=None,
                    repeat=None,
                    priority=Priority.normal,
                    response_policy=None,
                    queue_id=None,
                ).to_post_json()
                entries_list.append(task_payload)

            # Create all in one backend call, then fetch created rows once
            resp = self._store.create_many(entries_list=entries_list)
            created_log_ids: List[int] = []
            try:
                # Primary: explicit log_event_ids from backend
                created_log_ids = list(resp.get("log_event_ids") or [])
            except Exception:
                created_log_ids = []
            if not created_log_ids:
                # Some client variants may return a different key
                for alt in ("ids", "log_ids"):
                    try:
                        created_log_ids = list(resp.get(alt) or [])
                        if created_log_ids:
                            break
                    except Exception:
                        created_log_ids = []

            rows = []
            try:
                if created_log_ids:
                    rows = self._store.get_rows_by_log_ids(log_ids=created_log_ids)
            except Exception:
                rows = []
            # Fallback: fetch by pairwise (name AND description) to avoid collisions
            if not rows:
                try:
                    pair_clauses: list[str] = []
                    for spec in tasks:
                        nm = spec.get("name")
                        ds = spec.get("description")
                        if nm is None or ds is None:
                            continue
                        pair_clauses.append(
                            f"(name == {nm!r} and description == {ds!r})",
                        )
                    filter_expr = " or ".join(pair_clauses) if pair_clauses else None
                    rows = (
                        self._store.get_rows(
                            filter=filter_expr,
                            limit=max(10, len(tasks) * 2),
                            return_ids_only=False,
                        )
                        if filter_expr
                        else []
                    )
                except Exception:
                    rows = []
            # Derive task_ids from returned rows in the same order as input specs
            by_key: Dict[tuple[str, str], int] = {}
            for lg in rows:
                try:
                    e = getattr(lg, "entries", {}) or {}
                    nm = str(e.get("name"))
                    ds = str(e.get("description"))
                    by_key[(nm, ds)] = int(e.get("task_id"))
                except Exception:
                    continue
            created_ids = []
            for spec in tasks:
                tid = by_key.get((str(spec.get("name")), str(spec.get("description"))))
                if tid is not None:
                    created_ids.append(int(tid))

            # Robust fallback: fetch by names only if mapping failed / partial
            if len(created_ids) < len(tasks):
                try:
                    name_list = [str(spec.get("name")) for spec in tasks]
                    list_literal = "[" + ", ".join(repr(n) for n in name_list) + "]"
                    expr = f"name in {list_literal}"
                    ents = self._store.get_entries(filter=expr, limit=len(tasks))
                except Exception:
                    ents = []
                if ents:
                    by_name: Dict[str, int] = {}
                    for r in ents:
                        try:
                            by_name[str(r.get("name"))] = int(r.get("task_id"))
                        except Exception:
                            continue
                    created_ids = []
                    for spec in tasks:
                        nm = str(spec.get("name"))
                        if nm in by_name:
                            created_ids.append(int(by_name[nm]))

            # Reflect primed pointer in memory if we created one
            if primed_index is not None:
                try:
                    # Prefer 'rows' if available; else consult 'ents' fallback
                    primed_found: Optional[Dict[str, Any]] = None
                    if rows:
                        for r in rows:
                            e = getattr(r, "entries", {}) or {}
                            if e.get("status") == str(Status.primed):
                                primed_found = e
                                break
                    if primed_found is None:
                        try:
                            ents  # type: ignore[name-defined]
                        except Exception:
                            ents = []  # type: ignore[assignment]
                        for e in ents or []:
                            if e.get("status") == str(Status.primed):
                                primed_found = e
                                break
                    if primed_found is not None:
                        self._primed_task = dict(primed_found)
                except Exception:
                    pass
        else:
            # Create tasks sequentially to preserve ascending id assignment
            created_ids = []
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
                ):
                    if key in spec:
                        payload[key] = spec[key]
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
                    auto_primed_rows = self._filter_tasks(
                        filter=f"task_id in {created_ids} and status == 'primed'",
                    )
                except Exception:
                    auto_primed_rows = []
                for r in auto_primed_rows or []:
                    try:
                        self._update_task_status(
                            task_ids=r.get("task_id"),
                            new_status="queued",
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
                    self._update_task_status(task_ids=order_ids[0], new_status="primed")
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

    @_ts_log_tool_runtime
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
        self._store.delete(logs=log_id)
        return {
            "outcome": "task deleted",
            "details": {"task_id": task_id},
        }

    # Cancel Task(s)

    @_ts_log_tool_runtime
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
        completed_tasks = self._filter_tasks(filter="status == 'completed'")
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
        """Thin wrapper: delegate to queue-utils (prev pointer)."""
        return _q_prev(sched)

    @staticmethod
    def _sched_next(sched):
        """Thin wrapper: delegate to queue-utils (next pointer)."""
        return _q_next(sched)

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
    ) -> None:
        """Delegate to queue-utils to maintain symmetric neighbour links."""
        _q_sync_adjacent_links(self, task_id=task_id, schedule=schedule)

    # ────────────────────────────────────────────────────────────────────
    # Multi-queue helpers (public tools for the update loop)
    # ────────────────────────────────────────────────────────────────────

    def _allocate_new_queue_id(self) -> int:
        """Return a fresh integer queue identifier.

        Strategy – scan all tasks for existing top‑level ``queue_id`` numeric
        values and return ``max + 1`` (starting at 1). Queues are implicit in
        this scheduler: a queue exists as soon as at least one task carries a
        numeric ``queue_id``.
        """
        rows = self._filter_tasks()
        max_id = 0
        for r in rows:
            qid = r.get("queue_id")
            try:
                if isinstance(qid, int):
                    max_id = max(max_id, qid)
            except Exception:
                pass
        new_id = max_id + 1 if max_id >= 0 else 1
        return new_id

    def _head_row_for_queue(self, queue_id: Optional[int]) -> Optional[TaskRow]:
        """Best-effort fetch of the head row for a given queue.

        The "default" queue is represented by ``queue_id is None``.
        Only non-terminal tasks are considered part of a runnable queue.
        Returns ``None`` when no head exists (e.g., empty queue).
        """
        rows = [
            r
            for r in self._filter_tasks()
            if r.get("schedule") is not None
            and self._to_status(r.get("status")) not in self._TERMINAL_STATUSES
        ]
        heads: list[TaskRow] = []
        for r in rows:
            sched = r.get("schedule") or {}
            qid = r.get("queue_id")
            prev_id = sched.get("prev_task")
            if qid == queue_id and prev_id is None:
                heads.append(r)
        if not heads:
            return None
        assert (
            len(heads) == 1
        ), f"Multiple heads detected for queue_id={queue_id}: {heads}"
        return heads[0]

    def _walk_queue(self, head_row: TaskRow) -> list[TaskRow]:
        """Walk from ``head_row`` forward using ``next_task`` and return rows.

        Defensive against missing rows or broken links; stops at first gap.
        """
        ordered: list[TaskRow] = []
        cur = head_row
        seen: set[int] = set()
        while cur is not None:
            tid = cur.get("task_id")
            try:
                if isinstance(tid, int) and tid in seen:
                    break
                if isinstance(tid, int):
                    seen.add(tid)
            except Exception:
                pass
            ordered.append(cur)
            sched = cur.get("schedule") or {}
            nxt = sched.get("next_task")
            if nxt is None:
                break
            nxt_rows = self._filter_tasks(filter=f"task_id == {int(nxt)}", limit=1)
            cur = nxt_rows[0] if nxt_rows else None
        return ordered

    @_ts_log_tool_runtime
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
        rows = [
            r
            for r in self._filter_tasks()
            if r.get("schedule") is not None
            and self._to_status(r.get("status")) not in self._TERMINAL_STATUSES
        ]
        # heads are rows with prev_task == None
        heads: list[TaskRow] = [
            r for r in rows if (r.get("schedule") or {}).get("prev_task") is None
        ]
        out: list[Dict[str, Any]] = []
        for h in heads:
            sched = h.get("schedule") or {}
            start_at = sched.get("start_at")
            qid = h.get("queue_id")
            if not isinstance(qid, int):
                # Skip any legacy rows lacking a numeric queue_id
                continue
            chain = self._walk_queue(h)
            out.append(
                {
                    "queue_id": qid,
                    "queue_label": f"Q{qid}",
                    "head_id": h.get("task_id"),
                    "size": len(chain),
                    "start_at": start_at,
                },
            )
        return out

    @_ts_log_tool_runtime
    def _get_queue(
        self,
        *,
        queue_id: Optional[int] = None,
        strict: bool = True,
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
        head = self._head_row_for_queue(queue_id)
        if head is None:
            return []
        rows = self._walk_queue(head)
        ordered: List[Task] = []
        for row in rows:
            # Strip stale activation metadata on non-active rows
            _row = dict(row)
            try:
                if self._to_status(_row.get("status")) != Status.active:  # type: ignore[arg-type]
                    _row.pop("activated_by", None)
            except Exception:
                if str(_row.get("status")) != str(Status.active):
                    _row.pop("activated_by", None)
            ordered.append(Task(**_row))
        return ordered

    @_ts_log_tool_runtime
    def _walk_queue_from_task(self, *, task_id: int) -> List[Task]:
        """
        Walk the chain that contains `task_id` by following schedule.prev_task to
        the head and then schedule.next_task forward, returning rows as `Task`.

        This helper ignores the top-level queue_id and is used when a task does
        not carry a numeric queue_id but still belongs to a linked chain.
        """
        # Locate the starting row
        try:
            cur_row = self._get_single_row_or_raise(int(task_id))
        except Exception:
            return []

        # Walk to head using prev_task pointers
        head = cur_row
        try:
            while head is not None:
                prev_id = (head.get("schedule") or {}).get("prev_task")
                if prev_id is None:
                    break
                prev_rows = self._filter_tasks(
                    filter=f"task_id == {int(prev_id)}",
                    limit=1,
                )
                head = prev_rows[0] if prev_rows else None
        except Exception:
            pass

        if head is None:
            return []

        # Walk forward using next_task pointers; include terminal rows for context
        ordered: List[Task] = []
        node = head
        seen: set[int] = set()
        while node is not None:
            tid = node.get("task_id")
            try:
                tid_int = int(tid) if tid is not None else None
            except Exception:
                tid_int = None  # type: ignore[assignment]
            if tid_int is not None and tid_int in seen:
                break
            if tid_int is not None:
                seen.add(tid_int)
            # Strip stale activation metadata on non-active rows
            _row = dict(node)
            try:
                if self._to_status(_row.get("status")) != Status.active:  # type: ignore[arg-type]
                    _row.pop("activated_by", None)
            except Exception:
                if str(_row.get("status")) != str(Status.active):
                    _row.pop("activated_by", None)
            ordered.append(Task(**_row))
            nxt_id = (node.get("schedule") or {}).get("next_task")
            if nxt_id is None:
                break
            nxt_rows = self._filter_tasks(filter=f"task_id == {int(nxt_id)}", limit=1)
            node = nxt_rows[0] if nxt_rows else None

        return ordered

    @_ts_log_tool_runtime
    def _get_queue_for_task(self, *, task_id: int) -> List[Task]:
        """
        Return the runnable queue (head→tail) containing `task_id`.

        Strategy
        --------
        - If the row has a numeric `queue_id`, delegate to `_get_queue(queue_id)`.
        - Otherwise, walk the chain ignoring queue_id via `_walk_queue_from_task`.
        """
        try:
            row = self._get_single_row_or_raise(int(task_id))
        except Exception:
            return []

        qid = row.get("queue_id")
        if isinstance(qid, int):
            return self._get_queue(queue_id=qid)
        return self._walk_queue_from_task(task_id=int(task_id))

    @_ts_log_tool_runtime
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
        - If you need to insert or remove members from a queue, prefer
          :pyfunc:`_set_queue` or combine :pyfunc:`_move_tasks_to_queue` with
          this method.

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
        - Ensures exactly one head owns ``start_at`` (carried from the former
          head if present) and sets statuses consistently:
            • head with ``start_at`` → ``scheduled``;
            • non-heads → at most ``queued``.
        - The active task (if any) in this queue retains its ``active`` status.

        Guidance for callers (outer loop / LLM):
        - Always refresh the queue membership immediately before constructing `new_order`
          by calling `list_queues()` and `get_queue(queue_id=…)`.
        - If a task was started via `execute_isolated_by_id`, it is DETACHED and no longer a
          member of its former queue; do NOT include it in `new_order` for that queue.
        - This method asserts that `new_order` is an exact permutation of the current queue;
          if you see an assertion error, refresh state and reconstruct `new_order` accordingly.
        """
        # Build current queue membership directly from storage to avoid
        # assumptions about a single visible head during transitions.
        all_rows = self._filter_tasks()
        in_queue_rows: list[TaskRow] = [
            r
            for r in all_rows
            if r.get("schedule") is not None
            and r.get("queue_id") == queue_id
            and self._to_status(r.get("status")) not in self._TERMINAL_STATUSES
        ]
        current_set: set[int] = {int(r.get("task_id")) for r in in_queue_rows}
        if current_set != set(new_order):
            raise AssertionError(
                "new_order must be a permutation of the current queue. "
                f"Current members: {sorted(list(current_set))}; "
                f"Provided: {sorted(list(set(new_order)))}. "
                f"Refresh with list_queues() and get_queue(queue_id={queue_id}) "
                "then rebuild new_order accordingly.",
            )

        rows_by_id: Dict[int, TaskRow] = {
            r["task_id"]: r
            for r in self._filter_tasks(filter=f"task_id in {new_order}")
        }

        # Compute an invariant-preserving update plan via QueueEngine
        updates_per_log: Dict[int, Dict[str, Any]] = plan_reorder_queue(
            new_order=new_order,
            rows_by_id=rows_by_id,
            queue_id=queue_id,
        )

        # Persist through the central validated write funnel
        for tid, payload in updates_per_log.items():
            self._validated_write(
                task_id=tid,
                entries=payload,
                err_prefix=f"While reordering queue {queue_id} (task {tid}):",
            )

        # Auto-checkpoint after successful edit (best-effort)
        try:
            # Silent; the execute loop also checkpoints explicitly
            _ = self._queue_checkpoints  # guard attribute existence
            # Build lightweight snapshot of just this queue
            from ..common.llm_helpers import short_id as _short_id  # local import

            cid = _short_id(8)
            snap = {"label": "auto:_reorder_queue", "queues": []}
            try:
                order_now = [t.task_id for t in self._get_queue(queue_id=queue_id)]
            except Exception:
                order_now = list(new_order)
            head_start = None
            try:
                head = self._head_row_for_queue(queue_id)
                head_start = (
                    (head.get("schedule") or {}).get("start_at") if head else None
                )
            except Exception:
                head_start = None
            snap["queues"].append(
                {
                    "queue_id": queue_id,
                    "head_id": order_now[0] if order_now else None,
                    "start_at": head_start,
                    "order": order_now,
                },
            )
            self._queue_checkpoints[cid] = snap
            # Expose the checkpoint id in the return payload via outer scope variable
            _last_checkpoint_id = cid  # noqa: F841 (read by return below if available)
        except Exception:
            pass

        return {
            "outcome": "queue reordered",
            "details": {
                "queue_id": queue_id,
                "new_order": new_order,
                "checkpoint_id": locals().get("_last_checkpoint_id"),
            },
        }

    @_ts_log_tool_runtime
    def _move_tasks_to_queue(
        self,
        *,
        task_ids: List[int],
        queue_id: Optional[int] = None,
        position: Optional[str] = "back",
    ) -> ToolOutcome:
        """
        Move one or more runnable tasks to a specific queue and position.

        Parameters
        ----------
        task_ids : list[int]
            Identifiers of tasks to move. All tasks must be non-terminal. The
            currently active task cannot be moved.
        queue_id : int | None, default ``None``
            Target queue identifier. When
            you intend to create a brand-new queue, pass ``queue_id=None`` and
            this tool will allocate a fresh identifier automatically and return
            it in the details payload.
        position : {"front", "back"} | None, default "back"
            Where to insert the moved tasks relative to the target queue.
            - "front": insert as a block at the front (preserving the order of
              ``task_ids`` as provided);
            - "back": append at the end (preserving order);
            - None: keep target queue unchanged (only change queue membership; the
              tasks will not be linked unless you call `_reorder_queue` next).

        Behaviour
        ---------
        - Detaches the tasks from their original queues by fixing neighbour links.
        - Assigns the provided/allocated ``queue_id`` to the moved tasks.
        - If inserting at the front or back, updates neighbour links to splice the
          block into the target queue.
        - Preserves the target queue's head-level ``start_at`` by keeping it on the
          new head after insertion (if any).
        - Sets statuses consistently (head with ``start_at`` → ``scheduled``; others
          at most ``queued``). Active tasks are rejected.

        When to use
        -----------
        - Move tasks between queues or create a new queue and place a block at
          the front/back. For a single-shot, precise materialization (including
          removing other members), prefer :pyfunc:`_set_queue`.

        Returns
        -------
        ToolOutcome
            ``{"outcome": "tasks moved", "details": {"queue_id": <int>, "task_ids": [...]}}``
        """
        # Validate and normalise inputs
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if not task_ids:
            return {
                "outcome": "tasks moved",
                "details": {"queue_id": queue_id, "task_ids": []},
            }

        # Reject active and terminal tasks
        rows = self._filter_tasks(filter=f"task_id in {task_ids}")
        ids_found = {r.get("task_id") for r in rows}
        missing = [tid for tid in task_ids if tid not in ids_found]
        assert not missing, f"Unknown task ids: {missing}"
        for r in rows:
            st = self._to_status(r.get("status"))
            assert st not in self._TERMINAL_STATUSES, f"Task {r['task_id']} is terminal"
        # Guard against touching the active task via the central helper
        self._ensure_not_active_task(task_ids)
        # Reject moving trigger-based tasks into a runnable queue
        for r in rows:
            if r.get("trigger") is not None:
                raise ValueError(
                    f"Task {r['task_id']} is trigger-based and cannot be placed in the queue.",
                )

        # Allocate queue id when requested
        target_qid = queue_id if queue_id is not None else self._allocate_new_queue_id()

        # Snapshot the current target queue order BEFORE modifying membership to
        # avoid a transient multi-head state (which would make _get_queue assert
        # and tempt fallback paths to drop existing members).
        try:
            existing_order: List[int] = [
                t.task_id for t in self._get_queue(queue_id=target_qid)
            ]
        except Exception:
            existing_order = []

        # 1) Detach each task from its current queue (fix predecessor/successor)
        def _get_row(tid: int) -> TaskRow:
            return self._get_single_row_or_raise(tid)

        for tid in task_ids:
            r = _get_row(tid)
            sched = r.get("schedule") or {}
            prev_tid = sched.get("prev_task")
            next_tid = sched.get("next_task")
            # Fix predecessor's next pointer
            if prev_tid is not None:
                prev_row = self._get_single_row_or_raise(int(prev_tid))
                prev_sched = {**(prev_row.get("schedule") or {})}
                if prev_sched.get("next_task") == tid:
                    prev_sched["next_task"] = next_tid
                    self._validated_write(
                        task_id=int(prev_row["task_id"]),
                        entries={"schedule": prev_sched},
                        err_prefix=f"While detaching predecessor for move of task {tid}:",
                    )
            # Fix successor's prev pointer and remove start_at (no longer head)
            if next_tid is not None:
                next_row = self._get_single_row_or_raise(int(next_tid))
                next_sched = {**(next_row.get("schedule") or {})}
                if next_sched.get("prev_task") == tid:
                    next_sched.pop("start_at", None)
                    next_sched["prev_task"] = prev_tid
                    self._validated_write(
                        task_id=int(next_row["task_id"]),
                        entries={"schedule": next_sched},
                        err_prefix=f"While detaching successor for move of task {tid}:",
                    )

            # Clear this task's own linkage and queue id (will set below)
            new_sched: Dict[str, Any] = {
                "prev_task": None,
                "next_task": None,
            }
            self._validated_write(
                task_id=tid,
                entries={
                    "schedule": new_sched,
                    "status": Status.queued,
                    "queue_id": target_qid,
                },
                err_prefix=f"While moving task {tid} to queue {target_qid}:",
            )

        # 2) Materialize the target queue order via core primitive
        block = list(task_ids)
        # Compose new order from the pre-detach snapshot to preserve prior members
        base_order = [tid for tid in existing_order if tid not in block]
        if position == "front":
            new_order = block + base_order
        elif position == "back":
            new_order = base_order + block
        else:
            # Keep target queue order unchanged, but ensure members are present
            new_order = base_order + block

        if new_order:
            self._set_queue(queue_id=target_qid, order=new_order)

        # Auto-checkpoint after successful edit (best-effort)
        try:
            from ..common.llm_helpers import short_id as _short_id  # local import

            cid = _short_id(8)
            snap = {"label": "auto:_move_tasks_to_queue", "queues": []}
            # Capture target queue only for brevity
            order_now = [t.task_id for t in self._get_queue(queue_id=target_qid)]
            head_start = None
            try:
                head = self._head_row_for_queue(target_qid)
                head_start = (
                    (head.get("schedule") or {}).get("start_at") if head else None
                )
            except Exception:
                head_start = None
            snap["queues"].append(
                {
                    "queue_id": target_qid,
                    "head_id": order_now[0] if order_now else None,
                    "start_at": head_start,
                    "order": order_now,
                },
            )
            self._queue_checkpoints[cid] = snap
            _last_checkpoint_id = cid  # noqa: F841
        except Exception:
            pass

        return {
            "outcome": "tasks moved",
            "details": {
                "queue_id": target_qid,
                "task_ids": list(task_ids),
                "checkpoint_id": locals().get("_last_checkpoint_id"),
            },
        }

    # ------------------------------------------------------------------ #
    #  Atomic queue materialization                                       #
    # ------------------------------------------------------------------ #

    @_ts_log_tool_runtime
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

        rows = self._filter_tasks(filter=f"task_id in {order}")
        ids_found = {r.get("task_id") for r in rows}
        missing = [tid for tid in order if tid not in ids_found]
        assert not missing, f"Unknown task ids: {missing}"
        for r in rows:
            st = self._to_status(r.get("status"))
            assert st not in self._TERMINAL_STATUSES, f"Task {r['task_id']} is terminal"
        # Reject placing trigger-based tasks into a runnable queue
        for r in rows:
            if r.get("trigger") is not None:
                raise ValueError(
                    f"Task {r['task_id']} is trigger-based and cannot be placed in the queue.",
                )
        # Allow editing a queue that includes the currently active task; preserve its status below
        active_tid: Optional[int] = None
        try:
            if self._active_task is not None:
                active_tid = int(self._active_task.task_id)
        except Exception:
            active_tid = None

        # Allocate queue id when needed
        target_qid = queue_id if queue_id is not None else self._allocate_new_queue_id()

        # Capture existing head-level start_at BEFORE any mutations so it can be
        # restored onto the new head reliably (avoids losing it during neutralisation)
        existing_head_start: Optional[str] = None
        try:
            _orig_head = self._head_row_for_queue(target_qid)
            if _orig_head is not None:
                existing_head_start = (_orig_head.get("schedule") or {}).get("start_at")
        except Exception:
            existing_head_start = None
        # Fallback: if no head detected for this queue yet, derive from any current
        # member in the provided order that is a head and owns a start_at
        if existing_head_start is None:
            try:
                for _tid in order:
                    _row = self._get_single_row_or_raise(int(_tid))
                    _sched = _row.get("schedule") or {}
                    if (
                        _sched.get("start_at") is not None
                        and _sched.get("prev_task") is None
                    ):
                        existing_head_start = _sched.get("start_at")
                        break
            except Exception:
                pass

        # Remove any other members currently in the target queue (strict by queue_id)
        try:
            _rows_all = self._filter_tasks()
            current_members = [
                int(r.get("task_id"))
                for r in _rows_all
                if r.get("schedule") is not None
                and r.get("queue_id") == target_qid
                and self._to_status(r.get("status")) not in self._TERMINAL_STATUSES
            ]
        except Exception:
            current_members = []

        to_remove = [tid for tid in current_members if tid not in order]
        if to_remove:
            # Detach removed tasks: clear prev/next on schedule; set top-level queue_id=None
            for tid in to_remove:
                row = self._get_single_row_or_raise(int(tid))
                sched = {**(row.get("schedule") or {})}
                # Clear any neighbour pointers and start_at on non-heads
                sched.pop("prev_task", None)
                sched.pop("next_task", None)
                sched.pop("start_at", None)
                self._validated_write(
                    task_id=int(tid),
                    entries={
                        "schedule": sched or {},
                        # Ensure invariants: a detached task without start_at must not be 'scheduled'
                        "status": Status.queued,
                        "queue_id": None,
                    },
                    err_prefix=f"While clearing removed task {tid} from queue {target_qid}:",
                )

        # Ensure all specified tasks belong to target queue and have no conflicting links
        for tid in order:
            row = self._get_single_row_or_raise(int(tid))
            sched = {**(row.get("schedule") or {})}
            # neutralize prev/next; set precisely below
            sched["prev_task"] = None
            sched["next_task"] = None
            # Remove start_at for now; we will reapply for head only
            sched.pop("start_at", None)
            # Preserve 'active' status on the currently running task by omitting a status write
            prep_entries: Dict[str, Any] = {"schedule": sched, "queue_id": target_qid}
            if active_tid is None or int(tid) != int(active_tid):
                prep_entries["status"] = Status.queued  # temporary neutral state
            self._validated_write(
                task_id=int(tid),
                entries=prep_entries,
                err_prefix=f"While preparing task {tid} for queue materialization:",
            )

        # Rewire links to match order and apply head start_at
        for idx, tid in enumerate(order):
            prev_tid = None if idx == 0 else order[idx - 1]
            next_tid = None if idx == len(order) - 1 else order[idx + 1]
            sched = {
                "prev_task": prev_tid,
                "next_task": next_tid,
            }
            if idx == 0:
                # Prefer provided queue_start_at; else preserve previously-captured head start
                if queue_start_at is not None:
                    sched["start_at"] = queue_start_at
                elif existing_head_start is not None:
                    sched["start_at"] = existing_head_start

            # Derive status for non-active tasks only; keep the active task as 'active'
            write_entries: Dict[str, Any] = {"schedule": sched, "queue_id": target_qid}
            if active_tid is None or int(tid) != int(active_tid):
                desired_status = (
                    Status.scheduled
                    if (idx == 0 and ("start_at" in sched))
                    else Status.queued
                )
                write_entries["status"] = desired_status
            self._validated_write(
                task_id=int(tid),
                entries=write_entries,
                err_prefix=f"While materializing queue {target_qid} (task {tid}):",
            )

        # Safety: explicitly (re)apply start_at on the new head using the public
        # helper to enforce invariants, regardless of intermediate writes.
        try:
            if order:
                head_tid = int(order[0])
                source_start = (
                    queue_start_at
                    if queue_start_at is not None
                    else existing_head_start
                )
                if source_start is not None:
                    from datetime import datetime as _dt

                    _src = str(source_start)
                    if _src.endswith("Z"):
                        _src = _src.replace("Z", "+00:00")
                    try:
                        dt = (
                            source_start
                            if isinstance(source_start, _dt)
                            else _dt.fromisoformat(_src)
                        )
                    except Exception:
                        dt = source_start  # type: ignore[assignment]
                    self._update_task(task_id=head_tid, start_at=dt)  # type: ignore[arg-type]
        except Exception:
            pass

        # Auto-checkpoint
        try:
            from ..common.llm_helpers import short_id as _short_id  # local import

            cid = _short_id(8)
            snap = {"label": "auto:_set_queue", "queues": []}
            order_now = [t.task_id for t in self._get_queue(queue_id=target_qid)]
            head_start = None
            try:
                head = self._head_row_for_queue(target_qid)
                head_start = (
                    (head.get("schedule") or {}).get("start_at") if head else None
                )
            except Exception:
                head_start = None
            snap["queues"].append(
                {
                    "queue_id": target_qid,
                    "head_id": order_now[0] if order_now else None,
                    "start_at": head_start,
                    "order": order_now,
                },
            )
            self._queue_checkpoints[cid] = snap
            _last_checkpoint_id = cid  # noqa: F841
        except Exception:
            pass

        return {
            "outcome": "queue set",
            "details": {
                "queue_id": target_qid,
                "order": list(order),
                "checkpoint_id": locals().get("_last_checkpoint_id"),
            },
        }

    # ------------------------------------------------------------------ #
    #  Bulk low-level schedule edit (atomic)                              #
    # ------------------------------------------------------------------ #

    @_ts_log_tool_runtime
    def _set_schedules_atomic(
        self,
        *,
        schedules: List[Dict[str, Any]],
    ) -> ToolOutcome:
        """
        Apply multiple schedule edits atomically with graph validation.

        Each item: {"task_id": int, "schedule": {"queue_id": int | None,
        "prev_task": int | None, "next_task": int | None, "start_at"?: str}}

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

        # Build a local view and validate cross-refs
        by_id: Dict[int, Dict[str, Any]] = {}
        for item in schedules:
            tid = int(item.get("task_id"))
            sch = dict(item.get("schedule") or {})
            # Ignore any nested queue_id inside schedule; rely on top-level queue_id
            sch.pop("queue_id", None)
            by_id[tid] = sch

        # Fetch rows
        rows = self._filter_tasks(filter=f"task_id in {list(by_id.keys())}")
        ids_found = {r.get("task_id") for r in rows}
        missing = [tid for tid in by_id.keys() if tid not in ids_found]
        assert not missing, f"Unknown task ids: {missing}"
        for r in rows:
            st = self._to_status(r.get("status"))
            assert st not in self._TERMINAL_STATUSES, f"Task {r['task_id']} is terminal"
        self._ensure_not_active_task(list(by_id.keys()))

        # Cross-queue guard and adjacency graph
        graph: Dict[int, List[int]] = {tid: [] for tid in by_id.keys()}
        for tid, sch in by_id.items():
            # Use the provided top-level queue_id if present; otherwise derive from storage
            qid = None
            try:
                for it in schedules:
                    if int(it.get("task_id")) == int(tid):
                        qid = it.get("queue_id")
                        break
            except Exception:
                qid = None
            for nbr_key in ("prev_task", "next_task"):
                nbr = sch.get(nbr_key)
                if nbr is None:
                    continue
                nbr = int(nbr)
                # ensure referenced schedule present; fallback to storage view
                nbr_row = self._get_single_row_or_raise(nbr)
                nbr_sched = dict((nbr_row.get("schedule") or {}))
                nbr_qid = None
                # prefer top-level for neighbour if provided in the batch
                try:
                    for it in schedules:
                        if int(it.get("task_id")) == int(nbr):
                            nbr_qid = it.get("queue_id")
                            break
                except Exception:
                    nbr_qid = None
                if nbr_qid is None:
                    nbr_qid = nbr_row.get("queue_id")
                if nbr_qid != qid:
                    raise ValueError(
                        f"Cross-queue link rejected: task {tid} (qid={qid}) → {nbr_key}={nbr} (qid={nbr_qid}).",
                    )
                graph[tid].append(nbr)

        # Cycle/head validation by queue_id groups
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

        # Head/start_at rule: start_at only on heads
        for tid, sch in by_id.items():
            prev_tid = sch.get("prev_task")
            if sch.get("start_at") is not None and prev_tid is not None:
                raise ValueError(f"Only heads may define start_at (task {tid})")

        # Apply atomically: write each via validated funnel
        for tid, sch in by_id.items():
            is_head = sch.get("prev_task") is None
            desired_status = (
                Status.scheduled
                if (is_head and sch.get("start_at") is not None)
                else Status.queued
            )
            # Top-level queue_id must be provided explicitly per item or preserved from current row
            top_qid = None
            try:
                for it in schedules:
                    if int(it.get("task_id")) == int(tid):
                        top_qid = it.get("queue_id")
                        break
            except Exception:
                top_qid = None
            if top_qid is None:
                try:
                    current_row = self._get_single_row_or_raise(int(tid))
                    top_qid = current_row.get("queue_id")
                except Exception:
                    top_qid = None
            self._validated_write(
                task_id=int(tid),
                entries={
                    "schedule": sch,
                    "status": desired_status,
                    **({"queue_id": int(top_qid)} if isinstance(top_qid, int) else {}),
                },
                err_prefix=f"While applying set_schedules_atomic (task {tid}):",
            )

        return {"outcome": "schedules updated", "details": {"count": len(by_id)}}

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    # Deprecated: _explain_queue removed. Compose using _list_queues() and _get_queue().

    @_ts_log_tool_runtime
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
            - ``queue_name`` (str | None, optional): unused metadata; accepted for
              future compatibility.

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
                _head_row = self._get_single_row_or_raise(_head_tid)
                source_qid = _head_row.get("queue_id")
        except Exception:
            source_qid = None

        # Current queue snapshot (head→tail) for the identified source queue
        original = [t.task_id for t in self._get_queue(queue_id=source_qid)]
        if not original:
            return {"outcome": "queue partitioned", "details": {"queues": []}}

        # Normalise per-part order
        def _ordered(ids: List[int]) -> List[int]:
            if strategy == "as_list":
                return list(ids)
            # preserve original relative order
            rank = {tid: i for i, tid in enumerate(original)}
            return sorted(ids, key=lambda x: rank.get(x, 10**9))

        queue_start_ts = None  # Remember original queue-level timestamp
        head_row = self._head_row_for_queue(source_qid)
        if head_row is not None:
            queue_start_ts = (head_row.get("schedule") or {}).get("start_at")

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

        for j, tids in groups.items():
            ordered = _ordered(tids)
            qid = self._allocate_new_queue_id()
            # Materialize the new queue in one step via core primitive
            self._set_queue(queue_id=qid, order=ordered)
            # Apply queue_start_at if provided
            qstart = later_parts[j].get("queue_start_at")
            if qstart is not None and ordered:
                # Set via central helper to preserve invariants
                head_tid = ordered[0]
                self._update_task(task_id=head_tid, start_at=qstart)
            created.append({"queue_id": qid, "task_ids": ordered})

        # 2) Reduce the source queue to the first part (in chosen order)
        first_list = _ordered(list(first_ids))
        # Reorder source queue to include only these tasks: move out everything else already done
        if first_list:
            self._set_queue(queue_id=source_qid, order=first_list)
            # apply provided start_at or carry the original one
            fstart = parts[0].get("queue_start_at") if parts else None
            if fstart is not None:
                pass
            else:
                fstart = queue_start_ts
            if fstart is not None:
                head_tid = first_list[0]
                self._update_task(task_id=head_tid, start_at=fstart)

        details = {"default_queue": first_list, "new_queues": created}
        # Auto-checkpoint after successful edit (best-effort)
        try:
            from ..common.llm_helpers import short_id as _short_id  # local import

            cid = _short_id(8)
            snap = {"label": "auto:_partition_queue", "queues": []}
            for qinfo in self._list_queues():
                qid = qinfo.get("queue_id")
                order_now = [t.task_id for t in self._get_queue(queue_id=qid)]
                snap["queues"].append(
                    {
                        "queue_id": qid,
                        "head_id": qinfo.get("head_id"),
                        "start_at": qinfo.get("start_at"),
                        "order": order_now,
                    },
                )
            self._queue_checkpoints[cid] = snap
            _last_checkpoint_id = cid  # noqa: F841
        except Exception:
            pass

        return {
            "outcome": "queue partitioned",
            "details": {
                **details,
                "checkpoint_id": locals().get("_last_checkpoint_id"),
            },
        }

    # ------------------------------------------------------------------ #
    #  Centralised schedule/status write with invariant validation        #
    # ------------------------------------------------------------------ #

    @_ts_log_tool_runtime
    def _validated_write(
        self,
        *,
        task_id: int,
        entries: Dict[str, Any],
        err_prefix: str,
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
        current = self._get_single_row_or_raise(task_id)

        prospective_schedule = entries.get("schedule", current.get("schedule"))
        prospective_status = entries.get("status", current.get("status"))
        prospective_trigger = entries.get("trigger", current.get("trigger"))

        # Belt-and-braces: forbid setting status to 'active' via this funnel.
        norm_status = None
        if "status" in entries:
            try:
                norm_status = self._to_status(entries["status"])  # type: ignore[arg-type]
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
        if "schedule" in entries and prospective_schedule is not None:
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

                def _qid_of(tid: Optional[int]) -> Optional[int]:
                    if tid is None:
                        return None
                    rows = self._filter_tasks(filter=f"task_id == {int(tid)}", limit=1)
                    if not rows:
                        return None
                    srow = rows[0]
                    return srow.get("queue_id")

                # Only enforce when linkage exists
                for _nbr, _tid in (("prev_task", prev_tid), ("next_task", next_tid)):
                    if _tid is None:
                        continue
                    nbr_qid = _qid_of(_tid)
                    if nbr_qid != qid:
                        raise ValueError(
                            f"{err_prefix} cross-queue link rejected: {_nbr}={_tid} has queue_id={nbr_qid} "
                            f"but current task would be in queue_id={qid}. Use set_queue() or move_tasks_to_queue() "
                            f"followed by reorder_queue() to materialize chains within a single queue.",
                        )
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

        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        result = self._write_log_entries(logs=log_id, entries=entries)

        # Ensure neighbour symmetry whenever schedule changed
        if "schedule" in entries:
            self._sync_adjacent_links(task_id=task_id, schedule=prospective_schedule)

        return result

    # ------------------------------------------------------------------ #
    #  Centralised helpers for queue link manipulation                    #
    # ------------------------------------------------------------------ #

    @_ts_log_tool_runtime
    def _detach_from_queue_for_activation(
        self,
        *,
        task_id: int,
        detach: bool = True,
    ) -> None:
        _ops_detach_for_activation(
            self,
            task_id=task_id,
            detach=detach,
        )

    @_ts_log_tool_runtime
    def _attach_with_links(
        self,
        *,
        task_id: int,
        prev_task: Optional[int],
        next_task: Optional[int],
        head_start_at: Optional[str],
        err_prefix: str,
    ) -> None:
        _ops_attach_with_links(
            self,
            task_id=task_id,
            prev_task=prev_task,
            next_task=next_task,
            head_start_at=head_start_at,
            err_prefix=err_prefix,
        )

    _TERMINAL_STATUSES = {Status.completed, Status.cancelled, Status.failed}

    # ------------------------------------------------------------------ #
    #  Public lifecycle helpers                                           #
    # ------------------------------------------------------------------ #

    def reinstate_to_previous_queue(
        self,
        *,
        task_id: int,
        allow_active: bool = False,
    ) -> Dict[str, str]:
        """
        Public facade to restore a task to its prior queue/schedule position.

        Delegates to the internal `_reinstate_task_to_previous_queue` and maps
        the `allow_active` flag to the private `_allow_active` parameter.
        """
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
            task_id = self._primed_task["task_id"]
        if task_id is None:
            return

        rows = self._filter_tasks(filter=f"task_id == {task_id}", limit=1)
        row = rows[0] if rows else None
        # Only cache when the referenced row is actually in 'primed' state
        if row is not None and self._to_status(row.get("status")) == Status.primed:
            self._primed_task = row
        else:
            self._primed_task = None

    # Deprecated: _get_task_queue removed in favor of explicit helpers.

    # Small helper for ActiveQueue to await linkage stabilisation
    @_ts_log_tool_runtime
    def _get_linkage_barrier(self, *, task_id: int):
        try:
            return self._linkage_barriers.get(int(task_id))
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Pure neighbour selection helper                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _select_final_neighbours(
        *,
        task_id: int,
        was_head: bool,
        original_prev: Optional[int],
        original_next: Optional[int],
        queue_ids: list[int],
        is_viable: "Callable[[Optional[int]], bool]",
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Decide `(final_prev, final_next)` for reinstatement using a minimal, deterministic
        policy and no I/O.

        Policy:
        - If `was_head`: `final_prev=None`; `final_next` is `original_next` if viable,
          otherwise the current head (first in `queue_ids`) if different from `task_id`,
          otherwise `None`.
        - If middle: `final_prev` is `original_prev` if viable, else `None`;
          `final_next` is `original_next` if viable and distinct from `final_prev`, else `None`.
        - Avoid self-loops and identical prev/next; prefer keeping prev and dropping next.
        """
        current_head_id = queue_ids[0] if queue_ids else None

        def _clean(tid: Optional[int]) -> Optional[int]:
            return None if tid == task_id else tid

        if was_head:
            final_prev = None
            if is_viable(original_next):
                final_next = original_next
            else:
                final_next = (
                    current_head_id
                    if (current_head_id is not None and current_head_id != task_id)
                    else None
                )
        else:
            # Prefer restoring the original predecessor when still viable; else become head.
            final_prev = original_prev if is_viable(original_prev) else None
            # Preserve the original successor when viable and not equal to prev.
            final_next = (
                original_next
                if (is_viable(original_next) and original_next != final_prev)
                else None
            )

        final_prev = _clean(final_prev)
        final_next = _clean(final_next)
        if (
            final_prev is not None
            and final_next is not None
            and final_prev == final_next
        ):
            final_next = None

        return final_prev, final_next

    # Deprecated: _update_task_queue removed. Use _reorder_queue/_set_queue directly.

    # Legacy per-field updaters are no longer exposed as tools. Unified into _update_task.

    # Update Task(s) Status / Schedule / Deadline / Repetition / Priority

    @_ts_log_tool_runtime
    def _update_task_status(
        self,
        *,
        task_ids: Union[int, List[int]],
        new_status: str,
        allow_active: bool = False,
    ) -> Dict[str, str]:
        """
        Change the **lifecycle status** of one or many tasks.

        Notes:
        - Setting status to 'active' directly is forbidden. Activation is performed
          exclusively by the execution path (execute / execute_by_id).
        """
        # Forbid making anything active (unless explicitly allowed)
        new_status_enum = self._to_status(new_status)
        if new_status_enum == Status.active:
            raise ValueError(
                "Direct status changes to 'active' are not allowed; use the dedicated activation method.",
            )

        # Forbid touching the existing active task (always) – lifecycle changes
        # to the active task must go through the live ActiveTask handle.
        self._ensure_not_active_task(task_ids)

        # Invariant checks for queue/schedule-sensitive statuses
        if new_status_enum in {Status.scheduled, Status.queued}:
            rows = self._filter_tasks(filter=f"task_id in {task_ids}")
            for row in rows:
                self._validate_scheduled_invariants(
                    status=new_status_enum,
                    schedule=row.get("schedule"),
                    err_prefix=f"While changing status of task {row['task_id']}:",
                )

        # ToDo: replace with single API call once this task [https://app.clickup.com/t/86c3c1y63] is done
        log_ids = self._get_logs_by_task_ids(task_ids=task_ids)
        entries: Dict[str, Any] = {"status": new_status_enum}
        return self._write_log_entries(logs=log_ids, entries=entries, overwrite=True)

    @_ts_log_tool_runtime
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

        Returns
        -------
        dict
            Confirmation payload from the write operation.
        """

        # Forbid edits on the currently active task via scheduler APIs
        self._ensure_not_active_task(task_id)

        # Was 'trigger' explicitly provided (even if None)?
        _trigger_provided = trigger is not _UNSET

        # Fetch current row for invariants/derivations
        row = self._get_single_row_or_raise(int(task_id))
        current_sched = row.get("schedule") or {}

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
        ):
            raise ValueError("At least one field must be provided for an update.")

        # Mutually exclusive guard: trigger with any schedule/start_at
        if _trigger_provided and trigger is not None:
            # If the update itself adds a start_at or the current schedule is present, reject
            if start_at is not None:
                raise ValueError("Cannot set a trigger alongside a start_at schedule.")
            if row.get("schedule") is not None:
                raise ValueError(
                    "Cannot add a trigger while a schedule exists. Remove schedule first.",
                )

        # Build prospective schedule if start_at is supplied
        schedule_payload: Optional[Dict[str, Any]] = None
        if start_at is not None:
            # Disallow start_at when the task is trigger-based
            # Allow when the update explicitly clears the trigger in the same call
            if row.get("trigger") is not None and not (
                _trigger_provided and trigger is None
            ):
                raise ValueError(
                    "Cannot add/update *start_at* – the task is trigger-based.",
                )
            # Guard-rail: tasks with a predecessor cannot own start_at
            if self._sched_prev(current_sched) is not None:
                raise ValueError(
                    "Cannot set 'start_at' when the task has 'prev_task'. Move it to the queue head first.",
                )
            # Coerce datetime to ISO-8601 string if needed
            if not isinstance(start_at, str):
                try:
                    start_at = start_at.isoformat()  # type: ignore[assignment]
                except Exception:
                    pass
            schedule_payload = {
                "prev_task": self._sched_prev(current_sched),
                "next_task": self._sched_next(current_sched),
                "start_at": start_at,
            }

        # Determine desired status
        desired_status: Optional["Status"] = None
        if status is not None:
            # Forbid forcing 'active'
            status_enum = self._to_status(status)  # type: ignore[arg-type]
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
                status=(
                    desired_status if desired_status is not None else row.get("status")
                ),
                schedule=(
                    schedule_payload if schedule_payload is not None else current_sched
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
                if hasattr(r, "model_dump"):
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

        # If clearing a trigger (trigger explicitly None) and current status is triggerable
        if (
            _trigger_provided
            and (trigger is None)
            and (status is None)
            and self._to_status(row.get("status")) == Status.triggerable
        ):
            # Downgrade to queued when trigger removed (and not setting start_at)
            if schedule_payload is None:
                entries["status"] = Status.queued

        # Persist via central validated funnel when schedule/status involved; otherwise a direct write is fine
        if ("schedule" in entries) or ("status" in entries):
            return self._validated_write(
                task_id=task_id,
                entries=entries,
                err_prefix=f"While updating task {task_id}:",
            )
        else:
            log_id = self._get_logs_by_task_ids(task_ids=task_id)
            return self._write_log_entries(logs=log_id, entries=entries, overwrite=True)

    # Legacy per-field updaters are kept above as comments; use _update_task instead.

    # ────────────────────────────────────────────────────────────────────
    # Small internal helpers
    # ────────────────────────────────────────────────────────────────────

    def _new_llm_client(self, model: str) -> "unify.AsyncUnify":
        """Construct a configured AsyncUnify client for the given model."""
        return unify.AsyncUnify(
            model,
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )

    # ------------------------------------------------------------------ #
    #  Queue plan + checkpoints (shared helpers exposed as tools)        #
    # ------------------------------------------------------------------ #

    def validate_queue_plan(self, *, plan: Dict[str, Any] | str) -> Dict[str, Any]:  # type: ignore[valid-type]
        """Validate a proposed queue plan (dict or JSON string) and return the normalised structure."""
        import json as _json

        class _LaterGroup(BaseModel):
            task_ids: List[int] = Field(min_length=1)
            queue_start_at: Optional[str] = None

        class _QueuePlan(BaseModel):
            now: List[int] = Field(min_length=1)
            later_groups: List[_LaterGroup] = Field(default_factory=list)
            notes: Optional[str] = None

        try:
            parsed = _json.loads(plan) if isinstance(plan, str) else plan
        except Exception as _e:  # noqa: N806
            raise ValueError(f"Invalid plan: {_e}")
        model = _QueuePlan.model_validate(parsed)
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
        import json as _json

        class _LaterGroup(BaseModel):
            task_ids: List[int] = Field(min_length=1)
            queue_start_at: Optional[str] = None

        class _QueuePlan(BaseModel):
            now: List[int] = Field(min_length=1)
            later_groups: List[_LaterGroup] = Field(default_factory=list)
            notes: Optional[str] = None

        try:
            parsed = _json.loads(plan) if isinstance(plan, str) else plan
        except Exception as _e:
            raise ValueError(f"Invalid plan: {_e}")
        model = _QueuePlan.model_validate(parsed)
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
            all_q = self._list_queues()
        except Exception:
            all_q = []
        for q in all_q:
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

        from ..common.llm_helpers import short_id as _short_id  # local import

        cid = _short_id(8)
        self._queue_checkpoints[cid] = snapshot
        try:
            if str(os.getenv("UNITY_TS_PERSIST_CHECKPOINTS", "")).lower() in {
                "1",
                "true",
                "yes",
            }:
                self._persist_checkpoint(cid, label, snapshot)
        except Exception:
            pass
        return {"outcome": "checkpointed", "details": {"checkpoint_id": cid}}

    def revert_to_checkpoint(self, *, checkpoint_id: str) -> Dict[str, Any]:
        """Revert all queues to a previously created checkpoint."""
        snap = self._queue_checkpoints.get(str(checkpoint_id))
        if snap is None:
            try:
                if str(os.getenv("UNITY_TS_PERSIST_CHECKPOINTS", "")).lower() in {
                    "1",
                    "true",
                    "yes",
                }:
                    snap = self._load_checkpoint(str(checkpoint_id))
            except Exception:
                snap = None
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
                        head_row = self._get_single_row_or_raise(head_tid)
                        sched = {**(head_row.get("schedule") or {})}
                        sched["start_at"] = start_at
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
            if str(os.getenv("UNITY_TS_PERSIST_CHECKPOINTS", "")).lower() in {
                "1",
                "true",
                "yes",
            }:
                latest = self._get_latest_persisted_checkpoint()
                if latest is not None:
                    return {"outcome": "ok", "details": latest}
            return {
                "outcome": "none",
                "details": {"checkpoint_id": None, "label": None},
            }
        except Exception:
            return {
                "outcome": "none",
                "details": {"checkpoint_id": None, "label": None},
            }

    @staticmethod
    def _to_status(value: Union[Status, str]) -> Status:
        """Canonicalise a status-like value to the Status enum.

        Tolerate missing values by treating None as 'queued'.
        """
        if isinstance(value, Status):
            return value
        if value is None:
            return Status.queued
        return Status(value)

    # Default tool-policy helpers
    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require search_tasks on the first step; auto thereafter."""
        if step_index < 1 and "search_tasks" in current_tools:
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
        """Require ask on the first step; auto thereafter."""
        if step_index < 1 and "ask" in current_tools:
            return ("required", {"ask": current_tools["ask"]})
        return ("auto", current_tools)

    # Centralised simple-field writer
    def _update_fields_if_not_active(
        self,
        *,
        task_id: int,
        entries: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Update arbitrary fields on a single task, guarding against active-task edits.

        This consolidates the repetitive pattern used by name/description/deadline/
        repetition/priority updates without changing behaviour.
        """
        self._ensure_not_active_task(task_id)
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        return self._write_log_entries(logs=log_id, entries=entries)

    # ------------------------------------------------------------------ #
    #  Small centralised write helper                                     #
    # ------------------------------------------------------------------ #

    def _write_log_entries(
        self,
        *,
        logs: Union[int, "unify.Log", List[Union[int, "unify.Log"]]],
        entries: Dict[str, Any],
        overwrite: bool = True,
    ) -> Dict[str, str]:
        """
        Centralised adapter for log writes. Keeps all mutation calls going
        through one place in the scheduler.
        """
        return self._store.update(logs=logs, entries=entries, overwrite=overwrite)

    # ------------------------------------------------------------------ #
    #  Optional checkpoint persistence                                   #
    # ------------------------------------------------------------------ #

    def _persist_checkpoint(
        self,
        cid: str,
        label: Optional[str],
        snapshot: Dict[str, Any],
    ) -> None:
        try:
            ctx = f"{self._ctx}/Checkpoints"
            if ctx not in unify.get_contexts():
                unify.create_context(ctx)
                unify.create_fields(
                    {
                        "checkpoint_id": "str",
                        "label": "str",
                        "payload": "json",
                    },
                    context=ctx,
                )
            unify.log(
                context=ctx,
                new=True,
                checkpoint_id=cid,
                label=label,
                payload=snapshot,
            )
        except Exception:
            pass

    def _load_checkpoint(self, cid: str) -> Optional[Dict[str, Any]]:
        try:
            ctx = f"{self._ctx}/Checkpoints"
            rows = unify.get_logs(
                context=ctx,
                filter=f"checkpoint_id == {cid!r}",
                limit=1,
                return_ids_only=False,
            )
            if not rows:
                return None
            log = rows[0]
            entries = getattr(log, "entries", {}) or {}
            return entries.get("payload")
        except Exception:
            return None

    def _get_latest_persisted_checkpoint(self) -> Optional[Dict[str, Any]]:
        try:
            ctx = f"{self._ctx}/Checkpoints"
            rows = unify.get_logs(context=ctx, offset=0, limit=1, return_ids_only=False)
            if not rows:
                return None
            log = rows[-1]
            entries = getattr(log, "entries", {}) or {}
            return {
                "checkpoint_id": entries.get("checkpoint_id"),
                "label": entries.get("label"),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Best-effort helper                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _best_effort(func: "Callable[[], Any]") -> None:
        """
        Execute func and intentionally swallow any exceptions.

        Use only for non-critical maintenance paths (e.g., cache refresh,
        neighbour status fix-ups) where failure must not affect correctness.
        """
        try:
            func()
        except Exception:
            pass

    @staticmethod
    def _normalize_filter_expr(expr: Optional[str]) -> Optional[str]:
        """Return a storage-compatible filter expression.

        Currently performs a minimal rewrite so attribute-style access to
        nested schedule fields (e.g. ``schedule.start_at``) matches how values
        are stored in Unify (``schedule['start_at']``).  Keep this intentionally
        conservative to avoid altering semantics of unrelated expressions.
        """
        if not isinstance(expr, str):
            return expr
        try:
            return expr.replace(".start_at", "['start_at']")
        except Exception:
            return expr

    def _make_request_clarification_tool(
        self,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> Callable[[str], "asyncio.Future[str]"]:
        """Return an async tool that bubbles a question up and awaits the answer.

        Behaviour and integration notes
        --------------------------------
        - This tool exists only when the outer TaskScheduler loop has been given
          clarification queues. If those queues are not present, the outer loop
          MUST NOT ask the user questions as part of its final response. It must
          proceed using sensible defaults or best guesses and briefly state the
          assumptions used. If an inner tool asks for clarification but this
          outer loop lacks clarification queues, explicitly tell the inner tool
          that no clarification channel is available and provide reasonable
          default values or concrete best‑guess parameters instead.

        The returned coroutine raises RuntimeError if queues are not provided at call time.
        """

        async def _request(question: str) -> str:
            if clarification_up_q is None or clarification_down_q is None:
                raise RuntimeError(
                    "Clarification queues not supplied – cannot request clarification in this context.",
                )
            await clarification_up_q.put(question)
            return await clarification_down_q.get()

        return _request

    # ────────────────────────────────────────────────────────────────────
    # Small DRY helpers used by ask/update flows
    # ────────────────────────────────────────────────────────────────────

    def _start_loop(
        self,
        client: "unify.AsyncUnify",
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
    ) -> SteerableToolHandle:
        """Centralised wrapper around start_async_tool_use_loop."""
        return start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=loop_id,
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            log_steps=log_steps,
            preprocess_msgs=inject_broader_context,
            tool_policy=tool_policy,
        )

    def _maybe_add_clarification_tool(
        self,
        tools: ToolsDict,
        clarification_up_q: Optional[asyncio.Queue[str]],
        clarification_down_q: Optional[asyncio.Queue[str]],
    ) -> None:
        """Insert `request_clarification` only when both queues are provided."""
        if clarification_up_q is not None and clarification_down_q is not None:
            tools["request_clarification"] = self._make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
            )

    def _wrap_result_with_messages(
        self,
        handle: SteerableToolHandle,
        client: "unify.AsyncUnify",
    ) -> SteerableToolHandle:
        """Wrap handle.result() so it returns (answer, client.messages)."""
        original_result = handle.result

        async def wrapped_result():
            answer = await original_result()
            return answer, client.messages

        handle.result = wrapped_result  # type: ignore[assignment]
        return handle

    def _get_single_row_or_raise(self, task_id: int) -> TaskRow:
        """Fetch exactly one task row by id or raise ValueError."""
        rows = self._filter_tasks(filter=f"task_id == {task_id}", limit=1)
        if not rows:
            raise ValueError(f"No task found with id={task_id}")
        return rows[0]

    def _get_single_log_obj_or_raise(self, task_id: int) -> "unify.Log":
        """Fetch the unique unify.Log object for task_id or raise."""
        logs = self._get_logs_by_task_ids(task_ids=task_id, return_ids_only=False)
        assert len(logs) == 1, "Task IDs should be unique"
        return logs[0]  # type: ignore[return-value]

    # Reinstate a previously isolated-and-activated task back to its prior queue position

    def _reinstate_task_to_previous_queue(
        self,
        *,
        task_id: int,
        _allow_active: bool = False,
    ) -> ToolOutcome:
        # Delegate to manager; keep signature for backwards-compat with existing callers/tests
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
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
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
                r.get("status") == str(Status.active)
                for r in self._filter_tasks(filter="status == 'active'", limit=1)
            )
        except Exception:
            any_active = False
        if any_active:
            raise RuntimeError(
                "A task is marked as active, but no active handle is present – reconcile state before starting another task.",
            )

        return await self._execute_queue_internal(
            task_id=task_id,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    # (Removed) heuristic fallback reinstatement – rely on stored plan only

    # Search Across Tasks

    @_ts_log_tool_runtime
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
            Maximum number of results to return.

        Returns
        -------
        list[Task]
            Up to ``k`` matching tasks sorted by ascending cosine distance. If the
            similarity search yields fewer than ``k`` results and there are more
            than ``k`` tasks overall, the remainder is backfilled from
            ``unify.get_logs(limit=k)`` in returned order, skipping duplicates.
        """
        # 1) Primary: semantic similarity results (ordered). When references is None/empty,
        # the shared helper returns an empty list, and backfill-only logic applies.
        rows = fetch_top_k_by_references(self._ctx, references, k=k)
        filled = backfill_rows(
            self._ctx,
            rows,
            k,
            unique_id_field="task_id",
        )
        # Defensive read: drop stale activation metadata on non-active rows to avoid
        # Pydantic validation errors if any legacy writes left it behind.
        sanitized: list[Task] = []
        for lg in filled:
            row = dict(lg)
            try:
                if self._to_status(row.get("status")) != Status.active:  # type: ignore[arg-type]
                    row.pop("activated_by", None)
            except Exception:
                if str(row.get("status")) != str(Status.active):
                    row.pop("activated_by", None)
            sanitized.append(Task(**row))
        return sanitized

    @_ts_log_tool_runtime
    def _filter_tasks(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[TaskRow]:
        """
        Run a **column-wise Python expression** (`filter`) against every task
        and return the matching rows.

        Do *not* use this tool when searching for a task with a similar name
        or description. Trying to get an exact match on substrings (especially
        with multiple words) is very brittle, and likely to return no matches.
        The `search_tasks` tool is *much* more robust and accurate in such cases.

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
        filter = self._normalize_filter_expr(filter)
        rows = self._store.get_entries(
            filter=filter,
            offset=offset,
            limit=limit,
            # Avoid an extra backend call here by deriving private fields from the
            # cached schema instead of calling get_fields() again.
            exclude_fields=[
                name
                for name in self._get_columns().keys()
                if isinstance(name, str) and name.startswith("_")
            ],
        )

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

        return rows

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

    # ────────────────────────────────────────────────────────────────────
    # Column and metrics helpers (paralleling Contact/TranscriptManager)
    # ────────────────────────────────────────────────────────────────────

    def _get_columns(self) -> Dict[str, str]:
        """
        Return {column_name: column_type} for the tasks table.
        """
        return self._store.fields

    @_ts_log_tool_runtime
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

    @_ts_log_tool_runtime
    def _num_tasks(self) -> int:
        """Return the total number of tasks in the Tasks context."""
        return self._store.get_metric_count(key="task_id")

    # (Removed) LLM-based scope classifier

    # ------------------------------------------------------------------ #
    #  Steering intent classification (LLM-routed)                        #
    # ------------------------------------------------------------------ #

    async def _classify_steering_intent(
        self,
        message: str,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[str, str]:
        """Classify steering into: cancel | defer | pause | resume | continue | none."""
        try:
            client = self._new_llm_client("gpt-5@openai")
            system = (
                "You are a router that classifies an in-flight steering message.\n"
                "Labels: cancel | defer | pause | resume | continue | none.\n"
                "Definitions:\n"
                "- cancel: abandon/kill/drop the task (terminal).\n"
                "- defer: stop for now but resume later / return to prior queue/schedule.\n"
                "- pause: temporarily pause, expecting explicit resume soon.\n"
                "- resume: continue after a pause.\n"
                "- continue: keep going (no change).\n"
                "- none: message is not a steering instruction.\n"
                'Output ONLY JSON with rationale first: {"rationale": <short string>, "action": <label>, "reason": <short substring or null>}'
            )
            client.set_system_message(system)

            # Build a compact, recent-first transcript for added context
            def _format_ctx(
                ctx: Optional[List[Dict[str, Any]]],
                limit_chars: int = 2000,
            ) -> str:
                try:
                    if not ctx:
                        return "(no prior context)"
                    lines: List[str] = []
                    total = 0
                    for msg in reversed(ctx[-20:]):
                        role = str(msg.get("role", "")).strip() or "user"
                        content = str(msg.get("content", "")).strip()
                        line = f"{role}: {content}"
                        if total + len(line) > limit_chars:
                            break
                        lines.append(line)
                        total += len(line)
                    return "\n".join(reversed(lines)) if lines else "(no prior context)"
                except Exception:
                    return "(no prior context)"

            ctx_block = _format_ctx(parent_chat_context)
            user = (
                "Recent conversation (most recent last):\n"
                f"{ctx_block}\n\n"
                "Steering message:\n"
                f"{(message or '').strip()}"
            )
            raw = await client.generate(user)
            import json as _json

            data = None
            try:
                data = _json.loads(raw)
                action = str(data.get("action", "none")).strip().lower()
                reason = data.get("reason")
                if action not in {
                    "cancel",
                    "defer",
                    "pause",
                    "resume",
                    "continue",
                    "none",
                }:
                    action = "none"
                if reason is not None:
                    reason = str(reason)
                else:
                    reason = message
                return action, cast(str, reason)
            except Exception:
                # fallback: pattern match a minimal token
                low = (raw or "").lower()
                for tok in ["cancel", "defer", "pause", "resume", "continue"]:
                    if tok in low:
                        return tok, message
                return "none", message
        except Exception:
            return "none", message
