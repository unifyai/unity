from __future__ import annotations

import os
import unify
import asyncio
import functools
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from typing import Literal


from ..common.embed_utils import list_private_fields
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
from .types.activated_by import ActivatedBy

# Contact manager import (lazy at module level to avoid cycles in other modules)
from ..contact_manager.contact_manager import ContactManager
from ..common.model_to_fields import model_to_fields
from .prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
    build_execute_task_prompt,
)
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
from ..common.semantic_search import fetch_top_k_by_references, backfill_rows
import requests
from ..helpers import _handle_exceptions
from ._queue_utils import (
    sched_prev as _q_prev,
    sched_next as _q_next,
    sync_adjacent_links as _q_sync_adjacent_links,
    attach_with_links as _q_attach_with_links,
)


# ------------------------------------------------------------------ #
#  Typed reintegration plan                                          #
# ------------------------------------------------------------------ #
from .types.reintegration_plan import ReintegrationPlan


class TaskScheduler(BaseTaskScheduler):

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
        planner: Optional[BasePlanner] = None,
        rolling_summary_in_prompts: bool = True,
    ) -> None:
        """
        Create a scheduler responsible for creating, searching, updating and executing tasks in the current Unify context.

        Parameters
        ----------
        planner : BasePlanner | None, default ``None``
            Planner used to execute the steps of an active task. When ``None``, a
            ``SimulatedPlanner(timeout=20)`` is used.
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
            **methods_to_tool_dict(
                # Ask
                self.ask,
                # Creation / deletion / cancellation
                self._create_task,
                self._delete_task,
                self._cancel_tasks,
                # Queue manipulation
                self._update_task_queue,
                # Reintegration
                self._reinstate_task_to_previous_queue,
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

        if self._ctx not in unify.get_contexts():
            unify.create_context(
                self._ctx,
                unique_keys={"task_id": "counting", "instance_id": "counting"},
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
        else:
            # Ensure any new fields added to the Task model are present in the context
            try:
                existing_fields = unify.get_fields(context=self._ctx) or {}
            except Exception:
                existing_fields = {}
            model_fields = model_to_fields(Task)
            missing = {
                k: v for k, v in model_fields.items() if k not in existing_fields
            }
            if missing:
                unify.create_fields(
                    missing,
                    context=self._ctx,
                )

        # ID of the *single* task that is allowed to be in the **active**
        # state at any moment.  This will be maintained by a forthcoming
        # tool; until then it may legitimately stay as ``None``.
        # {'task_id': int, 'instance_id': int, 'handle': ActiveTask}
        self._active_task: Optional[Dict[str, Any]] = None
        primed_tasks = self._filter_tasks(filter="status == 'primed'")
        if primed_tasks:
            assert (
                len(primed_tasks) == 1
            ), f"More than one primed task found:\n{primed_tasks}"
            self._primed_task: Optional[Dict[str, Any]] = primed_tasks[0]
        else:
            self._primed_task: Optional[Dict[str, Any]] = None

        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # Stores a minimal corrective plan that allows us to restore the
        # previously queued/scheduled position of the most recently activated
        # task (when it was started in "isolate" mode).  This enables a
        # surgical reintegration after a user cancels the task and asks to
        # "do it as originally scheduled" without reverting unrelated changes.
        #
        # Shape:
        # @dataclass ReintegrationPlan
        #   task_id: int
        #   instance_id: int | None
        #   prev_task: int | None
        #   next_task: int | None
        #   start_at: str | None
        #   was_head: bool
        #   original_status: str | None
        self._reintegration_plan: Optional["ReintegrationPlan"] = None

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
        """
        Answer a free‑form question about tasks using the available read‑only tools.

        Parameters
        ----------
        text : str
            Natural language question.
        _return_reasoning_steps : bool, default ``False``
            When ``True``, ``handle.result()`` returns a tuple ``(answer, messages)`` where
            ``messages`` contains the full LLM trace.
        _log_tool_steps : bool, default ``True``
            Whether to include intermediate tool messages in the chat context shown to the caller.
        parent_chat_context : list[dict] | None, default ``None``
            Prior messages to seed the conversation.
        clarification_up_q : asyncio.Queue[str] | None, default ``None``
            Queue used to bubble clarification questions to the caller. Must be provided
            together with ``clarification_down_q`` for interactive sessions.
        clarification_down_q : asyncio.Queue[str] | None, default ``None``
            Queue on which answers to clarification questions are received.
        rolling_summary_in_prompts : bool | None, default ``None``
            Override for this call only; when ``None`` the instance default is used.
        tool_policy : {"default" | callable | None}, default "default"
            Policy controlling when tools are offered to the model. ``"default"`` requires
            a first tool step then switches to auto; a callable receives ``(step_idx, tools)``
            and must return a tuple ``(mode, tools)``.

        Returns
        -------
        SteerableToolHandle
            A live handle representing the interactive tool‑use session.
        """
        client = self._new_llm_client("o4-mini@openai")

        # ── 0.  Build a *live* tools-dict so the prompt reflects reality ───
        tools = dict(self._ask_tools)

        if clarification_up_q is not None or clarification_down_q is not None:
            tools["request_clarification"] = self._make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
            )

        # ── 1.  Inject the dynamic system-prompt ───────────────────────────
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

        # Prepare effective tool_policy – hard-code first step to use search_tasks
        if tool_policy == "default":
            effective_tool_policy = lambda i, current_tools: (
                (
                    "required",
                    {"search_tasks": current_tools["search_tasks"]},
                )
                if i < 1 and "search_tasks" in current_tools
                else ("auto", current_tools)
            )
        else:
            # pass through callable or None
            effective_tool_policy = tool_policy

        # ── 2.  Kick off the tool-use loop ────────────────────────────────
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_chat_context=parent_chat_context,
            log_steps=_log_tool_steps,
            preprocess_msgs=self._inject_broader_context,
            tool_policy=effective_tool_policy,
        )
        # Logging wrapper applied by decorator

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
        """
        Apply updates to tasks from a natural‑language request using the scheduler's write tools.

        Parameters
        ----------
        text : str
            Free‑form update request.
        _return_reasoning_steps : bool, default ``False``
            When ``True``, ``handle.result()`` returns a tuple ``(answer, messages)`` where
            ``messages`` contains the full LLM trace.
        _log_tool_steps : bool, default ``True``
            Whether to include intermediate tool messages in the chat context shown to the caller.
        parent_chat_context : list[dict] | None, default ``None``
            Prior messages to seed the conversation.
        clarification_up_q : asyncio.Queue[str] | None, default ``None``
            Queue used to bubble clarification questions to the caller. Must be provided
            together with ``clarification_down_q`` for interactive sessions.
        clarification_down_q : asyncio.Queue[str] | None, default ``None``
            Queue on which answers to clarification questions are received.
        rolling_summary_in_prompts : bool | None, default ``None``
            Override for this call only; when ``None`` the instance default is used.
        tool_policy : {"default" | callable | None}, default "default"
            Policy controlling when tools are offered to the model. ``"default"`` requires
            an initial ask step before writes; a callable receives ``(step_idx, tools)`` and
            must return a tuple ``(mode, tools)``.

        Returns
        -------
        SteerableToolHandle
            A live handle representing the interactive tool‑use session.
        """
        client = self._new_llm_client("o4-mini@openai")

        # ── 0.  Build a *live* tools-dict first (prompt needs it) ─────────
        tools = dict(self._update_tools)

        if clarification_up_q is not None or clarification_down_q is not None:
            tools["request_clarification"] = self._make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
            )

        # ── 1.  Inject the dynamic system-prompt ──────────────────────────
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

            def _default_update_policy(i: int, _tools: Dict[str, Any]):
                if i < 1 and "ask" in _tools:
                    return ("required", {"ask": _tools["ask"]})
                return ("auto", _tools)

            effective_tool_policy = _default_update_policy
        else:
            # pass through callable or None
            effective_tool_policy = tool_policy

        # ── 2.  Kick off interactive loop ─────────────────────────────────
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_chat_context=parent_chat_context,
            log_steps=_log_tool_steps,
            preprocess_msgs=self._inject_broader_context,
            tool_policy=effective_tool_policy,
        )
        # Logging wrapper applied by decorator

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
    @_log_manager_call.__func__("execute_task", "request")  # type: ignore[attr-defined]
    async def execute_task(
        self,
        text: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        """
        Promote a runnable task to active execution from free‑form input.

        Parameters
        ----------
        text : str
            Free‑form text. If it is a bare integer (e.g. ``"12"``), it is treated
            as a direct ``task_id`` fast‑path. Otherwise an auxiliary loop is used to
            identify the task and start it.
        parent_chat_context : list[dict] | None, default ``None``
            Prior messages to seed the conversation used for the reasoning flow.
        clarification_up_q : asyncio.Queue[str] | None, default ``None``
            Queue used to bubble clarification questions to the caller. Must be provided
            together with ``clarification_down_q`` for interactive sessions.
        clarification_down_q : asyncio.Queue[str] | None, default ``None``
            Queue on which answers to clarification questions are received.

        Returns
        -------
        SteerableToolHandle
            The live handle for the active task plan. When the fast‑path is taken,
            the returned handle is the underlying plan's handle (pass‑through).

        Notes
        -----
        Falls back to a reasoning loop if the direct ``task_id`` is invalid or the
        task cannot be started immediately.
        """

        freeform_text: str = text

        # ── Fast-path: direct numeric task_id ───────────────────────────────
        # When the user input *is* a plain integer we can skip the full
        # tool-resolution loop, execute the task immediately and hand back
        # the live ActiveTask handle.  This guarantees that callers who know
        # the id (including the unit-tests) observe the task promotion and
        # instance cloning *synchronously* after awaiting this method.

        stripped = freeform_text.strip()
        if stripped.isdigit():
            try:
                # Decide execution scope (A: isolate | B: chain) for this explicit request
                scope = await self._decide_execution_scope(
                    request_text=freeform_text,
                    parent_chat_context=parent_chat_context,
                )
                direct_handle = await self._execute_task_internal(
                    task_id=int(stripped),
                    parent_chat_context=parent_chat_context,
                    clarification_up_q=clarification_up_q,
                    clarification_down_q=clarification_down_q,
                    activated_by=ActivatedBy.explicit,
                    execution_scope=scope,
                )

                return direct_handle

            except (ValueError, RuntimeError):
                # Fall back to the slower, reasoning-based path when the id is
                # unknown or the task cannot be started directly (e.g. already
                # active).  Let the LLM ask for clarification / create a task.
                pass  # ↴ continue with regular flow

        return self._start_execute_task_loop(
            freeform_text=freeform_text,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

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
        activated_by: Optional[ActivatedBy] = None,
        execution_scope: Literal["isolate", "chain"] = "isolate",
    ) -> SteerableToolHandle:
        """
        Start the execution of a runnable task by its identifier.

        Parameters
        ----------
        task_id : int
            Identifier of the task to start. Must resolve to a single, non‑terminal,
            non‑active instance.
        parent_chat_context : list[dict] | None, default ``None``
            Prior messages to seed the conversation used for the planner execution.
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

        # 0. sanity checks
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

        # 1. Adjust queue linkages based on explicit activation scope
        self._detach_from_queue_for_activation(
            task_id=task_id,
            execution_scope=execution_scope,
        )

        # 2. build the *real* active plan
        plan_handle = await self._planner.execute(
            task_row["description"],
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        # 3. wrap it so we can keep the task-table in sync
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
        if self._to_status(task_row["status"]) == Status.triggerable or task_row.get(
            "repeat",
        ):
            self._clone_task_instance(task_row)

        # 4. Promote status → active (and record activation reason) and clear primed pointer if needed

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
    #  Helper – build and start the execute_task outer tool-use loop      #
    # ------------------------------------------------------------------ #
    def _start_execute_task_loop(
        self,
        *,
        freeform_text: str,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """Compose tools and prompt, then start the execute_task reasoning loop.

        Mirrors the pre-refactor behaviour byte-for-byte; extracted for clarity.
        """
        client = self._new_llm_client("o4-mini@openai")

        # ── tool definitions ────────────────────────────────────────────────
        async def _execute_task_by_id(*, task_id: int) -> SteerableToolHandle:  # type: ignore[valid-type]
            """Start the task with *task_id* and bubble up its handle (passthrough)."""

            # Decide execution scope (A: isolate | B: chain) for this explicit request
            scope = await self._decide_execution_scope(
                request_text=freeform_text,
                parent_chat_context=parent_chat_context,
            )
            handle = await self._execute_task_internal(
                task_id=task_id,
                parent_chat_context=parent_chat_context,
                clarification_up_q=clarification_up_q,
                clarification_down_q=clarification_down_q,
                activated_by=ActivatedBy.explicit,
                execution_scope=scope,
            )
            # 💡 signal pass-through so the outer loop adopts this handle
            setattr(handle, "__passthrough__", True)
            return handle

        async def request_clarification(question: str) -> str:  # type: ignore[valid-type]
            """Bubble *question* up to the caller and await the answer."""
            rc = self._make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
            )
            return await rc(question)

        # Wrap update to hard-code tool_policy=None while preserving metadata
        @functools.wraps(self.update, updated=())
        async def _update_no_forcing(*args, **kwargs):  # type: ignore[valid-type]
            kwargs["tool_policy"] = None
            return await self.update(*args, **kwargs)

        tools = methods_to_tool_dict(
            self.ask,
            _update_no_forcing,
            request_clarification,
            _execute_task_by_id,
            include_class_name=False,
        )

        # ── dynamic system prompt ───────────────────────────────────────────
        client.set_system_message(
            build_execute_task_prompt(tools),
        )

        outer_handle = start_async_tool_use_loop(
            client,
            freeform_text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.execute_task.__name__}",
            parent_chat_context=parent_chat_context,
            log_steps=True,
            preprocess_msgs=self._inject_broader_context,
        )

        return outer_handle

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
        # Normalise status to enum for consistent comparisons
        new_status_enum = self._to_status(new_status)
        entries: Dict[str, Any] = {"status": new_status_enum}
        # Only allow `activated_by` to be set during transition to 'active'
        if new_status_enum == Status.active and activated_by is not None:
            # Set only at the moment of activation; never overwrite later
            entries["activated_by"] = str(activated_by)

        result = unify.update_logs(
            logs=log_objs[0].id if hasattr(log_objs[0], "id") else log_objs[0],
            context=self._ctx,
            entries=entries,
            overwrite=True,
        )
        # Auto-clear reintegration plan on completion/failed to avoid stale replay.
        # Intentionally keep the plan on 'cancelled' so callers can reinstate
        # a cancelled isolated activation back to its prior queue position.
        try:
            if (
                self._reintegration_plan is not None
                and self._reintegration_plan.task_id == task_id
                and self._reintegration_plan.instance_id == instance_id
                and new_status_enum in {Status.completed, Status.failed}
            ):
                self._reintegration_plan = None
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
        Validate invariants related to queue linkage and scheduling.

        Invariants (when the task is not trigger-based):
        - If a task is at the head of the queue (prev_task is None) and defines
          a start_at, then its status must be 'scheduled' (not 'queued').
        - A task must not define both prev_task and start_at simultaneously –
          the queue-level timestamp lives on the head node only.
        - A 'primed' task must be the queue head (prev_task is None).
        - When setting status to 'scheduled', the task must have either a
          prev_task (it sits in the chain) or a start_at timestamp.

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

        # 'scheduled' requires either a chain position or a start_at
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
            response_policy=response_policy,
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
        schedule: Optional[Union[Schedule, dict]],
    ) -> None:
        """Delegate to queue-utils to maintain symmetric neighbour links."""
        _q_sync_adjacent_links(self, task_id=task_id, schedule=schedule)

    # ------------------------------------------------------------------ #
    #  Centralised schedule/status write with invariant validation        #
    # ------------------------------------------------------------------ #

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
        """
        current = self._get_single_row_or_raise(task_id)

        prospective_schedule = entries.get("schedule", current.get("schedule"))
        prospective_status = entries.get("status", current.get("status"))
        prospective_trigger = entries.get("trigger", current.get("trigger"))

        self._validate_scheduled_invariants(
            status=prospective_status,
            schedule=prospective_schedule,
            trigger=prospective_trigger,
            err_prefix=err_prefix,
        )

        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        result = unify.update_logs(
            logs=log_id,
            context=self._ctx,
            entries=entries,
            overwrite=True,
        )

        # Ensure neighbour symmetry whenever schedule changed
        if "schedule" in entries:
            self._sync_adjacent_links(task_id=task_id, schedule=prospective_schedule)

        return result

    # ------------------------------------------------------------------ #
    #  Centralised helpers for queue link manipulation                    #
    # ------------------------------------------------------------------ #

    def _detach_from_queue_for_activation(
        self,
        *,
        task_id: int,
        execution_scope: Literal["isolate", "chain"],
    ) -> None:
        """
        Detach the task from the runnable queue in preparation for activation.

        In 'isolate' mode the task is fully detached and the former neighbour
        pointers are reconnected. In 'chain' mode the task becomes the head of
        its current chain, preserving start_at on the head only.
        """
        candidate_rows = self._filter_tasks(
            filter=(
                f"task_id == {task_id} and status not in "
                "('completed','cancelled','failed','active')"
            ),
        )
        if not candidate_rows:
            raise ValueError(f"No runnable task found with id={task_id}")
        task_row = sorted(
            candidate_rows,
            key=lambda r: r.get("instance_id", 0),
        )[0]

        sched = task_row.get("schedule") or {}
        prev_tid = self._sched_prev(sched)
        next_tid = self._sched_next(sched)
        start_at = sched.get("start_at") if isinstance(sched, dict) else None

        def _get_log_obj(tid: int) -> unify.Log:
            logs = self._get_logs_by_task_ids(task_ids=tid, return_ids_only=False)
            assert len(logs) == 1, "Task IDs should be unique"
            return logs[0]  # type: ignore[return-value]

        # Record reintegration plan only for isolate scope
        if execution_scope == "isolate":
            self._reintegration_plan = ReintegrationPlan(
                task_id=task_id,
                instance_id=task_row.get("instance_id"),
                prev_task=prev_tid,
                next_task=next_tid,
                start_at=start_at,
                was_head=prev_tid is None,
                original_status=task_row.get("status"),
            )

        # Disconnect previous neighbour's next pointer
        if prev_tid is not None:
            prev_log = _get_log_obj(prev_tid)
            prev_sched = {
                **((getattr(prev_log, "entries", {}) or {}).get("schedule") or {}),
            }
            if prev_sched.get("next_task") == task_id:
                prev_sched["next_task"] = None
                unify.update_logs(
                    logs=prev_log.id if hasattr(prev_log, "id") else prev_log,
                    context=self._ctx,
                    entries={"schedule": prev_sched},
                    overwrite=True,
                )

        # Apply branch-specific rewiring
        if sched is not None:
            if execution_scope == "isolate":
                # Reconnect next backwards and transfer start_at to new head if needed
                if next_tid is not None:
                    next_log = _get_log_obj(next_tid)
                    next_sched = {
                        **(
                            (getattr(next_log, "entries", {}) or {}).get("schedule")
                            or {}
                        ),
                    }
                    if prev_tid is None:
                        next_sched["prev_task"] = None
                        if start_at is not None:
                            next_sched["start_at"] = start_at
                    else:
                        next_sched["prev_task"] = prev_tid
                        next_sched.pop("start_at", None)
                        # also set prev.next_task → next
                        if prev_tid is not None:
                            prev_log = _get_log_obj(prev_tid)
                            prev_sched = {
                                **(
                                    (getattr(prev_log, "entries", {}) or {}).get(
                                        "schedule",
                                    )
                                    or {}
                                ),
                            }
                            prev_sched["next_task"] = next_tid
                            unify.update_logs(
                                logs=(
                                    prev_log.id if hasattr(prev_log, "id") else prev_log
                                ),
                                context=self._ctx,
                                entries={"schedule": prev_sched},
                                overwrite=True,
                            )
                    unify.update_logs(
                        logs=next_log.id if hasattr(next_log, "id") else next_log,
                        context=self._ctx,
                        entries={"schedule": next_sched},
                        overwrite=True,
                    )

                # Finally, detach current task completely
                cur_log = _get_log_obj(task_id)
                if (getattr(cur_log, "entries", {}) or {}).get("schedule") is not None:
                    unify.update_logs(
                        logs=cur_log.id if hasattr(cur_log, "id") else cur_log,
                        context=self._ctx,
                        entries={"schedule": None},
                        overwrite=True,
                    )
            else:
                # Keep chain behind current task
                cur_log = _get_log_obj(task_id)
                new_sched: Dict[str, Any] = {"prev_task": None, "next_task": next_tid}
                if start_at is not None:
                    new_sched["start_at"] = start_at
                unify.update_logs(
                    logs=cur_log.id if hasattr(cur_log, "id") else cur_log,
                    context=self._ctx,
                    entries={"schedule": new_sched},
                    overwrite=True,
                )

                if next_tid is not None:
                    next_log = _get_log_obj(next_tid)
                    next_sched = {
                        **(
                            (getattr(next_log, "entries", {}) or {}).get("schedule")
                            or {}
                        ),
                    }
                    next_sched["prev_task"] = task_id
                    next_sched.pop("start_at", None)
                    unify.update_logs(
                        logs=next_log.id if hasattr(next_log, "id") else next_log,
                        context=self._ctx,
                        entries={"schedule": next_sched},
                        overwrite=True,
                    )

    def _attach_with_links(
        self,
        *,
        task_id: int,
        prev_task: Optional[int],
        next_task: Optional[int],
        head_start_at: Optional[str],
        err_prefix: str,
    ) -> None:
        """Delegate to queue-utils to attach with symmetric neighbour updates."""
        _q_attach_with_links(
            self,
            task_id=task_id,
            prev_task=prev_task,
            next_task=next_task,
            head_start_at=head_start_at,
            err_prefix=err_prefix,
        )

    _TERMINAL_STATUSES = {"completed", "cancelled", "failed"}

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
        self._primed_task = rows[0] if rows else None

    def _get_task_queue(
        self,
        task_id: Optional[int] = None,
    ) -> List[Task]:
        """
        Return the runnable task queue from head to tail.

        Parameters
        ----------
        task_id : int | None, default ``None``
            Optional starting node. When omitted the queue head is derived
            (prefer primed task, else first runnable with no ``prev_task``).

        Returns
        -------
        list[Task]
            Ordered list of non‑terminal tasks from head to tail.

        Notes
        -----
        Only rows actually traversed are loaded; the full table is not materialised.
        """

        # ----------------  helpers  ---------------- #
        def _get_task_by_task_id(tid: int) -> Optional[dict]:
            """Fetch exactly one task row by id or return None."""
            rows = self._filter_tasks(filter=f"task_id == {tid}", limit=1)
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
                head_candidates = self._filter_tasks(
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
            head_candidates = self._filter_tasks(
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
            row = self._filter_tasks(filter=f"task_id == {tid}", limit=1)[0]
            if row.get("trigger") is not None:
                raise ValueError(
                    f"Task {tid} is trigger-based and cannot be placed in the queue.",
                )
        # Collect every task that already has a schedule entry – we need its
        # linkage pointers *and* any existing start_at value.
        existing_logs = {
            t["task_id"]: t
            for t in self._filter_tasks()
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
            existing_status = self._to_status(
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
        Set, replace or clear a task's trigger.

        Parameters
        ----------
        task_id : int
            Identifier of the task to update.
        new_trigger : Trigger | dict | None
            Replacement trigger or ``None`` to remove it.

        Returns
        -------
        ToolOutcome
            Outcome payload with the updated task id.

        Raises
        ------
        ValueError
            If a trigger is added while a schedule exists.
        """

        self._ensure_not_active_task(task_id)

        current_rows = self._filter_tasks(filter=f"task_id == {task_id}", limit=1)
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

        # Ensure JSON-serialisable payload (pydantic → dict)
        entries: Dict[str, Any] = {
            "trigger": new_trigger.model_dump() if new_trigger is not None else None,
        }

        # ── status transitions ───────────────────────────────────────────
        cur_status = self._to_status(current["status"])
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
        return self._update_fields_if_not_active(
            task_id=task_id,
            entries={"name": new_name},
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
        return self._update_fields_if_not_active(
            task_id=task_id,
            entries={"description": new_description},
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
        new_status_enum = self._to_status(new_status)
        if new_status_enum == Status.active and not allow_active:
            raise ValueError(
                "Direct status changes to 'active' are not allowed; "
                "use the dedicated activation tool.",
            )

        # 2. Forbid touching the existing active task
        if not allow_active:
            self._ensure_not_active_task(task_ids)

        # ── Invariant check *per task* if new_status becomes 'scheduled' ─────
        if new_status_enum == Status.scheduled:
            rows = self._filter_tasks(filter=f"task_id in {task_ids}")
            for row in rows:
                self._validate_scheduled_invariants(
                    status=new_status_enum,
                    schedule=row.get("schedule"),
                    err_prefix=f"While changing status of task {row['task_id']}:",
                )
        # ── Invariant check when transitioning to 'queued' ───────────────
        if new_status_enum == Status.queued:
            rows = self._filter_tasks(filter=f"task_id in {task_ids}")
            for row in rows:
                self._validate_scheduled_invariants(
                    status=new_status_enum,
                    schedule=row.get("schedule"),
                    err_prefix=f"While changing status of task {row['task_id']}:",
                )

        # ToDo: replace with single API call once this task [https://app.clickup.com/t/86c3c1y63] is done
        log_ids = self._get_logs_by_task_ids(task_ids=task_ids)
        return unify.update_logs(
            logs=log_ids,
            context=self._ctx,
            entries={"status": new_status_enum},
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
        # Coerce to ISO-8601 string (Unify stores plain serialisable values)
        if isinstance(new_start_at, datetime):
            new_start_at = new_start_at.isoformat()

        # Fetch current row (needed for invariants & trigger check)
        current_rows = self._filter_tasks(filter=f"task_id == {task_id}", limit=1)

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

        return self._validated_write(
            task_id=task_id,
            entries={"schedule": sched_payload},
            err_prefix=f"While updating start_at for task {task_id}:",
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
        return self._update_fields_if_not_active(
            task_id=task_id,
            entries={"deadline": new_deadline},
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
        return self._update_fields_if_not_active(
            task_id=task_id,
            entries={"repeat": [r.model_dump() for r in new_repeat]},
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
        return self._update_fields_if_not_active(
            task_id=task_id,
            entries={"priority": new_priority},
        )

    # ────────────────────────────────────────────────────────────────────
    # Small internal helpers (Step 1)
    # ────────────────────────────────────────────────────────────────────

    def _new_llm_client(self, model: str) -> "unify.AsyncUnify":
        """Construct a configured AsyncUnify client for the given model."""
        return unify.AsyncUnify(
            model,
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

    @staticmethod
    def _to_status(value: Union[Status, str]) -> Status:
        """Canonicalise a status-like value to the Status enum."""
        return value if isinstance(value, Status) else Status(value)

    # Centralised simple-field writer (Step 2)
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
        return unify.update_logs(
            logs=log_id,
            context=self._ctx,
            entries=entries,
            overwrite=True,
        )

    def _make_request_clarification_tool(
        self,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> Callable[[str], "asyncio.Future[str]"]:
        """Return an async tool that bubbles a question up and awaits the answer.

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

    def _get_single_row_or_raise(self, task_id: int) -> Dict[str, Any]:
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
        task_id: Optional[int] = None,
    ) -> ToolOutcome:
        """
        Restore the most recently activated task (started in "isolate" scope)
        back to the exact queue/schedule position it occupied before activation.

        This operation is surgical: it modifies only the affected task and its
        immediate neighbours (prev/next). It does not revert unrelated changes
        to other tasks that may have happened in the meantime.

        Parameters
        ----------
        task_id : int | None, default ``None``
            Optionally enforce that the stored reintegration plan matches this
            id. When ``None``, the stored plan is applied as-is.

        Returns
        -------
        ToolOutcome
            Outcome payload confirming the reinstatement.

        Raises
        ------
        ValueError
            If no reintegration plan is available or the provided task_id
            does not match the stored plan.
        """
        plan = getattr(self, "_reintegration_plan", None)
        if not plan:
            raise ValueError("No reintegration plan available.")
        if task_id is not None and plan.task_id != task_id:
            raise ValueError(
                f"Reintegration plan exists for task_id={plan.task_id}, not {task_id}",
            )

        tid = plan.task_id
        prev_tid = plan.prev_task
        next_tid = plan.next_task
        was_head = bool(plan.was_head)  # original position
        original_start_at = plan.start_at
        original_status = plan.original_status

        # Fetch current affected logs (ids and entries)
        def _get_log_obj(tid_int: int) -> Optional[unify.Log]:
            rows = self._get_logs_by_task_ids(task_ids=tid_int, return_ids_only=False)
            if not rows:
                return None
            assert len(rows) == 1, "Task IDs should be unique"
            return rows[0]  # type: ignore[return-value]

        cur_log = _get_log_obj(tid)
        if cur_log is None:
            raise ValueError(f"Task id {tid} no longer exists")
        # Load current row dict for checks
        cur_rows = self._filter_tasks(filter=f"task_id == {tid}", limit=1)
        cur_row = cur_rows[0] if cur_rows else {}

        # Guard: must not be active at this moment
        if self._to_status(cur_row.get("status")) == Status.active:
            raise RuntimeError("Cannot reinstate while the task is active.")

        # Guard: if a trigger exists now, schedule restoration would violate exclusivity
        if cur_row.get("trigger") is not None:
            raise ValueError(
                "Task currently has a trigger; remove the trigger before restoring its schedule/queue position.",
            )

        # Idempotence: if there is already a schedule present, treat as a no-op
        if cur_row.get("schedule") is not None:
            self._reintegration_plan = None
            return {
                "outcome": "no-op: task already has a schedule",
                "details": {"task_id": tid},
            }

        # Determine viable neighbours (drift-aware)
        def _is_viable(neighbour_tid: Optional[int]) -> bool:
            if neighbour_tid is None:
                return False
            rows = self._filter_tasks(filter=f"task_id == {neighbour_tid}", limit=1)
            if not rows:
                return False
            return rows[0].get("status") not in self._TERMINAL_STATUSES

        # Current queue view (head, tail) for fallbacks
        queue_list = self._get_task_queue()
        queue_ids = [t.task_id for t in queue_list]
        current_head_id = queue_ids[0] if queue_ids else None
        current_tail_id = queue_ids[-1] if queue_ids else None

        # Compute final neighbours with graceful fallback
        final_prev: Optional[int]
        final_next: Optional[int]
        if was_head:
            final_prev = None
            if _is_viable(next_tid):
                final_next = next_tid
            else:
                # Place as new head; chain to current head if present
                final_next = (
                    current_head_id
                    if (current_head_id is not None and current_head_id != tid)
                    else None
                )
        else:
            final_prev = (
                prev_tid
                if _is_viable(prev_tid)
                else (
                    current_tail_id
                    if (current_tail_id is not None and current_tail_id != tid)
                    else None
                )
            )
            if _is_viable(next_tid) and next_tid != final_prev:
                final_next = next_tid
            else:
                final_next = None

        if final_prev == tid:
            final_prev = None
        if final_next == tid:
            final_next = None
        if (
            final_prev is not None
            and final_next is not None
            and final_prev == final_next
        ):
            # Prefer keeping prev and dropping next to avoid self-loop via neighbour
            final_next = None

        # Build payload for current task
        cur_sched: Dict[str, Any] = {
            "prev_task": final_prev,
            "next_task": final_next,
        }
        if was_head and original_start_at is not None:
            cur_sched["start_at"] = original_start_at

        # Decide restored status (respect primed invariants)
        desired_status = (
            self._to_status(str(original_status))
            if original_status is not None
            else Status.queued
        )
        if (
            desired_status == Status.primed
            and self._primed_task is not None
            and self._primed_task.get("task_id") != tid
        ):
            # Another primed exists – do not violate single-primed invariant
            desired_status = Status.queued

        # Invariant check for schedule/state combo before writing
        self._validate_scheduled_invariants(
            status=desired_status,
            schedule=cur_sched,
            err_prefix=f"While reinstating task {tid}:",
        )

        # Use centralised attach helper which performs symmetric neighbour updates
        self._attach_with_links(
            task_id=tid,
            prev_task=final_prev,
            next_task=final_next,
            head_start_at=(original_start_at if was_head else None),
            err_prefix=f"While reinstating task {tid}:",
        )

        # Restore the prior non-active status of the task instance
        if desired_status != Status.active:
            # Use per-instance update to avoid touching other instances
            self._update_task_status_instance(
                task_id=tid,
                instance_id=plan.instance_id,
                new_status=str(desired_status),
            )

        # Maintain primed cache if needed
        if desired_status == Status.primed:
            self._refresh_primed_cache(tid)

        # Clear plan – it has been applied
        self._reintegration_plan = None

        return {
            "outcome": "task reinstated to previous queue position",
            "details": {"task_id": tid},
        }

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
        return [Task(**lg) for lg in filled]

    def _filter_tasks(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
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
        if isinstance(filter, str):
            filter = filter.replace(".start_at", "['start_at']")
        return [
            log.entries
            for log in unify.get_logs(
                context=self._ctx,
                filter=filter,
                offset=offset,
                limit=limit,
                exclude_fields=list_private_fields(self._ctx),
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

    # ────────────────────────────────────────────────────────────────────
    # Column and metrics helpers (paralleling Contact/TranscriptManager)
    # ────────────────────────────────────────────────────────────────────

    def _get_columns(self) -> Dict[str, str]:
        """
        Return {column_name: column_type} for the tasks table.
        """
        proj = unify.active_project()
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields?project={proj}&context={self._ctx}"
        headers = {"Authorization": f"Bearer {os.environ['UNIFY_KEY']}"}
        response = requests.request("GET", url, headers=headers)
        _handle_exceptions(response)
        ret = response.json()
        return {k: v["data_type"] for k, v in ret.items()}

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
        ret = unify.get_logs_metric(
            metric="count",
            key="task_id",
            context=self._ctx,
        )
        if ret is None:
            return 0
        return int(ret)

    # ------------------------------------------------------------------ #
    #  Helper – decide execution scope (A: isolate | B: chain)           #
    # ------------------------------------------------------------------ #

    async def _decide_execution_scope(
        self,
        *,
        request_text: str,
        parent_chat_context: Optional[List[Dict[str, Any]]],
    ) -> Literal["isolate", "chain"]:
        """
        Use a compact LLM classification to decide whether the user intends to
        run this task in isolation (A) or together with tasks queued behind it (B).

        If ambiguous, default to "isolate".
        """
        try:
            client = self._new_llm_client("o4-mini@openai")
            system = (
                "Classify the user's intent for execution scope. "
                "Reply with exactly one word: isolate | chain. "
                "Choose 'chain' only if the user clearly requests running this task and subsequent queued tasks now. "
                "If uncertain, reply 'isolate'."
            )
            client.set_system_message(system)
            parts: List[str] = [f"Request: {request_text}"]
            if parent_chat_context:
                try:
                    ctx_preview = json.dumps(
                        parent_chat_context[-6:],
                        ensure_ascii=False,
                    )
                    parts.append(f"Context: {ctx_preview}")
                except Exception:
                    pass
            ans = await client.generate("\n\n".join(parts))
            ans = (ans or "").strip().lower()
            if ans == "chain" or ("chain" in ans and "isolate" not in ans):
                return "chain"
            return "isolate"
        except Exception:
            return "isolate"
