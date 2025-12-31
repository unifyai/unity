"""
Abstract base contracts for steerable tasks and the task scheduler.

This module defines two abstract interfaces:
- BaseActiveTask: a live, steerable task handle (pause, resume, interject, ask, stop)
- BaseTaskScheduler: the public surface for reading, mutating, and executing task lists

Implementations provide storage, I/O, and execution details. This module
specifies current behavior and method signatures only.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
import json

from pydantic import BaseModel
from ..common.async_tool_loop import SteerableToolHandle
from ..manager_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager


class BaseActiveTask(SteerableToolHandle, ABC):
    """
    Abstract interface for a live, steerable task.

    The activity can be paused, resumed, interjected, queried (ask), or
    stopped, and ultimately resolves to a single result string.
    """

    @abstractmethod
    def stop(
        self,
        *,
        cancel: bool,
        reason: Optional[str] = None,
    ) -> Optional[str]:
        """Stop the live activity with explicit intent.

        Parameters
        ----------
        cancel : bool
            When True, abandon the task (mark as cancelled). When False, defer and
            reinstate it back into its prior queue/schedule position where possible.
        reason : str | None
            Optional human‑readable reason for logging/auditing.
        """


class BaseTaskScheduler(BaseStateManager, metaclass=SingletonABCMeta):
    """
    Public contract for a task‑list manager.

    Managers expose three user‑facing methods:
    • `ask` – answer questions about the current task list (read‑only)
    • `update` – create, modify, delete, or reorder tasks and queues
    • `execute` – start task execution and return a live steerable handle

    Implementations choose their storage and execution strategy; this base
    class defines the required behavior and method signatures.

    Intended use
    ------------
    The TaskScheduler is responsible for activities that should be represented
    as first‑class Tasks – with names, descriptions, scheduling fields and a
    completion status – and for returning a live, steerable execution handle
    when starting such tasks.

    Scope and positioning (LLM‑facing)
    ----------------------------------
    Use this interface for activities that should be represented as durable
    Tasks with names, descriptions, scheduling fields and completion status.
    It returns a steerable execution handle when starting such tasks.
    """

    _as_caller_description: str = (
        "the TaskScheduler, executing a scheduled task on behalf of the end user"
    )

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Interrogate the **existing task list** (read‑only) and obtain a live
        :class:`SteerableToolHandle`.

        Purpose
        -------
        Use this method to locate and inspect tasks that already exist in the
        table: find task ids, check statuses, queue positions, schedules,
        deadlines, triggers, or summarise/compare existing entries. This call
        must never create, modify, delete or reorder tasks.

        Clarifications
        --------------
        Do not use this method to ask the human questions. If the caller needs
        clarification about a prospective/new task (e.g., start time, timezone,
        naming, scope), route the question via a dedicated
        ``request_clarification`` tool when available. If no clarification
        channel exists, proceed with sensible defaults/best‑guess values and
        state those assumptions in the outer loop's final reply.

        Do *not* request *how* the question should be answered; just ask the
        question in natural language and allow this `ask` method to determine
        the best method to answer it.

        Visual inputs policy
        --------------------
        • When relevant images are available, pass them via the ``images`` argument.
        • When delegating to another tool that declares an ``images`` parameter, forward the
          relevant images and rewrite/augment their annotations so they align with the delegated
          question or action (not the original user phrasing). Prefer AnnotatedImageRefs; preserve
          user‑referenced ordering when it matters.

        Task schema (reference)
        -----------------------
        {task_schema}

        Task fields – quick reference
        -----------------------------
        {task_fields_quickref}

        Examples
        --------
        • Good: "Which task covers the onboarding plan?" → identify the
          task_id so an update tool can be applied next.
        • Bad:  "What start time should I use for the task I am about to
          create?" → this is a human clarification; use
          ``request_clarification`` instead.

        Parameters
        ----------
        text : str
            Plain‑English question about existing tasks, e.g. "Which tasks are
            due tomorrow?".
        response_format : Type[BaseModel] | None, default ``None``
            Optional Pydantic model to request a structured answer. When provided,
            the final result should conform to this schema; otherwise a plain
            string answer is returned.
        _return_reasoning_steps : bool, default ``False``
            When *True*, :pymeth:`SteerableToolHandle.result` returns
            ``(answer, messages)`` – the first element is the assistant's
            reply, the second the hidden queue‑of‑thought.
        _log_tool_steps : bool, default ``True``
            If *True* the task‑scheduler logs every tool invocation to the
            server‑side logger.
        _parent_chat_context : list[dict] | None
            Optional read‑only conversation context to prepend to the internal
            tool‑use loop.
        _clarification_up_q / _clarification_down_q : asyncio.Queue[str] | None
            Duplex channels enabling interactive clarification questions. If
            supplied the LLM may push a follow‑up question onto
            *_clarification_up_q* and must read the human's answer from
            *_clarification_down_q*.

        Returns
        -------
        SteerableToolHandle
            Await :pymeth:`SteerableToolHandle.result` for the final answer or
            steer the interaction via ``pause()``, ``resume()``, ``interject()``
            or ``stop()``.
        """

    @abstractmethod
    async def update(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Apply a **mutation** request – create, edit, delete or reorder tasks –
        expressed in plain English and receive a steerable LLM handle.

        Do *not* request *how* the change should be implemented; just
        describe the desired end-state in natural language and allow the
        `update` method to determine the best method to apply it.

        Task schema (reference)
        -----------------------
        {task_schema}

        Task fields – quick reference
        -----------------------------
        {task_fields_quickref}

        Important execution boundary
        ----------------------------
        This method is not intended to be used by `execute` to rewrite
        schedules/ordering/start_at purely to begin execution. The `execute`
        flow determines the correct execution scope (isolate vs queue) and
        starts the task via an execution tool without mutating scheduling.
        Only use `update` within `execute` when the user explicitly asked to
        create a missing task or to change task fields before running.

        This method is not intended to be used to materialize transient
        conversational sessions. It should be used to create or modify durable
        Tasks and their scheduling/ordering.

        Visual inputs policy
        --------------------
        • When relevant images are available, pass them via the ``images`` argument.
        • When delegating to another tool that declares an ``images`` parameter (e.g., a read‑only ask),
          forward the relevant images and rewrite/augment their annotations so they align with the delegated
          request. Prefer AnnotatedImageRefs; preserve user‑referenced ordering when it matters.

        Natural-language ordering semantics
        ----------------------------------
        When the user expresses relative ordering constraints such as “A after B” or “A before B”,
        interpret this as a constraint on the runnable queue ordering between those tasks.

        Unless the user explicitly indicates that intermediate tasks are acceptable (e.g., “sometime
        after”, “later”, “not necessarily immediately”), treat “after/before” as an **adjacency**
        constraint: A should be placed immediately after/before B in the runnable queue.

        When enforcing such constraints, prefer minimal change: keep the relative order of other tasks
        stable unless moving them is required to satisfy the user’s expressed constraints.

        Please always be explicit about the *ordering* of tasks.
        If the order *doesn't* matter please say so explicitly.
        If the order *does* matter, and the tasks are given in the correct number order,
        please also say so. You must always be explicit.

        Please also always be explicit about whether a task is *due* by a certain `deadline`,
        or whether the task should `start_at` a certain date and time.
        These both represent different things.
        Tasks can have one, both, or neither of these specified.

        For tasks where the time duration is short and predictable (such as sending an email)
        then it's best to only set the `start_at` and omit the `deadline`.
        Either way, very explicit instructions regarding `start_at` and `deadline` must always be given.

        If the task is to be started *immediately*, then just put the current datetime as the `start_at`,
        and omit the deadline if one is not specified.

        All parameters mirror :pymeth:`ask`; refer there for detailed
        semantics.
        """

    @abstractmethod
    async def execute(
        self,
        task_id: int,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        isolated: Optional[bool] = None,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Start a runnable task by its identifier and return a live steerable handle.

        Implementations MUST return a *live* steerable handle whose public
        methods (pause, resume, interject, stop, result, …) continue to work
        throughout execution.

        Parameters
        ----------
        task_id : int
            Identifier of the task to start. Must reference a single, non‑terminal,
            non‑active instance.
        response_format : Type[BaseModel] | None, default ``None``
            Optional Pydantic model to request a structured result. When provided,
            the final result should conform to this schema.
        isolated : bool | None, default ``None``
            When ``True``, execute the task in isolation (detach from any queue).
            When ``False`` or ``None``, preserve queue/chained semantics.

        _parent_chat_context, _clarification_up_q, _clarification_down_q
            Optional execution context and clarification channels propagated to the
            underlying actor/plan.

        Execution delegation
        --------------------
        When a run-scoped execution environment is available, task execution may be
        delegated to that environment to maintain context continuity. Otherwise,
        execution proceeds through the scheduler's configured execution strategy.
        In both cases, a live steerable handle is returned that supports the full
        steering interface.

        Returns
        -------
        SteerableToolHandle
            Handle for the running task or queue head that supports pause, resume,
            interject, stop and result().

        Implementation note
        -------------------
        Only one task may be active at a time. Attempting to start another task
        while one is already running will raise ``RuntimeError``.

        Raises
        ------
        RuntimeError
            If another task is already active.
        ValueError
            When ``task_id`` cannot be found or is not runnable.
        """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Attach centralised docstring
BaseTaskScheduler.clear.__doc__ = CLEAR_METHOD_DOCSTRING

# Inject live Task schema into docstrings at import time
try:
    from .types.task import TaskBase as _DocTask

    _TASK_SCHEMA_JSON = json.dumps(_DocTask.model_json_schema(), indent=4)
    try:
        _schema = _DocTask.model_json_schema()
        _props = dict(_schema.get("properties", {}))
        _required = set(_schema.get("required", []))
        _lines: list[str] = []
        for _name, _meta in _props.items():
            _desc = _meta.get("description") or ""
            _opt = "" if _name in _required else " (optional)"
            _lines.append(f"• {_name}{_opt}: {_desc}")
        _TASK_FIELDS_QUICKREF = "\n".join(_lines)
    except Exception:
        _TASK_FIELDS_QUICKREF = ""
    BaseTaskScheduler.ask.__doc__ = (
        (BaseTaskScheduler.ask.__doc__ or "")
        .replace(
            "{task_schema}",
            _TASK_SCHEMA_JSON,
        )
        .replace("{task_fields_quickref}", _TASK_FIELDS_QUICKREF)
    )
    BaseTaskScheduler.update.__doc__ = (
        (BaseTaskScheduler.update.__doc__ or "")
        .replace(
            "{task_schema}",
            _TASK_SCHEMA_JSON,
        )
        .replace("{task_fields_quickref}", _TASK_FIELDS_QUICKREF)
    )
except Exception:
    # Best-effort doc enrichment only; leave docstrings unchanged on failure
    pass
