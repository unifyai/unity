from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable

from ..common.llm_helpers import SteerableToolHandle
from ..singleton_registry import SingletonABCMeta


class BaseActiveTask(SteerableToolHandle, ABC):
    """
    Abstract contract that every concrete active activity must satisfy.

    An active activity represents a long‑running operation that can be steered
    at runtime (pause / resume / interject / ask / stop) and that ultimately
    resolves to a single result string.

    Sub‑classes must provide concrete implementations of all abstract members
    below and expose them via ``valid_tools`` so that higher‑level agents (or
    the UI) can discover the currently available controls.
    """

    # Public API
    @abstractmethod
    async def ask(
        self,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
    ) -> SteerableToolHandle:
        """Ask a read-only question about the live activity and return a handle.

        Implementations should return a lightweight handle whose ``result()``
        yields the answer string (and may optionally include reasoning when
        ``_return_reasoning_steps`` is True)."""

    @property
    @abstractmethod
    def valid_tools(self) -> Dict[str, Callable]:
        """
        Map of public‑name → callable for the user‑accessible controls that are
        currently valid in the activity's lifecycle state.
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


class BaseTaskScheduler(ABC, metaclass=SingletonABCMeta):
    """
    *Public* contract that every concrete **task-list-manager** must satisfy.

    Managers expose exactly **two** user-facing methods:

    • `ask`    – answer questions about the current task list
    • `update` – create, modify or delete tasks and queues

    Implementations may use Unify logs, a local DB, a remote API or even a
    purely simulated LLM – but they all obey the signatures & docstrings below.
    """

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
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
        _return_reasoning_steps : bool, default ``False``
            When *True*, :pymeth:`SteerableToolHandle.result` returns
            ``(answer, messages)`` – the first element is the assistant's
            reply, the second the hidden queue‑of‑thought.
        _log_tool_steps : bool, default ``True``
            If *True* the task‑scheduler logs every tool invocation to the
            server‑side logger.
        parent_chat_context : list[dict] | None
            Optional read‑only conversation context to prepend to the internal
            tool‑use loop.
        clarification_up_q / clarification_down_q : asyncio.Queue[str] | None
            Duplex channels enabling interactive clarification questions. If
            supplied the LLM may push a follow‑up question onto
            *clarification_up_q* and must read the human's answer from
            *clarification_down_q*.

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
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Apply a **mutation** request – create, edit, delete or reorder tasks –
        expressed in plain English and receive a steerable LLM handle.

        Do *not* request *how* the change should be implemented; just
        describe the desired end-state in natural language and allow the
        `update` method to determine the best method to apply it.

        Important execution boundary
        ----------------------------
        This method is not intended to be used by `execute` to rewrite
        schedules/ordering/start_at purely to begin execution. The `execute`
        flow determines the correct execution scope (isolate vs queue) and
        starts the task via an execution tool without mutating scheduling.
        Only use `update` within `execute` when the user explicitly asked to
        create a missing task or to change task fields before running.

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
        text: str,
        *,
        isolated: Optional[bool] = None,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Start a **task** given a *free-form* textual instruction (*text*).

        Do *not* request *how* the task should be executed; state what you
        want to run in natural language and allow the `execute` method to
        determine the best method and steps.

        The assistant should interpret *text* to figure out which task the user
        wants to run.  Typical workflow:

        1. Call :py:meth:`TaskScheduler.ask` to identify the `task_id` (if the
           id is not explicitly mentioned in *text*).
        2. Internally execute the task – the implementation SHOULD expose a
           private ``_execute_by_id`` helper that returns a
           :class:`SteerableToolHandle` **and marks it
           for pass-through** so that the outer handle is upgraded transparently
           once the real execution begins.

        Execution scope via explicit reordering
        --------------------------------------
        The execute flow no longer uses implicit routing modes. Instead, it
        exposes queue inspection and an explicit reorder primitive so the agent
        can express desired behaviour directly:
        - Use `get_task_queue()` to read the current order.
        - Use `update_task_queue(original=[...], new=[...])` to place the desired
          subset/order at the head (e.g., "just X", "first two now", or "all").
        - Then call `execute_by_id(task_id=<head>)` to start.

        Never rewrite a task's `start_at` purely to begin execution, and do not
        write lifecycle `status` fields directly.
        - `ask` – to discover the relevant task and queue context
        - `get_task_queue` / `update_task_queue` – to explicitly shape the queue
        - `execute_by_id(task_id)` – to start the head after any necessary reordering

        Implementations MUST return a *live* steerable handle whose public
        methods (pause, resume, interject, stop, result, …) continue to work
        after the adoption.

        Parameters
        ----------
        isolated : bool | None, default ``None``
            Optional override for routing semantics when a numeric task id is
            supplied directly. When ``True``, the scheduler should execute the
            task in isolation (detached from the queue). When ``False``, the
            scheduler should use queue/chained semantics. When ``None``, routing
            is left to the implementation (e.g., LLM decides for free‑form text;
            numeric fast‑path may default to chained semantics).

        parent_chat_context, clarification_up_q, clarification_down_q
            Same purpose and semantics as in :pymeth:`ask`.

        Returns
        -------
        SteerableToolHandle
            Handle that ultimately yields the *task-specific* assistant
            dialogue.

        Raises
        ------
        RuntimeError
            If another task is already active.
        ValueError
            When no matching task could be identified.
        """

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
        Restore a task to its previous queue/schedule position after a defer‑stop.

        Parameters
        ----------
        task_id : int
            Identifier of the task to reinstate.
        allow_active : bool, default ``False``
            When ``True``, permit reinstatement while a task is currently active.

        Notes
        -----
        This public method is the canonical entrypoint. Concrete schedulers may
        delegate to an internal implementation. Backwards‑compatibility helpers
        should accept additional kwargs and ignore unknown ones.
        """
        raise NotImplementedError
