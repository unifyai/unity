from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..common.llm_helpers import SteerableToolHandle


class BaseTaskScheduler(ABC):
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
        Interrogate the current **task list** in natural language and obtain
        a *live* :class:`~unify.common.llm_helpers.SteerableToolHandle`.

        Parameters
        ----------
        text : str
            The user's plain-English question, e.g. *"Which tasks are due
            tomorrow?"*.
        _return_reasoning_steps : bool, default ``False``
            When *True*, :pymeth:`SteerableToolHandle.result` returns
            ``(answer, messages)`` – the first element is the assistant's
            reply, the second the hidden chain-of-thought.
        _log_tool_steps : bool, default ``True``
            If *True* the task-scheduler logs every tool invocation to the
            server-side logger.  Mainly useful for debugging.
        parent_chat_context : list[dict] | None
            Optional **read-only** conversation context to prepend to the
            internal tool-use loop.
        clarification_up_q / clarification_down_q : asyncio.Queue[str] | None
            Duplex channels enabling interactive *clarification* questions.
            If supplied the LLM may push a follow-up question onto
            *clarification_up_q* and must read the human's answer from
            *clarification_down_q*.

        Returns
        -------
        SteerableToolHandle
            Await :pymeth:`SteerableToolHandle.result` for the final answer or
            steer the interaction via ``pause()``, ``resume()``,
            ``interject()`` or ``stop()``.
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

        Please always be explicit about the *ordering* of tasks.
        If the order *doesn't* matter please say so explicitly.
        If the order *does* matter, and the tasks are given in the correct number order,
        please also say so. You must always be explicit.

        Please also always be explicit about whether a task is *due* by a certain `deadline`,
        or whether the task should `start_at` a certain date and time.
        These both represent different things.
        Tasks can have one, both, or neither of these specified.

        For tasks where the time duration is short and predictable (such as sending an email)
        then the scheduled `start_at` and the `deadline` should usually be set to the same datetime.
        Either way, very explicit instructions regarding `start_at` and `deadline` must be given.

        All parameters mirror :pymeth:`ask`; refer there for detailed
        semantics.
        """

    @abstractmethod
    async def execute_task(
        self,
        text: str,
        *,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Start a **task** given a *free-form* textual instruction (*text*).

        The assistant should interpret *text* to figure out which task the user
        wants to run.  Typical workflow:

        1. Call :py:meth:`TaskScheduler.ask` to identify the `task_id` (if the
           id is not explicitly mentioned in *text*).
        2. Internally execute the task – the implementation SHOULD expose a
           private ``_execute_task_by_id`` helper that returns a
           :class:`~unify.common.llm_helpers.SteerableToolHandle` **and marks it
           for pass-through** so that the outer handle is upgraded transparently
           once the real execution begins.

        Implementations MUST return a *live* steerable handle whose public
        methods (pause, resume, interject, stop, result, …) continue to work
        after the adoption.

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
