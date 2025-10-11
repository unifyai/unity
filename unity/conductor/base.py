"""
*Public* contract for every **Conductor** implementation.

The top-level manager unifies four sub-domains

• tasks (TaskScheduler)  • contacts  • transcripts  • knowledge-base

and it exposes exactly **three** conversational entry-points:

1. `ask`        – read-only Q&A across all domains
2. `request`    – read-write mutations (plus everything in *ask*)
3. `execute` – specialised surface for activating a queued task
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..common.async_tool_loop import SteerableToolHandle
from ..singleton_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from .types import StateManager


class BaseConductor(ABC, metaclass=SingletonABCMeta):
    # ------------------------------------------------------------------ #
    #  ask – read-only                                                   #
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
        Answer a **read-only question** that may reference tasks, contacts,
        transcripts *or* stored knowledge.

        Do *not* request *how* the question should be answered; just ask the
        question in natural language and allow the `ask` method to determine
        the best method to answer it.

        Parameters
        ----------
        text : str
            The exact user question (natural language).
        _return_reasoning_steps : bool, default ``False``
            When *True*, the handle's ``.result()`` yields
            ``(assistant_answer, hidden_messages)`` instead of just the answer.
        _log_tool_steps : bool, default ``True``
            Emit server-side logs for each internal tool call (debugging aid).
        parent_chat_context : list[dict] | None
            Optional **read-only** context inherited from a parent conversation
            and made visible to the inner tool loop.
        clarification_up_q / clarification_down_q : asyncio.Queue[str] | None
            Two-way channels enabling interactive clarification questions:
            the LLM places a question on *up* and blocks waiting for the human
            answer on *down*.

        Returns
        -------
        SteerableToolHandle
            Await ``handle.result()`` for the final answer or steer execution
            mid-flight via ``pause()``, ``resume()``, ``interject()`` or
            ``stop()``.
        """

    # ------------------------------------------------------------------ #
    #  request – read **and** write                                      #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def request(
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
        Execute a **mutation request** – create / edit / delete tasks, contacts
        or knowledge – and return a steerable LLM handle.

        Do *not* request *how* the change should be implemented; describe the
        desired end-state in natural language and allow the `request` method to
        determine the best method and tools to apply it.

        All parameters & return value mirror :py:meth:`ask`.
        """

    # ------------------------------------------------------------------ #
    #  clear – irreversible state wipe for a selected manager            #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def clear(self, target: StateManager) -> None:
        """
        {base}

        Parameters
        ----------
        target : StateManager
            Which manager to clear. Options include: CONTACTS, TRANSCRIPTS, KNOWLEDGE,
            TASKS, WEB_SEARCH, and forward-compat entries FUNCTIONS, GUIDANCE, IMAGES, SECRETS.
        """


BaseConductor.clear.__doc__ = (BaseConductor.clear.__doc__ or "").format(
    base=CLEAR_METHOD_DOCSTRING,
)
