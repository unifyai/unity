"""
*Public* contract that every concrete **knowledge-manager** must satisfy.

A knowledge-manager has to:
• **store** arbitrary facts expressed in natural language, evolving
  table-based schemas as required;
• **retrieve** facts on demand, optionally refactoring the schema to make
  queries easier.

Both operations return a :class:`~unify.common.llm_helpers.SteerableToolHandle`
so a caller (or higher-level agent) can pause, resume, interject, or cancel the
LLM reasoning loop.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..common.llm_helpers import SteerableToolHandle
from ..singleton_registry import SingletonABCMeta


class BaseKnowledgeManager(ABC, metaclass=SingletonABCMeta):
    """
    *Public* contract that every concrete **knowledge-manager** must satisfy.

    A knowledge-manager handles
    1. `store` – English instructions that **add** or **update** knowledge, and
    2. `retrieve` – natural-language questions that **query** this knowledge.

    Implementations may talk to a real vector store, an HTTP API, Unify logs,
    an in-memory mock, or an entirely simulated LLM – but they **all** expose
    exactly the two public methods documented below.
    """

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:  # noqa: D401 – full docstring below
        """
        Persist *new* knowledge or amend existing records, expressed in
        **plain English**. The LLM will translate the request into a series
        of table/column manipulations and data-writes.

        Do *not* request *how* the storage should be performed; describe the
        knowledge in natural language and allow the `update` method to
        determine the best method to apply it.

        Parameters
        ----------
        text : str
            User's instruction, e.g. *"Add that Tesla's battery warranty is
            eight years."*
        _return_reasoning_steps : bool, default ``False``
            When *True* the handle's :pyfunc:`result` yields
            ``(assistant_reply, messages)`` where *messages* is the hidden
            chain-of-thought.
        parent_chat_context : list[dict] | None
            Read-only message history from outer tool loops.
        clarification_up_q / clarification_down_q : asyncio.Queue[str] | None
            Optional duplex channels enabling the LLM to ask the user follow-up
            questions.

        Returns
        -------
        SteerableToolHandle
            Live handle to the running reasoning loop.
        """

    @abstractmethod
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Answer a **natural-language query** by reading from the knowledge
        store (and optionally reshaping the schema to make retrieval easier).

        Do *not* request *how* the question should be answered; just ask the
        question in natural language and allow the `ask` method to determine
        the best method to answer it.

        Parameters
        ----------
        text : str
            The user's question, e.g. *"What warranty information do we hold
            about Tesla vehicles?"*
        _return_reasoning_steps, parent_chat_context,
        clarification_up_q, clarification_down_q
            See :py:meth:`store`.

        Returns
        -------
        SteerableToolHandle
            Handle that eventually yields the answer text (and optionally the
            hidden reasoning steps).
        """

    @abstractmethod
    async def refactor(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        **Restructure the schema** of *all* knowledge tables **and** the
        contacts table so that data are de-duplicated, normalised and stored
        as clearly and efficiently as possible.

        Do *not* request *how* to perform the refactor; state the high-level
        intent in natural language and allow the `refactor` method to
        determine the best method and specific operations.

        Parameters
        ----------
        text : str
            A high-level English instruction, e.g.
            *"Remove duplicated company names and introduce surrogate primary
            keys where appropriate."* – the low-level operations (column
            renames, deletions, moves, etc.) are carried out by the LLM via
            the exposed table/column-manipulation tools.
        _return_reasoning_steps, parent_chat_context,
        clarification_up_q, clarification_down_q
            Behaviour identical to :py:meth:`store`.

        Returns
        -------
        SteerableToolHandle
            Handle whose :pyfunc:`result` yields a natural-language summary of
            every structural change (and, optionally, the hidden chain-of-
            thought when *_return_reasoning_steps* is *True*).
        """
