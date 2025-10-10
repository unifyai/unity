"""
*Public* contract that every concrete **knowledge-manager** must satisfy.

Exposes exactly three user‑facing operations:

• **ask**      — answer questions about stored knowledge (read‑only)
• **update**   — create or change knowledge expressed in plain English
• **refactor** — restructure schemas so data are clear, normalised and efficient

All operations return a :class:`SteerableToolHandle` so a caller (or higher‑level
agent) can pause, resume, interject or stop the LLM reasoning loop.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..common.async_tool_loop import SteerableToolHandle
from ..singleton_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING


class BaseKnowledgeManager(ABC, metaclass=SingletonABCMeta):
    """
    *Public* contract that every concrete **knowledge-manager** must satisfy.

    A knowledge‑manager exposes:

    • `ask`      — answer questions about knowledge that already exists
      (read‑only)
    • `update`   — create, amend, delete or merge knowledge via
      natural‑language instructions
    • `refactor` — restructure the schema across knowledge tables (and related
      contact tables) to improve clarity, normalisation and efficiency

    Implementations may talk to a real vector store, an HTTP API, Unity logs,
    an in‑memory mock or a simulated LLM – but they **all** expose exactly the
    public methods documented below.
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
        Apply a **mutation** request – create, edit, delete or merge knowledge –
        expressed in plain English and receive a steerable LLM handle.

        Do *not* request *how* the change should be implemented; describe the
        desired end‑state in natural language and allow the `update` method to
        determine the best method to apply it (e.g. table/column operations and
        data writes).

        Ask vs Clarification
        --------------------
        • `ask` is ONLY for inspecting/locating knowledge that ALREADY EXISTS
          (e.g., to verify stored facts or locate relevant records).
        • Do NOT use `ask` to ask the human for details about NEW knowledge
          being created/changed in this update request; call
          ``request_clarification`` when a clarification channel is available.
        • When no clarification tool exists, proceed with sensible defaults or
          best‑guess values and state those assumptions in the final reply.

        Parameters
        ----------
        text : str
            The user's request (e.g. *"Add that Tesla's battery warranty is
            eight years."*).
        _return_reasoning_steps : bool, default ``False``
            When *True*, :pyfunc:`SteerableToolHandle.result` returns a tuple
            ``(answer, messages)`` where *messages* is the invisible
            chain‑of‑thought exchanged with the LLM.
        parent_chat_context : list[dict] | None
            **Read‑only** conversation context to prepend to the tool loop.
        clarification_up_q / clarification_down_q : asyncio.Queue[str] | None
            Optional duplex channels enabling interactive follow‑ups. If
            supplied, the LLM may push a question onto *clarification_up_q* and
            must read the human's answer from *clarification_down_q*.

        Returns
        -------
        SteerableToolHandle
            Handle whose :pyfunc:`result` yields confirmation of the mutation
            and (optionally) reasoning steps.
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
        Interrogate the **existing knowledge** (read‑only) and obtain a live
        :class:`SteerableToolHandle`.

        Purpose
        -------
        Use this method to locate and inspect knowledge that already exists in
        the store: retrieve or compare facts, perform semantic searches,
        aggregate/summarise existing entries, or identify structures relevant to
        a subsequent update. This call must never create, modify or delete
        knowledge or schema.

        Clarifications
        --------------
        Do not use this method to ask the human follow‑up questions. If the
        caller needs clarification about what to retrieve (e.g., which topic,
        which time window, which source), route the question via a dedicated
        ``request_clarification`` tool when available. If no clarification
        channel exists, proceed with sensible defaults/best‑guess values and
        state those assumptions in the outer loop's final reply.

        Do *not* request *how* the question should be answered; just ask the
        question in natural language and allow this `ask` method to determine
        the best method to answer it.

        Examples
        --------
        • Good: "What warranty information do we hold about Tesla vehicles?"
          → retrieve relevant facts and cite their locations when possible.
        • Bad:  "What should the category name be for the new policy I'm about
          to add?" → this is a human clarification; use
          ``request_clarification`` instead.

        Parameters
        ----------
        text : str
            The user's plain‑English question (e.g. *"List return policies for
            ACME by effective date."*).
        _return_reasoning_steps : bool, default ``False``
            When ``True`` the handle's :pyfunc:`~SteerableToolHandle.result`
            yields ``(answer, messages)`` – the first element is the
            assistant's reply, the second the hidden chain‑of‑thought (useful
            for debugging).
        parent_chat_context : list[dict] | None
            Optional read‑only chat history that will be provided to all nested
            tool calls.
        clarification_up_q / clarification_down_q : asyncio.Queue[str] | None
            Duplex channels enabling interactive clarification questions. If
            supplied the LLM may push a follow‑up question onto
            *clarification_up_q* and must read the human's answer from
            *clarification_down_q*.

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
        contacts table so that data are de‑duplicated, normalised and stored as
        clearly and efficiently as possible.

        Do *not* request *how* to perform the refactor; state the high‑level
        intent in natural language and allow the `refactor` method to determine
        the best method and specific operations (e.g., column renames, moves,
        deletions, key introduction).

        Parameters
        ----------
        text : str
            A high‑level English instruction, e.g. *"Remove duplicated company
            names and introduce surrogate primary keys where appropriate."* –
            the low‑level operations are carried out by the LLM via the exposed
            table/column‑manipulation tools.
        _return_reasoning_steps, parent_chat_context,
        clarification_up_q, clarification_down_q
            Behaviour identical to :py:meth:`update`.

        Returns
        -------
        SteerableToolHandle
            Handle whose :pyfunc:`result` yields a natural‑language summary of
            every structural change (and, optionally, the hidden chain‑of‑
            thought when *_return_reasoning_steps* is *True*).
        """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Attach centralised docstring
BaseKnowledgeManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
