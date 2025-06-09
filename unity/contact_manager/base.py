# contact_manager/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from typing import Dict, List, Optional, Any

from ..common.llm_helpers import SteerableToolHandle


class BaseContactManager(ABC):
    """
    *Public* contract that every concrete **contact-manager** must satisfy.

    A contact-manager answers questions (`ask`) about stored contacts and
    handles English instructions (`update`) that create or change those
    contacts.  Implementations may talk to a real database, a remote CRM,
    an in-memory mock, or even a purely simulated LLM – but they **all**
    expose exactly the two public methods documented below.
    """

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Query the contact book in natural language and receive an **LLM
        reasoning handle** (`SteerableToolHandle`) that can be awaited,
        paused/resumed, interjected, or cancelled.

        Parameters
        ----------
        text : str
            The user's plain-English question (e.g. *"Show me Alice's phone
            number."*).
        _return_reasoning_steps : bool, default ``False``
            When *True*, :pyfunc:`SteerableToolHandle.result` returns a
            tuple ``(answer, messages)`` where *messages* is the invisible
            chain-of-thought exchanged with the LLM.
        parent_chat_context : list[dict] | None
            **Read-only** conversation context to prepend to the tool loop.
        clarification_up_q / clarification_down_q : asyncio.Queue[str] | None
            Optional duplex channels.  When supplied the LLM can ask the human
            follow-up questions via *up_q* and must read answers from
            *down_q*.

        Returns
        -------
        SteerableToolHandle
            A live handle that ultimately yields the assistant's answer and
            exposes steering operations (``pause``, ``resume``, ``interject``,
            ``stop``).
        """

    @abstractmethod
    def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Execute a natural-language **mutation** request on the contact book
        (create, update, delete, rename, …) and obtain a steerable handle to
        the LLM conversation.

        Parameters
        ----------
        text : str
            The user's request (e.g. *"Add Sarah Connor's phone number …"*).
        _return_reasoning_steps, parent_chat_context,
        clarification_up_q, clarification_down_q
            Same semantics as in :py:meth:`ask`.

        Returns
        -------
        SteerableToolHandle
            Handle whose :pyfunc:`result` yields confirmation of the mutation
            and (optionally) reasoning steps.
        """
