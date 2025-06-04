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
        Ask a *text* question about the contact book and obtain a
        :class:`~unity.common.llm_helpers.SteerableToolHandle` that lets the
        caller await the answer (and optionally the underlying reasoning
        messages) and steer the conversation further.

        Parameters
        ----------
        text:
            Natural-language query – e.g. *“Show me Alice’s phone number.”*
        _return_reasoning_steps:
            If *True*, the final ``result`` coroutine returns a
            ``(answer, messages)`` tuple where *messages* holds the
            model’s hidden chain-of-thought.
        parent_chat_context:
            Extra messages to prepend to the tool loop.
        clarification_up_q / clarification_down_q:
            Optional **async** queues used when the model needs to ask the
            human for disambiguation.

        Returns
        -------
        SteerableToolHandle
            Handle for awaiting / steering the conversation.
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
        Execute an English instruction that **creates** or **modifies**
        contacts (add, remove, rename, …).

        All parameters mirror :py:meth:`ask`.  See that method for details.
        """
