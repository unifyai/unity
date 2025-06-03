# unity/knowledge_manager/base.py
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..common.llm_helpers import SteerableToolHandle


class BaseKnowledgeManager(ABC):
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
    def store(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:  # noqa: D401 – full docstring below
        """
        Execute an English instruction that **adds to** or **refactors** the
        knowledge base (create tables/columns, insert rows, etc.).

        Parameters
        ----------
        text :
            Natural-language command – e.g. *“Add Adrian’s birthday: 2 Jun 1994.”*
        return_reasoning_steps :
            If *True*, :pyattr:`result` returns ``(answer, messages)`` where
            *messages* contains the model’s hidden chain-of-thought.
        parent_chat_context :
            Extra messages to prepend to the tool-use loop.
        clarification_up_q / clarification_down_q :
            Optional **async** queues used when the model needs to ask the
            human for more details.

        Returns
        -------
        SteerableToolHandle
            Handle for awaiting / steering the conversation.
        """

    @abstractmethod
    def retrieve(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Answer a *text* question by searching the stored knowledge.

        All parameters mirror :py:meth:`store`.  See that method for details.
        """
