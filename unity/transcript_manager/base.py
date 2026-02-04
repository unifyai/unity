from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel
from ..common.async_tool_loop import SteerableToolHandle
from ..manager_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager
from .types.message import Message
from .types.exchange import Exchange


class BaseTranscriptManager(BaseStateManager, metaclass=SingletonABCMeta):
    """
    *Public* contract that every concrete **transcript-manager** must satisfy.

    Exposes exactly one user-facing operation:

    • **ask** — answer questions about stored transcripts
    """

    _as_caller_description: str = (
        "the TranscriptManager, analyzing conversation transcripts"
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
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Interrogate the **existing transcripts** (read‑only) and obtain a live
        :class:`SteerableToolHandle`.

        Purpose
        -------
        Use this method to locate and analyse messages that already exist in the
        store: retrieve messages for a contact or exchange, filter by channel or
        time, perform semantic searches over content (and, when available, sender
        contact attributes), or summarise/compare existing entries in prose. This
        call must never create, modify or delete messages.

        Clarifications
        --------------
        Do not use this method to ask the human follow‑up questions. If the
        caller needs clarification about what to retrieve (e.g., which
        conversation, which date range, which person), route the question via a
        dedicated ``request_clarification`` tool when available. If no
        clarification channel exists, proceed with sensible defaults/best‑guess
        values and state those assumptions in the outer loop's final reply.

        Do *not* request *how* the question should be answered; just ask the
        question in natural language and allow this `ask` method to determine
        the best method to answer it.

        Examples
        --------
        • Good: "Show me the latest SMS from Alice" → identify by
          contact and medium, then fetch the most recent message (mention the
          relevant ``message_id``/``exchange_id`` when possible).
        • Bad:  "Should I email them again?" → this is a human decision/clarification;
          use ``request_clarification`` instead.

        Parameters
        ----------
        text : str
            Plain‑English question about existing transcripts, e.g. "Show me the
            latest SMS from Alice".
        response_format : Type[BaseModel] | None, default ``None``
            Optional Pydantic model to request a structured answer. When provided,
            the final result should conform to this schema; otherwise a plain
            string answer is returned.
        _return_reasoning_steps : bool, default ``False``
            When ``True`` the handle's :pyfunc:`~SteerableToolHandle.result`
            yields ``(answer, messages)`` – the first element is the assistant's
            reply, the second the hidden chain‑of‑thought (useful for debugging).
        _parent_chat_context : list[dict] | None
            Optional read‑only chat history that will be provided to all nested
            tool calls.
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
    def clear(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Additional public helpers (programmatic, non-LLM entrypoints)       #
    # ------------------------------------------------------------------ #
    def log_messages(
        self,
        messages: Union[
            Union[Dict[str, Any], Message],
            List[Union[Dict[str, Any], Message]],
        ],
        synchronous: bool = False,
    ) -> List[Message]:
        """
        Insert one or more transcript messages.

        Parameters
        ----------
        messages : dict | Message | list[dict | Message]
            Message payload(s) following the Message schema. Implementations
            must validate inputs and return the persisted Message models with
            assigned identifiers.
        synchronous : bool, default False
            Hint to publish related events synchronously (when supported).

        Returns
        -------
        list[Message]
            The created messages as validated Message models.
        """
        raise NotImplementedError

    def join_published(self) -> None:
        """
        Block until any internally queued publish operations have drained.

        Implementations may no-op if publishing is synchronous.
        """
        raise NotImplementedError

    @staticmethod
    def build_plain_transcript(
        messages: list[dict],
        *,
        contact_manager: Optional[Any] = None,
    ) -> str:
        """
        Return a plain-text transcript ("Full Name: content") for provided messages.

        The optional contact resolver may be supplied to map numeric sender ids
        to human-readable names. When omitted, an implementation-defined default
        resolution strategy may be used.
        """
        raise NotImplementedError

    def update_contact_id(
        self,
        *,
        original_contact_id: int,
        new_contact_id: int,
    ) -> Dict[str, Any]:
        """
        Replace all occurrences of one contact id with another across messages.

        The substitution applies to both sender_id and entries inside receiver_ids.

        Returns a summary payload describing how many messages were updated.
        """
        raise NotImplementedError

    def get_exchange_metadata(self, exchange_id: int) -> Exchange:
        """
        Fetch the Exchanges row for the given exchange_id as an Exchange model.
        """
        raise NotImplementedError

    def update_exchange_metadata(
        self,
        exchange_id: int,
        metadata: Dict[str, Any],
    ) -> Exchange:
        """
        Update (or create) metadata for the specified exchange and return the updated Exchange.
        """
        raise NotImplementedError

    def log_first_message_in_new_exchange(
        self,
        message: Union[Dict[str, Any], Message],
        *,
        exchange_initial_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Log the first message of a new exchange and set initial exchange metadata.

        Returns the newly assigned exchange_id.
        """
        raise NotImplementedError

    def filter_exchanges(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int | None = 100,
    ) -> Dict[str, Any]:
        """
        Filter Exchanges rows using a boolean Python expression evaluated per row.

        Parameters
        ----------
        filter : str | None, default None
            A Python boolean expression evaluated with column names in scope.
            When None, returns all exchanges.
        offset : int, default 0
            Zero-based index of the first result to include.
        limit : int | None, default 100
            Maximum number of records to return. Implementations may cap this value.

        Returns
        -------
        Dict[str, Any]
            A payload containing the matching exchanges (e.g., {"exchanges": [Exchange, ...]}).
        """
        raise NotImplementedError


# Attach centralised docstring
BaseTranscriptManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
