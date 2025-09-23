from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..common.async_tool_loop import SteerableToolHandle
from ..singleton_registry import SingletonABCMeta


class BaseTranscriptManager(ABC, metaclass=SingletonABCMeta):
    """
    *Public* contract that every concrete **transcript-manager** must satisfy.

    Exposes exactly two user-facing operations:

    • **ask**       — answer questions about stored transcripts
    • **summarize** — create & persist summaries of message exchanges
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
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
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
        • Good: "Show me the latest WhatsApp message from Alice" → identify by
          contact and medium, then fetch the most recent message (mention the
          relevant ``message_id``/``exchange_id`` when possible).
        • Bad:  "Should I email them again?" → this is a human decision/clarification;
          use ``request_clarification`` instead.

        Parameters
        ----------
        text : str
            Plain‑English question about existing transcripts, e.g. "Show me the
            latest WhatsApp message from Alice".
        _return_reasoning_steps : bool, default ``False``
            When ``True`` the handle's :pyfunc:`~SteerableToolHandle.result`
            yields ``(answer, messages)`` – the first element is the assistant's
            reply, the second the hidden chain‑of‑thought (useful for debugging).
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
            Await :pymeth:`SteerableToolHandle.result` for the final answer or
            steer the interaction via ``pause()``, ``resume()``, ``interject()``
            or ``stop()``.
        """

    async def summarize(
        self,
        *,
        from_exchanges: Optional[Union[int, List[int]]] = None,
        from_messages: Optional[Union[int, List[int]]] = None,
        omit_messages: Optional[List[int]] = None,
        guidance: Optional[str] = None,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Create a **concise summary** of one or more message exchanges
        (threads, phone calls, etc.) and persist it in the backing store.

        Do *not* request *how* the summary should be produced; state the goal
        and constraints in natural language and allow the `summarize` method
        to determine the best method to generate it.

        Parameters
        ----------
        from_exchanges : int | list[int] | None
            One or more **exchange-IDs** whose constituent messages should be
            summarised.  *Optional* but at least one of *from_exchanges* or
            *from_messages* **must** be supplied.
        from_messages : int | list[int] | None
            Explicit **message-ID(s)** to include in the summary.  Useful for
            stitching together snippets drawn from multiple exchanges.
        omit_messages : list[int] | None
            Message-IDs that should be **excluded** even if they appear in the
            two inclusion lists above.  Applied *last* and therefore
            overrides any overlap.
        guidance : str | None, default ``None``
            Optional *caller-supplied* hints that influence style or focus
            (e.g. *"Emphasise next-steps and deadlines"*).
        parent_chat_context, clarification_up_q, clarification_down_q
            Same semantics as in :py:meth:`ask`.

        Returns
        -------
        str
            The generated summary text (also written to persistent storage).
        """
        raise NotImplementedError(
            "Summarize functionality has been removed from TranscriptManager.",
        )
