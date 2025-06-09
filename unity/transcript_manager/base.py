from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..common.llm_helpers import SteerableToolHandle


class BaseTranscriptManager(ABC):
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
        Answer a **natural-language** question about the stored transcripts
        (emails, chats, calls …) and return a *live* ``SteerableToolHandle``
        to the LLM reasoning session.

        Parameters
        ----------
        text : str
            The user's free-form question (e.g. *"Show me the latest WhatsApp
            message from Alice"*).
        _return_reasoning_steps : bool, default ``False``
            When ``True`` the handle's :pyfunc:`~SteerableToolHandle.result`
            yields ``(answer, messages)`` where *messages* is the complete
            internal chat (useful for debugging).
        parent_chat_context : list[dict] | None
            Optional *read-only* chat history that will be provided to all
            nested tool calls.
        clarification_up_q / clarification_down_q : asyncio.Queue[str] | None
            Duplex channels enabling the LLM to **ask** the human for missing
            details (push to *up_q*) and **receive** the reply (read from
            *down_q*).

        Returns
        -------
        SteerableToolHandle
            A steerable handle that can be awaited, paused/resumed, stopped,
            or interjected with extra user turns.
        """

    @abstractmethod
    async def summarize(
        self,
        *,
        exchange_ids: Union[int, List[int]],
        guidance: Optional[str] = None,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> str:
        """
        Create a **concise summary** of one or more message exchanges
        (threads, phone calls, etc.) and persist it in the backing store.

        Parameters
        ----------
        exchange_ids : int | list[int]
            Identifier(s) of the exchange(s) to summarise.
        guidance : str | None, default ``None``
            Optional *caller-supplied* hints that influence style or focus
            (e.g. *"Emphasise next-steps and deadlines"*).
        parent_chat_context, clarification_up_q, clarification_down_q
            Same semantics as in :py:meth:`ask`.

        Returns
        -------
        str
            The generated summary (also written to persistent storage).
        """
