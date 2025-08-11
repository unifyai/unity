from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..common.llm_helpers import SteerableToolHandle
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
        Answer a **natural-language** question about the stored transcripts
        (emails, chats, calls …) and return a *live* ``SteerableToolHandle``
        to the LLM reasoning session.

        Do *not* request *how* the question should be answered; just ask the
        question in natural language and allow the `ask` method to determine
        the best method to answer it.

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
