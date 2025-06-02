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
        return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Answer a natural-language question about the transcript history.
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
        Summarise one or more message exchanges and store the result.
        """
