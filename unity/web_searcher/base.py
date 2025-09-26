from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from typing import Dict, List, Optional, Any

from ..common.async_tool_loop import SteerableToolHandle
from ..singleton_registry import SingletonABCMeta


class BaseWebSearcher(ABC, metaclass=SingletonABCMeta):
    """
    Public contract that every concrete web-search manager must satisfy.

    A web-search manager answers questions (ask) by orchestrating search,
    extraction and site traversal tools behind the scenes. Implementations may
    talk to a real provider SDK, a remote service, or a purely simulated LLM –
    but they all expose exactly the public method documented below.
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
        Answer a web research question and return a live SteerableToolHandle.

        Purpose
        -------
        Use this to research topics on the web. The implementation chooses among
        available tools (e.g., search, extract, crawl, map) and stops when
        sufficient evidence is gathered to provide a concise final answer.

        Clarifications
        --------------
        Do not ask the user questions in the final answer. When a clarification
        channel is provided via request_clarification, route any follow-ups there.
        If no clarification channel exists, proceed with sensible defaults or
        best‑guess values and state assumptions in the final answer when relevant.

        Parameters
        ----------
        text : str
            The user's plain‑English research question.
        _return_reasoning_steps : bool, default False
            When True, SteerableToolHandle.result returns (answer, messages)
            where messages are the internal tool-loop messages.
        parent_chat_context : list[dict] | None
            Read-only conversation context to prepend to the tool loop.
        clarification_up_q / clarification_down_q : asyncio.Queue[str] | None
            Optional duplex channels for interactive clarification.

        Returns
        -------
        SteerableToolHandle
            A live handle that ultimately yields the assistant's answer and
            exposes steering operations (pause, resume, interject, stop).
        """
        raise NotImplementedError
