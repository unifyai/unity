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
        Use this for external or web-based information needs. This includes:
        - General knowledge or best practices not stored in internal managers
        - Live or time-sensitive facts (e.g., "today", "yesterday", "this week",
          "latest", "current", "now") that cannot be reliably answered from an
          offline model snapshot

        Orchestration contract
        ----------------------
        - Callers must send a single high-level, natural-language question.
        - Do NOT fan-out multiple `ask` calls with provider/site hints (e.g., "site:").
        - The WebSearcher itself will perform source selection, parallel search,
          extraction, and aggregation. It may ask for clarification if needed.

        Query ergonomics (important)
        ----------------------------
        - Avoid redundant serial re-queries. If you need citations/links, ask for
          them in the initial question.
        - Only issue a second `ask` when the first response clearly indicates
          missing coverage or ambiguity that cannot be resolved without another
          targeted fetch.
        - Use multiple `ask` calls in parallel only when the user's message
          contains genuinely unrelated sub-questions; otherwise keep to a single
          `ask` and let this manager fan-out to sources internally.
        - Do not re-query just to "confirm" a previous `ask`; perform that
          validation within the initial request.

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
