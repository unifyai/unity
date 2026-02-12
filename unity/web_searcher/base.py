from __future__ import annotations

from abc import abstractmethod
import asyncio
from typing import Dict, List, Optional, Any, Type
from pydantic import BaseModel

from ..common.async_tool_loop import SteerableToolHandle
from ..manager_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager


class BaseWebSearcher(BaseStateManager, metaclass=SingletonABCMeta):
    """
    Public contract that every concrete web-search manager must satisfy.

    A web-search manager answers quick, one-off internet questions (ask) by
    orchestrating search and extraction tools behind the scenes. It is a
    lightweight, text-based retrieval engine — not a browser automation or
    gated-site access tool.
    """

    _as_caller_description: str = (
        "the WebSearcher, performing web research on behalf of the end user"
    )

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
        Answer a quick, one-off web question and return a live SteerableToolHandle.

        Purpose
        -------
        Use this for fast, basic internet queries against the public web:
        - "What are the headlines today?"
        - "What is the weather like in Paris this morning?"
        - "What is the current stock price of AAPL?"
        - "What is the Eisenhower Matrix?"
        - General knowledge, definitions, or best practices not stored internally
        - Live or time-sensitive facts (e.g., "today", "latest", "current", "now")

        NOT intended for
        ----------------
        - Accessing gated or authenticated websites (use Tavily + SecretManager +
          ComputerPrimitives directly via code-first plans for those)
        - Complex multi-step browser automation or "doing work" in a browser
        - Scraping behind login walls or paywalls

        This is purely a text-based retrieval engine for super-quick internet
        lookups. For anything requiring credentials, browser sessions, or
        multi-step web interaction, use the more expressive code-first plan
        approach with direct tool composition.

        Orchestration contract
        ----------------------
        - Callers must send a single high-level, natural-language question.
        - The WebSearcher will perform search, extraction, and aggregation
          internally. It may ask for clarification if needed.

        Query ergonomics (important)
        ----------------------------
        - Avoid redundant serial re-queries. If you need citations/links, ask for
          them in the initial question.
        - Include any required citations/links and, when relevant, the desired
          time window and scope in the initial question.
        - Only issue a second `ask` when the first response clearly indicates
          missing coverage or ambiguity that cannot be resolved without another
          targeted fetch.
        - Use multiple `ask` calls in parallel only when the user's message
          contains genuinely unrelated sub-questions; otherwise keep to a single
          `ask` and let this manager fan-out to sources internally.

        Clarifications
        --------------
        Do not ask the user questions in the final answer. When a clarification
        channel is provided via request_clarification, route any follow-ups there.
        If no clarification channel exists, proceed with sensible defaults or
        best-guess values and state assumptions in the final answer when relevant.

        Parameters
        ----------
        text : str
            The user's plain-English research question.
        response_format : Type[BaseModel] | None, default None
            Optional Pydantic model to request a structured answer. When provided,
            the final result should conform to this schema; otherwise a plain
            string answer is returned.
        _return_reasoning_steps : bool, default False
            When True, SteerableToolHandle.result returns (answer, messages)
            where messages are the internal tool-loop messages.
        _parent_chat_context : list[dict] | None
            Read-only conversation context to prepend to the tool loop.
        _clarification_up_q / _clarification_down_q : asyncio.Queue[str] | None
            Optional duplex channels for interactive clarification.

        Returns
        -------
        SteerableToolHandle
            A live handle that ultimately yields the assistant's answer and
            exposes steering operations (pause, resume, interject, stop).
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Attach centralised docstring
BaseWebSearcher.clear.__doc__ = CLEAR_METHOD_DOCSTRING
