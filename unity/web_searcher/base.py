from __future__ import annotations

from abc import abstractmethod
import asyncio
from typing import Dict, List, Optional, Any, Type
from pydantic import BaseModel

from ..common.async_tool_loop import SteerableToolHandle
from ..singleton_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager


class BaseWebSearcher(BaseStateManager, metaclass=SingletonABCMeta):
    """
    Public contract that every concrete web-search manager must satisfy.

    A web-search manager answers questions (ask) by orchestrating search,
    extraction and site traversal tools behind the scenes. Implementations may
    talk to a real provider SDK, a remote service, or a purely simulated LLM –
    but they all expose exactly the public method documented below.
    """

    _as_caller_description: str = (
        "the WebSearcher, performing web research on behalf of the end user"
    )

    @abstractmethod
    async def ask(
        self,
        text: str,
        *,
        _response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
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
        - Include any required citations/links and, when relevant, the desired time window and scope in the initial question; avoid provider- or engine-specific hints.
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
        _response_format : Type[BaseModel] | None, default None
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
    async def update(
        self,
        text: str,
        *,
        _response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Apply a mutation request related to the WebSearcher configuration and
        return a live SteerableToolHandle.

        Purpose
        -------
        Use this method to create, edit, or delete configuration records owned
        by the WebSearcher (e.g., entries in a `Websites` table capturing
        websites of interest). This method must not browse or answer web
        research questions; it manages stored metadata only.

        Guidance
        --------
        - Treat `host` as the natural unique key for a website entry.
        - After any mutation (create/delete), verify results via the read‑only
          `ask` surface (e.g., using catalog tools like `_filter_websites` or
          `_search_websites`).
        - Credentials must be referenced by integer `secret_id`s (foreign keys);
          never include raw secret values in messages or logs.

        Clarifications
        --------------
        Do not ask users questions in the final answer. When a clarification
        channel is provided, route follow‑ups there. If not provided, proceed
        with sensible defaults and state assumptions when relevant.

        Parameters
        ----------
        _response_format : Type[BaseModel] | None, default None
            Optional Pydantic model to request a structured outcome. When provided,
            the final result should conform to this schema; otherwise a plain
            string summary is returned.

        Returns
        -------
        SteerableToolHandle
            A handle exposing pause/resume/interject/stop; await result() for
            the final outcome summary.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Attach centralised docstring
BaseWebSearcher.clear.__doc__ = CLEAR_METHOD_DOCSTRING
