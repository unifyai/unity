from tavily import TavilyClient
import functools
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
import asyncio
from unity.settings import SETTINGS
from unity.common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from unity.common.read_only_ask_guard import ReadOnlyAskGuardHandle
from unity.common.llm_client import new_llm_client
from unity.common.llm_helpers import methods_to_tool_dict
from unity.common.tool_spec import ToolSpec
from unity.events.manager_event_logging import log_manager_call
from unity.events.event_bus import EVENT_BUS, Event
from unity.web_searcher import prompt_builders
from .base import BaseWebSearcher


class WebSearcher(BaseWebSearcher):
    """
    Text-based web retrieval engine for quick, one-off internet queries.
    """

    class Config:
        required_contexts = []

    def __init__(self):
        super().__init__()
        self.tavily_client = TavilyClient(api_key=SETTINGS.web.TAVILY_API_KEY or None)

        # Build the tools mapping once; copy when used
        ask_tools: Dict[str, Any] = methods_to_tool_dict(
            ToolSpec(fn=self._search, display_label="Searching the web"),
            ToolSpec(fn=self._extract, display_label="Extracting page content"),
            ToolSpec(fn=self._crawl, display_label="Crawling a website"),
            ToolSpec(fn=self._map, display_label="Mapping website structure"),
            include_class_name=False,
        )
        self.add_tools("ask", ask_tools)
        # Ensure any internal caches are present
        self._provision_storage()

    @functools.wraps(BaseWebSearcher.ask, updated=())
    @log_manager_call(
        "WebSearcher",
        "ask",
        payload_key="question",
        display_label="Searching the Web",
    )
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        client = new_llm_client()

        tools = dict(self.get_tools("ask"))
        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            from unity.common.llm_helpers import make_request_clarification_tool

            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

            async def _on_clar_req(q: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "WebSearcher",
                            "method": "ask",
                            "action": "clarification_request",
                            "question": q,
                        },
                    ),
                )

            async def _on_clar_ans(ans: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "WebSearcher",
                            "method": "ask",
                            "action": "clarification_answer",
                            "answer": ans,
                        },
                    ),
                )

        client.set_system_message(
            prompt_builders.build_ask_prompt(tools=tools).to_list(),
        )

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            response_format=response_format,
            handle_cls=(
                ReadOnlyAskGuardHandle if SETTINGS.UNITY_READONLY_ASK_GUARD else None
            ),
            clarification_queues=_clar_queues,
            on_clarification_request=_on_clar_req,
            on_clarification_answer=_on_clar_ans,
        )

        # If the caller requests reasoning steps, wrap the handle's result
        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore[attr-defined]

        return handle

    # ------------------------------------------------------------------ #
    #  Storage and lifecycle helpers                                     #
    # ------------------------------------------------------------------ #

    def _provision_storage(self) -> None:
        """
        Ensure internal caches exist (idempotent).

        Caches are kept process-local for last-operation snapshots.
        """
        if not hasattr(self, "_last_results"):
            self._last_results: List[Dict[str, Any]] = []
        if not hasattr(self, "_last_extractions"):
            self._last_extractions: Dict[str, Any] = {}
        if not hasattr(self, "_last_crawls"):
            self._last_crawls: Dict[str, Any] = {}
        if not hasattr(self, "_last_maps"):
            self._last_maps: Dict[str, Any] = {}

    @functools.wraps(BaseWebSearcher.clear, updated=())
    def clear(self) -> None:
        self._last_results = []
        self._last_extractions = {}
        self._last_crawls = {}
        self._last_maps = {}

    # ------------------------------------------------------------------ #
    #  Tavily tools                                                      #
    # ------------------------------------------------------------------ #

    def _search(
        self,
        query: str,
        *,
        max_results: int = 5,
        start_date: str = None,
        end_date: str = None,
        include_images: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Perform a web search and return a structured result.

        Parameters
        ----------
        query : str
            The search query.
        max_results : int, default 5
            Maximum number of results to return.
        start_date : str, default None
            Will return all results after the specified start date ( publish date ). Required to be written in the format YYYY-MM-DD.
        end_date : str, default None
            Will return all results before the specified end date ( publish date ). Required to be written in the format YYYY-MM-DD.
        include_images : bool, default False
            Also perform an image search and include the results in the response.

        Returns
        -------
        Dict[str, Any]
            Structured search output with keys:
            - "answer": Concise summary string.
            - "results": Ranked list of sources with titles, URLs and snippets.
            - "images": When requested, a list of related images (may be empty).
        """
        # Tavily rejects start_date == end_date. When the caller wants a
        # single day, widen the window by pushing start_date back one day.
        if start_date and end_date and start_date == end_date:
            try:
                from datetime import datetime, timedelta

                dt = datetime.strptime(start_date, "%Y-%m-%d")
                start_date = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
            except ValueError:
                pass

        response = self.tavily_client.search(
            query=query,
            max_results=max_results,
            start_date=start_date,
            end_date=end_date,
            include_images=include_images,
            include_answer=True,
        )
        return {
            "answer": response.get("answer", ""),
            "results": response.get("results", []),
            "images": response.get("images", []),
        }

    def _extract(
        self,
        urls: str | List[str],
        *,
        include_images: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract cleaned content from webpage URL.

        Parameters
        ----------
        urls : str | List[str]
            The URL to extract content from.
        include_images : bool, default False
            Also extract images from the URLs.

        Returns
        -------
        Dict[str, Any]
            Parsed content payload with keys:
            - "results": Successful extractions with cleaned content/metadata.
            - "failed_results": Any URLs that could not be extracted.
        """
        response = self.tavily_client.extract(urls=urls, include_images=include_images)
        return {
            "results": response.get("results", []),
            "failed_results": response.get("failed_results", []),
        }

    def _crawl(
        self,
        start_url: str,
        *,
        instructions: str | None = None,
        max_depth: int | None = None,
        max_breadth: int | None = None,
        limit: int | None = None,
        include_images: bool | None = None,
    ) -> Dict[str, Any]:
        """
        Graph-based website traversal.

        Parameters
        ----------
        start_url : str
            The root URL to begin the crawl.
        instructions : str, default None
            Natural language instructions for the crawler.
        max_depth : int | None, default None
            Maximum crawl depth (uses service defaults when None).
        max_breadth : int | None, default None
            Maximum number of links to follow per page (uses service defaults when None).
        limit : int | None, default None
            Overall limit on number of pages to crawl (uses service defaults when None).
        include_images : bool | None, default None
            Whether to include images in crawl results (uses service defaults when None).

        Returns
        -------
        Dict[str, Any]
            Crawl summary with keys:
            - "base_url": Normalised base host for the crawl session.
            - "results": List of discovered pages and associated content.
        """
        response = self.tavily_client.crawl(
            url=start_url,
            instructions=instructions,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            include_images=include_images,
        )
        return {
            "base_url": response.get("base_url"),
            "results": response.get("results", []),
        }

    def _map(
        self,
        url: str,
        *,
        instructions: str | None = None,
        max_depth: int | None = None,
        max_breadth: int | None = None,
        limit: int | None = None,
        include_images: bool | None = None,
    ) -> Dict[str, Any]:
        """
        Structured mapping over sources for complex research queries.

        Parameters
        ----------
        url : str
            The root URL to begin the mapping.
        instructions : str | None, default None
            Natural language guidance for the mapping process.
        max_depth : int | None, default None
            Maximum traversal depth (uses service defaults when None).
        max_breadth : int | None, default None
            Maximum number of branches to explore per step (uses service defaults when None).
        limit : int | None, default None
            Overall limit on the number of items to consider (uses service defaults when None).
        include_images : bool | None, default None
            Whether to include images in the mapped results (uses service defaults when None).

        Returns
        -------
        Dict[str, Any]
            Mapping summary with keys:
            - "base_url": Normalised base host when applicable.
            - "results": List of mapped items/pages relevant to the query.
        """
        response = self.tavily_client.map(
            url=url,
            instructions=instructions,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            include_images=include_images,
        )
        return {
            "base_url": response.get("base_url"),
            "results": response.get("results", []),
        }
