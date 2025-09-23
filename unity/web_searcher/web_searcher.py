import os
import json
from tavily import TavilyClient
from typing import List, Dict, Any, Optional
import unify
from unity.common.async_tool_loop import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from unity.common.llm_helpers import (
    inject_broader_context,
    methods_to_tool_dict,
)
from unity.events.manager_event_logging import log_manager_call
from unity.web_searcher import prompt_builders


class WebSearcher:
    """
    Manages web search and extraction.
    """

    def __init__(self):
        self.tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        # Build the tools mapping once; copy when used
        self._ask_tools: Dict[str, Any] = methods_to_tool_dict(
            self._search,
            self._extract,
            self._crawl,
            self._map,
            include_class_name=False,
        )

    @log_manager_call("WebSearcher", "ask", payload_key="question")
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        """
        Ask a web question. Uses an async tool-use loop with web tools.
        """
        client = self._new_llm_client("gpt-5@openai")

        tools = dict(self._ask_tools)

        client.set_system_message(
            prompt_builders.build_ask_prompt(tools=tools),
        )

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            preprocess_msgs=inject_broader_context,
        )

        # If the caller requests reasoning steps, wrap the handle's result
        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore[attr-defined]

        return handle

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

    # ------------------------------------------------------------------ #
    #  Small internal helpers (LLM client + tool policies)               #
    # ------------------------------------------------------------------ #

    def _new_llm_client(self, model: str) -> "unify.AsyncUnify":
        return unify.AsyncUnify(
            model,
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )
