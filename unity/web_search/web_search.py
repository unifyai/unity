from tavily import TavilyClient
from typing import List, Dict, Any, Optional
import os
import json
import unify
from ..common.llm_helpers import (
    start_async_tool_use_loop,
    inject_broader_context,
    TOOL_LOOP_LINEAGE,
    SteerableToolHandle,
)
from . import prompt_builders


class WebSearch:
    """
    Manages web search and extraction using the Tavily Python SDK.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = TavilyClient(api_key=self.api_key)

    # 1. Main entrypoint
    async def ask(
        self,
        text: str,
        *,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> SteerableToolHandle:
        """
        Ask a web question. Uses an async tool-use loop with web tools.
        """
        client = self._new_llm_client("gpt-5@openai")

        tools = {
            "search": self._search,
            "extract": self._extract,
            "crawl": self._crawl,
            "map": self._map,
        }

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

        return handle

    # 2. Core Tavily endpoints
    def _search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a web search and return top results.

        Parameters
        ----------
        query: str
            The query to search for.
        max_results: int
            The maximum number of results to return.
            Default is 5.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the answer, results, and images (if any).
        """
        response = self.client.search(
            query=query,
            max_results=max_results,
            include_answer=True,
        )
        return {
            "answer": response.get("answer", ""),
            "results": response.get("results", []),
            "images": response.get("images", []),
        }

    def _extract(self, url_to_extract: str) -> Dict[str, Any]:
        """
        Extract clean text from a given URL.
        """
        return self.client.extract(url=url_to_extract)

    def _crawl(self, start_url: str, depth: int = 1) -> Dict[str, Any]:
        """
        Crawl a site starting from a given URL.
        """
        return self.client.crawl(url=start_url, depth=depth)

    def _map(self, query: str) -> Dict[str, Any]:
        """
        Perform structured search (semantic mapping).
        """
        return self.client.map(query=query)

    # 4. LLM client helper
    def _new_llm_client(self, model: str) -> "unify.AsyncUnify":
        return unify.AsyncUnify(
            model,
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )

    # 3. Internal helper for summarisation
    def _summarise(self, prompt: str) -> str:
        """
        Call your LLM with the built prompt.
        Replace with actual OpenAI Responses/Anthropic call.
        """
        # Example placeholder
        return f"[LLM SUMMARY OUTPUT] {prompt[:200]}..."  # truncate preview
