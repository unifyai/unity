"""
BespokeRepairsAgent: Registry + facade for bespoke query functions.

This agent executes pre-defined queries without LLM orchestration.
Each query is a registered async function that can use execute_tool/execute_tools
along with any custom Python logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List

from unity.file_manager.managers.local import LocalFileManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QuerySpec:
    """Specification for a registered query."""

    query_id: str
    description: str
    fn: Callable[..., Coroutine[Any, Any, Any]]


_REGISTRY: Dict[str, QuerySpec] = {}


def register(query_id: str, description: str):
    """
    Decorator to register a query function.

    Usage:
        @register("my_query", "Description of what this query does")
        async def my_query(tools, **params):
            ...
    """

    def decorator(fn: Callable[..., Coroutine[Any, Any, Any]]):
        _REGISTRY[query_id] = QuerySpec(
            query_id=query_id,
            description=description,
            fn=fn,
        )
        return fn

    return decorator


def list_queries() -> List[Dict[str, str]]:
    """List all registered queries."""
    return [
        {"query_id": q.query_id, "description": q.description}
        for q in _REGISTRY.values()
    ]


def get_query(query_id: str) -> QuerySpec | None:
    """Get a query by ID."""
    return _REGISTRY.get(query_id)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────


class BespokeRepairsAgent:
    """
    Facade for executing registered bespoke queries.

    Usage:
        agent = BespokeRepairsAgent()

        # List available queries
        print(agent.list_queries())

        # Execute a query
        result = await agent.ask("repairs_per_day_per_operative", start_date="2025-07-01")
    """

    def __init__(self):
        self._fm = LocalFileManager()
        self._tools = None

    def get_tools(self) -> Dict[str, Any]:
        """Get tools from FileManager (cached)."""
        if self._tools is None:
            self._tools = dict(self._fm.get_tools("ask", include_sub_tools=True))
        return self._tools

    def list_queries(self) -> List[Dict[str, str]]:
        """List all available query IDs and descriptions."""
        return list_queries()

    async def ask(self, query_id: str, **params) -> Any:
        """
        Execute a registered query.

        Parameters
        ----------
        query_id : str
            The query to execute
        **params
            Parameters to pass to the query function

        Returns
        -------
        Whatever the query function returns

        Raises
        ------
        ValueError
            If query_id is not registered
        """
        spec = get_query(query_id)
        if spec is None:
            available = [q["query_id"] for q in self.list_queries()]
            raise ValueError(f"Unknown query_id '{query_id}'. Available: {available}")

        logger.info(f"Executing query '{query_id}' with params: {params}")
        tools = self.get_tools()
        result = await spec.fn(tools, **params)
        logger.info(f"Query '{query_id}' completed")
        return result
