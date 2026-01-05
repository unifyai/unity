"""
Query registry for BespokeRepairsAgent.

This module provides the @register decorator and query registry
for registering metric query functions.

Usage:
    from intranet.repairs_agent.static.registry import register

    @register("my_query", "Description of what this query does")
    async def my_query(tools, **params):
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class QuerySpec:
    """Specification for a registered query."""

    query_id: str
    description: str
    fn: Callable[..., Coroutine[Any, Any, Any]]
    registered_at: datetime = field(default_factory=datetime.now)


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
        if query_id in _REGISTRY:
            logger.warning(
                f"[Registry] Overwriting existing query '{query_id}' - "
                f"was registered at {_REGISTRY[query_id].registered_at}",
            )
        else:
            logger.debug(f"[Registry] Registering query: {query_id}")

        _REGISTRY[query_id] = QuerySpec(
            query_id=query_id,
            description=description,
            fn=fn,
        )
        return fn

    return decorator


def list_queries() -> List[Dict[str, str]]:
    """List all registered queries."""
    queries = [
        {"query_id": q.query_id, "description": q.description}
        for q in _REGISTRY.values()
    ]
    logger.debug(f"[Registry] Listed {len(queries)} registered queries")
    return queries


def get_query(query_id: str) -> QuerySpec | None:
    """Get a query by ID."""
    spec = _REGISTRY.get(query_id)
    if spec is None:
        logger.warning(f"[Registry] Query '{query_id}' not found in registry")
    return spec


def get_registered_count() -> int:
    """Get the number of registered queries."""
    return len(_REGISTRY)
