"""
BespokeRepairsAgent: Registry + facade for bespoke query functions.

This agent executes pre-defined queries without LLM orchestration.
Each query is a registered async function that can use execute_tool/execute_tools
along with any custom Python logic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

from unity.file_manager.managers.local import LocalFileManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Timing Utilities
# ─────────────────────────────────────────────────────────────────────────────


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


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

    def __init__(self) -> None:
        logger.info("[BespokeRepairsAgent] Initializing agent...")
        init_start = time.perf_counter()

        try:
            self._fm = LocalFileManager()
            logger.debug("[BespokeRepairsAgent] LocalFileManager initialized")
        except Exception as e:
            logger.error(
                f"[BespokeRepairsAgent] Failed to initialize LocalFileManager: {e}",
            )
            raise RuntimeError(f"Agent initialization failed: {e}") from e

        self._tools: Optional[Dict[str, Any]] = None
        self._tools_initialized = False

        init_elapsed = time.perf_counter() - init_start
        logger.info(
            f"[BespokeRepairsAgent] ✓ Agent initialized in {_format_duration(init_elapsed)} | "
            f"Registered queries: {get_registered_count()}",
        )

    def get_tools(self) -> Dict[str, Any]:
        """Get tools from FileManager (cached)."""
        if self._tools is None:
            logger.debug("[BespokeRepairsAgent] Loading tools from FileManager...")
            tools_start = time.perf_counter()

            try:
                self._tools = dict(self._fm.get_tools("ask", include_sub_tools=True))
                tools_elapsed = time.perf_counter() - tools_start

                tool_names = list(self._tools.keys())
                logger.info(
                    f"[BespokeRepairsAgent] ✓ Tools loaded in {_format_duration(tools_elapsed)} | "
                    f"Available: {', '.join(tool_names[:5])}{'...' if len(tool_names) > 5 else ''}",
                )
                self._tools_initialized = True

            except Exception as e:
                logger.error(f"[BespokeRepairsAgent] Failed to load tools: {e}")
                raise RuntimeError(f"Failed to load FileManager tools: {e}") from e

        return self._tools

    def list_queries(self) -> List[Dict[str, str]]:
        """List all available query IDs and descriptions."""
        logger.debug("[BespokeRepairsAgent] Listing available queries")
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
        RuntimeError
            If query execution fails
        """
        ask_start = time.perf_counter()
        logger.info(f"[BespokeRepairsAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[BespokeRepairsAgent] Starting query: {query_id}")

        # Log parameters (sanitized)
        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            logger.info(f"[BespokeRepairsAgent] Parameters: {param_str}")
        else:
            logger.info("[BespokeRepairsAgent] Parameters: (none)")

        # Validate query exists
        spec = get_query(query_id)
        if spec is None:
            available = [q["query_id"] for q in self.list_queries()]
            logger.error(
                f"[BespokeRepairsAgent] ✗ Unknown query '{query_id}'. "
                f"Available queries: {available}",
            )
            raise ValueError(
                f"Unknown query_id '{query_id}'. "
                f"Use --list to see available queries. "
                f"Available: {', '.join(available[:5])}{'...' if len(available) > 5 else ''}",
            )

        logger.debug(f"[BespokeRepairsAgent] Query found: {spec.description}")

        # Get tools
        try:
            tools = self.get_tools()
            logger.debug(f"[BespokeRepairsAgent] Tools ready ({len(tools)} available)")
        except Exception as e:
            logger.error(f"[BespokeRepairsAgent] ✗ Failed to get tools: {e}")
            raise

        # Execute query
        try:
            logger.info(f"[BespokeRepairsAgent] Executing query function...")
            exec_start = time.perf_counter()

            result = await spec.fn(tools, **params)

            exec_elapsed = time.perf_counter() - exec_start
            total_elapsed = time.perf_counter() - ask_start

            # Log result summary
            if isinstance(result, dict):
                result_count = len(result.get("results", []))
                total_val = result.get("total", "N/A")
                logger.info(
                    f"[BespokeRepairsAgent] ✓ Query '{query_id}' completed | "
                    f"Exec: {_format_duration(exec_elapsed)} | "
                    f"Total: {_format_duration(total_elapsed)} | "
                    f"Results: {result_count} groups | Total value: {total_val}",
                )
            else:
                logger.info(
                    f"[BespokeRepairsAgent] ✓ Query '{query_id}' completed | "
                    f"Exec: {_format_duration(exec_elapsed)} | "
                    f"Total: {_format_duration(total_elapsed)}",
                )

            logger.info(
                f"[BespokeRepairsAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            )
            return result

        except ValueError as e:
            elapsed = time.perf_counter() - ask_start
            logger.error(
                f"[BespokeRepairsAgent] ✗ Query '{query_id}' failed (ValueError) "
                f"after {_format_duration(elapsed)}: {e}",
            )
            raise

        except RuntimeError as e:
            elapsed = time.perf_counter() - ask_start
            logger.error(
                f"[BespokeRepairsAgent] ✗ Query '{query_id}' failed (RuntimeError) "
                f"after {_format_duration(elapsed)}: {e}",
            )
            raise

        except Exception as e:
            elapsed = time.perf_counter() - ask_start
            logger.exception(
                f"[BespokeRepairsAgent] ✗ Query '{query_id}' failed unexpectedly "
                f"after {_format_duration(elapsed)}",
            )
            raise RuntimeError(
                f"Query '{query_id}' failed unexpectedly: {type(e).__name__}: {e}",
            ) from e
