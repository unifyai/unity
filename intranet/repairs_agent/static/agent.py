"""
BespokeRepairsAgent: LLM-free facade for executing registered metric queries.

This agent provides direct, deterministic execution of pre-defined metric
functions without any LLM orchestration. Ideal for dashboards, automated
reports, and performance-critical use cases.

Usage:
    from intranet.repairs_agent.static import BespokeRepairsAgent

    agent = BespokeRepairsAgent()

    # List available queries
    print(agent.list_queries())

    # Execute a query
    result = await agent.ask("first_time_fix_rate", group_by="region")
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Dict, List, Optional

from unity.common.llm_client import new_llm_client
from unity.file_manager.managers.local import LocalFileManager

from ..config.prompt_builder import (
    build_analyst_system_prompt,
    build_analyst_user_prompt,
)
from .registry import get_query, get_registered_count, list_queries

logger = logging.getLogger(__name__)


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

    def __init__(self, skip_analysis: bool = False) -> None:
        """
        Initialize the BespokeRepairsAgent.

        Parameters
        ----------
        skip_analysis : bool
            If True, return raw metric results without LLM analysis.
            Useful for debugging or when only raw data is needed.
        """
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
        self._skip_analysis = skip_analysis
        self._llm_client = None if skip_analysis else new_llm_client()

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
        logger.info("[BespokeRepairsAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
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
            logger.info("[BespokeRepairsAgent] Executing query function...")
            exec_start = time.perf_counter()

            raw_result = await spec.fn(tools, **params)

            exec_elapsed = time.perf_counter() - exec_start

            # Convert to dict if Pydantic model
            if hasattr(raw_result, "model_dump"):
                result_dict = raw_result.model_dump()
            else:
                result_dict = raw_result

            # Log result summary
            if isinstance(result_dict, dict):
                result_count = len(result_dict.get("results", []))
                total_val = result_dict.get("total", "N/A")
                logger.info(
                    f"[BespokeRepairsAgent] ✓ Query '{query_id}' completed | "
                    f"Exec: {_format_duration(exec_elapsed)} | "
                    f"Results: {result_count} groups | Total value: {total_val}",
                )

            # If skip_analysis, return raw results
            if self._skip_analysis:
                total_elapsed = time.perf_counter() - ask_start
                logger.info(
                    "[BespokeRepairsAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                )
                return result_dict

            # Run LLM analysis
            logger.info("[BespokeRepairsAgent] Running LLM analysis...")
            analysis_start = time.perf_counter()

            # Get metric docstring for context
            metric_docstring = inspect.getdoc(spec.fn)

            # Extract plots from result
            plots = result_dict.get("plots", [])
            plots_list = [
                p.model_dump() if hasattr(p, "model_dump") else p for p in plots
            ]

            # Build prompts
            system_prompt = build_analyst_system_prompt()
            user_prompt = build_analyst_user_prompt(
                metric_name=query_id,
                metric_description=spec.description,
                metric_docstring=metric_docstring,
                params=params,
                results=result_dict,
                plots=plots_list,
            )

            # Call LLM
            response = await self._llm_client.generate(
                user_message=user_prompt,
                system_message=system_prompt,
            )

            analysis_elapsed = time.perf_counter() - analysis_start
            total_elapsed = time.perf_counter() - ask_start

            logger.info(
                f"[BespokeRepairsAgent] ✓ Analysis complete | "
                f"Analysis: {_format_duration(analysis_elapsed)} | "
                f"Total: {_format_duration(total_elapsed)}",
            )

            logger.info(
                "[BespokeRepairsAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            )

            # Return analysis with metadata
            return {
                "analysis": response,
                "metric_name": query_id,
                "metric_description": spec.description,
                "group_by": result_dict.get("group_by"),
                "total": result_dict.get("total"),
                "plots": plots_list,
                "raw_results": result_dict,
                "timings": {
                    "query_ms": round(exec_elapsed * 1000),
                    "analysis_ms": round(analysis_elapsed * 1000),
                    "total_ms": round(total_elapsed * 1000),
                },
            }

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
