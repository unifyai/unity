"""
Simple trajectory executor - runs tool calls and returns results.

This module provides utilities to execute pre-defined tool trajectories
without LLM orchestration. Used by BespokeRepairsAgent for deterministic
query execution.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from unity.common.tool_spec import normalise_tools

logger = logging.getLogger(__name__)


async def execute_tools(
    steps: List[Dict[str, Any]],
    tools: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Execute a list of tool calls.

    Parameters
    ----------
    steps : List[Dict]
        Each step: {"name": str, "arguments": dict}
    tools : Dict
        Tools from FileManager.get_tools()

    Returns
    -------
    List[Dict]
        Each result: {"name": str, "arguments": dict, "result": Any, "error": str | None}
    """
    normalized = normalise_tools(dict(tools))
    results = []

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(thread_name_prefix="trajectory_exec") as executor:
        for step in steps:
            name = step["name"]
            args = step.get("arguments", {})

            entry = {"name": name, "arguments": args, "result": None, "error": None}

            if name not in normalized:
                entry["error"] = f"Tool '{name}' not found"
                results.append(entry)
                logger.warning(f"Tool '{name}' not found, skipping")
                continue

            fn = normalized[name].fn

            try:
                if inspect.iscoroutinefunction(fn):
                    entry["result"] = await fn(**args)
                else:
                    entry["result"] = await loop.run_in_executor(
                        executor,
                        functools.partial(fn, **args),
                    )
            except Exception as e:
                entry["error"] = str(e)
                logger.error(f"Tool '{name}' failed: {e}")

            results.append(entry)

    return results


async def execute_tool(
    name: str,
    arguments: Dict[str, Any],
    tools: Dict[str, Any],
) -> Any:
    """
    Execute a single tool and return its result directly.

    Parameters
    ----------
    name : str
        Tool name
    arguments : Dict
        Tool arguments
    tools : Dict
        Tools from FileManager.get_tools()

    Returns
    -------
    Any
        The tool's result

    Raises
    ------
    RuntimeError
        If the tool fails
    """
    results = await execute_tools([{"name": name, "arguments": arguments}], tools)
    r = results[0]
    if r["error"]:
        raise RuntimeError(r["error"])
    return r["result"]
