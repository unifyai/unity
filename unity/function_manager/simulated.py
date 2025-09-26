from __future__ import annotations

import asyncio
import functools
import json
import os
from typing import Any, Dict, List, Optional

import unify

from ..common.async_tool_loop import SteerableToolHandle
from .base import BaseFunctionManager
from .types.function import Function


class _SimulatedFunctionHandle(SteerableToolHandle):
    def __init__(
        self,
        llm: unify.Unify,
        prompt: str,
        *,
        _return_reasoning_steps: bool,
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        self._want_steps = _return_reasoning_steps
        self._answer: Optional[str] = None
        self._msgs: List[Dict[str, Any]] = []
        self._done = False

    async def result(self):
        if not self._done:
            answer = await self._llm.generate(self._prompt)
            self._answer = answer
            self._msgs = [
                {"role": "user", "content": self._prompt},
                {"role": "assistant", "content": answer},
            ]
            self._done = True
        if self._want_steps:
            return self._answer, self._msgs
        return self._answer


class SimulatedFunctionManager(BaseFunctionManager):
    """
    Simulated function catalogue that never touches storage.

    Uses a single stateful LLM to fabricate plausible responses for listing,
    searching and similarity queries. Mutation endpoints acknowledge requests
    without persisting any state.
    """

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
        *,
        log_events: bool = False,
        rolling_summary_in_prompts: bool = True,
        simulation_guidance: Optional[str] = None,
    ) -> None:
        self._description = description
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance

        # One shared, *stateful* LLM for the simulation
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )

        columns = [{k: str(v.annotation)} for k, v in Function.model_fields.items()]

        sys_msg = (
            "You are a simulated function-catalogue assistant. There is no real "
            "storage; invent plausible functions and keep answers self-consistent.\n\n"
            f"Back-story: {self._description}\n\n"
            "Function columns available (simulated):\n" + json.dumps(columns)
        )
        self._llm.set_system_message(sys_msg)

    # ------------------------------------------------------------------ #
    # Public API – mirror BaseFunctionManager                            #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseFunctionManager.add_functions, updated=())
    def add_functions(
        self,
        *,
        implementations: str | List[str],
        preconditions: Optional[Dict[str, Dict]] = None,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        # No persistence – acknowledge each function name with a simulated status
        if isinstance(implementations, str):
            implementations = [implementations]
        results: Dict[str, str] = {}
        for src in implementations:
            # naive extract of a def name for acknowledgement
            name = "function"
            try:
                first = next(
                    line for line in src.splitlines() if line.strip().startswith("def ")
                )
                name = first.split("def ", 1)[1].split("(", 1)[0].strip()
            except Exception:
                pass
            results[name] = "added (simulated)"
        return results

    @functools.wraps(BaseFunctionManager.list_functions, updated=())
    def list_functions(
        self,
        *,
        include_implementations: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        # Produce a small fabricated list deterministically
        fake: Dict[str, Dict[str, Any]] = {
            "example": {
                "function_id": 1,
                "argspec": "(x: int, y: int) -> int",
                "docstring": "Add two integers (simulated)",
            },
        }
        if include_implementations:
            fake["example"][
                "implementation"
            ] = "def example(x: int, y: int) -> int:\n    return x + y\n"
        return fake

    @functools.wraps(BaseFunctionManager.get_precondition, updated=())
    def get_precondition(
        self,
        *,
        function_name: str,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        # Simulate that no explicit preconditions are stored
        return None

    @functools.wraps(BaseFunctionManager.delete_function, updated=())
    def delete_function(
        self,
        *,
        function_id: int,
        delete_dependents: bool = True,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        # Acknowledge deletion without side effects
        return {f"id={function_id}": "deleted (simulated)"}

    @functools.wraps(BaseFunctionManager.search_functions, updated=())
    def search_functions(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        # Query the LLM to fabricate a list of functions
        prompt = (
            "Simulate FunctionManager.search_functions. Return a JSON array of objects "
            "with fields name, function_id, argspec, docstring. Limit to "
            f"{limit} starting at {offset}. Filter expression (for flavor): {filter!r}."
        )

        async def _call() -> str:
            return await self._llm.generate(prompt)

        try:
            raw = asyncio.run(_call())
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        # Fallback minimal response
        return [
            {
                "name": "example",
                "function_id": 1,
                "argspec": "(x: int, y: int) -> int",
                "docstring": "Add two integers (simulated)",
            },
        ]

    @functools.wraps(BaseFunctionManager.search_functions_by_similarity, updated=())
    def search_functions_by_similarity(
        self,
        *,
        query: str,
        n: int = 5,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        prompt = (
            "Simulate FunctionManager.search_functions_by_similarity. Given the query, "
            "invent up to n plausible functions that might be similar. Return ONLY a JSON array "
            "of objects with keys name, function_id, argspec, score.\n\n"
            f"query={query!r}, n={n}"
        )

        async def _call() -> str:
            return await self._llm.generate(prompt)

        try:
            raw = asyncio.run(_call())
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return [
            {
                "name": "example",
                "function_id": 1,
                "argspec": "(x: int, y: int) -> int",
                "score": 0.12,
            },
        ]
