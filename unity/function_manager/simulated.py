from __future__ import annotations

import asyncio
import functools
import json
import os
from typing import Any, Dict, List, Optional

import unify

from .base import BaseFunctionManager
from .types.function import Function


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

        guidance = (
            f"\n\nSimulation guidance – prioritise hallucinating functions like these:\n{self._simulation_guidance}"
            if self._simulation_guidance
            else ""
        )

        sys_msg = (
            "You are a simulated function-catalogue assistant. There is no real "
            "storage; invent plausible functions and keep answers self-consistent.\n\n"
            f"Back-story: {self._description}{guidance}\n\n"
            "Function columns available (simulated):\n" + json.dumps(columns)
        )
        self._llm.set_system_message(sys_msg)

    # ------------------------------------------------------------------ #
    #  Internal helper: run async LLM from sync contexts                  #
    # ------------------------------------------------------------------ #
    def _run_async_sync(self, coro) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        import concurrent.futures

        def _runner() -> str:
            return asyncio.run(coro)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_runner).result()

    def _extract_json(self, text: str):
        """Extract JSON from an LLM response (raw, fenced, or embedded)."""
        import re

        # Direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Fenced block
        fenced = re.search(r"```(?:json)?\n([\s\S]*?)\n```", text)
        if fenced:
            candidate = fenced.group(1).strip()
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # First top-level JSON-looking span
        start = min(
            (
                i
                for i in [
                    text.find("{"),
                    text.find("[") if text.find("[") != -1 else 10**9,
                ]
                if i != -1
            ),
            default=-1,
        )
        if start != -1:
            # Heuristic: grab until matching closing brace/bracket
            stack = []
            end = start
            for idx, ch in enumerate(text[start:], start=start):
                if ch in "[{":
                    stack.append(ch)
                elif ch in "]}":
                    if not stack:
                        break
                    opening = stack.pop()
                    if not stack:
                        end = idx + 1
                        break
            if end > start:
                candidate = text[start:end]
                try:
                    return json.loads(candidate)
                except Exception:
                    pass

        raise ValueError("Could not extract JSON from LLM response")

    def _guidance_hint(self) -> str:
        return (self._simulation_guidance or "").strip()

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
        # Ask the stateful LLM to produce a catalogue snapshot aligned to guidance
        guidance = self._guidance_hint()
        prompt = (
            "Simulate FunctionManager.list_functions. Return ONLY a JSON object mapping "
            "function name -> {function_id, argspec, docstring"
            + (", implementation" if include_implementations else "")
            + "}. "
            "Ensure the catalogue reflects the following high-level domain guidance if provided.\n\n"
            f"Guidance: {guidance!r}"
        )

        def _call() -> str:
            return self._run_async_sync(self._llm.generate(prompt))

        raw = _call()
        data = self._extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError(
                "list_functions: expected a JSON object mapping name -> metadata",
            )
        return data

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
        guidance = self._guidance_hint()
        prompt = (
            "Simulate FunctionManager.search_functions. Return ONLY a JSON array of objects "
            "with fields name, function_id, argspec, docstring. "
            f"Limit to {limit} starting at {offset}. "
            f"Filter expression: {filter!r}. "
            "Ensure results reflect the high-level guidance when relevant.\n\n"
            f"Guidance: {guidance!r}"
        )

        def _call() -> str:
            return self._run_async_sync(self._llm.generate(prompt))

        raw = _call()
        data = self._extract_json(raw)
        if not isinstance(data, list):
            raise ValueError(
                "search_functions: expected a JSON array of function records",
            )
        return data

    @functools.wraps(BaseFunctionManager.search_functions_by_similarity, updated=())
    def search_functions_by_similarity(
        self,
        *,
        query: str,
        n: int = 5,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        guidance = self._guidance_hint()
        prompt = (
            "Simulate FunctionManager.search_functions_by_similarity. Given the natural-language query, "
            "invent up to n plausible functions that would exist in the current catalogue. "
            "Return ONLY a JSON array of objects with keys name, function_id, argspec, score. "
            "Bias the inventions to align with the high-level guidance if present.\n\n"
            f"query={query!r}, n={n}, guidance={guidance!r}"
        )

        def _call() -> str:
            return self._run_async_sync(self._llm.generate(prompt))

        raw = _call()
        data = self._extract_json(raw)
        if not isinstance(data, list):
            raise ValueError(
                "search_functions_by_similarity: expected a JSON array of function records with scores",
            )
        return data
