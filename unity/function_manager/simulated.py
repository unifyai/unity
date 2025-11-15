from __future__ import annotations

import asyncio
import functools
import json
import os
from typing import Any, Dict, List, Optional

import unify

from .base import BaseFunctionManager
from .types.function import Function
from ..common.simulated import (
    SimulatedLineage,
    simulated_llm_roundtrip,
)
from ..constants import LOGGER
import time


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
            "gpt-5@openai",
            reasoning_effort="high",
            service_tier="priority",
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
    #  Tool-call style logging helpers (only when no parent lineage)     #
    # ------------------------------------------------------------------ #
    def _tool_log_start(self, method: str, args: dict) -> tuple[str, str, float, bool]:
        label = SimulatedLineage.make_label(f"SimulatedFunctionManager.{method}")
        call_id = SimulatedLineage.extract_suffix(label) or ""
        t0 = time.perf_counter()
        log_enabled = not SimulatedLineage.has_outer()
        if log_enabled:
            try:
                LOGGER.info(
                    f"🛠️ [{label}] ToolCall Scheduled: {method} - {call_id} | args={json.dumps(args)}",
                )
            except Exception:
                pass
        return label, call_id, t0, log_enabled

    def _tool_log_end(
        self,
        method: str,
        *,
        label: str,
        call_id: str,
        t0: float,
        result_summary: dict,
        enabled: bool,
    ) -> None:
        if not enabled:
            return
        try:
            dt = time.perf_counter() - t0
            LOGGER.info(
                f"✅ [{label}] ToolCall Completed in {dt:.2f}s: {method} - {call_id} | result={json.dumps(result_summary)}",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Public API – mirror BaseFunctionManager                            #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseFunctionManager.add_functions, updated=())
    def add_functions(
        self,
        *,
        implementations: str | List[str],
        preconditions: Optional[Dict[str, Dict]] = None,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        label, call_id, t0, _log_tools = self._tool_log_start(
            "add_functions",
            {
                "implementations_count": (
                    len(implementations) if isinstance(implementations, list) else 1
                ),
                "has_preconditions": bool(preconditions),
            },
        )
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
        self._tool_log_end(
            "add_functions",
            label=label,
            call_id=call_id,
            t0=t0,
            result_summary={"total": len(results)},
            enabled=_log_tools,
        )
        return results

    @functools.wraps(BaseFunctionManager.list_functions, updated=())
    def list_functions(
        self,
        *,
        include_implementations: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        # Ask the stateful LLM to produce a catalogue snapshot aligned to guidance
        guidance = self._guidance_hint()
        label, call_id, t0, _log_tools = self._tool_log_start(
            "list_functions",
            {"include_implementations": include_implementations},
        )
        prompt = (
            "Simulate FunctionManager.list_functions. Return ONLY a JSON object mapping "
            "function name -> {function_id, argspec, docstring"
            + (", implementation" if include_implementations else "")
            + "}. "
            "Ensure the catalogue reflects the following high-level domain guidance if provided.\n\n"
            f"Guidance: {guidance!r}"
        )

        def _call() -> str:
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            return self._run_async_sync(
                simulated_llm_roundtrip(
                    self._llm,
                    label=label,
                    prompt=prompt,
                    sys_for_dump=sys_msg,
                    request_dump_body={
                        "model": getattr(self._llm, "model", None),
                        "messages": [{"role": "user", "content": prompt}],
                    },
                ),
            )

        raw = _call()
        data = self._extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError(
                "list_functions: expected a JSON object mapping name -> metadata",
            )
        self._tool_log_end(
            "list_functions",
            label=label,
            call_id=call_id,
            t0=t0,
            result_summary={"total": len(data)},
            enabled=_log_tools,
        )
        return data

    @functools.wraps(BaseFunctionManager.get_precondition, updated=())
    def get_precondition(
        self,
        *,
        function_name: str,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        label, call_id, t0, _log_tools = self._tool_log_start(
            "get_precondition",
            {"function_name": function_name},
        )
        # Simulate that no explicit preconditions are stored
        result = None
        self._tool_log_end(
            "get_precondition",
            label=label,
            call_id=call_id,
            t0=t0,
            result_summary={"present": result is not None},
            enabled=_log_tools,
        )
        return result

    @functools.wraps(BaseFunctionManager.delete_function, updated=())
    def delete_function(
        self,
        *,
        function_id: int,
        delete_dependents: bool = True,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        label, call_id, t0, _log_tools = self._tool_log_start(
            "delete_function",
            {"function_id": function_id, "delete_dependents": delete_dependents},
        )
        # Acknowledge deletion without side effects
        result = {f"id={function_id}": "deleted (simulated)"}
        self._tool_log_end(
            "delete_function",
            label=label,
            call_id=call_id,
            t0=t0,
            result_summary=result,
            enabled=_log_tools,
        )
        return result

    @functools.wraps(BaseFunctionManager.search_functions, updated=())
    def search_functions(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        guidance = self._guidance_hint()
        label, call_id, t0, _log_tools = self._tool_log_start(
            "search_functions",
            {"filter": filter, "offset": offset, "limit": limit},
        )
        prompt = (
            "Simulate FunctionManager.search_functions. Return ONLY a JSON array of objects "
            "with fields name, function_id, argspec, docstring. "
            f"Limit to {limit} starting at {offset}. "
            f"Filter expression: {filter!r}. "
            "Ensure results reflect the high-level guidance when relevant.\n\n"
            f"Guidance: {guidance!r}"
        )

        def _call() -> str:
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            return self._run_async_sync(
                simulated_llm_roundtrip(
                    self._llm,
                    label=label,
                    prompt=prompt,
                    sys_for_dump=sys_msg,
                    request_dump_body={
                        "model": getattr(self._llm, "model", None),
                        "messages": [{"role": "user", "content": prompt}],
                    },
                ),
            )

        raw = _call()
        data = self._extract_json(raw)
        if not isinstance(data, list):
            raise ValueError(
                "search_functions: expected a JSON array of function records",
            )
        self._tool_log_end(
            "search_functions",
            label=label,
            call_id=call_id,
            t0=t0,
            result_summary={"total": len(data)},
            enabled=_log_tools,
        )
        return data

    @functools.wraps(BaseFunctionManager.search_functions_by_similarity, updated=())
    def search_functions_by_similarity(
        self,
        *,
        query: str,
        n: int = 5,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        guidance = self._guidance_hint()
        label, call_id, t0, _log_tools = self._tool_log_start(
            "search_functions_by_similarity",
            {"query": query, "n": n},
        )
        prompt = (
            "Simulate FunctionManager.search_functions_by_similarity. Given the natural-language query, "
            "invent up to n plausible functions that would exist in the current catalogue. "
            "Return ONLY a JSON array of objects with keys name, function_id, argspec, score. "
            "Bias the inventions to align with the high-level guidance if present.\n\n"
            f"query={query!r}, n={n}, guidance={guidance!r}"
        )

        def _call() -> str:
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            return self._run_async_sync(
                simulated_llm_roundtrip(
                    self._llm,
                    label=label,
                    prompt=prompt,
                    sys_for_dump=sys_msg,
                    request_dump_body={
                        "model": getattr(self._llm, "model", None),
                        "messages": [{"role": "user", "content": prompt}],
                    },
                ),
            )

        raw = _call()
        data = self._extract_json(raw)
        if not isinstance(data, list):
            raise ValueError(
                "search_functions_by_similarity: expected a JSON array of function records with scores",
            )
        self._tool_log_end(
            "search_functions_by_similarity",
            label=label,
            call_id=call_id,
            t0=t0,
            result_summary={"total": len(data)},
            enabled=_log_tools,
        )
        return data

    @functools.wraps(BaseFunctionManager.clear, updated=())
    def clear(self) -> None:
        label, call_id, t0, _log_tools = self._tool_log_start("clear", {})
        type(self).__init__(
            self,
            description=getattr(
                self,
                "_description",
                "nothing fixed, make up some imaginary scenario",
            ),
            log_events=getattr(self, "_log_events", False),
            rolling_summary_in_prompts=getattr(
                self,
                "_rolling_summary_in_prompts",
                True,
            ),
            simulation_guidance=getattr(self, "_simulation_guidance", None),
        )
        self._tool_log_end(
            "clear",
            label=label,
            call_id=call_id,
            t0=t0,
            result_summary={"outcome": "reset"},
            enabled=_log_tools,
        )
