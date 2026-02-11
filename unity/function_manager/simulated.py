from __future__ import annotations

import asyncio
import functools
import json
from typing import Any, Dict, FrozenSet, List, Optional


from .base import BaseFunctionManager
from .types.function import Function
from ..common.simulated import (
    SimulatedLineage,
    simulated_llm_roundtrip,
    maybe_tool_log_scheduled,
    maybe_tool_log_completed,
)
from ..common.llm_client import new_llm_client


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
        filter_scope: Optional[str] = None,
        # Accept but ignore extra parameters for compatibility
        **kwargs: Any,
    ) -> None:
        self._description = description
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance
        self._filter_scope = filter_scope
        self._exclude_primitive_ids: Optional[FrozenSet[int]] = None
        self._exclude_compositional_ids: Optional[FrozenSet[int]] = None

        # One shared, *stateful* LLM for the simulation
        self._llm = new_llm_client(stateful=True)

        self._rebuild_system_message()

    def _rebuild_system_message(self) -> None:
        """Reconstruct the LLM system message from current state."""
        columns = [{k: str(v.annotation)} for k, v in Function.model_fields.items()]

        guidance = (
            f"\n\nSimulation guidance – prioritise hallucinating functions like these:\n{self._simulation_guidance}"
            if self._simulation_guidance
            else ""
        )

        scope_hint = (
            f"\n\nIMPORTANT – This instance has a filter_scope applied: {self._filter_scope!r}. "
            "Every query (list, filter, search) MUST only return functions whose metadata "
            "satisfies this boolean expression. For example, if the scope is "
            "\"language == 'python'\", never include shell/bash functions in results. "
            "Treat the scope as an implicit 'AND' condition on every read query."
            if self._filter_scope
            else ""
        )

        exclusion_hint = ""
        if self._exclude_primitive_ids or self._exclude_compositional_ids:
            parts = []
            if self._exclude_primitive_ids:
                parts.append(
                    f"primitive function_ids {sorted(self._exclude_primitive_ids)}",
                )
            if self._exclude_compositional_ids:
                parts.append(
                    f"compositional function_ids {sorted(self._exclude_compositional_ids)}",
                )
            exclusion_hint = (
                f"\n\nIMPORTANT – The following function IDs are excluded from all "
                f"read queries (they are prompt-injected via the environment and "
                f"must NOT appear in search/list/filter results): {', '.join(parts)}."
            )

        sys_msg = (
            "You are a simulated function-catalogue assistant. There is no real "
            "storage; invent plausible functions and keep answers self-consistent.\n\n"
            f"Back-story: {self._description}{guidance}{scope_hint}{exclusion_hint}\n\n"
            "Function columns available (simulated):\n" + json.dumps(columns)
        )
        self._llm.set_system_message(sys_msg)

    # ------------------------------------------------------------------ #
    #  Properties & setters (same contract as FunctionManager)            #
    # ------------------------------------------------------------------ #

    @property
    def filter_scope(self) -> Optional[str]:
        """A boolean expression applied to all simulated read queries."""
        return self._filter_scope

    @filter_scope.setter
    def filter_scope(self, value: Optional[str]) -> None:
        self._filter_scope = value
        self._rebuild_system_message()

    @property
    def exclude_primitive_ids(self) -> Optional[FrozenSet[int]]:
        """Primitive function IDs excluded from simulated queries."""
        return self._exclude_primitive_ids

    @exclude_primitive_ids.setter
    def exclude_primitive_ids(self, value: Optional[FrozenSet[int]]) -> None:
        self._exclude_primitive_ids = frozenset(value) if value else None
        self._rebuild_system_message()

    @property
    def exclude_compositional_ids(self) -> Optional[FrozenSet[int]]:
        """Compositional function IDs excluded from simulated queries."""
        return self._exclude_compositional_ids

    @exclude_compositional_ids.setter
    def exclude_compositional_ids(self, value: Optional[FrozenSet[int]]) -> None:
        self._exclude_compositional_ids = frozenset(value) if value else None
        self._rebuild_system_message()

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
        raise_on_error: bool = True,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        sched = maybe_tool_log_scheduled(
            "SimulatedFunctionManager.add_functions",
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
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "add_functions",
                {"total": len(results)},
                t0,
            )
        return results

    @functools.wraps(BaseFunctionManager.list_functions, updated=())
    def list_functions(
        self,
        *,
        include_implementations: bool = False,
        return_callable: bool = False,
        namespace: Optional[Dict[str, Any]] = None,
        also_return_metadata: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if also_return_metadata and not return_callable:
            raise ValueError("also_return_metadata requires return_callable=True")
        if return_callable and namespace is None:
            raise ValueError("namespace required when return_callable=True")

        # Ask the stateful LLM to produce a catalogue snapshot aligned to guidance
        guidance = self._guidance_hint()
        sched = maybe_tool_log_scheduled(
            "SimulatedFunctionManager.list_functions",
            "list_functions",
            {"include_implementations": include_implementations},
        )
        scope_clause = (
            f"\nScope constraint (MUST be satisfied by every returned function): {self._filter_scope!r}"
            if self._filter_scope
            else ""
        )
        prompt = (
            "Simulate FunctionManager.list_functions. Return ONLY a JSON object mapping "
            "function name -> {function_id, argspec, docstring"
            + (", implementation" if include_implementations else "")
            + "}. "
            "Ensure the catalogue reflects the following high-level domain guidance if provided.\n\n"
            f"Guidance: {guidance!r}{scope_clause}"
        )

        def _call() -> str:
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            # Prefer scheduled label when available for consistent logging
            label = (
                sched[0]
                if sched is not None and isinstance(sched, tuple) and len(sched) >= 1
                else SimulatedLineage.make_label(
                    "SimulatedFunctionManager.list_functions",
                )
            )
            return self._run_async_sync(
                simulated_llm_roundtrip(
                    self._llm,
                    label=label,
                    prompt=prompt,
                ),
            )

        raw = _call()
        data = self._extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError(
                "list_functions: expected a JSON object mapping name -> metadata",
            )
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "list_functions",
                {"total": len(data)},
                t0,
            )

        if not return_callable:
            return data

        assert namespace is not None  # validated above

        def _make_stub(fn_name: str):
            async def _stub(*args, **kwargs):
                return {
                    "simulated": True,
                    "name": fn_name,
                    "args": list(args),
                    "kwargs": kwargs,
                }

            _stub.__name__ = fn_name
            return _stub

        callables_map: Dict[str, Any] = {}
        for fn_name in list(data.keys()):
            if not isinstance(fn_name, str):
                continue
            cb = _make_stub(fn_name)
            namespace[fn_name] = cb
            callables_map[fn_name] = cb

        if also_return_metadata:
            return {"callables": callables_map, "metadata": data}  # type: ignore[return-value]

        return callables_map  # type: ignore[return-value]

    @functools.wraps(BaseFunctionManager.get_precondition, updated=())
    def get_precondition(
        self,
        *,
        function_name: str,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        sched = maybe_tool_log_scheduled(
            "SimulatedFunctionManager.get_precondition",
            "get_precondition",
            {"function_name": function_name},
        )
        # Simulate that no explicit preconditions are stored
        result = None
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "get_precondition",
                {"present": result is not None},
                t0,
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
        sched = maybe_tool_log_scheduled(
            "SimulatedFunctionManager.delete_function",
            "delete_function",
            {"function_id": function_id, "delete_dependents": delete_dependents},
        )
        # Acknowledge deletion without side effects
        result = {f"id={function_id}": "deleted (simulated)"}
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(label, cid, "delete_function", result, t0)
        return result

    @functools.wraps(BaseFunctionManager.filter_functions, updated=())
    def filter_functions(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        include_implementations: bool = True,
        return_callable: bool = False,
        namespace: Optional[Dict[str, Any]] = None,
        also_return_metadata: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if also_return_metadata and not return_callable:
            raise ValueError("also_return_metadata requires return_callable=True")
        if return_callable and namespace is None:
            raise ValueError("namespace required when return_callable=True")

        guidance = self._guidance_hint()
        sched = maybe_tool_log_scheduled(
            "SimulatedFunctionManager.filter_functions",
            "filter_functions",
            {"filter": filter, "offset": offset, "limit": limit},
        )
        scope_clause = (
            f"\nScope constraint (MUST be satisfied by every returned function): {self._filter_scope!r}"
            if self._filter_scope
            else ""
        )
        prompt = (
            "Simulate FunctionManager.filter_functions. Return ONLY a JSON array of objects "
            "with fields name, function_id, argspec, docstring. "
            f"Limit to {limit} starting at {offset}. "
            f"Filter expression: {filter!r}. "
            "Ensure results reflect the high-level guidance when relevant.\n\n"
            f"Guidance: {guidance!r}{scope_clause}"
        )

        def _call() -> str:
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            label = (
                sched[0]
                if sched is not None and isinstance(sched, tuple) and len(sched) >= 1
                else SimulatedLineage.make_label(
                    "SimulatedFunctionManager.filter_functions",
                )
            )
            return self._run_async_sync(
                simulated_llm_roundtrip(
                    self._llm,
                    label=label,
                    prompt=prompt,
                ),
            )

        raw = _call()
        data = self._extract_json(raw)
        if not isinstance(data, list):
            raise ValueError(
                "filter_functions: expected a JSON array of function records",
            )
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "filter_functions",
                {"total": len(data)},
                t0,
            )

        # Strip implementations if not requested
        if not include_implementations:
            data = [
                {k: v for k, v in rec.items() if k != "implementation"}
                for rec in data
                if isinstance(rec, dict)
            ]

        if not return_callable:
            return data

        assert namespace is not None  # validated above

        def _make_stub(fn_name: str):
            async def _stub(*args, **kwargs):
                return {
                    "simulated": True,
                    "name": fn_name,
                    "args": list(args),
                    "kwargs": kwargs,
                }

            _stub.__name__ = fn_name
            return _stub

        callables_list: List[Any] = []
        for rec in data:
            fn_name = rec.get("name") if isinstance(rec, dict) else None
            if not isinstance(fn_name, str):
                continue
            cb = _make_stub(fn_name)
            namespace[fn_name] = cb
            callables_list.append(cb)

        if also_return_metadata:
            return {"callables": callables_list, "metadata": data}  # type: ignore[return-value]

        return callables_list  # type: ignore[return-value]

    @functools.wraps(BaseFunctionManager.search_functions, updated=())
    def search_functions(
        self,
        *,
        query: str,
        n: int = 5,
        include_implementations: bool = True,
        return_callable: bool = False,
        namespace: Optional[Dict[str, Any]] = None,
        also_return_metadata: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if also_return_metadata and not return_callable:
            raise ValueError("also_return_metadata requires return_callable=True")
        if return_callable and namespace is None:
            raise ValueError("namespace required when return_callable=True")

        guidance = self._guidance_hint()
        sched = maybe_tool_log_scheduled(
            "SimulatedFunctionManager.search_functions",
            "search_functions",
            {"query": query, "n": n},
        )
        scope_clause = (
            f"\nScope constraint (MUST be satisfied by every returned function): {self._filter_scope!r}"
            if self._filter_scope
            else ""
        )
        prompt = (
            "Simulate FunctionManager.search_functions. Given the natural-language query, "
            "invent up to n plausible functions that would exist in the current catalogue. "
            "Return ONLY a JSON array of objects with keys name, function_id, argspec, score. "
            "Bias the inventions to align with the high-level guidance if present.\n\n"
            f"query={query!r}, n={n}, guidance={guidance!r}{scope_clause}"
        )

        def _call() -> str:
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            label = (
                sched[0]
                if sched is not None and isinstance(sched, tuple) and len(sched) >= 1
                else SimulatedLineage.make_label(
                    "SimulatedFunctionManager.search_functions",
                )
            )
            return self._run_async_sync(
                simulated_llm_roundtrip(
                    self._llm,
                    label=label,
                    prompt=prompt,
                ),
            )

        raw = _call()
        data = self._extract_json(raw)
        if not isinstance(data, list):
            raise ValueError(
                "search_functions: expected a JSON array of function records with scores",
            )
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "search_functions",
                {"total": len(data)},
                t0,
            )

        # Strip implementations if not requested
        if not include_implementations:
            data = [
                {k: v for k, v in rec.items() if k != "implementation"}
                for rec in data
                if isinstance(rec, dict)
            ]

        if not return_callable:
            return data

        assert namespace is not None  # validated above

        def _make_stub(fn_name: str):
            async def _stub(*args, **kwargs):
                return {
                    "simulated": True,
                    "name": fn_name,
                    "args": list(args),
                    "kwargs": kwargs,
                }

            _stub.__name__ = fn_name
            return _stub

        callables_list: List[Any] = []
        for rec in data:
            fn_name = rec.get("name") if isinstance(rec, dict) else None
            if not isinstance(fn_name, str):
                continue
            cb = _make_stub(fn_name)
            namespace[fn_name] = cb
            callables_list.append(cb)

        if also_return_metadata:
            return {"callables": callables_list, "metadata": data}  # type: ignore[return-value]

        return callables_list  # type: ignore[return-value]

    @functools.wraps(BaseFunctionManager.execute_function, updated=())
    async def execute_function(
        self,
        *,
        function_name: str,
        call_kwargs: Optional[Dict[str, Any]] = None,
        target_venv_id: Optional[int] = ...,
        state_mode: str = "stateless",
        session_id: int = 0,
        venv_pool: Optional[Any] = None,
        extra_namespaces: Optional[Dict[str, Any]] = None,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        sched = maybe_tool_log_scheduled(
            "SimulatedFunctionManager.execute_function",
            "execute_function",
            {
                "function_name": function_name,
                "call_kwargs": call_kwargs,
                "state_mode": state_mode,
            },
        )
        # Simulate execution without actually running any code
        result = {
            "result": {
                "simulated": True,
                "function_name": function_name,
                "call_kwargs": call_kwargs or {},
            },
            "error": None,
            "stdout": "",
            "stderr": "",
        }
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(
                label,
                cid,
                "execute_function",
                {"function_name": function_name, "success": True},
                t0,
            )
        return result

    @functools.wraps(BaseFunctionManager.clear, updated=())
    def clear(self) -> None:
        sched = maybe_tool_log_scheduled(
            "SimulatedFunctionManager.clear",
            "clear",
            {},
        )
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
            filter_scope=getattr(self, "_filter_scope", None),
        )
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(label, cid, "clear", {"outcome": "reset"}, t0)
