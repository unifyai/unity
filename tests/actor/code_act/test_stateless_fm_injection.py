"""Tests verifying that FM-discovered functions are available in stateless execute_code.

When the LLM discovers functions via FunctionManager discovery tools, those
callables are injected into session 0's namespace. These tests verify that
subsequent stateless execute_code calls can access those functions — the core
behavior introduced by the _fm_keys tracking and inject_globals plumbing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from tests.helpers import _handle_project
from unity.actor.code_act_actor import CodeActActor
from unity.actor.execution import (
    PythonExecutionSession,
    _CURRENT_SANDBOX,
    parts_to_text,
)


def _result_error(res: Any) -> Any:
    if isinstance(res, dict):
        return res.get("error")
    return getattr(res, "error", None)


def _result_stdout_text(res: Any) -> str:
    if isinstance(res, dict):
        stdout = res.get("stdout") or ""
    else:
        stdout = getattr(res, "stdout", "") or ""
    return parts_to_text(stdout) if isinstance(stdout, list) else str(stdout)


class _InjectingFunctionManager:
    """FunctionManager stub whose discovery tools inject a callable into the namespace.

    When search_functions is called with ``_return_callable=True`` and a
    ``_namespace`` dict, it exec's a real function into that namespace and
    returns metadata describing it. This mirrors what the real FM does via
    ``_inject_callables_for_functions``.
    """

    _include_primitives = False
    exclude_primitive_ids: frozenset = frozenset()
    exclude_compositional_ids: frozenset = frozenset()

    _FUNC_IMPL = "def sentinel_func():\n    return 'SENTINEL_OK'"
    _FUNC_META = [{"function_id": 1, "name": "sentinel_func", "docstring": "test fn"}]

    def search_functions(
        self,
        *,
        query: str,
        n: int = 5,
        include_implementations: bool = True,
        _return_callable: bool = False,
        _namespace: Optional[Dict[str, Any]] = None,
        _also_return_metadata: bool = False,
    ) -> Any:
        if _return_callable and _namespace is not None:
            exec(self._FUNC_IMPL, _namespace)
        if _also_return_metadata:
            return {"metadata": list(self._FUNC_META), "callables": {}}
        return list(self._FUNC_META)

    def filter_functions(self, **kwargs) -> Any:
        return self.search_functions(**kwargs)

    def list_functions(self, **kwargs) -> Any:
        return self.search_functions(**kwargs)

    def _get_function_data_by_name(self, *, name: str):
        return None

    def _get_primitive_data_by_name(self, *, name: str):
        return None


@pytest.mark.asyncio
@_handle_project
async def test_stateless_execute_code_can_call_fm_discovered_function():
    """After FM discovery injects a function into session 0, a subsequent
    stateless execute_code call can invoke that function without NameError."""

    fm = _InjectingFunctionManager()
    actor = CodeActActor(environments=[], function_manager=fm)

    tools = actor.get_tools("act")
    execute_code = tools["execute_code"]
    search_fn = tools["FunctionManager_search_functions"]

    sandbox = PythonExecutionSession(environments={}, computer_primitives=None)
    sb_token = _CURRENT_SANDBOX.set(sandbox)
    try:
        # Step 1: FM discovery — injects sentinel_func into sandbox.global_state
        await search_fn(query="sentinel")

        assert (
            "sentinel_func" in sandbox.global_state
        ), "FM discovery should have injected sentinel_func into the sandbox"
        assert (
            "sentinel_func" in sandbox._fm_keys
        ), "sentinel_func should be tracked in _fm_keys"

        # Step 2: stateless execute_code calling the FM-discovered function.
        res = await execute_code(
            thought="call the discovered function in a stateless session",
            code="result = sentinel_func()\nprint(result)",
            language="python",
            state_mode="stateless",
            session_id=None,
            session_name=None,
            venv_id=None,
            _notification_up_q=None,
        )

        assert (
            _result_error(res) is None
        ), f"Stateless execution should succeed but got: {_result_error(res)}"
        assert "SENTINEL_OK" in _result_stdout_text(res)
    finally:
        _CURRENT_SANDBOX.reset(sb_token)


@pytest.mark.asyncio
@_handle_project
async def test_stateless_does_not_inherit_intermediate_variables():
    """Stateless execute_code inherits FM functions but NOT intermediate
    variables created by prior stateful execution in session 0."""

    fm = _InjectingFunctionManager()
    actor = CodeActActor(environments=[], function_manager=fm)

    tools = actor.get_tools("act")
    execute_code = tools["execute_code"]
    search_fn = tools["FunctionManager_search_functions"]

    sandbox = PythonExecutionSession(environments={}, computer_primitives=None)
    sb_token = _CURRENT_SANDBOX.set(sandbox)
    try:
        # Step 1: FM discovery
        await search_fn(query="sentinel")

        # Step 2: stateful execution creates an intermediate variable
        res = await execute_code(
            thought="create an intermediate variable in session 0",
            code="my_intermediate_var = 42",
            language="python",
            state_mode="stateful",
            session_id=0,
            session_name=None,
            venv_id=None,
            _notification_up_q=None,
        )
        assert _result_error(res) is None

        # Verify the variable exists in session 0
        assert sandbox.global_state.get("my_intermediate_var") == 42

        # Step 3: stateless execution — FM function available, intermediate NOT
        res = await execute_code(
            thought="verify isolation: FM function works, intermediate var does not",
            code=(
                "parts = []\n"
                "parts.append(sentinel_func())\n"
                "try:\n"
                "    parts.append(str(my_intermediate_var))\n"
                "except NameError:\n"
                "    parts.append('NOT_FOUND')\n"
                "print('|'.join(parts))"
            ),
            language="python",
            state_mode="stateless",
            session_id=None,
            session_name=None,
            venv_id=None,
            _notification_up_q=None,
        )

        assert (
            _result_error(res) is None
        ), f"Stateless execution should succeed but got: {_result_error(res)}"
        stdout = _result_stdout_text(res)
        assert "SENTINEL_OK" in stdout, "FM function should be callable"
        assert (
            "NOT_FOUND" in stdout
        ), "Intermediate variable should NOT be available in stateless session"
    finally:
        _CURRENT_SANDBOX.reset(sb_token)
