"""
Tests for execute_function bare-handle unwrapping.

When execute_function calls a primitive that returns a SteerableToolHandle
with no meaningful side output (no stdout, no stderr, no error), it should
return the handle directly instead of wrapping it in an ExecutionResult dict.

This ensures the core loop adopts the handle via the bare-handle path
(adopt_nested, no intermediate LLM turn) rather than the composite path
(adopt_multi_nested, wasteful LLM turn that always calls wait()).
"""

from __future__ import annotations

import inspect
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from unity.actor.code_act_actor import CodeActActor
from unity.actor.execution import ExecutionResult
from unity.actor.environments import StateManagerEnvironment
from unity.common.async_tool_loop import SteerableToolHandle
from unity.common.llm_helpers import method_to_schema
from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.primitives import Primitives, PrimitiveScope
from unity.manager_registry import ManagerRegistry

pytestmark = pytest.mark.llm_call

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def configure_simulated_managers(monkeypatch: pytest.MonkeyPatch) -> None:
    from unity.settings import SETTINGS

    for name in ("CONTACT", "TASK", "TRANSCRIPT", "KNOWLEDGE", "GUIDANCE", "WEB"):
        monkeypatch.setenv(f"UNITY_{name}_IMPL", "simulated")
        attr = name.lower()
        if hasattr(SETTINGS, attr):
            monkeypatch.setattr(
                getattr(SETTINGS, attr),
                "IMPL",
                "simulated",
                raising=False,
            )

    for name in ("GUIDANCE", "WEB", "KNOWLEDGE"):
        monkeypatch.setenv(f"UNITY_{name}_ENABLED", "true")
        attr = name.lower()
        if hasattr(SETTINGS, attr):
            monkeypatch.setattr(
                getattr(SETTINGS, attr),
                "ENABLED",
                True,
                raising=False,
            )

    ManagerRegistry.clear()


@pytest_asyncio.fixture
async def execute_function_tool(
    configure_simulated_managers,
) -> AsyncIterator[Any]:
    """Yield the execute_function tool closure from a CodeActActor."""
    scope = PrimitiveScope(scoped_managers=frozenset({"contacts"}))
    primitives = Primitives(primitive_scope=scope)
    env = StateManagerEnvironment(primitives)
    fm = FunctionManager()
    actor = CodeActActor(environments=[env], function_manager=fm)

    tools = actor.get_tools("act")
    assert (
        "execute_function" in tools
    ), f"execute_function not found in tools: {list(tools.keys())}"
    fn = tools["execute_function"]
    if hasattr(fn, "fn"):
        fn = fn.fn

    try:
        yield fn
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class _FakeFunctionManager:
    def __init__(self):
        self.execute_in_venv = AsyncMock(
            return_value={
                "stdout": [],
                "stderr": [],
                "result": "venv ok",
                "error": None,
            },
        )

    def _get_function_data_by_name(self, *, name: str):
        if name != "stored_report":
            return None
        return {
            "function_id": 12,
            "name": "stored_report",
            "implementation": "def stored_report():\n    return 'default env'",
            "venv_id": 31,
            "is_primitive": False,
        }

    def search_functions(self, **kwargs):
        return {"metadata": []}

    def filter_functions(self, **kwargs):
        return {"metadata": []}

    def list_functions(self, **kwargs):
        return {"metadata": []}

    async def add_functions(self, **kwargs):
        return {"metadata": []}

    async def delete_function(self, **kwargs):
        return {"deleted": True}


@pytest.mark.asyncio
async def test_execute_function_does_not_expose_venv_id():
    fm = _FakeFunctionManager()
    actor = CodeActActor(
        function_manager=fm,  # type: ignore[arg-type]
        can_store=False,
    )

    try:
        execute_function = actor.get_tools("act")["execute_function"]
        if hasattr(execute_function, "fn"):
            execute_function = execute_function.fn

        signature = inspect.signature(execute_function)
        schema = method_to_schema(
            execute_function,
            tool_name="execute_function",
            include_class_name=False,
        )
    finally:
        await actor.close()

    assert "venv_id" not in signature.parameters
    assert "venv_id" not in schema["function"]["parameters"]["properties"]
    assert "venv_id" not in schema["function"]["description"]


@pytest.mark.asyncio
async def test_execute_function_uses_stored_venv_when_caller_omits_it():
    fm = _FakeFunctionManager()
    actor = CodeActActor(
        function_manager=fm,  # type: ignore[arg-type]
        can_store=False,
    )
    captured: dict[str, object] = {}

    async def _fake_execute(**kwargs):
        captured.update(kwargs)
        return {
            "stdout": [],
            "stderr": [],
            "result": "venv ok",
            "error": None,
            "language": kwargs["language"],
            "state_mode": kwargs["state_mode"],
            "session_id": kwargs["session_id"],
            "venv_id": kwargs["venv_id"],
            "session_created": False,
            "duration_ms": 0,
        }

    actor._session_executor.execute = AsyncMock(side_effect=_fake_execute)  # type: ignore[method-assign]

    try:
        execute_function = actor.get_tools("act")["execute_function"]
        if hasattr(execute_function, "fn"):
            execute_function = execute_function.fn

        result = await execute_function(
            function_name="stored_report",
            call_kwargs={},
        )
    finally:
        await actor.close()

    assert result.result == "venv ok"
    assert captured["venv_id"] == 31


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_execute_function_returns_bare_handle_for_primitive(
    execute_function_tool,
):
    """execute_function should return a bare SteerableToolHandle when
    the primitive produces no side output (stdout/stderr/error)."""
    result = await execute_function_tool(
        function_name="primitives.contacts.ask",
        call_kwargs={"text": "Who are my contacts?"},
    )

    assert isinstance(
        result,
        SteerableToolHandle,
    ), f"Expected bare SteerableToolHandle, got {type(result).__name__}: {result!r}"

    # Clean up the running handle.
    try:
        await result.stop("test cleanup")
    except Exception:
        pass


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_execute_function_returns_composite_when_side_output_present(
    execute_function_tool,
):
    """execute_function should return the full ExecutionResult when
    the execution produces stdout alongside the handle."""
    # A composed function that prints AND returns a handle — the print
    # output is meaningful intermediate content the LLM should observe.
    from unity.function_manager.function_manager import FunctionManager
    from unity.common.context_registry import ContextRegistry

    ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
    ContextRegistry.forget(FunctionManager, "Functions/Compositional")
    ContextRegistry.forget(FunctionManager, "Functions/Primitives")
    ContextRegistry.forget(FunctionManager, "Functions/Meta")

    fm = FunctionManager()
    fm.add_functions(
        implementations="""
async def ask_with_log(text: str):
    print("About to query contacts...")
    handle = await primitives.contacts.ask(text=text)
    return handle
""".strip(),
    )

    scope = PrimitiveScope(scoped_managers=frozenset({"contacts"}))
    primitives = Primitives(primitive_scope=scope)
    env = StateManagerEnvironment(primitives)
    actor = CodeActActor(environments=[env], function_manager=fm)

    try:
        tools = actor.get_tools("act")
        fn = tools["execute_function"]
        if hasattr(fn, "fn"):
            fn = fn.fn

        result = await fn(
            function_name="ask_with_log",
            call_kwargs={"text": "Who are my contacts?"},
        )

        # Composed function runs through the sandbox → stdout is captured
        # → result should be an ExecutionResult (not a bare handle).
        assert isinstance(result, (dict, ExecutionResult)), (
            f"Expected dict/ExecutionResult when stdout is present, "
            f"got {type(result).__name__}"
        )

        inner_result = (
            result.get("result")
            if isinstance(result, dict)
            else getattr(result, "result", None)
        )
        assert isinstance(inner_result, SteerableToolHandle), (
            f"Expected inner result to be SteerableToolHandle, "
            f"got {type(inner_result).__name__}"
        )

        try:
            await inner_result.stop("test cleanup")
        except Exception:
            pass
    finally:
        try:
            await actor.close()
        except Exception:
            pass
