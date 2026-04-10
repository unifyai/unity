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

from typing import Any, AsyncIterator

import pytest
import pytest_asyncio

from unity.actor.code_act_actor import CodeActActor
from unity.actor.execution import ExecutionResult
from unity.actor.environments import StateManagerEnvironment
from unity.common.async_tool_loop import SteerableToolHandle
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
