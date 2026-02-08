"""
Tests for steerable compositional functions.

Verifies that compositional functions can create and return steerable handles,
and that runtime detection via isinstance(result, SteerableToolHandle) works correctly.
"""

import asyncio

import pytest

from unity.common.async_tool_loop import SteerableToolHandle
from unity.function_manager.execution_env import create_execution_globals

# ────────────────────────────────────────────────────────────────────────────
# Runtime Detection Tests
# ────────────────────────────────────────────────────────────────────────────


def test_steerable_handle_runtime_detection():
    """Runtime isinstance check should correctly identify SteerableToolHandle subclasses."""
    globals_dict = create_execution_globals()

    assert globals_dict["SteerableToolHandle"] is SteerableToolHandle


def test_steerable_detection_with_non_handle():
    """Non-handle return values should not be detected as steerable."""
    # Plain values should not be steerable
    assert not isinstance("hello", SteerableToolHandle)
    assert not isinstance(42, SteerableToolHandle)
    assert not isinstance({"key": "value"}, SteerableToolHandle)
    assert not isinstance(None, SteerableToolHandle)

    # Async functions returning plain values are not steerable
    async def plain_function():
        return "plain result"

    result = asyncio.run(plain_function())
    assert not isinstance(result, SteerableToolHandle)


# ────────────────────────────────────────────────────────────────────────────
# Compositional Function Tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compositional_function_can_create_steerable_handle():
    """A compositional function should be able to create and return a SteerableToolHandle."""
    globals_dict = create_execution_globals()

    # Define a compositional function that creates a steerable handle
    code = """
async def my_steerable_workflow(goal: str):
    \"\"\"A steerable workflow that uses an async tool loop.\"\"\"
    client = new_llm_client()
    client.set_system_message("You are a helpful assistant. Respond briefly.")

    handle = start_async_tool_loop(
        client=client,
        message=goal,
        tools={},
        loop_id="test-steerable-workflow",
        timeout=30,
    )
    return handle
"""
    exec(code, globals_dict)

    # Call the function
    handle = await globals_dict["my_steerable_workflow"]("Say hello")

    assert isinstance(handle, SteerableToolHandle)

    # Clean up - stop the handle
    await handle.stop("test cleanup")
    try:
        await asyncio.wait_for(handle.result(), timeout=5.0)
    except Exception:
        pass  # May error due to early stop, that's fine


@pytest.mark.asyncio
async def test_steerable_handle_stop_method():
    """A steerable handle's stop() method should work correctly."""
    globals_dict = create_execution_globals()

    code = """
async def create_handle():
    client = new_llm_client()
    client.set_system_message("You are helpful.")
    return start_async_tool_loop(
        client=client,
        message="Count to 10 slowly",
        tools={},
        loop_id="test-stop",
        timeout=60,
    )
"""
    exec(code, globals_dict)

    handle = await globals_dict["create_handle"]()
    assert isinstance(handle, SteerableToolHandle)

    # Stop the handle
    handle.stop("stopping early")

    # Result should complete (possibly with cancellation message)
    result = await asyncio.wait_for(handle.result(), timeout=10.0)
    assert result is not None  # Should return something (even if cancelled)


@pytest.mark.asyncio
async def test_steerable_handle_result_method():
    """A steerable handle's result() method should return the final result."""
    globals_dict = create_execution_globals()

    code = """
async def create_simple_handle():
    client = new_llm_client()
    client.set_system_message("You are helpful. Be very brief - one word answers only.")
    return start_async_tool_loop(
        client=client,
        message="Say only the word 'done'",
        tools={},
        loop_id="test-result",
        timeout=30,
    )
"""
    exec(code, globals_dict)

    handle = await globals_dict["create_simple_handle"]()
    assert isinstance(handle, SteerableToolHandle)

    # Wait for result with timeout
    result = await asyncio.wait_for(handle.result(), timeout=30.0)
    assert result is not None
    assert isinstance(result, str)


# ────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ────────────────────────────────────────────────────────────────────────────


def test_non_steerable_function_returns_plain_value():
    """A non-steerable function should return plain values, not handles."""
    globals_dict = create_execution_globals()

    code = """
async def plain_workflow(x: int, y: int) -> int:
    return x + y
"""
    exec(code, globals_dict)

    result = asyncio.run(globals_dict["plain_workflow"](2, 3))

    assert not isinstance(result, SteerableToolHandle)
    assert result == 5


def test_steerable_detection_pattern():
    """The pattern for detecting steerable results should be straightforward."""
    globals_dict = create_execution_globals()

    # This is the pattern that SingleFunctionActor will use
    def is_steerable(result):
        """Check if a result is a steerable handle."""
        return isinstance(result, globals_dict["SteerableToolHandle"])

    # Test with various values
    assert not is_steerable("string")
    assert not is_steerable(123)
    assert not is_steerable({"dict": "value"})
    assert not is_steerable(None)
    assert not is_steerable([1, 2, 3])


@pytest.mark.asyncio
async def test_compositional_function_with_type_annotation():
    """Compositional function with SteerableToolHandle return type annotation should work."""
    globals_dict = create_execution_globals()

    # Function with explicit return type annotation
    code = """
from typing import Optional

async def typed_steerable_workflow(goal: str) -> SteerableToolHandle:
    \"\"\"A typed steerable workflow.\"\"\"
    client = new_llm_client()
    client.set_system_message("Be brief.")

    handle = start_async_tool_loop(
        client=client,
        message=goal,
        tools={},
        loop_id="test-typed",
        timeout=30,
    )
    return handle
"""
    exec(code, globals_dict)

    handle = await globals_dict["typed_steerable_workflow"]("Hi")
    assert isinstance(handle, SteerableToolHandle)

    # Clean up
    await handle.stop("cleanup")
    try:
        await asyncio.wait_for(handle.result(), timeout=5.0)
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────────────────
# CodeActActor Steerable Function Test
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compositional_function_returns_codeact_actor_handle():
    """
    A compositional function that wraps CodeActActor should be detected as steerable.

    This is a critical pattern: storing CodeActActor configurations as compositional
    functions allows for reusable, customizable agent workflows. The returned handle
    must be detected as steerable so the execution layer can forward steering operations.
    """
    globals_dict = create_execution_globals()

    # Define a compositional function that creates and returns a CodeActActor handle
    code = """
async def codeact_workflow(goal: str) -> SteerableToolHandle:
    \"\"\"
    A steerable workflow powered by CodeActActor.

    This pattern allows storing pre-configured CodeActActor setups as
    compositional functions that can be searched, retrieved, and executed
    with full steering support.
    \"\"\"
    from unity.actor.code_act_actor import CodeActActor

    # Create a CodeActActor with custom configuration
    actor = CodeActActor()

    # Start the actor - returns a SteerableToolHandle
    handle = await actor.act(
        description=goal,
        clarification_enabled=False,
    )

    return handle
"""
    exec(code, globals_dict)

    # Call the compositional function
    handle = await globals_dict["codeact_workflow"]("Say hello briefly")

    # Verify it's detected as steerable (the key assertion)
    assert isinstance(
        handle,
        SteerableToolHandle,
    ), "CodeActActor handle should be detected as SteerableToolHandle"

    # Verify handle methods are available (these would be forwarded by SingleFunctionActor)
    assert hasattr(handle, "interject")
    assert hasattr(handle, "stop")
    assert hasattr(handle, "pause")
    assert hasattr(handle, "resume")
    assert hasattr(handle, "result")

    # Clean up - stop the actor
    await handle.stop("test cleanup")
    try:
        await asyncio.wait_for(handle.result(), timeout=10.0)
    except Exception:
        pass  # May error due to early stop, that's expected
