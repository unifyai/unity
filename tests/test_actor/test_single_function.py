"""
Tests for SingleFunctionActor - a minimal actor that executes a single function.
"""

import asyncio
import pytest

from unity.actor.single_function_actor import (
    SingleFunctionActor,
)
from unity.function_manager.function_manager import FunctionManager
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# Helper functions to create test functions inside @_handle_project context
# ────────────────────────────────────────────────────────────────────────────


def _create_sync_function(fm: FunctionManager) -> dict:
    """Add a simple synchronous function to the FunctionManager."""
    implementation = '''
def greet_user(name: str = "World") -> str:
    """Greets a user by name."""
    return f"Hello, {name}!"
'''
    result = fm.add_functions(implementations=[implementation])
    assert result.get("greet_user") in ("added", "skipped: already exists")
    functions = fm.list_functions(include_implementations=True)
    return functions["greet_user"]


def _create_async_function(fm: FunctionManager) -> dict:
    """Add a simple async function to the FunctionManager."""
    implementation = '''
async def async_greeting(name: str = "World") -> str:
    """Asynchronously greets a user by name."""
    import asyncio
    await asyncio.sleep(0.01)
    return f"Async hello, {name}!"
'''
    result = fm.add_functions(implementations=[implementation])
    assert result.get("async_greeting") in ("added", "skipped: already exists")
    functions = fm.list_functions(include_implementations=True)
    return functions["async_greeting"]


def _create_slow_function(fm: FunctionManager) -> dict:
    """Add a slow function that can be cancelled."""
    implementation = '''
async def slow_task() -> str:
    """A slow task that takes a while to complete."""
    import asyncio
    await asyncio.sleep(10)
    return "Completed slowly"
'''
    result = fm.add_functions(implementations=[implementation])
    assert result.get("slow_task") in ("added", "skipped: already exists")
    functions = fm.list_functions(include_implementations=True)
    return functions["slow_task"]


def _create_failing_function(fm: FunctionManager) -> dict:
    """Add a function that raises an error."""
    implementation = '''
def failing_task() -> str:
    """A task that always fails."""
    raise ValueError("Intentional test failure")
'''
    result = fm.add_functions(implementations=[implementation])
    assert result.get("failing_task") in ("added", "skipped: already exists")
    functions = fm.list_functions(include_implementations=True)
    return functions["failing_task"]


# ────────────────────────────────────────────────────────────────────────────
# 1. Basic execution tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_execute_sync_function_by_id():
    """Execute a sync function by its ID."""
    fm = FunctionManager()
    simple_sync_function = _create_sync_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    function_id = simple_sync_function["function_id"]
    handle = await actor.act(
        description="greet someone",
        function_id=function_id,
        call_kwargs={"name": "Alice"},
    )

    result = await handle.result()
    assert "Hello, Alice!" in result
    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_execute_async_function_by_id():
    """Execute an async function by its ID."""
    fm = FunctionManager()
    simple_async_function = _create_async_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    function_id = simple_async_function["function_id"]
    handle = await actor.act(
        description="async greet",
        function_id=function_id,
        call_kwargs={"name": "Bob"},
    )

    result = await handle.result()
    assert "Async hello, Bob!" in result
    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_execute_function_by_description():
    """Execute a function found by semantic search."""
    fm = FunctionManager()
    _create_sync_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    # Search by description instead of ID
    handle = await actor.act(
        description="greet a user by their name",
        call_kwargs={"name": "Charlie"},
    )

    result = await handle.result()
    assert "Hello, Charlie!" in result
    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_execute_function_default_args():
    """Execute a function with default arguments."""
    fm = FunctionManager()
    simple_sync_function = _create_sync_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    function_id = simple_sync_function["function_id"]
    handle = await actor.act(
        description="greet",
        function_id=function_id,
        # No call_kwargs - should use default "World"
    )

    result = await handle.result()
    assert "Hello, World!" in result


# ────────────────────────────────────────────────────────────────────────────
# 2. Error handling tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_function_not_found_by_id():
    """Error when function ID doesn't exist."""
    fm = FunctionManager()

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    with pytest.raises(ValueError, match="No function found with ID"):
        await actor.act(
            description="anything",
            function_id=99999,
        )


@pytest.mark.asyncio
@_handle_project
async def test_function_not_found_by_description():
    """Error when no function matches description (with primitives excluded)."""
    fm = FunctionManager()
    # Don't add any functions

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    with pytest.raises(ValueError, match="No function found matching"):
        await actor.act(description="do something impossible", include_primitives=False)


@pytest.mark.asyncio
@_handle_project
async def test_function_execution_error():
    """Handle errors during function execution."""
    fm = FunctionManager()
    failing_function = _create_failing_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    function_id = failing_function["function_id"]
    handle = await actor.act(
        description="fail",
        function_id=function_id,
    )

    result = await handle.result()
    assert "Error:" in result or "failed" in result.lower()
    assert "Intentional test failure" in result
    assert handle.done()


# ────────────────────────────────────────────────────────────────────────────
# 3. Handle steering tests (mostly no-ops)
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_handle_pause_is_noop():
    """Pause should be a no-op that returns acknowledgment."""
    fm = FunctionManager()
    simple_sync_function = _create_sync_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    function_id = simple_sync_function["function_id"]
    handle = await actor.act(
        description="greet",
        function_id=function_id,
    )

    pause_result = await handle.pause()
    assert "acknowledged" in pause_result.lower() or "no effect" in pause_result.lower()

    # Function should still complete normally
    result = await handle.result()
    assert "Hello" in result


@pytest.mark.asyncio
@_handle_project
async def test_handle_resume_is_noop():
    """Resume should be a no-op that returns acknowledgment."""
    fm = FunctionManager()
    simple_sync_function = _create_sync_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    function_id = simple_sync_function["function_id"]
    handle = await actor.act(
        description="greet",
        function_id=function_id,
    )

    resume_result = await handle.resume()
    assert (
        "acknowledged" in resume_result.lower() or "no effect" in resume_result.lower()
    )

    result = await handle.result()
    assert "Hello" in result


@pytest.mark.asyncio
@_handle_project
async def test_handle_interject_is_noop():
    """Interject should be a no-op that returns acknowledgment."""
    fm = FunctionManager()
    simple_sync_function = _create_sync_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    function_id = simple_sync_function["function_id"]
    handle = await actor.act(
        description="greet",
        function_id=function_id,
    )

    interject_result = await handle.interject("change something")
    assert (
        "acknowledged" in interject_result.lower()
        or "no effect" in interject_result.lower()
    )

    result = await handle.result()
    assert "Hello" in result


@pytest.mark.asyncio
@_handle_project
async def test_handle_stop_cancels_execution():
    """Stop should cancel a running function."""
    fm = FunctionManager()
    slow_function = _create_slow_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    function_id = slow_function["function_id"]
    handle = await actor.act(
        description="slow",
        function_id=function_id,
    )

    # Wait a bit then stop
    await asyncio.sleep(0.05)
    assert not handle.done()

    stop_result = await handle.stop("Test cancellation")
    assert "stopped" in stop_result.lower()
    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_handle_ask_returns_status():
    """Ask should return information about the function status."""
    fm = FunctionManager()
    simple_async_function = _create_async_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    function_id = simple_async_function["function_id"]
    handle = await actor.act(
        description="async greet",
        function_id=function_id,
    )

    # Ask while running
    ask_handle = await handle.ask("What's happening?")
    ask_result = await ask_handle.result()
    assert isinstance(ask_result, str)

    # Wait for main execution to complete
    await handle.result()


# ────────────────────────────────────────────────────────────────────────────
# 4. Handle property tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_handle_done_property():
    """done() should reflect completion status."""
    fm = FunctionManager()
    simple_sync_function = _create_sync_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    function_id = simple_sync_function["function_id"]
    handle = await actor.act(
        description="greet",
        function_id=function_id,
    )

    # Wait for completion
    await handle.result()
    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_handle_get_history_is_empty():
    """get_history() should return empty list for single function."""
    fm = FunctionManager()
    simple_sync_function = _create_sync_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    function_id = simple_sync_function["function_id"]
    handle = await actor.act(
        description="greet",
        function_id=function_id,
    )

    history = handle.get_history()
    assert history == []

    await handle.result()


@pytest.mark.asyncio
@_handle_project
async def test_handle_clarification_queues_are_none():
    """Clarification queues should be None."""
    fm = FunctionManager()
    simple_sync_function = _create_sync_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    function_id = simple_sync_function["function_id"]
    handle = await actor.act(
        description="greet",
        function_id=function_id,
    )

    assert handle.clarification_up_q is None
    assert handle.clarification_down_q is None

    await handle.result()


# ────────────────────────────────────────────────────────────────────────────
# 5. Primitive lookup tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_get_primitive_by_name():
    """Should be able to get a primitive by its qualified name."""
    fm = FunctionManager()

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    primitive_data = actor._get_primitive_by_name("ContactManager.ask")

    assert primitive_data["name"] == "ContactManager.ask"
    assert primitive_data.get("is_primitive") is True
    assert "argspec" in primitive_data


@pytest.mark.asyncio
@_handle_project
async def test_get_primitive_by_name_not_found():
    """Should raise ValueError for unknown primitive name."""
    fm = FunctionManager()

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    with pytest.raises(ValueError, match="No primitive found"):
        actor._get_primitive_by_name("NonExistent.method")
