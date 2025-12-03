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
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager():
    """Provide a fresh FunctionManager for each test."""
    fm = FunctionManager()
    yield fm
    fm.clear()


@pytest.fixture
def simple_sync_function(function_manager):
    """Add a simple synchronous function to the FunctionManager."""
    implementation = '''
def greet_user(name: str = "World") -> str:
    """Greets a user by name."""
    return f"Hello, {name}!"
'''
    result = function_manager.add_functions(implementations=[implementation])
    assert result.get("greet_user") == "added"

    functions = function_manager.list_functions(include_implementations=True)
    return functions["greet_user"]


@pytest.fixture
def simple_async_function(function_manager):
    """Add a simple async function to the FunctionManager."""
    implementation = '''
async def async_greeting(name: str = "World") -> str:
    """Asynchronously greets a user by name."""
    import asyncio
    await asyncio.sleep(0.01)
    return f"Async hello, {name}!"
'''
    result = function_manager.add_functions(implementations=[implementation])
    assert result.get("async_greeting") == "added"

    functions = function_manager.list_functions(include_implementations=True)
    return functions["async_greeting"]


@pytest.fixture
def slow_function(function_manager):
    """Add a slow function that can be cancelled."""
    implementation = '''
async def slow_task() -> str:
    """A slow task that takes a while to complete."""
    import asyncio
    await asyncio.sleep(10)
    return "Completed slowly"
'''
    result = function_manager.add_functions(implementations=[implementation])
    assert result.get("slow_task") == "added"

    functions = function_manager.list_functions(include_implementations=True)
    return functions["slow_task"]


@pytest.fixture
def failing_function(function_manager):
    """Add a function that raises an error."""
    implementation = '''
def failing_task() -> str:
    """A task that always fails."""
    raise ValueError("Intentional test failure")
'''
    result = function_manager.add_functions(implementations=[implementation])
    assert result.get("failing_task") == "added"

    functions = function_manager.list_functions(include_implementations=True)
    return functions["failing_task"]


# ────────────────────────────────────────────────────────────────────────────
# 1. Basic execution tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_execute_sync_function_by_id(function_manager, simple_sync_function):
    """Execute a sync function by its ID."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
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
async def test_execute_async_function_by_id(function_manager, simple_async_function):
    """Execute an async function by its ID."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
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
async def test_execute_function_by_description(function_manager, simple_sync_function):
    """Execute a function found by semantic search."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
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
async def test_execute_function_default_args(function_manager, simple_sync_function):
    """Execute a function with default arguments."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
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
async def test_function_not_found_by_id(function_manager):
    """Error when function ID doesn't exist."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
    )

    with pytest.raises(ValueError, match="No function found with ID"):
        await actor.act(
            description="anything",
            function_id=99999,
        )


@pytest.mark.asyncio
@_handle_project
async def test_function_not_found_by_description(function_manager):
    """Error when no function matches description (with primitives excluded)."""
    # Don't add any functions
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
    )

    with pytest.raises(ValueError, match="No function found matching"):
        await actor.act(description="do something impossible", include_primitives=False)


@pytest.mark.asyncio
@_handle_project
async def test_function_execution_error(function_manager, failing_function):
    """Handle errors during function execution."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
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
async def test_handle_pause_is_noop(function_manager, simple_sync_function):
    """Pause should be a no-op that returns acknowledgment."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
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
async def test_handle_resume_is_noop(function_manager, simple_sync_function):
    """Resume should be a no-op that returns acknowledgment."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
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
async def test_handle_interject_is_noop(function_manager, simple_sync_function):
    """Interject should be a no-op that returns acknowledgment."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
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
async def test_handle_stop_cancels_execution(function_manager, slow_function):
    """Stop should cancel a running function."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
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
async def test_handle_ask_returns_status(function_manager, simple_async_function):
    """Ask should return information about the function status."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
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
async def test_handle_done_property(function_manager, simple_sync_function):
    """done() should reflect completion status."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
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
async def test_handle_get_history_is_empty(function_manager, simple_sync_function):
    """get_history() should return empty list for single function."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
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
async def test_handle_clarification_queues_are_none(
    function_manager,
    simple_sync_function,
):
    """Clarification queues should be None."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
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
async def test_get_primitive_by_name(function_manager):
    """Should be able to get a primitive by its qualified name."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
    )

    primitive_data = actor._get_primitive_by_name("ContactManager.ask")

    assert primitive_data["name"] == "ContactManager.ask"
    assert primitive_data.get("is_primitive") is True
    assert "argspec" in primitive_data


@pytest.mark.asyncio
@_handle_project
async def test_get_primitive_by_name_not_found(function_manager):
    """Should raise ValueError for unknown primitive name."""
    actor = SingleFunctionActor(
        action_provider=None,
        function_manager=function_manager,
    )

    with pytest.raises(ValueError, match="No primitive found"):
        actor._get_primitive_by_name("NonExistent.method")
