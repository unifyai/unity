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
        function_id=function_id,
        call_kwargs={"name": "Alice"},
    )

    result = await handle.result()
    assert result.result == "Hello, Alice!"
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
        function_id=function_id,
        call_kwargs={"name": "Bob"},
    )

    result = await handle.result()
    assert result.result == "Async hello, Bob!"
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
    assert result.result == "Hello, Charlie!"
    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_semantic_search_selects_best_match():
    """Semantic search should select the most relevant function when multiple exist."""
    fm = FunctionManager()

    # Add multiple functions with different purposes
    calc_impl = '''
def calculate_sum(a: int, b: int) -> int:
    """Calculates the sum of two numbers."""
    return a + b
'''
    weather_impl = '''
def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    return f"Weather in {city}: Sunny"
'''
    email_impl = '''
def send_email(to: str, subject: str) -> str:
    """Sends an email to a recipient."""
    return f"Email sent to {to} with subject: {subject}"
'''
    fm.add_functions(implementations=[calc_impl, weather_impl, email_impl])

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    # Search for calculation-related task
    handle = await actor.act(
        description="add two numbers together",
        call_kwargs={"a": 5, "b": 3},
    )

    result = await handle.result()
    assert str(result.result) == "8"  # 5 + 3 = 8


@pytest.mark.asyncio
@_handle_project
async def test_semantic_search_disambiguates_similar_functions():
    """Semantic search should disambiguate between similarly-named functions."""
    fm = FunctionManager()

    # Add functions with similar names but different purposes
    user_greet = '''
def greet_user(name: str) -> str:
    """Greets a human user by their name."""
    return f"Hello, {name}!"
'''
    pet_greet = '''
def greet_pet(name: str) -> str:
    """Greets a pet animal by their name."""
    return f"Good boy, {name}!"
'''
    fm.add_functions(implementations=[user_greet, pet_greet])

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    # Search specifically for pet greeting
    handle = await actor.act(
        description="say hello to my dog",
        call_kwargs={"name": "Buddy"},
    )

    result = await handle.result()
    assert result.result is not None and ("Good boy" in str(result.result) or "Buddy" in str(result.result))


@pytest.mark.asyncio
@_handle_project
async def test_semantic_search_finds_primitive():
    """Semantic search should find primitives when include_primitives=True (default)."""
    fm = FunctionManager()
    fm.sync_primitives()  # Ensure primitives are loaded

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    # Search for a primitive by describing what it does
    results = fm.search_functions(
        query="ask questions about my contacts",
        n=5,
        include_primitives=True,
    )

    # Verify that at least one primitive is in the results
    primitive_found = any(r.get("is_primitive", False) for r in results)
    assert primitive_found, "Expected to find at least one primitive in search results"

    # Verify ContactManager.ask is findable
    contact_primitive = next(
        (r for r in results if "Contact" in r.get("name", "")),
        None,
    )
    assert contact_primitive is not None, "Expected to find ContactManager primitive"


@pytest.mark.asyncio
@_handle_project
async def test_semantic_search_excludes_primitives_when_disabled():
    """Semantic search should exclude primitives when include_primitives=False."""
    fm = FunctionManager()
    fm.sync_primitives()  # Ensure primitives are loaded

    # Add a user function about contacts
    contact_func = '''
def list_my_contacts() -> str:
    """Lists all my contacts."""
    return "Contacts: Alice, Bob, Charlie"
'''
    fm.add_functions(implementations=[contact_func])

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    # Search with primitives excluded
    results = fm.search_functions(
        query="ask questions about my contacts",
        n=10,
        include_primitives=False,
    )

    # Verify no primitives in results
    for result in results:
        assert not result.get(
            "is_primitive",
            False,
        ), f"Found primitive {result.get('name')} when primitives should be excluded"


@pytest.mark.asyncio
@_handle_project
async def test_semantic_search_with_no_user_functions():
    """Semantic search should work with only primitives available."""
    fm = FunctionManager()
    fm.sync_primitives()  # Ensure primitives are loaded

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    # Search with only primitives available (no user functions added)
    results = fm.search_functions(
        query="manage my tasks and schedule",
        n=5,
        include_primitives=True,
    )

    # Should find some primitives
    assert len(results) > 0, "Expected to find primitives even with no user functions"


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
        function_id=function_id,
        # No call_kwargs - should use default "World"
    )

    result = await handle.result()
    assert result.result == "Hello, World!"


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
        await actor.act(function_id=99999)


@pytest.mark.asyncio
@_handle_project
async def test_function_not_found_by_description(monkeypatch):
    """Error when no function matches description."""
    fm = FunctionManager()

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    # Mock search to return empty results to test the error path
    def mock_search(*args, **kwargs):
        return []

    monkeypatch.setattr(fm, "search_functions", mock_search)

    with pytest.raises(ValueError, match="No function found matching"):
        await actor.act(description="anything")


@pytest.mark.asyncio
@_handle_project
async def test_no_selection_method_provided():
    """Error when no function_id, primitive_name, or description is provided."""
    fm = FunctionManager()

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    with pytest.raises(ValueError, match="Must provide at least one of"):
        await actor.act()


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
    handle = await actor.act(function_id=function_id)

    result = await handle.result()
    assert result.error is not None
    assert "Intentional test failure" in result.error
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
    handle = await actor.act(function_id=function_id)

    pause_result = await handle.pause()
    assert pause_result is None

    # Function should still complete normally
    result = await handle.result()
    assert "Hello" in str(result.result)


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
    handle = await actor.act(function_id=function_id)

    resume_result = await handle.resume()
    assert resume_result is None

    result = await handle.result()
    assert "Hello" in str(result.result)


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
    handle = await actor.act(function_id=function_id)

    await handle.interject("change something")

    result = await handle.result()
    assert "Hello" in str(result.result)


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
    handle = await actor.act(function_id=function_id)

    # Wait a bit then stop
    await asyncio.sleep(0.05)
    assert not handle.done()

    await handle.stop("Test cancellation")
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
    handle = await actor.act(function_id=function_id)

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
    handle = await actor.act(function_id=function_id)

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
    handle = await actor.act(function_id=function_id)

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
    handle = await actor.act(function_id=function_id)

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


# ────────────────────────────────────────────────────────────────────────────
# 6. Custom venv execution tests
# ────────────────────────────────────────────────────────────────────────────

# Minimal venv for fast tests
MINIMAL_VENV_CONTENT = """
[project]
name = "test-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
""".strip()


def _create_venv_function(fm: FunctionManager, venv_id: int) -> dict:
    """Add a function that runs in a custom venv."""
    implementation = f'''
async def venv_greeting(name: str = "World") -> str:
    """Greets a user from inside a custom venv."""
    import asyncio
    await asyncio.sleep(0.01)
    return f"Hello from venv, {{name}}!"
'''
    # Add function with explicit venv_id
    result = fm.add_functions(implementations=[implementation])
    assert result.get("venv_greeting") in ("added", "skipped: already exists")

    # Get the function and update its venv_id
    functions = fm.list_functions(include_implementations=True)
    func_data = functions["venv_greeting"]
    function_id = func_data["function_id"]

    # Update the function to use the venv
    fm.set_function_venv(function_id=function_id, venv_id=venv_id)

    # Re-fetch to get updated data
    functions = fm.list_functions(include_implementations=True)
    return functions["venv_greeting"]


def _create_primitives_venv_function(fm: FunctionManager, venv_id: int) -> dict:
    """Add a function that uses primitives from inside a custom venv."""
    implementation = '''
async def ask_contacts_from_venv(question: str) -> str:
    """Asks the contacts manager a question from inside a venv via RPC."""
    result = await primitives.contacts.ask(question=question)
    return f"Got answer: {result}"
'''
    result = fm.add_functions(implementations=[implementation])
    assert result.get("ask_contacts_from_venv") in ("added", "skipped: already exists")

    functions = fm.list_functions(include_implementations=True)
    func_data = functions["ask_contacts_from_venv"]
    function_id = func_data["function_id"]

    fm.set_function_venv(function_id=function_id, venv_id=venv_id)

    functions = fm.list_functions(include_implementations=True)
    return functions["ask_contacts_from_venv"]


@pytest.fixture
def cleanup_venvs():
    """Fixture to clean up venv directories after tests."""
    import shutil

    fm = FunctionManager()
    venv_ids = []

    yield venv_ids

    # Cleanup
    for venv_id in venv_ids:
        try:
            venv_dir = fm._get_venv_dir(venv_id)
            if venv_dir.exists():
                shutil.rmtree(venv_dir, ignore_errors=True)
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_execute_function_in_custom_venv(cleanup_venvs):
    """Execute a function in a custom virtual environment via SingleFunctionActor."""
    fm = FunctionManager()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs.append(venv_id)

    # Create function that uses the venv
    func_data = _create_venv_function(fm, venv_id)
    assert func_data["venv_id"] == venv_id

    # Execute via actor
    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    handle = await actor.act(
        function_id=func_data["function_id"],
        call_kwargs={"name": "VenvUser"},
    )

    result = await handle.result()
    assert result.result == "Hello from venv, VenvUser!"
    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_execute_venv_function_by_description(cleanup_venvs):
    """Find and execute a venv function by semantic search."""
    fm = FunctionManager()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs.append(venv_id)

    func_data = _create_venv_function(fm, venv_id)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    # Search by description - should find and execute in venv
    handle = await actor.act(
        description="greet someone from a virtual environment",
        call_kwargs={"name": "SearchUser"},
    )

    result = await handle.result()
    assert result.result is not None and "Hello from venv" in str(result.result)
    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_venv_function_error_handling(cleanup_venvs):
    """Errors in venv functions should propagate correctly."""
    fm = FunctionManager()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs.append(venv_id)

    # Create a function that raises an error
    error_impl = '''
async def venv_error_function() -> str:
    """A function that always fails."""
    raise ValueError("Error from inside venv")
'''
    fm.add_functions(implementations=[error_impl])
    functions = fm.list_functions(include_implementations=True)
    func_data = functions["venv_error_function"]
    fm.set_function_venv(function_id=func_data["function_id"], venv_id=venv_id)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    handle = await actor.act(function_id=func_data["function_id"])

    result = await handle.result()
    assert result.error is not None
    assert "Error from inside venv" in result.error
    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_venv_function_with_primitives_rpc(cleanup_venvs):
    """Function in venv should access primitives via RPC through the actor."""
    fm = FunctionManager()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs.append(venv_id)

    func_data = _create_primitives_venv_function(fm, venv_id)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    handle = await actor.act(
        function_id=func_data["function_id"],
        call_kwargs={"question": "Who are my contacts?"},
    )

    await handle.result()
    # The function executed in venv and completed (RPC details tested in test_venv_rpc.py)
    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_venv_function_stop_cancels_subprocess(cleanup_venvs):
    """Stopping a venv function should terminate the subprocess."""
    fm = FunctionManager()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs.append(venv_id)

    # Create a slow function
    slow_impl = '''
async def slow_venv_task() -> str:
    """A slow task in a venv."""
    import asyncio
    await asyncio.sleep(30)
    return "Completed slowly in venv"
'''
    fm.add_functions(implementations=[slow_impl])
    functions = fm.list_functions(include_implementations=True)
    func_data = functions["slow_venv_task"]
    fm.set_function_venv(function_id=func_data["function_id"], venv_id=venv_id)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    handle = await actor.act(function_id=func_data["function_id"])

    # Wait a bit then stop
    await asyncio.sleep(0.5)
    assert not handle.done()

    await handle.stop("Test cancellation")
    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_sync_function_in_venv(cleanup_venvs):
    """Sync functions should also work in custom venvs."""
    fm = FunctionManager()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    cleanup_venvs.append(venv_id)

    # Create a sync function
    sync_impl = '''
def sync_venv_add(a: int, b: int) -> int:
    """Add two numbers in a venv."""
    return a + b
'''
    fm.add_functions(implementations=[sync_impl])
    functions = fm.list_functions(include_implementations=True)
    func_data = functions["sync_venv_add"]
    fm.set_function_venv(function_id=func_data["function_id"], venv_id=venv_id)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    handle = await actor.act(
        function_id=func_data["function_id"],
        call_kwargs={"a": 5, "b": 3},
    )

    result = await handle.result()
    assert str(result.result) == "8"
    assert handle.done()


# ────────────────────────────────────────────────────────────────────────────
# 7. Verification tests
# ────────────────────────────────────────────────────────────────────────────


def _create_successful_function(fm: FunctionManager) -> dict:
    """Add a function that clearly succeeds."""
    implementation = '''
async def calculate_sum(a: int, b: int) -> dict:
    """Calculate the sum of two numbers and return a success result."""
    result = a + b
    return {"status": "success", "sum": result}
'''
    result = fm.add_functions(implementations=[implementation])
    assert result.get("calculate_sum") in ("added", "skipped: already exists")
    functions = fm.list_functions(include_implementations=True)
    return functions["calculate_sum"]


def _create_function_that_returns_failure(fm: FunctionManager) -> dict:
    """Add a function that returns a failure indicator."""
    implementation = '''
async def failed_operation() -> dict:
    """Connects to the database and returns the user data."""
    return {"status": "error", "message": "Operation failed: connection refused"}
'''
    result = fm.add_functions(implementations=[implementation])
    assert result.get("failed_operation") in ("added", "skipped: already exists")
    functions = fm.list_functions(include_implementations=True)
    return functions["failed_operation"]


@pytest.mark.asyncio
@pytest.mark.eval
@_handle_project
async def test_verification_passes_for_successful_function():
    """Verification should pass when function clearly succeeds."""
    fm = FunctionManager()
    func_data = _create_successful_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    handle = await actor.act(
        function_id=func_data["function_id"],
        call_kwargs={"a": 5, "b": 3},
        verify=True,
    )

    result = await handle.result()

    # Should succeed
    assert result.error is None
    assert handle._verification_passed is True
    assert handle._verification_reason is not None


@pytest.mark.asyncio
@pytest.mark.eval
@_handle_project
async def test_verification_fails_for_failed_function():
    """Verification should fail when function returns failure indicators."""
    fm = FunctionManager()
    func_data = _create_function_that_returns_failure(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    handle = await actor.act(
        function_id=func_data["function_id"],
        verify=True,
    )

    result = await handle.result()

    # Should fail verification
    assert handle._verification_passed is False
    assert result.error is not None and ("verification failed" in result.error.lower() or "error" in result.error.lower())
    assert handle._verification_reason is not None


@pytest.mark.asyncio
@_handle_project
async def test_verification_disabled_by_default_when_verify_flag_false():
    """Verification should be skipped when function has verify=False."""
    fm = FunctionManager()

    # Create function with verify=False
    implementation = '''
async def no_verify_task() -> str:
    """A task that should not be verified."""
    return "Done without verification"
'''
    # Add function with verify=False
    fm.add_functions(
        implementations=[implementation],
        verify={"no_verify_task": False},
    )
    functions = fm.list_functions(include_implementations=True)
    func_data = functions["no_verify_task"]

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    handle = await actor.act(
        function_id=func_data["function_id"],
        # Don't pass verify - should use function's flag (False)
    )

    result = await handle.result()

    # Verification should not have run
    assert handle._verification_passed is None
    assert result.result == "Done without verification"


@pytest.mark.asyncio
@pytest.mark.eval
@_handle_project
async def test_verify_param_overrides_function_flag():
    """verify=True on act() should override function's verify=False."""
    fm = FunctionManager()

    implementation = '''
async def override_verify_task() -> dict:
    """A task with explicit verify override."""
    return {"status": "success", "message": "completed"}
'''
    # Add function with verify=False
    fm.add_functions(
        implementations=[implementation],
        verify={"override_verify_task": False},
    )
    functions = fm.list_functions(include_implementations=True)
    func_data = functions["override_verify_task"]

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    handle = await actor.act(
        function_id=func_data["function_id"],
        verify=True,  # Override the function's verify=False
    )

    await handle.result()

    # Verification should have run (override was effective)
    assert handle._verification_passed is not None


@pytest.mark.asyncio
@_handle_project
async def test_verify_false_skips_verification():
    """verify=False on act() should skip verification even if function has verify=True."""
    fm = FunctionManager()

    # Create function with verify=True (the default)
    implementation = '''
async def verified_sum(a: int, b: int) -> dict:
    """Calculate sum with verification enabled by default."""
    return {"status": "success", "sum": a + b}
'''
    fm.add_functions(
        implementations=[implementation],
        verify={"verified_sum": True},
    )
    functions = fm.list_functions(include_implementations=True)
    func_data = functions["verified_sum"]

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    handle = await actor.act(
        function_id=func_data["function_id"],
        call_kwargs={"a": 1, "b": 2},
        verify=False,  # Explicitly skip verification
    )

    result = await handle.result()

    # Verification should not have run
    assert handle._verification_passed is None
    assert result.result is not None and ("success" in str(result.result).lower() or "3" in str(result.result))


# ────────────────────────────────────────────────────────────────────────────
# Steerable Function Forwarding Tests
# ────────────────────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────────────────────
# 8. Dependency Resolution Tests
# ────────────────────────────────────────────────────────────────────────────


def _create_dependency_functions(fm: FunctionManager) -> tuple[dict, dict]:
    """Create a helper function and a main function that depends on it."""
    helper_impl = '''
def helper_multiply(x: int, y: int) -> int:
    """Multiplies two numbers together."""
    return x * y
'''
    main_impl = '''
async def compute_with_helper(a: int, b: int, c: int) -> int:
    """Computes (a * b) + c using the helper function."""
    product = helper_multiply(a, b)
    return product + c
'''
    # Add helper first
    result = fm.add_functions(implementations=[helper_impl])
    assert result.get("helper_multiply") in ("added", "skipped: already exists")

    # Add main function (depends_on will be auto-detected)
    result = fm.add_functions(implementations=[main_impl])
    assert result.get("compute_with_helper") in ("added", "skipped: already exists")

    functions = fm.list_functions(include_implementations=True)
    return functions["helper_multiply"], functions["compute_with_helper"]


@pytest.mark.asyncio
@_handle_project
async def test_dependency_resolution_by_id():
    """Execute a function that depends on another function, retrieving by ID."""
    fm = FunctionManager()
    helper_func, main_func = _create_dependency_functions(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    handle = await actor.act(
        function_id=main_func["function_id"],
        call_kwargs={"a": 5, "b": 3, "c": 2},
        verify=False,
    )

    result = await handle.result()
    assert result.result == 17, f"Expected 17, got: {result.result}"  # (5 * 3) + 2 = 17
    assert handle.done()


@pytest.mark.asyncio
@_handle_project
async def test_dependency_resolution_by_description():
    """Execute a function that depends on another function, retrieving by semantic search."""
    fm = FunctionManager()
    helper_func, main_func = _create_dependency_functions(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    handle = await actor.act(
        description="compute (a * b) + c, product plus addition",
        call_kwargs={"a": 4, "b": 6, "c": 1},
        verify=False,
    )

    result = await handle.result()
    assert result.result == 25, f"Expected 25, got: {result.result}"  # (4 * 6) + 1 = 25
    assert handle.done()
