"""
Tests for execute_function with target_venv_id override.

Coverage
========
✓ Execute function with default venv (from function table)
✓ Execute function with explicit venv override
✓ Execute function in default environment (target_venv_id=None)
✓ Execute venv-less function in a specific venv
✓ Error handling for non-existent functions
"""

from __future__ import annotations

import shutil
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager
from unity.file_manager.managers.local import LocalFileManager
from unity.common.context_registry import ContextRegistry
from unity.manager_registry import ManagerRegistry

# ────────────────────────────────────────────────────────────────────────────
# Sample Functions
# ────────────────────────────────────────────────────────────────────────────

SIMPLE_SYNC_FUNCTION = """
def add_numbers(a, b):
    \"\"\"Add two numbers together.\"\"\"
    return a + b
""".strip()

SIMPLE_ASYNC_FUNCTION = """
async def greet(name):
    \"\"\"Return a greeting.\"\"\"
    return f"Hello, {name}!"
""".strip()

FUNCTION_WITH_PRIMITIVES = """
async def ask_contacts(question):
    \"\"\"Ask contacts a question via primitives.\"\"\"
    result = await primitives.contacts.ask(question=question)
    return result
""".strip()

# Minimal venv content for testing
MINIMAL_VENV_CONTENT = """
[project]
name = "test-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager_factory(tmp_path):
    """Factory fixture that creates FunctionManager instances.

    Uses tmp_path as the LocalFileManager root so function files are written to
    an ephemeral directory rather than ~/Unity/Local (which on macOS's
    case-insensitive filesystem can collide with the repo checkout).
    """
    managers = []

    def _create():
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/Compositional")
        ContextRegistry.forget(FunctionManager, "Functions/Primitives")
        ContextRegistry.forget(FunctionManager, "Functions/Meta")
        # Clear the LocalFileManager singleton so we can create one rooted at tmp_path
        ManagerRegistry.clear()
        local_fm = LocalFileManager(root=str(tmp_path / "Local"))
        fm = FunctionManager(file_manager=local_fm)
        managers.append(fm)
        return fm

    yield _create

    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


@pytest.fixture
def mock_primitives():
    """Create a mock primitives object for testing."""
    primitives = MagicMock()
    primitives.contacts = MagicMock()
    primitives.contacts.ask = AsyncMock(return_value="Alice is a test contact")
    return primitives


# ────────────────────────────────────────────────────────────────────────────
# Test: Default Behavior (Use Function's Venv)
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_default_no_venv(function_manager_factory):
    """Execute a function that has no venv in default environment."""
    fm = function_manager_factory()

    # Add a simple function with no venv
    fm.add_functions(implementations=SIMPLE_SYNC_FUNCTION)

    # Execute with default behavior (should use function's venv_id which is None)
    result = await fm.execute_function(
        function_name="add_numbers",
        call_kwargs={"a": 2, "b": 3},
    )

    assert result["error"] is None, f"Unexpected error: {result['error']}"
    assert result["result"] == 5


@_handle_project
@pytest.mark.asyncio
async def test_execute_async_function_default_no_venv(function_manager_factory):
    """Execute an async function that has no venv in default environment."""
    fm = function_manager_factory()

    # Add an async function with no venv
    fm.add_functions(implementations=SIMPLE_ASYNC_FUNCTION)

    # Execute with default behavior
    result = await fm.execute_function(
        function_name="greet",
        call_kwargs={"name": "World"},
    )

    assert result["error"] is None, f"Unexpected error: {result['error']}"
    assert result["result"] == "Hello, World!"


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_with_primitives_default_env(
    function_manager_factory,
    mock_primitives,
):
    """Execute a function with primitives in default environment."""
    fm = function_manager_factory()

    # Add function that uses primitives
    fm.add_functions(implementations=FUNCTION_WITH_PRIMITIVES)

    # Execute with primitives
    result = await fm.execute_function(
        function_name="ask_contacts",
        call_kwargs={"question": "Who is Alice?"},
        extra_namespaces={"primitives": mock_primitives},
    )

    assert result["error"] is None, f"Unexpected error: {result['error']}"
    assert result["result"] == "Alice is a test contact"
    mock_primitives.contacts.ask.assert_called_once_with(question="Who is Alice?")


# ────────────────────────────────────────────────────────────────────────────
# Test: Override to Specific Venv
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_override_to_venv(function_manager_factory):
    """Execute a venv-less function in a specific venv via override."""
    fm = function_manager_factory()

    # Add a simple function with no venv
    fm.add_functions(implementations=SIMPLE_SYNC_FUNCTION)

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        # Execute in the venv (override from None to venv_id)
        result = await fm.execute_function(
            function_name="add_numbers",
            call_kwargs={"a": 10, "b": 20},
            target_venv_id=venv_id,  # Override!
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"
        assert result["result"] == 30
    finally:
        # Cleanup
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_override_to_different_venv(function_manager_factory):
    """Execute a function associated with venv A in venv B."""
    fm = function_manager_factory()

    # Create two venvs
    venv_a = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_b = fm.add_venv(venv=MINIMAL_VENV_CONTENT.replace("test-venv", "test-venv-b"))

    # Add a function associated with venv A
    fm.add_functions(implementations=SIMPLE_SYNC_FUNCTION)
    # Get the function and associate it with venv A
    listing = fm.list_functions()
    func_id = listing["add_numbers"]["function_id"]
    fm.set_function_venv(function_id=func_id, venv_id=venv_a)

    try:
        # Execute in venv B (override from venv A to venv B)
        result = await fm.execute_function(
            function_name="add_numbers",
            call_kwargs={"a": 100, "b": 200},
            target_venv_id=venv_b,  # Override to different venv!
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"
        assert result["result"] == 300
    finally:
        # Cleanup
        for vid in [venv_a, venv_b]:
            venv_dir = fm._get_venv_dir(vid)
            if venv_dir.exists():
                shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# Test: Override to Default Environment (None)
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_execute_venv_function_in_default_env(function_manager_factory):
    """Execute a venv function in default environment via override."""
    fm = function_manager_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Add a function and associate with venv
    fm.add_functions(implementations=SIMPLE_SYNC_FUNCTION)
    listing = fm.list_functions()
    func_id = listing["add_numbers"]["function_id"]
    fm.set_function_venv(function_id=func_id, venv_id=venv_id)

    # Execute in default env (override from venv_id to None)
    result = await fm.execute_function(
        function_name="add_numbers",
        call_kwargs={"a": 5, "b": 7},
        target_venv_id=None,  # Override to default env!
    )

    assert result["error"] is None, f"Unexpected error: {result['error']}"
    assert result["result"] == 12


# ────────────────────────────────────────────────────────────────────────────
# Test: Error Handling
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_execute_nonexistent_function_raises(function_manager_factory):
    """Executing a non-existent function raises ValueError."""
    fm = function_manager_factory()

    with pytest.raises(ValueError, match="not found"):
        await fm.execute_function(
            function_name="does_not_exist",
            call_kwargs={},
        )


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_runtime_error(function_manager_factory):
    """Function runtime errors are captured in result."""
    fm = function_manager_factory()

    error_function = """
def raise_error():
    raise ValueError("Intentional error")
""".strip()

    fm.add_functions(implementations=error_function)

    result = await fm.execute_function(
        function_name="raise_error",
        call_kwargs={},
    )

    assert result["error"] is not None
    assert "Intentional error" in result["error"]


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_wrong_args(function_manager_factory):
    """Passing wrong arguments results in error."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SIMPLE_SYNC_FUNCTION)

    result = await fm.execute_function(
        function_name="add_numbers",
        call_kwargs={"x": 1, "y": 2},  # Wrong arg names
    )

    assert result["error"] is not None
    # Should mention missing argument or unexpected keyword
    assert (
        "argument" in result["error"].lower() or "unexpected" in result["error"].lower()
    )


# ────────────────────────────────────────────────────────────────────────────
# Test: Stdout/Stderr Capture
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_captures_stdout(function_manager_factory):
    """Stdout from function execution is captured."""
    fm = function_manager_factory()

    print_function = """
def print_message(msg):
    print(f"Output: {msg}")
    return "done"
""".strip()

    fm.add_functions(implementations=print_function)

    result = await fm.execute_function(
        function_name="print_message",
        call_kwargs={"msg": "Hello"},
    )

    assert result["error"] is None
    assert result["result"] == "done"
    assert "Output: Hello" in result["stdout"]


# ────────────────────────────────────────────────────────────────────────────
# Test: Verify Default Behavior Uses Function's Venv
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_uses_stored_venv_by_default(function_manager_factory):
    """When target_venv_id is not specified, use function's stored venv."""
    fm = function_manager_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Add function and associate with venv
    fm.add_functions(implementations=SIMPLE_SYNC_FUNCTION)
    listing = fm.list_functions()
    func_id = listing["add_numbers"]["function_id"]
    fm.set_function_venv(function_id=func_id, venv_id=venv_id)

    try:
        # Execute without specifying target_venv_id - should use venv_id from table
        result = await fm.execute_function(
            function_name="add_numbers",
            call_kwargs={"a": 1, "b": 1},
            # target_venv_id not specified - defaults to function's venv
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"
        assert result["result"] == 2
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
