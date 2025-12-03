"""
Tests for RPC access to primitives from custom virtual environments.

Tests that functions running in custom venvs can call back to the main process
to access primitives (state managers) and computer_primitives.
"""

import pytest
import shutil
from unittest.mock import AsyncMock, MagicMock

from unity.function_manager.function_manager import FunctionManager
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project


# Sample pyproject.toml with minimal dependencies
MINIMAL_VENV_CONTENT = """
[project]
name = "test-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Test Functions that Use Primitives
# ────────────────────────────────────────────────────────────────────────────

# Function that calls primitives.contacts.ask
PRIMITIVES_ASK_FUNCTION = """
async def ask_contacts(question: str) -> str:
    \"\"\"Ask the contacts manager a question via RPC.\"\"\"
    result = await primitives.contacts.ask(question=question)
    return result
""".strip()

# Function that calls multiple primitives
MULTI_PRIMITIVE_FUNCTION = """
async def multi_primitive_call() -> dict:
    \"\"\"Call multiple primitives.\"\"\"
    contacts_result = await primitives.contacts.ask(question="Who is Alice?")
    knowledge_result = await primitives.knowledge.ask(question="What is 2+2?")
    return {
        "contacts": contacts_result,
        "knowledge": knowledge_result,
    }
""".strip()

# Function that calls computer_primitives
COMPUTER_PRIMITIVES_FUNCTION = """
async def use_computer(selector: str) -> str:
    \"\"\"Call computer_primitives via RPC.\"\"\"
    result = await computer_primitives.click(selector=selector)
    return result
""".strip()

# Function that uses primitives and returns their result
SIMPLE_PRIMITIVES_FUNCTION = """
async def get_contact_count() -> int:
    \"\"\"Get a count from contacts.\"\"\"
    result = await primitives.contacts.list_all()
    return len(result) if isinstance(result, list) else 0
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager_factory():
    """Factory fixture that creates FunctionManager instances."""
    managers = []

    def _create():
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/Compositional")
        ContextRegistry.forget(FunctionManager, "Functions/Primitives")
        ContextRegistry.forget(FunctionManager, "Functions/Meta")
        fm = FunctionManager()
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
    """Create a mock primitives object for testing RPC."""
    primitives = MagicMock()

    # Mock contacts manager
    primitives.contacts = MagicMock()
    primitives.contacts.ask = AsyncMock(return_value="Alice is a test contact")
    primitives.contacts.list_all = AsyncMock(
        return_value=[{"name": "Alice"}, {"name": "Bob"}],
    )

    # Mock knowledge manager
    primitives.knowledge = MagicMock()
    primitives.knowledge.ask = AsyncMock(return_value="4")

    return primitives


@pytest.fixture
def mock_computer_primitives():
    """Create a mock computer_primitives object for testing RPC."""
    computer = MagicMock()
    computer.click = AsyncMock(return_value="clicked")
    computer.type_text = AsyncMock(return_value="typed")
    return computer


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_execute_with_primitives_rpc(
    function_manager_factory,
    mock_primitives,
):
    """Function in venv should be able to call primitives via RPC."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=PRIMITIVES_ASK_FUNCTION,
            call_kwargs={"question": "Who is Alice?"},
            is_async=True,
            primitives=mock_primitives,
        )

        # Should succeed without errors
        assert result["error"] is None, f"Unexpected error: {result['error']}"

        # The mock should have been called
        mock_primitives.contacts.ask.assert_called_once_with(question="Who is Alice?")

        # Result should be from the mock
        assert result["result"] == "Alice is a test contact"
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_execute_with_multiple_primitive_calls(
    function_manager_factory,
    mock_primitives,
):
    """Function should be able to make multiple RPC calls."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=MULTI_PRIMITIVE_FUNCTION,
            call_kwargs={},
            is_async=True,
            primitives=mock_primitives,
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"

        # Both mocks should have been called
        mock_primitives.contacts.ask.assert_called_once()
        mock_primitives.knowledge.ask.assert_called_once()

        # Result should contain both responses
        assert result["result"]["contacts"] == "Alice is a test contact"
        assert result["result"]["knowledge"] == "4"
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_execute_with_computer_primitives_rpc(
    function_manager_factory,
    mock_computer_primitives,
):
    """Function in venv should be able to call computer_primitives via RPC."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=COMPUTER_PRIMITIVES_FUNCTION,
            call_kwargs={"selector": "#button"},
            is_async=True,
            computer_primitives=mock_computer_primitives,
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"

        # The mock should have been called
        mock_computer_primitives.click.assert_called_once_with(selector="#button")

        # Result should be from the mock
        assert result["result"] == "clicked"
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_execute_without_primitives_errors_gracefully(
    function_manager_factory,
):
    """Function calling primitives without them provided should get an error."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=PRIMITIVES_ASK_FUNCTION,
            call_kwargs={"question": "test"},
            is_async=True,
            primitives=None,  # No primitives provided
        )

        # Should have an error about primitives not being available
        assert result["error"] is not None
        assert (
            "primitives" in result["error"].lower() or "rpc" in result["error"].lower()
        )
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_primitives_list_returns_correct_count(
    function_manager_factory,
    mock_primitives,
):
    """Function should receive and process RPC results correctly."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_PRIMITIVES_FUNCTION,
            call_kwargs={},
            is_async=True,
            primitives=mock_primitives,
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"

        # Mock returns 2 contacts
        assert result["result"] == 2
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_rpc_error_propagates_to_function(
    function_manager_factory,
):
    """Errors from RPC calls should propagate back to the function."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Create a mock that raises an error
    mock_primitives = MagicMock()
    mock_primitives.contacts = MagicMock()
    mock_primitives.contacts.ask = AsyncMock(
        side_effect=ValueError("Simulated RPC error"),
    )

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=PRIMITIVES_ASK_FUNCTION,
            call_kwargs={"question": "test"},
            is_async=True,
            primitives=mock_primitives,
        )

        # The error should be captured
        assert result["error"] is not None
        assert "Simulated RPC error" in result["error"] or "RPC" in result["error"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
