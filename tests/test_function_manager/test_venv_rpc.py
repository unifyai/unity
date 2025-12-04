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


# ────────────────────────────────────────────────────────────────────────────
# Error Propagation in Full Chain Tests
# ────────────────────────────────────────────────────────────────────────────


# Function that raises an error immediately
FUNCTION_RAISES_IMMEDIATELY = """
async def raise_immediately() -> str:
    \"\"\"Raises an error before any RPC.\"\"\"
    raise ValueError("Immediate function error")
""".strip()


# Function that uses an invalid primitive path
INVALID_PRIMITIVE_PATH_FUNCTION = """
async def call_invalid_primitive() -> str:
    \"\"\"Call a non-existent primitive method.\"\"\"
    result = await primitives.nonexistent.fake_method(arg="test")
    return result
""".strip()


# Function that captures and re-raises RPC errors
FUNCTION_RERAISES_RPC_ERROR = """
async def reraise_rpc_error() -> str:
    \"\"\"Call primitive and reraise with additional context.\"\"\"
    try:
        result = await primitives.contacts.ask(question="test")
        return result
    except Exception as e:
        raise RuntimeError(f"Wrapped: {e}") from e
""".strip()


# Function that makes multiple calls, one of which fails
PARTIAL_FAILURE_FUNCTION = """
async def partial_failure() -> dict:
    \"\"\"First call succeeds, second fails.\"\"\"
    first = await primitives.contacts.ask(question="first")
    second = await primitives.knowledge.ask(question="second")  # Will fail
    return {"first": first, "second": second}
""".strip()


# Sync function that uses primitives (tests sync RPC error handling)
SYNC_PRIMITIVES_FUNCTION = """
def sync_ask_contacts(question: str) -> str:
    \"\"\"Sync function calling primitives via RPC.\"\"\"
    result = primitives.contacts.ask(question=question)
    return result
""".strip()


@_handle_project
@pytest.mark.asyncio
async def test_function_error_before_rpc(function_manager_factory):
    """Errors in function code before any RPC should propagate correctly."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=FUNCTION_RAISES_IMMEDIATELY,
            call_kwargs={},
            is_async=True,
        )

        # The error should be captured
        assert result["error"] is not None
        assert "Immediate function error" in result["error"]
        assert "ValueError" in result["error"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_different_exception_types_propagate(function_manager_factory):
    """Various exception types from primitives should propagate with type info."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Test different exception types
    exception_types = [
        (KeyError, "key_not_found"),
        (RuntimeError, "runtime issue"),
        (TypeError, "wrong type"),
        (AttributeError, "missing attribute"),
    ]

    for exc_type, exc_msg in exception_types:
        mock_primitives = MagicMock()
        mock_primitives.contacts = MagicMock()
        mock_primitives.contacts.ask = AsyncMock(side_effect=exc_type(exc_msg))

        try:
            result = await fm.execute_in_venv(
                venv_id=venv_id,
                implementation=PRIMITIVES_ASK_FUNCTION,
                call_kwargs={"question": "test"},
                is_async=True,
                primitives=mock_primitives,
            )

            # The error message should be captured
            assert (
                result["error"] is not None
            ), f"Expected error for {exc_type.__name__}"
            assert exc_msg in result["error"], (
                f"Expected '{exc_msg}' in error for {exc_type.__name__}, "
                f"got: {result['error']}"
            )
        finally:
            pass  # Don't cleanup between iterations

    # Final cleanup
    venv_dir = fm._get_venv_dir(venv_id)
    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_rpc_error_is_wrapped_in_runtime_error(function_manager_factory):
    """RPC errors should be wrapped in RuntimeError with clear message."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    mock_primitives = MagicMock()
    mock_primitives.contacts = MagicMock()
    mock_primitives.contacts.ask = AsyncMock(
        side_effect=ValueError("Original error message"),
    )

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=PRIMITIVES_ASK_FUNCTION,
            call_kwargs={"question": "test"},
            is_async=True,
            primitives=mock_primitives,
        )

        assert result["error"] is not None
        # Error should contain RPC context
        assert "RPC" in result["error"] or "Original error message" in result["error"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_function_reraises_with_context(function_manager_factory):
    """Function that wraps RPC error should preserve both messages."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    mock_primitives = MagicMock()
    mock_primitives.contacts = MagicMock()
    mock_primitives.contacts.ask = AsyncMock(
        side_effect=ValueError("Original RPC error"),
    )

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=FUNCTION_RERAISES_RPC_ERROR,
            call_kwargs={},
            is_async=True,
            primitives=mock_primitives,
        )

        assert result["error"] is not None
        # Should contain the wrapped context
        assert "Wrapped" in result["error"] or "RuntimeError" in result["error"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_partial_failure_in_multiple_rpc_calls(function_manager_factory):
    """When one of multiple RPC calls fails, error should propagate."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # First call succeeds, second fails
    mock_primitives = MagicMock()
    mock_primitives.contacts = MagicMock()
    mock_primitives.contacts.ask = AsyncMock(return_value="success")
    mock_primitives.knowledge = MagicMock()
    mock_primitives.knowledge.ask = AsyncMock(
        side_effect=RuntimeError("Second call failed"),
    )

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=PARTIAL_FAILURE_FUNCTION,
            call_kwargs={},
            is_async=True,
            primitives=mock_primitives,
        )

        # Error should be from the second call
        assert result["error"] is not None
        assert "Second call failed" in result["error"]

        # First call should have been made
        mock_primitives.contacts.ask.assert_called_once()
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_sync_function_rpc_error(function_manager_factory):
    """Sync functions should also get RPC errors propagated."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    mock_primitives = MagicMock()
    mock_primitives.contacts = MagicMock()
    # For sync functions, the mock needs to return a value or raise synchronously
    mock_primitives.contacts.ask = MagicMock(
        side_effect=ValueError("Sync RPC error"),
    )

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=SYNC_PRIMITIVES_FUNCTION,
            call_kwargs={"question": "test"},
            is_async=False,
            primitives=mock_primitives,
        )

        assert result["error"] is not None
        assert "Sync RPC error" in result["error"] or "RPC" in result["error"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_computer_primitives_error_propagates(function_manager_factory):
    """Errors from computer_primitives RPC should also propagate."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    mock_computer = MagicMock()
    mock_computer.click = AsyncMock(
        side_effect=RuntimeError("Browser not available"),
    )

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=COMPUTER_PRIMITIVES_FUNCTION,
            call_kwargs={"selector": "#button"},
            is_async=True,
            computer_primitives=mock_computer,
        )

        assert result["error"] is not None
        assert "Browser not available" in result["error"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_subprocess_crash_handled_gracefully(function_manager_factory):
    """If the subprocess crashes, error should be returned not raised."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Function that causes subprocess to exit
    crash_function = """
import sys
def crash_subprocess() -> str:
    \"\"\"Force subprocess to exit.\"\"\"
    sys.exit(1)
""".strip()

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=crash_function,
            call_kwargs={},
            is_async=False,
        )

        # Should get an error, not crash the main process
        assert result["error"] is not None
        # Could be "Subprocess ended unexpectedly" or similar
        assert (
            "error" in result["error"].lower()
            or "unexpected" in result["error"].lower()
        )
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_invalid_implementation_syntax_error(function_manager_factory):
    """Syntax errors in implementation should be caught and reported."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Invalid Python syntax
    invalid_function = """
def broken_syntax(
    \"\"\"Missing close paren and colon.\"\"\"
    return "never reached"
""".strip()

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=invalid_function,
            call_kwargs={},
            is_async=False,
        )

        assert result["error"] is not None
        assert "SyntaxError" in result["error"] or "syntax" in result["error"].lower()
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_import_error_in_function(function_manager_factory):
    """Import errors in function code should be caught and reported."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Function that imports non-existent module
    import_error_function = """
async def import_nonexistent() -> str:
    \"\"\"Try to import a module that doesn't exist.\"\"\"
    import nonexistent_module_xyz123
    return "never reached"
""".strip()

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=import_error_function,
            call_kwargs={},
            is_async=True,
        )

        assert result["error"] is not None
        assert (
            "ModuleNotFoundError" in result["error"]
            or "ImportError" in result["error"]
            or "nonexistent_module" in result["error"]
        )
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_error_includes_traceback(function_manager_factory):
    """Error messages should include stack trace information."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Function with nested call that fails
    nested_error_function = """
async def nested_error() -> str:
    \"\"\"Error in nested function.\"\"\"
    def inner():
        def innermost():
            raise ValueError("Deep error")
        return innermost()
    return inner()
""".strip()

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=nested_error_function,
            call_kwargs={},
            is_async=True,
        )

        assert result["error"] is not None
        assert "Deep error" in result["error"]
        # Should have traceback info
        assert "innermost" in result["error"] or "Traceback" in result["error"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_stdout_stderr_captured_with_error(
    function_manager_factory,
    mock_primitives,
):
    """stdout/stderr should still be captured when an error occurs."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Function that prints before failing
    print_then_fail = """
async def print_then_fail() -> str:
    \"\"\"Print something then raise.\"\"\"
    print("This is stdout before failure")
    raise ValueError("Failure after print")
""".strip()

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=print_then_fail,
            call_kwargs={},
            is_async=True,
        )

        assert result["error"] is not None
        assert "Failure after print" in result["error"]
        # stdout should still be captured
        assert "stdout before failure" in result["stdout"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# ComputerPrimitives via Venv RPC - Comprehensive Tests
# ────────────────────────────────────────────────────────────────────────────


# Function that uses computer_primitives.act
CP_ACT_FUNCTION = """
async def use_act(instruction: str) -> str:
    \"\"\"Use computer_primitives.act via RPC.\"\"\"
    result = await computer_primitives.act(instruction=instruction)
    return result
""".strip()


# Function that uses computer_primitives.observe
CP_OBSERVE_FUNCTION = """
async def use_observe() -> str:
    \"\"\"Use computer_primitives.observe via RPC.\"\"\"
    result = await computer_primitives.observe()
    return result
""".strip()


# Function that uses computer_primitives.query
CP_QUERY_FUNCTION = """
async def use_query(question: str) -> str:
    \"\"\"Use computer_primitives.query via RPC.\"\"\"
    result = await computer_primitives.query(question=question)
    return result
""".strip()


# Function that uses computer_primitives.navigate
CP_NAVIGATE_FUNCTION = """
async def use_navigate(url: str) -> str:
    \"\"\"Use computer_primitives.navigate via RPC.\"\"\"
    result = await computer_primitives.navigate(url=url)
    return result
""".strip()


# Function that uses computer_primitives.get_links
CP_GET_LINKS_FUNCTION = """
async def use_get_links() -> list:
    \"\"\"Use computer_primitives.get_links via RPC.\"\"\"
    result = await computer_primitives.get_links()
    return result
""".strip()


# Function that uses computer_primitives.get_content
CP_GET_CONTENT_FUNCTION = """
async def use_get_content() -> str:
    \"\"\"Use computer_primitives.get_content via RPC.\"\"\"
    result = await computer_primitives.get_content()
    return result
""".strip()


# Function that uses computer_primitives.reason
CP_REASON_FUNCTION = """
async def use_reason(request: str, context: str) -> str:
    \"\"\"Use computer_primitives.reason via RPC.\"\"\"
    result = await computer_primitives.reason(request=request, context=context)
    return result
""".strip()


# Function that chains multiple computer_primitives calls
CP_CHAIN_FUNCTION = """
async def chain_computer_primitives(url: str, question: str) -> dict:
    \"\"\"Chain multiple computer_primitives calls.\"\"\"
    nav_result = await computer_primitives.navigate(url=url)
    observe_result = await computer_primitives.observe()
    query_result = await computer_primitives.query(question=question)
    return {
        "navigate": nav_result,
        "observe": observe_result,
        "query": query_result,
    }
""".strip()


# Function that uses both primitives and computer_primitives
MIXED_PRIMITIVES_FUNCTION = """
async def use_both_primitives(contact_question: str, browser_url: str) -> dict:
    \"\"\"Use both primitives and computer_primitives in one function.\"\"\"
    contact_result = await primitives.contacts.ask(question=contact_question)
    nav_result = await computer_primitives.navigate(url=browser_url)
    return {
        "contacts": contact_result,
        "navigate": nav_result,
    }
""".strip()


@pytest.fixture
def full_mock_computer_primitives():
    """Create a comprehensive mock of computer_primitives for testing."""
    computer = MagicMock()
    computer.click = AsyncMock(return_value="clicked #button")
    computer.act = AsyncMock(return_value="action performed")
    computer.observe = AsyncMock(
        return_value="Page shows login form with email/password fields",
    )
    computer.query = AsyncMock(return_value="The page title is 'Dashboard'")
    computer.navigate = AsyncMock(return_value="navigated to url")
    computer.get_links = AsyncMock(
        return_value=[
            {"text": "Home", "href": "/"},
            {"text": "About", "href": "/about"},
            {"text": "Contact", "href": "/contact"},
        ],
    )
    computer.get_content = AsyncMock(
        return_value="<html><body>Page content here</body></html>",
    )
    computer.reason = AsyncMock(
        return_value="Based on analysis, the user wants to login",
    )
    computer.type_text = AsyncMock(return_value="typed text successfully")
    return computer


@_handle_project
@pytest.mark.asyncio
async def test_computer_primitives_act(
    function_manager_factory,
    full_mock_computer_primitives,
):
    """computer_primitives.act should work via RPC."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=CP_ACT_FUNCTION,
            call_kwargs={"instruction": "Click the login button"},
            is_async=True,
            computer_primitives=full_mock_computer_primitives,
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"
        full_mock_computer_primitives.act.assert_called_once_with(
            instruction="Click the login button",
        )
        assert result["result"] == "action performed"
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_computer_primitives_observe(
    function_manager_factory,
    full_mock_computer_primitives,
):
    """computer_primitives.observe should work via RPC."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=CP_OBSERVE_FUNCTION,
            call_kwargs={},
            is_async=True,
            computer_primitives=full_mock_computer_primitives,
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"
        full_mock_computer_primitives.observe.assert_called_once()
        assert "login form" in result["result"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_computer_primitives_query(
    function_manager_factory,
    full_mock_computer_primitives,
):
    """computer_primitives.query should work via RPC."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=CP_QUERY_FUNCTION,
            call_kwargs={"question": "What is the page title?"},
            is_async=True,
            computer_primitives=full_mock_computer_primitives,
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"
        full_mock_computer_primitives.query.assert_called_once_with(
            question="What is the page title?",
        )
        assert "Dashboard" in result["result"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_computer_primitives_navigate(
    function_manager_factory,
    full_mock_computer_primitives,
):
    """computer_primitives.navigate should work via RPC."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=CP_NAVIGATE_FUNCTION,
            call_kwargs={"url": "https://example.com"},
            is_async=True,
            computer_primitives=full_mock_computer_primitives,
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"
        full_mock_computer_primitives.navigate.assert_called_once_with(
            url="https://example.com",
        )
        assert result["result"] == "navigated to url"
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_computer_primitives_get_links_returns_list(
    function_manager_factory,
    full_mock_computer_primitives,
):
    """computer_primitives.get_links should return a list via RPC."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=CP_GET_LINKS_FUNCTION,
            call_kwargs={},
            is_async=True,
            computer_primitives=full_mock_computer_primitives,
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"
        full_mock_computer_primitives.get_links.assert_called_once()

        # Result should be a list of links
        assert isinstance(result["result"], list)
        assert len(result["result"]) == 3
        assert result["result"][0]["text"] == "Home"
        assert result["result"][1]["href"] == "/about"
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_computer_primitives_get_content(
    function_manager_factory,
    full_mock_computer_primitives,
):
    """computer_primitives.get_content should return HTML content via RPC."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=CP_GET_CONTENT_FUNCTION,
            call_kwargs={},
            is_async=True,
            computer_primitives=full_mock_computer_primitives,
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"
        full_mock_computer_primitives.get_content.assert_called_once()
        assert "<html>" in result["result"]
        assert "Page content" in result["result"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_computer_primitives_reason(
    function_manager_factory,
    full_mock_computer_primitives,
):
    """computer_primitives.reason should work via RPC."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=CP_REASON_FUNCTION,
            call_kwargs={
                "request": "Analyze user intent",
                "context": "User clicked login button",
            },
            is_async=True,
            computer_primitives=full_mock_computer_primitives,
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"
        full_mock_computer_primitives.reason.assert_called_once_with(
            request="Analyze user intent",
            context="User clicked login button",
        )
        assert "login" in result["result"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_computer_primitives_chain_multiple_calls(
    function_manager_factory,
    full_mock_computer_primitives,
):
    """Chaining multiple computer_primitives calls should work via RPC."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=CP_CHAIN_FUNCTION,
            call_kwargs={
                "url": "https://example.com",
                "question": "What is on the page?",
            },
            is_async=True,
            computer_primitives=full_mock_computer_primitives,
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"

        # All three methods should have been called
        full_mock_computer_primitives.navigate.assert_called_once_with(
            url="https://example.com",
        )
        full_mock_computer_primitives.observe.assert_called_once()
        full_mock_computer_primitives.query.assert_called_once_with(
            question="What is on the page?",
        )

        # Result should be a dict with all three results
        assert result["result"]["navigate"] == "navigated to url"
        assert "login form" in result["result"]["observe"]
        assert "Dashboard" in result["result"]["query"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_mixed_primitives_and_computer_primitives(
    function_manager_factory,
    mock_primitives,
    full_mock_computer_primitives,
):
    """Using both primitives and computer_primitives in same function should work."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=MIXED_PRIMITIVES_FUNCTION,
            call_kwargs={
                "contact_question": "Who is Alice?",
                "browser_url": "https://example.com",
            },
            is_async=True,
            primitives=mock_primitives,
            computer_primitives=full_mock_computer_primitives,
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"

        # Both should have been called
        mock_primitives.contacts.ask.assert_called_once_with(question="Who is Alice?")
        full_mock_computer_primitives.navigate.assert_called_once_with(
            url="https://example.com",
        )

        # Result should have both
        assert result["result"]["contacts"] == "Alice is a test contact"
        assert result["result"]["navigate"] == "navigated to url"
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_computer_primitives_without_primitives(
    function_manager_factory,
    full_mock_computer_primitives,
):
    """Using only computer_primitives (no primitives) should work."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=CP_OBSERVE_FUNCTION,
            call_kwargs={},
            is_async=True,
            primitives=None,  # Explicitly no primitives
            computer_primitives=full_mock_computer_primitives,
        )

        assert result["error"] is None, f"Unexpected error: {result['error']}"
        assert "login form" in result["result"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_computer_primitives_error_with_details(
    function_manager_factory,
):
    """ComputerPrimitives errors should include detailed error info."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    mock_computer = MagicMock()
    mock_computer.navigate = AsyncMock(
        side_effect=RuntimeError("Connection refused: browser not responding"),
    )

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=CP_NAVIGATE_FUNCTION,
            call_kwargs={"url": "https://example.com"},
            is_async=True,
            computer_primitives=mock_computer,
        )

        assert result["error"] is not None
        assert "Connection refused" in result["error"]
        assert "browser not responding" in result["error"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_computer_primitives_partial_chain_failure(
    function_manager_factory,
):
    """When one call in a chain fails, error should include which call failed."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    mock_computer = MagicMock()
    mock_computer.navigate = AsyncMock(return_value="navigated")
    mock_computer.observe = AsyncMock(side_effect=TimeoutError("Page load timeout"))
    mock_computer.query = AsyncMock(return_value="query result")

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=CP_CHAIN_FUNCTION,
            call_kwargs={
                "url": "https://example.com",
                "question": "test",
            },
            is_async=True,
            computer_primitives=mock_computer,
        )

        assert result["error"] is not None
        assert "Page load timeout" in result["error"]

        # Navigate should have been called (before the failure)
        mock_computer.navigate.assert_called_once()
        # Query should NOT have been called (after the failure)
        mock_computer.query.assert_not_called()
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_computer_primitives_without_computer_primitives_arg(
    function_manager_factory,
):
    """Calling computer_primitives without providing it should error gracefully."""
    fm = function_manager_factory()
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=CP_OBSERVE_FUNCTION,
            call_kwargs={},
            is_async=True,
            computer_primitives=None,  # Not provided
        )

        assert result["error"] is not None
        assert (
            "computer_primitives" in result["error"].lower()
            or "rpc" in result["error"].lower()
        )
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
