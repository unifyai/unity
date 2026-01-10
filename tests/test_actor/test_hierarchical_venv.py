"""
Tests for venv function execution via HierarchicalActor.

Tests that HierarchicalActor can call functions that run in custom virtual
environments. These functions are treated as atomic, opaque callables that
execute via subprocess with RPC access to primitives.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from unity.actor.hierarchical_actor import (
    HierarchicalActor,
    _VenvFunctionProxy,
)
from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.primitives import ComputerPrimitives
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# Sample Venv Content and Functions
# ────────────────────────────────────────────────────────────────────────────

MINIMAL_VENV_CONTENT = """
[project]
name = "test-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
""".strip()


SIMPLE_VENV_FUNCTION = """
async def simple_add(a: int, b: int) -> int:
    \"\"\"Add two numbers in the venv.\"\"\"
    return a + b
""".strip()


VENV_FUNCTION_WITH_PRIMITIVES = """
async def get_contact_name(question: str) -> str:
    \"\"\"Get contact info via RPC.\"\"\"
    result = await primitives.contacts.ask(question=question)
    return f"Got: {result}"
""".strip()


VENV_FUNCTION_WITH_COMPUTER = """
async def click_element(selector: str) -> str:
    \"\"\"Click an element via computer primitives RPC.\"\"\"
    await computer_primitives.click(selector=selector)
    return f"Clicked: {selector}"
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
def mock_computer_primitives():
    """Provides a mock ComputerPrimitives."""
    provider = MagicMock(spec=ComputerPrimitives)
    provider.computer = MagicMock()
    provider.computer.act = AsyncMock(return_value="Action completed.")
    provider.computer.observe = AsyncMock(return_value="Observation complete.")
    provider.computer.get_screenshot = AsyncMock(return_value=b"fake_screenshot")
    provider.computer.get_current_url = AsyncMock(return_value="https://example.com")
    provider.computer.backend = MagicMock()
    provider.click = AsyncMock(return_value=None)
    provider.close = AsyncMock()
    return provider


@pytest.fixture
def mock_primitives():
    """Create a mock primitives object for testing."""
    primitives = MagicMock()
    primitives.contacts = MagicMock()
    primitives.contacts.ask = AsyncMock(return_value="Alice is a contact")
    primitives.knowledge = MagicMock()
    primitives.knowledge.ask = AsyncMock(return_value="Knowledge result")
    return primitives


# ────────────────────────────────────────────────────────────────────────────
# VenvFunctionProxy Unit Tests
# ────────────────────────────────────────────────────────────────────────────


def test_proxy_has_correct_metadata(mock_primitives, mock_computer_primitives):
    """Proxy should expose function name and docstring."""
    func_data = {
        "name": "test_function",
        "venv_id": 1,
        "implementation": SIMPLE_VENV_FUNCTION,
        "docstring": "Add two numbers in the venv.",
        "argspec": "(a: int, b: int) -> int",
    }
    mock_fm = MagicMock(spec=FunctionManager)
    mock_plan = MagicMock()
    mock_plan.action_log = []

    proxy = _VenvFunctionProxy(
        function_manager=mock_fm,
        func_data=func_data,
        plan=mock_plan,
        primitives=mock_primitives,
        computer_primitives=mock_computer_primitives,
    )

    assert proxy.__name__ == "test_function"
    assert proxy.__doc__ == "Add two numbers in the venv."


@pytest.mark.asyncio
async def test_proxy_calls_execute_in_venv(
    mock_primitives,
    mock_computer_primitives,
):
    """Proxy should call execute_in_venv when invoked."""
    func_data = {
        "name": "simple_add",
        "venv_id": 1,
        "implementation": SIMPLE_VENV_FUNCTION,
        "docstring": "Add two numbers.",
        "argspec": "(a: int, b: int) -> int",
    }
    mock_fm = MagicMock(spec=FunctionManager)
    mock_fm.execute_in_venv = AsyncMock(
        return_value={"result": 5, "error": None, "stdout": "", "stderr": ""},
    )
    mock_plan = MagicMock()
    mock_plan.action_log = []

    proxy = _VenvFunctionProxy(
        function_manager=mock_fm,
        func_data=func_data,
        plan=mock_plan,
        primitives=mock_primitives,
        computer_primitives=mock_computer_primitives,
    )

    result = await proxy(a=2, b=3)

    assert result == 5
    mock_fm.execute_in_venv.assert_called_once()
    call_kwargs = mock_fm.execute_in_venv.call_args.kwargs
    assert call_kwargs["venv_id"] == 1
    assert call_kwargs["call_kwargs"] == {"a": 2, "b": 3}


@pytest.mark.asyncio
async def test_proxy_raises_on_error(
    mock_primitives,
    mock_computer_primitives,
):
    """Proxy should raise RuntimeError when venv execution fails."""
    func_data = {
        "name": "failing_function",
        "venv_id": 1,
        "implementation": "async def fail(): raise ValueError('oops')",
        "docstring": "A failing function.",
        "argspec": "() -> None",
    }
    mock_fm = MagicMock(spec=FunctionManager)
    mock_fm.execute_in_venv = AsyncMock(
        return_value={
            "result": None,
            "error": "ValueError: oops",
            "stdout": "",
            "stderr": "Traceback...",
        },
    )
    mock_plan = MagicMock()
    mock_plan.action_log = []

    proxy = _VenvFunctionProxy(
        function_manager=mock_fm,
        func_data=func_data,
        plan=mock_plan,
        primitives=mock_primitives,
        computer_primitives=mock_computer_primitives,
    )

    with pytest.raises(RuntimeError, match="ValueError: oops"):
        await proxy()


@pytest.mark.asyncio
async def test_proxy_logs_calls(
    mock_primitives,
    mock_computer_primitives,
):
    """Proxy should log calls to action_log."""
    func_data = {
        "name": "logged_function",
        "venv_id": 1,
        "implementation": SIMPLE_VENV_FUNCTION,
        "docstring": "A logged function.",
        "argspec": "(a: int, b: int) -> int",
    }
    mock_fm = MagicMock(spec=FunctionManager)
    mock_fm.execute_in_venv = AsyncMock(
        return_value={"result": 10, "error": None, "stdout": "", "stderr": ""},
    )
    mock_plan = MagicMock()
    mock_plan.action_log = []

    proxy = _VenvFunctionProxy(
        function_manager=mock_fm,
        func_data=func_data,
        plan=mock_plan,
        primitives=mock_primitives,
        computer_primitives=mock_computer_primitives,
    )

    await proxy(a=5, b=5)

    assert len(mock_plan.action_log) == 1
    assert "logged_function" in mock_plan.action_log[0]
    assert "a=5" in mock_plan.action_log[0]


# ────────────────────────────────────────────────────────────────────────────
# HierarchicalActor Venv Integration Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_inject_venv_function_proxies(
    function_manager_factory,
    mock_computer_primitives,
):
    """_inject_venv_function_proxies should add venv functions to execution_namespace."""
    fm = function_manager_factory()

    # Create a venv and add a function to it
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Add the function with venv_id
    fm.add_functions(implementations=[SIMPLE_VENV_FUNCTION])

    # Update the function to have the venv_id
    fm.set_function_venv(function_name="simple_add", venv_id=venv_id)

    # Create a mock actor and plan
    with patch.object(
        HierarchicalActor,
        "__init__",
        lambda self, **kwargs: None,
    ):
        actor = HierarchicalActor()
        actor.function_manager = fm

        # Create a mock plan with execution_namespace
        mock_plan = MagicMock()
        mock_plan.execution_namespace = {}
        mock_plan.action_log = []

        # Inject venv function proxies
        await actor._inject_venv_function_proxies(mock_plan, mock_computer_primitives)

        # Check that the proxy was injected
        assert "simple_add" in mock_plan.execution_namespace
        proxy = mock_plan.execution_namespace["simple_add"]
        assert isinstance(proxy, _VenvFunctionProxy)
        assert proxy.__name__ == "simple_add"


NORMAL_ADD_FUNCTION = """
async def normal_add(a: int, b: int) -> int:
    \"\"\"Add two numbers normally.\"\"\"
    return a + b
""".strip()


@_handle_project
@pytest.mark.asyncio
async def test_inject_library_functions_skips_venv_functions(
    function_manager_factory,
):
    """_inject_library_functions should skip venv functions (they're injected as proxies)."""
    fm = function_manager_factory()

    # Create a venv and add a venv function
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    # Add both functions
    fm.add_functions(implementations=[SIMPLE_VENV_FUNCTION, NORMAL_ADD_FUNCTION])

    # Set the venv_id for the venv function
    fm.set_function_venv(function_name="simple_add", venv_id=venv_id)

    # Create a mock actor
    with patch.object(
        HierarchicalActor,
        "__init__",
        lambda self, **kwargs: None,
    ):
        actor = HierarchicalActor()
        actor.function_manager = fm

        # Test code that references both functions
        base_code = """
async def main():
    x = await simple_add(1, 2)
    y = await normal_add(3, 4)
    return x + y
"""

        injected_code, skip_verify = await actor._inject_library_functions(base_code)

        # Normal function should be injected as code
        assert "normal_add" in injected_code
        assert "async def normal_add" in injected_code

        # Venv function (simple_add) should NOT be injected as code
        # Count occurrences - if it appears, it should only be from the call site
        simple_add_count = injected_code.count("simple_add")
        async_def_simple_add_count = injected_code.count("async def simple_add")

        # The implementation should not be injected
        assert (
            async_def_simple_add_count == 0
        ), "Venv function implementation should not be injected"


VENV_FUNC_IMPL = """
async def venv_func(x: int) -> int:
    \"\"\"Venv function.\"\"\"
    return x * 2
""".strip()


@_handle_project
@pytest.mark.asyncio
async def test_venv_functions_listed_in_search(function_manager_factory):
    """Venv functions should be returned by search_functions for injection detection."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    fm.add_functions(implementations=[VENV_FUNC_IMPL])
    fm.set_function_venv(function_name="venv_func", venv_id=venv_id)

    # Search for venv functions specifically
    venv_functions = fm.search_functions(filter="venv_id != None", limit=100)

    assert len(venv_functions) == 1
    assert venv_functions[0]["name"] == "venv_func"
    assert venv_functions[0]["venv_id"] == venv_id


# ────────────────────────────────────────────────────────────────────────────
# End-to-End Tests (Mocked)
# ────────────────────────────────────────────────────────────────────────────


MULTIPLY_IMPL = """
async def multiply(a: int, b: int) -> int:
    \"\"\"Multiply two numbers.\"\"\"
    return a * b
""".strip()


@_handle_project
@pytest.mark.asyncio
async def test_proxy_call_with_mock_execute_in_venv(
    function_manager_factory,
    mock_primitives,
    mock_computer_primitives,
):
    """Test the full flow of calling a venv function proxy with mocked execution."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    fm.add_functions(implementations=[MULTIPLY_IMPL])
    fm.set_function_venv(function_name="multiply", venv_id=venv_id)

    # Create a mock plan
    mock_plan = MagicMock()
    mock_plan.action_log = []

    # Get the function data
    funcs = fm.search_functions(filter="venv_id != None", limit=1)
    func_data = funcs[0]

    # Mock execute_in_venv
    with patch.object(
        fm,
        "execute_in_venv",
        new=AsyncMock(
            return_value={
                "result": 42,
                "error": None,
                "stdout": "",
                "stderr": "",
            },
        ),
    ):
        proxy = _VenvFunctionProxy(
            function_manager=fm,
            func_data=func_data,
            plan=mock_plan,
            primitives=mock_primitives,
            computer_primitives=mock_computer_primitives,
        )

        result = await proxy(a=6, b=7)

        assert result == 42
        fm.execute_in_venv.assert_called_once()


@_handle_project
@pytest.mark.asyncio
async def test_proxy_handles_stdout_capture(
    function_manager_factory,
    mock_primitives,
    mock_computer_primitives,
):
    """Test that stdout from venv function is logged."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    func_data = {
        "name": "print_func",
        "venv_id": venv_id,
        "implementation": "async def print_func():\n    print('Hello from venv')\n    return 'done'",
        "docstring": "Prints and returns.",
        "argspec": "() -> str",
    }

    mock_plan = MagicMock()
    mock_plan.action_log = []

    with patch.object(
        fm,
        "execute_in_venv",
        new=AsyncMock(
            return_value={
                "result": "done",
                "error": None,
                "stdout": "Hello from venv\n",
                "stderr": "",
            },
        ),
    ):
        proxy = _VenvFunctionProxy(
            function_manager=fm,
            func_data=func_data,
            plan=mock_plan,
            primitives=mock_primitives,
            computer_primitives=mock_computer_primitives,
        )

        result = await proxy()

        assert result == "done"
        # Check that stdout was logged
        stdout_logged = any("Hello from venv" in log for log in mock_plan.action_log)
        assert stdout_logged, f"Expected stdout in logs, got: {mock_plan.action_log}"


# ────────────────────────────────────────────────────────────────────────────
# E2E Tests - Real Venv Execution with RPC
# ────────────────────────────────────────────────────────────────────────────

import shutil


@pytest.fixture
def cleanup_venvs(function_manager_factory):
    """Fixture to clean up venv directories after tests."""
    fm = function_manager_factory()
    venv_ids = []

    def track(venv_id):
        venv_ids.append(venv_id)
        return venv_id

    yield fm, track

    for venv_id in venv_ids:
        try:
            venv_dir = fm._get_venv_dir(venv_id)
            if venv_dir.exists():
                shutil.rmtree(venv_dir, ignore_errors=True)
        except Exception:
            pass


VENV_RPC_FUNCTION = """
async def get_contact_via_rpc(question: str):
    '''Get contact info via RPC to primitives.'''
    result = await primitives.contacts.ask(question=question)
    return f"Got: {result}"
""".strip()


@_handle_project
@pytest.mark.asyncio
async def test_proxy_e2e_with_real_rpc(cleanup_venvs):
    """E2E test: _VenvFunctionProxy should work with real RPC to primitives."""
    fm, track = cleanup_venvs

    venv_id = track(fm.add_venv(venv=MINIMAL_VENV_CONTENT))

    func_data = {
        "name": "get_contact_via_rpc",
        "venv_id": venv_id,
        "implementation": VENV_RPC_FUNCTION,
        "docstring": "Get contact info via RPC.",
        "argspec": "(question: str) -> str",
    }

    mock_plan = MagicMock()
    mock_plan.action_log = []

    # Real primitives mock that will be called via RPC
    real_primitives = MagicMock()
    real_primitives.contacts = MagicMock()
    real_primitives.contacts.ask = AsyncMock(return_value="Alice is contact #1")

    proxy = _VenvFunctionProxy(
        function_manager=fm,
        func_data=func_data,
        plan=mock_plan,
        primitives=real_primitives,
        computer_primitives=MagicMock(),
    )

    result = await proxy(question="Who is the first contact?")

    assert "Got: Alice is contact #1" in result
    real_primitives.contacts.ask.assert_called_once_with(
        question="Who is the first contact?",
    )


VENV_COMPUTER_RPC_FUNCTION = """
async def click_button(selector: str):
    '''Click a button via computer primitives RPC.'''
    await computer_primitives.click(selector=selector)
    return f"Clicked: {selector}"
""".strip()


@_handle_project
@pytest.mark.asyncio
async def test_proxy_e2e_with_computer_primitives_rpc(cleanup_venvs):
    """E2E test: _VenvFunctionProxy should work with computer primitives RPC."""
    fm, track = cleanup_venvs

    venv_id = track(fm.add_venv(venv=MINIMAL_VENV_CONTENT))

    func_data = {
        "name": "click_button",
        "venv_id": venv_id,
        "implementation": VENV_COMPUTER_RPC_FUNCTION,
        "docstring": "Click a button.",
        "argspec": "(selector: str) -> str",
    }

    mock_plan = MagicMock()
    mock_plan.action_log = []

    mock_computer = MagicMock()
    mock_computer.click = AsyncMock(return_value=None)

    proxy = _VenvFunctionProxy(
        function_manager=fm,
        func_data=func_data,
        plan=mock_plan,
        primitives=MagicMock(),
        computer_primitives=mock_computer,
    )

    result = await proxy(selector="#submit-btn")

    assert "Clicked: #submit-btn" in result
    mock_computer.click.assert_called_once_with(selector="#submit-btn")


VENV_MIXED_RPC_FUNCTION = """
async def search_and_click(query: str, button: str):
    '''Search via primitives and click via computer.'''
    search_result = await primitives.knowledge.ask(question=query)
    await computer_primitives.click(selector=button)
    return f"Found '{search_result}', clicked '{button}'"
""".strip()


@_handle_project
@pytest.mark.asyncio
async def test_proxy_e2e_mixed_primitives_and_computer(cleanup_venvs):
    """E2E test: Function can use both primitives and computer_primitives."""
    fm, track = cleanup_venvs

    venv_id = track(fm.add_venv(venv=MINIMAL_VENV_CONTENT))

    func_data = {
        "name": "search_and_click",
        "venv_id": venv_id,
        "implementation": VENV_MIXED_RPC_FUNCTION,
        "docstring": "Search and click.",
        "argspec": "(query: str, button: str) -> str",
    }

    mock_plan = MagicMock()
    mock_plan.action_log = []

    mock_primitives = MagicMock()
    mock_primitives.knowledge = MagicMock()
    mock_primitives.knowledge.ask = AsyncMock(return_value="Result from knowledge")

    mock_computer = MagicMock()
    mock_computer.click = AsyncMock(return_value=None)

    proxy = _VenvFunctionProxy(
        function_manager=fm,
        func_data=func_data,
        plan=mock_plan,
        primitives=mock_primitives,
        computer_primitives=mock_computer,
    )

    result = await proxy(query="What is X?", button="#confirm")

    assert "Found 'Result from knowledge'" in result
    assert "clicked '#confirm'" in result
    mock_primitives.knowledge.ask.assert_called_once()
    mock_computer.click.assert_called_once()


VENV_ERROR_FUNCTION = """
async def failing_rpc():
    '''Function that gets an RPC error.'''
    result = await primitives.contacts.ask(question="cause error")
    return result
""".strip()


@_handle_project
@pytest.mark.asyncio
async def test_proxy_e2e_rpc_error_propagates(cleanup_venvs):
    """E2E test: RPC errors should propagate through the proxy."""
    fm, track = cleanup_venvs

    venv_id = track(fm.add_venv(venv=MINIMAL_VENV_CONTENT))

    func_data = {
        "name": "failing_rpc",
        "venv_id": venv_id,
        "implementation": VENV_ERROR_FUNCTION,
        "docstring": "Function that fails.",
        "argspec": "() -> str",
    }

    mock_plan = MagicMock()
    mock_plan.action_log = []

    mock_primitives = MagicMock()
    mock_primitives.contacts = MagicMock()
    mock_primitives.contacts.ask = AsyncMock(
        side_effect=RuntimeError("Database connection failed"),
    )

    proxy = _VenvFunctionProxy(
        function_manager=fm,
        func_data=func_data,
        plan=mock_plan,
        primitives=mock_primitives,
        computer_primitives=MagicMock(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await proxy()

    assert "Database connection failed" in str(exc_info.value)
