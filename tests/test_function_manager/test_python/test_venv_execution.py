"""
Tests for executing functions in custom virtual environments.

Tests the venv preparation, subprocess execution, and result handling.
"""

import pytest
import shutil

from unity.function_manager.function_manager import FunctionManager
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project

# Sample pyproject.toml with minimal dependencies (fast to sync)
MINIMAL_VENV_CONTENT = """
[project]
name = "test-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
""".strip()

# A venv with a simple extra dependency
NUMPY_VENV_CONTENT = """
[project]
name = "numpy-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24.0",
]
""".strip()

# Simple sync function
SYNC_FUNCTION = """
def add_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b
""".strip()

# Simple async function
ASYNC_FUNCTION = """
async def multiply_async(x: int, y: int) -> int:
    \"\"\"Multiply two numbers asynchronously.\"\"\"
    import asyncio
    await asyncio.sleep(0.01)
    return x * y
""".strip()

# Function that uses numpy (requires numpy venv)
NUMPY_FUNCTION = """
def create_array(size: int) -> str:
    \"\"\"Create a numpy array and return its string representation.\"\"\"
    import numpy as np
    arr = np.zeros(size)
    return str(arr)
""".strip()

# Function that captures stdout
STDOUT_FUNCTION = """
def print_and_return(msg: str) -> str:
    \"\"\"Print a message and return it.\"\"\"
    print(f"Printed: {msg}")
    return msg
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager_factory():
    """
    Factory fixture that creates FunctionManager instances.
    """
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

    # Cleanup all created managers
    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# 1. Venv Preparation Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_prepare_venv_creates_directory(function_manager_factory):
    """prepare_venv should create the venv directory structure."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        python_path = await fm.prepare_venv(venv_id=venv_id)

        assert python_path.exists()
        assert python_path.name == "python"

        venv_dir = fm._get_venv_dir(venv_id)
        assert venv_dir.exists()
        assert (venv_dir / "pyproject.toml").exists()
        assert (venv_dir / "venv_runner.py").exists()
    finally:
        # Cleanup
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_prepare_venv_is_idempotent(function_manager_factory):
    """Calling prepare_venv twice should return the same path and be quick."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        # First call creates the venv
        python_path_1 = await fm.prepare_venv(venv_id=venv_id)
        assert fm.is_venv_ready(venv_id=venv_id) is True

        # Second call should return the same path (and be much faster since no sync needed)
        python_path_2 = await fm.prepare_venv(venv_id=venv_id)

        assert python_path_1 == python_path_2
        assert fm.is_venv_ready(venv_id=venv_id) is True
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_is_venv_ready_returns_false_initially(function_manager_factory):
    """is_venv_ready should return False for a new venv."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    assert fm.is_venv_ready(venv_id=venv_id) is False


@_handle_project
@pytest.mark.asyncio
async def test_is_venv_ready_returns_true_after_prepare(function_manager_factory):
    """is_venv_ready should return True after prepare_venv."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        await fm.prepare_venv(venv_id=venv_id)
        assert fm.is_venv_ready(venv_id=venv_id) is True
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_prepare_venv_nonexistent_raises(function_manager_factory):
    """prepare_venv should raise for non-existent venv_id."""
    fm = function_manager_factory()

    with pytest.raises(ValueError, match="not found"):
        await fm.prepare_venv(venv_id=99999)


# ────────────────────────────────────────────────────────────────────────────
# 2. Execution Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_execute_sync_function_in_venv(function_manager_factory):
    """execute_in_venv should run a sync function and return the result."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=SYNC_FUNCTION,
            call_kwargs={"a": 3, "b": 5},
            is_async=False,
        )

        assert result["error"] is None
        assert result["result"] == 8
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_execute_async_function_in_venv(function_manager_factory):
    """execute_in_venv should run an async function and return the result."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=ASYNC_FUNCTION,
            call_kwargs={"x": 4, "y": 7},
            is_async=True,
        )

        assert result["error"] is None
        assert result["result"] == 28
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_execute_captures_stdout(function_manager_factory):
    """execute_in_venv should capture stdout from the function."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=STDOUT_FUNCTION,
            call_kwargs={"msg": "hello"},
            is_async=False,
        )

        assert result["error"] is None
        assert result["result"] == "hello"
        assert "Printed: hello" in result["stdout"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_execute_with_custom_dependency(function_manager_factory):
    """execute_in_venv should work with venv-specific dependencies."""
    fm = function_manager_factory()

    # Create venv with numpy
    venv_id = fm.add_venv(venv=NUMPY_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=NUMPY_FUNCTION,
            call_kwargs={"size": 3},
            is_async=False,
        )

        assert result["error"] is None
        # numpy array of zeros should be "[0. 0. 0.]"
        assert "0." in result["result"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_execute_with_missing_dependency_fails(function_manager_factory):
    """execute_in_venv should fail if dependency is missing."""
    fm = function_manager_factory()

    # Create minimal venv (no numpy)
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=NUMPY_FUNCTION,
            call_kwargs={"size": 3},
            is_async=False,
        )

        # Should have an error about missing numpy
        assert result["error"] is not None
        assert (
            "numpy" in result["error"].lower() or "no module" in result["error"].lower()
        )
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_with_error(function_manager_factory):
    """execute_in_venv should capture function errors."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

    error_function = """
def raises_error():
    raise ValueError("Intentional test error")
""".strip()

    try:
        result = await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=error_function,
            call_kwargs={},
            is_async=False,
        )

        assert result["error"] is not None
        assert "Intentional test error" in result["error"]
    finally:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
