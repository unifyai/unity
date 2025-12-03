"""
Tests for virtual environment storage and foreign key cascading in FunctionManager.

Tests the Functions/VirtualEnvs context, including:
- Creating and listing virtual environments
- Associating functions with virtual environments
- Cascading SET NULL on venv deletion
"""

import pytest

from unity.function_manager.function_manager import FunctionManager
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project


# Sample pyproject.toml content for testing
SAMPLE_VENV_CONTENT = """
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "test-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
]
""".strip()

SAMPLE_VENV_CONTENT_2 = """
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "ml-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
]
""".strip()

SIMPLE_FUNCTION = """
async def add_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers together.\"\"\"
    return a + b
""".strip()

SIMPLE_FUNCTION_2 = """
async def multiply_numbers(a: int, b: int) -> int:
    \"\"\"Multiply two numbers together.\"\"\"
    return a * b
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager_factory():
    """
    Factory fixture that creates FunctionManager instances.

    Returns a callable that creates a FunctionManager. This ensures the
    FunctionManager is instantiated AFTER @_handle_project sets up the
    test-specific context, providing proper isolation for parallel tests.
    """
    managers = []

    def _create():
        # Forget FunctionManager's cached contexts to ensure we get
        # fresh contexts for this test's active context (set by @_handle_project)
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
# 1. Virtual Environment CRUD Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_add_venv_returns_id(function_manager_factory):
    """add_venv should return an auto-assigned venv_id."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)

    assert isinstance(venv_id, int)
    assert venv_id >= 0


@_handle_project
@pytest.mark.asyncio
async def test_add_multiple_venvs_have_unique_ids(function_manager_factory):
    """Multiple venvs should have unique auto-incrementing IDs."""
    fm = function_manager_factory()

    venv_id_1 = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    venv_id_2 = fm.add_venv(venv=SAMPLE_VENV_CONTENT_2)

    assert venv_id_1 != venv_id_2
    assert isinstance(venv_id_1, int)
    assert isinstance(venv_id_2, int)


@_handle_project
@pytest.mark.asyncio
async def test_get_venv_returns_content(function_manager_factory):
    """get_venv should return the venv content by ID."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    result = fm.get_venv(venv_id=venv_id)

    assert result is not None
    assert result["venv_id"] == venv_id
    assert result["venv"] == SAMPLE_VENV_CONTENT


@_handle_project
@pytest.mark.asyncio
async def test_get_venv_nonexistent_returns_none(function_manager_factory):
    """get_venv should return None for non-existent ID."""
    fm = function_manager_factory()

    result = fm.get_venv(venv_id=99999)

    assert result is None


@_handle_project
@pytest.mark.asyncio
async def test_list_venvs_returns_all(function_manager_factory):
    """list_venvs should return all virtual environments."""
    fm = function_manager_factory()

    venv_id_1 = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    venv_id_2 = fm.add_venv(venv=SAMPLE_VENV_CONTENT_2)

    venvs = fm.list_venvs()

    assert len(venvs) == 2
    venv_ids = {v["venv_id"] for v in venvs}
    assert venv_id_1 in venv_ids
    assert venv_id_2 in venv_ids


@_handle_project
@pytest.mark.asyncio
async def test_delete_venv_removes_entry(function_manager_factory):
    """delete_venv should remove the virtual environment."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    assert fm.get_venv(venv_id=venv_id) is not None

    result = fm.delete_venv(venv_id=venv_id)

    assert result is True
    assert fm.get_venv(venv_id=venv_id) is None


@_handle_project
@pytest.mark.asyncio
async def test_delete_venv_nonexistent_returns_false(function_manager_factory):
    """delete_venv should return False for non-existent ID."""
    fm = function_manager_factory()

    result = fm.delete_venv(venv_id=99999)

    assert result is False


@_handle_project
@pytest.mark.asyncio
async def test_update_venv_changes_content(function_manager_factory):
    """update_venv should change the venv content."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    result = fm.update_venv(venv_id=venv_id, venv=SAMPLE_VENV_CONTENT_2)

    assert result is True
    updated = fm.get_venv(venv_id=venv_id)
    assert updated["venv"] == SAMPLE_VENV_CONTENT_2


@_handle_project
@pytest.mark.asyncio
async def test_update_venv_nonexistent_returns_false(function_manager_factory):
    """update_venv should return False for non-existent ID."""
    fm = function_manager_factory()

    result = fm.update_venv(venv_id=99999, venv=SAMPLE_VENV_CONTENT)

    assert result is False


# ────────────────────────────────────────────────────────────────────────────
# 2. Function-VirtualEnv Association Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_function_venv_id_defaults_to_none(function_manager_factory):
    """Functions should have venv_id=None by default."""
    fm = function_manager_factory()

    fm.add_functions(implementations=[SIMPLE_FUNCTION])
    functions = fm.list_functions()

    assert "add_numbers" in functions
    # The function should exist; venv_id defaults to None
    func_data = functions["add_numbers"]
    # venv_id might not be in the basic list, let's check via get_function_venv
    venv = fm.get_function_venv(function_id=func_data["function_id"])
    assert venv is None


@_handle_project
@pytest.mark.asyncio
async def test_set_function_venv_associates_venv(function_manager_factory):
    """set_function_venv should associate a function with a venv."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    fm.add_functions(implementations=[SIMPLE_FUNCTION])
    functions = fm.list_functions()
    func_id = functions["add_numbers"]["function_id"]

    result = fm.set_function_venv(function_id=func_id, venv_id=venv_id)

    assert result is True
    venv = fm.get_function_venv(function_id=func_id)
    assert venv is not None
    assert venv["venv_id"] == venv_id
    assert venv["venv"] == SAMPLE_VENV_CONTENT


@_handle_project
@pytest.mark.asyncio
async def test_set_function_venv_to_none_removes_association(function_manager_factory):
    """set_function_venv with None should remove the association."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    fm.add_functions(implementations=[SIMPLE_FUNCTION])
    functions = fm.list_functions()
    func_id = functions["add_numbers"]["function_id"]

    fm.set_function_venv(function_id=func_id, venv_id=venv_id)
    assert fm.get_function_venv(function_id=func_id) is not None

    result = fm.set_function_venv(function_id=func_id, venv_id=None)

    assert result is True
    assert fm.get_function_venv(function_id=func_id) is None


@_handle_project
@pytest.mark.asyncio
async def test_set_function_venv_nonexistent_function_returns_false(
    function_manager_factory,
):
    """set_function_venv should return False for non-existent function."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    result = fm.set_function_venv(function_id=99999, venv_id=venv_id)

    assert result is False


# ────────────────────────────────────────────────────────────────────────────
# 3. Cascade Deletion Tests (SET NULL)
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_delete_venv_cascades_set_null_to_functions(function_manager_factory):
    """Deleting a venv should SET NULL on functions that reference it."""
    fm = function_manager_factory()

    # Create venv and associate it with a function
    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    fm.add_functions(implementations=[SIMPLE_FUNCTION])
    functions = fm.list_functions()
    func_id = functions["add_numbers"]["function_id"]
    fm.set_function_venv(function_id=func_id, venv_id=venv_id)

    # Verify the association exists
    assert fm.get_function_venv(function_id=func_id) is not None

    # Delete the venv
    fm.delete_venv(venv_id=venv_id)

    # Function should now have venv_id=None (cascaded SET NULL)
    venv = fm.get_function_venv(function_id=func_id)
    assert venv is None


@_handle_project
@pytest.mark.asyncio
async def test_delete_venv_cascades_to_multiple_functions(function_manager_factory):
    """Deleting a venv should SET NULL on all functions that reference it."""
    fm = function_manager_factory()

    # Create venv and associate it with multiple functions
    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    fm.add_functions(implementations=[SIMPLE_FUNCTION, SIMPLE_FUNCTION_2])
    functions = fm.list_functions()

    func_id_1 = functions["add_numbers"]["function_id"]
    func_id_2 = functions["multiply_numbers"]["function_id"]

    fm.set_function_venv(function_id=func_id_1, venv_id=venv_id)
    fm.set_function_venv(function_id=func_id_2, venv_id=venv_id)

    # Verify both associations exist
    assert fm.get_function_venv(function_id=func_id_1) is not None
    assert fm.get_function_venv(function_id=func_id_2) is not None

    # Delete the venv
    fm.delete_venv(venv_id=venv_id)

    # Both functions should now have venv_id=None
    assert fm.get_function_venv(function_id=func_id_1) is None
    assert fm.get_function_venv(function_id=func_id_2) is None


@_handle_project
@pytest.mark.asyncio
async def test_delete_venv_does_not_affect_functions_with_other_venv(
    function_manager_factory,
):
    """Deleting a venv should not affect functions referencing a different venv."""
    fm = function_manager_factory()

    # Create two venvs
    venv_id_1 = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    venv_id_2 = fm.add_venv(venv=SAMPLE_VENV_CONTENT_2)

    # Create two functions, each with a different venv
    fm.add_functions(implementations=[SIMPLE_FUNCTION, SIMPLE_FUNCTION_2])
    functions = fm.list_functions()

    func_id_1 = functions["add_numbers"]["function_id"]
    func_id_2 = functions["multiply_numbers"]["function_id"]

    fm.set_function_venv(function_id=func_id_1, venv_id=venv_id_1)
    fm.set_function_venv(function_id=func_id_2, venv_id=venv_id_2)

    # Delete only the first venv
    fm.delete_venv(venv_id=venv_id_1)

    # First function should have venv_id=None
    assert fm.get_function_venv(function_id=func_id_1) is None

    # Second function should still have its venv
    venv = fm.get_function_venv(function_id=func_id_2)
    assert venv is not None
    assert venv["venv_id"] == venv_id_2


@_handle_project
@pytest.mark.asyncio
async def test_function_deletion_does_not_delete_venv(function_manager_factory):
    """Deleting a function should not delete the associated venv."""
    fm = function_manager_factory()

    # Create venv and associate with function
    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    fm.add_functions(implementations=[SIMPLE_FUNCTION])
    functions = fm.list_functions()
    func_id = functions["add_numbers"]["function_id"]
    fm.set_function_venv(function_id=func_id, venv_id=venv_id)

    # Delete the function
    fm.delete_function(function_id=func_id)

    # Venv should still exist
    venv = fm.get_venv(venv_id=venv_id)
    assert venv is not None
    assert venv["venv"] == SAMPLE_VENV_CONTENT
