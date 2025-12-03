"""
Tests for custom function collection and synchronization.

Tests the auto-sync mechanism for functions defined in the custom/ folder.
"""

import pytest

from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.custom_functions import (
    collect_custom_functions,
    compute_custom_functions_hash,
)
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project


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
# 1. Collection Tests
# ────────────────────────────────────────────────────────────────────────────


def test_collect_custom_functions_finds_decorated_functions():
    """collect_custom_functions should find functions with @custom_function decorator."""
    functions = collect_custom_functions()

    # Should have found the example functions
    assert "example_add" in functions
    assert "example_uppercase" in functions


def test_collect_custom_functions_excludes_auto_sync_false():
    """Functions with auto_sync=False should not be collected."""
    functions = collect_custom_functions()

    # draft_function_not_synced has auto_sync=False
    assert "draft_function_not_synced" not in functions


def test_collect_custom_functions_has_required_fields():
    """Collected functions should have all required metadata fields."""
    functions = collect_custom_functions()

    assert "example_add" in functions
    func = functions["example_add"]

    # Check required fields
    assert "name" in func
    assert func["name"] == "example_add"
    assert "argspec" in func
    assert "a: int" in func["argspec"]
    assert "b: int" in func["argspec"]
    assert "docstring" in func
    assert "Add two integers" in func["docstring"]
    assert "implementation" in func
    assert "return a + b" in func["implementation"]
    assert "custom_hash" in func
    assert len(func["custom_hash"]) == 16  # SHA256 truncated to 16 chars
    assert "embedding_text" in func
    assert func["is_primitive"] is False


def test_collect_custom_functions_respects_decorator_options():
    """Decorator options should be reflected in collected metadata."""
    functions = collect_custom_functions()

    # example_add has default verify=True
    assert functions["example_add"]["verify"] is True

    # example_uppercase has verify=False
    assert functions["example_uppercase"]["verify"] is False


def test_compute_custom_functions_hash_is_deterministic():
    """The aggregate hash should be deterministic."""
    hash1 = compute_custom_functions_hash()
    hash2 = compute_custom_functions_hash()

    assert hash1 == hash2
    assert len(hash1) == 16


# ────────────────────────────────────────────────────────────────────────────
# 2. Sync Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_functions_inserts_new_functions(function_manager_factory):
    """sync_custom_functions should insert new functions into the DB."""
    fm = function_manager_factory()

    # Initial sync should insert functions
    result = fm.sync_custom_functions()

    assert result is True  # Sync was performed

    # Check that functions are in the DB
    functions = fm.list_functions()

    assert "example_add" in functions
    assert "example_uppercase" in functions
    assert "draft_function_not_synced" not in functions  # auto_sync=False


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_functions_is_idempotent(function_manager_factory):
    """Calling sync_custom_functions twice should not re-sync if unchanged."""
    fm = function_manager_factory()

    # First sync
    result1 = fm.sync_custom_functions()
    assert result1 is True

    # Reset the synced flag to allow re-checking
    fm._custom_functions_synced = False

    # Second sync should skip (hash matches)
    result2 = fm.sync_custom_functions()
    assert result2 is False


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_functions_preserves_function_id(function_manager_factory):
    """Sync should preserve function_id when content matches."""
    fm = function_manager_factory()

    # First sync
    fm.sync_custom_functions()
    functions = fm.list_functions()
    original_id = functions["example_add"]["function_id"]

    # Reset and sync again
    fm._custom_functions_synced = False
    fm.sync_custom_functions()

    functions = fm.list_functions()
    assert functions["example_add"]["function_id"] == original_id


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_functions_has_custom_hash(function_manager_factory):
    """Synced custom functions should have custom_hash field set."""
    fm = function_manager_factory()

    fm.sync_custom_functions()

    # Get the full function data including custom_hash
    db_functions = fm._get_custom_functions_from_db()

    assert "example_add" in db_functions
    assert db_functions["example_add"]["custom_hash"] is not None
    assert len(db_functions["example_add"]["custom_hash"]) == 16


@_handle_project
@pytest.mark.asyncio
async def test_sync_overwrites_user_function_with_same_name(function_manager_factory):
    """Custom functions should overwrite user-added functions with same name."""
    fm = function_manager_factory()

    # Add a user function with the same name as a custom function
    user_impl = """
async def example_add(a: int, b: int) -> int:
    '''User version of example_add.'''
    return a + b + 100  # Different implementation
"""
    fm.add_functions(implementations=[user_impl])

    # Verify user function was added
    functions = fm.list_functions()
    assert "example_add" in functions
    user_function_id = functions["example_add"]["function_id"]

    # Now sync custom functions - should overwrite
    fm.sync_custom_functions()

    # The function should now be the custom version
    functions = fm.list_functions(include_implementations=True)
    assert "example_add" in functions

    # The implementation should be from the custom folder, not user
    assert "return a + b + 100" not in functions["example_add"]["implementation"]
    assert "return a + b" in functions["example_add"]["implementation"]

    # It should have a custom_hash now
    db_functions = fm._get_custom_functions_from_db()
    assert "example_add" in db_functions


@_handle_project
@pytest.mark.asyncio
async def test_user_function_without_custom_hash_is_preserved(function_manager_factory):
    """User-added functions with different names should not be affected."""
    fm = function_manager_factory()

    # Add a user function with a unique name
    user_impl = """
async def my_unique_user_function(x: int) -> int:
    '''A unique user function.'''
    return x * 3
"""
    fm.add_functions(implementations=[user_impl])

    # Sync custom functions
    fm.sync_custom_functions()

    # User function should still be there
    functions = fm.list_functions()
    assert "my_unique_user_function" in functions
    assert "example_add" in functions  # Custom function also there
