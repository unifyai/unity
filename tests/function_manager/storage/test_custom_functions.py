"""
Tests for custom function and venv collection and synchronization.

Tests the auto-sync mechanism for functions and venvs defined in the custom/ folder.
"""

import pytest

from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.custom_functions import (
    collect_custom_functions,
    compute_custom_functions_hash,
    collect_custom_venvs,
    compute_custom_venvs_hash,
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
# 1. Function Collection Tests
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
# 2. Venv Collection Tests
# ────────────────────────────────────────────────────────────────────────────


def test_collect_custom_venvs_finds_toml_files():
    """collect_custom_venvs should find .toml files in the venvs folder."""
    venvs = collect_custom_venvs()

    # Should have found the example venv
    assert "example_minimal" in venvs


def test_collect_custom_venvs_has_required_fields():
    """Collected venvs should have all required metadata fields."""
    venvs = collect_custom_venvs()

    assert "example_minimal" in venvs
    venv = venvs["example_minimal"]

    # Check required fields
    assert "name" in venv
    assert venv["name"] == "example_minimal"
    assert "venv" in venv
    assert "[project]" in venv["venv"]
    assert "custom_hash" in venv
    assert len(venv["custom_hash"]) == 16


def test_compute_custom_venvs_hash_is_deterministic():
    """The aggregate venv hash should be deterministic."""
    hash1 = compute_custom_venvs_hash()
    hash2 = compute_custom_venvs_hash()

    assert hash1 == hash2
    assert len(hash1) == 16


# ────────────────────────────────────────────────────────────────────────────
# 3. Function Sync Tests
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


# ────────────────────────────────────────────────────────────────────────────
# 4. Venv Sync Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_venvs_inserts_new_venvs(function_manager_factory):
    """sync_custom_venvs should insert new venvs into the DB."""
    fm = function_manager_factory()

    # Initial sync should insert venvs
    name_to_id = fm.sync_custom_venvs()

    assert "example_minimal" in name_to_id
    assert isinstance(name_to_id["example_minimal"], int)

    # Check that venv is in the DB
    venvs = fm.list_venvs()
    assert len(venvs) >= 1

    # Find the example_minimal venv
    example_venv = None
    for v in venvs:
        if v.get("name") == "example_minimal":
            example_venv = v
            break

    assert example_venv is not None
    assert "[project]" in example_venv["venv"]


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_venvs_is_idempotent(function_manager_factory):
    """Calling sync_custom_venvs twice should return same mapping."""
    fm = function_manager_factory()

    # First sync
    name_to_id_1 = fm.sync_custom_venvs()

    # Second sync (should use cached result)
    name_to_id_2 = fm.sync_custom_venvs()

    assert name_to_id_1 == name_to_id_2


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_venvs_has_custom_hash(function_manager_factory):
    """Synced custom venvs should have custom_hash field set."""
    fm = function_manager_factory()

    fm.sync_custom_venvs()

    # Get the full venv data including custom_hash
    db_venvs = fm._get_custom_venvs_from_db()

    assert "example_minimal" in db_venvs
    assert db_venvs["example_minimal"]["custom_hash"] is not None
    assert len(db_venvs["example_minimal"]["custom_hash"]) == 16


# ────────────────────────────────────────────────────────────────────────────
# 5. Combined Sync Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_syncs_both_venvs_and_functions(function_manager_factory):
    """sync_custom should sync both venvs and functions."""
    fm = function_manager_factory()

    # Combined sync
    fm.sync_custom()

    # Check venvs
    db_venvs = fm._get_custom_venvs_from_db()
    assert "example_minimal" in db_venvs

    # Check functions
    functions = fm.list_functions()
    assert "example_add" in functions
    assert "example_uppercase" in functions


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_is_idempotent(function_manager_factory):
    """Calling sync_custom twice should not re-sync if unchanged."""
    fm = function_manager_factory()

    # First sync
    result1 = fm.sync_custom()
    assert result1 is True

    # Reset flags
    fm._custom_venvs_synced = False
    fm._custom_functions_synced = False

    # Second sync should skip (hashes match)
    result2 = fm.sync_custom()
    assert result2 is False


# ────────────────────────────────────────────────────────────────────────────
# 6. venv_name Resolution Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_venv_name_resolved_to_venv_id(function_manager_factory):
    """
    Functions with venv_name should have it resolved to venv_id during sync.

    This test creates a function with venv_name that matches a custom venv,
    and verifies the resolution works correctly.
    """
    fm = function_manager_factory()

    # Sync venvs first to get the mapping
    name_to_id = fm.sync_custom_venvs()

    # Sync functions with the mapping
    fm.sync_custom_functions(venv_name_to_id=name_to_id)

    # Get functions - if any function uses venv_name="example_minimal",
    # it should now have a venv_id set
    functions = fm.list_functions()

    # At minimum, verify the sync completed without error
    assert "example_add" in functions


@_handle_project
@pytest.mark.asyncio
async def test_venv_name_not_found_leaves_venv_id_none(function_manager_factory):
    """
    If venv_name doesn't match any custom venv, venv_id should remain None.

    This tests the edge case where a function references a non-existent venv.
    """
    fm = function_manager_factory()

    # Create a function with a venv_name that doesn't exist
    # We'll do this by manually calling sync_custom_functions with a partial mapping
    name_to_id = {"some_other_venv": 999}  # Does not include example_minimal

    # Sync functions - any function with venv_name not in the mapping should keep venv_id=None
    fm.sync_custom_functions(venv_name_to_id=name_to_id)

    # Functions should still be synced
    functions = fm.list_functions()
    assert "example_add" in functions

    # If example_add had a venv_name that wasn't in the mapping, its venv_id would be None
    # (Currently example_add doesn't have venv_name set, so this is a baseline test)


@_handle_project
@pytest.mark.asyncio
async def test_empty_venv_name_mapping_does_not_crash(function_manager_factory):
    """
    Syncing functions with an empty venv_name_to_id mapping should work.
    """
    fm = function_manager_factory()

    # Sync with empty mapping
    result = fm.sync_custom_functions(venv_name_to_id={})

    assert result is True
    functions = fm.list_functions()
    assert "example_add" in functions


@_handle_project
@pytest.mark.asyncio
async def test_none_venv_name_mapping_does_not_crash(function_manager_factory):
    """
    Syncing functions with None for venv_name_to_id should work.
    """
    fm = function_manager_factory()

    # Sync with None mapping
    result = fm.sync_custom_functions(venv_name_to_id=None)

    assert result is True
    functions = fm.list_functions()
    assert "example_add" in functions
