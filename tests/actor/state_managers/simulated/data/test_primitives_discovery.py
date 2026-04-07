"""
Tests for Actor discovering and using DataManager primitives.

These tests verify that:
1. DataManager is properly registered in Primitives
2. Actor can access DataManager methods via primitives.data
3. Basic DataManager operations work within Actor context
4. Sync methods are wrapped to be async (without modifying the singleton)
"""

from __future__ import annotations

import asyncio

from unity.function_manager.primitives import (
    Primitives,
    _AsyncPrimitiveWrapper,
    _create_async_wrapper,
)

# ────────────────────────────────────────────────────────────────────────────
# Discovery Tests
# ────────────────────────────────────────────────────────────────────────────


def test_data_manager_in_primitives():
    """DataManager should be accessible via primitives.data."""
    primitives = Primitives()
    dm = primitives.data
    assert dm is not None
    assert hasattr(dm, "filter")
    assert hasattr(dm, "search")
    assert hasattr(dm, "reduce")
    assert hasattr(dm, "filter_join")
    assert hasattr(dm, "reduce_join")
    assert hasattr(dm, "search_join")


def test_data_manager_has_expected_methods():
    """DataManager should have all expected public methods."""
    primitives = Primitives()
    dm = primitives.data

    # Table operations
    assert callable(getattr(dm, "create_table", None))
    assert callable(getattr(dm, "describe_table", None))
    assert callable(getattr(dm, "list_tables", None))
    assert callable(getattr(dm, "delete_table", None))

    # Query operations
    assert callable(getattr(dm, "filter", None))
    assert callable(getattr(dm, "search", None))
    assert callable(getattr(dm, "reduce", None))

    # Join operations
    assert callable(getattr(dm, "filter_join", None))
    assert callable(getattr(dm, "reduce_join", None))
    assert callable(getattr(dm, "search_join", None))
    assert callable(getattr(dm, "filter_multi_join", None))
    assert callable(getattr(dm, "search_multi_join", None))

    # Mutation operations
    assert callable(getattr(dm, "insert_rows", None))
    assert callable(getattr(dm, "update_rows", None))
    assert callable(getattr(dm, "delete_rows", None))

    # Embedding operations
    assert callable(getattr(dm, "ensure_vector_column", None))
    assert callable(getattr(dm, "vectorize_rows", None))


def test_data_manager_metadata_registered():
    """DataManager metadata should be in ToolSurfaceRegistry."""
    from unity.function_manager.primitives import get_registry

    registry = get_registry()
    spec = registry.get_manager_spec("data")
    assert spec is not None
    assert spec.domain == "Data Operations & Ingestion"
    methods = registry.primitive_methods(manager_alias="data")
    assert "filter" in methods
    assert "search" in methods
    assert "reduce" in methods


def test_data_manager_in_primitive_registry():
    """DataManager should be registered in ToolSurfaceRegistry."""
    from unity.function_manager.primitives import get_registry

    registry = get_registry()
    spec = registry.get_manager_spec("data")
    assert spec is not None
    assert spec.primitive_class_path == "unity.data_manager.data_manager.DataManager"


def test_data_manager_alias_to_getter():
    """DataManager should have entry in _ALIAS_TO_GETTER."""
    from unity.function_manager.primitives.runtime import _ALIAS_TO_GETTER

    assert "data" in _ALIAS_TO_GETTER
    assert _ALIAS_TO_GETTER["data"] == "get_data_manager"


# ────────────────────────────────────────────────────────────────────────────
# Integration with FileManager
# ────────────────────────────────────────────────────────────────────────────


def test_file_manager_has_data_manager_access():
    """FileManager should delegate to DataManager internally."""
    primitives = Primitives()
    fm = primitives.files

    # FileManager should have a _data_manager property
    assert hasattr(fm, "_data_manager")
    dm = fm._data_manager
    assert dm is not None

    # Both should be the same type
    from unity.data_manager.base import BaseDataManager

    assert isinstance(dm, BaseDataManager)


# ────────────────────────────────────────────────────────────────────────────
# Async Patching Tests
# ────────────────────────────────────────────────────────────────────────────


def test_data_manager_sync_methods_patched_to_async():
    """DataManager sync methods should be patched to be async via primitives."""
    primitives = Primitives()
    dm = primitives.data

    # DataManager has sync methods that should now be async
    assert asyncio.iscoroutinefunction(
        dm.filter,
    ), "filter should be async after patching"
    assert asyncio.iscoroutinefunction(
        dm.search,
    ), "search should be async after patching"
    assert asyncio.iscoroutinefunction(
        dm.reduce,
    ), "reduce should be async after patching"
    assert asyncio.iscoroutinefunction(
        dm.insert_rows,
    ), "insert_rows should be async after patching"


def test_file_manager_sync_methods_patched_to_async():
    """FileManager sync methods should be patched to be async via primitives."""
    primitives = Primitives()
    fm = primitives.files

    # FileManager has sync methods that should now be async
    assert asyncio.iscoroutinefunction(
        fm.describe,
    ), "describe should be async after patching"
    assert asyncio.iscoroutinefunction(
        fm.filter_files,
    ), "filter_files should be async after patching"
    assert asyncio.iscoroutinefunction(
        fm.reduce,
    ), "reduce should be async after patching"


def test_file_manager_already_async_methods_unchanged():
    """FileManager methods that are already async should remain async."""
    primitives = Primitives()
    fm = primitives.files

    # ask_about_file is already async - should still be async
    assert asyncio.iscoroutinefunction(
        fm.ask_about_file,
    ), "ask_about_file should remain async"


def test_async_patching_is_idempotent():
    """Creating multiple wrappers for the same manager should work."""
    from unity.manager_registry import ManagerRegistry

    dm = ManagerRegistry.get_data_manager()

    # Create two wrappers (using manager alias, not class name)
    wrapper1 = _create_async_wrapper(dm, "data")
    wrapper2 = _create_async_wrapper(dm, "data")

    # Both wrappers should work
    assert asyncio.iscoroutinefunction(wrapper1.filter)
    assert asyncio.iscoroutinefunction(wrapper2.filter)

    # Original manager should not be modified
    assert not asyncio.iscoroutinefunction(dm.filter), "Original sync method unchanged"


def test_primitives_returns_async_wrapper():
    """primitives.data should return an async wrapper that delegates to DataManager."""
    from unity.manager_registry import ManagerRegistry

    primitives = Primitives()
    dm_wrapper = primitives.data

    # Should be a wrapper instance
    assert isinstance(dm_wrapper, _AsyncPrimitiveWrapper), "Should return wrapper"

    # Should have async methods
    assert asyncio.iscoroutinefunction(dm_wrapper.filter)

    # Original singleton should remain sync
    dm_original = ManagerRegistry.get_data_manager()
    assert not asyncio.iscoroutinefunction(dm_original.filter), "Original unchanged"


def test_primitives_files_returns_async_wrapper():
    """primitives.files should return an async wrapper that delegates to FileManager."""
    from unity.manager_registry import ManagerRegistry

    primitives = Primitives()
    fm_wrapper = primitives.files

    # Should be a wrapper instance
    assert isinstance(fm_wrapper, _AsyncPrimitiveWrapper), "Should return wrapper"

    # Original singleton should remain sync (for methods that are sync)
    fm_original = ManagerRegistry.get_file_manager()
    # describe is sync on the original
    assert not asyncio.iscoroutinefunction(fm_original.describe), "Original unchanged"


def test_async_managers_not_wrapped():
    """Managers with async methods should be returned directly."""
    primitives = Primitives()

    # ContactManager has async methods - returned directly, not wrapped
    cm = primitives.contacts
    assert asyncio.iscoroutinefunction(cm.ask), "ContactManager.ask should be async"
    assert asyncio.iscoroutinefunction(
        cm.update,
    ), "ContactManager.update should be async"
