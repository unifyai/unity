"""
Tests for Actor discovering and using DataManager primitives.

These tests verify that:
1. DataManager is properly registered in Primitives
2. Actor can access DataManager methods via primitives.data
3. Basic DataManager operations work within Actor context
"""

from __future__ import annotations


from unity.function_manager.primitives import Primitives


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

    # Plot operations
    assert callable(getattr(dm, "plot", None))
    assert callable(getattr(dm, "plot_batch", None))


def test_data_manager_metadata_registered():
    """DataManager metadata should be in MANAGER_METADATA."""
    from unity.function_manager.primitives import MANAGER_METADATA

    assert "data" in MANAGER_METADATA
    meta = MANAGER_METADATA["data"]
    assert meta["domain"] == "Data Operations & Pipelines"
    assert "filter" in meta["methods"]
    assert "search" in meta["methods"]
    assert "reduce" in meta["methods"]


def test_data_manager_in_primitive_classes():
    """DataManager should be in PRIMITIVE_CLASSES."""
    from unity.function_manager.primitives import PRIMITIVE_CLASSES
    from unity.data_manager.data_manager import DataManager

    assert DataManager in PRIMITIVE_CLASSES


def test_data_manager_class_to_getter():
    """DataManager should have entry in _CLASS_TO_GETTER."""
    from unity.function_manager.primitives import _CLASS_TO_GETTER

    assert "DataManager" in _CLASS_TO_GETTER
    assert _CLASS_TO_GETTER["DataManager"] == "get_data_manager"


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
