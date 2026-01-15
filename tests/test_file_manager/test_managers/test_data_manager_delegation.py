"""
Tests for FileManager delegation to DataManager.

These tests verify that FileManager correctly delegates data operations
to the underlying DataManager instance.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ────────────────────────────────────────────────────────────────────────────
# Delegation smoke tests
# ────────────────────────────────────────────────────────────────────────────


def test_file_manager_has_data_manager_property(file_manager):
    """FileManager should have a _data_manager property."""
    assert hasattr(file_manager, "_data_manager")
    dm = file_manager._data_manager
    assert dm is not None


def test_filter_files_returns_list(file_manager, tmp_path: Path):
    """filter_files should return a list (delegated to DataManager)."""
    # Create and ingest a file
    p = tmp_path / "filter_test.txt"
    p.write_text("test content")
    file_manager.ingest_files(str(p))

    # Filter should return list
    rows = file_manager.filter_files()
    assert isinstance(rows, list)


def test_search_files_returns_list(file_manager, tmp_path: Path):
    """search_files should return a list (delegated to DataManager)."""
    # Create and ingest a file
    p = tmp_path / "search_test.txt"
    p.write_text("test content for semantic search")
    file_manager.ingest_files(str(p))

    # Use describe to get context
    storage = file_manager.describe(file_path=str(p))
    if storage.has_document:
        # Search should return list
        rows = file_manager.search_files(
            context=storage.document.context_path,
            references={"summary": "semantic search"},
            limit=5,
        )
        assert isinstance(rows, list)


def test_reduce_returns_result(file_manager, tmp_path: Path):
    """reduce should return a result (delegated to DataManager)."""
    # Create and ingest multiple files
    for i in range(3):
        p = tmp_path / f"reduce_test_{i}.txt"
        p.write_text(f"content {i}")
        file_manager.ingest_files(str(p))

    # Reduce should return count
    result = file_manager.reduce(metric="count", column="file_id")
    assert isinstance(result, (int, float, dict, list))


def test_list_columns_returns_dict(file_manager):
    """list_columns should return a dict (uses DataManager internally)."""
    cols = file_manager.list_columns()
    assert isinstance(cols, dict)
    assert "file_id" in cols


def test_list_columns_with_context(file_manager, tmp_path: Path):
    """list_columns with context should return columns for that context."""
    p = tmp_path / "cols_test.csv"
    p.write_text("a,b,c\n1,2,3\n4,5,6\n")
    file_manager.ingest_files(str(p))

    storage = file_manager.describe(file_path=str(p))
    if storage.has_tables:
        cols = file_manager.list_columns(context=storage.tables[0].context_path)
        assert isinstance(cols, dict)
        assert len(cols) > 0


# ────────────────────────────────────────────────────────────────────────────
# describe() API tests
# ────────────────────────────────────────────────────────────────────────────


def test_describe_by_file_path(file_manager, tmp_path: Path):
    """describe() should work with file_path parameter."""
    p = tmp_path / "desc_path.txt"
    p.write_text("test")
    file_manager.ingest_files(str(p))

    storage = file_manager.describe(file_path=str(p))
    assert storage.file_id is not None
    assert storage.file_path == str(p)


def test_describe_by_file_id(file_manager, tmp_path: Path):
    """describe() should work with file_id parameter."""
    p = tmp_path / "desc_id.txt"
    p.write_text("test")
    file_manager.ingest_files(str(p))

    # First get the file_id
    rows = file_manager.filter_files(filter=f"file_path == '{str(p)}'")
    assert rows
    file_id = rows[0]["file_id"]

    # Now describe by file_id
    storage = file_manager.describe(file_id=file_id)
    assert storage.file_id == file_id
    assert storage.file_path == str(p)


def test_describe_includes_document_context(file_manager, tmp_path: Path):
    """describe() should include document context for text files."""
    p = tmp_path / "desc_doc.txt"
    p.write_text("Document content for testing")
    file_manager.ingest_files(str(p))

    storage = file_manager.describe(file_path=str(p))
    assert storage.has_document
    assert storage.document is not None
    assert storage.document.context_path is not None
    assert "/Content" in storage.document.context_path


def test_describe_includes_table_contexts(file_manager, tmp_path: Path):
    """describe() should include table contexts for CSV files."""
    p = tmp_path / "desc_table.csv"
    p.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
    file_manager.ingest_files(str(p))

    storage = file_manager.describe(file_path=str(p))
    if storage.has_tables:
        assert len(storage.tables) > 0
        table = storage.tables[0]
        assert table.name is not None
        assert table.context_path is not None
        assert "/Tables/" in table.context_path


def test_describe_raises_for_missing_file(file_manager):
    """describe() should raise ValueError for non-existent file."""
    with pytest.raises(ValueError, match="not found"):
        file_manager.describe(file_path="/nonexistent/file.txt")


def test_describe_raises_without_identifier(file_manager):
    """describe() should raise ValueError if neither file_path nor file_id provided."""
    with pytest.raises(ValueError, match="Either file_path or file_id"):
        file_manager.describe()


# ────────────────────────────────────────────────────────────────────────────
# Workflow integration tests
# ────────────────────────────────────────────────────────────────────────────


def test_describe_then_filter_workflow(file_manager, tmp_path: Path):
    """describe() → filter_files() workflow should work correctly."""
    p = tmp_path / "workflow.csv"
    p.write_text("name,value\nalpha,10\nbeta,20\n")
    file_manager.ingest_files(str(p))

    # Step 1: describe to get context path
    storage = file_manager.describe(file_path=str(p))
    assert storage.has_tables

    # Step 2: filter using context path from describe
    table_ctx = storage.tables[0].context_path
    rows = file_manager.filter_files(context=table_ctx)
    assert isinstance(rows, list)


def test_describe_then_search_workflow(file_manager, tmp_path: Path):
    """describe() → search_files() workflow should work correctly."""
    p = tmp_path / "workflow_search.txt"
    p.write_text("Machine learning and artificial intelligence concepts")
    file_manager.ingest_files(str(p))

    # Step 1: describe to get context path
    storage = file_manager.describe(file_path=str(p))
    if not storage.has_document:
        pytest.skip("No document context created")

    # Step 2: search using context path from describe
    doc_ctx = storage.document.context_path
    rows = file_manager.search_files(
        context=doc_ctx,
        references={"summary": "machine learning"},
        limit=5,
    )
    assert isinstance(rows, list)


def test_describe_then_reduce_workflow(file_manager, tmp_path: Path):
    """describe() → reduce() workflow should work correctly."""
    p = tmp_path / "workflow_reduce.csv"
    p.write_text("category,amount\nA,100\nB,200\nA,150\n")
    file_manager.ingest_files(str(p))

    # Step 1: describe to get context path
    storage = file_manager.describe(file_path=str(p))
    if not storage.has_tables:
        pytest.skip("No table context created")

    # Step 2: reduce using context path from describe
    table_ctx = storage.tables[0].context_path
    result = file_manager.reduce(
        context=table_ctx,
        metric="sum",
        column="amount",
    )
    assert result is not None
