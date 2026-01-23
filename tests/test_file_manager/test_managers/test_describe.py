"""
Comprehensive tests for FileManager.describe() API.

Tests cover all cases for the describe tooling:
- File exists on filesystem but not indexed
- File indexed but parsing failed
- File indexed and parsed successfully
- Access by file_path vs file_id
- Document context (PDF, DOCX, etc.)
- Table contexts (CSV, XLSX sheets)
- Schema information
- Status fields (filesystem_exists, indexed_exists, parsed_status)
- Storage configuration (storage_id, table_ingest)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pytest

# ────────────────────────────────────────────────────────────────────────────
# Status Field Tests
# ────────────────────────────────────────────────────────────────────────────


def test_describe_file_not_indexed(file_manager, tmp_path: Path) -> None:
    """describe() for a file that exists but is not indexed."""
    # Create a file but don't ingest it
    test_file = tmp_path / "not_indexed.txt"
    test_file.write_text("This file exists but is not indexed.")

    storage = file_manager.describe(file_path=str(test_file))

    # Status checks
    assert storage.filesystem_exists is True
    assert storage.indexed_exists is False
    assert storage.parsed_status is None
    assert storage.file_id is None

    # No storage info
    assert storage.document is None
    assert storage.tables == []
    assert storage.has_document is False
    assert storage.has_tables is False


def test_describe_file_not_exists(file_manager) -> None:
    """describe() for a file path that doesn't exist."""
    storage = file_manager.describe(file_path="/nonexistent/path/to/file.txt")

    assert storage.filesystem_exists is False
    assert storage.indexed_exists is False
    assert storage.parsed_status is None
    assert storage.file_id is None


def test_describe_file_indexed_successfully(file_manager, tmp_path: Path) -> None:
    """describe() for a file that is indexed and parsed successfully."""
    test_file = tmp_path / "indexed_success.txt"
    test_file.write_text("This is a successfully indexed file.")

    # Ingest the file
    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    # Status checks
    assert storage.filesystem_exists is True
    assert storage.indexed_exists is True
    assert storage.parsed_status == "success"
    assert storage.file_id is not None

    # Identity
    assert storage.file_path == str(test_file)
    assert storage.source_uri is not None
    assert storage.index_context is not None


def test_describe_returns_ingest_config(file_manager, tmp_path: Path) -> None:
    """describe() returns ingest configuration fields."""
    test_file = tmp_path / "ingest_config_test.csv"
    test_file.write_text("col1,col2\n1,2\n3,4\n")

    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    # Storage config fields should be present
    assert isinstance(storage.storage_id, str) and len(storage.storage_id) > 0
    assert isinstance(storage.table_ingest, bool)
    assert storage.file_format is not None


# ────────────────────────────────────────────────────────────────────────────
# Access Method Tests (file_path vs file_id)
# ────────────────────────────────────────────────────────────────────────────


def test_describe_by_file_path(file_manager, tmp_path: Path) -> None:
    """describe() by file_path resolves correctly."""
    test_file = tmp_path / "by_path.txt"
    test_file.write_text("Access by path test.")

    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    assert storage.indexed_exists is True
    assert storage.file_path == str(test_file)
    assert storage.file_id is not None


def test_describe_by_file_id(file_manager, tmp_path: Path) -> None:
    """describe() by file_id resolves correctly."""
    test_file = tmp_path / "by_id.txt"
    test_file.write_text("Access by file_id test.")

    file_manager.ingest_files(str(test_file))

    # Get file_id from filter
    rows = file_manager.filter_files(filter=f"file_path == {str(test_file)!r}")
    assert rows
    file_id = rows[0].get("file_id")
    assert file_id is not None

    # Describe by file_id
    storage = file_manager.describe(file_id=file_id)

    assert storage.indexed_exists is True
    assert storage.file_id == file_id
    assert storage.file_path == str(test_file)


def test_describe_requires_file_path_or_file_id(file_manager) -> None:
    """describe() raises if neither file_path nor file_id provided."""
    with pytest.raises(ValueError, match="Either file_path or file_id"):
        file_manager.describe()


def test_describe_by_invalid_file_id(file_manager) -> None:
    """describe() for a non-existent file_id returns not indexed."""
    storage = file_manager.describe(file_id=999999)

    assert storage.indexed_exists is False
    assert storage.file_id is None


# ────────────────────────────────────────────────────────────────────────────
# Document Context Tests
# ────────────────────────────────────────────────────────────────────────────


def test_describe_document_context_txt(file_manager, tmp_path: Path) -> None:
    """describe() for a text file shows document context."""
    test_file = tmp_path / "document.txt"
    test_file.write_text("This is a document with content.")

    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    assert storage.has_document is True
    assert storage.document is not None
    assert "/Content" in storage.document.context_path
    assert storage.document.column_schema is not None


def test_describe_document_schema(file_manager, tmp_path: Path) -> None:
    """describe() returns schema information for document context."""
    test_file = tmp_path / "schema_test.txt"
    test_file.write_text("Document for schema testing.")

    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    if storage.has_document and storage.document:
        schema = storage.document.column_schema
        assert schema is not None
        # Should have column info
        assert hasattr(schema, "columns")
        assert hasattr(schema, "column_names")


# ────────────────────────────────────────────────────────────────────────────
# Table Context Tests
# ────────────────────────────────────────────────────────────────────────────


def test_describe_csv_has_tables(file_manager, tmp_path: Path) -> None:
    """describe() for a CSV file shows table context."""
    test_file = tmp_path / "data.csv"
    test_file.write_text("name,value,category\nAlice,100,A\nBob,200,B\nCharlie,150,A\n")

    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    assert storage.has_tables is True
    assert len(storage.tables) >= 1

    # First table should have context_path and schema
    table = storage.tables[0]
    assert "/Tables/" in table.context_path
    assert table.name is not None
    assert table.column_schema is not None


def test_describe_table_schema_columns(file_manager, tmp_path: Path) -> None:
    """describe() returns column info for table contexts."""
    test_file = tmp_path / "columns_test.csv"
    test_file.write_text("id,name,score\n1,Alice,85\n2,Bob,72\n")

    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    if storage.has_tables and storage.tables:
        table = storage.tables[0]
        schema = table.column_schema
        col_names = schema.column_names

        # Should contain our columns
        assert "id" in col_names or "name" in col_names or "score" in col_names


def test_describe_xlsx_multiple_sheets(
    file_manager,
    supported_file_examples: Dict[str, Dict[str, Any]],
) -> None:
    """describe() for XLSX shows multiple table contexts for sheets."""
    # Find xlsx example if available
    xlsx_spec = supported_file_examples.get("xlsx")
    if not xlsx_spec:
        pytest.skip("No xlsx test file available")

    xlsx_path = xlsx_spec["path"]
    file_manager.ingest_files(str(xlsx_path))

    storage = file_manager.describe(file_path=str(xlsx_path))

    # XLSX should have tables (sheets)
    assert storage.has_tables is True
    # Should have at least one table
    assert len(storage.tables) >= 1


def test_describe_table_names_property(file_manager, tmp_path: Path) -> None:
    """FileStorageMap.table_names returns list of table names."""
    test_file = tmp_path / "table_names.csv"
    test_file.write_text("a,b\n1,2\n")

    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    if storage.has_tables:
        assert isinstance(storage.table_names, list)
        assert all(isinstance(n, str) for n in storage.table_names)


# ────────────────────────────────────────────────────────────────────────────
# Context Path Tests
# ────────────────────────────────────────────────────────────────────────────


def test_describe_context_paths_use_file_id(file_manager, tmp_path: Path) -> None:
    """describe() returns context paths that use file_id (not file_path)."""
    test_file = tmp_path / "context_path_test.csv"
    test_file.write_text("x,y\n1,2\n")

    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    # Context paths should contain file_id
    file_id = storage.file_id
    assert file_id is not None

    if storage.has_document:
        assert str(file_id) in storage.document.context_path

    if storage.has_tables:
        for table in storage.tables:
            assert str(file_id) in table.context_path


def test_describe_all_context_paths_property(file_manager, tmp_path: Path) -> None:
    """FileStorageMap.all_context_paths returns all queryable paths."""
    test_file = tmp_path / "all_paths.csv"
    test_file.write_text("col\nval\n")

    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    all_paths = storage.all_context_paths
    assert isinstance(all_paths, list)

    # Should include document and/or table paths
    if storage.has_document:
        assert storage.document.context_path in all_paths
    if storage.has_tables:
        for table in storage.tables:
            assert table.context_path in all_paths


# ────────────────────────────────────────────────────────────────────────────
# Helper Method Tests
# ────────────────────────────────────────────────────────────────────────────


def test_describe_get_table_by_name(file_manager, tmp_path: Path) -> None:
    """FileStorageMap.get_table(name) returns TableInfo."""
    test_file = tmp_path / "get_table.csv"
    test_file.write_text("a,b\n1,2\n")

    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    if storage.has_tables:
        table_name = storage.table_names[0]
        table = storage.get_table(table_name)
        assert table is not None
        assert table.name == table_name


def test_describe_get_table_by_context(file_manager, tmp_path: Path) -> None:
    """FileStorageMap.get_table_by_context returns TableInfo."""
    test_file = tmp_path / "get_by_ctx.csv"
    test_file.write_text("x,y\n1,2\n")

    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    if storage.has_tables:
        ctx_path = storage.tables[0].context_path
        table = storage.get_table_by_context(ctx_path)
        assert table is not None
        assert table.context_path == ctx_path


# ────────────────────────────────────────────────────────────────────────────
# Searchable Column Tests
# ────────────────────────────────────────────────────────────────────────────


def test_describe_schema_searchable_columns(file_manager, tmp_path: Path) -> None:
    """ContextSchema.searchable_columns property works."""
    test_file = tmp_path / "searchable.csv"
    test_file.write_text("text,value\nhello,1\nworld,2\n")

    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    if storage.has_tables:
        schema = storage.tables[0].column_schema
        # searchable_columns should be a list (may be empty if no embeddings)
        assert isinstance(schema.searchable_columns, list)


def test_describe_column_info_properties(file_manager, tmp_path: Path) -> None:
    """ColumnInfo has expected properties."""
    test_file = tmp_path / "col_info.csv"
    test_file.write_text("name,score\nAlice,100\n")

    file_manager.ingest_files(str(test_file))

    storage = file_manager.describe(file_path=str(test_file))

    if storage.has_tables and storage.tables[0].column_schema.columns:
        col = storage.tables[0].column_schema.columns[0]
        assert hasattr(col, "name")
        assert hasattr(col, "data_type")
        assert hasattr(col, "is_searchable")
        assert hasattr(col, "embedding_column")


# ────────────────────────────────────────────────────────────────────────────
# Workflow Integration Tests
# ────────────────────────────────────────────────────────────────────────────


def test_describe_then_filter_workflow(file_manager, tmp_path: Path) -> None:
    """describe() → filter() workflow using context_path."""
    test_file = tmp_path / "workflow_filter.csv"
    test_file.write_text("name,amount\nAlice,100\nBob,200\nCharlie,150\n")

    file_manager.ingest_files(str(test_file))

    # Step 1: Describe to get context path
    storage = file_manager.describe(file_path=str(test_file))
    assert storage.has_tables

    # Step 2: Use context path for filter
    ctx = storage.tables[0].context_path
    results = file_manager.filter_files(context=ctx, filter="amount > 100")

    assert isinstance(results, list)
    # Should have Bob (200) and Charlie (150)
    assert len(results) >= 1


def test_describe_then_reduce_workflow(file_manager, tmp_path: Path) -> None:
    """describe() → reduce() workflow using context_path."""
    test_file = tmp_path / "workflow_reduce.csv"
    test_file.write_text("category,value\nA,10\nA,20\nB,30\n")

    file_manager.ingest_files(str(test_file))

    # Step 1: Describe
    storage = file_manager.describe(file_path=str(test_file))
    assert storage.has_tables

    # Step 2: Use context path for reduce
    ctx = storage.tables[0].context_path
    result = file_manager.reduce(context=ctx, metric="sum", columns="value")

    assert result is not None
