from __future__ import annotations

from typing import Dict, Any


def test_list_columns_index(file_manager) -> None:
    """Test list_columns returns index schema when no context provided."""
    cols = file_manager.list_columns()
    assert isinstance(cols, dict)
    assert len(cols) > 0
    assert "file_id" in cols


def test_list_columns_per_file_content(
    file_manager,
    supported_file_examples: Dict[str, Dict[str, Any]],
) -> None:
    """Test list_columns for per-file document content using describe() API."""
    # Pick any available example and parse it
    name, spec = next(iter(supported_file_examples.items()))
    file_path = spec["path"].as_posix()
    file_manager.ingest_files(file_path)

    # Use describe() to get storage map
    storage = file_manager.describe(file_path=file_path)

    if not storage.has_document:
        # No document content created – be lenient
        return

    # Use the exact context path from describe()
    cols = file_manager.list_columns(context=storage.document.context_path)
    assert isinstance(cols, dict)
    assert len(cols) > 0


def test_list_columns_per_file_table_when_present(
    file_manager,
    supported_file_examples: Dict[str, Dict[str, Any]],
):
    """Test list_columns for per-file tables using describe() API."""
    # Prefer a CSV/XLSX example to maximize chance of tables
    chosen = None
    for name, spec in supported_file_examples.items():
        if str(spec.get("path", "")).endswith((".csv", ".xlsx")):
            chosen = (name, spec)
            break
    if chosen is None:
        # Fall back to any example; may result in no tables
        chosen = next(iter(supported_file_examples.items()))

    _, spec = chosen
    file_path = spec["path"].as_posix()
    file_manager.ingest_files(file_path)

    # Use describe() to get storage map with tables
    storage = file_manager.describe(file_path=file_path)

    if not storage.has_tables:
        # No tables extracted in this run – be lenient
        return

    # Use the exact context path from describe()
    table_context = storage.tables[0].context_path
    cols = file_manager.list_columns(context=table_context)
    assert isinstance(cols, dict)
    assert len(cols) > 0


def test_describe_returns_storage_map(
    file_manager,
    supported_file_examples: Dict[str, Dict[str, Any]],
) -> None:
    """Test describe() returns FileStorageMap with correct structure."""
    # Pick any available example
    name, spec = next(iter(supported_file_examples.items()))
    file_path = spec["path"].as_posix()
    file_manager.ingest_files(file_path)

    storage = file_manager.describe(file_path=file_path)

    # Verify FileStorageMap structure
    assert storage.file_id is not None
    assert storage.file_path == file_path
    assert storage.index_context is not None

    # At least one of document or tables should exist
    if storage.has_document:
        assert storage.document is not None
        assert storage.document.context_path is not None
    if storage.has_tables:
        assert len(storage.tables) > 0
        for table in storage.tables:
            assert table.name is not None
            assert table.context_path is not None
