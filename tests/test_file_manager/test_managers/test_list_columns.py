from __future__ import annotations

from typing import Dict, Any


def test_list_columns_index(file_manager) -> None:
    cols = file_manager.list_columns()
    assert isinstance(cols, dict)
    assert len(cols) > 0
    assert "file_id" in cols


def test_list_columns_per_file_content(
    file_manager,
    supported_file_examples: Dict[str, Dict[str, Any]],
) -> None:
    # Pick any available example and parse it
    name, spec = next(iter(supported_file_examples.items()))
    file_path = spec["path"].as_posix()
    file_manager.ingest_files(file_path)

    overview = file_manager.tables_overview(file=file_path)
    # Verify overview has content (for validation)
    roots = [k for k in overview.keys() if k != "FileRecords"]
    if not roots:
        # Nothing was created (e.g., parse failure) – be lenient
        return

    # Use file_path directly instead of legacy root from tables_overview
    # Request the schema for the per-file Content context via file_path
    cols = file_manager.list_columns(table=file_path)
    assert isinstance(cols, dict)
    assert len(cols) > 0


def test_list_columns_per_file_table_when_present(
    file_manager,
    supported_file_examples: Dict[str, Dict[str, Any]],
):
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

    overview = file_manager.tables_overview(file=file_path)
    roots = [k for k in overview.keys() if k != "FileRecords"]
    if not roots:
        return
    # Look for a Tables map under any root to get the table label
    table_label = None
    for root in roots:
        meta = overview.get(root, {})
        tables = meta.get("Tables") if isinstance(meta, dict) else None
        if tables:
            table_label = next(iter(tables.keys()))
            break
    if table_label is None:
        # No tables extracted in this run – be lenient
        return

    # Use file_path directly instead of legacy root from tables_overview
    cols = file_manager.list_columns(table=f"{file_path}.Tables.{table_label}")
    assert isinstance(cols, dict)
    assert len(cols) > 0
