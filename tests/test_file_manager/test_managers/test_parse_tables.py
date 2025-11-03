"""
Per-table ingestion tests for spreadsheets (CSV/XLSX with multi-tab).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_csv_per_table_context(file_manager, tmp_path: Path):
    # Create a simple CSV
    csv_path = tmp_path / "people.csv"
    csv_path.write_text(
        "Name,Age,City\nJohn,30,NYC\nJane,25,LDN\n",
        encoding="utf-8",
    )
    display_name = file_manager.import_file(csv_path)

    # Parse file
    result = file_manager.parse(display_name)
    print(result)
    assert result[display_name]["status"] == "success"

    # Verify a per-table context exists
    import unify

    ctxs = unify.get_contexts()
    table_ctx_candidates = [
        name for name in ctxs.keys() if "/Tables/" in name and "people" in name
    ]
    assert table_ctx_candidates, "No table context candidates found"

    # Optionally, fetch a few rows from one table context
    sample_ctx = table_ctx_candidates[0]
    rows = unify.get_logs(context=sample_ctx, limit=10)
    # Should have header-derived columns
    assert rows, "No rows found"
    assert set(["Name", "Age", "City"]).issubset(
        set(rows[0].entries.keys()),
    ), "Columns not found"
    assert len(rows) > 0, "No rows found"
    assert len(rows[0].entries) == 3, "Columns not found"
    assert len(rows[1].entries) == 3, "Columns not found"
    assert rows[0].entries["Name"] == "Jane", "Name not found"
    assert rows[0].entries["Age"] == "25", "Age not found"
    assert rows[0].entries["City"] == "LDN", "City not found"
    assert rows[1].entries["Name"] == "John", "Name not found"
    assert rows[1].entries["Age"] == "30", "Age not found"
    assert rows[1].entries["City"] == "NYC", "City not found"


@pytest.mark.asyncio
@_handle_project
async def test_xlsx_multi_tab_per_table_context(file_manager):
    # Use sample multi-tab workbooks from tests
    sample_dir = Path(__file__).parents[1] / "sample"
    retail = sample_dir / "retail_data.xlsx"
    workforce = sample_dir / "workforce_data.xlsx"

    for path in [retail, workforce]:
        if path.exists():
            display_name = file_manager.import_file(path)
            res = file_manager.parse(display_name)
            print(res)
            assert res[display_name]["status"] == "success"
        else:
            print(f"Path {path} does not exist")
            assert False

    # Verify multiple per-table contexts exist (one per tab)
    import unify

    ctxs = unify.get_contexts()
    table_ctx_candidates = [
        name
        for name in ctxs.keys()
        if "/Tables/" in name
        and any(k in name for k in ["retail_data", "workforce_data"])
    ]
    assert len(table_ctx_candidates) >= 2
