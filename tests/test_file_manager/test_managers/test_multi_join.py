from __future__ import annotations

from pathlib import Path

from tests.helpers import _handle_project
from unity.file_manager.types import FilePipelineConfig


@_handle_project
def test_filter_multi_join_chain(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    # Prepare a simple text file so Content context exists
    p = tmp_path / "mjoin_src.txt"
    p.write_text("seed row for content context")
    name = str(p)
    fm.ingest_files(name, config=FilePipelineConfig())

    # Use file_path directly instead of legacy root from tables_overview
    # Chain a two-step multi-join using the same file_path (self-join) as a smoke test
    out = fm.filter_multi_join(
        joins=[
            {
                "tables": [name, name],
                "join_expr": f"{name}.row_id == {name}.row_id",
                "select": {f"{name}.row_id": "rid"},
            },
            {
                "tables": ["$prev", name],
                # Join the derived rid from $prev to the real row_id on the file context
                "join_expr": "rid == row_id",
                "select": {"rid": "rid"},
            },
        ],
        result_where=None,
        result_limit=5,
        result_offset=0,
    )
    assert isinstance(out, dict) and "rows" in out


@_handle_project
def test_search_multi_join_chain_backfill(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    p = tmp_path / "mjoin_search_src.txt"
    p.write_text("seed row for content context")
    name = str(p)
    fm.ingest_files(name, config=FilePipelineConfig())

    # Use file_path directly instead of legacy root from tables_overview
    # No references → backfill path; ensure it does not error and returns a list
    rows = fm.search_multi_join(
        joins=[
            {
                "tables": [name, name],
                "join_expr": f"{name}.row_id == {name}.row_id",
                "select": {f"{name}.row_id": "rid"},
            },
        ],
        references=None,
        k=1,
        filter=None,
    )
    assert isinstance(rows, list)
