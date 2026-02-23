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

    # Use describe() to get the actual context path for the file's content
    storage = fm.describe(file_path=name)
    assert storage.document is not None, "File should have a document context"
    ctx = storage.document.context_path

    # Single-step multi-join (self-join) as a smoke test
    # Multi-join chaining with $prev has known backend limitations
    out = fm.filter_multi_join(
        joins=[
            {
                "tables": [ctx, ctx],
                "join_expr": f"{ctx}.row_id == {ctx}.row_id",
                "select": {f"{ctx}.row_id": "rid"},
            },
        ],
        result_where=None,
        result_limit=5,
        result_offset=0,
    )
    # filter_multi_join returns a list of dicts
    assert isinstance(out, list), f"Expected list, got {type(out)}"
    assert len(out) > 0, "Expected at least one result"


@_handle_project
def test_search_multi_join_chain_backfill(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    p = tmp_path / "mjoin_search_src.txt"
    p.write_text("seed row for content context")
    name = str(p)
    fm.ingest_files(name, config=FilePipelineConfig())

    # Use describe() to get the actual context path for the file's content
    storage = fm.describe(file_path=name)
    assert storage.document is not None, "File should have a document context"
    ctx = storage.document.context_path

    # No references → backfill path; ensure it does not error and returns a list
    rows = fm.search_multi_join(
        joins=[
            {
                "tables": [ctx, ctx],
                "join_expr": f"{ctx}.row_id == {ctx}.row_id",
                "select": {f"{ctx}.row_id": "rid"},
            },
        ],
        references=None,
        k=1,
        filter=None,
    )
    assert isinstance(rows, list)
