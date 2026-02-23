from __future__ import annotations

from pathlib import Path


from tests.helpers import _handle_project
from unity.file_manager.types import FilePipelineConfig


@_handle_project
def test_filter_join_with_logical_names(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    # Prepare a small CSV-like table by parsing text with a detected table (simulated by rows)
    # Use two files to make join meaningful
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("header1,header2\nA1,B1\nA2,B2\n")
    b.write_text("header1,header3\nA1,C1\nX2,C2\n")

    fa, fb = str(a), str(b)
    fm.ingest_files([fa, fb], config=FilePipelineConfig())

    # Use describe() to get the actual context paths
    storage_a = fm.describe(file_path=fa)
    storage_b = fm.describe(file_path=fb)
    assert storage_a.document is not None, "File A should have a document context"
    assert storage_b.document is not None, "File B should have a document context"
    ctx_a = storage_a.document.context_path
    ctx_b = storage_b.document.context_path

    # Join Content contexts by row_id; this is a smoke test for context path resolution
    out = fm.filter_join(
        tables=[ctx_a, ctx_b],
        join_expr=f"{ctx_a}.row_id == {ctx_b}.row_id",
        select={
            f"{ctx_a}.file_id": "left_id",
            f"{ctx_b}.file_id": "right_id",
        },
        mode="inner",
        left_where=None,
        right_where=None,
        result_where=None,
        result_limit=10,
        result_offset=0,
    )
    # filter_join returns a list of dicts
    assert isinstance(out, list), f"Expected list, got {type(out)}"
