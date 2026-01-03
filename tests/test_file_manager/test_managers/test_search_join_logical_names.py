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

    # Use file_path directly instead of legacy root from tables_overview
    # We will join Content contexts by a trivial select; this is a smoke test for file_path resolution
    out = fm.filter_join(
        tables=[fa, fb],
        join_expr=f"{fa}.row_id == {fb}.row_id",
        select={
            f"{fa}.file_id": "left_id",
            f"{fb}.file_id": "right_id",
        },
        mode="inner",
        left_where=None,
        right_where=None,
        result_where=None,
        result_limit=10,
        result_offset=0,
    )
    assert isinstance(out, dict) and "rows" in out
