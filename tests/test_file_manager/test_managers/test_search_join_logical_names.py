from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import _handle_project
from unity.file_manager.types import FilePipelineConfig


@pytest.mark.unit
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
    fm.parse([fa, fb], config=FilePipelineConfig())

    # Overview provides logical names; fetch roots
    ov_a = fm._tables_overview(file=fa)
    roots_a = [k for k, v in ov_a.items() if isinstance(v, dict) and "Content" in v]
    assert roots_a
    root_a = roots_a[0]

    ov_b = fm._tables_overview(file=fb)
    roots_b = [k for k, v in ov_b.items() if isinstance(v, dict) and "Content" in v]
    assert roots_b
    root_b = roots_b[0]

    # With no explicit tables extracted in this basic text path, we still check the join wrapper accepts logical names
    # We will join Content contexts by a trivial select; this is a smoke test for logical name resolution
    out = fm._filter_join(
        tables=[root_a, root_b],
        join_expr=f"{root_a}.row_id == {root_b}.row_id",
        select={
            f"{root_a}.file_id": "left_id",
            f"{root_b}.file_id": "right_id",
        },
        mode="inner",
        left_where=None,
        right_where=None,
        result_where=None,
        result_limit=10,
        result_offset=0,
    )
    assert isinstance(out, dict) and "rows" in out
