from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import _handle_project
from unity.file_manager.types import FilePipelineConfig


@pytest.mark.unit
@_handle_project
def test_filter_multi_join_chain(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    # Prepare a simple text file so Content context exists
    p = tmp_path / "mjoin_src.txt"
    p.write_text("seed row for content context")
    name = str(p)
    fm.parse(name, config=FilePipelineConfig())

    # Resolve logical root from overview
    ov = fm._tables_overview(file=name)
    roots = [k for k, v in ov.items() if isinstance(v, dict) and "Content" in v]
    assert roots, "Expected a per-file root in tables_overview"
    root = roots[0]

    # Chain a two-step multi-join using the same root (self-join) as a smoke test
    out = fm._filter_multi_join(
        joins=[
            {
                "tables": [root, root],
                "join_expr": f"{root}.row_id == {root}.row_id",
                "select": {f"{root}.row_id": "rid"},
            },
            {
                "tables": ["$prev", root],
                "join_expr": "rid == rid",
                "select": {"rid": "rid"},
            },
        ],
        result_where=None,
        result_limit=5,
        result_offset=0,
    )
    assert isinstance(out, dict) and "rows" in out


@pytest.mark.unit
@_handle_project
def test_search_multi_join_chain_backfill(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    p = tmp_path / "mjoin_search_src.txt"
    p.write_text("seed row for content context")
    name = str(p)
    fm.parse(name, config=FilePipelineConfig())

    ov = fm._tables_overview(file=name)
    roots = [k for k, v in ov.items() if isinstance(v, dict) and "Content" in v]
    assert roots
    root = roots[0]

    # No references → backfill path; ensure it does not error and returns a list
    rows = fm._search_multi_join(
        joins=[
            {
                "tables": [root, root],
                "join_expr": f"{root}.row_id == {root}.row_id",
                "select": {f"{root}.row_id": "rid"},
            },
        ],
        references=None,
        k=1,
        filter=None,
    )
    assert isinstance(rows, list)
