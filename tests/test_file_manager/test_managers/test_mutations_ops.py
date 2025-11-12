from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import _handle_project
from unity.file_manager.types import FilePipelineConfig


@pytest.mark.unit
@_handle_project
def test_rename_updates_index_and_contexts(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    p = tmp_path / "rename_src.txt"
    p.write_text("hello")
    src = str(p)

    res = fm.parse(src, config=FilePipelineConfig())
    item = res[src]
    item = item if isinstance(item, dict) else item.model_dump()
    assert item.get("status") == "success"

    # Perform rename via manager
    fm._rename_file(target_id_or_path=src, new_name="rename_dst.txt")
    dst = str(p.with_name("rename_dst.txt"))

    # Old row should be gone; new row should exist
    rows_old = fm._filter_files(filter=f"file_path == '{src}'")
    rows_new = fm._filter_files(filter=f"file_path == '{dst}'")
    assert len(rows_old) == 0
    assert any(r.get("file_path") == dst for r in rows_new)

    # Content context should resolve for the new path in tables_overview
    ov = fm._tables_overview(file=dst)
    roots = [v for k, v in ov.items() if isinstance(v, dict) and "Content" in v]
    assert roots and "/Content" in str(roots[0]["Content"].get("context", ""))


@pytest.mark.unit
@_handle_project
def test_move_updates_index_and_contexts(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    src_path = tmp_path / "move_src.txt"
    src_path.write_text("hello")
    src = str(src_path)

    fm.parse(src, config=FilePipelineConfig())

    new_dir = tmp_path / "sub"
    new_dir.mkdir(parents=True, exist_ok=True)
    fm._move_file(target_id_or_path=src, new_parent_path=str(new_dir))
    dst = str(new_dir / "move_src.txt")

    rows_old = fm._filter_files(filter=f"file_path == '{src}'")
    rows_new = fm._filter_files(filter=f"file_path == '{dst}'")
    assert len(rows_old) == 0
    assert any(r.get("file_path") == dst for r in rows_new)


@pytest.mark.unit
@_handle_project
def test_delete_removes_index_row(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    p = tmp_path / "delete_me.txt"
    p.write_text("bye")
    name = str(p)
    fm.parse(name, config=FilePipelineConfig())

    rows = fm._filter_files(filter=f"file_path == '{name}'")
    assert rows and rows[0].get("file_id") is not None
    fid = int(rows[0].get("file_id"))

    fm._delete_file(file_id=fid)
    rows_after = fm._filter_files(filter=f"file_path == '{name}'")
    assert len(rows_after) == 0
