from __future__ import annotations

from pathlib import Path


from tests.helpers import _handle_project
from unity.file_manager.types import FilePipelineConfig


@_handle_project
def test_rename_updates_index_and_contexts(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    p = tmp_path / "rename_src.txt"
    p.write_text("hello")
    src = str(p)

    res = fm.ingest_files(src, config=FilePipelineConfig())
    item = res[src]
    # All returns are now Pydantic models - use attribute access
    assert item.status == "success"

    # Perform rename via manager
    fm.rename_file(file_id_or_path=src, new_name="rename_dst.txt")
    dst = str(p.with_name("rename_dst.txt"))

    # Old row should be gone; new row should exist
    rows_old = fm.filter_files(filter=f"file_path == '{src}'")
    rows_new = fm.filter_files(filter=f"file_path == '{dst}'")
    assert len(rows_old) == 0
    assert any(r.get("file_path") == dst for r in rows_new)

    # Content context should resolve for the new path using describe()
    storage = fm.describe(file_path=dst)
    assert storage.has_document, "Expected document context after rename"
    assert "/Content" in storage.document.context_path


@_handle_project
def test_move_updates_index_and_contexts(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    src_path = tmp_path / "move_src.txt"
    src_path.write_text("hello")
    src = str(src_path)

    fm.ingest_files(src, config=FilePipelineConfig())

    new_dir = tmp_path / "sub"
    new_dir.mkdir(parents=True, exist_ok=True)
    fm.move_file(file_id_or_path=src, new_parent_path=str(new_dir))
    dst = str(new_dir / "move_src.txt")

    rows_old = fm.filter_files(filter=f"file_path == '{src}'")
    rows_new = fm.filter_files(filter=f"file_path == '{dst}'")
    assert len(rows_old) == 0
    assert any(r.get("file_path") == dst for r in rows_new)


@_handle_project
def test_delete_removes_index_row(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    p = tmp_path / "delete_me.txt"
    p.write_text("bye")
    name = str(p)
    fm.ingest_files(name, config=FilePipelineConfig())

    rows = fm.filter_files(filter=f"file_path == '{name}'")
    assert rows and rows[0].get("file_id") is not None
    fid = int(rows[0].get("file_id"))

    # Test deletion using file_id_or_path with file_id
    fm.delete_file(file_id_or_path=fid)
    rows_after = fm.filter_files(filter=f"file_path == '{name}'")
    assert len(rows_after) == 0


@_handle_project
def test_delete_using_file_path(file_manager, tmp_path: Path):
    """Test that deletion works with file_path as well as file_id."""
    fm = file_manager
    fm.clear()

    p = tmp_path / "delete_by_path.txt"
    p.write_text("delete me by path")
    name = str(p)
    fm.ingest_files(name, config=FilePipelineConfig())

    rows_before = fm.filter_files(filter=f"file_path == '{name}'")
    assert len(rows_before) >= 1

    # Test deletion using file_id_or_path with file_path (fully qualified)
    fm.delete_file(file_id_or_path=name)
    rows_after = fm.filter_files(filter=f"file_path == '{name}'")
    assert len(rows_after) == 0
