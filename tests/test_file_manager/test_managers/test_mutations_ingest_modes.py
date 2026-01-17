from __future__ import annotations

from pathlib import Path

from tests.helpers import _handle_project
from unity.file_manager.types import FilePipelineConfig


@_handle_project
def test_unified_mode_rename_preserves_unified_content(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    p = tmp_path / "unified_src.txt"
    p.write_text("hello unified")
    name = str(p)

    cfg = FilePipelineConfig()
    cfg.ingest.storage_id = "UnifiedDocs"
    cfg.ingest.table_ingest = False

    fm.ingest_files(name, config=cfg)

    # Before rename: describe should return file storage map
    storage_before = fm.describe(file_path=name)
    # Unified mode uses a different context structure, file should be indexed
    assert storage_before.file_id is not None

    # Rename the underlying file
    fm.rename_file(file_id_or_path=name, new_name="unified_dst.txt")
    new_name = str(p.with_name("unified_dst.txt"))

    # After rename: describe should find the file at new path
    storage_after = fm.describe(file_path=new_name)
    assert storage_after.file_id is not None


@_handle_project
def test_per_file_mode_move_updates_content_root(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()

    src = tmp_path / "move_src_pf.txt"
    src.write_text("pf")
    name = str(src)

    cfg = FilePipelineConfig()  # default per_file
    fm.ingest_files(name, config=cfg)

    # Move into a subfolder
    sub = tmp_path / "sub"
    sub.mkdir(parents=True, exist_ok=True)
    fm.move_file(file_id_or_path=name, new_parent_path=str(sub))
    new_name = str(sub / src.name)

    # describe() should resolve the Content context for the new path
    storage = fm.describe(file_path=new_name)
    assert storage.has_document, "Expected document context after move"
    assert "/Content" in storage.document.context_path
