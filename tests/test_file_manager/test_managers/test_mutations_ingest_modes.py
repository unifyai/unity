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
    cfg.ingest.mode = "unified"
    cfg.ingest.unified_label = "UnifiedDocs"
    cfg.ingest.table_ingest = False

    fm.ingest_files(name, config=cfg)

    # Before rename: file-scoped overview should expose the unified Content key
    ov_before = fm.tables_overview(file=name)
    assert "UnifiedDocs" in ov_before and "Content" in ov_before["UnifiedDocs"]

    # Rename the underlying file
    fm.rename_file(file_id_or_path=name, new_name="unified_dst.txt")
    new_name = str(p.with_name("unified_dst.txt"))

    # After rename: the unified Content key remains under the same label
    ov_after = fm.tables_overview(file=new_name)
    assert "UnifiedDocs" in ov_after and "Content" in ov_after["UnifiedDocs"]


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

    # The file-scoped overview should resolve the Content context for the new path
    ov = fm.tables_overview(file=new_name)
    roots = [k for k, v in ov.items() if isinstance(v, dict) and "Content" in v]
    assert roots and any(
        "/Content" in ov[roots[0]]["Content"].get("context", "") for _ in [0]
    )
