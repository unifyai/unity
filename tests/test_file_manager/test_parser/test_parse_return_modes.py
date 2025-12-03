from __future__ import annotations

from pathlib import Path

import pytest
from unity.file_manager.types import FilePipelineConfig


def test_return_mode_full(file_manager, tmp_path: Path):
    fm = file_manager
    p = tmp_path / "return_full.txt"
    p.write_text("alpha beta gamma")
    cfg = FilePipelineConfig()
    cfg.output.return_mode = "full"
    out = fm.ingest_files(str(p), config=cfg)
    assert isinstance(out, dict)
    item = out[str(p)]
    assert isinstance(item, dict)
    assert "status" in item and "file_format" in item


def test_return_mode_none(file_manager, tmp_path: Path):
    fm = file_manager
    p = tmp_path / "return_none.txt"
    p.write_text("alpha beta gamma")
    cfg = FilePipelineConfig()
    cfg.output.return_mode = "none"
    out = fm.ingest_files(str(p), config=cfg)
    item = out[str(p)]
    assert set(["file_path", "status", "total_records"]).issubset(set(item.keys()))


@pytest.mark.unit
def test_ingest_single_file(file_manager, tmp_path: Path):
    """Test ingest_files with a single file returns proper structure."""
    fm = file_manager
    p = tmp_path / "return_sync.txt"
    p.write_text("sync ingest content")
    cfg = FilePipelineConfig()
    results = fm.ingest_files(str(p), config=cfg)
    assert results and isinstance(results.get(str(p)), dict)
    assert results[str(p)].get("file_path") == str(p)
