from __future__ import annotations

from pathlib import Path

import pytest
from unity.file_manager.types import (
    FilePipelineConfig,
    ParsedFile,
    IngestedMinimal,
    BaseIngestedFile,
    IngestPipelineResult,
)


def test_return_mode_full(file_manager, tmp_path: Path):
    """Test that full return mode returns ParsedFile Pydantic model."""
    fm = file_manager
    p = tmp_path / "return_full.txt"
    p.write_text("alpha beta gamma")
    cfg = FilePipelineConfig()
    cfg.output.return_mode = "full"
    out = fm.ingest_files(str(p), config=cfg)
    # out is IngestPipelineResult Pydantic model
    assert isinstance(out, IngestPipelineResult)
    item = out[str(p)]
    # full mode returns ParsedFile Pydantic model
    assert isinstance(item, ParsedFile)
    assert item.status in ("success", "error")
    assert hasattr(item, "file_format")


def test_return_mode_none(file_manager, tmp_path: Path):
    """Test that none return mode returns IngestedMinimal Pydantic model."""
    fm = file_manager
    p = tmp_path / "return_none.txt"
    p.write_text("alpha beta gamma")
    cfg = FilePipelineConfig()
    cfg.output.return_mode = "none"
    out = fm.ingest_files(str(p), config=cfg)
    item = out[str(p)]
    # none mode returns IngestedMinimal Pydantic model
    assert isinstance(item, IngestedMinimal)
    assert hasattr(item, "file_path")
    assert hasattr(item, "status")
    assert hasattr(item, "total_records")


@pytest.mark.unit
def test_ingest_single_file(file_manager, tmp_path: Path):
    """Test ingest_files with a single file returns proper structure."""
    fm = file_manager
    p = tmp_path / "return_sync.txt"
    p.write_text("sync ingest content")
    cfg = FilePipelineConfig()
    results = fm.ingest_files(str(p), config=cfg)
    # results is IngestPipelineResult Pydantic model
    assert isinstance(results, IngestPipelineResult)
    assert str(p) in results
    item = results[str(p)]
    # default (compact) mode returns BaseIngestedFile or subclass
    assert isinstance(item, BaseIngestedFile)
    assert item.file_path == str(p)
