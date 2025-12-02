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
    out = fm.parse(str(p), config=cfg)
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
    out = fm.parse(str(p), config=cfg)
    item = out[str(p)]
    assert set(["file_path", "status", "total_records"]).issubset(set(item.keys()))


@pytest.mark.asyncio
async def test_async_single(file_manager, tmp_path: Path):
    fm = file_manager
    p = tmp_path / "return_async.txt"
    p.write_text("async parse content")
    cfg = FilePipelineConfig()
    results = []
    async for r in fm.parse_async(str(p), config=cfg):
        results.append(r)
    assert results and isinstance(results[0], dict)
    assert results[0].get("file_path") == str(p)
