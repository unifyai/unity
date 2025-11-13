"""
Config and context structure tests for FileManager.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import _handle_project
from unity.file_manager.types import (
    FilePipelineConfig,
    ParseConfig,
    IngestConfig,
    EmbeddingsConfig,
    EmbeddingSpec,
)


@pytest.mark.unit
@_handle_project
def test_per_file_contexts_created(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    p = tmp_path / "ctx_test.txt"
    p.write_text("Simple content for per-file context test.")
    name = str(p)

    res = fm.parse(name)
    _item = res[name]
    _item = _item if isinstance(_item, dict) else _item.model_dump()
    assert _item["status"] == "success"

    print(f"_item.file_path: {_item.get('file_path')}")

    overview = fm._tables_overview(file=name)
    print(f"overview: {overview}")
    assert isinstance(overview, dict)
    # Global entry
    assert "FileRecords" in overview and isinstance(overview["FileRecords"], dict)
    # Find file root entry (shape: { Content: {...}, Tables?: {...} })
    roots = [v for k, v in overview.items() if isinstance(v, dict) and "Content" in v]
    assert roots, "Expected a per-file root with a Content entry"
    content_meta = roots[0]["Content"]
    assert "/Content" in str(content_meta.get("context", ""))


@pytest.mark.unit
@_handle_project
def test_unified_mode_context_created(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    p1 = tmp_path / "u_ctx1.txt"
    p2 = tmp_path / "u_ctx2.txt"
    p1.write_text("Content A")
    p2.write_text("Content B")
    n1 = str(p1)
    n2 = str(p2)

    cfg = FilePipelineConfig(
        ingest=IngestConfig(mode="unified", unified_label="Docs"),
    )
    res = fm.parse([n1, n2], config=cfg)
    _i1 = res[n1]
    _i1 = _i1 if isinstance(_i1, dict) else _i1.model_dump()
    _i2 = res[n2]
    _i2 = _i2 if isinstance(_i2, dict) else _i2.model_dump()
    assert _i1["status"] == "success"
    assert _i2["status"] == "success"

    # Unified context should exist under the unified label
    ov = fm._tables_overview(file="Docs")
    assert isinstance(ov, dict) and len(ov) >= 1
    # Unified label entry should exist with Content
    assert "Docs" in ov and "Content" in ov["Docs"]
    assert "/Content" in str(ov["Docs"]["Content"].get("context", ""))


@pytest.mark.asyncio
@_handle_project
async def test_parse_async_batching_and_kwargs(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    paths = []
    names = []
    for i in range(3):
        p = tmp_path / f"async_{i}.txt"
        p.write_text(f"Row {i}")
        paths.append(p)
        names.append(str(p))

    cfg = FilePipelineConfig(parse=ParseConfig(batch_size=2, parser_kwargs={}))
    results = []
    async for r in fm.parse_async(names, config=cfg):
        results.append(r if isinstance(r, dict) else r.model_dump())

    assert len(results) == 3
    assert all(r.get("status") in ("success", "error") for r in results)
    assert any(r.get("status") == "success" for r in results)


@pytest.mark.unit
@_handle_project
def test_embedding_specs_smoke(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    p = tmp_path / "emb.txt"
    p.write_text("Content for embedding test.")
    name = str(p)

    cfg = FilePipelineConfig(
        ingest=IngestConfig(mode="per_file"),
        embed=EmbeddingsConfig(
            strategy="after",
            specs=[
                EmbeddingSpec(
                    context="per_file",
                    source_column="summary",
                    target_column="_summary_emb",
                ),
            ],
        ),
    )

    res = fm.parse(name, config=cfg)
    _item = res[name]
    _item = _item if isinstance(_item, dict) else _item.model_dump()
    assert _item["status"] == "success"
    # Column existence for index embeddings may be model-driven; ensure schema still accessible
    cols = fm._list_columns()
    assert "file_path" in cols and "status" in cols


@pytest.mark.unit
@_handle_project
def test_table_ingest_toggle_off_skips_tables_contexts(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    p = tmp_path / "no_tables.txt"
    p.write_text("Plain text; no tables expected.")
    name = str(p)

    cfg = FilePipelineConfig(ingest=IngestConfig(table_ingest=False))
    res = fm.parse(name, config=cfg)
    _item = res[name]
    _item = _item if isinstance(_item, dict) else _item.model_dump()
    assert _item["status"] == "success"

    ov = fm._tables_overview(file=name)
    assert isinstance(ov, dict)
    # When table_ingest=False, the root should not include a "Tables" map
    roots = [v for k, v in ov.items() if isinstance(v, dict) and "Content" in v]
    assert roots, "Expected a per-file root with Content"
    assert "Tables" not in roots[0]
