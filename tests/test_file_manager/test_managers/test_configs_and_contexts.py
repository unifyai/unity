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
    p = tmp_path / "ctx_test.txt"
    p.write_text("Simple content for per-file context test.")
    name = fm.import_file(p)

    res = fm.parse(name)
    assert res[name]["status"] == "success"

    overview = fm._tables_overview(file=name)
    assert isinstance(overview, dict)
    # At least the Content context should exist
    assert any("/Content" in str(meta.get("context", "")) for meta in overview.values())


@pytest.mark.unit
@_handle_project
def test_unified_mode_context_created(file_manager, tmp_path: Path):
    fm = file_manager
    p1 = tmp_path / "u_ctx1.txt"
    p2 = tmp_path / "u_ctx2.txt"
    p1.write_text("Content A")
    p2.write_text("Content B")
    n1 = fm.import_file(p1)
    n2 = fm.import_file(p2)

    cfg = FilePipelineConfig(
        ingest=IngestConfig(mode="unified", unified_label="Docs"),
    )
    res = fm.parse([n1, n2], config=cfg)
    assert res[n1]["status"] == "success"
    assert res[n2]["status"] == "success"

    # Unified context should exist under the unified label
    ov = fm._tables_overview(file="Docs")
    assert isinstance(ov, dict) and len(ov) >= 1
    assert any(
        "Docs" in str(m.get("context", "")) and "/Content" in str(m.get("context", ""))
        for m in ov.values()
    )


@pytest.mark.asyncio
@_handle_project
async def test_parse_async_batching_and_kwargs(file_manager, tmp_path: Path):
    fm = file_manager
    paths = []
    names = []
    for i in range(3):
        p = tmp_path / f"async_{i}.txt"
        p.write_text(f"Row {i}")
        paths.append(p)
        names.append(fm.import_file(p))

    cfg = FilePipelineConfig(parse=ParseConfig(batch_size=2, parser_kwargs={}))
    results = []
    async for r in fm.parse_async(names, config=cfg):
        results.append(r)

    assert len(results) == 3
    assert all(r.get("status") in ("success", "error") for r in results)
    assert any(r.get("status") == "success" for r in results)


@pytest.mark.unit
@_handle_project
def test_embedding_specs_smoke(file_manager, tmp_path: Path):
    fm = file_manager
    p = tmp_path / "emb.txt"
    p.write_text("Content for embedding test.")
    name = fm.import_file(p)

    cfg = FilePipelineConfig(
        ingest=IngestConfig(mode="per_file"),
        embed=EmbeddingsConfig(
            embed_along=True,
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
    assert res[name]["status"] == "success"
    # Column existence for index embeddings may be model-driven; ensure schema still accessible
    cols = fm._list_columns()
    assert "file_path" in cols and "status" in cols


@pytest.mark.unit
@_handle_project
def test_table_ingest_toggle_off_skips_tables_contexts(file_manager, tmp_path: Path):
    fm = file_manager
    p = tmp_path / "no_tables.txt"
    p.write_text("Plain text; no tables expected.")
    name = fm.import_file(p)

    cfg = FilePipelineConfig(ingest=IngestConfig(table_ingest=False))
    res = fm.parse(name, config=cfg)
    assert res[name]["status"] == "success"

    ov = fm._tables_overview(file=name)
    assert isinstance(ov, dict)
    # There should be no per-file Tables contexts when explicitly disabled
    assert not any("/Tables/" in str(meta.get("context", "")) for meta in ov.values())
