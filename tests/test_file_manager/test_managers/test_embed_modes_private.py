from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import pytest
import unify

from tests.helpers import _handle_project
from unity.file_manager.types import (
    FilePipelineConfig,
    IngestConfig,
    EmbeddingsConfig,
    FileEmbeddingSpec,
    TableEmbeddingSpec,
    PluginsConfig,
    BusinessContextSpec,
    TableBusinessContextSpec,
)


def _make_records(n: int) -> List[Dict[str, Any]]:
    # Minimal content rows compatible with FileContent schema
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "content_type": "paragraph",
                "title": f"t{i}",
                "summary": f"sum{i}",
                "content_text": f"text{i}",
            },
        )
    return rows


class _MetaStub:
    def __init__(self, tables: List[Any] | None = None) -> None:
        self.tables = tables or []
        self.mime_type = "text/plain"
        self.parser_name = "Stub"
        self.processing_time = 0.0


class _TableStub:
    def __init__(self, rows: List[Any], sheet_name: str | None = None) -> None:
        self.rows = rows
        self.columns = None
        self.sheet_name = sheet_name
        self.section_path = None


class _DocStub:
    def __init__(self, tables: List[Any] | None = None) -> None:
        self.metadata = _MetaStub(tables=tables or [])
        self.processing_status = "completed"

    # Only used by index entry; not called in these tests
    def to_parse_result(self, *a, **kw) -> Dict[str, Any]:
        return {}


@pytest.mark.unit
@_handle_project
def test_embed_off_no_columns(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    file_path = "synthetic_off.txt"
    doc = _DocStub()
    records = _make_records(3)
    result = {
        "status": "success",
        "total_records": len(records),
        "file_format": "txt",
        "records": records,
    }

    cfg = FilePipelineConfig(
        ingest=IngestConfig(mode="per_file"),
        embed=EmbeddingsConfig(
            strategy="off",
            file_specs=[
                FileEmbeddingSpec(
                    file_path="*",
                    context="per_file",
                    tables=[
                        TableEmbeddingSpec(
                            table="*",
                            source_columns=["summary"],
                            target_columns=["_summary_emb"],
                        ),
                    ],
                ),
            ],
        ),
    )

    # Ingest then attempt embed (should early-return and not create columns)
    inserted_ids = fm._ingest(
        file_path=file_path,
        document=doc,
        result=result,
        config=cfg,
    )
    fm._embed(
        file_path=file_path,
        document=doc,
        result=result,
        inserted_ids=inserted_ids,
        config=cfg,
    )
    ctx = fm._ctx_for_file(file_path)
    fields = unify.get_fields(context=ctx)
    assert "_summary_emb" not in fields


@pytest.mark.unit
@_handle_project
def test_embed_after_single_hook_and_columns(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    file_path = "synthetic_after.txt"
    doc = _DocStub()
    records = _make_records(4)
    result = {
        "status": "success",
        "total_records": len(records),
        "file_format": "txt",
        "records": records,
    }

    calls = {"pre": 0, "post": 0}

    def pre_hook(manager, filename, result, document, config):
        calls["pre"] += 1

    def post_hook(manager, filename, result, document, config):
        calls["post"] += 1

    cfg = FilePipelineConfig(
        ingest=IngestConfig(mode="per_file"),
        embed=EmbeddingsConfig(
            strategy="after",
            hooks_per_chunk=True,  # irrelevant for 'after', should still be once
            file_specs=[
                FileEmbeddingSpec(
                    file_path="*",
                    context="per_file",
                    tables=[
                        TableEmbeddingSpec(
                            table="*",
                            source_columns=["summary"],
                            target_columns=["_summary_emb"],
                        ),
                    ],
                ),
            ],
        ),
        plugins=PluginsConfig(pre_embed=[pre_hook], post_embed=[post_hook]),
    )

    inserted_ids = fm._ingest(
        file_path=file_path,
        document=doc,
        result=result,
        config=cfg,
    )
    fm._embed(
        file_path=file_path,
        document=doc,
        result=result,
        inserted_ids=inserted_ids,
        config=cfg,
    )
    # Hooks should run exactly once in 'after' mode
    assert calls["pre"] == 1
    assert calls["post"] == 1

    ctx = fm._ctx_for_file(file_path)
    fields = unify.get_fields(context=ctx)
    assert "_summary_emb" in fields


@pytest.mark.unit
@_handle_project
def test_embed_along_content_hooks_per_chunk(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    file_path = "synthetic_along.txt"
    doc = _DocStub()
    # 5 records with chunk size 2 → expect 3 chunks
    records = _make_records(5)
    result = {
        "status": "success",
        "total_records": len(records),
        "file_format": "txt",
        "records": records,
    }

    calls = {"pre": 0, "post": 0}

    def pre_hook(manager, filename, result, document, config):
        calls["pre"] += 1

    def post_hook(manager, filename, result, document, config):
        calls["post"] += 1

    cfg = FilePipelineConfig(
        ingest=IngestConfig(mode="per_file", content_rows_batch_size=2),
        embed=EmbeddingsConfig(
            strategy="along",
            hooks_per_chunk=True,
            file_specs=[
                FileEmbeddingSpec(
                    file_path="*",
                    context="per_file",
                    tables=[
                        TableEmbeddingSpec(
                            table="*",
                            source_columns=["summary"],
                            target_columns=["_summary_emb"],
                        ),
                    ],
                ),
            ],
        ),
        plugins=PluginsConfig(pre_embed=[pre_hook], post_embed=[post_hook]),
    )

    fm._ingest_and_embed(file_path=file_path, document=doc, result=result, config=cfg)
    # Hooks should run at least once per chunk
    expected_chunks = math.ceil(len(records) / cfg.ingest.content_rows_batch_size)
    assert calls["pre"] >= expected_chunks
    assert calls["post"] >= expected_chunks

    ctx = fm._ctx_for_file(file_path)
    fields = unify.get_fields(context=ctx)
    assert "_summary_emb" in fields


@pytest.mark.unit
def test_resolve_embed_strategy_auto_without_parse():
    # Build a fake document with tables to simulate size
    small_doc = _DocStub(tables=[_TableStub(rows=[1, 2])])  # 2 rows
    big_doc = _DocStub(tables=[_TableStub(rows=list(range(10)))])  # 10 rows

    # Auto with threshold 5 → small → after, big → along
    cfg = FilePipelineConfig(
        embed=EmbeddingsConfig(strategy="auto", large_threshold=5, specs=[]),
    )
    from unity.file_manager.managers.ops import resolve_embed_strategy as _res

    res_small = {"total_records": 0}
    res_big = {"total_records": 0}
    assert _res(small_doc, res_small, cfg) == "after"
    assert _res(big_doc, res_big, cfg) == "along"


# ---------------- Additional cross-format strategy and table tests ---------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "label,total_records,total_table_rows,threshold,expected",
    [
        ("pdf_small", 100, 0, 2000, "after"),
        ("pdf_large", 2500, 0, 2000, "along"),
        ("docx_small", 50, 0, 2000, "after"),
        ("docx_large", 5000, 0, 2000, "along"),
        ("csv_small_tables", 0, 10, 2000, "after"),
        ("csv_large_tables", 0, 5000, 2000, "along"),
        ("xlsx_small_tables", 0, 100, 2000, "after"),
        ("xlsx_large_tables", 0, 10000, 2000, "along"),
    ],
)
def test_auto_strategy_across_formats(
    label: str,
    total_records: int,
    total_table_rows: int,
    threshold: int,
    expected: str,
):
    # Create a stub document with a specific number of table rows
    tables = []
    if total_table_rows > 0:
        tables.append(
            _TableStub(rows=list(range(total_table_rows)), sheet_name="Sheet1"),
        )
    doc = _DocStub(tables=tables)
    result = {"total_records": total_records}
    cfg = FilePipelineConfig(
        embed=EmbeddingsConfig(strategy="auto", large_threshold=threshold, specs=[]),
    )
    from unity.file_manager.managers.ops import resolve_embed_strategy as _res

    assert _res(doc, result, cfg) == expected


@pytest.mark.unit
@_handle_project
@pytest.mark.parametrize(
    "file_name,sheet_name,num_rows,batch_size,target_columns",
    [
        (
            "large.csv",
            None,
            6,
            2,
            ["_Name_emb", "_City_emb"],
        ),  # CSV: no sheet name, multiple columns
        (
            "large.xlsx",
            "Sheet1",
            6,
            2,
            ["_Name_emb", "_City_emb"],
        ),  # XLSX: with sheet name, multiple columns
    ],
)
def test_table_embeddings_along_for_csv_and_xlsx(
    file_manager,
    tmp_path: Path,
    file_name: str,
    sheet_name: str | None,
    num_rows: int,
    batch_size: int,
    target_columns: list[str],
):
    """
    Ensure that along mode creates embedding columns on per-file table contexts for both CSV and XLSX-like inputs.
    Demonstrates consolidated embedding spec with multiple columns per table.
    """
    fm = file_manager
    fm.clear()

    # Build document with a single table and N rows
    rows = [["Name", "Age", "City", "Country"]] + [
        [f"User{i}", 20 + i, f"City{i}", f"Country{i}"] for i in range(num_rows)
    ]
    tbl = _TableStub(rows=rows, sheet_name=sheet_name)
    doc = _DocStub(tables=[tbl])
    result = {
        "status": "success",
        "total_records": 0,
        "file_format": "csv" if file_name.endswith(".csv") else "xlsx",
        "records": [],
    }

    # Use consolidated embedding spec with multiple columns per table
    cfg = FilePipelineConfig(
        ingest=IngestConfig(
            mode="per_file",
            table_rows_batch_size=batch_size,
            business_contexts=[
                BusinessContextSpec(
                    file_path=file_name,
                    tables=[
                        TableBusinessContextSpec(
                            table=sheet_name
                            or "large",  # Use sheet name or filename-derived table name
                            column_descriptions={
                                "Name": "User's full name",
                                "City": "City where the user resides",
                                "Age": "User's age in years",
                                "Country": "Country of residence",
                            },
                            table_description="User directory with demographic information",
                        ),
                    ],
                ),
            ],
        ),
        embed=EmbeddingsConfig(
            strategy="along",
            hooks_per_chunk=True,
            file_specs=[
                FileEmbeddingSpec(
                    file_path="*",
                    context="per_file_table",
                    tables=[
                        TableEmbeddingSpec(
                            table="*",
                            source_columns=[
                                "Name",
                                "City",
                            ],  # Multiple columns in single spec
                            target_columns=target_columns,
                        ),
                    ],
                ),
            ],
        ),
    )
    # Run along pipeline directly
    fm._ingest_and_embed(file_path=file_name, document=doc, result=result, config=cfg)

    # Locate a per-file table in the overview and assert the embedding columns exist
    ov = fm._tables_overview(file=file_name)
    roots = [k for k, v in ov.items() if isinstance(v, dict) and "Tables" in v]
    assert roots, "Expected per-file root with Tables"
    tables = ov[roots[0]]["Tables"]
    assert isinstance(tables, dict) and len(tables) >= 1
    logical = next(iter(tables.keys()))
    # Use file_path directly instead of legacy root from tables_overview
    cols = fm._list_columns(table=f"{file_name}.Tables.{logical}")
    # Verify all target columns were created (consolidated spec with multiple columns)
    for target_col in target_columns:
        assert target_col in cols, f"Expected embedding column {target_col} to exist"
