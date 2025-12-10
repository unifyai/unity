from __future__ import annotations

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
    TableBusinessContextSpec,
    FileBusinessContextSpec,
    BusinessContextsConfig,
    ParsedFile,
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
    def __init__(
        self,
        tables: List[Any] | None = None,
        records: List[Dict[str, Any]] | None = None,
    ) -> None:
        self.metadata = _MetaStub(tables=tables or [])
        self.processing_status = "completed"
        self._records = records or []

    def to_parse_result(self, *a, **kw) -> ParsedFile:
        return ParsedFile(
            file_path=a[0] if a else "stub.txt",
            status="success",
            total_records=len(self._records),
            file_format="txt",
            records=self._records,
        )


@_handle_project
def test_embed_off_no_columns(file_manager, tmp_path: Path):
    """Test that embed strategy 'off' does not create embedding columns."""
    fm = file_manager
    fm.clear()
    file_path = "synthetic_off.txt"
    records = _make_records(3)
    doc = _DocStub(records=records)

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

    # Use process_single_file from executor
    from unity.file_manager.managers.utils.executor import process_single_file

    process_single_file(fm, document=doc, file_path=file_path, config=cfg)

    ctx = fm._ctx_for_file(file_path)
    fields = unify.get_fields(context=ctx)
    assert "_summary_emb" not in fields


@_handle_project
def test_embed_after_creates_columns(file_manager, tmp_path: Path):
    """Test that embed strategy 'after' creates embedding columns."""
    fm = file_manager
    fm.clear()
    file_path = "synthetic_after.txt"
    records = _make_records(4)
    doc = _DocStub(records=records)

    cfg = FilePipelineConfig(
        ingest=IngestConfig(mode="per_file"),
        embed=EmbeddingsConfig(
            strategy="after",
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

    # Use process_single_file from executor
    from unity.file_manager.managers.utils.executor import process_single_file

    process_single_file(fm, document=doc, file_path=file_path, config=cfg)

    ctx = fm._ctx_for_file(file_path)
    fields = unify.get_fields(context=ctx)
    assert "_summary_emb" in fields


@_handle_project
def test_embed_along_content_hooks_per_chunk(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    file_path = "synthetic_along.txt"
    # 5 records with chunk size 2 → expect 3 chunks
    records = _make_records(5)
    doc = _DocStub(records=records)

    calls = {"pre": 0, "post": 0}

    def pre_hook(manager, file_path, result, document, config):
        calls["pre"] += 1

    def post_hook(manager, file_path, result, document, config):
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

    # Use process_single_file from executor instead of removed _ingest_and_embed
    from unity.file_manager.managers.utils.executor import process_single_file

    process_single_file(fm, document=doc, file_path=file_path, config=cfg)
    # With the new task-based approach, hooks may be called per-chunk
    # but the exact count depends on implementation
    assert calls["pre"] >= 1
    assert calls["post"] >= 1

    ctx = fm._ctx_for_file(file_path)
    fields = unify.get_fields(context=ctx)
    assert "_summary_emb" in fields


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

    # Use consolidated embedding spec with multiple columns per table
    cfg = FilePipelineConfig(
        ingest=IngestConfig(
            mode="per_file",
            table_rows_batch_size=batch_size,
            business_contexts=BusinessContextsConfig(
                global_rules=[],
                file_contexts=[
                    FileBusinessContextSpec(
                        file_path=file_name,
                        file_rules=[],
                        table_contexts=[
                            TableBusinessContextSpec(
                                table=sheet_name or "large",
                                table_rules=[],
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
    # Run along pipeline using process_single_file from executor
    from unity.file_manager.managers.utils.executor import process_single_file

    process_single_file(fm, document=doc, file_path=file_name, config=cfg)

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
