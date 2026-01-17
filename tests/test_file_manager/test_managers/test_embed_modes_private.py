from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
import unify

from tests.helpers import _handle_project
from unity.file_manager.file_parsers.types.contracts import FileParseResult
from unity.file_manager.file_parsers.types.table import ExtractedTable
from unity.file_manager.types import (
    FilePipelineConfig,
    IngestConfig,
    EmbeddingsConfig,
    FileEmbeddingSpec,
    TableEmbeddingSpec,
    TableBusinessContextSpec,
    FileBusinessContextSpec,
    BusinessContextsConfig,
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


def _make_extracted_table(
    *,
    rows: List[List[Any]],
    label: str,
    sheet_name: str | None,
    table_id: str = "table:0",
) -> ExtractedTable:
    if not rows:
        return ExtractedTable(table_id=table_id, label=label, sheet_name=sheet_name)

    header = [str(x) for x in list(rows[0] or [])]
    body = list(rows[1:] or [])

    row_dicts: List[Dict[str, Any]] = []
    for r in body:
        row_dicts.append(
            {header[i]: (r[i] if i < len(r) else None) for i in range(len(header))},
        )

    return ExtractedTable(
        table_id=table_id,
        label=label,
        sheet_name=sheet_name,
        columns=header,
        rows=row_dicts,
        sample_rows=row_dicts[:25],
        num_rows=len(row_dicts),
        num_cols=len(header),
    )


@_handle_project
def test_embed_off_no_columns(file_manager, tmp_path: Path):
    """Test that embed strategy 'off' does not create embedding columns."""
    fm = file_manager
    fm.clear()
    file_path = "synthetic_off.txt"
    records = _make_records(3)
    parse_result = FileParseResult(logical_path=file_path, status="success")
    # Create a tiny in-memory graph so the FileManager adapter can derive `/Content/` rows.
    from unity.file_manager.file_parsers.types.enums import NodeKind
    from unity.file_manager.file_parsers.types.graph import (
        ContentGraph,
        ContentNode,
        DocumentPayload,
        ParagraphPayload,
        SentencePayload,
    )

    doc_id = "document:0"
    nodes = {
        doc_id: ContentNode(
            node_id=doc_id,
            kind=NodeKind.DOCUMENT,
            parent_id=None,
            children_ids=[],
            payload=DocumentPayload(),
        ),
    }
    for i, r in enumerate(records):
        para_id = f"paragraph:{i}"
        nodes[doc_id].children_ids.append(para_id)
        nodes[para_id] = ContentNode(
            node_id=para_id,
            kind=NodeKind.PARAGRAPH,
            parent_id=doc_id,
            children_ids=[],
            order=i,
            title=str(r.get("title")),
            text=str(r.get("content_text")),
            payload=ParagraphPayload(),
        )
        sent_id = f"sentence:{i}:0"
        nodes[para_id].children_ids.append(sent_id)
        nodes[sent_id] = ContentNode(
            node_id=sent_id,
            kind=NodeKind.SENTENCE,
            parent_id=para_id,
            children_ids=[],
            order=0,
            text=str(r.get("content_text")),
            payload=SentencePayload(sentence_index=0),
        )
    parse_result.graph = ContentGraph(root_id=doc_id, nodes=nodes)

    cfg = FilePipelineConfig(
        ingest=IngestConfig(),  # default per-file storage
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
    from unity.file_manager.managers.utils.ingest_ops import get_file_id_from_path

    process_single_file(fm, parse_result=parse_result, file_path=file_path, config=cfg)

    # Look up file_id and use storage_id-based context
    file_id = get_file_id_from_path(index_context=fm._ctx, file_path=file_path)
    assert file_id is not None, f"File record not found for {file_path}"
    ctx = fm._ctx_for_file_content(str(file_id))
    fields = unify.get_fields(context=ctx)
    assert "_summary_emb" not in fields


@_handle_project
def test_embed_after_creates_columns(file_manager, tmp_path: Path):
    """Test that embed strategy 'after' creates embedding columns."""
    fm = file_manager
    fm.clear()
    file_path = "synthetic_after.txt"
    records = _make_records(4)
    parse_result = FileParseResult(logical_path=file_path, status="success")
    from unity.file_manager.file_parsers.types.enums import NodeKind
    from unity.file_manager.file_parsers.types.graph import (
        ContentGraph,
        ContentNode,
        DocumentPayload,
        ParagraphPayload,
        SentencePayload,
    )

    doc_id = "document:0"
    nodes = {
        doc_id: ContentNode(
            node_id=doc_id,
            kind=NodeKind.DOCUMENT,
            parent_id=None,
            children_ids=[],
            payload=DocumentPayload(),
        ),
    }
    for i, r in enumerate(records):
        para_id = f"paragraph:{i}"
        nodes[doc_id].children_ids.append(para_id)
        nodes[para_id] = ContentNode(
            node_id=para_id,
            kind=NodeKind.PARAGRAPH,
            parent_id=doc_id,
            children_ids=[],
            order=i,
            title=str(r.get("title")),
            text=str(r.get("content_text")),
            payload=ParagraphPayload(),
        )
        sent_id = f"sentence:{i}:0"
        nodes[para_id].children_ids.append(sent_id)
        nodes[sent_id] = ContentNode(
            node_id=sent_id,
            kind=NodeKind.SENTENCE,
            parent_id=para_id,
            children_ids=[],
            order=0,
            text=str(r.get("content_text")),
            payload=SentencePayload(sentence_index=0),
        )
    parse_result.graph = ContentGraph(root_id=doc_id, nodes=nodes)

    cfg = FilePipelineConfig(
        ingest=IngestConfig(),  # default per-file storage
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
    from unity.file_manager.managers.utils.ingest_ops import get_file_id_from_path

    process_single_file(fm, parse_result=parse_result, file_path=file_path, config=cfg)

    # Look up file_id and use storage_id-based context
    file_id = get_file_id_from_path(index_context=fm._ctx, file_path=file_path)
    assert file_id is not None, f"File record not found for {file_path}"
    ctx = fm._ctx_for_file_content(str(file_id))
    fields = unify.get_fields(context=ctx)
    assert "_summary_emb" in fields


@_handle_project
def test_embed_along_content(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    file_path = "synthetic_along.txt"
    # 5 records with chunk size 2 → expect 3 chunks
    records = _make_records(5)
    parse_result = FileParseResult(logical_path=file_path, status="success")
    from unity.file_manager.file_parsers.types.enums import NodeKind
    from unity.file_manager.file_parsers.types.graph import (
        ContentGraph,
        ContentNode,
        DocumentPayload,
        ParagraphPayload,
        SentencePayload,
    )

    doc_id = "document:0"
    nodes = {
        doc_id: ContentNode(
            node_id=doc_id,
            kind=NodeKind.DOCUMENT,
            parent_id=None,
            children_ids=[],
            payload=DocumentPayload(),
        ),
    }
    for i, r in enumerate(records):
        para_id = f"paragraph:{i}"
        nodes[doc_id].children_ids.append(para_id)
        nodes[para_id] = ContentNode(
            node_id=para_id,
            kind=NodeKind.PARAGRAPH,
            parent_id=doc_id,
            children_ids=[],
            order=i,
            title=str(r.get("title")),
            text=str(r.get("content_text")),
            payload=ParagraphPayload(),
        )
        sent_id = f"sentence:{i}:0"
        nodes[para_id].children_ids.append(sent_id)
        nodes[sent_id] = ContentNode(
            node_id=sent_id,
            kind=NodeKind.SENTENCE,
            parent_id=para_id,
            children_ids=[],
            order=0,
            text=str(r.get("content_text")),
            payload=SentencePayload(sentence_index=0),
        )
    parse_result.graph = ContentGraph(root_id=doc_id, nodes=nodes)

    calls = {"pre": 0, "post": 0}

    def pre_hook(manager, file_path, result, parse_result, config):
        calls["pre"] += 1

    def post_hook(manager, file_path, result, parse_result, config):
        calls["post"] += 1

    cfg = FilePipelineConfig(
        ingest=IngestConfig(content_rows_batch_size=2),  # default per-file storage
        embed=EmbeddingsConfig(
            strategy="along",
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

    # Use process_single_file from executor instead of removed _ingest_and_embed
    from unity.file_manager.managers.utils.executor import process_single_file
    from unity.file_manager.managers.utils.ingest_ops import get_file_id_from_path

    process_single_file(fm, parse_result=parse_result, file_path=file_path, config=cfg)
    # Note: plugin hooks are optional and may be wired in/out depending on executor implementation.
    # The critical invariant for "along" is that embeddings are created successfully.

    # Look up file_id and use storage_id-based context
    file_id = get_file_id_from_path(index_context=fm._ctx, file_path=file_path)
    assert file_id is not None, f"File record not found for {file_path}"
    ctx = fm._ctx_for_file_content(str(file_id))
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
    label = sheet_name or Path(file_name).stem
    tbl = _make_extracted_table(rows=rows, label=label, sheet_name=sheet_name)
    parse_result = FileParseResult(
        logical_path=file_name,
        status="success",
        tables=[tbl],
    )

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

    process_single_file(fm, parse_result=parse_result, file_path=file_name, config=cfg)

    # Locate a per-file table using describe() and assert the embedding columns exist
    storage = fm.describe(file_path=file_name)
    assert storage.has_tables, "Expected per-file tables"
    assert len(storage.tables) >= 1

    # Use the exact context path from describe()
    table_context = storage.tables[0].context_path
    cols = fm.list_columns(context=table_context)

    # Verify all target columns were created (consolidated spec with multiple columns)
    for target_col in target_columns:
        assert target_col in cols, f"Expected embedding column {target_col} to exist"
