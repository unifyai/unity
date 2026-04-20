from __future__ import annotations


import pytest

from tests.helpers import _handle_project
from unity.common.pipeline.instrumentation import PipelineInstrumentation
from unity.file_manager.managers.local import LocalFileManager
from unity.file_manager.file_parsers.types.contracts import FileParseResult
from unity.file_manager.file_parsers.types.enums import NodeKind
from unity.file_manager.file_parsers.types.graph import (
    ContentGraph,
    ContentNode,
    DocumentPayload,
    ParagraphPayload,
)
from unity.file_manager.types.config import FilePipelineConfig


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_file_manager_reduce_param_shapes(file_manager: LocalFileManager, tmp_path):
    # Start from a clean slate so metrics only see rows created in this test
    file_manager.clear()

    # Seed the index using process_single_file from the executor
    from unity.file_manager.managers.utils.executor import process_single_file

    cfg = FilePipelineConfig()

    # Create a couple of dummy files in the FileRecords index
    for i in range(2):
        doc_id = "document:0"
        para_id = "paragraph:0"
        parse_result = FileParseResult(
            logical_path=f"dummy_{i}.txt",
            status="success",
            graph=ContentGraph(
                root_id=doc_id,
                nodes={
                    doc_id: ContentNode(
                        node_id=doc_id,
                        kind=NodeKind.DOCUMENT,
                        parent_id=None,
                        children_ids=[para_id],
                        payload=DocumentPayload(),
                    ),
                    para_id: ContentNode(
                        node_id=para_id,
                        kind=NodeKind.PARAGRAPH,
                        parent_id=doc_id,
                        children_ids=[],
                        text=f"content_{i}",
                        payload=ParagraphPayload(),
                    ),
                },
            ),
        )
        process_single_file(
            file_manager,
            parse_result=parse_result,
            file_path=f"dummy_{i}.txt",
            config=cfg,
            instrumentation=PipelineInstrumentation(),
        )

    # Single column, no grouping
    scalar = file_manager.reduce(metric="sum", columns="file_id")
    assert isinstance(scalar, (int, float))

    # Multiple columns, no grouping
    multi = file_manager.reduce(metric="max", columns=["file_id"])
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"file_id"}

    # Single column, group_by string
    grouped_str = file_manager.reduce(
        metric="sum",
        columns="file_id",
        group_by="status",
    )
    assert isinstance(grouped_str, dict)

    # Multiple columns, group_by string
    grouped_str_multi = file_manager.reduce(
        metric="min",
        columns=["file_id"],
        group_by="status",
    )
    assert isinstance(grouped_str_multi, dict)

    # Single column, group_by list
    grouped_list = file_manager.reduce(
        metric="sum",
        columns="file_id",
        group_by=["status", "file_id"],
    )
    assert isinstance(grouped_list, dict)

    # Multiple columns, group_by list
    grouped_list_multi = file_manager.reduce(
        metric="mean",
        columns=["file_id"],
        group_by=["status", "file_id"],
    )
    assert isinstance(grouped_list_multi, dict)

    # Filter as string
    filtered_scalar = file_manager.reduce(
        metric="sum",
        columns="file_id",
        filter="file_id >= 0",
    )
    assert isinstance(filtered_scalar, (int, float))

    # Filter as per-column dict
    filtered_multi = file_manager.reduce(
        metric="sum",
        columns=["file_id"],
        filter={"file_id": "file_id >= 0"},
    )
    assert isinstance(filtered_multi, dict)
