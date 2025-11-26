from __future__ import annotations

import pytest
from types import SimpleNamespace

from tests.helpers import _handle_project
from tests.assertion_helpers import assertion_failed
from unity.file_manager.managers.local import LocalFileManager
from unity.file_manager.types.config import FilePipelineConfig


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_file_manager_reduce_param_shapes(file_manager: LocalFileManager, tmp_path):
    # Start from a clean slate so metrics only see rows created in this test
    file_manager.clear()

    # Manually seed the index and per-file Content via the private _ingest helper,
    # mirroring what parse() would normally do but with lightweight dummy data.
    cfg = FilePipelineConfig()
    doc = SimpleNamespace(metadata=SimpleNamespace(tables=[]))
    base_result: dict[str, object] = {
        "status": "success",
        "error": None,
        "summary": "dummy summary",
        "file_format": None,
        "file_size": 123,
        "total_records": 1,
        "processing_time": 0.0,
        "created_at": "2025-01-01T00:00:00Z",
        "modified_at": "2025-01-01T00:00:00Z",
        "confidence_score": 1.0,
        "key_topics": [],
        "named_entities": {},
        "content_tags": [],
    }

    # Create a couple of dummy files in the FileRecords index
    for i in range(2):
        result = dict(base_result)
        file_manager._ingest(
            file_path=f"dummy_{i}.txt",
            document=doc,
            result=result,
            config=cfg,
        )

    # Single key, no grouping
    scalar = file_manager._reduce(metric="sum", keys="file_id")
    assert isinstance(scalar, (int, float))

    # Multiple keys, no grouping
    multi = file_manager._reduce(metric="max", keys=["file_id"])
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"file_id"}

    # Single key, group_by string
    grouped_str = file_manager._reduce(
        metric="sum",
        keys="file_id",
        group_by="status",
    )
    assert isinstance(grouped_str, dict)

    # Multiple keys, group_by string
    grouped_str_multi = file_manager._reduce(
        metric="min",
        keys=["file_id"],
        group_by="status",
    )
    assert isinstance(grouped_str_multi, dict)

    # Single key, group_by list
    grouped_list = file_manager._reduce(
        metric="sum",
        keys="file_id",
        group_by=["status", "file_id"],
    )
    assert isinstance(grouped_list, dict)

    # Multiple keys, group_by list
    grouped_list_multi = file_manager._reduce(
        metric="mean",
        keys=["file_id"],
        group_by=["status", "file_id"],
    )
    assert isinstance(grouped_list_multi, dict)

    # Filter as string
    filtered_scalar = file_manager._reduce(
        metric="sum",
        keys="file_id",
        filter="file_id >= 0",
    )
    assert isinstance(filtered_scalar, (int, float))

    # Filter as per-key dict
    filtered_multi = file_manager._reduce(
        metric="sum",
        keys=["file_id"],
        filter={"file_id": "file_id >= 0"},
    )
    assert isinstance(filtered_multi, dict)


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_ask_uses_reduce_for_numeric_aggregation(
    file_manager: LocalFileManager,
    tmp_path,
):
    """Verify LLM uses reduce tool for numeric aggregation questions."""
    # Start from a clean slate so metrics only see rows created in this test
    file_manager.clear()

    # Manually seed the index via the private _ingest helper
    cfg = FilePipelineConfig()
    doc = SimpleNamespace(metadata=SimpleNamespace(tables=[]))
    base_result: dict[str, object] = {
        "status": "success",
        "error": None,
        "summary": "dummy summary",
        "file_format": None,
        "file_size": 123,
        "total_records": 1,
        "processing_time": 0.0,
        "created_at": "2025-01-01T00:00:00Z",
        "modified_at": "2025-01-01T00:00:00Z",
        "confidence_score": 1.0,
        "key_topics": [],
        "named_entities": {},
        "content_tags": [],
    }

    # Create a couple of dummy files in the FileRecords index
    for i in range(3):
        result = dict(base_result)
        result["file_size"] = 100 + i * 50  # Varying file sizes
        file_manager._ingest(
            file_path=f"dummy_{i}.txt",
            document=doc,
            result=result,
            config=cfg,
        )

    handle = await file_manager.ask(
        "What is the average file size for all files?",
        _return_reasoning_steps=True,
    )
    answer, steps = await handle.result()

    # Assert reduce tool was called
    reduce_called = any(
        any(
            "reduce" in (tc.get("function", {}).get("name", "") or "").lower()
            for tc in (step.get("tool_calls") or [])
        )
        for step in steps
        if step.get("role") == "assistant"
    )
    assert reduce_called, assertion_failed(
        "reduce tool to be called",
        f"steps without reduce: {[s for s in steps if s.get('role') == 'assistant']}",
        steps,
        "LLM should use reduce tool for numeric aggregation",
    )
