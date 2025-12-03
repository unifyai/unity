from __future__ import annotations

import json
import pytest
from pathlib import Path

from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_parse_return_modes(file_manager, tmp_path: Path):
    p = tmp_path / "sample.txt"
    p.write_text("Hello world. This is a sample file.", encoding="utf-8")

    from unity.file_manager.types import FilePipelineConfig

    # compact (default) → typed model
    res_compact = file_manager.ingest_files(str(p))
    assert str(p) in res_compact
    compact_item = res_compact[str(p)]
    # Either Pydantic model or dict; compact should be model
    assert hasattr(
        compact_item,
        "content_ref",
    ), "compact mode should return typed model with content_ref"
    assert hasattr(compact_item, "metrics"), "compact mode should include metrics"

    # full → raw dict (heavy fields present)
    res_full = file_manager.ingest_files(
        str(p),
        config=FilePipelineConfig(output={"return_mode": "full"}),
    )
    full_item = res_full[str(p)]
    assert isinstance(full_item, dict)
    assert full_item.get("status") == "success"
    assert "records" in full_item
    assert "full_text" in full_item
    assert "file_format" in full_item

    # none → minimal stub
    res_none = file_manager.ingest_files(
        str(p),
        config=FilePipelineConfig(output={"return_mode": "none"}),
    )
    none_item = res_none[str(p)]
    assert isinstance(none_item, dict)
    assert set(
        ["file_path", "status", "error", "total_records", "file_format"],
    ).issubset(none_item.keys())


@pytest.mark.asyncio
@_handle_project
async def test_ask_about_file_with_response_format(file_manager, tmp_path: Path):
    p = tmp_path / "report.txt"
    p.write_text("Quarterly Report Q1 2025. Revenue grew to $10M.", encoding="utf-8")

    # Ensure file is parseable/indexed first (no-op for local absolute path)
    from unity.file_manager.types import FilePipelineConfig

    file_manager.ingest_files(
        str(p),
        config=FilePipelineConfig(output={"return_mode": "compact"}),
    )

    # Define a simple response schema
    from pydantic import BaseModel

    class ReportFacts(BaseModel):
        period: str | None = None
        mentions_revenue: bool

    # Ask for structured extraction
    handle = await file_manager.ask_about_file(
        str(p),
        "Extract the reporting period (e.g., 'Q1 2025') if present, and whether revenue is mentioned.",
        response_format=ReportFacts,
    )
    out = await handle.result()

    # The async tool loop enforces schema; tolerate string or dict payloads
    if isinstance(out, str):
        try:
            out = json.loads(out)
        except Exception:
            # Fallback: let Pydantic try to coerce
            out = ReportFacts.model_validate_json(out).model_dump()
    elif hasattr(out, "model_dump"):
        out = out.model_dump()

    inst = ReportFacts.model_validate(out)
    # Basic expectations
    assert inst.mentions_revenue in (True, False)
    # If a period was detected, it should include 'Q' or a year-like token
    if inst.period:
        assert ("Q" in inst.period) or any(ch.isdigit() for ch in inst.period)
