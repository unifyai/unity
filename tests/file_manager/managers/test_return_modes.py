"""Tests for ingest_files return modes and typed model output.

This module tests all three return modes (compact, full, none) and verifies
that the correct Pydantic models are returned for each.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from tests.helpers import _handle_project

# =============================================================================
# RETURN MODE TESTS
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_parse_return_modes(file_manager, tmp_path: Path):
    """Test all three return modes return appropriate Pydantic models."""
    p = tmp_path / "sample.txt"
    p.write_text("Hello world. This is a sample file.", encoding="utf-8")

    from unity.file_manager.types import (
        FilePipelineConfig,
        BaseIngestedFile,
        IngestedMinimal,
    )
    from unity.file_manager.types.ingest import IngestedFullFile

    # compact (default) → typed BaseIngestedFile model
    res_compact = file_manager.ingest_files(str(p))
    assert str(p) in res_compact
    compact_item = res_compact[str(p)]
    # compact mode should return typed model with content_ref
    assert isinstance(compact_item, BaseIngestedFile)
    assert hasattr(compact_item, "content_ref")
    assert hasattr(compact_item, "metrics")
    # Observability: compact metrics should include parser timing derived from FileParseTrace
    assert compact_item.metrics.processing_time is not None

    # full → IngestedFullFile Pydantic model (parse artifacts + lowered rows + refs/metrics)
    res_full = file_manager.ingest_files(
        str(p),
        config=FilePipelineConfig(output={"return_mode": "full"}),
    )
    full_item = res_full[str(p)]
    assert isinstance(full_item, IngestedFullFile)
    assert full_item.status == "success"
    assert hasattr(full_item, "content_rows")
    assert hasattr(full_item, "full_text")
    assert hasattr(full_item, "file_format")
    assert full_item.trace is not None

    # none → IngestedMinimal Pydantic model (minimal stub)
    res_none = file_manager.ingest_files(
        str(p),
        config=FilePipelineConfig(output={"return_mode": "none"}),
    )
    none_item = res_none[str(p)]
    assert isinstance(none_item, IngestedMinimal)
    assert hasattr(none_item, "file_path")
    assert hasattr(none_item, "status")
    assert hasattr(none_item, "error")
    assert hasattr(none_item, "total_records")
    assert hasattr(none_item, "file_format")


# =============================================================================
# COMPACT MODEL TYPE TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_compact_returns_typed_models_by_format(
    file_manager,
    supported_file_examples: dict,
):
    """Test that compact mode returns format-specific Pydantic model subclasses."""
    from unity.file_manager.types import (
        BaseIngestedFile,
        IngestedPDF,
        IngestedDocx,
        IngestedXlsx,
        IngestedCsv,
        ContentRef,
        TableRef,
        FileMetrics,
        FileFormat,
        MimeType,
    )

    # Map extensions to expected compact model classes
    ext_to_model = {
        ".pdf": IngestedPDF,
        ".docx": IngestedDocx,
        ".xlsx": IngestedXlsx,
        ".csv": IngestedCsv,
    }

    # Iterate known sample files and assert typed models are returned by default (compact mode)
    for filename, example_data in supported_file_examples.items():
        ext = example_data["format"].lower()
        if ext not in ext_to_model:
            continue  # skip formats without a concrete subclass

        display_name = str(example_data["path"])  # absolute path
        res = file_manager.ingest_files(display_name)  # default compact
        assert display_name in res
        item = res[display_name]

        # Type: should be the specific subclass (not a dict)
        assert isinstance(item, ext_to_model[ext])
        assert isinstance(item, BaseIngestedFile)

        # Common fields and types
        assert isinstance(item.content_ref, ContentRef)
        assert isinstance(item.tables_ref, list)
        assert all(isinstance(t, TableRef) for t in item.tables_ref)
        assert isinstance(item.metrics, FileMetrics)
        assert item.status in ("success", "error")

        # Enums are present and correctly typed
        assert item.file_format is None or isinstance(item.file_format, FileFormat)
        assert item.mime_type is None or isinstance(item.mime_type, MimeType)

        # Subclass-specific field presence (shape checks only)
        if isinstance(item, IngestedPDF):
            assert hasattr(item, "page_count")
            assert hasattr(item, "total_sections")
        if isinstance(item, IngestedXlsx):
            assert hasattr(item, "sheet_count")
            assert isinstance(item.sheet_names, list)
        if isinstance(item, IngestedCsv):
            assert hasattr(item, "table_count")


# =============================================================================
# ASK ABOUT FILE WITH RESPONSE FORMAT
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_ask_about_file_with_response_format(file_manager, tmp_path: Path):
    """Test that ask_about_file respects response_format schema."""
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
