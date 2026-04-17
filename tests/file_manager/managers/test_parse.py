"""
FileManager parse functionality tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_parse_single(file_manager, supported_file_examples: dict):
    """Test parsing a single file."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = str(example_data["path"])  # absolute path

    from unity.file_manager.types import FilePipelineConfig
    from unity.file_manager.types.ingest import IngestedFullFile

    # Request full mode to assert parse artifacts + FileManager lowering invariants
    result = file_manager.ingest_files(
        display_name,
        config=FilePipelineConfig(output={"return_mode": "full"}),
    )

    assert display_name in result
    # Full mode returns IngestedFullFile Pydantic model
    item = result[display_name]
    assert isinstance(item, IngestedFullFile)
    assert item.status == "success"
    assert isinstance(item.content_rows, list) and item.content_rows

    # Trace/observability should be present in full mode
    assert item.trace is not None
    assert isinstance(item.trace.backend, str) and item.trace.backend


@pytest.mark.asyncio
async def test_parse_multiple(file_manager, supported_file_examples: dict):
    """Test parsing multiple files at once."""
    # Import all example files
    display_names = []
    for filename, example_data in supported_file_examples.items():
        display_name = str(example_data["path"])  # absolute path
        display_names.append(display_name)

    # Parse all files - compact is default, returns Pydantic models
    results = file_manager.ingest_files(display_names)

    assert len(results) == len(display_names)
    for display_name in display_names:
        assert display_name in results
        item = results[display_name]
        # All returns are now Pydantic models - use attribute access
        assert item.status == "success"


@pytest.mark.asyncio
async def test_parse_with_options(file_manager, supported_file_examples: dict):
    """Test parsing with custom options."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = str(example_data["path"])  # absolute path

    # Parse with options via config (forwarded to parser)
    from unity.file_manager.types import FilePipelineConfig, ParseConfig

    cfg = FilePipelineConfig(
        parse=ParseConfig(max_concurrent_parses=1),
    )
    result = file_manager.ingest_files(display_name, config=cfg)

    assert display_name in result
    item = result[display_name]
    # All returns are now Pydantic models - use attribute access
    assert item.status == "success"


@pytest.mark.asyncio
async def test_parse_can_emit_run_ledger(file_manager, tmp_path: Path):
    txt = tmp_path / "ledger.txt"
    txt.write_text("Alpha paragraph.\n\nBeta paragraph.", encoding="utf-8")
    ledger_path = tmp_path / "run_ledger.jsonl"

    from unity.file_manager.types import FilePipelineConfig

    result = file_manager.ingest_files(
        str(txt),
        config=FilePipelineConfig(
            diagnostics={
                "enable_run_ledger": True,
                "run_ledger_file": str(ledger_path),
            },
        ),
    )

    assert result[str(txt)].status == "success"
    assert ledger_path.exists()

    records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records

    run_ids = {record["run_id"] for record in records}
    assert len(run_ids) == 1

    record_types = {record["record_type"] for record in records}
    assert {"run", "file", "stage"} <= record_types

    stage_names = {
        record["stage_name"] for record in records if record["record_type"] == "stage"
    }
    assert "file_record" in stage_names
    assert "ingest_content" in stage_names

    file_records = [record for record in records if record["record_type"] == "file"]
    assert file_records
    assert file_records[0]["status"] == "success"


@pytest.mark.asyncio
async def test_parse_can_emit_correlated_progress_events(file_manager, tmp_path: Path):
    from unittest.mock import patch

    csv = tmp_path / "people.csv"
    csv.write_text("Name,Age,City\nJohn,30,NYC\nJane,25,LDN\n", encoding="utf-8")
    progress_path = tmp_path / "progress.jsonl"

    from unity.file_manager.types import FilePipelineConfig

    cfg = FilePipelineConfig(
        diagnostics={
            "enable_progress": True,
            "progress_mode": "json_file",
            "progress_file": str(progress_path),
            "verbosity": "high",
        },
    )

    with patch(
        "unity.file_manager.parse_adapter.lowering.content_rows.summarize_table_profile",
        return_value="stub table summary",
    ):
        result = file_manager.ingest_files(str(csv), config=cfg)

    assert result[str(csv)].status == "success"
    assert progress_path.exists()

    events = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events
    assert all(event.get("event_id") for event in events)

    run_ids = {event["run_id"] for event in events if event.get("run_id")}
    assert len(run_ids) == 1

    parse_events = [event for event in events if event["phase"] == "parse"]
    assert {event["status"] for event in parse_events} >= {"started", "completed"}

    stage_events = [
        event
        for event in events
        if event["phase"] in {"file_record", "ingest_content", "ingest_table"}
    ]
    assert stage_events
    assert all(event.get("stage_id") for event in stage_events)
    assert all(event.get("file_id") is not None for event in stage_events)
    assert all(event.get("storage_id") for event in stage_events)

    table_events = [event for event in events if event["phase"] == "ingest_table"]
    assert table_events
    assert table_events[0].get("table_id")
    assert table_events[0]["meta"]["table_label"] == "people"
    assert table_events[0]["meta"]["row_count"] == 2

    file_complete = [event for event in events if event["phase"] == "file_complete"]
    assert file_complete
    assert file_complete[0]["status"] == "completed"
    assert file_complete[0]["meta"]["parse_backend"] == "native_csv_backend"


@pytest.mark.asyncio
async def test_parse_can_emit_estimated_cost_ledger(file_manager, tmp_path: Path):
    from unittest.mock import patch

    csv = tmp_path / "cost_people.csv"
    csv.write_text("Name,Age,City\nJohn,30,NYC\nJane,25,LDN\n", encoding="utf-8")
    cost_ledger_path = tmp_path / "cost_ledger.jsonl"
    artifact_root = tmp_path / "artifacts"

    from unity.file_manager.types import FilePipelineConfig

    cfg = FilePipelineConfig(
        transport={
            "table_input_mode": "materialized_artifact",
            "artifact_root_dir": str(artifact_root),
        },
        cost={
            "enable_cost_ledger": True,
            "cost_ledger_file": str(cost_ledger_path),
            "tenant_id": "tenant-local",
            "environment": "test",
        },
    )

    with patch(
        "unity.file_manager.parse_adapter.lowering.content_rows.summarize_table_profile",
        return_value="stub table summary",
    ):
        result = file_manager.ingest_files(str(csv), config=cfg)

    assert result[str(csv)].status == "success"
    assert cost_ledger_path.exists()

    ledgers = [
        json.loads(line)
        for line in cost_ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(ledgers) == 1

    ledger = ledgers[0]
    assert ledger["record_type"] == "cost_ledger"
    assert ledger["tenant_id"] == "tenant-local"
    assert ledger["environment"] == "test"
    assert ledger["estimated_total"] > 0

    line_items = ledger["line_items"]
    components = {item["component"] for item in line_items}
    assert "parse_compute_cpu" in components
    assert "parse_compute_memory" in components
    assert "artifact_storage" in components
    assert "row_ingest_requests" in components
    assert "row_ingest_storage" in components
    assert "observability" in components

    artifact_item = next(
        item for item in line_items if item["component"] == "artifact_storage"
    )
    assert artifact_item["meta"]["artifact_bytes"] > 0
    assert artifact_item["meta"]["artifact_count"] == 1


def test_executor_retry_policy_retries_transient_errors_and_stops_on_non_retryable():
    from unity.common.pipeline import run_with_retry as _run_with_retry
    from unity.file_manager.types import RetryConfig

    retry_cfg = RetryConfig(
        max_retries=3,
        retry_delay_seconds=0.0,
        jitter_ratio=0.0,
        retry_mode="transient_only",
    )

    transient_attempts = {"count": 0}

    def flaky() -> str:
        transient_attempts["count"] += 1
        if transient_attempts["count"] < 3:
            raise TimeoutError("timed out")
        return "ok"

    transient_result = _run_with_retry(flaky, {}, retry_config=retry_cfg, label="flaky")
    assert transient_result.success is True
    assert transient_result.value == "ok"
    assert transient_result.retries == 2
    assert transient_attempts["count"] == 3

    fatal_attempts = {"count": 0}

    def fatal() -> None:
        fatal_attempts["count"] += 1
        raise ValueError("bad input")

    fatal_result = _run_with_retry(fatal, {}, retry_config=retry_cfg, label="fatal")
    assert fatal_result.success is False
    assert fatal_result.failure_kind == "non_retryable"
    assert fatal_attempts["count"] == 1


@pytest.mark.asyncio
async def test_parse_empty(file_manager, sample_files: Path):
    """Test parsing an empty file."""
    # Use empty file by absolute path
    empty_file = sample_files / "empty.txt"
    display_name = str(empty_file)

    from unity.file_manager.types import FilePipelineConfig
    from unity.file_manager.types.ingest import IngestedFullFile

    # Request full mode to test lowering semantics on empty file
    result = file_manager.ingest_files(
        display_name,
        config=FilePipelineConfig(output={"return_mode": "full"}),
    )

    assert display_name in result
    # Full mode returns IngestedFullFile Pydantic model
    item = result[display_name]
    assert isinstance(item, IngestedFullFile)
    # Empty file should still parse successfully
    assert item.status == "success"
    # May have a document structure but content should be empty
    rows = list(item.content_rows or [])
    assert rows, "Expected at least a document row for an empty file"


@pytest.mark.asyncio
async def test_parse_supported(file_manager, supported_file_examples: dict):
    """Test parsing files in all supported formats."""
    # Add all example files to the file manager
    display_names = []
    for filename, example_data in supported_file_examples.items():
        display_name = str(example_data["path"])  # absolute path
        display_names.append(display_name)

    # Test parsing each file individually
    from unity.file_manager.types import FilePipelineConfig
    from unity.file_manager.types.ingest import IngestedFullFile

    for display_name in display_names:
        result = file_manager.ingest_files(
            display_name,
            config=FilePipelineConfig(output={"return_mode": "full"}),
        )

        assert display_name in result
        item = result[display_name]
        assert isinstance(item, IngestedFullFile)
        assert item.status == "success"
        assert len(item.content_rows) > 0

        # Trace identity should always be coherent at the parser boundary.
        assert item.trace is not None
        assert item.trace.logical_path == display_name
        assert (
            isinstance(item.trace.source_local_path, str)
            and item.trace.source_local_path
        )
        assert item.trace.parsed_local_path is not None
        assert isinstance(item.trace.conversion_chain, list)

        from unity.file_manager.file_parsers.types.enums import ContentType

        # If this is a spreadsheet (csv or xlsx), ensure we emitted sheet/table catalog rows
        # and extracted at least one concrete table batch.
        file_format = getattr(item.file_format, "value", item.file_format)
        if str(file_format) in ("csv", "xlsx"):
            ctypes = {r.content_type for r in (item.content_rows or [])}
            assert ContentType.SHEET in ctypes
            assert ContentType.TABLE in ctypes
            assert item.tables, "Expected extracted table batches for spreadsheets"

        # For PDFs/DOCX, assert the "good bits" pipeline steps were invoked (structure + summaries + metadata).
        if item.trace.backend in ("pdf_backend", "ms_word_backend"):
            step_names = {s.name for s in (item.trace.steps or [])}
            assert "docling_convert" in step_names
            assert "docling_index_structure" in step_names
            assert "llm_enrichment" in step_names
            assert any(
                s in step_names
                for s in (
                    "build_document_graph_hybrid",
                    "build_document_graph_native",
                    "build_document_graph_text_fallback",
                )
            )
            ctypes = {r.content_type for r in (item.content_rows or [])}
            assert ContentType.SECTION in ctypes
            # Metadata is optional best-effort; when present it must already be stringified for FileRecords/search.
            if item.metadata is not None and isinstance(item.metadata, dict):
                assert isinstance(item.metadata.get("key_topics", ""), str)
                assert isinstance(item.metadata.get("named_entities", ""), str)
                assert isinstance(item.metadata.get("content_tags", ""), str)


@pytest.mark.asyncio
async def test_parse_multiple_supported(
    file_manager,
    supported_file_examples: dict,
):
    """Test parsing multiple files in supported formats."""
    # Add all example files to the file manager
    display_names = []
    for filename, example_data in supported_file_examples.items():
        display_name = str(example_data["path"])  # absolute path
        display_names.append(display_name)

    # Parse all files at once
    from unity.file_manager.types import FilePipelineConfig
    from unity.file_manager.types.ingest import IngestedFullFile

    result = file_manager.ingest_files(
        display_names,
        config=FilePipelineConfig(output={"return_mode": "full"}),
    )

    for display_name in display_names:
        assert display_name in result
        file_result = result[display_name]
        assert isinstance(file_result, IngestedFullFile)
        assert file_result.status == "success"
        assert len(file_result.content_rows) > 0

    # Sanity check that at least one parsed file produced extracted tables.
    assert any(bool(getattr(r, "tables", []) or []) for r in result.values())


@pytest.mark.asyncio
async def test_parse_trace_backend_routing(file_manager, tmp_path: Path):
    """Smoke-test backend routing via FileParseResult.trace.backend + basic lowering invariants."""
    from unittest.mock import patch

    from unity.file_manager.file_parsers.types.formats import FileFormat
    from unity.file_manager.file_parsers.types.enums import ContentType
    from unity.file_manager.types import (
        BusinessContextsConfig,
        FileBusinessContextSpec,
        FilePipelineConfig,
        TableBusinessContextSpec,
    )

    # --------------------------- TXT -> text_backend --------------------------- #
    txt = tmp_path / "route.txt"
    txt.write_text("Hello world. Second sentence.", encoding="utf-8")
    txt_path = str(txt)
    txt_res = file_manager.ingest_files(
        txt_path,
        config=FilePipelineConfig(output={"return_mode": "full"}),
    )
    out_txt = txt_res[txt_path]
    assert out_txt.status == "success"
    assert out_txt.trace is not None
    assert out_txt.trace.backend == "text_backend"
    assert out_txt.trace.logical_path == txt_path
    assert out_txt.trace.source_local_path and out_txt.trace.parsed_local_path
    assert out_txt.trace.parsed_local_path == out_txt.trace.source_local_path
    assert out_txt.trace.conversion_chain == []
    assert out_txt.file_format == FileFormat.TXT
    assert out_txt.graph is not None
    ctypes_txt = {r.content_type for r in (out_txt.content_rows or [])}
    assert ContentType.DOCUMENT in ctypes_txt
    assert ContentType.PARAGRAPH in ctypes_txt

    # ---------------------- CSV -> native_csv_backend ------------------------ #
    csv = tmp_path / "people.csv"
    csv.write_text("Name,Age,City\nJohn,30,NYC\nJane,25,LDN\n", encoding="utf-8")
    csv_path = str(csv)
    cfg = FilePipelineConfig(
        output={"return_mode": "full"},
        ingest={
            "business_contexts": BusinessContextsConfig(
                global_rules=[],
                file_contexts=[
                    FileBusinessContextSpec(
                        file_path=csv_path,
                        file_rules=[],
                        table_contexts=[
                            TableBusinessContextSpec(
                                table="people",
                                table_rules=[],
                                column_descriptions={"Name": "Person's full name"},
                                table_description="People directory",
                            ),
                        ],
                    ),
                ],
            ),
        },
    )
    with patch(
        "unity.file_manager.parse_adapter.lowering.content_rows.summarize_table_profile",
        return_value="stub table summary",
    ):
        csv_res = file_manager.ingest_files(csv_path, config=cfg)
    out_csv = csv_res[csv_path]
    assert out_csv.status == "success"
    assert out_csv.trace is not None
    assert out_csv.trace.backend == "native_csv_backend"
    assert out_csv.trace.logical_path == csv_path
    assert out_csv.trace.source_local_path and out_csv.trace.parsed_local_path
    assert out_csv.trace.parsed_local_path == out_csv.trace.source_local_path
    assert out_csv.trace.conversion_chain == []
    assert out_csv.file_format == FileFormat.CSV
    assert out_csv.tables and out_csv.tables[0].label == "people"
    ctypes_csv = {r.content_type for r in (out_csv.content_rows or [])}
    assert ContentType.SHEET in ctypes_csv
    assert ContentType.TABLE in ctypes_csv
    table_rows = [
        r for r in (out_csv.content_rows or []) if r.content_type == ContentType.TABLE
    ]
    assert any((r.summary or "") == "stub table summary" for r in table_rows)

    # ---------------------- UNKNOWN ext -> TXT fallback ----------------------- #
    unk = tmp_path / "mystery.unknown"
    unk.write_text("Fallback text parse", encoding="utf-8")
    unk_path = str(unk)
    unk_res = file_manager.ingest_files(
        unk_path,
        config=FilePipelineConfig(output={"return_mode": "full"}),
    )
    out_unk = unk_res[unk_path]
    assert out_unk.status == "success"
    assert out_unk.trace is not None
    assert out_unk.trace.backend == "text_backend"
    assert out_unk.trace.logical_path == unk_path
    assert out_unk.trace.source_local_path and out_unk.trace.parsed_local_path
    assert out_unk.trace.parsed_local_path == out_unk.trace.source_local_path
    assert out_unk.trace.conversion_chain == []
    assert out_unk.file_format == FileFormat.TXT
