from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import _handle_project
from unity.file_manager.types import (
    BaseIngestedFile,
    FileFormat,
    FilePipelineConfig,
    IngestPipelineResult,
    IngestedCsv,
    IngestedMinimal,
    ParseConfig,
    IngestedXlsx,
)
from unity.file_manager.types.ingest import IngestedFullFile


def _assert_non_empty(s: str, *, label: str) -> None:
    assert isinstance(s, str), f"{label} must be str"
    assert s.strip(), f"{label} must be non-empty"


@_handle_project
def test_return_mode_full_returns_ingested_full_file(
    rootless_file_manager,
    tmp_path: Path,
):
    fm = rootless_file_manager
    p = tmp_path / "return_full.txt"
    p.write_text("alpha beta gamma", encoding="utf-8")

    cfg = FilePipelineConfig()
    cfg.output.return_mode = "full"
    cfg.embed.strategy = "off"

    out = fm.ingest_files(str(p), config=cfg)
    assert isinstance(out, IngestPipelineResult)

    item = out[str(p)]
    assert isinstance(item, IngestedFullFile)
    assert item.file_path == str(p)
    assert item.status == "success"
    _assert_non_empty(item.full_text, label="full_text")
    _assert_non_empty(item.summary, label="summary")
    assert item.content_rows, "Expected lowered /Content rows in full mode"


@_handle_project
def test_return_mode_full_metadata_is_plain_dict_and_strings(
    rootless_file_manager,
    tmp_path: Path,
):
    fm = rootless_file_manager
    p = tmp_path / "return_full_meta.txt"
    p.write_text(
        "Acme Corp shipped 12 widgets to Berlin on 2025-01-01.",
        encoding="utf-8",
    )

    cfg = FilePipelineConfig()
    cfg.output.return_mode = "full"
    cfg.embed.strategy = "off"

    out = fm.ingest_files(str(p), config=cfg)
    item = out[str(p)]
    assert isinstance(item, IngestedFullFile)
    assert item.status == "success"

    assert isinstance(item.metadata, dict)
    # These are required to be comma-separated strings (not lists) in parse outputs.
    assert isinstance(item.metadata.get("key_topics", ""), str)
    assert isinstance(item.metadata.get("named_entities", ""), str)
    assert isinstance(item.metadata.get("content_tags", ""), str)


@_handle_project
def test_return_mode_none_returns_ingested_minimal(
    rootless_file_manager,
    tmp_path: Path,
):
    fm = rootless_file_manager
    p = tmp_path / "return_none.txt"
    p.write_text("alpha beta gamma", encoding="utf-8")

    cfg = FilePipelineConfig()
    cfg.output.return_mode = "none"
    cfg.embed.strategy = "off"

    out = fm.ingest_files(str(p), config=cfg)
    assert isinstance(out, IngestPipelineResult)

    item = out[str(p)]
    assert isinstance(item, IngestedMinimal)
    assert item.file_path == str(p)
    assert item.status == "success"


@_handle_project
def test_return_mode_none_reports_total_records_and_file_format(
    rootless_file_manager,
    tmp_path: Path,
):
    fm = rootless_file_manager
    p = tmp_path / "return_none_counts.txt"
    p.write_text("Header\n\nParagraph one. Paragraph two.", encoding="utf-8")

    cfg = FilePipelineConfig()
    cfg.output.return_mode = "none"
    cfg.embed.strategy = "off"

    out = fm.ingest_files(str(p), config=cfg)
    item = out[str(p)]
    assert isinstance(item, IngestedMinimal)
    assert item.status == "success"
    assert item.total_records > 0
    assert item.file_format is not None
    assert "txt" in item.file_format.lower()


@_handle_project
def test_return_mode_compact_returns_reference_first_model(
    rootless_file_manager,
    tmp_path: Path,
):
    fm = rootless_file_manager
    p = tmp_path / "return_compact.txt"
    p.write_text("alpha beta gamma", encoding="utf-8")

    cfg = FilePipelineConfig()
    cfg.output.return_mode = "compact"
    cfg.embed.strategy = "off"

    out = fm.ingest_files(str(p), config=cfg)
    assert isinstance(out, IngestPipelineResult)

    item = out[str(p)]
    assert isinstance(item, BaseIngestedFile)
    assert item.file_path == str(p)
    assert item.status == "success"
    assert item.content_ref is not None


@_handle_project
def test_return_mode_compact_populates_content_ref_and_summary_excerpt(
    rootless_file_manager,
    tmp_path: Path,
):
    fm = rootless_file_manager
    p = tmp_path / "return_compact_refs.txt"
    p.write_text("alpha beta gamma delta epsilon", encoding="utf-8")

    cfg = FilePipelineConfig()
    cfg.output.return_mode = "compact"
    cfg.embed.strategy = "off"

    out = fm.ingest_files(str(p), config=cfg)
    item = out[str(p)]
    assert isinstance(item, BaseIngestedFile)
    assert item.status == "success"
    assert item.file_format == FileFormat.TXT
    assert item.content_ref.context
    assert item.content_ref.record_count > 0
    assert (item.summary_excerpt or "").strip()
    assert item.metrics.processing_time is not None


@pytest.mark.parametrize(("kind",), [("csv",), ("xlsx",)])
@_handle_project
def test_return_mode_compact_selects_spreadsheet_models_for_csv_and_xlsx(
    rootless_file_manager,
    tmp_path: Path,
    kind: str,
    write_minimal_xlsx,
):
    fm = rootless_file_manager
    if kind == "csv":
        p = tmp_path / "book.csv"
        p.write_text("A,B\n1,2\n3,4\n", encoding="utf-8")
        expected_cls = IngestedCsv
        expected_fmt = FileFormat.CSV
    else:
        p = tmp_path / "book.xlsx"
        write_minimal_xlsx(p, sheets=[("Sheet1", [["A", "B"], ["1", "2"], ["3", "4"]])])
        expected_cls = IngestedXlsx
        expected_fmt = FileFormat.XLSX

    cfg = FilePipelineConfig()
    cfg.output.return_mode = "compact"
    cfg.embed.strategy = "off"

    out = fm.ingest_files(str(p), config=cfg)
    item = out[str(p)]
    assert isinstance(item, expected_cls)
    assert item.status == "success"
    assert item.file_format == expected_fmt
    assert item.content_ref.context
    assert item.content_ref.record_count >= 0

    if kind == "csv":
        assert item.table_count is not None
        assert item.table_count >= 1
        assert len(item.tables_ref) >= 1
    else:
        assert item.sheet_count is not None
        assert item.table_count is not None


@_handle_project
def test_return_mode_full_does_not_mix_raw_file_parse_result_on_error(
    rootless_file_manager,
    tmp_path: Path,
):
    fm = rootless_file_manager
    missing = tmp_path / "missing_full.txt"
    assert not missing.exists()

    cfg = FilePipelineConfig()
    cfg.output.return_mode = "full"
    cfg.embed.strategy = "off"

    out = fm.ingest_files(str(missing), config=cfg)
    assert isinstance(out, IngestPipelineResult)

    item = out[str(missing)]
    assert isinstance(item, IngestedFullFile)
    assert item.status == "error"


@_handle_project
def test_ingest_files_is_best_effort_for_mixed_success_and_failure(
    rootless_file_manager,
    tmp_path: Path,
):
    """
    Deep integration: ingest_files should not catastrophically fail when one input is invalid.
    It must return a per-file result for *all* requested paths.
    """
    fm = rootless_file_manager
    good = tmp_path / "ok.txt"
    good.write_text("alpha beta gamma", encoding="utf-8")
    missing = tmp_path / "missing.txt"
    assert not missing.exists()

    cfg = FilePipelineConfig()
    cfg.output.return_mode = "compact"
    cfg.embed.strategy = "off"

    out = fm.ingest_files([str(good), str(missing)], config=cfg)
    assert isinstance(out, IngestPipelineResult)

    ok_item = out[str(good)]
    bad_item = out[str(missing)]

    assert isinstance(ok_item, BaseIngestedFile)
    assert ok_item.status == "success"
    assert isinstance(bad_item, BaseIngestedFile)
    assert bad_item.status == "error"


@_handle_project
def test_file_manager_passes_parse_config_backend_overrides_into_file_parser_full_mode(
    rootless_file_manager,
    tmp_path: Path,
):
    """
    End-to-end: FileManager.ingest_files must pass the typed ParseConfig through to FileParser.parse_batch,
    so backend hot-swaps work without any manual registry/config parsing in FileManager.
    """
    fm = rootless_file_manager
    p = tmp_path / "doc.txt"
    p.write_text("hello", encoding="utf-8")

    cfg = FilePipelineConfig()
    cfg.output.return_mode = "full"
    cfg.embed.strategy = "off"
    cfg.parse = ParseConfig(
        max_concurrent_parses=3,
        backend_class_paths_by_format={
            "txt": "tests.file_manager.file_parser.conftest.StubTxtBackendA",
        },
    )

    out = fm.ingest_files(str(p), config=cfg)
    item = out[str(p)]
    assert isinstance(item, IngestedFullFile)
    assert item.status == "success"
    assert item.trace is not None
    assert item.trace.backend == "stub_txt_a"
    assert "stub_txt_a" in (item.summary or "")


@_handle_project
def test_file_manager_can_hotswap_backend_overrides_between_calls(
    rootless_file_manager,
    tmp_path: Path,
):
    fm = rootless_file_manager
    p = tmp_path / "doc.txt"
    p.write_text("hello", encoding="utf-8")

    cfg_a = FilePipelineConfig()
    cfg_a.output.return_mode = "full"
    cfg_a.embed.strategy = "off"
    cfg_a.parse = ParseConfig(
        max_concurrent_parses=1,
        backend_class_paths_by_format={
            "txt": "tests.file_manager.file_parser.conftest.StubTxtBackendA",
        },
    )

    cfg_b = FilePipelineConfig()
    cfg_b.output.return_mode = "full"
    cfg_b.embed.strategy = "off"
    cfg_b.parse = ParseConfig(
        max_concurrent_parses=1,
        backend_class_paths_by_format={
            "txt": "tests.file_manager.file_parser.conftest.StubTxtBackendB",
        },
    )

    out_a = fm.ingest_files(str(p), config=cfg_a)
    out_b = fm.ingest_files(str(p), config=cfg_b)

    a = out_a[str(p)]
    b = out_b[str(p)]
    assert (
        isinstance(a, IngestedFullFile)
        and a.trace is not None
        and a.trace.backend == "stub_txt_a"
    )
    assert (
        isinstance(b, IngestedFullFile)
        and b.trace is not None
        and b.trace.backend == "stub_txt_b"
    )


@_handle_project
def test_file_manager_parse_config_can_route_multiple_formats_to_different_backends_in_one_run(
    rootless_file_manager,
    tmp_path: Path,
):
    fm = rootless_file_manager
    txt = tmp_path / "doc.txt"
    pdf = tmp_path / "doc.pdf"
    txt.write_text("hello", encoding="utf-8")
    pdf.write_bytes(b"%PDF-1.4 fake")

    cfg = FilePipelineConfig()
    cfg.output.return_mode = "full"
    cfg.embed.strategy = "off"
    cfg.parse = ParseConfig(
        max_concurrent_parses=4,
        backend_class_paths_by_format={
            "txt": "tests.file_manager.file_parser.conftest.StubTxtBackendA",
            "pdf": "tests.file_manager.file_parser.conftest.StubPdfBackend",
        },
    )

    out = fm.ingest_files([str(txt), str(pdf)], config=cfg)
    a = out[str(txt)]
    b = out[str(pdf)]
    assert (
        isinstance(a, IngestedFullFile)
        and a.trace is not None
        and a.trace.backend == "stub_txt_a"
    )
    assert (
        isinstance(b, IngestedFullFile)
        and b.trace is not None
        and b.trace.backend == "stub_pdf"
    )
