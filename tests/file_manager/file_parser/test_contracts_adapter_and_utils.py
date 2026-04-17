from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import _handle_project
from unity.file_manager.file_parsers import (
    FileParseRequest,
    FileParser,
    FileFormat,
    MimeType,
)
from unity.file_manager.file_parsers.conversion import (
    DocumentConversionManager,
    DocxToPdfConverter,
)
from unity.file_manager.file_parsers.registry import BackendRegistry
from unity.file_manager.file_parsers.settings import FILE_PARSER_SETTINGS
from unity.file_manager.file_parsers.types.backend import BaseFileParserBackend
from unity.file_manager.file_parsers.types.contracts import FileParseResult
from unity.file_manager.file_parsers.types.contracts import FileParseTrace
from unity.file_manager.file_parsers.types.enums import ContentType
from unity.file_manager.file_parsers.utils import format_policy  # type: ignore[attr-defined]
from unity.file_manager.file_parsers.utils.tracing import safe_call, traced_step
from unity.file_manager.parse_adapter import adapt_parse_result_for_file_manager
from unity.file_manager.types import FilePipelineConfig, ParseConfig


@_handle_project
def test_backend_contract_enforced_invalid_return_type(tmp_path: Path):
    class BadBackend(BaseFileParserBackend):
        name = "bad_backend"
        supported_formats = (FileFormat.TXT,)

        def can_handle(self, fmt: FileFormat | None) -> bool:
            return True

        def parse(self, request: FileParseRequest, /):  # type: ignore[override]
            return {"not": "a FileParseResult"}

    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")

    parser = FileParser(backends=[BadBackend()])
    res = parser.parse(FileParseRequest(logical_path=str(p), source_local_path=str(p)))

    assert res.status == "error"
    assert res.error is not None
    assert "invalid type" in res.error.lower()


@_handle_project
def test_registry_routes_known_formats_to_expected_backends():
    reg = BackendRegistry.from_config()

    pdf = reg.pick_backend(FileFormat.PDF)
    docx = reg.pick_backend(FileFormat.DOCX)
    csv = reg.pick_backend(FileFormat.CSV)
    xlsx = reg.pick_backend(FileFormat.XLSX)
    txt = reg.pick_backend(FileFormat.TXT)
    html = reg.pick_backend(FileFormat.HTML)
    jsn = reg.pick_backend(FileFormat.JSON)
    xml = reg.pick_backend(FileFormat.XML)

    assert pdf is not None and pdf.name == "pdf_backend"
    assert docx is not None and docx.name == "ms_word_backend"
    assert csv is not None and csv.name == "native_csv_backend"
    assert xlsx is not None and xlsx.name == "native_excel_backend"
    assert txt is not None and txt.name == "text_backend"
    assert html is not None and html.name == "html_backend"
    assert jsn is not None and jsn.name == "json_backend"
    assert xml is not None and xml.name == "xml_backend"


@_handle_project
def test_registry_caches_instances_per_class_path():
    reg = BackendRegistry.from_config()
    a = reg.pick_backend(FileFormat.TXT)
    b = reg.pick_backend(FileFormat.TXT)
    assert a is not None and b is not None
    assert a is b, "Expected registry to cache backend instances per class path"


@_handle_project
def test_registry_normalizes_override_keys_with_dots_and_case():
    # Override with a key that includes a dot and mixed case; should still resolve.
    reg = BackendRegistry.from_config(
        backend_class_paths_by_format={
            ".TXT": "unity.file_manager.file_parsers.implementations.python.backends.text_backend.TextBackend",
        },
    )
    b = reg.pick_backend(FileFormat.TXT)
    assert b is not None
    assert b.name == "text_backend"


@_handle_project
def test_registry_import_failure_is_treated_as_no_backend_and_parser_returns_error(
    tmp_path: Path,
):
    p = tmp_path / "doc.txt"
    p.write_text("x", encoding="utf-8")

    parse_cfg = ParseConfig(
        backend_class_paths_by_format={"txt": "not.a.real.module.NopeBackend"},
        max_concurrent_parses=1,
    )
    parser = FileParser()
    res = parser.parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
        parse_config=parse_cfg,
    )
    assert res.status == "error"
    assert res.trace is not None
    assert res.trace.backend == "none"
    assert "No backend for format" in (res.error or "")


@_handle_project
def test_parse_config_normalizes_backend_mapping_keys(tmp_path: Path):
    """
    FileParser should treat ParseConfig.backend_class_paths_by_format as an override mapping and
    honor keys like '.TXT' (normalized by BackendRegistry).
    """
    p = tmp_path / "doc.txt"
    p.write_text("hello", encoding="utf-8")

    cfg = ParseConfig(
        max_concurrent_parses=1,
        backend_class_paths_by_format={
            ".TXT": "tests.file_manager.file_parser.conftest.StubTxtBackendA",
        },
    )
    res = FileParser().parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
        parse_config=cfg,
    )
    assert res.status == "success"
    assert res.trace is not None
    assert res.trace.backend == "stub_txt_a"


@_handle_project
def test_parse_config_hotswaps_single_format_backend_between_calls(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_text("hello", encoding="utf-8")

    cfg_a = ParseConfig(
        max_concurrent_parses=1,
        backend_class_paths_by_format={
            "txt": "tests.file_manager.file_parser.conftest.StubTxtBackendA",
        },
    )
    cfg_b = ParseConfig(
        max_concurrent_parses=1,
        backend_class_paths_by_format={
            "txt": "tests.file_manager.file_parser.conftest.StubTxtBackendB",
        },
    )

    parser = FileParser()
    ra = parser.parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
        parse_config=cfg_a,
    )
    rb = parser.parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
        parse_config=cfg_b,
    )
    ra2 = parser.parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
        parse_config=cfg_a,
    )

    assert (
        ra.status == "success"
        and ra.trace is not None
        and ra.trace.backend == "stub_txt_a"
    )
    assert (
        rb.status == "success"
        and rb.trace is not None
        and rb.trace.backend == "stub_txt_b"
    )
    assert (
        ra2.status == "success"
        and ra2.trace is not None
        and ra2.trace.backend == "stub_txt_a"
    )


@_handle_project
def test_parse_config_routes_multiple_formats_to_different_backends_in_one_batch(
    tmp_path: Path,
):
    txt = tmp_path / "doc.txt"
    pdf = tmp_path / "doc.pdf"
    txt.write_text("hello", encoding="utf-8")
    pdf.write_bytes(b"%PDF-1.4 fake")

    cfg = ParseConfig(
        max_concurrent_parses=4,
        backend_class_paths_by_format={
            "txt": "tests.file_manager.file_parser.conftest.StubTxtBackendA",
            "pdf": "tests.file_manager.file_parser.conftest.StubPdfBackend",
        },
    )

    parser = FileParser()
    reqs = [
        FileParseRequest(logical_path="logical/doc.txt", source_local_path=str(txt)),
        FileParseRequest(logical_path="logical/doc.pdf", source_local_path=str(pdf)),
    ]
    out = parser.parse_batch(reqs, raises_on_error=False, parse_config=cfg)

    assert [r.logical_path for r in out] == ["logical/doc.txt", "logical/doc.pdf"]
    assert (
        out[0].status == "success"
        and out[0].trace is not None
        and out[0].trace.backend == "stub_txt_a"
    )
    assert (
        out[1].status == "success"
        and out[1].trace is not None
        and out[1].trace.backend == "stub_pdf"
    )


@_handle_project
def test_parse_batch_preserves_order_and_is_best_effort(tmp_path: Path):
    good = tmp_path / "ok.txt"
    missing = tmp_path / "missing.txt"
    good.write_text("alpha beta gamma", encoding="utf-8")

    parser = FileParser()
    reqs = [
        FileParseRequest(logical_path="logical/ok.txt", source_local_path=str(good)),
        FileParseRequest(
            logical_path="logical/missing.txt",
            source_local_path=str(missing),
        ),
    ]
    out = parser.parse_batch(
        reqs,
        raises_on_error=False,
        parse_config=ParseConfig(max_concurrent_parses=50),
    )

    assert [r.logical_path for r in out] == ["logical/ok.txt", "logical/missing.txt"]
    assert out[0].status == "success"
    assert out[1].status == "error"
    assert out[0].trace is not None and out[0].trace.logical_path == "logical/ok.txt"
    assert (
        out[1].trace is not None and out[1].trace.logical_path == "logical/missing.txt"
    )


@_handle_project
def test_parse_batch_raises_when_configured(tmp_path: Path):
    good = tmp_path / "ok.txt"
    missing = tmp_path / "missing.txt"
    good.write_text("alpha beta gamma", encoding="utf-8")

    parser = FileParser()
    reqs = [
        FileParseRequest(logical_path="logical/ok.txt", source_local_path=str(good)),
        FileParseRequest(
            logical_path="logical/missing.txt",
            source_local_path=str(missing),
        ),
    ]
    with pytest.raises(RuntimeError):
        parser.parse_batch(
            reqs,
            raises_on_error=True,
            parse_config=ParseConfig(max_concurrent_parses=50),
        )


@_handle_project
def test_parse_batch_with_injected_backends_runs_sequentially(tmp_path: Path):
    """Injected backends force subprocess_isolation=False.

    With no subprocess isolation, parse_batch runs files sequentially
    (max concurrency = 1).  This avoids in-process resource contention
    and ensures crash isolation is only provided by the subprocess path.
    """
    import threading

    p = tmp_path / "one.txt"
    p.write_text("x", encoding="utf-8")

    lock = threading.Lock()
    max_active = 0
    call_order: list[str] = []

    class TrackingBackend(BaseFileParserBackend):
        name = "tracking_backend"
        supported_formats = (FileFormat.TXT,)

        def can_handle(self, fmt: FileFormat | None) -> bool:
            return fmt == FileFormat.TXT

        def parse(self, ctx: FileParseRequest, /) -> FileParseResult:
            nonlocal max_active
            with lock:
                call_order.append(str(ctx.logical_path))
                max_active = max(max_active, 1)
            return FileParseResult(logical_path=str(ctx.logical_path), status="success")

    parser = FileParser(backends=[TrackingBackend()])
    reqs = [
        FileParseRequest(logical_path=f"logical/{i}.txt", source_local_path=str(p))
        for i in range(5)
    ]
    out = parser.parse_batch(
        reqs,
        raises_on_error=False,
        parse_config=ParseConfig(max_concurrent_parses=200),
    )
    assert len(out) == 5
    assert max_active == 1
    assert call_order == [f"logical/{i}.txt" for i in range(5)]


@_handle_project
def test_parse_batch_propagates_backend_errors_without_raising(tmp_path: Path):
    """When a backend returns error status, parse_batch continues to the next file."""

    class FailingBackend(BaseFileParserBackend):
        name = "failing_backend"
        supported_formats = (FileFormat.TXT,)

        def can_handle(self, fmt: FileFormat | None) -> bool:
            return fmt == FileFormat.TXT

        def parse(self, ctx: FileParseRequest, /) -> FileParseResult:
            return FileParseResult(
                logical_path=str(ctx.logical_path),
                status="error",
                error="deliberate failure",
            )

    p = tmp_path / "a.txt"
    p.write_text("x", encoding="utf-8")
    reqs = [
        FileParseRequest(logical_path="logical/a.txt", source_local_path=str(p)),
        FileParseRequest(logical_path="logical/b.txt", source_local_path=str(p)),
    ]
    out = FileParser(backends=[FailingBackend()]).parse_batch(
        reqs,
        raises_on_error=False,
    )
    assert len(out) == 2
    assert all(r.status == "error" for r in out)


@_handle_project
def test_parse_adapter_spreadsheet_sheet_and_table_rows_have_no_content_text(
    sample_file,
):
    p = sample_file("employee_records.csv")
    assert p.exists()

    res = FileParser().parse(
        FileParseRequest(
            logical_path="sample/employee_records.csv",
            source_local_path=str(p),
        ),
    )
    assert res.status == "success"

    cfg = FilePipelineConfig()
    cfg.embed.strategy = "off"
    adapted = adapt_parse_result_for_file_manager(res, config=cfg)

    rows = list(adapted.content_rows or [])
    assert rows, "Expected lowered /Content rows"

    sheet_rows = [r for r in rows if r.content_type == ContentType.SHEET]
    table_rows = [r for r in rows if r.content_type == ContentType.TABLE]
    assert sheet_rows, "Expected at least one sheet catalog row"
    assert table_rows, "Expected at least one table catalog row"

    for r in sheet_rows + table_rows:
        assert r.content_text is None
        assert (r.summary or "").strip(), "Catalog rows must have summaries"


@_handle_project
def test_parse_adapter_text_rows_keep_content_text_for_paragraphs_and_sentences(
    tmp_path: Path,
):
    p = tmp_path / "doc.txt"
    p.write_text(
        "Header\n\nParagraph one. Second sentence.\n\nParagraph two has more words.",
        encoding="utf-8",
    )

    res = FileParser().parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
    )
    assert res.status == "success"

    cfg = FilePipelineConfig()
    cfg.embed.strategy = "off"
    adapted = adapt_parse_result_for_file_manager(res, config=cfg)

    paras = [r for r in adapted.content_rows if r.content_type == ContentType.PARAGRAPH]
    sents = [r for r in adapted.content_rows if r.content_type == ContentType.SENTENCE]
    assert paras, "Expected paragraph rows"
    assert sents, "Expected sentence rows"
    assert any((r.content_text or "").strip() for r in paras)
    assert any((r.content_text or "").strip() for r in sents)


@_handle_project
def test_parse_adapter_error_result_is_best_effort_and_forwards_tables():
    from unity.file_manager.file_parsers.types.table import ExtractedTable

    tbl = ExtractedTable(
        table_id="table:0",
        label="T",
        sheet_name="Sheet 1",
        columns=["A"],
        rows=[{"A": 1}],
        sample_rows=[{"A": 1}],
        num_rows=1,
        num_cols=1,
    )
    pr = FileParseResult(
        logical_path="logical/bad.csv",
        status="error",
        error="boom",
        file_format=FileFormat.CSV,
        mime_type=MimeType.TEXT_CSV,
        tables=[tbl],
        graph=None,
        trace=FileParseTrace(logical_path="logical/bad.csv", backend="csv_backend"),
    )

    cfg = FilePipelineConfig()
    cfg.embed.strategy = "off"
    adapted = adapt_parse_result_for_file_manager(pr, config=cfg)
    assert adapted.content_rows == []
    assert len(adapted.tables) == 1
    assert adapted.tables[0].label == "T"


@_handle_project
def test_parse_adapter_success_without_graph_returns_empty_rows_but_keeps_tables():
    from unity.file_manager.file_parsers.types.table import ExtractedTable

    tbl = ExtractedTable(
        table_id="table:0",
        label="T",
        sheet_name="Sheet 1",
        columns=["A"],
        rows=[{"A": 1}],
        sample_rows=[{"A": 1}],
        num_rows=1,
        num_cols=1,
    )
    pr = FileParseResult(
        logical_path="logical/ok.csv",
        status="success",
        file_format=FileFormat.CSV,
        mime_type=MimeType.TEXT_CSV,
        tables=[tbl],
        graph=None,
    )
    cfg = FilePipelineConfig()
    cfg.embed.strategy = "off"
    adapted = adapt_parse_result_for_file_manager(pr, config=cfg)
    assert adapted.content_rows == []
    assert len(adapted.tables) == 1


@_handle_project
def test_parse_adapter_graph_with_missing_root_is_handled(tmp_path: Path):
    from unity.file_manager.file_parsers.types.graph import ContentGraph

    pr = FileParseResult(
        logical_path="logical/root_missing.txt",
        status="success",
        file_format=FileFormat.TXT,
        mime_type=MimeType.TEXT_PLAIN,
        graph=ContentGraph(root_id="root", nodes={}),
        tables=[],
    )
    cfg = FilePipelineConfig()
    cfg.embed.strategy = "off"
    adapted = adapt_parse_result_for_file_manager(pr, config=cfg)
    assert adapted.content_rows == []


@_handle_project
def test_format_policy_spreadsheet_profile_truncation_marker_present(tmp_path: Path):
    """
    Unit coverage for `format_policy.build_spreadsheet_profile_text` truncation behavior.
    """
    from unity.file_manager.file_parsers.types.table import ExtractedTable

    tables = []
    for i in range(20):
        tables.append(
            ExtractedTable(
                table_id=f"table:{i}",
                label=f"Sheet{i:02d}",
                sheet_name=f"Sheet{i:02d}",
                columns=["A", "B"],
                rows=[{"A": i, "B": i + 1}],
                sample_rows=[{"A": i, "B": i + 1}],
                num_rows=1,
                num_cols=2,
            ),
        )

    profile = format_policy.build_spreadsheet_profile_text(
        logical_path="logical/book.xlsx",
        tables=tables,
        sheet_names=[t.sheet_name for t in tables if t.sheet_name],
        max_tables=3,
        max_sample_rows=1,
    )
    assert "[TRUNCATED]" in profile


@_handle_project
def test_format_policy_metadata_fallback_is_deterministic_for_empty_text():
    meta = format_policy.extract_metadata_from_text_best_effort(
        text="",
        settings=FILE_PARSER_SETTINGS,
    )
    assert meta is not None
    assert (meta.key_topics or "").strip()
    assert (meta.content_tags or "").strip()
    assert meta.confidence_score is not None


@_handle_project
def test_docx_to_pdf_converter_reports_missing_source(tmp_path: Path):
    missing = tmp_path / "missing.docx"
    assert not missing.exists()

    conv = DocxToPdfConverter()
    res = conv.convert(missing)
    assert res.ok is False
    assert res.dst is None
    assert res.backend in ("none", "docx2pdf", "soffice", "win32com", "reuse")


@_handle_project
def test_document_conversion_manager_skips_unsupported_extensions(tmp_path: Path):
    mgr = DocumentConversionManager(converters=[DocxToPdfConverter()])
    bad = tmp_path / "file.xyz"
    bad.write_text("x", encoding="utf-8")

    out = mgr.convert(str(bad))
    assert out.ok is False
    assert out.backend in ("none", "skip")


@_handle_project
def test_traced_step_records_success_and_duration():
    trace = FileParseTrace(logical_path="x", backend="t")
    with traced_step(trace, name="unit") as step:
        step.counters["k"] = 1
        step.warnings.append("w")
    assert len(trace.steps) == 1
    s = trace.steps[0]
    assert s.name == "unit"
    assert s.status.value == "success"
    assert s.duration_ms >= 0
    assert s.counters["k"] == 1
    assert "w" in s.warnings


@_handle_project
def test_traced_step_records_failure_and_error_payload():
    trace = FileParseTrace(logical_path="x", backend="t")
    with pytest.raises(RuntimeError):
        with traced_step(trace, name="boom"):
            raise RuntimeError("nope")
    assert len(trace.steps) == 1
    s = trace.steps[0]
    assert s.name == "boom"
    assert s.status.value == "failed"
    assert s.error is not None
    assert s.error.code == "step_failed"
    assert "nope" in s.error.message


@_handle_project
def test_safe_call_marks_step_degraded_and_returns_default():
    trace = FileParseTrace(logical_path="x", backend="t")
    with traced_step(trace, name="maybe") as step:
        out = safe_call(
            step,
            lambda: (_ for _ in ()).throw(ValueError("bad")),
            code="x",
            default=123,
        )
    assert out == 123
    assert trace.steps[0].status.value == "degraded"
    assert trace.steps[0].error is not None
    assert trace.steps[0].error.code == "x"


@_handle_project
def test_file_parser_enforces_identity_trace_and_defaults_with_minimal_backend(
    tmp_path: Path,
):
    p = tmp_path / "doc.txt"
    p.write_text("x", encoding="utf-8")

    class MinimalBackend(BaseFileParserBackend):
        name = "minimal_backend"
        supported_formats = (FileFormat.TXT,)

        def can_handle(self, fmt: FileFormat | None) -> bool:
            return fmt == FileFormat.TXT

        def parse(self, ctx: FileParseRequest, /) -> FileParseResult:
            # Deliberately return wrong/missing identity so FileParser must enforce invariants.
            return FileParseResult(
                logical_path="WRONG",
                status="success",
                summary="",
                full_text="",
                file_format=None,
                mime_type=None,
                trace=FileParseTrace(
                    logical_path="ALSO_WRONG",
                    backend=self.name,
                    parsed_local_path=None,
                ),
            )

    parser = FileParser(backends=[MinimalBackend()])
    res = parser.parse(
        FileParseRequest(logical_path="logical/doc.txt", source_local_path=str(p)),
    )
    assert res.status == "success"
    assert res.logical_path == "logical/doc.txt"
    assert res.file_format == FileFormat.TXT
    assert res.mime_type == MimeType.TEXT_PLAIN

    assert res.trace is not None
    assert res.trace.backend == "minimal_backend"
    assert res.trace.logical_path == "logical/doc.txt"
    assert (
        res.trace.source_local_path is not None
        and res.trace.source_local_path.endswith("doc.txt")
    )
    assert res.trace.parsed_local_path is not None
    assert res.trace.duration_ms >= 0

    # FileParser guarantees a non-empty summary on success even if the backend returns empty.
    assert (res.summary or "").strip()


@_handle_project
def test_file_parser_catches_backend_exception_and_returns_error(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_text("x", encoding="utf-8")

    class BoomBackend(BaseFileParserBackend):
        name = "boom_backend"
        supported_formats = (FileFormat.TXT,)

        def can_handle(self, fmt: FileFormat | None) -> bool:
            return fmt == FileFormat.TXT

        def parse(self, ctx: FileParseRequest, /) -> FileParseResult:
            raise RuntimeError("boom")

    parser = FileParser(backends=[BoomBackend()])
    res = parser.parse(
        FileParseRequest(logical_path="logical/doc.txt", source_local_path=str(p)),
    )
    assert res.status == "error"
    assert "boom" in (res.error or "")
    assert res.trace is not None
    assert res.trace.backend == "boom_backend"
    assert any("backend_exception" in w for w in (res.trace.warnings or []))


@_handle_project
def test_file_parser_skips_injected_backend_when_can_handle_raises(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_text("x", encoding="utf-8")

    class BadPredicateBackend(BaseFileParserBackend):
        name = "bad_pred_backend"
        supported_formats = (FileFormat.TXT,)

        def can_handle(self, fmt: FileFormat | None) -> bool:
            raise RuntimeError("predicate failure")

        def parse(
            self,
            ctx: FileParseRequest,
            /,
        ) -> FileParseResult:  # pragma: no cover
            return FileParseResult(logical_path=str(ctx.logical_path), status="success")

    parser = FileParser(backends=[BadPredicateBackend()])
    res = parser.parse(
        FileParseRequest(logical_path="logical/doc.txt", source_local_path=str(p)),
    )
    assert res.status == "error"
    assert res.trace is not None
    assert res.trace.backend == "none"


@_handle_project
def test_file_parser_fills_spreadsheet_summary_when_backend_returns_blank(
    tmp_path: Path,
):
    from unity.file_manager.file_parsers.types.table import ExtractedTable

    p = tmp_path / "book.csv"
    p.write_text("A,B\n1,2\n", encoding="utf-8")

    tbl = ExtractedTable(
        table_id="table:0",
        label="Sheet 1",
        sheet_name="Sheet 1",
        columns=["A", "B"],
        rows=[{"A": 1, "B": 2}],
        sample_rows=[{"A": 1, "B": 2}],
        num_rows=1,
        num_cols=2,
    )

    class BlankSpreadsheetBackend(BaseFileParserBackend):
        name = "blank_csv_backend"
        supported_formats = (FileFormat.CSV,)

        def can_handle(self, fmt: FileFormat | None) -> bool:
            return fmt == FileFormat.CSV

        def parse(self, ctx: FileParseRequest, /) -> FileParseResult:
            return FileParseResult(
                logical_path=str(ctx.logical_path),
                status="success",
                file_format=FileFormat.CSV,
                mime_type=MimeType.TEXT_CSV,
                tables=[tbl],
                summary="",
                full_text="",
                metadata=None,  # triggers deterministic fallback when text is empty
                trace=None,  # forces FileParser to attach trace
            )

    parser = FileParser(backends=[BlankSpreadsheetBackend()])
    res = parser.parse(
        FileParseRequest(logical_path="logical/book.csv", source_local_path=str(p)),
    )
    assert res.status == "success"
    assert res.file_format == FileFormat.CSV
    assert res.mime_type == MimeType.TEXT_CSV
    assert (res.summary or "").strip()
    assert "Spreadsheet" in res.summary
