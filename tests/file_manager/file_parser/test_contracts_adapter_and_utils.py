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
    assert csv is not None and csv.name == "csv_backend"
    assert xlsx is not None and xlsx.name == "ms_excel_backend"
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
def test_parse_batch_enforces_global_concurrency_cap_of_8(tmp_path: Path):
    """
    Deep unit test: regardless of requested ParseConfig.max_concurrent_parses,
    FileParser caps actual parallelism to 8.

    This test is deterministic: it measures max simultaneous backend.parse() calls using a blocking backend.
    """
    import threading

    # One real file is enough; backend doesn't need to read it.
    p = tmp_path / "one.txt"
    p.write_text("x", encoding="utf-8")

    lock = threading.Lock()
    active = 0
    max_active = 0
    reached = threading.Event()
    release = threading.Event()

    class BlockingBackend(BaseFileParserBackend):
        name = "blocking_backend"
        supported_formats = (FileFormat.TXT,)

        def can_handle(self, fmt: FileFormat | None) -> bool:
            return fmt == FileFormat.TXT

        def parse(self, ctx: FileParseRequest, /) -> FileParseResult:
            nonlocal active, max_active
            with lock:
                active += 1
                if active > max_active:
                    max_active = active
                if active >= 8:
                    reached.set()

            # Block until the test releases us (avoid hangs with a finite wait).
            release.wait(timeout=5.0)
            with lock:
                active -= 1

            # Return a minimal "success" result with empty fields so FileParser invariants don't trigger LLM calls.
            return FileParseResult(logical_path=str(ctx.logical_path), status="success")

    parser = FileParser(backends=[BlockingBackend()])
    reqs = [
        FileParseRequest(logical_path=f"logical/{i}.txt", source_local_path=str(p))
        for i in range(30)
    ]

    out_holder: dict[str, list[FileParseResult]] = {}

    def _run() -> None:
        out_holder["out"] = parser.parse_batch(
            reqs,
            raises_on_error=False,
            parse_config=ParseConfig(max_concurrent_parses=200),
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    assert reached.wait(
        timeout=5.0,
    ), "Expected FileParser.parse_batch to reach its hard cap concurrency of 8"
    release.set()
    t.join(timeout=10.0)
    assert (
        not t.is_alive()
    ), "parse_batch thread did not finish after releasing blocking backend"

    out = out_holder["out"]
    assert len(out) == len(reqs)
    assert max_active == 8


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (2, 2),
        (5, 5),
        (99, 8),
    ],
)
@_handle_project
def test_parse_batch_concurrency_is_derived_from_parse_config(
    tmp_path: Path,
    requested: int,
    expected: int,
):
    import threading

    p = tmp_path / "one.txt"
    p.write_text("x", encoding="utf-8")

    lock = threading.Lock()
    active = 0
    max_active = 0
    reached = threading.Event()
    release = threading.Event()

    class BlockingBackend(BaseFileParserBackend):
        name = "blocking_backend"
        supported_formats = (FileFormat.TXT,)

        def can_handle(self, fmt: FileFormat | None) -> bool:
            return fmt == FileFormat.TXT

        def parse(self, ctx: FileParseRequest, /) -> FileParseResult:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                if active >= expected:
                    reached.set()
            release.wait(timeout=5.0)
            with lock:
                active -= 1
            return FileParseResult(logical_path=str(ctx.logical_path), status="success")

    parser = FileParser(backends=[BlockingBackend()])
    reqs = [
        FileParseRequest(logical_path=f"logical/{i}.txt", source_local_path=str(p))
        for i in range(30)
    ]
    out_holder: dict[str, list[FileParseResult]] = {}

    def _run() -> None:
        out_holder["out"] = parser.parse_batch(
            reqs,
            raises_on_error=False,
            parse_config=ParseConfig(max_concurrent_parses=requested),
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    assert reached.wait(timeout=5.0)
    release.set()
    t.join(timeout=10.0)
    assert not t.is_alive()
    assert len(out_holder["out"]) == len(reqs)
    assert max_active == expected


@_handle_project
def test_parse_batch_defensive_path_handles_parse_raising(monkeypatch):
    """
    Cover the extremely defensive parse_batch branch where fut.result() raises.
    """
    parser = FileParser()

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    # parse_batch submits `_parse_single` into the executor; patch that to force `fut.result()` to raise.
    monkeypatch.setattr(parser, "_parse_single", boom)
    reqs = [
        FileParseRequest(logical_path="logical/a.txt", source_local_path="a.txt"),
        FileParseRequest(logical_path="logical/b.txt", source_local_path="b.txt"),
    ]
    out = parser.parse_batch(
        reqs,
        raises_on_error=False,
        parse_config=ParseConfig(max_concurrent_parses=8),
    )
    assert len(out) == 2
    assert out[0].status == "error"
    assert out[0].trace is not None
    assert out[0].trace.backend == "file_parser"
    assert any("parse_batch_exception" in w for w in (out[0].trace.warnings or []))


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
