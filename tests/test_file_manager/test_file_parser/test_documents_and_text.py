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
from unity.file_manager.file_parsers.types.enums import NodeKind


def _assert_non_empty(s: str, *, label: str) -> None:
    assert isinstance(s, str), f"{label} must be str"
    assert s.strip(), f"{label} must be non-empty"


@pytest.mark.parametrize(
    ("ext", "mime"),
    [
        (".txt", MimeType.TEXT_PLAIN),
        (".md", MimeType.TEXT_MARKDOWN),
    ],
)
@_handle_project
def test_text_like_formats_parse_via_text_backend(
    tmp_path: Path,
    ext: str,
    mime: MimeType,
):
    p = tmp_path / f"doc{ext}"
    if ext == ".md":
        p.write_text("# Heading\n\nParagraph one.\n\nParagraph two.", encoding="utf-8")
    else:
        p.write_text("Hello world. Second sentence.", encoding="utf-8")

    res = FileParser().parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
    )

    assert res.status == "success"
    assert res.trace is not None
    assert res.trace.backend == "text_backend"
    assert res.mime_type == mime

    _assert_non_empty(res.full_text, label="full_text")
    _assert_non_empty(res.summary, label="summary")

    assert res.metadata is not None
    _assert_non_empty(res.metadata.key_topics, label="metadata.key_topics")
    _assert_non_empty(res.metadata.content_tags, label="metadata.content_tags")

    assert res.graph is not None
    kinds = {n.kind for n in res.graph.nodes.values()}
    assert NodeKind.SECTION in kinds


@pytest.mark.parametrize(
    ("ext", "mime", "payload_text"),
    [
        (
            ".html",
            MimeType.TEXT_HTML,
            "<html><body><h1>Title</h1><p>First paragraph.</p></body></html>",
        ),
        (
            ".json",
            MimeType.APPLICATION_JSON,
            '{"title":"Example","items":[{"k":1},{"k":2}]}',
        ),
        (
            ".xml",
            MimeType.APPLICATION_XML,
            "<root><item id='1'>alpha</item><item id='2'>beta</item></root>",
        ),
    ],
)
@_handle_project
def test_html_xml_json_parse_via_base_document_backend_with_fallback(
    tmp_path: Path,
    ext: str,
    mime: MimeType,
    payload_text: str,
):
    p = tmp_path / f"doc{ext}"
    p.write_text(payload_text, encoding="utf-8")

    res = FileParser().parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
    )
    assert res.status == "success"
    assert res.trace is not None
    expected_backend = {
        ".html": "html_backend",
        ".json": "json_backend",
        ".xml": "xml_backend",
    }[ext]
    assert res.trace.backend == expected_backend
    assert res.mime_type == mime
    assert res.file_format in (FileFormat.HTML, FileFormat.XML, FileFormat.JSON)

    _assert_non_empty(res.full_text, label="full_text")
    _assert_non_empty(res.summary, label="summary")
    assert res.metadata is not None
    _assert_non_empty(res.metadata.key_topics, label="metadata.key_topics")
    _assert_non_empty(res.metadata.content_tags, label="metadata.content_tags")
    assert res.graph is not None


@pytest.mark.parametrize(
    ("content", "needles"),
    [
        (
            "First paragraph\n\nSecond paragraph\n\nThird paragraph",
            ["First paragraph", "Second paragraph", "Third paragraph"],
        ),
        ("Special: café naïve € 你好", ["café", "naïve", "€", "你好"]),
        ("", []),
    ],
)
@_handle_project
def test_txt_variants_multi_paragraph_special_chars_and_empty(
    tmp_path: Path,
    content: str,
    needles: list[str],
):
    p = tmp_path / "variants.txt"
    p.write_text(content, encoding="utf-8")

    res = FileParser().parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
    )

    assert res.status == "success"
    assert res.file_format == FileFormat.TXT
    assert res.mime_type == MimeType.TEXT_PLAIN

    assert res.trace is not None
    assert res.trace.backend == "text_backend"

    assert isinstance(res.full_text, str)
    for n in needles:
        assert n in res.full_text

    # Summary is always populated (falls back to logical_path for empty files).
    _assert_non_empty(res.summary, label="summary")

    assert res.metadata is not None
    _assert_non_empty(res.metadata.key_topics, label="metadata.key_topics")
    _assert_non_empty(res.metadata.content_tags, label="metadata.content_tags")


@_handle_project
def test_unknown_extension_is_parsed_as_best_effort_text(tmp_path: Path):
    p = tmp_path / "weird.xyz"
    p.write_text("unsupported content", encoding="utf-8")
    res = FileParser().parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
    )
    assert res.status == "success"
    assert res.file_format == FileFormat.TXT
    assert res.mime_type == MimeType.TEXT_PLAIN
    assert res.trace is not None
    assert res.trace.backend == "text_backend"
    _assert_non_empty(res.full_text, label="full_text")


@_handle_project
def test_unknown_extension_respects_provided_mime_type(tmp_path: Path):
    """
    If callers provide a mime_type explicitly, FileParser should NOT override it even when the
    extension is unknown (it will still treat the format as best-effort TXT).
    """
    p = tmp_path / "weird.unknownext"
    p.write_text("hello", encoding="utf-8")
    res = FileParser().parse(
        FileParseRequest(
            logical_path=str(p),
            source_local_path=str(p),
            mime_type=MimeType.APPLICATION_JSON,
        ),
    )
    assert res.status == "success"
    assert res.file_format == FileFormat.TXT
    assert res.mime_type == MimeType.APPLICATION_JSON


@pytest.mark.parametrize(
    ("fname", "expected_fmt", "expected_backend"),
    [
        ("IT_Department_Policy_Document.pdf", FileFormat.PDF, "pdf_backend"),
        (
            "SmartHome_Hub_X200_Technical_Documentation.docx",
            FileFormat.DOCX,
            "ms_word_backend",
        ),
    ],
)
@_handle_project
def test_document_formats_have_text_summary_metadata_and_structure(
    fname: str,
    expected_fmt: FileFormat,
    expected_backend: str,
    sample_file,
):
    p = sample_file(fname)
    logical = f"sample/{fname}"

    res = FileParser().parse(
        FileParseRequest(
            logical_path=logical,
            source_local_path=str(p),
        ),
    )

    assert res.status == "success"
    assert res.file_format == expected_fmt
    assert res.trace is not None
    assert res.trace.backend == expected_backend
    assert res.trace.logical_path == logical

    _assert_non_empty(res.full_text, label="full_text")
    _assert_non_empty(res.summary, label="summary")

    assert res.metadata is not None
    _assert_non_empty(res.metadata.key_topics, label="metadata.key_topics")
    _assert_non_empty(res.metadata.content_tags, label="metadata.content_tags")
    assert res.metadata.confidence_score is not None

    assert res.graph is not None
    kinds = {n.kind for n in res.graph.nodes.values()}
    assert NodeKind.SECTION in kinds
    assert NodeKind.PARAGRAPH in kinds


@_handle_project
def test_text_backend_populates_text_summary_metadata_and_graph(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_text(
        "TITLE\n\nFirst paragraph. Second sentence.\n\nSecond paragraph with more text.",
        encoding="utf-8",
    )

    res = FileParser().parse(
        FileParseRequest(logical_path=str(p), source_local_path=str(p)),
    )

    assert res.status == "success"
    assert res.file_format == FileFormat.TXT
    assert res.trace is not None
    assert res.trace.backend == "text_backend"

    _assert_non_empty(res.full_text, label="full_text")
    _assert_non_empty(res.summary, label="summary")

    assert res.metadata is not None
    _assert_non_empty(res.metadata.key_topics, label="metadata.key_topics")
    _assert_non_empty(res.metadata.content_tags, label="metadata.content_tags")
    assert res.metadata.confidence_score is not None

    assert res.graph is not None
    kinds = {n.kind for n in res.graph.nodes.values()}
    assert NodeKind.SECTION in kinds
    assert NodeKind.PARAGRAPH in kinds
    assert NodeKind.SENTENCE in kinds
