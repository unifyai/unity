"""
Basic DoclingParser functionality tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import _handle_project
from unity.file_manager.parser import DoclingParser


@pytest.mark.unit
@_handle_project
def test_parser_initialization():
    """Test parser initializes with correct defaults."""
    parser = DoclingParser()

    assert parser.parser_name == "DoclingParser"
    assert parser.parser_version == "1.0.0"
    assert parser.max_chunk_size == 500
    assert parser.chunk_overlap == 200
    assert parser.sentence_chunk_size == 512
    assert parser.use_hybrid_chunking is False
    assert parser.extract_images is True
    assert parser.extract_tables is True
    assert parser.use_llm_enrichment is True


@pytest.mark.unit
@_handle_project
def test_parser_custom_initialization():
    """Test parser with custom parameters."""
    parser = DoclingParser(
        max_chunk_size=1000,
        chunk_overlap=100,
        sentence_chunk_size=256,
        use_hybrid_chunking=True,
        extract_images=False,
        extract_tables=False,
        use_llm_enrichment=False,
        parser_name="CustomParser",
        parser_version="2.0.0",
    )

    assert parser.parser_name == "CustomParser"
    assert parser.parser_version == "2.0.0"
    assert parser.max_chunk_size == 1000
    assert parser.chunk_overlap == 100
    assert parser.sentence_chunk_size == 256
    assert parser.use_hybrid_chunking is True
    assert parser.extract_images is False
    assert parser.extract_tables is False
    assert parser.use_llm_enrichment is False


@pytest.mark.unit
@_handle_project
def test_parse_simple_text(parser, supported_format_files):
    """Test parsing a simple text file."""
    txt_files = supported_format_files[".txt"]["files"]
    simple_file = txt_files["simple"]
    doc = parser.parse(simple_file)

    # Check document structure
    assert doc.document_id is not None
    assert doc.metadata is not None
    assert doc.metadata.file_name == "test_simple.txt"
    assert doc.metadata.file_type == "text/plain"
    assert doc.metadata.parser_name == "DoclingParser"

    # Check content
    assert len(doc.sections) >= 1
    section = doc.sections[0]
    assert len(section.paragraphs) >= 1

    # Verify text content
    full_text = doc.to_plain_text()
    assert "simple text file" in full_text


@pytest.mark.unit
@_handle_project
def test_parse_multi_paragraph(parser, supported_format_files):
    """Test parsing text with multiple paragraphs."""
    txt_files = supported_format_files[".txt"]["files"]
    multi_file = txt_files["multi_paragraph"]
    doc = parser.parse(multi_file)

    # Should have at least one section
    assert len(doc.sections) >= 1
    section = doc.sections[0]

    # Should have content (parser may combine paragraphs)
    assert len(section.paragraphs) >= 1
    # Check that all paragraph content is present in the section
    section_text = section.content_text or ""
    assert "First paragraph" in section_text
    assert "Second paragraph" in section_text
    assert "Third paragraph" in section_text

    # Check content preservation
    full_text = doc.to_plain_text()
    assert "First paragraph" in full_text
    assert "Second paragraph" in full_text
    assert "Third paragraph" in full_text


@pytest.mark.unit
@_handle_project
def test_parse_empty_file(parser, supported_format_files):
    """Test parsing an empty file."""
    txt_files = supported_format_files[".txt"]["files"]
    empty_file = txt_files["empty"]
    doc = parser.parse(empty_file)

    # Should still create a valid document
    assert doc.document_id is not None
    assert doc.metadata is not None

    # But with no content
    assert len(doc.sections) == 0 or (
        len(doc.sections) == 1 and len(doc.sections[0].paragraphs) == 0
    )

    # Plain text should be empty
    assert doc.to_plain_text().strip() == ""


@pytest.mark.unit
@_handle_project
def test_parse_special_characters(parser, supported_format_files):
    """Test parsing text with special characters."""
    txt_files = supported_format_files[".txt"]["files"]
    special_file = txt_files["special_chars"]
    doc = parser.parse(special_file)

    # Check special characters are preserved
    full_text = doc.to_plain_text()
    assert "café" in full_text
    assert "naïve" in full_text
    assert "€100" in full_text
    assert "你好" in full_text


@pytest.mark.unit
@_handle_project
def test_parse_nonexistent_file(parser):
    """Test parsing a non-existent file."""
    with pytest.raises(FileNotFoundError):
        parser.parse(Path("/nonexistent/file.txt"))


@pytest.mark.unit
@_handle_project
def test_flat_records_conversion(parser, supported_format_files):
    """Test conversion to flat records format."""
    txt_files = supported_format_files[".txt"]["files"]
    simple_file = txt_files["simple"]
    doc = parser.parse(simple_file)

    records = doc.to_flat_records()

    # Should have records
    assert len(records) > 0

    # Check record structure
    for record in records:
        assert "document_id" in record
        assert "content_type" in record
        assert "content_text" in record
        assert record["content_type"] in [
            "document",
            "section",
            "paragraph",
            "sentence",
        ]

    # Document record should exist
    doc_records = [r for r in records if r["content_type"] == "document"]
    assert len(doc_records) == 1

    # Content should be preserved
    all_content = " ".join(r.get("content_text", "") for r in records)
    assert "simple text file" in all_content
