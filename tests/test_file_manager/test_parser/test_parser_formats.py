"""
Comprehensive tests for parsing different file formats with DoclingParser.
Tests all supported formats (.txt, .pdf, .docx) with rigorous validation.
"""

from __future__ import annotations

import time

import pytest

from tests.helpers import _handle_project


@pytest.mark.unit
@_handle_project
def test_parse_txt_simple(parser, supported_format_files):
    """Test parsing simple text file."""
    txt_files = supported_format_files[".txt"]["files"]
    txt_file = txt_files["simple"]
    doc = parser.parse(txt_file)

    # Check metadata
    assert doc.metadata.file_type == "text/plain"
    assert doc.metadata.file_name.endswith(".txt")

    # Check content is preserved
    full_text = doc.to_plain_text()
    assert "simple text file" in full_text.lower()

    # Should have basic structure
    assert len(doc.sections) >= 0  # May have 0 or more sections
    assert doc.processing_status == "completed"


@pytest.mark.unit
@_handle_project
def test_parse_txt_multi_paragraph(parser, supported_format_files):
    """Test parsing multi-paragraph text file."""
    txt_files = supported_format_files[".txt"]["files"]
    txt_file = txt_files["multi_paragraph"]
    doc = parser.parse(txt_file)

    # Check metadata
    assert doc.metadata.file_type == "text/plain"

    # Check content preservation
    full_text = doc.to_plain_text()
    assert "First paragraph" in full_text
    assert "Second paragraph" in full_text
    assert "Third paragraph" in full_text

    # Should have structure
    assert len(doc.sections) >= 1
    assert doc.processing_status == "completed"


@pytest.mark.unit
@_handle_project
def test_parse_txt_special_characters(parser, supported_format_files):
    """Test parsing text file with special characters."""
    txt_files = supported_format_files[".txt"]["files"]
    txt_file = txt_files["special_chars"]
    doc = parser.parse(txt_file)

    # Check metadata
    assert doc.metadata.file_type == "text/plain"

    # Check Unicode handling
    full_text = doc.to_plain_text()
    # Check for at least some special characters from our fixture
    has_special_chars = any(
        char in full_text for char in ["café", "naïve", "€", "你好"]
    )
    assert has_special_chars, f"Expected special characters in: {full_text}"

    assert doc.processing_status == "completed"


@pytest.mark.unit
@_handle_project
def test_parse_pdf_file(parser):
    """Test parsing PDF file from sample directory."""
    from pathlib import Path

    # Use the actual PDF file from sample directory
    sample_dir = Path(__file__).parent.parent / "sample"
    pdf_file = sample_dir / "IT_Department_Policy_Document.pdf"

    if not pdf_file.exists():
        pytest.skip("PDF sample file not found")

    doc = parser.parse(pdf_file)

    # Check metadata
    assert doc.metadata.file_type == "application/pdf"
    assert doc.metadata.file_name == "IT_Department_Policy_Document.pdf"
    assert doc.metadata.file_size > 0

    # Check content extraction
    full_text = doc.to_plain_text()
    assert len(full_text.strip()) > 0, "PDF should contain extractable text"

    # Should have some structure (PDFs typically have multiple sections)
    assert len(doc.sections) >= 1
    assert doc.processing_status == "completed"


@pytest.mark.unit
@_handle_project
def test_parse_docx_file(parser):
    """Test parsing DOCX file from sample directory."""
    from pathlib import Path

    # Use the actual DOCX file from sample directory
    sample_dir = Path(__file__).parent.parent / "sample"
    docx_file = sample_dir / "SmartHome_Hub_X200_Technical_Documentation.docx"

    if not docx_file.exists():
        pytest.skip("DOCX sample file not found")

    doc = parser.parse(docx_file)

    # Check metadata
    assert (
        doc.metadata.file_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert doc.metadata.file_name == "SmartHome_Hub_X200_Technical_Documentation.docx"
    assert doc.metadata.file_size > 0

    # Check content extraction
    full_text = doc.to_plain_text()
    assert len(full_text.strip()) > 0, "DOCX should contain extractable text"

    # DOCX files typically have good structure
    assert len(doc.sections) >= 1
    assert doc.processing_status == "completed"


@pytest.mark.unit
@_handle_project
def test_parse_empty_txt_file(parser, supported_format_files):
    """Test parsing an empty text file."""
    txt_files = supported_format_files[".txt"]["files"]
    empty_file = txt_files["empty"]
    doc = parser.parse(empty_file)

    # Check metadata
    assert doc.metadata.file_type == "text/plain"
    assert doc.processing_status == "completed"

    # Empty file should have empty content
    full_text = doc.to_plain_text()
    assert full_text.strip() == ""

    # May have 0 sections for empty files
    assert len(doc.sections) >= 0


@pytest.mark.unit
@_handle_project
def test_parse_formats_comprehensive(parser):
    """Test parsing all supported formats comprehensively."""
    from pathlib import Path

    # Test that parser only supports the expected formats
    supported_formats = parser.supported_formats
    expected_formats = {".txt", ".pdf", ".docx"}

    # Verify all expected formats are supported
    for fmt in expected_formats:
        assert fmt in supported_formats, f"Expected format {fmt} to be supported"

    # Test each supported format with sample files
    sample_dir = Path(__file__).parent.parent / "sample"

    # Test PDF if available
    pdf_file = sample_dir / "IT_Department_Policy_Document.pdf"
    if pdf_file.exists():
        pdf_doc = parser.parse(pdf_file)
        assert pdf_doc.metadata.file_type == "application/pdf"
        assert pdf_doc.processing_status == "completed"
        assert len(pdf_doc.to_plain_text().strip()) > 0

    # Test DOCX if available
    docx_file = sample_dir / "SmartHome_Hub_X200_Technical_Documentation.docx"
    if docx_file.exists():
        docx_doc = parser.parse(docx_file)
        assert (
            docx_doc.metadata.file_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert docx_doc.processing_status == "completed"
        assert len(docx_doc.to_plain_text().strip()) > 0


@pytest.mark.unit
@_handle_project
def test_parse_binary_format_metadata(parser):
    """Test that binary formats (PDF, DOCX) have correct metadata."""
    from pathlib import Path

    sample_dir = Path(__file__).parent.parent / "sample"

    # Test PDF metadata
    pdf_file = sample_dir / "IT_Department_Policy_Document.pdf"
    if pdf_file.exists():
        pdf_doc = parser.parse(pdf_file)

        # Check all required metadata fields
        assert pdf_doc.metadata.file_name == "IT_Department_Policy_Document.pdf"
        assert pdf_doc.metadata.file_type == "application/pdf"
        assert pdf_doc.metadata.file_size > 0
        assert pdf_doc.metadata.parser_name == "DoclingParser"
        assert pdf_doc.metadata.parser_version == "1.0.0"
        assert pdf_doc.metadata.created_at is not None
        assert pdf_doc.metadata.processed_at is not None

    # Test DOCX metadata
    docx_file = sample_dir / "SmartHome_Hub_X200_Technical_Documentation.docx"
    if docx_file.exists():
        docx_doc = parser.parse(docx_file)

        # Check all required metadata fields
        assert (
            docx_doc.metadata.file_name
            == "SmartHome_Hub_X200_Technical_Documentation.docx"
        )
        assert (
            docx_doc.metadata.file_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert docx_doc.metadata.file_size > 0
        assert docx_doc.metadata.parser_name == "DoclingParser"
        assert docx_doc.metadata.parser_version == "1.0.0"
        assert docx_doc.metadata.created_at is not None
        assert docx_doc.metadata.processed_at is not None


# =============================================================================
# COMPREHENSIVE DYNAMIC FORMAT TESTS
# =============================================================================


@pytest.mark.unit
@_handle_project
def test_all_supported_formats_dynamic(
    parser,
    supported_format_files,
    parser_validation_suite,
):
    """Test dynamic parsing of all supported formats with rigorous validation."""
    validation = parser_validation_suite

    for fmt, format_info in supported_format_files.items():
        print(f"\nTesting format: {fmt}")

        # Test the simple variant for each format
        if format_info.get("is_binary", False):
            # For binary formats, use sample files
            sample_files = [
                f
                for f in format_info["files"].values()
                if f.name.startswith("file_example")
            ]
            if sample_files:
                file_path = sample_files[0]  # Use first sample file
                print(f"Testing binary file: {file_path}")

                doc = parser.parse(file_path)

                # Validate basic structure
                validation["validate_metadata"](doc, format_info["mime_type"])
                validation["validate_structure"](
                    doc,
                    format_info["structure_expectations"],
                )

                # For binary files, just ensure we got some content
                assert (
                    doc.to_plain_text().strip()
                ), f"Binary file {fmt} should extract some text content"
        else:
            # For text formats, test the simple variant
            simple_file = format_info["files"]["simple"]
            print(f"Testing text file: {simple_file}")

            doc = parser.parse(simple_file)

            # Validate metadata
            validation["validate_metadata"](doc, format_info["mime_type"])

            # Validate structure meets expectations
            validation["validate_structure"](doc, format_info["structure_expectations"])

            # Validate content preservation
            if format_info["validation_patterns"]:
                validation["validate_content"](doc, format_info["validation_patterns"])


@pytest.mark.unit
@_handle_project
def test_empty_files_all_formats(parser, supported_format_files):
    """Test handling of empty files across all supported formats."""

    for fmt, format_info in supported_format_files.items():
        if format_info.get("is_binary", False):
            continue  # Skip binary formats - can't have empty binary files

        if "empty" not in format_info["files"]:
            continue

        print(f"\nTesting empty file for format: {fmt}")
        empty_file = format_info["files"]["empty"]

        doc = parser.parse(empty_file)

        # Should still create valid document
        assert doc.document_id is not None
        assert doc.metadata is not None
        assert doc.processing_status == "completed"

        # Content should be empty
        assert doc.to_plain_text().strip() == ""
        assert len(doc.sections) == 0


@pytest.mark.unit
@_handle_project
def test_flat_records_all_formats(
    parser,
    supported_format_files,
    parser_validation_suite,
):
    """Test flat record conversion for all supported formats."""
    validation = parser_validation_suite

    for fmt, format_info in supported_format_files.items():
        print(f"\nTesting flat records for format: {fmt}")

        # Use sample file for binary, simple file for text
        if format_info.get("is_binary", False):
            sample_files = [
                f
                for f in format_info["files"].values()
                if f.name.startswith("file_example")
            ]
            if not sample_files:
                continue
            file_path = sample_files[0]
        else:
            file_path = format_info["files"]["simple"]

        doc = parser.parse(file_path)
        records = doc.to_flat_records()

        # Validate record structure
        validation["validate_records"](records)

        # Check format-specific record expectations
        expectations = format_info["structure_expectations"]
        doc_records = [r for r in records if r["content_type"] == "document"]
        section_records = [r for r in records if r["content_type"] == "section"]
        para_records = [r for r in records if r["content_type"] == "paragraph"]

        assert len(doc_records) == 1
        if expectations["min_sections"] > 0:
            assert len(section_records) >= expectations["min_sections"]
        if expectations["min_paragraphs"] > 0:
            assert len(para_records) >= expectations["min_paragraphs"]


@pytest.mark.unit
@_handle_project
def test_content_preservation_across_formats(parser, supported_format_files):
    """Test that content is properly preserved across different formats."""

    for fmt, format_info in supported_format_files.items():
        if format_info.get("is_binary", False):
            continue  # Binary file content depends on actual file content

        print(f"\nTesting content preservation for: {fmt}")

        # Test with different variants and their expected patterns
        for variant in ["simple", "multi_paragraph"]:
            if variant not in format_info["files"]:
                continue

            test_file = format_info["files"][variant]
            doc = parser.parse(test_file)
            full_text = doc.to_plain_text()

            # Should have substantial content
            assert (
                len(full_text.strip()) > 10
            ), f"Format {fmt} variant {variant} should extract substantial content"

            # Validate patterns based on the specific variant
            if variant == "simple" and format_info["validation_patterns"]:
                # For simple variant, check the validation patterns
                for pattern in format_info["validation_patterns"]:
                    assert (
                        pattern.lower() in full_text.lower()
                    ), f"Pattern '{pattern}' not preserved in {fmt} {variant}"
            elif variant == "multi_paragraph":
                # For multi-paragraph, check for paragraph structure
                assert (
                    "paragraph" in full_text.lower()
                ), f"Multi-paragraph variant should contain 'paragraph' in {fmt}"
                assert (
                    len(full_text.split("\n\n")) >= 2
                ), f"Multi-paragraph should have multiple paragraphs in {fmt}"


@pytest.mark.unit
@_handle_project
def test_metadata_consistency_across_formats(parser, supported_format_files):
    """Test that metadata is consistently set across all formats."""

    for fmt, format_info in supported_format_files.items():
        print(f"\nTesting metadata for format: {fmt}")

        # Use appropriate file for testing
        if format_info.get("is_binary", False):
            sample_files = [
                f
                for f in format_info["files"].values()
                if f.name.startswith("file_example")
            ]
            if not sample_files:
                continue
            file_path = sample_files[0]
        else:
            file_path = format_info["files"]["simple"]

        doc = parser.parse(file_path)

        # Check required metadata fields
        assert doc.metadata.file_type == format_info["mime_type"]
        assert doc.metadata.parser_name == "DoclingParser"
        assert doc.metadata.parser_version is not None

        # Check file-specific metadata
        assert doc.metadata.file_name == file_path.name
        assert doc.metadata.file_size is not None
        assert doc.metadata.file_size >= 0

        # Check processing metadata
        assert doc.processing_status == "completed"
        assert doc.error_message is None


@pytest.mark.unit
@_handle_project
def test_large_files_performance(
    parser,
    supported_format_files,
    performance_benchmarks,
):
    """Test performance with larger files where available."""
    benchmarks = performance_benchmarks

    for fmt, format_info in supported_format_files.items():
        if format_info.get("is_binary", False):
            continue  # Binary file size is fixed

        # Test with large variant if available
        if "large" not in format_info["files"]:
            continue

        print(f"\nTesting large file performance for: {fmt}")
        large_file = format_info["files"]["large"]

        # Get file size to determine benchmark
        file_size = large_file.stat().st_size
        if file_size < 1024:  # < 1KB
            max_time = benchmarks["max_parse_time"]["small_file"]
        elif file_size < 100 * 1024:  # < 100KB
            max_time = benchmarks["max_parse_time"]["medium_file"]
        else:
            max_time = benchmarks["max_parse_time"]["large_file"]

        # Time the parsing
        start_time = time.time()
        doc = parser.parse(large_file)
        parse_time = time.time() - start_time

        # Validate performance
        assert (
            parse_time <= max_time
        ), f"Parsing {fmt} took {parse_time:.2f}s, max allowed: {max_time}s"

        # Validate parsing succeeded
        assert doc.processing_status == "completed"
        assert len(doc.to_plain_text().strip()) > 100  # Should have substantial content
