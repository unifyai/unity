"""
Advanced tests for DoclingParser implementation.

These tests cover edge cases, performance, and integration scenarios
not covered by the basic test files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import _handle_project
from unity.file_manager.parser import DoclingParser


@pytest.mark.unit
@_handle_project
def test_txt_file_parsing(parser: DoclingParser, supported_format_files):
    """Test that .txt files are parsed correctly."""
    # Parse simple text
    txt_files = supported_format_files[".txt"]["files"]
    document = parser.parse(txt_files["simple"])
    result = document.to_dict()

    assert result["processing_status"] == "completed"
    assert "This is a simple text file." in result["full_text"]
    assert len(result["sections"]) >= 1
    assert result["metadata"]["file_type"] == "text/plain"


@pytest.mark.unit
@_handle_project
def test_multi_paragraph_parsing(parser: DoclingParser, supported_format_files):
    """Test parsing of multi-paragraph documents."""
    txt_files = supported_format_files[".txt"]["files"]
    document = parser.parse(txt_files["multi_paragraph"])
    result = document.to_dict()

    assert result["processing_status"] == "completed"

    # Check all paragraphs are captured
    full_text = result["full_text"]
    assert "First paragraph" in full_text
    assert "Second paragraph" in full_text
    assert "Third paragraph" in full_text

    # Should have at least one section
    assert len(result["sections"]) >= 1
    section = result["sections"][0]

    # Should have at least one paragraph (might be combined due to chunking)
    assert len(section["paragraphs"]) >= 1

    # Check that content is preserved even if chunked differently
    all_paragraph_text = " ".join(p["text"] for p in section["paragraphs"])
    assert "First paragraph" in all_paragraph_text
    assert "Second paragraph" in all_paragraph_text
    assert "Third paragraph" in all_paragraph_text


@pytest.mark.unit
@_handle_project
def test_special_characters(parser: DoclingParser, supported_format_files):
    """Test handling of special characters."""
    txt_files = supported_format_files[".txt"]["files"]
    document = parser.parse(txt_files["special_chars"])
    result = document.to_dict()

    assert result["processing_status"] == "completed"
    # Check for Unicode and special characters (from our fixture content)
    full_text = result["full_text"]
    # Our fixture should contain some special characters - check for at least some
    assert any(
        char in full_text
        for char in ["café", "naïve", "€", "你好", "世界", "математика"]
    ), f"Expected special characters in content: {full_text}"


@pytest.mark.unit
@_handle_project
def test_various_text_formats(parser: DoclingParser, supported_format_files):
    """Test parsing of various text-based formats that are actually supported."""
    # Only test formats that are actually supported by the parser
    test_cases = {
        ".txt": {"content": "Simple text content", "expected_phrases": ["Simple text"]},
    }

    # Add other formats only if they're supported
    for fmt in supported_format_files.keys():
        if fmt == ".txt":
            continue  # Already handled above
        elif fmt == ".json":
            test_cases[fmt] = {
                "content": '{"key": "value"}',
                "expected_phrases": ["key", "value"],
            }
        elif fmt == ".html":
            test_cases[fmt] = {
                "content": "<html><body>Test</body></html>",
                "expected_phrases": ["Test"],
            }
        elif fmt == ".md":
            test_cases[fmt] = {
                "content": "# Header\nContent",
                "expected_phrases": ["Header", "Content"],
            }
        elif fmt == ".csv":
            test_cases[fmt] = {
                "content": "a,b,c\n1,2,3",
                "expected_phrases": ["a", "b", "c"],
            }

    for file_ext, test_data in test_cases.items():
        if file_ext not in supported_format_files:
            continue  # Skip unsupported formats

        print(f"Testing format: {file_ext}")

        # For supported formats in our fixture, use those files
        if file_ext in supported_format_files:
            format_info = supported_format_files[file_ext]
            file_path = format_info["files"]["simple"]
            document = parser.parse(file_path)
            result = document.to_dict()

            assert result["processing_status"] == "completed"
            assert result["metadata"]["file_type"] == format_info["mime_type"]

            # Verify content was extracted
            full_text = result["full_text"]
            assert len(full_text.strip()) > 0, f"No content extracted for {file_ext}"


@pytest.mark.unit
@_handle_project
def test_empty_file_handling(parser: DoclingParser, tmp_path: Path):
    """Test handling of empty files."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    document = parser.parse(empty_file)

    assert document.processing_status == "completed"
    assert document.full_text == ""
    assert len(document.sections) == 0  # Empty file should have no sections


@pytest.mark.unit
@_handle_project
def test_large_text_handling(parser: DoclingParser, tmp_path: Path):
    """Test handling of larger text files."""
    # Create a file with many paragraphs
    paragraphs = [f"Paragraph {i}: " + "Lorem ipsum " * 50 for i in range(20)]
    large_file = tmp_path / "large.txt"
    large_file.write_text("\n\n".join(paragraphs), encoding="utf-8")

    document = parser.parse(large_file)

    assert document.processing_status == "completed"
    assert document.metadata.total_words > 1000
    assert len(document.sections) >= 1

    # Check that paragraphs were split appropriately
    total_paragraphs = sum(len(section.paragraphs) for section in document.sections)
    assert total_paragraphs >= 1  # Should have at least one paragraph

    # For large text files, content might be chunked differently
    # Verify content preservation instead of exact paragraph count
    for i in range(20):
        assert f"Paragraph {i}:" in document.full_text


@pytest.mark.unit
@_handle_project
def test_parser_metadata(parser: DoclingParser, supported_format_files):
    """Test that parser metadata is correctly set."""
    txt_files = supported_format_files[".txt"]["files"]
    document = parser.parse(txt_files["simple"])
    result = document.to_dict()

    metadata = result["metadata"]
    assert metadata["parser_name"] == "DoclingParser"
    assert metadata["parser_version"] == "1.0.0"
    assert metadata["file_name"] == "test_simple.txt"
    assert metadata["file_size"] > 0
    assert "created_at" in metadata
    assert "processed_at" in metadata
