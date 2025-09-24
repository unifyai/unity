"""
FileManager parse functionality tests.
"""

from __future__ import annotations

from pathlib import Path
from unity.file_manager.file_manager import FileManager


def test_parse_single_file(supported_file_examples: dict):
    """Test parsing a single file."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    file_manager = FileManager()
    display_name = file_manager._add_file(example_data["path"])

    result = file_manager.parse(display_name)

    assert display_name in result
    assert result[display_name]["status"] == "success"
    assert "records" in result[display_name]
    assert isinstance(result[display_name]["records"], list)

    # Check metadata
    metadata = result[display_name].get("metadata", {})
    assert "file_type" in metadata
    assert "file_size" in metadata


def test_parse_multiple_files(supported_file_examples: dict):
    """Test parsing multiple files at once."""
    # Import all example files
    display_names = []
    file_manager = FileManager()
    for filename, example_data in supported_file_examples.items():
        display_name = file_manager._add_file(example_data["path"])
        display_names.append(display_name)

    # Parse all files
    results = file_manager.parse(display_names)

    assert len(results) == len(display_names)
    for display_name in display_names:
        assert display_name in results
        assert results[display_name]["status"] == "success"


def test_parse_with_options(supported_file_examples: dict):
    """Test parsing with custom options."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    file_manager = FileManager()
    display_name = file_manager._add_file(example_data["path"])

    # Parse with options (these are passed to the parser)
    result = file_manager.parse(
        display_name,
        max_chunk_size=100,
        chunk_overlap=20,
    )

    assert display_name in result
    assert result[display_name]["status"] == "success"


def test_parse_empty_file(sample_files: Path):
    """Test parsing an empty file."""
    # Import empty file (this fixture still creates empty.txt)
    empty_file = sample_files / "empty.txt"
    file_manager = FileManager()
    display_name = file_manager._add_file(empty_file)

    result = file_manager.parse(display_name)

    assert display_name in result
    # Empty file should still parse successfully
    assert result[display_name]["status"] == "success"
    # May have a document structure but content should be empty
    records = result[display_name]["records"]
    if records:
        # All content should be empty
        all_content = " ".join(
            str(record.get("content_text", "")) for record in records
        ).strip()
        assert (
            all_content == "empty"
        ), "Empty file with no content should fallback to using the title as content"


def test_parse_supported_formats(supported_file_examples: dict):
    """Test parsing files in all supported formats."""
    # Add all example files to the file manager
    display_names = []
    file_manager = FileManager()
    for filename, example_data in supported_file_examples.items():
        display_name = file_manager._add_file(example_data["path"])
        display_names.append(display_name)

    # Test parsing each file individually
    for display_name in display_names:
        result = file_manager.parse(display_name)

        assert display_name in result
        assert result[display_name]["status"] == "success"
        assert len(result[display_name]["records"]) > 0

        # Check that content is preserved
        records = result[display_name]["records"]
        all_content = " ".join(
            str(record.get("content_text", "")) for record in records
        )
        assert all_content.strip()  # Should have some content


def test_parse_multiple_supported_files(supported_file_examples: dict):
    """Test parsing multiple files in supported formats."""
    # Add all example files to the file manager
    display_names = []
    file_manager = FileManager()
    for filename, example_data in supported_file_examples.items():
        display_name = file_manager._add_file(example_data["path"])
        display_names.append(display_name)

    # Parse all files at once
    result = file_manager.parse(display_names)

    for display_name in display_names:
        assert display_name in result
        file_result = result[display_name]
        assert file_result["status"] == "success"
        assert len(file_result["records"]) > 0
