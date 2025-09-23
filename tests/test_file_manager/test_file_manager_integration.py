"""
Integration tests for FileManager.

These tests verify FileManager behavior with real file operations
and parser integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from unity.file_manager.file_manager import FileManager
from tests.helpers import _handle_project


@pytest.fixture()
def temp_dir(tmp_path: Path) -> Path:
    """Legacy fixture for backward compatibility."""
    d = tmp_path / "src"
    d.mkdir(parents=True, exist_ok=True)
    (d / "a.txt").write_text("hello world", encoding="utf-8")
    (d / "a (1).txt").write_text("duplicate name test", encoding="utf-8")
    (d / "b.json").write_text('{"k":1}', encoding="utf-8")
    return d


@_handle_project
@pytest.mark.unit
def test_unique_name_generation_on_conflict(tmp_path: Path):
    """Test unique name generation when importing files with same names."""
    # Create two identical names in different folders and import sequentially
    src1 = tmp_path / "one"
    src2 = tmp_path / "two"
    src1.mkdir()
    src2.mkdir()
    (src1 / "conflict.pdf").write_text("x", encoding="utf-8")
    (src2 / "conflict.pdf").write_text("y", encoding="utf-8")

    fm = FileManager()
    first = fm.import_directory(src1)
    second = fm.import_directory(src2)
    all_names = fm.list()
    assert "conflict.pdf" in all_names
    assert "conflict (1).pdf" in all_names
    assert len(set(all_names)) == len(all_names)


@_handle_project
@pytest.mark.unit
def test_parse_all_supported_formats(supported_file_examples: dict):
    """Test parsing of all supported file formats."""
    fm = FileManager()

    for filename, example_data in supported_file_examples.items():
        # Add the file to file manager
        display_name = fm._add_file(example_data["path"])

        # Parse the file
        result = fm.parse(display_name)

        # Check result structure
        assert display_name in result
        file_result = result[display_name]
        assert file_result["status"] == "success"
        assert "records" in file_result
        assert "metadata" in file_result

        # Check content was parsed - combine all record content
        all_content = " ".join(
            str(record.get("content_text", "")) for record in file_result["records"]
        )

        # Check that content was extracted
        assert (
            all_content.strip()
        ), f"Expected some content from {filename} but got empty"

        # For files with known expected phrases, verify them
        if example_data["expected_phrases"]:
            for expected_phrase in example_data["expected_phrases"]:
                assert (
                    expected_phrase.lower() in all_content.lower()
                ), f"Expected '{expected_phrase}' in {filename} content"

        # For sample files, just ensure we got substantial content
        if example_data.get("is_sample_file", False):
            assert (
                len(all_content.strip()) > 10
            ), f"Expected substantial content from sample file {filename}"


@_handle_project
@pytest.mark.unit
def test_parse_error_handling(tmp_path: Path):
    """Test handling of files that can't be parsed."""
    # Create a file with an unsupported extension
    bad_file = tmp_path / "bad.xyz"
    bad_file.write_text("unsupported content", encoding="utf-8")

    fm = FileManager()
    fm.import_directory(tmp_path)

    # Should still import the file
    assert fm.exists("bad.xyz")

    # Parsing should return a result (basic text parsing as fallback)
    result = fm.parse("bad.xyz")
    assert "bad.xyz" in result
    file_result = result["bad.xyz"]

    # Should parse successfully as text
    assert file_result["status"] == "success"
    assert len(file_result["records"]) > 0

    # Content should be preserved
    all_content = " ".join(
        str(record.get("content_text", "")) for record in file_result["records"]
    )
    assert "unsupported content" in all_content


@_handle_project
@pytest.mark.unit
@pytest.mark.asyncio
async def test_ask_tool_loop_uses_parse(temp_dir: Path):
    """Test that the ask method correctly uses the parse tool."""
    fm = FileManager()
    fm.import_directory(temp_dir)
    # Ask a trivial question which should rely on parse()
    name = next(n for n in fm.list() if n.endswith(".txt"))
    handle = await fm.ask(name, f"What does {name} contain?")
    ans = await handle.result()
    assert isinstance(ans, str)
    assert ans  # non-empty answer


@_handle_project
@pytest.mark.unit
def test_file_content_preservation(supported_file_examples: dict):
    """Test that file content is preserved correctly during parsing."""
    fm = FileManager()

    for filename, example_data in supported_file_examples.items():
        # Add the file to file manager
        display_name = fm._add_file(example_data["path"])

        result = fm.parse(display_name)
        assert display_name in result
        file_result = result[display_name]
        assert file_result["status"] == "success"

        # Combine all content from records
        all_content = " ".join(
            str(record.get("content_text", "")) for record in file_result["records"]
        )

        # Ensure content was preserved and extracted
        assert all_content.strip(), f"Expected content to be preserved in {filename}"

        # Test specific phrases for files with known content
        if example_data["expected_phrases"]:
            for phrase in example_data["expected_phrases"]:
                assert (
                    phrase.lower() in all_content.lower()
                ), f"Expected '{phrase}' in {filename}"

        # Ensure we have proper document structure
        assert len(file_result["records"]) > 0


@_handle_project
@pytest.mark.unit
def test_document_structure_integrity(supported_file_examples: dict):
    """Test that all supported formats produce proper document structure."""
    fm = FileManager()

    for filename, example_data in supported_file_examples.items():
        display_name = fm._add_file(example_data["path"])
        result = fm.parse(display_name)

        assert display_name in result
        file_result = result[display_name]
        assert file_result["status"] == "success"

        records = file_result["records"]
        assert len(records) > 0, f"Expected records for {filename}"

        # Check document structure integrity
        doc_records = [r for r in records if r.get("content_type") == "document"]
        assert (
            len(doc_records) == 1
        ), f"Should have exactly one document record for {filename}"

        # Verify hierarchical structure
        section_records = [r for r in records if r.get("content_type") == "section"]
        para_records = [r for r in records if r.get("content_type") == "paragraph"]

        # Document should have sections and paragraphs (or at least one of them)
        assert (
            len(section_records) > 0 or len(para_records) > 0
        ), f"Expected structural content in {filename}"


@_handle_project
@pytest.mark.unit
def test_file_manager_singleton():
    """Test that FileManager is a singleton."""
    fm1 = FileManager()
    fm2 = FileManager()
    assert fm1 is fm2
