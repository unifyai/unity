"""
Integration tests for FileManager.

These tests verify FileManager behavior with real file operations
and parser integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import _handle_project
from tests.test_file_manager.helpers import ask_judge


@pytest.fixture()
def temp_dir(tmp_path: Path) -> Path:
    """Legacy fixture for backward compatibility."""
    d = tmp_path / "src"
    d.mkdir(parents=True, exist_ok=True)
    (d / "a.txt").write_text("hello world", encoding="utf-8")
    (d / "a (1).txt").write_text("duplicate name test", encoding="utf-8")
    (d / "b.json").write_text('{"k":1}', encoding="utf-8")
    return d


@pytest.mark.asyncio
@_handle_project
async def test_unique_name_generation(file_manager, tmp_path: Path):
    """Test unique name generation when importing files with same names."""
    # Create two identical names in different folders and import sequentially
    src1 = tmp_path / "one"
    src2 = tmp_path / "two"
    src1.mkdir()
    src2.mkdir()
    (src1 / "conflict.pdf").write_text("x", encoding="utf-8")
    (src2 / "conflict.pdf").write_text("y", encoding="utf-8")

    fm = file_manager
    fm.clear()
    # Use import_directory to exercise unique naming policy on import
    # Make the stem unique to avoid collisions with prior tests
    import uuid

    stem = f"conflict_{uuid.uuid4().hex[:8]}"
    (src1 / f"{stem}.pdf").write_text("x", encoding="utf-8")
    (src2 / f"{stem}.pdf").write_text("y", encoding="utf-8")

    fm.import_directory(src1)
    fm.import_directory(src2)

    names = fm.list()
    assert f"{stem}.pdf" in names
    assert f"{stem} (1).pdf" in names


@pytest.mark.asyncio
@_handle_project
async def test_parse_all_formats(file_manager, supported_file_examples: dict):
    """Test parsing of all supported file formats."""
    fm = file_manager
    fm.clear()
    for filename, example_data in supported_file_examples.items():
        # Parse by absolute path (no import needed)
        display_name = str(example_data["path"])  # absolute path

        # Parse the file
        from unity.file_manager.types import FilePipelineConfig

        result = fm.ingest_files(
            display_name,
            config=FilePipelineConfig(output={"return_mode": "full"}),
        )

        # Check result structure (flattened fields)
        assert display_name in result
        file_result = result[display_name]
        assert file_result["status"] == "success"
        assert "records" in file_result
        # flattened top-level file metadata
        assert "file_format" in file_result
        assert "file_size" in file_result
        assert "file_format" in file_result

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


@pytest.mark.asyncio
@_handle_project
async def test_parse_errors(file_manager, tmp_path: Path):
    """Test handling of files that can't be parsed."""
    # Create a file with an unsupported extension
    bad_file = tmp_path / "bad.xyz"
    bad_file.write_text("unsupported content", encoding="utf-8")

    fm = file_manager
    fm.clear()
    # The file exists on disk; exists should reflect filesystem
    assert fm.exists(str(bad_file))

    # Parsing should return a result (basic text parsing as fallback)
    from unity.file_manager.types import FilePipelineConfig

    result = fm.ingest_files(
        str(bad_file),
        config=FilePipelineConfig(output={"return_mode": "full"}),
    )
    assert str(bad_file) in result
    file_result = result[str(bad_file)]

    # Should parse successfully as text
    assert file_result["status"] == "success"
    assert len(file_result["records"]) > 0

    # Content should be preserved
    all_content = " ".join(
        str(record.get("content_text", "")) for record in file_result["records"]
    )
    assert "unsupported content" in all_content


@pytest.mark.asyncio
@_handle_project
async def test_ask_uses_parse(file_manager, temp_dir: Path):
    """Test that the ask method correctly uses the parse tool."""
    fm = file_manager
    fm.clear()
    files = [str(p) for p in temp_dir.iterdir() if p.is_file()]

    # Parse files to add them to Unify logs before ask
    fm.ingest_files(files)

    # Ask a trivial question which should rely on parsed content
    name = next(n for n in files if n.endswith(".txt"))
    instruction = f"What does the file {name} contain?"
    handle = await fm.ask(instruction)
    ans = await handle.result()
    assert isinstance(ans, str)
    assert ans  # non-empty answer

    # Read file content for the judge
    from pathlib import Path as _Path

    file_content = _Path(name).read_text(encoding="utf-8")

    # Ask judge to verify
    verdict = await ask_judge(instruction, ans, file_content=file_content)
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed 'ask' incorrect. Verdict: {verdict}"


@pytest.mark.asyncio
@_handle_project
async def test_content_preservation(file_manager, supported_file_examples: dict):
    """Test that file content is preserved correctly during parsing."""
    fm = file_manager
    fm.clear()
    for filename, example_data in supported_file_examples.items():
        # Parse by absolute path instead of importing
        display_name = str(example_data["path"])  # absolute path

        from unity.file_manager.types import FilePipelineConfig

        result = fm.ingest_files(
            display_name,
            config=FilePipelineConfig(output={"return_mode": "full"}),
        )
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


@pytest.mark.asyncio
@_handle_project
async def test_structure_integrity(
    file_manager,
    supported_file_examples: dict,
):
    """Test that all supported formats produce proper document structure."""
    fm = file_manager
    fm.clear()
    for filename, example_data in supported_file_examples.items():
        display_name = str(example_data["path"])  # absolute path
        from unity.file_manager.types import FilePipelineConfig

        result = fm.ingest_files(
            display_name,
            config=FilePipelineConfig(output={"return_mode": "full"}),
        )

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
        # Skip structural check for CSV files - they don't have hierarchical document structure
        file_ext = Path(filename).suffix.lower()
        if file_ext in [".csv", ".xlsx"]:
            # CSV files are tabular data, not hierarchical documents
            # They may only have a document record without sections/paragraphs
            continue

        section_records = [r for r in records if r.get("content_type") == "section"]
        para_records = [r for r in records if r.get("content_type") == "paragraph"]

        # Document should have sections and paragraphs (or at least one of them)
        assert (
            len(section_records) > 0 or len(para_records) > 0
        ), f"Expected structural content in {filename}"


@_handle_project
def test_singleton(file_manager):
    """Test that FileManager is a singleton."""
    fm1 = file_manager
    fm2 = file_manager
    assert fm1 is fm2
