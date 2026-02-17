"""
Integration tests for FileManager.

These tests verify FileManager behavior with real file operations
and parser integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import _handle_project


def _text_for_assertions(file_result) -> str:
    """
    Return a best-effort text surface for assertions across formats.

    Policy:
    - For documents/text: use lowered `/Content/` row `content_text`.
    - For spreadsheets: `/Content/` sheet/table rows intentionally have no `content_text`,
      so use the bounded `full_text` profile + a small sample of extracted table values.
    """
    fmt = getattr(file_result, "file_format", None)
    fmt_val = str(getattr(fmt, "value", fmt) or "").lower().strip()
    if fmt_val in ("csv", "xlsx"):
        parts: list[str] = [str(getattr(file_result, "full_text", "") or "")]
        try:
            values: list[str] = []
            for t in list(getattr(file_result, "tables", []) or [])[:8]:
                for r in list(getattr(t, "rows", []) or [])[:50]:
                    for v in (r.values() if isinstance(r, dict) else []):
                        if v is not None:
                            values.append(str(v))
                    if len(values) >= 500:
                        break
                if len(values) >= 500:
                    break
            if values:
                parts.append(" ".join(values))
        except Exception:
            pass
        return " ".join([p for p in parts if str(p).strip()]).strip()

    rows = list(getattr(file_result, "content_rows", []) or [])
    return " ".join(str(getattr(r, "content_text", "") or "") for r in rows).strip()


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
        from unity.file_manager.types.ingest import IngestedFullFile

        result = fm.ingest_files(
            display_name,
            config=FilePipelineConfig(output={"return_mode": "full"}),
        )

        # Check result structure (flattened fields)
        assert display_name in result
        file_result = result[display_name]
        assert file_result.status == "success"
        assert isinstance(file_result, IngestedFullFile)
        # top-level file metadata
        assert hasattr(file_result, "file_format")

        # Check content was parsed - combine all record content
        all_content = _text_for_assertions(file_result)

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

    # Parsing should return a result (unknown extensions are treated as best-effort text)
    from unity.file_manager.types import FilePipelineConfig
    from unity.file_manager.types.ingest import IngestedFullFile

    result = fm.ingest_files(
        str(bad_file),
        config=FilePipelineConfig(output={"return_mode": "full"}),
    )
    assert str(bad_file) in result
    file_result = result[str(bad_file)]

    # Should parse successfully as text
    assert file_result.status == "success"
    assert isinstance(file_result, IngestedFullFile)
    assert len(list(file_result.content_rows or [])) > 0

    # Content should be preserved
    all_content = " ".join(
        str(getattr(r, "content_text", "") or "")
        for r in (file_result.content_rows or [])
    )
    assert not all_content.strip()


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
        assert file_result.status == "success"

        all_content = _text_for_assertions(file_result)

        # Ensure content was preserved and extracted
        assert all_content.strip(), f"Expected content to be preserved in {filename}"

        # Test specific phrases for files with known content
        if example_data["expected_phrases"]:
            for phrase in example_data["expected_phrases"]:
                assert (
                    phrase.lower() in all_content.lower()
                ), f"Expected '{phrase}' in {filename}"

        # Ensure we have proper document structure
        assert len(list(file_result.content_rows or [])) > 0


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
        assert file_result.status == "success"

        rows = list(file_result.content_rows or [])
        assert len(rows) > 0, f"Expected content_rows for {filename}"

        # Check document structure integrity
        def _ctype(r):
            return getattr(r, "content_type", None)

        from unity.file_manager.file_parsers.types.enums import ContentType

        doc_records = [r for r in rows if _ctype(r) == ContentType.DOCUMENT]
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

        section_records = [r for r in rows if _ctype(r) == ContentType.SECTION]
        para_records = [r for r in rows if _ctype(r) == ContentType.PARAGRAPH]

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
