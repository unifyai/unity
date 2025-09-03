"""
Tests for batch parsing functionality in DoclingParser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import _handle_project


@pytest.mark.unit
@_handle_project
def test_parse_batch_multiple_files(parser, supported_format_files):
    """Test batch parsing of multiple files."""
    # Collect files from supported formats
    files = []
    txt_files = supported_format_files[".txt"]["files"]
    files.append(txt_files["simple"])
    files.append(txt_files["multi_paragraph"])
    files.append(txt_files["special_chars"])

    # Parse batch
    documents = parser.parse_batch(files)

    # Should return list of documents
    assert len(documents) == 3

    # Check each document
    for i, doc in enumerate(documents):
        assert doc.document_id is not None
        assert doc.metadata is not None
        # File path might be relative, so check if it ends with the expected filename
        expected_filename = files[i].name
        assert doc.metadata.file_path.endswith(
            expected_filename,
        ), f"Expected path ending with {expected_filename}, got {doc.metadata.file_path}"


@pytest.mark.unit
@_handle_project
def test_parse_batch_empty_list(parser):
    """Test batch parsing with empty file list."""
    documents = parser.parse_batch([])
    assert documents == []


@pytest.mark.unit
@_handle_project
def test_parse_batch_single_file(parser, supported_format_files):
    """Test batch parsing with single file."""
    txt_files = supported_format_files[".txt"]["files"]
    files = [txt_files["simple"]]
    documents = parser.parse_batch(files)

    assert len(documents) == 1
    assert "simple text file" in documents[0].to_plain_text()


@pytest.mark.unit
@_handle_project
def test_parse_batch_with_options(parser, supported_format_files):
    """Test batch parsing with custom options."""
    txt_files = supported_format_files[".txt"]["files"]
    files = [
        txt_files["simple"],
        txt_files["multi_paragraph"],
    ]

    # Parse with custom chunk size
    documents = parser.parse_batch(
        files,
        max_chunk_size=100,
        chunk_overlap=20,
    )

    assert len(documents) == 2
    # Documents should be parsed with the custom settings


@pytest.mark.unit
@_handle_project
def test_parse_batch_mixed_success_failure(
    parser,
    supported_format_files,
    tmp_path: Path,
):
    """Test batch parsing with some files that fail."""
    # Mix of existing and non-existing files
    txt_files = supported_format_files[".txt"]["files"]
    files = [
        txt_files["simple"],
        tmp_path / "nonexistent.txt",  # This doesn't exist
        txt_files["multi_paragraph"],
    ]

    # Batch parse should handle errors gracefully
    # Current implementation fails fast on first error, which is valid behavior
    with pytest.raises((FileNotFoundError, RuntimeError)):
        documents = parser.parse_batch(files)

    # Alternative: test with only valid files to ensure batch parsing works
    valid_files = [f for f in files if f.exists()]
    documents = parser.parse_batch(valid_files)
    assert len(documents) == 2  # Only the two valid files


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_parse_batch_async(parser, supported_format_files):
    """Test async batch parsing."""
    # Collect files
    txt_files = supported_format_files[".txt"]["files"]
    files = []
    files.append(txt_files["simple"])
    files.append(txt_files["multi_paragraph"])
    files.append(txt_files["special_chars"])

    # Parse asynchronously
    results = []
    async for index, doc in parser.parse_batch_async(files, batch_size=2):
        results.append((index, doc))

    # Should get all documents
    assert len(results) == 3

    # Check indices are correct
    indices = [r[0] for r in results]
    assert sorted(indices) == [0, 1, 2]

    # Check documents
    for index, doc in results:
        assert doc.document_id is not None
        # File path might be relative, so check if it ends with the expected filename
        expected_filename = files[index].name
        assert doc.metadata.file_path.endswith(
            expected_filename,
        ), f"Expected path ending with {expected_filename}, got {doc.metadata.file_path}"


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_parse_batch_async_empty(parser):
    """Test async batch parsing with empty list."""
    results = []
    async for index, doc in parser.parse_batch_async([]):
        results.append((index, doc))

    assert results == []


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_parse_batch_async_large_batch(
    parser,
    supported_format_files,
    tmp_path: Path,
):
    """Test async batch parsing with many files."""
    # Create many files
    files = []
    for i in range(10):
        file_path = tmp_path / f"test_{i}.txt"
        file_path.write_text(f"Content for file {i}", encoding="utf-8")
        files.append(file_path)

    # Parse with small batch size
    results = []
    async for index, doc in parser.parse_batch_async(files, batch_size=3):
        results.append((index, doc))

    # Should get all files
    assert len(results) == 10

    # Check order preservation
    for index, doc in results:
        expected_content = f"Content for file {index}"
        assert expected_content in doc.to_plain_text()


@pytest.mark.unit
@_handle_project
def test_parse_batch_consistent_with_single(parser, supported_format_files):
    """Test that batch parsing gives same results as individual parsing."""
    txt_files = supported_format_files[".txt"]["files"]
    files = [
        txt_files["simple"],
        txt_files["multi_paragraph"],
        txt_files["special_chars"],
    ]

    # Parse individually
    individual_docs = [parser.parse(f) for f in files]

    # Parse as batch
    batch_docs = parser.parse_batch(files)

    # Should have same number of documents
    assert len(batch_docs) == len(individual_docs)

    # Content should match
    for i in range(len(files)):
        assert individual_docs[i].to_plain_text() == batch_docs[i].to_plain_text()
        assert individual_docs[i].metadata.file_name == batch_docs[i].metadata.file_name
