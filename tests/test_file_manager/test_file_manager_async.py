"""
FileManager async functionality tests.
"""

from __future__ import annotations


import pytest

from unity.file_manager.file_manager import FileManager


@pytest.mark.asyncio
async def test_parse_async_single_file(supported_file_examples: dict):
    file_manager = FileManager()
    """Test async parsing of a single file."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = file_manager._add_file(example_data["path"])

    # Parse asynchronously
    results = []
    async for result in file_manager.parse_async(display_name):
        results.append(result)

    assert len(results) == 1
    assert results[0]["filename"] == display_name
    assert results[0]["status"] == "success"


@pytest.mark.asyncio
async def test_parse_async_multiple_files(supported_file_examples: dict):
    file_manager = FileManager()
    """Test async parsing of multiple files."""
    # Import all example files
    imported = []
    for filename, example_data in supported_file_examples.items():
        display_name = file_manager._add_file(example_data["path"])
        imported.append(display_name)

    # Parse asynchronously
    results = []
    filenames_seen = set()
    async for result in file_manager.parse_async(imported):
        results.append(result)
        filenames_seen.add(result["filename"])

    # Should get result for each file
    assert len(results) == len(imported)
    assert filenames_seen == set(imported)

    # All should have a status
    for result in results:
        assert "status" in result
        assert result["status"] in ["success", "error"]


@pytest.mark.asyncio
async def test_parse_async_with_batch_size(supported_file_examples: dict):
    file_manager = FileManager()
    """Test async parsing with custom batch size."""
    # Import multiple files
    imported = []
    for filename, example_data in supported_file_examples.items():
        display_name = file_manager._add_file(example_data["path"])
        imported.append(display_name)

    # Parse with small batch size
    results = []
    async for result in file_manager.parse_async(imported, batch_size=2):
        results.append(result)

    assert len(results) == len(imported)


@pytest.mark.asyncio
async def test_parse_async_mixed_files(supported_file_examples: dict):
    file_manager = FileManager()
    """Test async parsing with mix of existing and non-existing files."""
    # Import one file
    filename, example_data = next(iter(supported_file_examples.items()))
    existing = file_manager._add_file(example_data["path"])

    # Mix of existing and non-existing
    files_to_parse = [existing, "nonexistent.txt", "another_missing.txt"]

    results = []
    errors = []
    async for result in file_manager.parse_async(files_to_parse):
        results.append(result)
        if result["status"] == "error":
            errors.append(result)

    assert len(results) == 3
    assert len(errors) == 2  # Two non-existent files

    # Check error messages
    for error_result in errors:
        assert "not found" in error_result["error"].lower()


@pytest.mark.asyncio
async def test_parse_async_empty_list():
    file_manager = FileManager()
    """Test async parsing with empty file list."""
    results = []
    async for result in file_manager.parse_async([]):
        results.append(result)

    assert len(results) == 0


@pytest.mark.asyncio
async def test_parse_async_with_options(supported_file_examples: dict):
    file_manager = FileManager()
    """Test async parsing with parser options."""
    # Import files
    imported = []
    for filename, example_data in supported_file_examples.items():
        display_name = file_manager._add_file(example_data["path"])
        imported.append(display_name)

    # Parse with options
    results = []
    async for result in file_manager.parse_async(
        imported,
        batch_size=3,
        max_chunk_size=100,
        chunk_overlap=20,
    ):
        results.append(result)

    assert len(results) == len(imported)
    # Verify all parsed successfully
    for result in results:
        assert result["status"] == "success"
