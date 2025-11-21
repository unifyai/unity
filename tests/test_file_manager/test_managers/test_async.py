"""
FileManager async functionality tests.
"""

from __future__ import annotations


import pytest


@pytest.mark.asyncio
async def test_parse_async_single_file(file_manager, supported_file_examples: dict):
    """Test async parsing of a single file."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = str(example_data["path"])  # absolute path; no import needed

    # Parse asynchronously (defaults)
    results = []
    async for result in file_manager.parse_async(display_name):
        results.append(result)

    assert len(results) == 1
    _r0 = results[0] if isinstance(results[0], dict) else results[0].model_dump()
    assert _r0["file_path"] == display_name
    assert _r0["status"] == "success"


@pytest.mark.asyncio
async def test_parse_async_multiple_files(file_manager, supported_file_examples: dict):
    """Test async parsing of multiple files."""
    # Use absolute paths for all example files
    imported = [str(ex["path"]) for _, ex in supported_file_examples.items()]

    # Parse asynchronously
    results = []
    filenames_seen = set()
    async for result in file_manager.parse_async(imported):
        _r = result if isinstance(result, dict) else result.model_dump()
        results.append(_r)
        filenames_seen.add(_r["file_path"])

    # Should get result for each file
    assert len(results) == len(imported)
    assert filenames_seen == set(imported)

    # All should have a status
    for result in results:
        assert "status" in result
        assert result["status"] in ["success", "error"]


@pytest.mark.asyncio
async def test_parse_async_with_batch_size(file_manager, supported_file_examples: dict):
    """Test async parsing with custom batch size."""
    # Use absolute paths
    imported = [str(ex["path"]) for _, ex in supported_file_examples.items()]

    # Parse with small batch size via config
    from unity.file_manager.types import FilePipelineConfig, ParseConfig

    cfg = FilePipelineConfig(parse=ParseConfig(batch_size=2))
    results = []
    async for result in file_manager.parse_async(imported, config=cfg):
        results.append(result if isinstance(result, dict) else result.model_dump())

    assert len(results) == len(imported)


@pytest.mark.asyncio
async def test_parse_async_mixed_files(file_manager, supported_file_examples: dict):
    """Test async parsing with mix of existing and non-existing files."""
    # One existing file by absolute path
    filename, example_data = next(iter(supported_file_examples.items()))
    existing = str(example_data["path"])  # absolute path

    # Mix of existing and non-existing
    files_to_parse = [existing, "nonexistent.txt", "another_missing.txt"]

    results = []
    errors = []
    async for result in file_manager.parse_async(files_to_parse):
        _r = result if isinstance(result, dict) else result.model_dump()
        results.append(_r)
        if _r["status"] == "error":
            errors.append(_r)

    assert len(results) == 3
    assert len(errors) == 2  # Two non-existent files

    # Check error messages
    for error_result in errors:
        assert "not found" in str(error_result.get("error", "")).lower()


@pytest.mark.asyncio
async def test_parse_async_empty_list(file_manager):
    """Test async parsing with empty file list."""
    results = []
    async for result in file_manager.parse_async([]):
        results.append(result)

    assert len(results) == 0


@pytest.mark.asyncio
async def test_parse_async_with_options(file_manager, supported_file_examples: dict):
    """Test async parsing with parser options."""
    # Use absolute paths
    imported = [str(ex["path"]) for _, ex in supported_file_examples.items()]

    # Parse with options via config
    from unity.file_manager.types import FilePipelineConfig, ParseConfig

    cfg = FilePipelineConfig(
        parse=ParseConfig(
            batch_size=3,
            parser_kwargs={"max_chunk_size": 100, "chunk_overlap": 20},
        ),
    )
    results = []
    async for result in file_manager.parse_async(imported, config=cfg):
        results.append(result if isinstance(result, dict) else result.model_dump())

    assert len(results) == len(imported)
    # Verify all parsed successfully
    for result in results:
        assert result["status"] == "success"
