"""
FileManager parse functionality tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_parse_single(file_manager, supported_file_examples: dict):
    """Test parsing a single file."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = str(example_data["path"])  # absolute path

    from unity.file_manager.types import FilePipelineConfig, ParsedFile

    # Request full mode to assert heavy fields like 'records'
    result = file_manager.ingest_files(
        display_name,
        config=FilePipelineConfig(output={"return_mode": "full"}),
    )

    assert display_name in result
    # Full mode returns ParsedFile Pydantic model
    item = result[display_name]
    assert isinstance(item, ParsedFile)
    assert item.status == "success"
    assert isinstance(item.records, list)

    # Check flattened metadata
    assert hasattr(item, "file_format")
    assert hasattr(item, "file_size")


@pytest.mark.asyncio
async def test_parse_multiple(file_manager, supported_file_examples: dict):
    """Test parsing multiple files at once."""
    # Import all example files
    display_names = []
    for filename, example_data in supported_file_examples.items():
        display_name = str(example_data["path"])  # absolute path
        display_names.append(display_name)

    # Parse all files - compact is default, returns Pydantic models
    results = file_manager.ingest_files(display_names)

    assert len(results) == len(display_names)
    for display_name in display_names:
        assert display_name in results
        item = results[display_name]
        # All returns are now Pydantic models - use attribute access
        assert item.status == "success"


@pytest.mark.asyncio
async def test_parse_with_options(file_manager, supported_file_examples: dict):
    """Test parsing with custom options."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = str(example_data["path"])  # absolute path

    # Parse with options via config (forwarded to parser)
    from unity.file_manager.types import FilePipelineConfig, ParseConfig

    cfg = FilePipelineConfig(
        parse=ParseConfig(parser_kwargs={"max_chunk_size": 100, "chunk_overlap": 20}),
    )
    result = file_manager.ingest_files(display_name, config=cfg)

    assert display_name in result
    item = result[display_name]
    # All returns are now Pydantic models - use attribute access
    assert item.status == "success"


@pytest.mark.asyncio
async def test_parse_empty(file_manager, sample_files: Path):
    """Test parsing an empty file."""
    # Use empty file by absolute path
    empty_file = sample_files / "empty.txt"
    display_name = str(empty_file)

    from unity.file_manager.types import FilePipelineConfig, ParsedFile

    # Request full mode to test 'records' semantics on empty file
    result = file_manager.ingest_files(
        display_name,
        config=FilePipelineConfig(output={"return_mode": "full"}),
    )

    assert display_name in result
    # Full mode returns ParsedFile Pydantic model
    item = result[display_name]
    assert isinstance(item, ParsedFile)
    # Empty file should still parse successfully
    assert item.status == "success"
    # May have a document structure but content should be empty
    records = item.records
    if records:
        # All content should be empty
        all_content = " ".join(
            str(record.get("content_text", "")) for record in records
        ).strip()
        assert (
            "empty" in all_content.lower()
        ), "Empty file with no content should fallback to using the title as content"


@pytest.mark.asyncio
async def test_parse_supported(file_manager, supported_file_examples: dict):
    """Test parsing files in all supported formats."""
    # Add all example files to the file manager
    display_names = []
    for filename, example_data in supported_file_examples.items():
        display_name = str(example_data["path"])  # absolute path
        display_names.append(display_name)

    # Test parsing each file individually
    from unity.file_manager.types import FilePipelineConfig, ParsedFile

    for display_name in display_names:
        result = file_manager.ingest_files(
            display_name,
            config=FilePipelineConfig(output={"return_mode": "full"}),
        )

        assert display_name in result
        item = result[display_name]
        assert isinstance(item, ParsedFile)
        assert item.status == "success"
        assert len(item.records) > 0

        # If this is a spreadsheet (csv or xlsx), ensure per-table context is present
        file_format = item.file_format
        if file_format in ("csv", "xlsx"):
            try:
                import unify

                ctxs = unify.get_contexts(prefix=f"{unify.active_project()}/")
                table_ctx_candidates = [
                    name for name in ctxs.keys() if "/Tables/" in name
                ]
                assert (
                    table_ctx_candidates
                ), "Expected per-table contexts for spreadsheets"
            except Exception:
                pass


@pytest.mark.asyncio
async def test_parse_multiple_supported(
    file_manager,
    supported_file_examples: dict,
):
    """Test parsing multiple files in supported formats."""
    # Add all example files to the file manager
    display_names = []
    for filename, example_data in supported_file_examples.items():
        display_name = str(example_data["path"])  # absolute path
        display_names.append(display_name)

    # Parse all files at once
    from unity.file_manager.types import FilePipelineConfig, ParsedFile

    result = file_manager.ingest_files(
        display_names,
        config=FilePipelineConfig(output={"return_mode": "full"}),
    )

    for display_name in display_names:
        assert display_name in result
        file_result = result[display_name]
        assert isinstance(file_result, ParsedFile)
        assert file_result.status == "success"
        assert len(file_result.records) > 0

    # Sanity check that at least one table context exists
    try:
        import unify

        ctxs = unify.get_contexts(prefix=f"{unify.active_project()}/")
        assert any("/Tables/" in name for name in ctxs.keys())
    except Exception:
        pass
