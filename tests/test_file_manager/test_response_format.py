"""Tests for GlobalFileManager response_format parameter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path

from unity.file_manager.simulated import (
    SimulatedFileManager,
    SimulatedGlobalFileManager,
)
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# Response format schemas
# ────────────────────────────────────────────────────────────────────────────


class FileQueryResult(BaseModel):
    """Structured result from a file query."""

    files_count: int = Field(..., description="Number of files found")
    file_names: List[str] = Field(..., description="Names of matching files")
    total_size_bytes: Optional[int] = Field(
        None,
        description="Total size of files in bytes",
    )
    summary: str = Field(..., description="Brief natural language summary")


class FileOrganizeResult(BaseModel):
    """Structured result after file organization operation."""

    success: bool = Field(..., description="Whether the operation was successful")
    operations_performed: List[str] = Field(
        default_factory=list,
        description="List of operations performed",
    )
    files_affected: int = Field(0, description="Number of files affected")
    summary: str = Field(..., description="Summary of what was done")


# ────────────────────────────────────────────────────────────────────────────
# Simulated FileManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simulated_file_manager_with_files():
    """Fixture for SimulatedFileManager with pre-loaded files."""
    fm = SimulatedFileManager()
    fm.clear_simulated_files()
    fm.add_simulated_file(
        "report.pdf",
        records=[{"content": "Quarterly financial report"}],
        metadata={"mime_type": "application/pdf"},
        full_text="Quarterly financial report content here",
        description="Q4 financial report",
    )
    fm.add_simulated_file(
        "notes.txt",
        records=[{"content": "Meeting notes from team standup"}],
        metadata={"mime_type": "text/plain"},
        full_text="Meeting notes from team standup discussion",
        description="Team meeting notes",
    )
    yield fm
    fm.clear_simulated_files()


@pytest.mark.asyncio
@_handle_project
async def test_simulated_ask_response_format(simulated_file_manager_with_files):
    """Simulated FileManager.ask should return structured output when response_format is provided."""
    fm = simulated_file_manager_with_files

    handle = await fm.ask(
        "How many files are there and what are their names?",
        response_format=FileQueryResult,
    )
    result = await handle.result()

    # Should be valid JSON conforming to the schema
    parsed = FileQueryResult.model_validate_json(result)

    assert isinstance(parsed.files_count, int)
    assert parsed.files_count >= 0
    assert isinstance(parsed.file_names, list)
    assert parsed.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_organize_response_format(simulated_file_manager_with_files):
    """Simulated FileManager.organize should return structured output when response_format is provided."""
    fm = simulated_file_manager_with_files

    handle = await fm.organize(
        "Rename notes.txt to meeting_notes.txt",
        response_format=FileOrganizeResult,
    )
    result = await handle.result()

    parsed = FileOrganizeResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert isinstance(parsed.operations_performed, list)
    assert isinstance(parsed.files_affected, int)
    assert parsed.summary.strip(), "Summary should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# Simulated GlobalFileManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simulated_global_file_manager(simulated_file_manager_with_files):
    """Fixture for SimulatedGlobalFileManager."""
    return SimulatedGlobalFileManager("Demo global file manager for testing.")


@pytest.mark.asyncio
@_handle_project
async def test_simulated_global_ask_response_format(simulated_global_file_manager):
    """Simulated GlobalFileManager.ask should return structured output when response_format is provided."""
    gfm = simulated_global_file_manager

    handle = await gfm.ask(
        "List all available files across filesystems",
        response_format=FileQueryResult,
    )
    result = await handle.result()

    parsed = FileQueryResult.model_validate_json(result)

    assert isinstance(parsed.files_count, int)
    assert isinstance(parsed.file_names, list)
    assert parsed.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_global_organize_response_format(simulated_global_file_manager):
    """Simulated GlobalFileManager.organize should return structured output when response_format is provided."""
    gfm = simulated_global_file_manager

    handle = await gfm.organize(
        "Move all reports to a dedicated folder",
        response_format=FileOrganizeResult,
    )
    result = await handle.result()

    parsed = FileOrganizeResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert isinstance(parsed.operations_performed, list)
    assert parsed.summary.strip(), "Summary should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# Real GlobalFileManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_real_ask_response_format(
    fm_root,
    file_manager,
    global_file_manager,
):
    """Real GlobalFileManager.ask should return structured output when response_format is provided."""
    # Create some test files
    root = Path(fm_root)
    (root / "test_doc.txt").write_text("Test document content")
    (root / "data.csv").write_text("col1,col2\n1,2\n3,4")

    # Parse the files
    file_manager.parse(["test_doc.txt", "data.csv"])

    gfm = global_file_manager
    handle = await gfm.ask(
        "How many files are available and what are they named?",
        response_format=FileQueryResult,
    )
    result = await handle.result()

    parsed = FileQueryResult.model_validate_json(result)

    assert isinstance(parsed.files_count, int)
    assert isinstance(parsed.file_names, list)
    assert parsed.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_real_organize_response_format(
    fm_root,
    file_manager,
    global_file_manager,
):
    """Real GlobalFileManager.organize should return structured output when response_format is provided."""
    # Create a test file
    root = Path(fm_root)
    (root / "to_rename.txt").write_text("File to be renamed")

    # Parse the file
    file_manager.parse(["to_rename.txt"])

    gfm = global_file_manager
    handle = await gfm.organize(
        "Rename to_rename.txt to renamed_file.txt",
        response_format=FileOrganizeResult,
    )
    result = await handle.result()

    parsed = FileOrganizeResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert isinstance(parsed.operations_performed, list)
    assert parsed.summary.strip(), "Summary should be non-empty"
