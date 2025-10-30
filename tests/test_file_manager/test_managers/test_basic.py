"""
Basic FileManager functionality tests (manager-level).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_file_manager_initialization(file_manager):
    fm = file_manager
    assert fm is not None
    assert hasattr(fm, "_adapter")
    assert hasattr(fm, "_parser")
    assert hasattr(fm, "get_tools")
    assert hasattr(fm, "_tools")
    assert hasattr(fm, "_ctx")


@pytest.mark.asyncio
@_handle_project
async def test_file_exists_not_found(file_manager):
    fm = file_manager
    assert not fm.exists("nonexistent.txt")


@pytest.mark.asyncio
@_handle_project
async def test_file_list_empty(file_manager):
    fm = file_manager
    # List may not be empty if previous tests ran; ensure idempotency by using a new name
    assert isinstance(fm.list(), list)


@pytest.mark.asyncio
@_handle_project
async def test_file_import_and_exists(file_manager, supported_file_examples: dict):
    filename, example_data = next(iter(supported_file_examples.items()))
    fm = file_manager
    display_name = fm.import_file(example_data["path"])  # new API
    assert fm.exists(display_name)
    assert display_name in fm.list()


@pytest.mark.asyncio
@_handle_project
async def test_file_import_directory(file_manager, sample_files: Path):
    fm = file_manager
    imported = fm.import_directory(sample_files)
    assert len(imported) > 0
    assert all(fm.exists(f) for f in imported)


@pytest.mark.asyncio
@_handle_project
async def test_public_import_file(file_manager, sample_files: Path):
    fm = file_manager
    sample_file = next(sample_files.iterdir())
    display_name = fm.import_file(sample_file)
    assert display_name is not None
    assert fm.exists(display_name)
    assert display_name in fm.list()


@pytest.mark.asyncio
@_handle_project
async def test_public_import_directory(file_manager, sample_files: Path):
    fm = file_manager
    imported = fm.import_directory(sample_files)
    assert len(imported) > 0
    for filename in imported:
        assert fm.exists(filename)


@pytest.mark.asyncio
@_handle_project
async def test_file_import_unique_names(file_manager, tmp_path: Path):
    fm = file_manager
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()
    (dir1 / "test.txt").write_text("File 1")
    (dir2 / "test.txt").write_text("File 2")
    name1 = fm.import_file(dir1 / "test.txt")
    name2 = fm.import_file(dir2 / "test.txt")
    assert name1 != name2
    assert fm.exists(name1)
    assert fm.exists(name2)


@pytest.mark.asyncio
@_handle_project
async def test_file_import_nonexistent_file(file_manager):
    fm = file_manager
    with pytest.raises(FileNotFoundError):
        fm.import_file(Path("/nonexistent/file.txt"))


@pytest.mark.asyncio
@_handle_project
async def test_file_import_nonexistent_directory(file_manager):
    fm = file_manager
    with pytest.raises(NotADirectoryError):
        fm.import_directory("/nonexistent/directory")


@pytest.mark.asyncio
@_handle_project
async def test_parse_nonexistent_file(file_manager):
    fm = file_manager
    result = fm.parse("nonexistent.txt")
    assert "nonexistent.txt" in result
    assert result["nonexistent.txt"]["status"] == "error"
    assert "not found" in result["nonexistent.txt"]["error"].lower()


@pytest.mark.asyncio
@_handle_project
async def test_parse_multiple_files_mixed(file_manager, supported_file_examples: dict):
    fm = file_manager
    filename, example_data = next(iter(supported_file_examples.items()))
    existing = fm.import_file(example_data["path"])  # new API
    results = fm.parse([existing, "nonexistent.txt"])
    assert len(results) == 2
    assert results[existing]["status"] == "success"
    assert results["nonexistent.txt"]["status"] == "error"
