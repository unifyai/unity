"""
Basic FileManager functionality tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from unity.file_manager.file_manager import FileManager
from tests.helpers import _handle_project


@_handle_project
def test_file_manager_initialization():
    """Test FileManager initializes correctly."""
    file_manager = FileManager()
    assert file_manager is not None
    assert hasattr(file_manager, "_tmp_dir")
    assert hasattr(file_manager, "_display_to_path")
    assert hasattr(file_manager, "_parser")
    assert hasattr(file_manager, "_tools")
    assert hasattr(file_manager, "_ctx")


@_handle_project
def test_file_exists_not_found():
    """Test exists() returns False for non-existent file."""
    file_manager = FileManager()
    assert not file_manager.exists("nonexistent.txt")


@_handle_project
def test_file_list_empty():
    """Test list() returns empty list when no files added."""
    file_manager = FileManager()
    assert file_manager.list() == []


@_handle_project
def test_file_import_and_exists(supported_file_examples: dict):
    """Test importing a file and checking existence."""
    # Get the first available test file
    filename, example_data = next(iter(supported_file_examples.items()))
    file_manager = FileManager()
    display_name = file_manager._add_file(example_data["path"])

    assert file_manager.exists(display_name)
    assert display_name in file_manager.list()


@_handle_project
def test_file_import_directory(sample_files: Path):
    """Test importing all files from a directory."""
    file_manager = FileManager()
    imported = file_manager.import_directory(sample_files)

    assert len(imported) > 0
    assert all(file_manager.exists(f) for f in imported)
    assert set(imported) == set(file_manager.list())


@_handle_project
def test_public_import_file(sample_files: Path):
    """Test the public import_file method."""
    # Get first available file
    file_manager = FileManager()
    sample_file = next(sample_files.iterdir())

    # Import using public method
    display_name = file_manager.import_file(sample_file)

    assert display_name is not None
    assert file_manager.exists(display_name)
    assert display_name in file_manager.list()


@_handle_project
def test_public_import_directory(sample_files: Path):
    """Test the public import_directory method (already tested above but for completeness)."""
    file_manager = FileManager()
    imported = file_manager.import_directory(sample_files)

    assert len(imported) > 0
    for filename in imported:
        assert file_manager.exists(filename)
        assert filename in file_manager.list()


@_handle_project
def test_file_import_unique_names(tmp_path: Path):
    """Test that duplicate filenames get unique display names."""
    file_manager = FileManager()
    # Create files with same name in different directories
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()

    (dir1 / "test.txt").write_text("File 1")
    (dir2 / "test.txt").write_text("File 2")

    # Import both files
    name1 = file_manager._add_file(dir1 / "test.txt")
    name2 = file_manager._add_file(dir2 / "test.txt")

    # Should have different display names
    assert name1 != name2
    assert file_manager.exists(name1)
    assert file_manager.exists(name2)

    # One should be "test.txt", the other "test (1).txt"
    assert name1 == "test.txt"
    assert name2 == "test (1).txt"


@_handle_project
def test_file_import_nonexistent_file():
    """Test importing a non-existent file raises error."""
    file_manager = FileManager()
    with pytest.raises(FileNotFoundError):
        file_manager._add_file(Path("/nonexistent/file.txt"))


@_handle_project
def test_file_import_nonexistent_directory():
    """Test importing from non-existent directory raises error."""
    file_manager = FileManager()
    with pytest.raises(NotADirectoryError):
        file_manager.import_directory("/nonexistent/directory")


@_handle_project
def test_parse_nonexistent_file():
    """Test parsing non-existent file returns error."""
    file_manager = FileManager()
    result = file_manager.parse("nonexistent.txt")

    assert "nonexistent.txt" in result
    assert result["nonexistent.txt"]["status"] == "error"
    assert "not found" in result["nonexistent.txt"]["error"].lower()


@_handle_project
def test_parse_multiple_files_mixed(supported_file_examples: dict):
    """Test parsing multiple files with mixed existence."""
    file_manager = FileManager()
    # Import one file
    filename, example_data = next(iter(supported_file_examples.items()))
    existing_file = file_manager._add_file(example_data["path"])

    # Parse mix of existing and non-existing
    results = file_manager.parse([existing_file, "nonexistent.txt"])

    assert len(results) == 2
    assert results[existing_file]["status"] == "success"
    assert results["nonexistent.txt"]["status"] == "error"
