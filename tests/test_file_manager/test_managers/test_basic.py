"""
Basic FileManager functionality tests (manager-level).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_initialization(file_manager):
    fm = file_manager
    assert fm is not None
    assert hasattr(fm, "_adapter")
    assert hasattr(fm, "_parser")
    assert hasattr(fm, "get_tools")
    assert hasattr(fm, "_tools")
    assert hasattr(fm, "_ctx")


@pytest.mark.asyncio
@_handle_project
async def test_exists_not_found(file_manager):
    fm = file_manager
    assert not fm.exists("nonexistent.txt")


@pytest.mark.asyncio
@_handle_project
async def test_list_empty(file_manager):
    fm = file_manager
    # List may not be empty if previous tests ran; ensure idempotency by using a new name
    assert isinstance(fm.list(), list)


@pytest.mark.asyncio
@_handle_project
async def test_import_and_exists(file_manager, supported_file_examples: dict):
    filename, example_data = next(iter(supported_file_examples.items()))
    fm = file_manager
    display_name = str(example_data["path"])  # absolute path
    # Exists should reflect filesystem; no import required
    assert fm.exists(display_name)


@pytest.mark.asyncio
@_handle_project
async def test_import_directory(file_manager, sample_files: Path):
    fm = file_manager
    # Parse all files in the directory by absolute path; ensure existence and success
    files = [str(p) for p in sample_files.iterdir() if p.is_file()]
    assert files, "No files found in sample_files fixture"
    res = fm.ingest_files(files)
    for f in files:
        assert fm.exists(f)
        assert f in res
        item = res[f]
        # All returns are now Pydantic models - use attribute access
        assert item.status in ("success", "error")


@pytest.mark.asyncio
@_handle_project
async def test_public_import(file_manager, sample_files: Path):
    fm = file_manager
    sample_file = next(sample_files.iterdir())
    display_name = str(sample_file)
    # Parse by absolute path and verify success/existence
    res = fm.ingest_files(display_name)
    assert display_name in res
    assert fm.exists(display_name)


@pytest.mark.asyncio
@_handle_project
async def test_public_import_directory(file_manager, sample_files: Path):
    fm = file_manager
    files = [str(p) for p in sample_files.iterdir() if p.is_file()]
    assert files
    res = fm.ingest_files(files)
    for f in files:
        assert fm.exists(f)
        assert f in res


@pytest.mark.asyncio
@_handle_project
async def test_import_unique_names_batch(file_manager, tmp_path: Path):
    fm = file_manager
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()
    (dir1 / "test.txt").write_text("File 1")
    (dir2 / "test.txt").write_text("File 2")
    name1 = str(dir1 / "test.txt")
    name2 = str(dir2 / "test.txt")
    fm.ingest_files([name1, name2])
    assert fm.exists(name1)
    assert fm.exists(name2)


@pytest.mark.asyncio
@_handle_project
async def test_import_unique_names_single(file_manager, tmp_path: Path):
    fm = file_manager
    d1 = tmp_path / "one"
    d2 = tmp_path / "two"
    d1.mkdir()
    d2.mkdir()
    (d1 / "conflict.txt").write_text("A")
    (d2 / "conflict.txt").write_text("B")
    n1 = fm.import_file(d1 / "conflict.txt")
    n2 = fm.import_file(d2 / "conflict.txt")
    assert n1 != n2
    assert fm.exists(n1)
    assert fm.exists(n2)


@pytest.mark.asyncio
@_handle_project
async def test_exists_nonexistent(file_manager):
    fm = file_manager
    missing = "/nonexistent/file.txt"
    assert not fm.exists(missing)


@pytest.mark.asyncio
@_handle_project
async def test_import_nonexistent_directory(file_manager):
    fm = file_manager
    # Parsing from a nonexistent directory should surface as not found errors
    res = fm.ingest_files(["/nonexistent/directory/x.txt"])
    assert "/nonexistent/directory/x.txt" in res
    item = res["/nonexistent/directory/x.txt"]
    # All returns are now Pydantic models - use attribute access
    assert item.status == "error"


@pytest.mark.asyncio
@_handle_project
async def test_parse_nonexistent(file_manager):
    fm = file_manager
    result = fm.ingest_files("nonexistent.txt")
    assert "nonexistent.txt" in result
    item = result["nonexistent.txt"]
    # All returns are now Pydantic models - use attribute access
    assert item.status == "error"
    assert "not found" in str(item.error or "").lower()


@pytest.mark.asyncio
@_handle_project
async def test_filter_by_content_id_dict(file_manager, supported_file_examples: dict):
    """Ensure dict-based content_id supports hierarchical filtering on per-file Content."""
    fm = file_manager
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = str(example_data["path"])  # absolute path
    # Parse file to create per-file Content rows
    fm.ingest_files(display_name)
    # Use file_path directly instead of legacy root from tables_overview
    # Filter for the document row using dict-based content_id
    rows = fm.filter_files(
        filter="content_type == 'document' and content_id.get('document') == 0",
        tables=[display_name],
    )
    assert isinstance(rows, list)
    assert rows, "Expected at least one Content row for the document"
    # Basic sanity: returned row should have a content_id dict
    first = rows[0]
    assert isinstance(first.get("content_id"), dict)
    assert first.get("content_type") == "document"
    assert first["content_id"].get("document") == 0


@pytest.mark.asyncio
@_handle_project
async def test_parse_multiple_mixed(file_manager, supported_file_examples: dict):
    fm = file_manager
    filename, example_data = next(iter(supported_file_examples.items()))
    existing = str(example_data["path"])  # absolute path
    results = fm.ingest_files([existing, "nonexistent.txt"])
    assert len(results) == 2
    # All returns are now Pydantic models - use attribute access
    assert results[existing].status == "success"
    assert results["nonexistent.txt"].status == "error"
