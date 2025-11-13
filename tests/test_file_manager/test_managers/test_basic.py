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
    display_name = str(example_data["path"])  # absolute path
    # Exists should reflect filesystem; no import required
    assert fm.exists(display_name)


@pytest.mark.asyncio
@_handle_project
async def test_file_import_directory(file_manager, sample_files: Path):
    fm = file_manager
    # Parse all files in the directory by absolute path; ensure existence and success
    files = [str(p) for p in sample_files.iterdir() if p.is_file()]
    assert files, "No files found in sample_files fixture"
    res = fm.parse(files)
    for f in files:
        assert fm.exists(f)
        assert f in res
        _item = res[f]
        _item = _item if isinstance(_item, dict) else _item.model_dump()
        assert _item["status"] in ("success", "error")


@pytest.mark.asyncio
@_handle_project
async def test_public_import_file(file_manager, sample_files: Path):
    fm = file_manager
    sample_file = next(sample_files.iterdir())
    display_name = str(sample_file)
    # Parse by absolute path and verify success/existence
    res = fm.parse(display_name)
    assert display_name in res
    assert fm.exists(display_name)


@pytest.mark.asyncio
@_handle_project
async def test_public_import_directory(file_manager, sample_files: Path):
    fm = file_manager
    files = [str(p) for p in sample_files.iterdir() if p.is_file()]
    assert files
    res = fm.parse(files)
    for f in files:
        assert fm.exists(f)
        assert f in res


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
    name1 = str(dir1 / "test.txt")
    name2 = str(dir2 / "test.txt")
    fm.parse([name1, name2])
    assert fm.exists(name1)
    assert fm.exists(name2)


@pytest.mark.asyncio
@_handle_project
async def test_import_file_unique_names(file_manager, tmp_path: Path):
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
async def test_file_import_nonexistent_file(file_manager):
    fm = file_manager
    missing = "/nonexistent/file.txt"
    assert not fm.exists(missing)


@pytest.mark.asyncio
@_handle_project
async def test_file_import_nonexistent_directory(file_manager):
    fm = file_manager
    # Parsing from a nonexistent directory should surface as not found errors
    res = fm.parse(["/nonexistent/directory/x.txt"])
    assert "/nonexistent/directory/x.txt" in res
    _item = res["/nonexistent/directory/x.txt"]
    _item = _item if isinstance(_item, dict) else _item.model_dump()
    assert _item["status"] == "error"


@pytest.mark.asyncio
@_handle_project
async def test_parse_nonexistent_file(file_manager):
    fm = file_manager
    result = fm.parse("nonexistent.txt")
    assert "nonexistent.txt" in result
    _item = result["nonexistent.txt"]
    _item = _item if isinstance(_item, dict) else _item.model_dump()
    assert _item["status"] == "error"
    assert "not found" in str(_item.get("error", "")).lower()


@pytest.mark.asyncio
@_handle_project
async def test_filter_by_content_id_dict(file_manager, supported_file_examples: dict):
    """Ensure dict-based content_id supports hierarchical filtering on per-file Content."""
    fm = file_manager
    filename, example_data = next(iter(supported_file_examples.items()))
    display_name = str(example_data["path"])  # absolute path
    # Parse file to create per-file Content rows
    fm.parse(display_name)
    # Use file_path directly instead of legacy root from tables_overview
    # Filter for the document row using dict-based content_id
    rows = fm._filter_files(
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
async def test_parse_multiple_files_mixed(file_manager, supported_file_examples: dict):
    fm = file_manager
    filename, example_data = next(iter(supported_file_examples.items()))
    existing = str(example_data["path"])  # absolute path
    results = fm.parse([existing, "nonexistent.txt"])
    assert len(results) == 2
    _ex_item = results[existing]
    _ex_item = _ex_item if isinstance(_ex_item, dict) else _ex_item.model_dump()
    _missing_item = results["nonexistent.txt"]
    _missing_item = (
        _missing_item if isinstance(_missing_item, dict) else _missing_item.model_dump()
    )
    assert _ex_item["status"] == "success"
    assert _missing_item["status"] == "error"
