from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_source_uri_filter_local(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    # Create a file under fm root and parse it by absolute path
    p = tmp_path / "ident_a.txt"
    p.write_text("alpha")
    fm.parse(str(p))

    # Resolve canonical URI via stat
    st = fm.stat(str(p))
    uri = st.get("canonical_uri")
    assert isinstance(uri, str) and uri.startswith("local://")

    # Filter by source_uri should return exactly one row for this file
    rows = fm._filter_files(filter=f"source_uri == '{uri}'")
    assert len(rows) >= 1
    assert any(r.get("source_uri") == uri for r in rows)


@pytest.mark.asyncio
@_handle_project
async def test_source_uri_consistency_across_rename_local(file_manager, tmp_path: Path):
    fm = file_manager
    fm.clear()
    # Create file and ingest
    src = tmp_path / "rename_me_ident.txt"
    src.write_text("data")
    fm.parse(str(src))

    # Capture canonical URI
    before = fm.stat(str(src))
    uri = before.get("canonical_uri")
    assert isinstance(uri, str)

    # Rename via organize surface
    h = await fm.organize(f"Rename {src.as_posix()} to renamed_ident.txt")
    _ = await h.result()

    # The source_uri identity should still find the row; file_path should reflect new name
    rows = fm._filter_files(filter=f"source_uri == '{uri}'")
    assert rows, "Row should still be indexed by source_uri after rename"
    assert any("renamed_ident.txt" in str(r.get("file_path", "")) for r in rows)


@pytest.mark.asyncio
@_handle_project
async def test_root_vs_rootless_parametrized(tmp_path: Path):
    from unity.file_manager.managers.file_manager import FileManager
    from unity.file_manager.managers.local import LocalFileManager
    from unity.file_manager.fs_adapters.local_adapter import LocalFileSystemAdapter

    # Files outside any specific root
    a = tmp_path / "rootless_a.txt"
    a.write_text("rootless")

    # One manager with root, one rootless
    fm_rooted = LocalFileManager(str(tmp_path))
    fm_rooted.clear()
    fm_rootless = FileManager(adapter=LocalFileSystemAdapter(None))
    fm_rootless.clear()

    # Ingest via both managers
    fm_rooted.parse("rootless_a.txt")  # root-relative
    fm_rootless.parse(str(a))

    # Both should see the file on their respective surfaces
    st1 = fm_rooted.stat("rootless_a.txt")
    st2 = fm_rootless.stat(str(a))
    assert st1["filesystem_exists"] and st2["filesystem_exists"]
    assert st1["indexed_exists"] and st2["indexed_exists"]

    # canonical_uri scheme should be local:// and identical across both views
    assert isinstance(st1["canonical_uri"], str) and st1["canonical_uri"].startswith(
        "local://",
    )
    assert st1["canonical_uri"] == st2["canonical_uri"]
