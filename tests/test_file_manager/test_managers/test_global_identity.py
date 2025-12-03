from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_global_identity_filter_and_tools(tmp_path: Path):
    from unity.file_manager.managers.local import LocalFileManager
    from unity.file_manager.managers.file_manager import FileManager
    from unity.file_manager.fs_adapters.local_adapter import LocalFileSystemAdapter
    from unity.file_manager.global_file_manager import GlobalFileManager

    # Setup two local managers: one rooted, one rootless
    fm_rooted = LocalFileManager(str(tmp_path))
    fm_rootless = FileManager(adapter=LocalFileSystemAdapter(None))

    # Create two files under the rooted tree
    f1 = tmp_path / "ident_main.txt"
    f2 = tmp_path / "ident_other.txt"
    f1.write_text("identity main")
    f2.write_text("identity other")

    # Ingest f1 into both managers
    fm_rooted.ingest_files("ident_main.txt")  # root-relative
    fm_rootless.ingest_files(str(f1))  # absolute

    # Build a GlobalFileManager over both
    gfm = GlobalFileManager([fm_rooted, fm_rootless])

    # 1) stat returns the same canonical_uri on both managers
    s_rooted = fm_rooted.stat("ident_main.txt")
    s_rootless = fm_rootless.stat(str(f1))
    assert s_rooted["canonical_uri"] and s_rootless["canonical_uri"]
    assert s_rooted["canonical_uri"] == s_rootless["canonical_uri"]
    canon = s_rooted["canonical_uri"]

    # 2) filter by source_uri works per manager
    rows_rooted = fm_rooted._filter_files(filter=f"source_uri == '{canon}'")
    rows_rootless = fm_rootless._filter_files(filter=f"source_uri == '{canon}'")
    assert rows_rooted and rows_rootless

    # 3) Global ask doesn't break identity-based filtering
    h_ask = await gfm.ask("List available filesystems and overall inventory")
    _ = await h_ask.result()
    rows_after_ask_rooted = fm_rooted._filter_files(filter=f"source_uri == '{canon}'")
    assert rows_after_ask_rooted

    # 4) Global organize on a different file does not affect f1's source_uri lookup
    # (avoid rename of f1 since source_uri currently encodes absolute path)
    h_org = await gfm.organize(
        f"Rename /{f2.name} to renamed_other.txt and then move renamed_other.txt to /",
    )
    _ = await h_org.result()
    rows_after_org_rootless = fm_rootless._filter_files(
        filter=f"source_uri == '{canon}'",
    )
    assert rows_after_org_rootless
