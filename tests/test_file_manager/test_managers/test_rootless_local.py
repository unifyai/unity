from __future__ import annotations


import pytest


@pytest.mark.asyncio
async def test_rootless_local_manager_abs_paths(tmp_path, rootless_file_manager):
    # Create files outside any configured root (rootless adapter)
    a = tmp_path / "outside_a.txt"
    b = tmp_path / "outside_b.txt"
    a.write_text("rootless A")
    b.write_text("rootless B")

    fm = rootless_file_manager

    # stat before/after parse
    s0 = fm.stat(str(a))
    assert s0["filesystem_exists"] is True and s0["indexed_exists"] is False

    fm.ingest_files([str(a), str(b)])
    s1 = fm.stat(str(a))
    s2 = fm.stat(str(b))
    assert s1["filesystem_exists"] is True and s1["indexed_exists"] is True
    assert s2["filesystem_exists"] is True and s2["indexed_exists"] is True

    # ask_about_file should accept absolute path
    h = await fm.ask_about_file(str(a), "What does this file contain?")
    ans = await h.result()
    assert isinstance(ans, str) and ans.strip()

    # organize operations should accept absolute path as well
    h2 = await fm.organize(
        f"Rename {a.as_posix()} to renamed_outside.txt and move {b.as_posix()} to {tmp_path.as_posix()}",
    )
    ans2 = await h2.result()
    assert isinstance(ans2, str) and ans2.strip()
