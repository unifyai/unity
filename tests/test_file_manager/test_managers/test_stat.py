from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_stat_reports_canonical_uri_and_existence(tmp_path, file_manager):
    fm = file_manager  # LocalFileManager bound to session fm_root

    # Create a file under the fm root
    root = Path(fm._adapter._root)  # type: ignore[attr-defined]
    (root / "stat_demo.txt").write_text("stat demo")

    rel = "stat_demo.txt"
    abs_path = (root / rel).as_posix()

    # 1) Before parse: filesystem exists, index doesn't
    s1 = fm.stat(rel)
    assert s1["filesystem_exists"] is True
    assert s1["indexed_exists"] is False
    assert isinstance(s1.get("canonical_uri"), (str, type(None)))

    # 2) After parse: indexed exists
    fm.parse(rel)
    s2 = fm.stat(rel)
    assert s2["filesystem_exists"] is True
    assert s2["indexed_exists"] is True
    assert s2.get("parsed_status") in ("success", "completed", None)

    # 3) Absolute path also resolves to same canonical uri
    s3 = fm.stat(abs_path)
    assert s3["filesystem_exists"] is True
    assert s3["indexed_exists"] is True
    # Canonical URIs should be consistent
    if s2.get("canonical_uri") and s3.get("canonical_uri"):
        assert s2["canonical_uri"] == s3["canonical_uri"]
