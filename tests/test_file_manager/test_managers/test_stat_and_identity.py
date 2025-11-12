from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_extract_filesystem_type_and_safe(file_manager):
    fm = file_manager

    # Filesystem type extraction strips details in brackets
    assert fm._extract_filesystem_type("Local [/tmp/root]") == "Local"
    assert fm._extract_filesystem_type("CodeSandbox [abc]") == "CodeSandbox"
    assert fm._extract_filesystem_type("") == "Unknown"

    # _safe compresses and removes path punctuation, preserving tail
    s = fm._safe("/very/long/path/to/a/Report.v1.2.pdf")
    assert all(c.isalnum() or c in "_-" for c in s)
    assert s.endswith("Report_v1_2_pdf")


@pytest.mark.asyncio
async def test_resolve_to_uri_and_stat(tmp_path: Path, file_manager):
    fm = file_manager

    # Create a file under the fm root and parse
    root = Path(fm._adapter._root)  # type: ignore[attr-defined]
    p = root / "id_stat_demo.txt"
    p.write_text("content")

    abs_path = p.as_posix()
    uri = fm._resolve_to_uri(abs_path)
    assert isinstance(uri, (str, type(None)))
    if isinstance(uri, str):
        assert uri.startswith(("local://", "codesandbox://", "interact://"))

    s1 = fm.stat(abs_path)
    assert s1["filesystem_exists"] is True
    assert s1["indexed_exists"] in (False, True)

    # After parse, indexed_exists should be True
    fm.parse(abs_path)
    s2 = fm.stat(abs_path)
    assert s2["filesystem_exists"] is True
    assert s2["indexed_exists"] is True
    # Canonical uri should be stable
    if s1.get("canonical_uri") and s2.get("canonical_uri"):
        assert s1["canonical_uri"] == s2["canonical_uri"]
