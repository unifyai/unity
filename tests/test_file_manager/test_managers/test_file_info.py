"""Consolidated tests for FileManager.file_info tool.

This module tests all aspects of the _file_info tool including:
- Basic existence and status checks
- Lookup by file_path, file_id, and source_uri
- Source URI resolution and consistency
- Cross-provider identity (e.g., Google Drive stubs)
- Root vs rootless adapter scenarios
- Global FileManager identity handling
- Identity helper methods (safe, _extract_filesystem_type, _resolve_to_uri)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pytest
from tests.helpers import _handle_project


# =============================================================================
# BASIC FILE INFO TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_file_info_reports_source_uri_and_existence(tmp_path, file_manager):
    """Test basic file_info output: filesystem/index existence and source_uri."""
    fm = file_manager

    root = Path(fm._adapter._root)  # type: ignore[attr-defined]
    (root / "info_demo.txt").write_text("file info demo")

    rel = "info_demo.txt"
    abs_path = (root / rel).as_posix()

    # Before parse: filesystem exists, index doesn't
    info1 = fm.file_info(identifier=rel)
    assert info1.filesystem_exists is True
    assert info1.indexed_exists is False
    assert isinstance(info1.source_uri, (str, type(None)))

    # After parse: indexed exists
    fm.ingest_files(rel)
    info2 = fm.file_info(identifier=rel)
    assert info2.filesystem_exists is True
    assert info2.indexed_exists is True
    assert info2.parsed_status in ("success", "completed", None)

    # Absolute path also resolves to same source uri
    info3 = fm.file_info(identifier=abs_path)
    assert info3.filesystem_exists is True
    assert info3.indexed_exists is True
    if info2.source_uri and info3.source_uri:
        assert info2.source_uri == info3.source_uri


@pytest.mark.asyncio
async def test_file_info_by_file_id(tmp_path, file_manager):
    """Test file_info lookup by numeric file_id."""
    fm = file_manager

    root = Path(fm._adapter._root)  # type: ignore[attr-defined]
    (root / "info_by_id.txt").write_text("lookup by id")
    fm.ingest_files("info_by_id.txt")

    info_by_path = fm.file_info(identifier="info_by_id.txt")
    assert info_by_path.indexed_exists is True

    # Get file_id from the index
    rows = fm.filter_files(filter="file_path.endswith('info_by_id.txt')")
    assert rows
    file_id = rows[0].get("file_id")
    assert file_id is not None

    # Lookup by file_id
    info_by_id = fm.file_info(identifier=file_id)
    assert info_by_id.indexed_exists is True
    assert info_by_id.source_uri == info_by_path.source_uri


@pytest.mark.asyncio
async def test_file_info_ingest_mode_fields(tmp_path, file_manager):
    """Test that file_info returns all ingest/identity fields."""
    fm = file_manager

    root = Path(fm._adapter._root)  # type: ignore[attr-defined]
    (root / "ingest_mode_test.txt").write_text("check ingest fields")
    fm.ingest_files("ingest_mode_test.txt")

    info = fm.file_info(identifier="ingest_mode_test.txt")
    assert info.indexed_exists is True
    # Check all identity fields are present as attributes
    assert hasattr(info, "ingest_mode")
    assert hasattr(info, "unified_label")
    assert hasattr(info, "table_ingest")
    assert hasattr(info, "source_provider")
    assert hasattr(info, "file_format")


# =============================================================================
# IDENTITY HELPER TESTS
# =============================================================================


def test_extract_filesystem_type_and_safe(file_manager):
    """Test identity helper methods: _extract_filesystem_type and safe."""
    fm = file_manager

    # Filesystem type extraction strips details in brackets
    assert fm._extract_filesystem_type("Local [/tmp/root]") == "Local"
    assert fm._extract_filesystem_type("CodeSandbox [abc]") == "CodeSandbox"
    assert fm._extract_filesystem_type("") == "Unknown"

    # safe compresses and removes path punctuation, preserving tail
    s = fm.safe("/very/long/path/to/a/Report.v1.2.pdf")
    assert all(c.isalnum() or c in "_-" for c in s)
    assert s.endswith("Report_v1_2_pdf")


@pytest.mark.asyncio
async def test_resolve_to_uri_and_file_info(tmp_path: Path, file_manager):
    """Test _resolve_to_uri consistency with file_info."""
    fm = file_manager

    root = Path(fm._adapter._root)  # type: ignore[attr-defined]
    p = root / "id_info_demo.txt"
    p.write_text("content")

    abs_path = p.as_posix()
    uri = fm._resolve_to_uri(abs_path)
    assert isinstance(uri, (str, type(None)))
    if isinstance(uri, str):
        assert uri.startswith(("local://", "codesandbox://", "interact://"))

    info1 = fm.file_info(identifier=abs_path)
    assert info1.filesystem_exists is True
    assert info1.indexed_exists in (False, True)

    # After parse, indexed_exists should be True
    fm.ingest_files(abs_path)
    info2 = fm.file_info(identifier=abs_path)
    assert info2.filesystem_exists is True
    assert info2.indexed_exists is True
    if info1.source_uri and info2.source_uri:
        assert info1.source_uri == info2.source_uri


# =============================================================================
# SOURCE URI FILTER AND CONSISTENCY TESTS
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_source_uri_filter_local(file_manager, tmp_path: Path):
    """Test filtering by source_uri after file_info lookup."""
    fm = file_manager
    fm.clear()

    p = tmp_path / "ident_a.txt"
    p.write_text("alpha")
    fm.ingest_files(str(p))

    info = fm.file_info(identifier=str(p))
    uri = info.source_uri
    assert isinstance(uri, str) and uri.startswith("local://")

    rows = fm.filter_files(filter=f"source_uri == '{uri}'")
    assert len(rows) >= 1
    assert any(r.get("source_uri") == uri for r in rows)


@pytest.mark.asyncio
@_handle_project
async def test_root_vs_rootless_source_uri_consistency(tmp_path: Path):
    """Test source_uri consistency between rooted and rootless managers."""
    from unity.file_manager.managers.file_manager import FileManager
    from unity.file_manager.managers.local import LocalFileManager
    from unity.file_manager.filesystem_adapters.local_adapter import (
        LocalFileSystemAdapter,
    )

    a = tmp_path / "rootless_a.txt"
    a.write_text("rootless")

    fm_rooted = LocalFileManager(str(tmp_path))
    fm_rooted.clear()
    fm_rootless = FileManager(adapter=LocalFileSystemAdapter(None))
    fm_rootless.clear()

    fm_rooted.ingest_files("rootless_a.txt")
    fm_rootless.ingest_files(str(a))

    info1 = fm_rooted.file_info(identifier="rootless_a.txt")
    info2 = fm_rootless.file_info(identifier=str(a))
    assert info1.filesystem_exists and info2.filesystem_exists
    assert info1.indexed_exists and info2.indexed_exists

    assert isinstance(info1.source_uri, str) and info1.source_uri.startswith(
        "local://",
    )
    assert info1.source_uri == info2.source_uri


# =============================================================================
# ROOTLESS ADAPTER TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_rootless_local_manager_file_info(tmp_path, rootless_file_manager):
    """Test file_info with rootless adapter and absolute paths."""
    a = tmp_path / "outside_a.txt"
    b = tmp_path / "outside_b.txt"
    a.write_text("rootless A")
    b.write_text("rootless B")

    fm = rootless_file_manager

    info0 = fm.file_info(identifier=str(a))
    assert info0.filesystem_exists is True and info0.indexed_exists is False

    fm.ingest_files([str(a), str(b)])
    info1 = fm.file_info(identifier=str(a))
    info2 = fm.file_info(identifier=str(b))
    assert info1.filesystem_exists is True and info1.indexed_exists is True
    assert info2.filesystem_exists is True and info2.indexed_exists is True

    # ask_about_file should accept absolute path
    h = await fm.ask_about_file(str(a), "What does this file contain?")
    ans = await h.result()
    assert isinstance(ans, str) and ans.strip()


# =============================================================================
# CROSS-PROVIDER IDENTITY TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_gdrive_source_uri_stub():
    """Test file_info with a Google Drive stub adapter."""
    from unity.file_manager.filesystem_adapters.base import BaseFileSystemAdapter
    from unity.file_manager.types.filesystem import (
        FileSystemCapabilities,
        FileReference,
    )
    from unity.file_manager.managers.file_manager import FileManager

    class GoogleDriveStubAdapter(BaseFileSystemAdapter):
        def __init__(
            self,
            initial_files: Optional[
                Dict[str, Tuple[str, bytes, str | None, str | None]]
            ] = None,
        ):
            self._files: Dict[str, Tuple[str, bytes, str | None, str | None]] = dict(
                initial_files or {},
            )
            self._caps = FileSystemCapabilities(
                can_read=True,
                can_rename=False,
                can_move=False,
                can_delete=False,
            )

        @property
        def name(self) -> str:
            return "GoogleDrive"

        @property
        def uri_name(self) -> str:
            return "gdrive"

        @property
        def capabilities(self) -> FileSystemCapabilities:
            return self._caps

        def _resolve(
            self,
            key: str,
        ) -> Tuple[str, Tuple[str, bytes, str | None, str | None]]:
            if key in self._files:
                return key, self._files[key]
            lookup = key.lstrip("/")
            for fid, (nm, data, mime, mtime) in self._files.items():
                if nm == lookup:
                    return fid, (nm, data, mime, mtime)
            raise FileNotFoundError(key)

        def iter_files(self, root: Optional[str] = None) -> Iterable[FileReference]:
            for fid, (nm, data, mime, mtime) in self._files.items():
                yield FileReference(
                    path=f"/{nm}",
                    name=nm,
                    provider=self.name,
                    uri=f"{self.uri_name}://{fid}",
                    size_bytes=len(data),
                    modified_at=mtime,
                    mime_type=mime,
                )

        def get_file(self, path: str) -> FileReference:
            fid, (nm, data, mime, mtime) = self._resolve(path)
            return FileReference(
                path=f"/{nm}",
                name=nm,
                provider=self.name,
                uri=f"{self.uri_name}://{fid}",
                size_bytes=len(data),
                modified_at=mtime,
                mime_type=mime,
            )

        def exists(self, path: str) -> bool:
            try:
                self._resolve(path)
                return True
            except FileNotFoundError:
                return False

        def list(self, root: Optional[str] = None) -> List[str]:
            return [ref.path.lstrip("/") for ref in self.iter_files(root)]

        def open_bytes(self, path: str) -> bytes:
            _, (_, data, _, _) = self._resolve(path)
            return data

        def export_file(self, path: str, destination_dir: str) -> str:
            fid, (nm, data, _, _) = self._resolve(path)
            dest_dir = Path(destination_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / nm
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(data)
            return str(dest_path)

        def export_directory(self, path: str, destination_dir: str) -> List[str]:
            out: List[str] = []
            for fid in list(self._files.keys()):
                try:
                    out.append(self.export_file(fid, destination_dir))
                except Exception:
                    continue
            return out

    stub = GoogleDriveStubAdapter(
        initial_files={
            "id_123": ("stub_doc.txt", b"stub content for gdrive", "text/plain", None),
        },
    )
    fm = FileManager(adapter=stub)

    # file_info should yield a gdrive:// URI using either id or path
    info_by_id = fm.file_info(identifier="id_123")
    info_by_path = fm.file_info(identifier="/stub_doc.txt")
    assert isinstance(info_by_id.source_uri, str) and info_by_id.source_uri.startswith(
        "gdrive://",
    )
    assert info_by_id.source_uri == info_by_path.source_uri

    # Filtering by source_uri should be supported after parse
    fm.ingest_files("id_123")
    uri = info_by_id.source_uri
    rows = fm.filter_files(filter=f"source_uri == '{uri}'")
    assert rows


# =============================================================================
# GLOBAL FILE MANAGER IDENTITY TESTS
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_global_file_manager_identity(tmp_path: Path):
    """Test file_info consistency across GlobalFileManager with multiple adapters."""
    from unity.file_manager.managers.local import LocalFileManager
    from unity.file_manager.managers.file_manager import FileManager
    from unity.file_manager.filesystem_adapters.local_adapter import (
        LocalFileSystemAdapter,
    )
    from unity.file_manager.global_file_manager import GlobalFileManager

    fm_rooted = LocalFileManager(str(tmp_path))
    fm_rootless = FileManager(adapter=LocalFileSystemAdapter(None))

    f1 = tmp_path / "ident_main.txt"
    f2 = tmp_path / "ident_other.txt"
    f1.write_text("identity main")
    f2.write_text("identity other")

    fm_rooted.ingest_files("ident_main.txt")
    fm_rootless.ingest_files(str(f1))

    gfm = GlobalFileManager([fm_rooted, fm_rootless])

    # file_info returns the same source_uri on both managers
    info_rooted = fm_rooted.file_info(identifier="ident_main.txt")
    info_rootless = fm_rootless.file_info(identifier=str(f1))
    assert info_rooted.source_uri and info_rootless.source_uri
    assert info_rooted.source_uri == info_rootless.source_uri
    canon = info_rooted.source_uri

    # filter by source_uri works per manager
    rows_rooted = fm_rooted.filter_files(filter=f"source_uri == '{canon}'")
    rows_rootless = fm_rootless.filter_files(filter=f"source_uri == '{canon}'")
    assert rows_rooted and rows_rootless

    # list_filesystems should work
    filesystems = gfm.list_filesystems()
    assert isinstance(filesystems, list) and len(filesystems) > 0
