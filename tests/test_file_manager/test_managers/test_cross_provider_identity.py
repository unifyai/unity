from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pytest


@pytest.mark.asyncio
async def test_gdrive_source_uri_and_stat_stub():
    # Inline stub adapter to avoid adding production code
    from unity.file_manager.fs_adapters.base import BaseFileSystemAdapter
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

    # Create a stub store with one file id
    stub = GoogleDriveStubAdapter(
        initial_files={
            "id_123": ("stub_doc.txt", b"stub content for gdrive", "text/plain", None),
        },
    )
    fm = FileManager(adapter=stub)

    # stat should yield a gdrive:// URI using either id or path
    st_by_id = fm.stat("id_123")
    st_by_path = fm.stat("/stub_doc.txt")
    assert isinstance(st_by_id.get("canonical_uri"), str) and st_by_id[
        "canonical_uri"
    ].startswith("gdrive://")
    assert st_by_id["canonical_uri"] == st_by_path["canonical_uri"]

    # Filtering by source_uri should be supported after parse
    fm.parse("id_123")
    uri = st_by_id["canonical_uri"]
    rows = fm._filter_files(filter=f"source_uri == '{uri}'")
    assert rows
