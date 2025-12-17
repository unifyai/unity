from __future__ import annotations

from typing import Iterable, Optional, List, Dict, Any
import os

from unify.utils import http

from unity.session_details import SESSION_DETAILS
from unity.file_manager.filesystem_adapters.base import BaseFileSystemAdapter
from unity.file_manager.types.filesystem import FileSystemCapabilities, FileReference


class CodeSandboxFileSystemAdapter(BaseFileSystemAdapter):
    """Adapter for CodeSandbox filesystem SDK.

    Notes
    -----
    This adapter expects a client instance that exposes a ``fs`` namespace with
    methods compatible with the CodeSandbox SDK, e.g. ``readdir``, ``readFile``,
    ``rename``, ``copy``, ``remove`` as documented in the File System SDK
    reference.

    When a client is not provided, mutating operations raise ``NotImplementedError``
    and listing/reads return empty/defaults.
    """

    def __init__(
        self,
        sandbox_id: str,
        auth_token: str | None = None,
        *,
        client: Optional[Any] = None,
        service_base_url: Optional[str] = None,
    ):
        self._sandbox_id = sandbox_id
        self._token = auth_token or os.environ.get("CODESANDBOX_API_TOKEN") or ""
        # Optional direct SDK client; when not provided, we route via the local codesandbox-service
        self._client = client
        self._service_base = (
            service_base_url
            or os.environ.get("CODESANDBOX_SERVICE_BASE_URL")
            or f"http://localhost:{os.environ.get('CODESANDBOX_SERVICE_PORT','3100')}"
        ).rstrip("/")
        self._connected = False
        self._caps = FileSystemCapabilities(
            can_read=True,
            can_rename=True,
            can_move=True,
            can_delete=True,
        )

    @property
    def name(self) -> str:
        return f"CodeSandbox"

    @property
    def uri_name(self) -> str:
        return "csb"

    @property
    def capabilities(self) -> FileSystemCapabilities:
        return self._caps

    def _join(self, *parts: str) -> str:
        p = "/".join(str(x).strip("/") for x in parts if str(x))
        return "/" + p if not p.startswith("/") else p

    def _is_dir_entry(self, entry: Dict[str, Any]) -> bool:
        # SDK returns directory entries with a type or isDir flag depending on client
        kind = entry.get("type")
        if isinstance(kind, str) and kind.lower() in {"dir", "directory", "folder"}:
            return True
        return bool(entry.get("isDir") or entry.get("isDirectory"))

    def _headers(self) -> Dict[str, str]:
        # Mirror agent-service header style; service only checks presence
        unify_key = os.environ.get("UNIFY_KEY", "")
        assistant_email = SESSION_DETAILS.assistant.email
        return {"authorization": f"Bearer {unify_key} {assistant_email}".strip()}

    def _ensure_connected(self) -> None:
        if self._client is not None:
            return
        if self._connected:
            return
        try:
            url = f"{self._service_base}/sandboxes/{self._sandbox_id}/connect"
            resp = http.post(
                url,
                headers=self._headers(),
                timeout=30,
                raise_for_status=False,
            )
            if resp.status_code < 400:
                self._connected = True
        except Exception:
            # Leave as not connected; subsequent calls will retry
            self._connected = False

    def _readdir_once(self, directory: str) -> List[str]:
        if self._client is not None and hasattr(self._client, "fs"):
            try:
                items = self._client.fs.readdir(directory)  # type: ignore[attr-defined]
                names: List[str] = []
                for it in items or []:
                    names.append(
                        (
                            it
                            if isinstance(it, str)
                            else str(dict(it).get("name") or dict(it).get("path") or "")
                        ),
                    )
                return [self._join(directory, n) for n in names if n]
            except Exception:
                return []
        # HTTP fallback
        self._ensure_connected()
        try:
            url = f"{self._service_base}/fs/{self._sandbox_id}/readdir"
            resp = http.get(
                url,
                params={"dir": directory},
                headers=self._headers(),
                timeout=60,
                raise_for_status=False,
            )
            if resp.status_code >= 400:
                return []
            payload = resp.json() or {}
            items = payload.get("items", [])
            names: List[str] = []
            for it in items:
                names.append(
                    (
                        it
                        if isinstance(it, str)
                        else str(dict(it).get("name") or dict(it).get("path") or "")
                    ),
                )
            return [self._join(directory, n) for n in names if n]
        except Exception:
            return []

    def _is_dir_http(self, path: str) -> bool:
        # Try the stat endpoint first
        try:
            url = f"{self._service_base}/fs/{self._sandbox_id}/stat"
            resp = http.get(
                url,
                params={"path": path},
                headers=self._headers(),
                timeout=60,
                raise_for_status=False,
            )
            if resp.status_code < 400:
                stat = (resp.json() or {}).get("stat", {})
                return bool(stat.get("isDir"))
        except Exception:
            pass
        return False

    def _readdir_recursive(self, base: str) -> List[str]:
        stack = [base]
        files: List[str] = []
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            entries = self._readdir_once(cur)
            for path in entries:
                # Determine if directory
                is_dir = False
                if self._client is not None:
                    # Best-effort: CodeSandbox client may expose stat
                    try:
                        if hasattr(self._client.fs, "stat"):
                            st = self._client.fs.stat(path)  # type: ignore[attr-defined]
                            is_dir = bool(
                                getattr(st, "isDir", False)
                                or (isinstance(st, dict) and st.get("isDir")),
                            )
                        else:
                            is_dir = False
                    except Exception:
                        is_dir = False
                else:
                    is_dir = self._is_dir_http(path)
                if is_dir:
                    stack.append(path)
                else:
                    files.append(path)
        return files

    def iter_files(self, root: Optional[str] = None) -> Iterable[FileReference]:
        base = root or "/"
        for path in self._readdir_recursive(base):
            name = path.rsplit("/", 1)[-1]
            try:
                raw = self._client.fs.readFile(path) if self._client is not None else b""  # type: ignore[attr-defined]
                size = len(bytes(raw)) if raw is not None else None
            except Exception:
                raw, size = None, None
            yield FileReference(
                path=path,
                name=name,
                provider=self.name,
                uri=f"{self.uri_name}://{self._sandbox_id}{path}",
                size_bytes=size,
                modified_at=None,
                mime_type=None,
            )

    def get_file(self, path: str) -> FileReference:
        p = path if str(path).startswith("/") else self._join("/", path)
        name = p.rsplit("/", 1)[-1]
        size = None
        if self._client is not None:
            try:
                raw = self._client.fs.readFile(p)  # type: ignore[attr-defined]
                size = len(bytes(raw)) if raw is not None else None
            except Exception:
                size = None
        return FileReference(
            path=p,
            name=name,
            provider=self.name,
            uri=f"{self.uri_name}://{self._sandbox_id}{p}",
            size_bytes=size,
        )

    def exists(self, path: str) -> bool:
        """Check if a file exists in the CodeSandbox workspace."""
        p = path if str(path).startswith("/") else self._join("/", path)
        try:
            if self._client is not None and hasattr(self._client.fs, "stat"):
                stat = self._client.fs.stat(p)  # type: ignore[attr-defined]
                is_file = not bool(
                    getattr(stat, "isDir", False)
                    or (isinstance(stat, dict) and stat.get("isDir")),
                )
                return is_file
            else:
                # HTTP fallback - try to stat
                self._ensure_connected()

                url = f"{self._service_base}/fs/{self._sandbox_id}/stat"
                resp = http.get(
                    url,
                    params={"path": p},
                    headers=self._headers(),
                    timeout=60,
                    raise_for_status=False,
                )
                if resp.status_code < 400:
                    stat = (resp.json() or {}).get("stat", {})
                    return not bool(stat.get("isDir"))
                return False
        except Exception:
            return False

    def list(self, root: Optional[str] = None) -> List[str]:
        """List all file paths in the CodeSandbox workspace."""
        try:
            return [ref.path.lstrip("/") for ref in self.iter_files(root)]
        except Exception:
            return []

    def open_bytes(self, path: str) -> bytes:
        p = path if str(path).startswith("/") else self._join("/", path)
        if self._client is not None and hasattr(self._client, "fs"):
            raw = self._client.fs.readFile(p)  # type: ignore[attr-defined]
            return bytes(raw) if not isinstance(raw, (bytes, bytearray)) else bytes(raw)
        # HTTP fallback
        self._ensure_connected()
        url = f"{self._service_base}/fs/{self._sandbox_id}/readFile"
        resp = http.get(
            url,
            params={"path": p},
            headers=self._headers(),
            timeout=120,
            raise_for_status=False,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"CodeSandbox readFile failed: {resp.status_code}")
        return resp.content

    def export_file(self, path: str, destination_dir: str) -> str:
        """Export (download) a file from CodeSandbox workspace to local destination directory.

        Uses the CodeSandbox download API when available for efficient file transfer.
        """
        from pathlib import Path as _PathLib
        import urllib.request

        p = path if str(path).startswith("/") else self._join("/", path)

        # Try using the download API if available (generates a download URL)
        file_bytes = None
        if self._client is not None and hasattr(self._client.fs, "download"):
            try:
                # Generate download URL (valid for 5 minutes per SDK docs)
                download_info = self._client.fs.download(p)  # type: ignore[attr-defined]
                download_url = (
                    download_info.get("downloadUrl")
                    if isinstance(download_info, dict)
                    else getattr(download_info, "downloadUrl", None)
                )

                if download_url:
                    # Download from the generated URL
                    with urllib.request.urlopen(download_url) as response:
                        file_bytes = response.read()
            except Exception:
                # Fall back to readFile if download fails
                pass

        # Fallback to readFile if download API not available or failed
        if file_bytes is None:
            file_bytes = self.open_bytes(path)

        # Create destination directory structure
        dest_dir = _PathLib(destination_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Preserve the original filename
        dest_path = dest_dir / path.lstrip("/")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Write bytes to destination file
        dest_path.write_bytes(file_bytes)

        return str(dest_path)

    def export_directory(self, path: str, destination_dir: str) -> List[str]:
        """Export (download) all files from a directory in CodeSandbox workspace.

        Can use the download API for directories to get a zip file (optimization opportunity).
        """
        from pathlib import Path as _PathLib
        import urllib.request
        import zipfile
        import tempfile

        exported: List[str] = []

        # Try using directory download API if available (returns zip file)
        if self._client is not None and hasattr(self._client.fs, "download"):
            try:
                p = path if str(path).startswith("/") else self._join("/", path)
                download_info = self._client.fs.download(p)  # type: ignore[attr-defined]
                download_url = (
                    download_info.get("downloadUrl")
                    if isinstance(download_info, dict)
                    else getattr(download_info, "downloadUrl", None)
                )

                if download_url:
                    # Download zip file
                    with urllib.request.urlopen(download_url) as response:
                        zip_data = response.read()

                    # Extract zip to destination
                    with tempfile.NamedTemporaryFile(
                        suffix=".zip",
                        delete=False,
                    ) as tmp_zip:
                        tmp_zip.write(zip_data)
                        tmp_zip_path = tmp_zip.name

                    try:
                        dest_dir_path = _PathLib(destination_dir)
                        dest_dir_path.mkdir(parents=True, exist_ok=True)

                        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
                            zip_ref.extractall(dest_dir_path)
                            # Return list of extracted file paths
                            for member in zip_ref.namelist():
                                if not member.endswith("/"):  # Skip directories
                                    exported.append(str(dest_dir_path / member))
                    finally:
                        import os

                        try:
                            os.unlink(tmp_zip_path)
                        except Exception:
                            pass

                    return exported
            except Exception:
                # Fall back to individual file export
                pass

        # Fallback: export files individually
        try:
            for file_ref in self.iter_files(path):
                try:
                    exported_path = self.export_file(file_ref.path, destination_dir)
                    exported.append(exported_path)
                except Exception:
                    continue
        except Exception:
            pass

        return exported

    def rename(self, path: str, new_name: str) -> FileReference:
        old_path = path if str(path).startswith("/") else self._join("/", path)
        parent = old_path.rsplit("/", 1)[0] or "/"
        new_path = self._join(parent, new_name)
        if self._client is not None and hasattr(self._client, "fs"):
            self._client.fs.rename(old_path, new_path)  # type: ignore[attr-defined]
        else:
            self._ensure_connected()
            url = f"{self._service_base}/fs/{self._sandbox_id}/rename"
            resp = http.post(
                url,
                json={"oldPath": old_path, "newPath": new_path},
                headers=self._headers(),
                timeout=60,
                raise_for_status=False,
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"CodeSandbox rename failed: {resp.status_code} {resp.text[:200]}",
                )
        return FileReference(
            path=new_path,
            name=new_name,
            provider=self.name,
            uri=f"{self.uri_name}://{self._sandbox_id}{new_path}",
        )

    def move(self, path: str, new_parent_path: str) -> FileReference:
        src = path if str(path).startswith("/") else self._join("/", path)
        name = src.rsplit("/", 1)[-1]
        dst_parent = (
            new_parent_path
            if str(new_parent_path).startswith("/")
            else self._join("/", new_parent_path)
        )
        new_path = self._join(dst_parent, name)
        if self._client is not None and hasattr(self._client, "fs"):
            # Rename effectively moves within workspace
            self._client.fs.rename(src, new_path)  # type: ignore[attr-defined]
        else:
            self._ensure_connected()
            url = f"{self._service_base}/fs/{self._sandbox_id}/move"
            resp = http.post(
                url,
                json={"oldPath": src, "newParentPath": dst_parent},
                headers=self._headers(),
                timeout=60,
                raise_for_status=False,
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"CodeSandbox move failed: {resp.status_code} {resp.text[:200]}",
                )
        return FileReference(
            path=new_path,
            name=name,
            provider=self.name,
            uri=f"{self.uri_name}://{self._sandbox_id}{new_path}",
        )

    def delete(self, path: str) -> None:
        """Delete a file from the CodeSandbox workspace."""
        if not self._caps.can_delete:
            raise PermissionError("Delete not permitted by backend policy")
        p = path if str(path).startswith("/") else self._join("/", path)
        if self._client is not None and hasattr(self._client.fs, "unlink"):
            self._client.fs.unlink(p)  # type: ignore[attr-defined]
        else:
            self._ensure_connected()
            url = f"{self._service_base}/fs/{self._sandbox_id}/delete"
            resp = http.post(
                url,
                json={"path": p},
                headers=self._headers(),
                timeout=60,
                raise_for_status=False,
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"CodeSandbox delete failed: {resp.status_code} {resp.text[:200]}",
                )
