"""Local filesystem adapter with optional managed VM file sync."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

from unity.file_manager.filesystem_adapters.base import BaseFileSystemAdapter
from unity.file_manager.types.filesystem import FileSystemCapabilities, FileReference

if TYPE_CHECKING:
    from unity.file_manager.sync import SyncManager


class LocalFileSystemAdapter(BaseFileSystemAdapter):
    """Adapter for a local directory tree with optional VM sync.

    This adapter operates on ~ (home directory) for user files. When sync is
    enabled and a managed VM is configured (via SESSION_DETAILS.desktop_url),
    the home directory is synchronized with the VM via rclone SFTP.

    Sync lifecycle:
    - Job start: Bidirectional sync with --resync (start_sync → bisync)
    - File write: Push changed file to VM (notify_file_write)
    - Periodic: Bidirectional sync for remote changes (bisync every 30s)
    - Job end: Final bisync to VM (stop_sync → bisync)
    """

    def __init__(
        self,
        root: str | None = None,
        *,
        enable_sync: bool = False,
    ):
        """Initialize LocalFileSystemAdapter.

        Parameters
        ----------
        root : str | None, default None
            Root directory for file operations. Defaults to ~ (home).
        enable_sync : bool, default False
            Whether to enable VM file sync. Actual sync only occurs if
            SESSION_DETAILS.desktop_url is configured.
        """
        if root is None:
            from unity.file_manager.settings import get_local_root

            root = get_local_root()

        self._root = Path(root).expanduser().resolve()

        # Ensure root directory exists
        self._root.mkdir(parents=True, exist_ok=True)

        self._caps = FileSystemCapabilities(
            can_read=True,
            can_rename=True,
            can_move=True,
            can_delete=True,
        )

        # Sync component (lazy initialization)
        self._enable_sync = enable_sync
        self._sync_manager: Optional["SyncManager"] = None

    @property
    def name(self) -> str:
        return "Local"

    @property
    def uri_name(self) -> str:
        return "local"

    @property
    def capabilities(self) -> FileSystemCapabilities:
        return self._caps

    # ----------------------- Sync Properties ----------------------- #

    @property
    def sync_enabled(self) -> bool:
        """Whether file sync is configured and enabled."""
        if self._sync_manager is None:
            return False
        return self._sync_manager.enabled

    @property
    def sync_started(self) -> bool:
        """Whether file sync has been started."""
        if self._sync_manager is None:
            return False
        return self._sync_manager._started

    # ----------------------- Core IO Methods ----------------------- #

    def _abspath(self, p: str) -> Path:
        # Support both absolute and root-relative inputs.
        # Absolute paths are allowed anywhere on the local filesystem.
        q = Path(p)
        if not q.is_absolute():
            q = (self._root / p.lstrip("/")).resolve()
            return q
        return q.resolve()

    def iter_files(self, root: Optional[str] = None) -> Iterable[FileReference]:
        base = self._abspath(root or ".")
        if not base.exists():
            return []
        for p in base.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(self._root)).replace("\\", "/")
                yield FileReference(
                    path=("/" + rel if not rel.startswith("/") else rel),
                    name=p.name,
                    provider=self.name,
                    uri=f"{self.uri_name}://{p.resolve().as_posix().lstrip('/')}",
                    size_bytes=p.stat().st_size,
                    modified_at=None,
                    mime_type=None,
                )

    def get_file(self, path: str) -> FileReference:
        p = self._abspath(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(path)
        rel = str(p.relative_to(self._root)).replace("\\", "/")
        return FileReference(
            path=("/" + rel if not rel.startswith("/") else rel),
            name=p.name,
            provider=self.name,
            uri=f"{self.uri_name}://{p.resolve().as_posix().lstrip('/')}",
            size_bytes=p.stat().st_size,
            modified_at=None,
            mime_type=None,
        )

    def exists(self, path: str) -> bool:
        """Check if a file exists (optimized for local filesystem)."""
        try:
            p = self._abspath(path)
            return p.exists() and p.is_file()
        except (PermissionError, FileNotFoundError):
            return False
        except Exception:
            return False

    def list(self, root: Optional[str] = None) -> List[str]:
        """List all file paths in the local filesystem."""
        try:
            return [ref.path.lstrip("/") for ref in self.iter_files(root)]
        except Exception:
            return []

    def open_bytes(self, path: str) -> bytes:
        p = self._abspath(path)
        return p.read_bytes()

    def export_file(self, path: str, destination_dir: str) -> str:
        """Export (copy) a file from local filesystem to destination directory."""
        source_path = self._abspath(path)
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        dest_dir = Path(destination_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Preserve the original filename
        dest_path = dest_dir / path.lstrip("/")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_path, dest_path)

        return str(dest_path)

    def export_directory(self, path: str, destination_dir: str) -> List[str]:
        """Export (copy) all files from a directory to destination directory."""
        exported: List[str] = []
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
        if not self._caps.can_rename:
            raise PermissionError("Rename not permitted by backend policy")
        p = self._abspath(path)
        dest = p.with_name(new_name)
        p.rename(dest)
        # Handle absolute paths that may be outside adapter root
        try:
            rel = str(dest.relative_to(self._root)).replace("\\", "/")
            path_str = "/" + rel if not rel.startswith("/") else rel
        except ValueError:
            # Path is outside adapter root, use absolute path
            path_str = dest.as_posix()
        return FileReference(
            path=path_str,
            name=dest.name,
            provider=self.name,
            uri=f"{self.uri_name}://{dest.resolve().as_posix().lstrip('/')}",
        )

    def move(self, path: str, new_parent_path: str) -> FileReference:
        if not self._caps.can_move:
            raise PermissionError("Move not permitted by backend policy")
        p = self._abspath(path)
        new_parent = self._abspath(new_parent_path)
        new_parent.mkdir(parents=True, exist_ok=True)
        dest = new_parent / p.name
        p.rename(dest)
        # Handle absolute paths that may be outside adapter root
        try:
            rel = str(dest.relative_to(self._root)).replace("\\", "/")
            path_str = "/" + rel if not rel.startswith("/") else rel
        except ValueError:
            # Path is outside adapter root, use absolute path
            path_str = dest.as_posix()
        return FileReference(
            path=path_str,
            name=dest.name,
            provider=self.name,
            uri=f"{self.uri_name}://{dest.resolve().as_posix().lstrip('/')}",
        )

    def delete(self, path: str) -> None:
        if not self._caps.can_delete:
            raise PermissionError("Delete not permitted by backend policy")
        p = self._abspath(path)
        if not p.exists():
            raise FileNotFoundError(path)
        if not p.is_file():
            raise ValueError(f"Path is not a file: {path}")
        p.unlink()

    # ------------------------- High-level import APIs ------------------------- #

    @staticmethod
    def _unique_name(existing: set[str], desired: str) -> str:
        base = Path(desired).stem
        ext = Path(desired).suffix
        name = f"{base}{ext}"
        if name not in existing:
            return name
        i = 1
        while True:
            candidate = f"{base} ({i}){ext}"
            if candidate not in existing:
                return candidate
            i += 1

    def import_file(self, source_path: str) -> str:
        src = Path(source_path).expanduser().resolve()
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(str(src))
        # Copy into root with unique name
        try:
            existing = {p.name for p in self._root.iterdir() if p.is_file()}
        except Exception:
            existing = set()
        desired = src.name
        unique = self._unique_name(existing, desired)
        dest = (self._root / unique).resolve()
        shutil.copy2(src, dest)
        return unique

    def import_directory(self, directory: str) -> List[str]:
        p = Path(directory).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise NotADirectoryError(str(directory))
        added: List[str] = []
        for child in sorted(p.iterdir()):
            if child.is_file():
                try:
                    added.append(self.import_file(str(child)))
                except Exception:
                    continue
        return added

    def register_existing_file(
        self,
        path: str,
        *,
        display_name: Optional[str] = None,
        protected: bool = False,
    ) -> str:
        p = Path(path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(str(p))
        name = display_name or p.name
        try:
            existing = {x.name for x in self._root.iterdir() if x.is_file()}
        except Exception:
            existing = set()
        if name in existing and (self._root / name).resolve() != p:
            name = self._unique_name(existing, name)
        # If file is already under root with the same name, don't copy
        dest = (self._root / name).resolve()
        if dest != p:
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
            except Exception:
                # As best-effort registration, if copy fails and file already exists under dest, accept
                if not dest.exists():
                    raise
        # protected flag is advisory for this adapter; higher layers enforce
        return name

    def is_protected(self, display_name: str) -> bool:
        # Local adapter does not persist protection flags; always False
        return False

    def save_file_to_downloads(
        self,
        filename: str,
        contents: bytes,
        *,
        sync: bool = False,
    ) -> str:
        """Save bytes to Downloads directory.

        Parameters
        ----------
        filename : str
            Desired filename for the saved file.
        contents : bytes
            File contents.
        sync : bool, default False
            If True and sync is active, trigger sync to remote VM.
            Note: When True, this method schedules an async sync task.

        Returns
        -------
        str
            Relative path to saved file (e.g., "Downloads/report.pdf")
        """
        downloads_dir = (self._root / "Downloads").resolve()
        try:
            downloads_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        desired = Path(filename).name or "downloaded_file"
        try:
            existing = {p.name for p in downloads_dir.iterdir() if p.is_file()}
        except Exception:
            existing = set()
        unique = self._unique_name(existing, desired)
        target_path = downloads_dir / unique
        with open(target_path, "wb") as f:
            f.write(contents)

        relative_path = f"Downloads/{unique}"

        # Schedule sync if requested and sync is active
        if sync and self._sync_manager is not None and self._sync_manager._started:
            abs_path = str(self._root / relative_path)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._sync_manager.on_file_write(abs_path))
            except RuntimeError:
                # No running loop - sync will happen on next poll
                print(f"[LocalFS] No event loop for sync, will sync on next poll")

        return relative_path

    def resolve_display_name(self, display_name: str) -> Optional[str]:
        candidate = (self._root / display_name).expanduser().resolve()
        if candidate.exists():
            return str(candidate)
        # Check Downloads namespace
        dl = (self._root / "Downloads" / Path(display_name).name).resolve()
        if dl.exists():
            return str(dl)
        return None

    # ----------------------- Async Sync Methods ----------------------- #

    async def start_sync(self) -> bool:
        """Start file synchronization with managed VM.

        Called during manager initialization on job start.
        Creates the SyncManager lazily and initiates sync.

        Returns
        -------
        bool
            True if sync started successfully, False otherwise.
        """
        if not self._enable_sync:
            print("[LocalFS] Sync disabled by constructor flag")
            return False

        # Lazy create SyncManager to allow SESSION_DETAILS to be populated first
        if self._sync_manager is None:
            from unity.file_manager.sync import SyncManager

            self._sync_manager = SyncManager()

        if not self._sync_manager.enabled:
            print("[LocalFS] Sync not enabled (no desktop_url)")
            return False

        return await self._sync_manager.start()

    async def stop_sync(self) -> None:
        """Stop file synchronization with final sync to VM."""
        if self._sync_manager is not None:
            await self._sync_manager.stop()

    async def notify_file_write(self, path: str) -> None:
        """Notify sync manager of a file write.

        Parameters
        ----------
        path : str
            Absolute path to the written file.
        """
        if self._sync_manager is not None and self._sync_manager._started:
            await self._sync_manager.on_file_write(path)

    async def notify_file_delete(self, path: str) -> None:
        """Notify sync manager of a file deletion.

        Parameters
        ----------
        path : str
            Absolute path to the deleted file.
        """
        if self._sync_manager is not None and self._sync_manager._started:
            await self._sync_manager.on_file_delete(path)

    async def refresh_from_remote(self) -> bool:
        """Manually refresh files from remote VM.

        Useful before reading files that may have changed on desktop.

        Returns
        -------
        bool
            True if refresh succeeded, False otherwise.
        """
        if self._sync_manager is None or not self._sync_manager._started:
            return False
        result = await self._sync_manager.sync_remote_changes()
        return result.success
