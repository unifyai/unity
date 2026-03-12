from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Iterator, List

from unity.file_manager.types.filesystem import (
    FileSystemCapabilities,
    FileReference,
    FolderReference,
)


class BaseFileSystemAdapter(ABC):
    """Thin, synchronous adapter around a concrete filesystem backend.

    Responsibilities:
    - Discover files/folders and basic metadata
    - Fetch file bytes for parsing/analysis
    - Perform guarded mutations (rename/move) when supported

    No LLM. No parsing. Pure IO.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier (stable over process lifetime)."""

    @property
    @abstractmethod
    def capabilities(self) -> FileSystemCapabilities:
        """Advertised capabilities of this backend."""

    # Listing / lookup ----------------------------------------------------- #
    @abstractmethod
    def iter_files(self, root: Optional[str] = None) -> Iterable[FileReference]:
        """Yield file references under an optional root path."""

    @abstractmethod
    def get_file(self, path: str) -> FileReference:
        """Return a single file reference by backend id or canonical path."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file exists in the filesystem.

        Each adapter should implement this using their underlying filesystem API.

        Parameters
        ----------
        path : str
            Backend-specific file identifier or canonical path.

        Returns
        -------
        bool
            True if the file exists, False otherwise.
        """

    @abstractmethod
    def list(self, root: Optional[str] = None) -> List[str]:
        """List all file paths in the filesystem.

        Each adapter should implement this using their underlying filesystem API.
        Paths should be returned without leading slashes for display purposes.

        Parameters
        ----------
        root : str | None
            Optional root path to list files under.

        Returns
        -------
        list[str]
            List of file paths without leading slashes.
        """

    # Content --------------------------------------------------------------- #
    @abstractmethod
    def open_bytes(self, path: str) -> bytes:
        """Return raw bytes for a file."""

    @abstractmethod
    def export_file(self, path: str, destination_dir: str) -> str:
        """
        Export a file from the underlying filesystem to a local destination directory.

        This method copies/downloads the file from the adapter's filesystem to a local
        directory, preserving the original filename. This is used by the FileManager
        to bring files into a local temp directory for parsing.

        Parameters
        ----------
        path : str
            The file identifier or path in the adapter's filesystem.
        destination_dir : str
            Local directory path where the file should be exported.

        Returns
        -------
        str
            Full path to the exported file in the destination directory.

        Raises
        ------
        FileNotFoundError
            If the source file doesn't exist.
        PermissionError
            If export is not allowed or destination is not writable.

        Examples
        --------
        >>> adapter.export_file("document.pdf", "/tmp/export_dir")
        "/tmp/export_dir/document.pdf"
        """

    @abstractmethod
    def export_directory(self, path: str, destination_dir: str) -> List[str]:
        """
        Export all files from a directory in the underlying filesystem to a local destination.

        Each adapter should implement this using their underlying filesystem API,
        optimizing for batch operations where possible (e.g., zip download).

        Parameters
        ----------
        path : str
            The directory identifier or path in the adapter's filesystem.
        destination_dir : str
            Local directory path where files should be exported.

        Returns
        -------
        list[str]
            List of full paths to exported files in the destination directory.
        """

    # Mutations (capability-guarded) --------------------------------------- #
    def rename(
        self,
        path: str,
        new_name: str,
    ) -> FileReference:  # pragma: no cover - default not supported
        raise NotImplementedError("Rename not supported by this adapter")

    def move(
        self,
        path: str,
        new_parent_path: str,
    ) -> FileReference:  # pragma: no cover - default not supported
        raise NotImplementedError("Move not supported by this adapter")

    def delete(self, path: str) -> None:  # pragma: no cover - default not supported
        """
        Delete a file from the underlying filesystem.

        Parameters
        ----------
        path : str
            Backend-specific file identifier or canonical path.

        Raises
        ------
        NotImplementedError
            If deletion is not supported by this adapter.
        PermissionError
            If the file is protected and cannot be deleted.
        FileNotFoundError
            If the file doesn't exist.
        """
        raise NotImplementedError("Delete not supported by this adapter")

    # Optional: folder-level iteration for adapters that expose folders
    def iter_folders(
        self,
        root: Optional[str] = None,
    ) -> Iterator[FolderReference]:  # pragma: no cover - default not supported
        raise NotImplementedError("iter_folders not supported by this adapter")

    # ------------------------- High-level import APIs ------------------------- #
    # These are optional convenience methods. Adapters that support persistent
    # storage should implement them; others may raise NotImplementedError.

    def import_file(
        self,
        source_path: str,
    ) -> str:  # pragma: no cover - default not supported
        """Import/copy a single file into the adapter's persistent store and return a display name."""
        raise NotImplementedError("import_file not supported by this adapter")

    def import_directory(
        self,
        directory: str,
    ) -> List[str]:  # pragma: no cover - default not supported
        """Import/copy all files in a directory (non-recursive). Return display names."""
        raise NotImplementedError("import_directory not supported by this adapter")

    def register_existing_file(
        self,
        path: str,
        *,
        display_name: Optional[str] = None,
        protected: bool = False,
    ) -> str:  # pragma: no cover - default not supported
        """Register an already-existing file for read-only access without copying."""
        raise NotImplementedError(
            "register_existing_file not supported by this adapter",
        )

    def is_protected(
        self,
        display_name: str,
    ) -> bool:  # pragma: no cover - default not supported
        return False

    def save_attachment(
        self,
        attachment_id: str,
        filename: str,
        contents: bytes,
    ) -> str:  # pragma: no cover - default not supported
        """Save bytes to the Attachments directory as {attachment_id}_{filename} and return the display name."""
        raise NotImplementedError(
            "save_attachment not supported by this adapter",
        )

    def resolve_display_name(
        self,
        display_name: str,
    ) -> Optional[str]:  # pragma: no cover - default not supported
        """Best-effort resolve of a display name to an absolute path in the backing store."""
        return None
