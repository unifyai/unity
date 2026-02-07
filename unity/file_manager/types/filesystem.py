from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class FileSystemCapabilities(BaseModel):
    """Capabilities advertised by a filesystem backend.

    Read operations should always be supported. Mutations are opt-in per backend.
    """

    can_read: bool = True
    can_rename: bool = False
    can_move: bool = False
    can_delete: bool = False
    can_create: bool = False


class FileReference(BaseModel):
    """A file reference as returned by a filesystem adapter."""

    path: str = Field(
        ...,
        description="Canonical path within the backend (leading '/' when applicable)",
    )
    name: str
    provider: Optional[str] = Field(
        default=None,
        description="Stable provider/adapter name (e.g., 'Local').",
    )
    uri: Optional[str] = Field(
        default=None,
        description="Canonical provider URI (e.g., local:///abs/path, gdrive://fileId)",
    )
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    modified_at: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class FolderReference(BaseModel):
    """A folder reference (when adapters expose folder-level introspection)."""

    path: str
    name: str
    extra: Dict[str, Any] = Field(default_factory=dict)
