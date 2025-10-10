from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

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
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    modified_at: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class FolderReference(BaseModel):
    """A folder reference (when adapters expose folder-level introspection)."""

    path: str
    name: str
    extra: Dict[str, Any] = Field(default_factory=dict)


OperationType = Literal["rename", "move"]


class OperationAction(BaseModel):
    """A single planned operation (rename/move) with a rationale."""

    operation: OperationType
    target_id_or_path: str
    new_name: Optional[str] = None
    new_parent_path: Optional[str] = None
    reason: str = ""


class OperationPlan(BaseModel):
    """A set of actions that together satisfy an organizational goal."""

    goal: str
    actions: List[OperationAction] = Field(default_factory=list)
    can_execute: bool = False
