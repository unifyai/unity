"""Pydantic data model for File records stored by the FileManager."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

UNASSIGNED = -1


class FileRecord(BaseModel):
    """
    Lightweight index row for a file in FileRecords/<alias>.

    This schema mirrors the FileManager table schema and should remain lean; the
    heavy parsed content lives under per-file contexts.
    """

    file_id: int = Field(
        default=UNASSIGNED,
        description="Unique identifier for the file",
        ge=UNASSIGNED,
    )

    # Core identification
    file_path: str = Field(
        ...,
        description="Filesystem path or display name of the file (adapter-specific).",
    )

    # Processing status
    status: str = Field(
        default="success",
        description="Processing status: 'success' or 'error'.",
    )

    # Error handling
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'error'.",
    )

    # Parsed content is no longer stored in the top-level File row. Records
    # are logged under per-file table contexts. Keep the schema lean here.

    # Flattened metadata fields
    file_type: Optional[str] = Field(
        default=None,
        description="Type/format of the file",
    )
    file_size: Optional[int] = Field(
        default=None,
        description="Size of the file in bytes",
    )
    total_records: Optional[int] = Field(
        default=None,
        description="Total parsed content rows for this file",
    )
    processing_time: Optional[float] = Field(
        default=None,
        description="Parsing time in seconds",
    )
    created_at: Optional[str] = Field(
        default=None,
        description="Creation timestamp (ISO-8601)",
    )
    modified_at: Optional[str] = Field(
        default=None,
        description="Last modified timestamp (ISO-8601)",
    )

    # Summary and semantic annotations
    confidence_score: Optional[float] = Field(
        default=None,
        description="Overall confidence score from the parser/LLM",
    )
    key_topics: List[str] = Field(
        default_factory=list,
        description="Key topics detected in the document",
    )
    named_entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Named entities grouped by type",
    )
    content_tags: List[str] = Field(
        default_factory=list,
        description="Freeform content tags",
    )
    summary: Optional[str] = Field(
        default="",
        description="Short summary extracted from the document",
    )


class File(BaseModel):
    """
    Per-file context row schema used for storing flattened, hierarchical
    records extracted from a single file. The per-file context lives under
    "<base>/File/<alias>/<file_path>" and uses counters to represent the
    document→section→paragraph→sentence hierarchy. Extracted tabular content
    for a file is stored under separate per-table contexts (no predefined
    fields) at "<base>/File/<alias>/<file_path>/Tables/<table>".
    """

    # Foreign keys / identifiers
    file_id: int = Field(
        default=UNASSIGNED,
        ge=UNASSIGNED,
        description="Foreign key to the FileRecord (global index)",
    )
    # Unique id within the per-file content context
    content_id: int = Field(
        default=UNASSIGNED,
        ge=UNASSIGNED,
        description="Unique content row id",
    )

    # Hierarchy counters
    document_id: int | None = Field(
        default=None,
        description="Document id (root counter)",
    )
    section_id: int | None = Field(
        default=None,
        description="Section id (scoped to document)",
    )
    image_id: int | None = Field(
        default=None,
        description="Image id (scoped to section)",
    )
    table_id: int | None = Field(
        default=None,
        description="Table id (scoped to section)",
    )
    paragraph_id: int | None = Field(
        default=None,
        description="Paragraph id (scoped to section)",
    )
    sentence_id: int | None = Field(
        default=None,
        description="Sentence id (scoped to paragraph)",
    )

    # Core fields
    file_path: str = Field(..., description="File path")
    content_type: str = Field(
        ...,
        description="'document' | 'section' | 'paragraph' | 'sentence' | 'image' | 'table'",
    )
    title: str | None = Field(default=None, description="Heading or inferred title")
    summary: str | None = Field(
        default=None,
        description="Rich summary used for embeddings",
    )
    content_text: str | None = Field(default=None, description="Original raw text")
