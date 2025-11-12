"""Pydantic data model for File records stored by the FileManager."""

from __future__ import annotations

from typing import Dict, List, Optional, Literal

from pydantic import BaseModel, Field
from unity.file_manager.parser.types.enums import FileFormat  # unified enums

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
    source_uri: str | None = Field(
        default=None,
        description="Canonical provider URI for the file (e.g., local:///abs/path, gdrive://fileId).",
    )
    source_provider: str | None = Field(
        default=None,
        description="Provider/adapter name (e.g., Local, GoogleDrive, CodeSandbox).",
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
    # Unified type/format fields
    file_format: Optional[FileFormat] = Field(
        default=None,
        description="Canonical file format (e.g., pdf, docx, xlsx, csv)",
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

    # Ingestion layout metadata
    ingest_mode: Literal["per_file", "unified"] = Field(
        default="per_file",
        description="Ingestion layout mode used for this file ('per_file' or 'unified').",
    )
    unified_label: Optional[str] = Field(
        default=None,
        description="Unified label bucket name when ingest_mode == 'unified' (otherwise None).",
    )
    table_ingest: bool = Field(
        default=True,
        description="Whether tables were ingested into per-file table contexts for this file.",
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

    @staticmethod
    def to_file_record_entry(
        *,
        file_path: str,
        source_uri: Optional[str],
        source_provider: Optional[str],
        result: Dict[str, object],
        ingest_mode: Literal["per_file", "unified"],
        unified_label: Optional[str],
        table_ingest: bool,
    ) -> Dict[str, object]:
        """
        Build a storage entry for FileRecords/<alias> from parse result and identity.
        """
        # TODO: Remove hack for the sheet ingestion
        return {
            "file_path": file_path,
            "source_uri": source_uri,
            "source_provider": source_provider,
            "status": result.get("status"),
            "error": result.get("error"),
            "summary": result.get("summary"),
            "file_format": result.get("file_format"),
            "file_size": result.get("file_size"),
            "total_records": result.get("total_records"),
            "processing_time": result.get("processing_time"),
            "created_at": result.get("created_at"),
            "modified_at": result.get("modified_at"),
            "confidence_score": result.get("confidence_score"),
            "key_topics": result.get("key_topics"),
            "named_entities": result.get("named_entities"),
            "content_tags": result.get("content_tags"),
            "ingest_mode": ingest_mode,
            "unified_label": (unified_label if ingest_mode == "unified" else None),
            "table_ingest": bool(table_ingest),
        }


class FileIdentity(BaseModel):
    """
    Consolidated identity for a file managed by FileManager.

    This model captures source identity and ingest layout so callers can
    derive correct contexts for Content and Tables for both per_file and
    unified ingestion modes.
    """

    file_path: str = Field(
        ...,
        description="Filesystem path or logical display name of the file.",
    )
    source_provider: Optional[str] = Field(
        default=None,
        description="Provider/adapter name (e.g., Local, GoogleDrive, CodeSandbox).",
    )
    source_uri: Optional[str] = Field(
        default=None,
        description="Canonical provider URI (e.g., local:///abs/path, gdrive://fileId).",
    )
    ingest_mode: Literal["per_file", "unified"] = Field(
        default="per_file",
        description="Ingestion layout mode for this file.",
    )
    unified_label: Optional[str] = Field(
        default=None,
        description="Unified bucket label when ingest_mode=='unified'.",
    )
    table_ingest: bool = Field(
        default=True,
        description="True when tables are persisted in per‑file table contexts.",
    )
    file_format: Optional[FileFormat] = Field(
        default=None,
        description="Canonical file format (e.g., pdf, docx, xlsx, csv)",
    )


class FileContent(BaseModel):
    """
    Per-file context row schema used for storing flattened, hierarchical
    records extracted from a single file. The per-file context lives under
    "<base>/Files/<alias>/<file_path>/Content/" and uses counters to represent the
    document→section→paragraph→sentence hierarchy. Extracted tabular content
    for a file is stored under separate per-table contexts (no predefined
    fields) at "<base>/Files/<alias>/<file_path>/Tables/<table>".
    """

    # Unique id within the per-file content context (row identifier)
    row_id: int = Field(
        default=UNASSIGNED,
        ge=UNASSIGNED,
        description="Unique row id for Content context",
    )

    # Foreign keys / identifiers
    file_id: int = Field(
        default=UNASSIGNED,
        ge=UNASSIGNED,
        description="Foreign key to the FileRecord (global index)",
    )

    # Consolidated hierarchical identifier map (document/section/paragraph/sentence/table/image)
    # Examples:
    #  - document row: {"document": 0}
    #  - section row:  {"document": 0, "section": 2}
    #  - paragraph row:{"document": 0, "section": 2, "paragraph": 1}
    #  - sentence row: {"document": 0, "section": 2, "paragraph": 1, "sentence": 3}
    #  - table row:    {"document": 0, "section": 2, "table": 0}
    #  - image row:    {"document": 0, "section": 2, "image": 0}
    content_id: Optional[Dict[str, int]] = Field(
        default=None,
        description="Consolidated hierarchical id map for this row",
    )

    # Core fields
    content_type: str = Field(
        ...,
        description="'document' | 'section' | 'paragraph' | 'sentence' | 'image' | 'table'",
    )
    title: str | None = Field(default=None, description="Heading or inferred title")
    content_text: str | None = Field(default=None, description="Original raw text")
    summary: str | None = Field(
        default=None,
        description="Rich summary used for embeddings",
    )

    @staticmethod
    def to_file_content_entries(
        *,
        file_id: int,
        rows: List[Dict[str, object]],
        id_layout: Literal["map", "columns", "string"] = "map",
    ) -> List[Dict[str, object]]:
        """
        Build per-file Content entries from parser rows, attaching file_id and
        dropping fields not represented in the schema.
        """
        out: List[Dict[str, object]] = []
        allowed_core = {
            "content_type",
            "title",
            "summary",
            "content_text",
            "content_id",
        }
        legacy_cols = {
            "document_id",
            "section_id",
            "paragraph_id",
            "sentence_id",
            "image_id",
            "table_id",
        }
        for rec in rows or []:
            base = {k: v for k, v in rec.items() if k in allowed_core}
            if id_layout == "columns":
                for k in legacy_cols:
                    if k in rec:
                        base[k] = rec[k]
            base["file_id"] = int(file_id)
            out.append(base)
        return out
