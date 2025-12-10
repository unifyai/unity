"""Pydantic data model for File records stored by the FileManager."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, Union

from pydantic import BaseModel, Field
from unity.file_manager.parser.types.enums import FileFormat  # unified enums

UNASSIGNED = -1


class ParsedFile(BaseModel):
    """
    Strictly typed output from file parsing.

    This model captures the complete output from Document.to_parse_result() and
    can be used to derive FileRecord entries and Content rows for ingestion.
    It replaces the loosely typed Dict[str, Any] that was previously used.

    Attributes
    ----------
    file_path : str
        The file path or identifier.
    status : Literal["success", "error"]
        Processing status.
    error : str | None
        Error message if status is 'error'.
    records : list[dict]
        Parsed content records (flexible schema based on parser output).
    full_text : str
        Full text content extracted from the file.
    summary : str
        Document summary.
    file_format : str | None
        File format (e.g., 'pdf', 'docx').
    file_size : int | None
        File size in bytes.
    total_records : int
        Number of parsed content records.
    processing_time : float | None
        Parsing time in seconds.
    created_at : str | None
        File creation timestamp (ISO-8601).
    modified_at : str | None
        File modification timestamp (ISO-8601).
    confidence_score : float | None
        Parser confidence score.
    key_topics : list[str]
        Extracted key topics.
    named_entities : dict
        Named entities by type.
    content_tags : list[str]
        Content tags.
    """

    file_path: str = Field(
        ...,
        description="The file path or identifier.",
    )
    status: Literal["success", "error"] = Field(
        default="success",
        description="Processing status.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'error'.",
    )
    records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Parsed content records (flexible schema).",
    )
    full_text: str = Field(
        default="",
        description="Full text content extracted from the file.",
    )
    summary: str = Field(
        default="",
        description="Document summary.",
    )
    file_format: Optional[Union[str, FileFormat]] = Field(
        default=None,
        description="File format (e.g., 'pdf', 'docx').",
    )
    file_size: Optional[int] = Field(
        default=None,
        description="File size in bytes.",
    )
    total_records: int = Field(
        default=0,
        description="Number of parsed content records.",
    )
    processing_time: Optional[float] = Field(
        default=None,
        description="Parsing time in seconds.",
    )
    created_at: Optional[str] = Field(
        default=None,
        description="File creation timestamp (ISO-8601).",
    )
    modified_at: Optional[str] = Field(
        default=None,
        description="File modification timestamp (ISO-8601).",
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="Parser confidence score.",
    )
    key_topics: List[str] = Field(
        default_factory=list,
        description="Extracted key topics.",
    )
    named_entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Named entities by type.",
    )
    content_tags: List[str] = Field(
        default_factory=list,
        description="Content tags.",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for backwards compatibility with code expecting dicts."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedFile":
        """Create from dict for backwards compatibility."""
        return cls(**data)

    @classmethod
    def error_result(cls, file_path: str, error: str) -> "ParsedFile":
        """Create an error result."""
        return cls(
            file_path=file_path,
            status="error",
            error=error,
        )


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
        result: Union["ParsedFile", Dict[str, Any]],
        ingest_mode: Literal["per_file", "unified"],
        unified_label: Optional[str],
        table_ingest: bool,
    ) -> Dict[str, object]:
        """
        Build a storage entry for FileRecords/<alias> from parse result and identity.

        Parameters
        ----------
        file_path : str
            The file path.
        source_uri : str | None
            Canonical provider URI.
        source_provider : str | None
            Provider name.
        result : ParsedFile | dict
            Parse result (ParsedFile model or legacy dict).
        ingest_mode : Literal["per_file", "unified"]
            Ingestion mode.
        unified_label : str | None
            Unified label when ingest_mode is 'unified'.
        table_ingest : bool
            Whether tables are ingested.

        Returns
        -------
        dict
            Storage entry for FileRecords.
        """
        # Handle both ParsedFile and dict for backwards compatibility
        if isinstance(result, ParsedFile):
            data = result.to_dict()
        else:
            data = result

        return {
            "file_path": file_path,
            "source_uri": source_uri,
            "source_provider": source_provider,
            "status": data.get("status"),
            "error": data.get("error"),
            "summary": data.get("summary"),
            "file_format": data.get("file_format"),
            "file_size": data.get("file_size"),
            "total_records": data.get("total_records"),
            "processing_time": data.get("processing_time"),
            "created_at": data.get("created_at"),
            "modified_at": data.get("modified_at"),
            "confidence_score": data.get("confidence_score"),
            "key_topics": data.get("key_topics"),
            "named_entities": data.get("named_entities"),
            "content_tags": data.get("content_tags"),
            "ingest_mode": ingest_mode,
            "unified_label": (unified_label if ingest_mode == "unified" else None),
            "table_ingest": bool(table_ingest),
        }


class FileInfo(BaseModel):
    """
    Comprehensive file information combining filesystem status and ingest identity.

    This model provides a complete picture of a file's state across both the
    raw filesystem and the parsed/indexed state. Use this to understand:
    - Whether the file exists on disk (filesystem_exists)
    - Whether it has been indexed (indexed_exists)
    - Its parse status (parsed_status)
    - Its ingest layout (ingest_mode, unified_label, table_ingest)
    """

    # Filesystem status
    file_path: str = Field(
        ...,
        description="Filesystem path as queried.",
    )
    filesystem_exists: bool = Field(
        default=False,
        description="True if the file currently exists on disk.",
    )

    # Index status
    indexed_exists: bool = Field(
        default=False,
        description="True if the file has a row in FileRecords index.",
    )
    parsed_status: Optional[Literal["success", "error"]] = Field(
        default=None,
        description="Parse status: 'success', 'error', or None if not indexed.",
    )

    # Identity fields
    source_provider: Optional[str] = Field(
        default=None,
        description="Provider/adapter name (e.g., Local, GoogleDrive, CodeSandbox).",
    )
    source_uri: Optional[str] = Field(
        default=None,
        description="Canonical provider URI (e.g., local:///abs/path, gdrive://fileId).",
    )

    # Ingest layout
    ingest_mode: Literal["per_file", "unified"] = Field(
        default="per_file",
        description="Ingestion layout: 'per_file' or 'unified'.",
    )
    unified_label: Optional[str] = Field(
        default=None,
        description="Unified bucket label when ingest_mode=='unified'.",
    )
    table_ingest: bool = Field(
        default=True,
        description="True when tables are in per-file /Tables/ contexts.",
    )

    # File metadata (when indexed)
    file_format: Optional[FileFormat] = Field(
        default=None,
        description="Canonical file format (e.g., pdf, docx, xlsx, csv).",
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
