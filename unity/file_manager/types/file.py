"""Pydantic data model for File records stored by the FileManager."""

from __future__ import annotations

from typing import Dict, List, Optional, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field
from unity.file_manager.file_parsers.types.enums import ContentType
from unity.file_manager.file_parsers.types.formats import FileFormat

if TYPE_CHECKING:
    from unity.file_manager.file_parsers.types.contracts import FileParseResult

UNASSIGNED = -1
FileIngestionStatus = Literal["queued", "ingesting", "success", "error"]


class FileRecordFields(BaseModel):
    """Common fields for FileRecords index rows (shared by row payload + stored rows)."""

    # Core identification
    file_path: str = Field(
        ...,
        description="Filesystem path or display name of the file (adapter-specific).",
    )
    source_uri: Optional[str] = Field(
        default=None,
        description="Canonical provider URI for the file (e.g., local:///abs/path, gdrive://fileId).",
    )
    source_provider: Optional[str] = Field(
        default=None,
        description="Provider/adapter name (e.g., Local).",
    )

    # Processing status
    status: Optional[Literal["success", "error"]] = Field(
        default=None,
        description="Terminal parse outcome once ingestion completes.",
    )
    ingestion_status: Optional[FileIngestionStatus] = Field(
        default=None,
        description=(
            "Background ingest lifecycle state. "
            "`queued`/`ingesting` mean a background attachment job is still running; "
            "`success`/`error` mean the queued job has reached a terminal state."
        ),
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message when status == 'error'.",
    )

    # Flattened metadata fields
    file_format: Optional[FileFormat] = Field(
        default=None,
        description="Canonical file format (e.g., pdf, docx, xlsx, csv).",
    )
    file_size: Optional[int] = Field(
        default=None,
        ge=0,
        description="Size of the file in bytes.",
    )
    total_records: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total parsed content rows for this file.",
    )
    processing_time: Optional[float] = Field(
        default=None,
        ge=0,
        description="Parsing time in seconds.",
    )
    created_at: Optional[str] = Field(
        default=None,
        description="Creation timestamp (ISO-8601).",
    )
    modified_at: Optional[str] = Field(
        default=None,
        description="Last modified timestamp (ISO-8601).",
    )

    # Ingestion layout metadata
    storage_id: str = Field(
        default="",
        description=(
            "Context path identifier for this file's storage. "
            "Empty string means auto-assigned (uses str(file_id)). "
            "When multiple files share the same storage_id, their content "
            "is stored in a shared context."
        ),
    )
    table_ingest: bool = Field(
        default=True,
        description="Whether tables were ingested into per-file table contexts for this file.",
    )

    # Summary and semantic annotations
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall confidence score from the parser/LLM.",
    )
    key_topics: str = Field(
        default="",
        description="Comma-separated key topics detected in the file.",
    )
    named_entities: str = Field(
        default="",
        description="Comma-separated named entities detected in the file.",
    )
    content_tags: str = Field(
        default="",
        description="Comma-separated content tags for retrieval.",
    )
    summary: Optional[str] = Field(
        default="",
        description="Short summary extracted from the document.",
    )

    authoring_assistant_id: Optional[int] = Field(
        default=None,
        description=(
            "Assistant id of the body that published this catalog row. "
            "File bytes stay on the authoring body's local disk; this field "
            "only records which body first indexed the file so reviewers can "
            "attribute Hive-shared catalog rows to their origin body."
        ),
    )


class FileRecordRow(FileRecordFields):
    """
    Client-side payload written to the `FileRecords/<alias>` index context.

    The server is responsible for assigning `file_id` (auto-counting on the index).
    """


class FileRecord(FileRecordFields):
    """
    Lightweight index row for a file in FileRecords/<alias>.

    This schema mirrors the FileManager table schema and should remain lean; the
    heavy parsed content lives under per-file contexts.
    """

    file_id: int = Field(
        default=UNASSIGNED,
        description="Server-assigned unique identifier for the file.",
        ge=UNASSIGNED,
    )

    @staticmethod
    def to_file_record_entry(
        *,
        file_path: str,
        source_uri: Optional[str],
        source_provider: Optional[str],
        parse_result: "FileParseResult",
        storage_id: str,
        table_ingest: bool,
        file_size: Optional[int] = None,
        created_at: Optional[str] = None,
        modified_at: Optional[str] = None,
        total_records: Optional[int] = None,
        document_summary: Optional[str] = None,
    ) -> "FileRecordRow":
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
        parse_result : FileParseResult
            Strict parse output produced by `unity.file_manager.file_parsers`.
        storage_id : str
            Context path identifier. Empty string for auto-assignment (str(file_id)).
        table_ingest : bool
            Whether tables are ingested.

        Returns
        -------
        FileRecordRow
            Typed row payload for insertion into FileRecords/<alias>.
        """
        from unity.file_manager.file_parsers.types.contracts import FileParseResult

        if not isinstance(parse_result, FileParseResult):
            raise TypeError(
                f"parse_result must be FileParseResult, got: {type(parse_result)!r}",
            )

        meta = getattr(parse_result, "metadata", None)

        # Derive processing_time (seconds) from trace (ms)
        processing_time: Optional[float] = None
        try:
            tr = getattr(parse_result, "trace", None)
            dur_ms = getattr(tr, "duration_ms", None) if tr is not None else None
            if dur_ms is not None:
                processing_time = float(dur_ms) / 1000.0
        except Exception:
            processing_time = None

        # Prefer explicit document_summary when the parse result does not carry one.
        summary_val = (getattr(parse_result, "summary", "") or "").strip()
        if not summary_val and document_summary:
            summary_val = str(document_summary).strip()

        from unity.common.authoring import authoring_assistant_id

        return FileRecordRow(
            file_path=file_path,
            source_uri=source_uri,
            source_provider=source_provider,
            status=str(getattr(parse_result, "status", "error") or "error"),
            error=getattr(parse_result, "error", None),
            summary=summary_val,
            file_format=getattr(parse_result, "file_format", None),
            file_size=file_size,
            total_records=total_records,
            processing_time=processing_time,
            created_at=created_at,
            modified_at=modified_at,
            confidence_score=(
                getattr(meta, "confidence_score", None) if meta is not None else None
            ),
            key_topics=(getattr(meta, "key_topics", "") if meta is not None else ""),
            named_entities=(
                getattr(meta, "named_entities", "") if meta is not None else ""
            ),
            content_tags=(
                getattr(meta, "content_tags", "") if meta is not None else ""
            ),
            storage_id=storage_id,
            table_ingest=bool(table_ingest),
            authoring_assistant_id=authoring_assistant_id(),
        )


class FileInfo(BaseModel):
    """
    Comprehensive file information combining filesystem status and ingest identity.

    This model provides a complete picture of a file's state across both the
    raw filesystem and the parsed/indexed state. Use this to understand:
    - Whether the file exists on disk (filesystem_exists)
    - Whether it has been indexed (indexed_exists)
    - Its parse status (parsed_status)
    - Its storage_id (context path identifier)
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
        description="Provider/adapter name (e.g., Local).",
    )
    source_uri: Optional[str] = Field(
        default=None,
        description="Canonical provider URI (e.g., local:///abs/path, gdrive://fileId).",
    )

    # Storage layout
    storage_id: str = Field(
        default="",
        description=(
            "Context path identifier for this file's storage. "
            "Use with describe() or context builders to get full paths."
        ),
    )
    table_ingest: bool = Field(
        default=True,
        description="True when tables are in /Tables/ contexts.",
    )

    # File metadata (when indexed)
    file_format: Optional[FileFormat] = Field(
        default=None,
        description="Canonical file format (e.g., pdf, docx, xlsx, csv).",
    )


class DocumentFields(BaseModel):
    """
    Common fields for per-file `/Content/` (document) rows.

    .. note::
        The underlying Unify context path remains `/Content/` for backward compatibility.
        Only the API type names have been updated (FileContent* → Document*).
    """

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
    content_type: ContentType = Field(
        ...,
        description="Stable content type vocabulary for retrieval/navigation.",
    )
    title: Optional[str] = Field(default=None, description="Heading or inferred title.")
    content_text: Optional[str] = Field(default=None, description="Original raw text.")
    summary: Optional[str] = Field(
        default=None,
        description="Rich summary used for embeddings.",
    )


class DocumentRow(DocumentFields):
    """
    Client-side payload written to a per-file `/Content/` context.

    The server is responsible for assigning `row_id` (auto-counting on the context).
    """


class Document(DocumentFields):
    """
    Per-file context row schema used for storing flattened, hierarchical
    records extracted from a single file. The per-file context lives under
    "<base>/Files/<alias>/<file_id>/Content/" and uses counters to represent the
    document→section→paragraph→sentence hierarchy. Extracted tabular content
    for a file is stored under separate per-table contexts (no predefined
    fields) at "<base>/Files/<alias>/<file_id>/Tables/<table>".

    .. note::
        The underlying Unify context path is `/Content/` (not `/Document/`)
        for backward compatibility.
    """

    # Unique id within the per-file content context (row identifier)
    row_id: int = Field(
        default=UNASSIGNED,
        ge=UNASSIGNED,
        description="Unique row id for Content context",
    )

    @staticmethod
    def to_document_entries(
        *,
        file_id: int,
        rows: List["DocumentRow"],
    ) -> List["DocumentRow"]:
        """
        Build per-file Content entries from parser rows, attaching file_id and
        dropping fields not represented in the schema.
        """
        out: List[DocumentRow] = []
        for r in list(rows or []):
            out.append(r.model_copy(update={"file_id": int(file_id)}))
        return out


# Backward compatibility aliases (deprecated)
FileContentFields = DocumentFields
FileContentRow = DocumentRow
FileContent = Document


class FileTableRefRow(BaseModel):
    """Typed preview row for a per-file table reference."""

    name: str
    context: str
    row_count: int = Field(default=0, ge=0)
    columns: List[str] = Field(default_factory=list)
