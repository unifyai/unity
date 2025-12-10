"""Pydantic models for ingested file results returned by FileManager.ingest_files.

These models represent the structured output after files have been parsed and
ingested into Unify contexts. They are compact, reference-first models that
avoid heavy payloads (full_text, raw records) and instead provide pointers
(content_ref, tables_ref) to query details via manager tools.

The naming convention uses "Ingested*" to distinguish from "ParsedFile" which
represents the full/raw parsing output with all heavy fields included.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field
from unity.file_manager.parser.types.enums import FileFormat, MimeType


class ContentRef(BaseModel):
    """Reference to the per-file content context with lightweight metrics."""

    context: str = Field(..., description="Unify context for per-file content rows")
    record_count: int = Field(0, ge=0, description="Number of content rows ingested")
    text_chars: int = Field(
        0,
        ge=0,
        description="Approximate number of text characters available",
    )


class TableRef(BaseModel):
    """Reference to a per-file table context with a tiny preview of its schema."""

    name: str = Field(
        ...,
        description="Logical table label (e.g., sheet name or section label)",
    )
    context: str = Field(..., description="Unify context for this per-file table")
    row_count: int = Field(0, ge=0, description="Number of rows in the table")
    columns: List[str] = Field(
        default_factory=list,
        description="Preview of column names (truncated)",
    )


class FileMetrics(BaseModel):
    """Basic file-level metrics helpful for routing without heavy payloads."""

    file_size: Optional[int] = Field(default=None, description="Size in bytes")
    processing_time: Optional[float] = Field(
        default=None,
        description="Processing time in seconds",
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="Parser/LLM confidence score",
    )


class BaseIngestedFile(BaseModel):
    """Compact, reference-first ingest result returned by FileManager.ingest_files.

    Heavy artifacts (full_text, records, table rows) are NOT included. Use the
    provided `content_ref` and `tables_ref` to query details via manager tools.
    """

    # Identity and status
    file_path: str = Field(
        ...,
        description="Filesystem path or display name of the file",
    )
    source_uri: Optional[str] = Field(
        default=None,
        description="Canonical provider URI (e.g., local:///abs/path)",
    )
    display_path: Optional[str] = Field(
        default=None,
        description="Human-friendly path for prompts",
    )
    file_format: Optional[FileFormat] = Field(
        default=None,
        description="Canonical file format (e.g., pdf, docx, xlsx, csv)",
    )
    mime_type: Optional[MimeType] = Field(
        default=None,
        description="MIME type (e.g., application/pdf)",
    )
    status: str = Field(default="success", description="'success' or 'error'")
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'error'",
    )

    # Timeline
    created_at: Optional[str] = Field(
        default=None,
        description="Creation timestamp (ISO-8601)",
    )
    modified_at: Optional[str] = Field(
        default=None,
        description="Last modified timestamp (ISO-8601)",
    )

    # Compact content pointers
    summary_excerpt: str = Field(
        default="",
        description="Short excerpt/summary (truncated) to orient the LLM without large token cost",
    )
    content_ref: ContentRef = Field(
        default_factory=lambda: ContentRef(context="", record_count=0, text_chars=0),
    )
    tables_ref: List[TableRef] = Field(default_factory=list)
    metrics: FileMetrics = Field(default_factory=FileMetrics)


class IngestedPDF(BaseIngestedFile):
    """Ingested PDF with document-oriented metrics."""

    page_count: Optional[int] = Field(default=None, description="Total pages detected")
    total_sections: Optional[int] = Field(
        default=None,
        description="Number of sections extracted",
    )
    image_count: Optional[int] = Field(
        default=None,
        description="Number of images extracted",
    )
    table_count: Optional[int] = Field(
        default=None,
        description="Number of tables extracted",
    )
    total_records: Optional[int] = Field(
        default=None,
        description="Flattened content row count",
    )


class IngestedDocx(BaseIngestedFile):
    """Ingested DOCX with document-oriented metrics."""

    total_sections: Optional[int] = Field(default=None)
    image_count: Optional[int] = Field(default=None)
    table_count: Optional[int] = Field(default=None)
    total_records: Optional[int] = Field(default=None)


class IngestedDoc(BaseIngestedFile):
    """Ingested DOC with document-oriented metrics."""

    total_sections: Optional[int] = Field(default=None)
    image_count: Optional[int] = Field(default=None)
    table_count: Optional[int] = Field(default=None)
    total_records: Optional[int] = Field(default=None)


class IngestedXlsx(BaseIngestedFile):
    """Ingested XLSX with spreadsheet-oriented metrics."""

    sheet_count: Optional[int] = Field(
        default=None,
        description="Number of sheets/tables extracted",
    )
    sheet_names: List[str] = Field(
        default_factory=list,
        description="Names of sheets (when available)",
    )
    table_count: Optional[int] = Field(
        default=None,
        description="Number of tables extracted",
    )


class IngestedCsv(BaseIngestedFile):
    """Ingested CSV with table-oriented metrics."""

    table_count: Optional[int] = Field(
        default=None,
        description="Number of tables extracted (typically 1)",
    )


class IngestedMinimal(BaseModel):
    """Minimal stub for 'none' return mode - just status info, no content refs."""

    file_path: str = Field(..., description="The file path")
    status: str = Field(default="success", description="'success' or 'error'")
    error: Optional[str] = Field(default=None, description="Error message if any")
    total_records: int = Field(default=0, description="Number of records parsed")
    file_format: Optional[str] = Field(default=None, description="File format")


# Union type for all ingested file models (compact mode)
IngestedFileUnion = Union[
    BaseIngestedFile,
    IngestedPDF,
    IngestedDocx,
    IngestedDoc,
    IngestedXlsx,
    IngestedCsv,
]

# Import ParsedFile for full mode - this is the raw parsing output
from unity.file_manager.types.file import ParsedFile

# Type for individual file results depending on return mode
# - compact: IngestedFileUnion (reference-first, no heavy fields)
# - full: ParsedFile (complete parsing output with records, full_text, etc.)
# - none: IngestedMinimal (just status stub)
FileResultType = Union[IngestedFileUnion, ParsedFile, IngestedMinimal]


class PipelineStatistics(BaseModel):
    """Global statistics for the ingest pipeline run."""

    total_files: int = Field(0, ge=0, description="Total number of files processed")
    success_count: int = Field(
        0,
        ge=0,
        description="Number of files ingested successfully",
    )
    error_count: int = Field(0, ge=0, description="Number of files that failed")
    total_duration_ms: float = Field(
        0.0,
        ge=0,
        description="Total pipeline duration in milliseconds",
    )
    total_content_rows: int = Field(
        0,
        ge=0,
        description="Total content rows ingested across all files",
    )
    total_table_rows: int = Field(
        0,
        ge=0,
        description="Total table rows ingested across all files",
    )


class IngestPipelineResult(BaseModel):
    """Container model for FileManager.ingest_files output.

    Provides structured access to per-file ingest results and global statistics.
    This is the top-level return type for the ingest_files method.

    Supports multiple return modes:
    - "compact" (default): IngestedFileUnion Pydantic models
    - "full": ParsedFile Pydantic models (complete with records, full_text)
    - "none": IngestedMinimal Pydantic models (just status stub)
    """

    model_config = {"arbitrary_types_allowed": True}

    files: Dict[str, FileResultType] = Field(
        default_factory=dict,
        description="Mapping of file_path → ingested file result (Pydantic model)",
    )
    statistics: PipelineStatistics = Field(
        default_factory=PipelineStatistics,
        description="Global pipeline statistics",
    )

    def __getitem__(self, key: str) -> FileResultType:
        """Allow dict-like access: result[file_path]."""
        return self.files[key]

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator: file_path in result."""
        return key in self.files

    def __iter__(self):
        """Iterate over file paths."""
        return iter(self.files)

    def __len__(self) -> int:
        """Number of files in the result."""
        return len(self.files)

    def items(self):
        """Iterate over (file_path, result) pairs."""
        return self.files.items()

    def keys(self):
        """Return file paths."""
        return self.files.keys()

    def values(self):
        """Return file results."""
        return self.files.values()

    def get(self, key: str, default=None) -> Optional[FileResultType]:
        """Get a file result with optional default."""
        return self.files.get(key, default)

    def update(
        self,
        other: Union[Dict[str, FileResultType], "IngestPipelineResult"],
    ) -> None:
        """Update files dict with another mapping."""
        if isinstance(other, IngestPipelineResult):
            self.files.update(other.files)
        else:
            self.files.update(other)

    @classmethod
    def from_results(
        cls,
        results: Dict[str, FileResultType],
        total_duration_ms: float = 0.0,
    ) -> "IngestPipelineResult":
        """Build an IngestPipelineResult from a dict of file results.

        Automatically computes statistics from the individual file results.
        All results must be Pydantic models (IngestedFileUnion, ParsedFile, or IngestedMinimal).
        """

        def _get_status(r: BaseModel) -> Optional[str]:
            return getattr(r, "status", None)

        def _get_content_rows(r: BaseModel) -> int:
            # For compact models with content_ref
            cr = getattr(r, "content_ref", None)
            if cr:
                return getattr(cr, "record_count", 0)
            # For ParsedFile (full mode)
            return getattr(r, "total_records", 0)

        def _get_table_rows(r: BaseModel) -> int:
            tr = getattr(r, "tables_ref", None)
            if tr:
                return sum(getattr(t, "row_count", 0) for t in tr)
            return 0

        stats = PipelineStatistics(
            total_files=len(results),
            success_count=sum(
                1 for r in results.values() if _get_status(r) == "success"
            ),
            error_count=sum(1 for r in results.values() if _get_status(r) == "error"),
            total_duration_ms=total_duration_ms,
            total_content_rows=sum(_get_content_rows(r) for r in results.values()),
            total_table_rows=sum(_get_table_rows(r) for r in results.values()),
        )
        return cls(files=results, statistics=stats)
