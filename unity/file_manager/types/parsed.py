from __future__ import annotations

from typing import List, Optional

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
        description="Parsing time in seconds",
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="Parser/LLM confidence score",
    )


class BaseParsedFile(BaseModel):
    """Compact, reference-first parse result returned by FileManager.parse.

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
    content_ref: ContentRef
    tables_ref: List[TableRef] = Field(default_factory=list)
    metrics: FileMetrics = Field(default_factory=FileMetrics)


class ParsedPDF(BaseParsedFile):
    """Parsed PDF with document-oriented metrics."""

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


class ParsedDocx(BaseParsedFile):
    """Parsed DOCX with document-oriented metrics."""

    total_sections: Optional[int] = Field(default=None)
    image_count: Optional[int] = Field(default=None)
    table_count: Optional[int] = Field(default=None)
    total_records: Optional[int] = Field(default=None)


class ParsedDoc(BaseParsedFile):
    """Parsed DOC with document-oriented metrics."""

    total_sections: Optional[int] = Field(default=None)
    image_count: Optional[int] = Field(default=None)
    table_count: Optional[int] = Field(default=None)
    total_records: Optional[int] = Field(default=None)


class ParsedXlsx(BaseParsedFile):
    """Parsed XLSX with spreadsheet-oriented metrics."""

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


class ParsedCsv(BaseParsedFile):
    """Parsed CSV with table-oriented metrics."""

    table_count: Optional[int] = Field(
        default=None,
        description="Number of tables extracted (typically 1)",
    )
