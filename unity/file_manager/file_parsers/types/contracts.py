"""
Parse contracts for `unity.file_manager.file_parsers`.

This module defines the *canonical* typed boundary between the FileManager and
the FileParser.

Boundary
--------
- Input: `FileParseRequest`
- Output: `FileParseResult`

The parser is allowed to produce richer artifacts (e.g., `ContentGraph`) and
observability data (`FileParseTrace`) but must always return them wrapped in the
strict `FileParseResult` model.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .formats import FileFormat, MimeType
from .graph import ContentGraph
from .json_types import JsonObject
from .table import ExtractedTable

SUMMARY_UNSET: str = "__UNSET__"
"""Sentinel value for ``FileParseResult.summary`` before enrichment runs."""

METADATA_UNSET: None = None
"""Sentinel for ``FileParseResult.metadata`` before enrichment runs.

``None`` is safe here because downstream code already guards with
``if result.metadata is not None``.
"""


class FileParseRequest(BaseModel):
    """
    Canonical parse request.

    Identity model
    --------------
    - `logical_path` is the stable, external identifier for the file (used for
      FileRecords, returned results, and context naming).
    - `source_local_path` is the on-disk path the backend should actually read from.
      This may be a temporary exported path produced by a filesystem adapter.
    """

    logical_path: str = Field(
        ...,
        description="Stable external identifier (adapter path) used by FileManager.",
    )
    source_local_path: str = Field(
        ...,
        description="On-disk path used for reading/conversion (may be a temp export).",
    )
    file_format: Optional[FileFormat] = Field(default=None)
    mime_type: Optional[MimeType] = Field(default=None)


class StepStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"
    DEGRADED = "degraded"


class ParseError(BaseModel):
    """Structured error payload for traceability."""

    code: str = Field(..., description="Stable machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    exception_type: Optional[str] = Field(default=None)
    details: JsonObject = Field(default_factory=dict)


class StepTrace(BaseModel):
    """Trace information for a single pipeline step."""

    name: str
    status: StepStatus = StepStatus.SUCCESS
    duration_ms: float = 0.0
    counters: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[ParseError] = None


class ConversionHop(BaseModel):
    """Trace a single file conversion hop (e.g., DOC → PDF)."""

    operation: str = Field(
        ...,
        description="High-level operation name (e.g., 'doc_to_pdf').",
    )
    src: str = Field(..., description="Source local path before conversion.")
    dst: Optional[str] = Field(
        default=None,
        description="Destination local path after conversion.",
    )
    backend: str = Field(
        default="",
        description="Concrete converter backend identifier (e.g., 'soffice', 'docx2pdf', 'reuse').",
    )
    ok: bool = Field(default=False, description="Whether the conversion hop succeeded.")
    message: str = Field(default="", description="Converter message/debug details.")


class FileParseTrace(BaseModel):
    """Trace information for an entire parse operation."""

    logical_path: str
    backend: str = Field(..., description="Backend identifier (e.g., 'pdf_backend')")
    file_format: Optional[FileFormat] = None
    mime_type: Optional[MimeType] = None

    status: StepStatus = StepStatus.SUCCESS
    duration_ms: float = 0.0
    steps: List[StepTrace] = Field(default_factory=list)

    counters: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

    # Optional debugging identity (do not use for contexts/records)
    source_local_path: Optional[str] = None
    parsed_local_path: Optional[str] = Field(
        default=None,
        description=(
            "Local path actually parsed after conversions. "
            "If omitted, callers should treat it as equal to source_local_path."
        ),
    )
    conversion_chain: List[ConversionHop] = Field(
        default_factory=list,
        description="Ordered conversion hops from source_local_path to parsed_local_path.",
    )


class FileParseMetadata(BaseModel):
    """
    Metadata extracted during parsing.

    Note: for embeddings/search, these fields are stored as comma-separated strings.
    """

    key_topics: str = Field(default="", description="Comma-separated key topics")
    named_entities: str = Field(
        default="",
        description="Comma-separated named entities",
    )
    content_tags: str = Field(default="", description="Comma-separated content tags")

    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class FileParseResult(BaseModel):
    """
    Strict output for a parse operation.

    This is the only output object file parser backends are allowed to return.

    Note: the FileManager may further *adapt* this parse output into its own
    ingestion/storage row models (e.g. `/Content/` rows). The parser itself does
    not own that transformation.
    """

    logical_path: str
    status: Literal["success", "error"] = "success"
    error: Optional[str] = None

    file_format: Optional[FileFormat] = None
    mime_type: Optional[MimeType] = None

    # High-level artifacts produced by the parser backend
    tables: List[ExtractedTable] = Field(default_factory=list)

    # Lightweight document-level summary (safe for FileRecords).
    # Starts as ``SUMMARY_UNSET`` — enrichment replaces it with a real
    # string (possibly ``""`` for truly empty documents).  Using a
    # non-null string sentinel avoids None-propagation issues through
    # Pydantic contexts and embedding pipelines.
    summary: str = Field(
        default=SUMMARY_UNSET,
        description=(
            "SUMMARY_UNSET means enrichment has not run yet.  "
            "Empty string means enrichment ran but produced nothing."
        ),
    )

    # Optional text.
    # - For documents (PDF/DOCX/TXT): extracted text is useful for debugging and enrichment.
    # - For spreadsheets (CSV/XLSX): this should be a **bounded profile** (columns + sample rows),
    #   not a full dump of the dataset.
    full_text: str = ""

    # ``None`` = enrichment has not run.  Populated model = enrichment ran.
    metadata: Optional[FileParseMetadata] = None

    # Observability + debug
    trace: Optional[FileParseTrace] = None
    graph: Optional[ContentGraph] = None
