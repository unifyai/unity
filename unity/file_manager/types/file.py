"""Pydantic data model for File records stored by the FileManager."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

UNASSIGNED = -1


class File(BaseModel):
    """
    Structured representation of a parsed file record, matching _create_result_dict output.
    This model is used both for the FileManager table schema and for storing parsed results.
    """

    file_id: int = Field(
        default=UNASSIGNED,
        description="Unique identifier for the file",
        ge=UNASSIGNED,
    )

    # Core identification
    filename: str = Field(
        ...,
        description="Display filename unique within the session.",
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

    # Parsed content
    records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Flat records from document parsing (from Document.to_flat_records()).",
    )

    # Full text content
    full_text: Optional[str] = Field(
        default="",
        description="Complete plain text content of the parsed file.",
    )

    # File metadata (nested structure matching _create_result_dict)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="""File metadata including:
        - document_id: str
        - total_records: int
        - processing_time: float
        - file_path: str
        - file_type: str
        - file_size: int
        - created_at: datetime
        - modified_at: datetime
        """,
    )

    description: Optional[str] = Field(
        default="",
        description="Description of the file.",
    )
