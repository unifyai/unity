"""
Typed extracted table representation used at the parser ↔ FileManager boundary.

This is intentionally separate from the ContentGraph so FileManager ingestion can
operate without needing to understand internal graph structure.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .json_types import JsonObject


class ExtractedTable(BaseModel):
    """
    A table extracted from a file, ready for ingestion into `/Tables/<label>`.

    Notes
    -----
    - `rows` should be a list[dict] with string keys so it can be passed directly
      to Unify `create_logs` helpers.
    - `sample_rows` is a bounded preview used to build RAG-friendly table catalog
      rows in `/Content/` (not necessarily the first N rows; can be stratified).
    """

    table_id: str = Field(..., description="Stable table identifier within this parse")
    label: str = Field(
        ...,
        description="Human-meaningful table label used for context naming (will be sanitized by FileManager.safe)",
    )
    sheet_name: Optional[str] = Field(
        default=None,
        description="Worksheet/sheet name when the source is a spreadsheet",
    )

    columns: List[str] = Field(default_factory=list)
    rows: List[JsonObject] = Field(default_factory=list)

    # Bounded preview for catalog / summary generation
    sample_rows: List[JsonObject] = Field(default_factory=list)

    num_rows: Optional[int] = None
    num_cols: Optional[int] = None

    # RAG-friendly description (optional; can be filled by summarization steps)
    table_summary: Optional[str] = None
