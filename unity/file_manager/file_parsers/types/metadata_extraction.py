from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class DocumentMetadataExtraction(BaseModel):
    """Pydantic model for LLM-based document metadata extraction."""

    document_type: Literal[
        "policy",
        "procedure",
        "guideline",
        "handbook",
        "form",
        "template",
        "other",
    ] = Field(description="Type of document - use exact values only")

    category: Literal[
        "safety",
        "hr",
        "finance",
        "operations",
        "maintenance",
        "legal",
        "tenancy",
        "general",
    ] = Field(description="Document category - use exact values only")

    key_topics: List[str] = Field(
        description="Main topics/themes (3-8 items, snake_case format)",
        min_length=3,
        max_length=8,
    )

    named_entities: Dict[str, List[str]] = Field(
        description="Extracted entities organized by type",
        default_factory=dict,
    )

    content_tags: List[str] = Field(
        description="Searchable keywords for query matching (5-12 items)",
        min_length=5,
        max_length=12,
    )

    confidence_score: float = Field(
        description="Confidence in extraction accuracy (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
