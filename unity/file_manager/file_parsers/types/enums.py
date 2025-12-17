"""
Enums shared across the parsing subsystem.

These enums enforce a strict, finite vocabulary for:
- `NodeKind`: internal ContentGraph node kinds
- `ContentType`: persisted `/Content/` row content_type values
"""

from __future__ import annotations

from enum import Enum


class NodeKind(str, Enum):
    """Node kind for internal `ContentGraph` nodes."""

    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    SHEET = "sheet"
    TABLE = "table"
    IMAGE = "image"

    # Internal / provenance-oriented kinds (not necessarily lowered)
    PAGE = "page"
    BLOCK = "block"
    SPAN = "span"
    OTHER = "other"


class ContentType(str, Enum):
    """
    Persisted `/Content/` content_type values.

    This vocabulary is intentionally small and stable so RAG agents can reason
    about content navigation reliably.
    """

    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    SHEET = "sheet"
    TABLE = "table"
    IMAGE = "image"
