"""
Typed ContentNode graph.

This is the canonical internal representation produced by file parser backends.
It is format-agnostic: nodes form a graph (typically a tree + cross-links) and
each node may carry typed payload and provenance.

Important distinction
---------------------
- The graph can be rich (e.g., include `page` nodes for PDFs, `sheet` nodes for XLSX).
- The FileManager storage contract remains stable and simple:
  `/Content/` rows are produced by a lowering step and use a small set of
  `content_type` values (document/section/paragraph/sentence/image/table/sheet).
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


from .enums import NodeKind
from .json_types import JsonObject


class CoordOrigin(str):
    """Coordinate origin label used for bounding boxes."""


class BBox(BaseModel):
    """A simple bounding box (used for provenance)."""

    l: float
    t: float
    r: float
    b: float
    origin: Optional[Literal["TOPLEFT", "BOTTOMLEFT"]] = None


class Provenance(BaseModel):
    """Where a node came from in the source file."""

    page_no: Optional[int] = None
    bbox: Optional[BBox] = None
    source_ref: Optional[str] = None  # e.g. a docling self_ref or external id
    sheet_name: Optional[str] = None  # for spreadsheets


# ----------------------------- Payloads ---------------------------------- #


class BasePayload(BaseModel):
    """Base payload with a discriminator field."""

    # Use a string discriminator for Pydantic's union; the node kind enum is
    # enforced at `ContentNode.kind`.
    type: str


class DocumentPayload(BasePayload):
    type: Literal["document"] = "document"
    title: Optional[str] = None


class SectionPayload(BasePayload):
    type: Literal["section"] = "section"
    level: Optional[int] = None
    path: Optional[List[str]] = None


class ParagraphPayload(BasePayload):
    type: Literal["paragraph"] = "paragraph"


class SentencePayload(BasePayload):
    type: Literal["sentence"] = "sentence"
    sentence_index: Optional[int] = None


class SheetPayload(BasePayload):
    type: Literal["sheet"] = "sheet"
    sheet_index: Optional[int] = None
    sheet_name: Optional[str] = None


class TablePayload(BasePayload):
    type: Literal["table"] = "table"
    label: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    # A bounded sample used for RAG summaries (not necessarily full rows)
    sample_rows: List[JsonObject] = Field(default_factory=list)
    num_rows: Optional[int] = None
    num_cols: Optional[int] = None


class ImagePayload(BasePayload):
    type: Literal["image"] = "image"
    caption: Optional[str] = None
    # Reference to stored bytes is backend-specific; the pipeline will standardize later.
    image_ref: Optional[str] = None


class PagePayload(BasePayload):
    type: Literal["page"] = "page"
    page_no: Optional[int] = None


class GenericPayload(BasePayload):
    type: Literal["other"] = "other"
    data: JsonObject = Field(default_factory=dict)


NodePayload = (
    DocumentPayload
    | SectionPayload
    | ParagraphPayload
    | SentencePayload
    | SheetPayload
    | TablePayload
    | ImagePayload
    | PagePayload
    | GenericPayload
)


class ContentNode(BaseModel):
    """A single node in the parsed content graph."""

    node_id: str
    kind: NodeKind
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    order: Optional[int] = None

    # Textual fields used for RAG and lowering.
    title: Optional[str] = None
    text: Optional[str] = None
    summary: Optional[str] = None

    # Structured metadata
    provenance: Optional[Provenance] = None
    payload: Optional[NodePayload] = None
    meta: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_payload_consistency(self) -> "ContentNode":
        if self.payload is not None:
            if str(getattr(self.payload, "type", None)) != str(
                getattr(self.kind, "value", self.kind),
            ):
                raise ValueError(
                    f"payload.type ({getattr(self.payload, 'type', None)!r}) must match kind ({self.kind!r})",
                )
        return self


class ContentGraph(BaseModel):
    """A typed graph of content nodes for a single file."""

    root_id: str
    nodes: Dict[str, ContentNode] = Field(default_factory=dict)

    def get(self, node_id: str) -> ContentNode:
        return self.nodes[node_id]

    def children(self, node_id: str) -> List[ContentNode]:
        node = self.get(node_id)
        return [self.get(cid) for cid in node.children_ids]
