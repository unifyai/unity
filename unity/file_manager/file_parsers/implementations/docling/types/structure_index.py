from __future__ import annotations

"""Docling-local typed index structures.

These types are used only by the Docling implementation to carry derived
structure for:
- mapping Docling item refs to heading paths, and
- attaching tables/images to the correct section path.
"""

from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field


class DoclingHeadingOrderItem(BaseModel):
    text: str
    level: int = Field(ge=1)
    self_ref: str
    path: List[str] = Field(default_factory=list)


class DoclingStructureIndex(BaseModel):
    """Derived structure index for a single DoclingDocument."""

    ref_to_path: Dict[str, Tuple[str, ...]] = Field(default_factory=dict)
    heading_refs: Set[str] = Field(default_factory=set)
    heading_ref_to_level: Dict[str, int] = Field(default_factory=dict)
    heading_order: List[DoclingHeadingOrderItem] = Field(default_factory=list)
    title: Optional[str] = None
