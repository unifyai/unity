"""Payload model for ToolLoop events."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ToolLoopPayload(BaseModel):
    """Payload for tool loop LLM messages.

    Published for each LLM message during async tool loop execution,
    capturing both the raw message and loop context.
    """

    model_config = ConfigDict(extra="allow")

    message: Dict[str, Any] = Field(
        description="Raw LLM message dict (explicitly untyped to handle varied shapes)",
    )
    method: str = Field(description="Public method that spawned this loop (loop_id)")
    hierarchy: List[str] = Field(
        default_factory=list,
        description="Lineage of nested loops",
    )
    # TODO: remove hierarchy_label once frontend migrates to hierarchy-only
    # tree building -- it is now trivially "->".join(hierarchy).
    hierarchy_label: str = Field(
        default="",
        description="Human-readable hierarchy label (deprecated: derivable from hierarchy)",
    )
    origin: Optional[str] = Field(default=None, description="Origin identifier")
