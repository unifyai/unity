from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass

from unity.image_manager.image_manager import ImageHandle


@dataclass
class DetectedEvent:
    """
    Represents a candidate event detected by the ScreenShareManager's first-pass analysis.
    """

    timestamp: float
    detection_reason: str
    image_handle: ImageHandle
    preliminary_label: Optional[str] = None


class KeyEvent(BaseModel):
    """
    Represents a single, fully annotated event, combining a timestamp with a rich description.
    This is used internally for creating summaries and providing recent event history.
    """

    timestamp: float = Field(..., description="The precise timestamp of the event.")
    image_annotation: str = Field(
        ...,
        description="The rich, contextual description of the event.",
    )
    representative_timestamp: float = Field(
        ...,
        description="The timestamp of the frame representing this event.",
    )


class TurnAnalysisResponse(BaseModel):
    """
    The structured output from the LLM after analyzing a user's turn.
    (Used by the original, more complex annotation prompt logic).
    """

    events: List[KeyEvent] = Field(default_factory=list)
