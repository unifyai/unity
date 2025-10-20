from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass

# This import is now needed for the new DetectedEvent class
from unity.image_manager.image_manager import ImageHandle


@dataclass
class DetectedEvent:
    """
    Represents a candidate event detected by the ScreenShareManager's first-pass analysis.
    This is a lightweight object containing a pending ImageHandle that can be later enriched
    with a full annotation.
    """
    timestamp: float
    detection_reason: str  # A brief, machine-readable reason, e.g., "visual_change", "user_speech"
    image_handle: ImageHandle # A pending handle to the representative "after" frame.
    preliminary_label: Optional[str] = None # Optional cheap label from the detection step.


class KeyEvent(BaseModel):
    """
    Represents a single, discrete, meaningful event identified within a user's turn.
    This is now primarily used for the *output* of the rich annotation step.
    """

    timestamp: float = Field(
        ...,
        description="The precise timestamp (in seconds, matching the media stream time) of when this specific event occurred. Example: 15.2",
    )
    image_annotation: str = Field(
        ...,
        description="A rich, contextual description of what the representative image shows and why it is relevant to the user's turn and the overall session context.",
    )
    representative_timestamp: float = Field(
        ...,
        description="The timestamp of the single 'AFTER' frame that best represents the visual state of this event. This must exactly match one of the timestamps provided in the input.",
    )


class TurnAnalysisResponse(BaseModel):
    """
    The structured output from the LLM after analyzing a user's turn, containing all identified key events.
    """

    events: List[KeyEvent] = Field(
        default_factory=list,
        description="A chronologically ordered list of all meaningful events that occurred during the user's turn.",
    )