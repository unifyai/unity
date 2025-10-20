from typing import List, Optional
from pydantic import BaseModel, Field


class KeyEvent(BaseModel):
    """
    Represents a single, discrete, meaningful event identified within a user's turn.
    """

    timestamp: float = Field(
        ...,
        description="The precise timestamp (in seconds, matching the media stream time) of when this specific event occurred. Example: 15.2",
    )
    image_annotation: str = Field(
        ...,
        description="A description of what the representative image shows and why it is relevant to the user's turn and the overall session context.",
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
