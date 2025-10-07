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
    event_description: str = Field(
        ...,
        description="A concise, third-person summary of what occurred at this moment (e.g., 'User clicked the submit button').",
    )
    representative_timestamp: float = Field(
        ...,
        description="The timestamp of the single 'AFTER' frame that best represents the visual state of this event. This must exactly match one of the timestamps provided in the input.",
    )
    triggering_phrase: Optional[str] = Field(
        None,
        description="If the event was directly referenced in the user's speech, this is the exact substring from the speech transcript that corresponds to the event.",
    )


class TurnAnalysisResponse(BaseModel):
    """
    The structured output from the LLM after analyzing a user's turn, containing all identified key events.
    """

    events: List[KeyEvent] = Field(
        default_factory=list,
        description="A chronologically ordered list of all meaningful events that occurred during the user's turn.",
    )
