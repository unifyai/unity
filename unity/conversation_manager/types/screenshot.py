from __future__ import annotations

from datetime import datetime
from typing import NamedTuple


class ScreenshotEntry(NamedTuple):
    """A single screenshot captured during screen sharing or webcam, paired with context."""

    gcs_uri: str
    utterance: str
    timestamp: datetime
    source: str  # "assistant" | "user" (screen share) | "webcam"
    local_message_id: int | None = None
