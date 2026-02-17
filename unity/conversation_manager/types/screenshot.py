from __future__ import annotations

from datetime import datetime
from typing import NamedTuple


class ScreenshotEntry(NamedTuple):
    """A single screenshot captured during screen sharing, paired with context."""

    b64: str
    utterance: str
    timestamp: datetime
    source: str  # "assistant" (assistant's desktop) or "user" (user's screen share)
    local_message_id: int | None = None
