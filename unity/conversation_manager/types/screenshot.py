from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import NamedTuple


class ScreenshotEntry(NamedTuple):
    """A single screenshot captured during screen sharing or webcam, paired with context."""

    b64: str
    utterance: str
    timestamp: datetime
    source: str  # "assistant" | "user" (screen share) | "webcam" | "google_meet"
    local_message_id: int | None = None
    filepath: str | None = None


_SOURCE_SUBFOLDER = {
    "assistant": "Assistant",
    "user": "User",
    "webcam": "Webcam",
    "google_meet": "GoogleMeet",
}


def generate_screenshot_path(entry: ScreenshotEntry) -> str:
    """Compute a deterministic filepath for a screenshot (no I/O)."""
    subfolder = _SOURCE_SUBFOLDER.get(entry.source, entry.source.title())
    directory = Path("Screenshots") / subfolder
    stem = entry.timestamp.strftime("%Y-%m-%dT%H-%M-%S.%f")
    return str(directory / f"{stem}.jpg")


def write_screenshot_to_disk(entry: ScreenshotEntry, path: str) -> None:
    """Write screenshot bytes to disk. Safe to call from a background task."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(entry.b64))
