"""Re-export the existing Message model as the payload type for Message events."""

from __future__ import annotations

from ...transcript_manager.types.message import Message as MessagePayload

__all__ = ["MessagePayload"]
