"""Payload model for DesktopPrimitiveInvoked events."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DesktopPrimitivePayload(BaseModel):
    """Fired when a primitives.computer.desktop.* method is called.

    Used by the ConversationManager to gate desktop fast-path tool exposure.
    """

    method: str = Field(
        description="Desktop method that was invoked (act, observe, etc.)",
    )
