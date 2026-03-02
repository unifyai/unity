"""Payload models for desktop primitive events."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DesktopPrimitivePayload(BaseModel):
    """Fired when a primitives.computer.desktop.* method is called.

    Used by the ConversationManager to gate desktop fast-path tool exposure.
    """

    method: str = Field(
        description="Desktop method that was invoked (act, observe, etc.)",
    )


class DesktopActCompletedPayload(BaseModel):
    """Fired when primitives.computer.desktop.act() completes.

    Carries the instruction and the agent's summary of what was done so the CM
    can notify both the slow brain and fast brain during interactive
    screen-sharing sessions. The actual visual state is already captured by
    the regular screen-share screenshot pipeline on the next brain turn.
    """

    instruction: str = Field(description="The instruction that was executed.")
    summary: str = Field(description="Agent's description of what was done.")
