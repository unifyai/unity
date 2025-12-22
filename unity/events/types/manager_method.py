"""Payload model for ManagerMethod events."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ManagerMethodPayload(BaseModel):
    """Payload for manager method invocations (ask/update/execute).

    Published at method entry (phase=incoming) and exit (phase=outgoing),
    as well as for mid-flight actions like clarifications.
    """

    model_config = ConfigDict(extra="allow")

    manager: str = Field(description="Name of the manager being called")
    method: str = Field(description="Method name (ask, update, execute, etc.)")
    source: Optional[str] = Field(default=None, description="Caller source identifier")
    phase: Optional[str] = Field(
        default=None,
        description="Event phase: 'incoming' or 'outgoing'",
    )
    action: Optional[str] = Field(
        default=None,
        description="Action being performed on a handle",
    )

    # Flexible fields for incoming/outgoing payloads
    question: Optional[str] = Field(
        default=None,
        description="Incoming question text (for ask methods)",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Incoming instructions text (for update/execute methods)",
    )
    answer: Optional[str] = Field(
        default=None,
        description="Outgoing answer/result text",
    )
