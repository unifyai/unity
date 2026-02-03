"""Payload model for Comms events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class CommsPayload(BaseModel):
    """Payload for Comms events (phone calls, SMS, email, etc.).

    Comms events have varied shapes depending on subtype (PhoneCallReceived,
    SMSReceived, etc.). The payload_cls field on the Event identifies the
    specific subtype. This model captures common fields and allows extras.
    """

    model_config = ConfigDict(extra="allow")

    # Common fields across most Comms events
    contact: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Contact info dict",
    )
    content: Optional[str] = Field(
        default=None,
        description="Message content if applicable",
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Event timestamp",
    )
