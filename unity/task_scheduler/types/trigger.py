"""Trigger definition for inbound communication events that activate a task."""

from __future__ import annotations

from unity.conversation_manager.types import Medium
from typing import List, Optional

from pydantic import BaseModel, Field


class Trigger(BaseModel):
    """
    Describe an **in-bound communication event** that should *start* the task.
    """

    medium: Medium = Field(
        ...,
        description="Which channel (SMS / email / call / …) must be observed",
    )
    from_contact_ids: Optional[List[int]] = Field(
        default=None,
        description=(
            "Only messages/calls from these contacts count.  "
            "Leave unset to match *any* sender."
        ),
    )
    omit_contact_ids: Optional[List[int]] = Field(
        default=None,
        description="Explicitly ignore these contacts (overrides *from_contact_ids*)",
    )
    interrupt: bool = Field(
        default=False,
        description=(
            "If **True** an on-going task is *pre-empted* and queued behind "
            "this one; otherwise the caller/writer is put on hold."
        ),
    )
    recurring: bool = Field(
        default=False,
        description=(
            "If **True** the task returns to the *triggerable* state after "
            "completion so future events re-activate it."
        ),
    )
