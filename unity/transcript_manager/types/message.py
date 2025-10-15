from enum import StrEnum
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from typing import Literal
from ...image_manager.types import ImageRefs

UNASSIGNED = -1


class Medium(StrEnum):
    UNIFY_MESSAGE = "unify_message"
    UNIFY_CALL = "unify_call"
    UNIFY_MEET = "unify_meet"
    EMAIL = "email"
    SMS_MESSAGE = "sms_message"
    PHONE_CALL = "phone_call"
    WHATSAPP_MSG = "whatsapp_message"
    WHATSAPP_CALL = "whatsapp_call"
    GOOGLE_MEET = "google_meet"


class ScreenShareAnnotation(BaseModel):
    caption: str
    image: str
    type: Literal["vision", "speech"]


class Message(BaseModel):
    message_id: int = Field(description="Unique identifier for the message", ge=-1)
    medium: Medium = Field(
        description="The communication channel used for this message",
    )
    sender_id: int = Field(description="ID of the contact who sent the message")
    receiver_ids: list[int] = Field(
        description="IDs of the contact(s) who received the message.",
        min_length=1,
    )
    timestamp: datetime = Field(
        description="When the message was sent/received in ISO-8601 format",
    )
    content: str = Field(description="The actual text content of the message")
    exchange_id: int = Field(
        description="ID of the conversation thread this message belongs to",
        ge=-1,
    )
    images: ImageRefs = Field(
        default_factory=lambda: ImageRefs.model_validate([]),
        description=(
            "List of image references aligned to the text by freeform explanation. "
            "Use ImageRefs with RawImageRef and AnnotatedImageRef entries."
        ),
    )
    call_utterance_timestamp: str = Field(
        default="",
        description="Timestamp of the utterance associated with calls",
    )
    # call_url: str = Field(
    #     default="",
    #     description="URL of the recorded call file associated with the call",
    # )
    screen_share: dict[str, ScreenShareAnnotation] = Field(
        default_factory=dict,
        description="Mapping of timestamps to screen share annotation objects, capturing key visual events.",
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_sentinel(cls, data: dict) -> dict:
        """Pre-processing hook to

        Ensure *message_id* has the **UNASSIGNED** sentinel when omitted so
        downstream code can rely on its presence.
        """
        # Guarantee sentinel for id ------------------------------------------------
        data.setdefault("message_id", UNASSIGNED)
        data.setdefault("exchange_id", UNASSIGNED)
        return data

    # Don’t serialise the sentinel value when POSTing
    def to_post_json(self) -> dict:
        """Dump payload for POST; omit the dummy id."""
        exclude = set()
        if self.exchange_id == UNASSIGNED:
            exclude.add("exchange_id")
        if self.message_id == UNASSIGNED:
            exclude.add("message_id")

        payload = self.model_dump(mode="json", exclude=exclude)

        return payload


VALID_MEDIA: tuple[str, ...] = tuple(m.value for m in Medium)
