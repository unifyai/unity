from enum import StrEnum
from pydantic import BaseModel, Field, model_validator, field_validator
from datetime import datetime
import re
from typing import Literal

UNASSIGNED = -1


class Medium(StrEnum):
    SMS_MESSAGE = "sms_message"
    EMAIL = "email"
    WHATSAPP_MSG = "whatsapp_message"
    PHONE_CALL = "phone_call"
    WHATSAPP_CALL = "whatsapp_call"
    UNIFY_MESSAGE = "unify_message"
    UNIFY_CALL = "unify_call"
    GOOGLE_MEET = "google_meet"
    UNIFY_MEET = "unify_meet"


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
    images: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Mapping of json.dumps strings like '[x:y]' → image_id (int). "
            "Supports negative indices and open-ended ranges (e.g., '[6:]', '[:10]')."
        ),
    )
    call_utterance_timestamp: str = Field(
        default="",
        description="Timestamp of the utterance associated with calls",
    )
    call_url: str = Field(
        default="",
        description="URL of the recorded call file associated with the call",
    )
    screen_share: dict[str, ScreenShareAnnotation] = Field(
        default_factory=dict,
        description="Mapping of timestamps to screen share annotation objects, capturing key visual events.",
    )

    @field_validator("images", mode="before")
    @classmethod
    def _validate_images(cls, v):
        """Ensure images is a dict[str, int] with keys like "[x:y]".

        Rules:
        - Key must strictly match "[x:y]" with optional negative or open ends.
          Regex: ^\[\s*(-?\d+)?\s*:\s*(-?\d+)?\s*\]$
        - Value must be coercible to int (image_id).
        - None → {}.
        """
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise TypeError("images must be a dict[str, int]")
        pattern = re.compile(r"^\[\s*(-?\d+)?\s*:\s*(-?\d+)?\s*\]$")
        out: dict[str, int] = {}
        for k, val in v.items():
            if not isinstance(k, str):
                raise ValueError("images keys must be strings like '[x:y]'")
            if not pattern.fullmatch(k):
                raise ValueError(
                    f"images key '{k}' must match '[x:y]' with optional negative or open bounds",
                )
            try:
                out[k] = int(val)
            except Exception as exc:
                raise ValueError(
                    f"images value for key '{k}' must be an integer image_id",
                ) from exc
        return out

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
