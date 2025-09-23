from enum import StrEnum
from pydantic import BaseModel, Field, model_validator
from datetime import datetime

UNASSIGNED = -1


class Medium(StrEnum):
    SMS_MESSAGE = "sms_message"
    EMAIL = "email"
    WHATSAPP_MSG = "whatsapp_message"
    PHONE_CALL = "phone_call"
    WHATSAPP_CALL = "whatsapp_call"
    UNIFY_CHAT = "unify_chat"


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
