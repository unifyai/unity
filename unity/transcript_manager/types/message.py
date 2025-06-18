from enum import StrEnum
from pydantic import BaseModel, Field


class Medium(StrEnum):
    SMS_MESSAGE = "sms_message"
    EMAIL = "email"
    WHATSAPP_MSG = "whatsapp_message"
    PHONE_CALL = "phone_call"
    WHATSAPP_CALL = "whatsapp_call"


class Message(BaseModel):
    message_id: int = Field(description="Unique identifier for the message")
    medium: Medium = Field(
        description="The communication channel used for this message",
    )
    sender_id: int = Field(description="ID of the contact who sent the message")
    receiver_id: int = Field(description="ID of the contact who received the message")
    timestamp: str = Field(
        description="When the message was sent/received in ISO-8601 format",
    )
    content: str = Field(description="The actual text content of the message")
    exchange_id: int = Field(
        description="ID of the conversation thread this message belongs to",
    )


VALID_MEDIA: tuple[str, ...] = tuple(m.value for m in Medium)
