from enum import StrEnum
from pydantic import BaseModel


class Medium(StrEnum):
    SMS_MESSAGE = "sms_message"
    EMAIL = "email"
    WHATSAPP_MSG = "whatsapp_message"
    PHONE_CALL = "phone_call"
    WHATSAPP_CALL = "whatsapp_call"


class Message(BaseModel):
    medium: Medium
    sender_id: int
    receiver_id: int
    timestamp: str
    content: str
    exchange_id: int


VALID_MEDIA: tuple[str, ...] = tuple(m.value for m in Medium)
