from typing import Literal
from pydantic import BaseModel
from datetime import datetime


class Message(BaseModel):
    medium: Literal[
        "sms message",
        "email",
        "whatsapp message",
        "phone_call",
        "whatsapp_call",
    ]
    sender: str
    receiver: str
    timestamp: datetime
    content: str
    exchange_id: int
