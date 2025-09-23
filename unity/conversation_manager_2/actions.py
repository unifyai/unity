import os
from typing import Literal, Optional, Union

import redis
import aiohttp
from pydantic import BaseModel, Field


# conductor
class AskConductor(BaseModel):
    action_name: Literal["ask_conductor"]
    query: str


# wait
class WaitForNextEvent(BaseModel):
    action_name: Literal["wait"]


# comms actions (main user)
# whatsapp has some issues, will deal with it later
# class SendWhatsapp(BaseModel):
#     ...


class SendEmail(BaseModel):
    action_name: Literal["send_email"]
    subject: str
    body: str


class SendSMS(BaseModel):
    action_name: Literal["send_sms"]
    number_or_id: str = Field(
        ..., description="Exact number or contact id of the contact to sms"
    )
    message: str


class MakeCall(BaseModel):
    action_name: Literal["make_call"]
    number_or_id: str = Field(
        ..., description="Exact number or contact id of the contact to call"
    )


actions = Union[WaitForNextEvent, SendSMS, MakeCall]


class ResponsePhone(BaseModel):
    thoughts: str
    phone_utterance: str
    actions: Optional[list[actions]]


class Response(BaseModel):
    thoughts: str
    actions: Optional[list[actions]]


RESPONSES_MODEL = {"call": ResponsePhone, "gmeet": ResponsePhone, "text": Response}


headers = {"Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY')}"}


async def _send_sms_message_via_number(to_number: str, message: str) -> str:
    """
    Send an SMS message using the SMS provider API.

    Args:
        to_number: The recipient's phone number
        message: The message content to send

    Returns:
        str: The response from the SMS API
    """
    from_number = os.getenv("ASSISTANT_NUMBER")

    print(f"Sending SMS from {from_number} to {to_number}: {message}")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{os.getenv('UNITY_COMMS_URL')}/phone/send-text",
            headers=headers,
            json={
                "From": from_number,
                "To": to_number,
                "Body": message,
            },
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                print(e)
            response_text = await response.text()
            print(f"Response: {response_text}")
            return response_text
