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
    email_or_id: str =  Field(
        ..., description="Exact email or contact id of the contact to email"
    )
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


actions = Union[WaitForNextEvent, SendSMS, SendEmail, MakeCall]


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


async def _send_email_via_address(
    to_email: str,
    subject: str,
    content: str,
    message_id: str=None,
) -> str:
    """
    Send an SMS message using the SMS provider API.

    Args:
        to_email: The email address to send the email to
        subject: The subject of the email
        content: The message content to send
        message_id: The message ID of the email to reply to

    Returns:
        str: The response from the email API
    """
    from_email = os.getenv("ASSISTANT_EMAIL")

    print(
        f"Sending email from {from_email} to {to_email}: {content}, {subject} {message_id}"
    )
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{os.getenv('UNITY_COMMS_URL')}/email/send",
            headers=headers,
            json={
                "from": from_email,
                "to": to_email,
                "subject": subject,
                "body": content,
                "in_reply_to": message_id,
            },
        ) as response:
            response.raise_for_status()
            response_text = await response.text()
            print(f"Response: {response_text}")
            return response_text

async def _start_call(
    from_number: str,
    to_number: str,
) -> str:
    """
    Send a call using the call provider API.

    Args:
        from_number: The sender's phone number
        to_number: The recipient's phone number

    Returns:
        str: The response
    """
    print(f"Sending call from {from_number} to {to_number}")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{os.getenv('UNITY_COMMS_URL')}/phone/send-call",
            headers=headers,
            json={"From": from_number, "To": to_number, "NewCall": "true"},
        ) as response:
            response.raise_for_status()
            response_text = await response.text()
            print(f"Response: {response_text}")
            return response_text