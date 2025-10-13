import os
from typing import Literal, Optional, Union
import asyncio
import aiohttp
from pydantic import BaseModel, Field, create_model


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
    """Comms method to send emails"""

    action_name: Literal["send_email"]
    contact_id: str = Field(
        ...,
        description="contact id, should be -1 if you can not infer the contact from the active conversation, otherwise the contact's id as shown in active conversations",
    )
    first_name: str
    surname: Optional[str]
    email_address: str
    subject: str
    body: str


class SendSMS(BaseModel):
    """Comms method to send sms"""

    action_name: Literal["send_sms"]
    contact_id: str = Field(
        ...,
        description="contact id, should be -1 if you can not infer the contact from the active conversation, otherwise the contact's id as shown in active conversations",
    )
    first_name: str
    surname: Optional[str]
    phone_number: str
    message: str


class MakeCall(BaseModel):
    """Comms method to make outbound calls"""

    action_name: Literal["make_call"]
    contact_id: str = Field(
        ...,
        description="contact id, should be -1 if you can not infer the contact from the active conversation, otherwise the contact's id as shown in active conversations",
    )
    first_name: Optional[str]
    surname: Optional[str]
    phone_number: str


class SendUnifyMessage(BaseModel):
    """Send a message to the boss chat (no-phone medium)"""

    action_name: Literal["send_unify_message"]
    message: str
    contact_id: Literal["1"] = "1"


def build_dynamic_response_models(
    include_email: bool = True,
    include_sms: bool = True,
    include_call: bool = True,
):
    """
    Dynamically create response models with conditional actions based on available contact info.

    Args:
        include_email: Whether SendEmail action should be available
        include_sms: Whether SendSMS action should be available
        include_call: Whether MakeCall action should be available

    Returns:
        dict: Response models for different modes (call, gmeet, text)
    """
    # Build list of available action types
    available_actions = [AskConductor, WaitForNextEvent]  # Always available

    if include_email:
        available_actions.append(SendEmail)
    if include_sms:
        available_actions.append(SendSMS)
    if include_call:
        available_actions.append(MakeCall)
    # Unify message is always available for text mode
    available_actions.append(SendUnifyMessage)

    # Create dynamic Union of available actions
    ActionsUnion = Union[tuple(available_actions)]

    # Dynamically create Response model for text mode
    DynamicResponse = create_model(
        "DynamicResponse",
        thoughts=(str, ...),
        actions=(Optional[list[ActionsUnion]], ...),
        __base__=BaseModel,
    )

    # Dynamically create ResponsePhone model for call/gmeet modes
    DynamicResponsePhone = create_model(
        "DynamicResponsePhone",
        thoughts=(str, ...),
        phone_utterance=(str, ...),
        actions=(Optional[list[ActionsUnion]], ...),
        __base__=BaseModel,
    )

    return {
        "call": DynamicResponsePhone,
        "gmeet": DynamicResponsePhone,
        "text": DynamicResponse,
    }


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
                return {"success": False}
            return await response.json()


async def _send_email_via_address(
    to_email: str,
    subject: str,
    content: str,
    message_id: str = None,
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
        f"Sending email from {from_email} to {to_email}: {content}, {subject} {message_id}",
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
            try:
                response.raise_for_status()
            except Exception:
                return {"success": False}
            return await response.json()


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
            try:
                response.raise_for_status()
            except Exception:
                return {
                    "success": False,
                    "error": f"Failed to initiate call to {to_number}",
                }
            return await response.json()


async def add_email_attachments(
    attachments: list[dict[str, str]],
    receiver_email: str,
    gmail_message_id: str,
) -> None:
    """
    Download attachments via the /attachment endpoint and write placeholder files.

    Each attachment item should be of the form: {"id": str, "filename": str}
    For now, writes an empty placeholder file with the same filename.
    """
    if not attachments:
        return

    print("Saving email attachments...")
    async with aiohttp.ClientSession() as session:
        for att in attachments:
            try:
                att_id = att.get("id", "")
                raw_filename = att.get("filename") or f"attachment_{att_id}"
                # very basic filename sanitization
                safe_filename = os.path.basename(raw_filename)

                url = f"{os.getenv('UNITY_COMMS_URL')}/email/attachment"
                params = {
                    "receiver_email": receiver_email,
                    "gmail_message_id": gmail_message_id,
                    "attachment_id": att_id,
                    # "filename": safe_filename,
                }

                async with session.get(url, headers=headers, params=params) as resp:
                    data = await resp.read()

                from unity.file_manager.file_manager import FileManager

                file_manager = FileManager()
                await asyncio.to_thread(
                    file_manager.save_file_to_downloads,
                    safe_filename,
                    data,
                )

                print(
                    f"Downloaded attachment {safe_filename} (size={len(data)} bytes) — placeholder file written",
                )
            except Exception as e:
                print(f"Failed to fetch/write attachment '{att}': {e}")
