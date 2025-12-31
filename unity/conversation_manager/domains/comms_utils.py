from dotenv import load_dotenv
import json
import asyncio
import aiohttp
import os

from unity.session_details import DEFAULT_ASSISTANT_ID, SESSION_DETAILS
from unity.settings import SETTINGS

load_dotenv()
headers = {"Authorization": f"Bearer {SETTINGS.ORCHESTRA_ADMIN_KEY}"}

# Lazily initialized publisher (avoids import-time GCP auth failures in tests)
_publisher = None


def _get_publisher():
    """Get or create the GCP Pub/Sub publisher client."""
    global _publisher
    if _publisher is None:
        from google.cloud import pubsub_v1

        _publisher = pubsub_v1.PublisherClient()
    return _publisher


async def send_sms_message_via_number(to_number: str, content: str) -> str:
    """
    Send an SMS message using the SMS provider API.

    Args:
        to_number: The recipient's phone number
        content: The message content to send

    Returns:
        str: The response from the SMS API
    """
    from_number = SESSION_DETAILS.assistant.number
    if not from_number:
        return {"success": False}

    print(f"Sending SMS from {from_number} to {to_number}: {content}")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SETTINGS.conversation.COMMS_URL}/phone/send-text",
            headers=headers,
            json={
                "From": from_number,
                "To": to_number,
                "Body": content,
            },
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                print(e)
                return {"success": False}
            return await response.json()


async def send_unify_message(content: str, contact_id: int = 1) -> dict:
    """
    Send a unify message to a contact.

    Args:
        content: The message content to send.
        contact_id: The target contact's ID. Defaults to 1 (boss).

    Returns:
        dict with "success" key indicating delivery status.
    """
    assistant_id = SESSION_DETAILS.assistant.id
    staging_suffix = (
        "-staging"
        if SETTINGS.STAGING and DEFAULT_ASSISTANT_ID not in assistant_id
        else ""
    )
    topic_name = f"unity-{assistant_id}{staging_suffix}"
    publisher = _get_publisher()
    topic_path = publisher.topic_path("responsive-city-458413-a2", topic_name)

    print(f"Sending unify message to contact_id={contact_id}: {content}")
    message_data = {
        "thread": "unify_message_outbound",
        "event": {"content": content, "role": "assistant", "contact_id": contact_id},
    }
    try:
        # Publish with attributes
        future = publisher.publish(
            topic_path,
            json.dumps(message_data).encode("utf-8"),
            thread="unify_message_outbound",
        )
        message_id = future.result()
        print(f"Unify message published with ID: {message_id}")
        if message_id:
            return {"success": True}
        else:
            return {"success": False}
    except Exception as e:
        print(f"Error sending unify message: {e}")
        return {"success": False, "error": str(e)}


async def send_email_via_address(
    to_email: str,
    subject: str,
    body: str,
    email_id: str = None,
) -> str:
    """
    Send an SMS message using the SMS provider API.

    Args:
        to_email: The email address to send the email to
        subject: The subject of the email
        body: The message body to send
        email_id: The email identifier of the message to reply to (threading id)

    Returns:
        str: The response from the email API
    """
    from_email = SESSION_DETAILS.assistant.email
    if not from_email:
        return {"success": False}

    print(
        f"Sending email from {from_email} to {to_email}: {body}, {subject} {email_id}",
    )
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SETTINGS.conversation.COMMS_URL}/gmail/send",
            headers=headers,
            json={
                "from": from_email,
                "to": to_email,
                "subject": subject,
                "body": body,
                "in_reply_to": email_id,
            },
        ) as response:
            try:
                response.raise_for_status()
            except Exception:
                return {"success": False}
            return await response.json()


async def start_call(to_number: str) -> str:
    """
    Send a call using the call provider API.

    Args:
        to_number: The recipient's phone number

    Returns:
        str: The response
    """
    from_number = SESSION_DETAILS.assistant.number
    print(f"Sending call from {from_number} to {to_number}")
    if not from_number:
        return {"success": False}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SETTINGS.conversation.COMMS_URL}/phone/send-call",
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

                url = f"{SETTINGS.conversation.COMMS_URL}/gmail/attachment"
                params = {
                    "receiver_email": receiver_email,
                    "gmail_message_id": gmail_message_id,
                    "attachment_id": att_id,
                    # "filename": safe_filename,
                }

                async with session.get(url, headers=headers, params=params) as resp:
                    data = await resp.read()

                from unity.file_manager.managers.local import (
                    LocalFileManager as FileManager,
                )

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
