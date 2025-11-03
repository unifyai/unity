import os
import asyncio
import aiohttp

headers = {"Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY')}"}


async def send_sms_message_via_number(to_number: str, message: str) -> str:
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


async def send_email_via_address(
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


async def start_call(
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

