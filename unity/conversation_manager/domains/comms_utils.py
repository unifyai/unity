from dotenv import load_dotenv
import asyncio
import base64
import aiohttp
import json
import mimetypes
import os
from pathlib import Path
import uuid

from unity.logger import LOGGER
from unity.common.hierarchical_logger import ICONS
from unity.session_details import SESSION_DETAILS
from unity.settings import SETTINGS

load_dotenv()
headers = {"Authorization": f"Bearer {SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()}"}

# Lazily initialized publisher (avoids import-time GCP auth failures in tests)
_publisher = None


def _get_publisher():
    """Get or create the GCP Pub/Sub publisher client."""
    global _publisher
    if _publisher is None:
        from google.cloud import pubsub_v1

        _publisher = pubsub_v1.PublisherClient()
    return _publisher


def _use_local_comms() -> bool:
    enabled = getattr(SETTINGS.conversation, "LOCAL_COMMS_ENABLED", False)
    mode = getattr(SETTINGS.conversation, "LOCAL_COMMS_MODE", "hosted")
    return enabled is True or (isinstance(mode, str) and mode == "local")


def _local_comms_base_url() -> str:
    public_url = SETTINGS.conversation.LOCAL_COMMS_PUBLIC_URL.strip()
    if public_url:
        return public_url.rstrip("/")
    return (
        f"http://{SETTINGS.conversation.LOCAL_COMMS_HOST}:"
        f"{SETTINGS.conversation.LOCAL_COMMS_PORT}"
    )


async def _publish_local_outbox_async(payload: dict) -> bool:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_local_comms_base_url()}/local/comms/outbox",
            json=payload,
        ) as response:
            return response.status < 400


def _publish_local_outbox_sync(payload: dict) -> bool:
    import requests

    response = requests.post(
        f"{_local_comms_base_url()}/local/comms/outbox",
        json=payload,
        timeout=10,
    )
    return response.status_code < 400


def _inline_attachment_bytes(attachment: dict) -> bytes | None:
    encoded = attachment.get("content_base64")
    if not encoded:
        return None
    if attachment.get("content_encoding") == "hex":
        return bytes.fromhex(encoded)
    return base64.b64decode(encoded.encode("ascii"))


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

    if _use_local_comms():
        from unity.conversation_manager.local_providers import twilio as local_twilio

        try:
            return await local_twilio.send_sms_message(to_number, from_number, content)
        except Exception as e:
            LOGGER.error(f"{ICONS['comms_outbound']} {e}")
            return {"success": False, "error": str(e)}

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
                LOGGER.error(f"{ICONS['comms_outbound']} {e}")
                return {"success": False}
            return await response.json()


async def send_whatsapp_message(
    to_number: str,
    content: str,
    user_name: str = "",
    agent_name: str = "",
    media_url: str | None = None,
) -> dict:
    """
    Send a WhatsApp message via the Communication service.

    Communication automatically handles the WhatsApp 24h session window:
    if the window is open, ``content`` is sent as free-form text; if closed,
    it falls back to an approved greeting template with ``content`` appended.

    Args:
        to_number: The recipient's WhatsApp number (E.164)
        content: The message content to send
        user_name: Recipient's first name (used in template fallback)
        agent_name: Assistant's first name (used in template fallback)
        media_url: Publicly accessible URL of a media attachment (one per
            message — WhatsApp constraint).  Supported types: images, audio,
            video, PDF, DOC/XLSX when inside the 24h window.

    Returns:
        dict with 'success' key indicating delivery status.
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    if agent_id is None:
        return {"success": False}

    if _use_local_comms():
        from unity.conversation_manager.local_providers import twilio as local_twilio

        from_number = (
            SESSION_DETAILS.assistant.whatsapp_number
            or SESSION_DETAILS.assistant.number
        )
        if not from_number:
            return {"success": False, "error": "No WhatsApp sender number configured"}
        try:
            return await local_twilio.send_whatsapp_message(
                to_number=to_number,
                from_number=from_number,
                body=content,
                media_url=media_url,
            )
        except Exception as e:
            LOGGER.error(f"{ICONS['comms_outbound']} WhatsApp send failed: {e}")
            return {"success": False, "error": str(e)}

    payload = {
        "to": to_number,
        "body": content,
        "assistant_id": agent_id,
        "user_name": user_name,
        "agent_name": agent_name,
    }
    if media_url:
        payload["media_url"] = media_url

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SETTINGS.conversation.COMMS_URL}/whatsapp/send",
            headers=headers,
            json=payload,
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                LOGGER.error(f"{ICONS['comms_outbound']} WhatsApp send failed: {e}")
                return {"success": False}
            return await response.json()


async def send_unify_message(
    content: str,
    contact_id: int = 1,
    attachment: dict | None = None,
) -> dict:
    """
    Send a unify message to a contact, optionally with an attachment.

    Args:
        content: The message content to send.
        contact_id: The target contact's ID. Defaults to 1 (boss).
        attachment: Optional attachment dict with keys:
            - id: Unique identifier for the attachment
            - filename: The name of the file
            - url: Signed URL to download the file

    Returns:
        dict with "success" key indicating delivery status.
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    event_data = {"content": content, "role": "assistant", "contact_id": contact_id}
    if attachment:
        event_data["attachments"] = [attachment]

    if _use_local_comms():
        success = await _publish_local_outbox_async(
            {
                "thread": "unify_message_outbound",
                "event": event_data,
            },
        )
        return {"success": success}

    env_suffix = SETTINGS.ENV_SUFFIX if agent_id is not None else ""
    topic_name = f"unity-{agent_id}{env_suffix}"
    publisher = _get_publisher()
    topic_path = publisher.topic_path(SETTINGS.GCP_PROJECT_ID, topic_name)

    message_data = {
        "thread": "unify_message_outbound",
        "event": event_data,
    }
    try:
        # Publish with attributes
        future = publisher.publish(
            topic_path,
            json.dumps(message_data).encode("utf-8"),
            thread="unify_message_outbound",
        )
        message_id = future.result()
        LOGGER.debug(
            f"{ICONS['comms_outbound']} Unify message published with ID: {message_id}",
        )
        if message_id:
            return {"success": True}
        else:
            return {"success": False}
    except Exception as e:
        LOGGER.error(f"{ICONS['comms_outbound']} Error sending unify message: {e}")
        return {"success": False, "error": str(e)}


def publish_system_error(error_message: str, error_type: str = "unknown") -> None:
    """Publish a system error to the assistant's Pub/Sub topic.

    This is a best-effort, fire-and-forget publish used to notify the console
    that the container hit an unrecoverable error (OOM, unhandled exception, etc.)
    so the UI can show a user-friendly warning instead of going silent.

    Args:
        error_message: Human-readable description of the error.
        error_type: Structured error type for console classification. One of:
            ``oom``, ``startup_failed``, ``init_failed``, ``message_failed``,
            ``recovering``, ``unknown``.

    Uses a synchronous publish (no await) so it can be called from both sync
    and async contexts, including signal handlers and thread-pool callbacks.
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    if agent_id is None:
        return

    if _use_local_comms():
        try:
            _publish_local_outbox_sync(
                {
                    "thread": "system_error",
                    "event": {
                        "content": error_message,
                        "error_type": error_type,
                    },
                },
            )
        except Exception as e:
            LOGGER.error(
                f"{ICONS['comms_outbound']} Failed to publish system error: {e}",
            )
        return

    env_suffix = SETTINGS.ENV_SUFFIX if agent_id is not None else ""
    topic_name = f"unity-{agent_id}{env_suffix}"
    try:
        publisher = _get_publisher()
        topic_path = publisher.topic_path(SETTINGS.GCP_PROJECT_ID, topic_name)
        message_data = {
            "thread": "system_error",
            "event": {
                "content": error_message,
                "error_type": error_type,
            },
        }
        future = publisher.publish(
            topic_path,
            json.dumps(message_data).encode("utf-8"),
            thread="system_error",
        )
        future.result(timeout=5)
        LOGGER.debug(
            f"{ICONS['comms_outbound']} Published system error [{error_type}]: {error_message}",
        )
    except Exception as e:
        LOGGER.error(f"{ICONS['comms_outbound']} Failed to publish system error: {e}")


async def complete_api_message(
    api_message_id: str,
    response: str | None = None,
    attachments: list[dict] | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Mark an API message as completed in Orchestra, optionally with a response."""
    orchestra_url = SETTINGS.ORCHESTRA_URL
    body: dict = {"response": response}
    if attachments:
        body["attachments"] = [
            {
                "id": att.get("id", ""),
                "filename": att.get("filename", ""),
                "gs_url": att.get("gs_url", ""),
                "content_type": att.get("content_type"),
                "size_bytes": att.get("size_bytes"),
            }
            for att in attachments
        ]
    if tags:
        body["tags"] = tags
    async with aiohttp.ClientSession() as session:
        async with session.put(
            f"{orchestra_url}/admin/messages/{api_message_id}/complete",
            headers=headers,
            json=body,
        ) as resp:
            try:
                resp.raise_for_status()
            except Exception as e:
                LOGGER.error(
                    f"{ICONS['comms_outbound']} Failed to complete API message: {e}",
                )
                return {"success": False}
            return {"success": True}


async def publish_assistant_desktop_ready(
    binding_id: str,
    desktop_url: str,
    liveview_url: str,
    vm_type: str,
) -> None:
    """Publish desktop-ready notification to the assistant's Pub/Sub topic.

    The Console subscribes to this thread to update the liveview iframe.
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    if _use_local_comms():
        try:
            await _publish_local_outbox_async(
                {
                    "thread": "assistant_desktop_ready",
                    "event": {
                        "binding_id": binding_id,
                        "desktop_url": desktop_url,
                        "liveview_url": liveview_url,
                        "vm_type": vm_type,
                    },
                },
            )
        except Exception as e:
            LOGGER.error(
                f"{ICONS['comms_outbound']} Error publishing assistant_desktop_ready: {e}",
            )
        return

    env_suffix = SETTINGS.ENV_SUFFIX if agent_id is not None else ""
    topic_name = f"unity-{agent_id}{env_suffix}"
    publisher = _get_publisher()
    topic_path = publisher.topic_path(SETTINGS.GCP_PROJECT_ID, topic_name)

    message_data = {
        "thread": "assistant_desktop_ready",
        "event": {
            "binding_id": binding_id,
            "desktop_url": desktop_url,
            "liveview_url": liveview_url,
            "vm_type": vm_type,
        },
    }
    try:
        future = publisher.publish(
            topic_path,
            json.dumps(message_data).encode("utf-8"),
            thread="assistant_desktop_ready",
        )
        future.result()
        LOGGER.debug(
            f"{ICONS['comms_outbound']} Published assistant_desktop_ready to {topic_name}",
        )
    except Exception as e:
        LOGGER.error(
            f"{ICONS['comms_outbound']} Error publishing assistant_desktop_ready: {e}",
        )


async def upload_unify_attachment(
    file_content: bytes,
    filename: str,
    assistant_id: int | None = None,
) -> dict:
    """
    Upload a file attachment for use in outbound Unify messages.

    Args:
        file_content: The raw bytes of the file to upload.
        filename: The name of the file.
        assistant_id: Optional assistant ID for organizing storage.

    Returns:
        dict with attachment details: {"id": str, "filename": str, "url": str}
        or {"success": False, "error": str} on failure.
    """
    if assistant_id is None:
        assistant_id = SESSION_DETAILS.assistant.agent_id

    if _use_local_comms():
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        attachment = {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "content_base64": base64.b64encode(file_content).decode("ascii"),
            "content_type": content_type,
            "size_bytes": len(file_content),
        }
        attachment["url"] = (
            f"{_local_comms_base_url()}/local/comms/attachments/{attachment['id']}"
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_local_comms_base_url()}/local/comms/attachments",
                json=attachment,
            ) as response:
                if response.status >= 400:
                    detail = await response.text()
                    return {"success": False, "error": detail}
        return attachment

    from io import BytesIO

    adapters_url = SETTINGS.conversation.ADAPTERS_URL

    LOGGER.debug(
        f"{ICONS['comms_outbound']} Uploading unify attachment: {filename} ({len(file_content)} bytes)",
    )

    # Create form data for multipart upload
    form_data = aiohttp.FormData()
    form_data.add_field(
        "file",
        BytesIO(file_content),
        filename=filename,
        content_type="application/octet-stream",
    )
    form_data.add_field("assistant_id", str(assistant_id))

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{adapters_url}/unify/attachment",
            headers=headers,
            data=form_data,
        ) as response:
            try:
                body = await response.text()
                if response.status >= 400:
                    try:
                        detail = json.loads(body).get("error", body)
                    except (json.JSONDecodeError, AttributeError):
                        detail = body
                    error_msg = f"Upload rejected ({response.status}): {detail}"
                    LOGGER.debug(
                        f"{ICONS['comms_outbound']} Failed to upload unify attachment: {error_msg}",
                    )
                    return {"success": False, "error": error_msg}
                result = json.loads(body)
                LOGGER.debug(f"{ICONS['comms_outbound']} Uploaded attachment: {result}")
                return result
            except Exception as e:
                LOGGER.debug(
                    f"{ICONS['comms_outbound']} Failed to upload unify attachment: {e}",
                )
                return {"success": False, "error": str(e)}


async def send_discord_message(
    to: str | None = None,
    channel_id: str | None = None,
    body: str = "",
    bot_id: str | None = None,
    media_url: str | None = None,
) -> dict:
    """Send a Discord message via the Communication service.

    Supports two modes:
    - DM: pass ``to`` (Discord user snowflake). The route is resolved
      server-side and a DM channel is opened automatically.
    - Channel reply: pass ``channel_id`` and ``bot_id``. The message is
      posted directly to the channel.

    Args:
        to: Recipient's Discord user ID (for DMs).
        channel_id: Discord channel ID (for channel replies).
        body: The text content to send.
        bot_id: Pool bot ID (required for channel messages, optional for DMs).
        media_url: Optional URL of a media attachment to embed.

    Returns:
        dict with 'success' key and optionally 'message_id', 'channel_id'.
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    if agent_id is None:
        return {"success": False}

    payload: dict = {
        "body": body,
        "assistant_id": agent_id,
    }
    if to:
        payload["to"] = to
    if channel_id:
        payload["channel_id"] = channel_id
    if bot_id:
        payload["bot_id"] = bot_id
    if media_url:
        payload["media_url"] = media_url

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SETTINGS.conversation.COMMS_URL}/discord/send",
            headers=headers,
            json=payload,
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                LOGGER.error(f"{ICONS['comms_outbound']} Discord send failed: {e}")
                return {"success": False}
            result = await response.json()
            result["success"] = True
            return result


async def send_email_via_address(
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    email_id: str | None = None,
    attachment: dict | None = None,
) -> dict:
    """
    Send an email using the email provider API.

    Args:
        to: List of recipient email addresses.
        subject: The subject of the email.
        body: The message body to send.
        cc: Optional list of CC email addresses.
        bcc: Optional list of BCC email addresses.
        email_id: The email identifier of the message to reply to (threading id).
        attachment: Optional attachment dict with keys:
            - filename: The name of the file
            - content_base64: Base64-encoded file contents

    Returns:
        dict: Response with 'success' bool and optionally 'error' message
    """
    if _use_local_comms():
        from unity.conversation_manager.local_providers import email as local_email

        return await local_email.send_email(
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
            email_id=email_id,
            attachment=attachment,
        )

    from_email = SESSION_DETAILS.assistant.email
    if not from_email:
        return {"success": False, "error": "No sender email configured"}

    payload = {
        "from": from_email,
        "to": to,
        "subject": subject,
        "body": body,
        "in_reply_to": email_id,
    }
    if cc:
        payload["cc"] = cc
    if bcc:
        payload["bcc"] = bcc
    if attachment:
        payload["attachment"] = attachment

    provider = SESSION_DETAILS.assistant.email_provider
    send_path = "/outlook/send" if provider == "microsoft_365" else "/gmail/send"

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SETTINGS.conversation.COMMS_URL}{send_path}",
            headers=headers,
            json=payload,
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                return {"success": False, "error": str(e)}
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
    if not from_number:
        return {"success": False}

    from unity.conversation_manager.domains.call_manager import make_room_name

    assistant_id = str(SESSION_DETAILS.assistant.agent_id)
    room_name = make_room_name(assistant_id, "phone")

    if _use_local_comms():
        from unity.conversation_manager.local_providers import twilio as local_twilio

        try:
            return await local_twilio.start_call(to_number, from_number, room_name)
        except Exception:
            return {
                "success": False,
                "error": f"Failed to initiate call to {to_number}",
            }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SETTINGS.conversation.COMMS_URL}/phone/send-call",
            headers=headers,
            json={
                "From": from_number,
                "To": to_number,
                "room_name": room_name,
            },
        ) as response:
            try:
                response.raise_for_status()
            except Exception:
                return {
                    "success": False,
                    "error": f"Failed to initiate call to {to_number}",
                }
            return await response.json()


async def start_whatsapp_call(
    to_number: str,
    agent_name: str,
    room_name: str,
) -> dict:
    """
    Initiate a WhatsApp voice call via the Communication service.

    Communication checks call permission with Orchestra and decides the method:
    - Permission granted → places outbound call directly (returns method: "direct")
    - Permission not granted → sends invite template (returns method: "invite")

    Args:
        to_number: The recipient's WhatsApp number (E.164)
        agent_name: Assistant's first name (used in invite template)
        room_name: Pre-built LiveKit room name

    Returns:
        dict with 'success', 'method' ("direct"|"invite"), and other fields.
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    if agent_id is None:
        return {"success": False}

    if _use_local_comms():
        from unity.conversation_manager.local_providers import twilio as local_twilio

        from_number = (
            SESSION_DETAILS.assistant.whatsapp_number
            or SESSION_DETAILS.assistant.number
        )
        if not from_number:
            return {"success": False, "error": "No WhatsApp sender number configured"}
        try:
            return await local_twilio.start_whatsapp_call(
                to_number=to_number,
                from_number=from_number,
                room_name=room_name,
            )
        except Exception:
            return {
                "success": False,
                "error": f"Failed to initiate WhatsApp call to {to_number}",
            }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SETTINGS.conversation.COMMS_URL}/whatsapp/send-call",
            headers=headers,
            json={
                "to": to_number,
                "assistant_id": agent_id,
                "agent_name": agent_name,
                "room_name": room_name,
            },
        ) as response:
            try:
                response.raise_for_status()
            except Exception:
                return {
                    "success": False,
                    "error": f"Failed to initiate WhatsApp call to {to_number}",
                }
            return await response.json()


async def add_email_attachments(
    attachments: list[dict[str, str]],
    receiver_email: str,
    message_id: str,
) -> None:
    """
    Download email attachments and save to Attachments folder.

    Each attachment item should be of the form: {"id": str, "filename": str}
    """
    if not attachments:
        return

    provider = SESSION_DETAILS.assistant.email_provider
    is_outlook = provider == "microsoft_365"

    LOGGER.debug(f"{ICONS['comms_outbound']} Saving email attachments...")
    async with aiohttp.ClientSession() as session:
        for att in attachments:
            try:
                att_id = att.get("id", "")
                raw_filename = att.get("filename") or f"attachment_{att_id}"
                safe_filename = os.path.basename(raw_filename)
                inline_bytes = _inline_attachment_bytes(att)
                if inline_bytes is not None:
                    data = inline_bytes
                else:
                    if is_outlook:
                        url = f"{SETTINGS.conversation.COMMS_URL}/outlook/attachment"
                        params = {
                            "user_email": receiver_email,
                            "message_id": message_id,
                            "attachment_id": att_id,
                        }
                    else:
                        url = f"{SETTINGS.conversation.COMMS_URL}/gmail/attachment"
                        params = {
                            "receiver_email": receiver_email,
                            "gmail_message_id": message_id,
                            "attachment_id": att_id,
                        }
                    async with session.get(url, headers=headers, params=params) as resp:
                        data = await resp.read()

                from unity.manager_registry import ManagerRegistry

                file_manager = ManagerRegistry.get_file_manager()
                await asyncio.to_thread(
                    file_manager.save_attachment,
                    att_id,
                    safe_filename,
                    data,
                )

                LOGGER.debug(
                    f"{ICONS['comms_outbound']} Downloaded email attachment {safe_filename} (size={len(data)} bytes)",
                )
            except Exception as e:
                LOGGER.error(
                    f"{ICONS['comms_outbound']} Failed to fetch/write attachment '{att}': {e}",
                )


async def _get_signed_url_from_gs_url(
    session: aiohttp.ClientSession,
    gs_url: str,
) -> str:
    """
    Request a signed URL from Orchestra for a gs:// path.

    Args:
        session: aiohttp session for making requests
        gs_url: The gs:// URL to get a signed URL for

    Returns:
        The signed HTTPS URL for downloading the file
    """
    orchestra_url = SETTINGS.ORCHESTRA_URL
    # Use the user's API key (not admin key) for Orchestra API calls
    user_api_key = SESSION_DETAILS.unify_key
    user_headers = {"Authorization": f"Bearer {user_api_key}"}
    async with session.post(
        f"{orchestra_url}/storage/signed-url",
        headers=user_headers,
        json={"gcs_uri": gs_url},
    ) as resp:
        resp.raise_for_status()
        result = await resp.json()
        return result.get("signed_url", "")


async def _download_single_attachment(
    session: aiohttp.ClientSession,
    att: dict[str, str],
    adapter,
) -> str | None:
    """Download one attachment and write it to disk. Returns the display name, or None on failure."""
    att_id = att.get("id", "")
    raw_filename = att.get("filename") or f"attachment_{att_id}"
    safe_filename = os.path.basename(raw_filename)

    target_path = Path(adapter._root) / "Attachments" / f"{att_id}_{safe_filename}"
    if target_path.exists() and target_path.stat().st_size > 0:
        LOGGER.debug(
            f"{ICONS['comms_outbound']} Attachment {safe_filename} already on disk, skipping download",
        )
        return f"Attachments/{att_id}_{safe_filename}"

    url = att.get("url")
    gs_url = att.get("gs_url")

    if not url and gs_url:
        try:
            url = await _get_signed_url_from_gs_url(session, gs_url)
        except Exception as e:
            LOGGER.error(
                f"{ICONS['comms_outbound']} Failed to get signed URL for {gs_url}: {e}",
            )
            url = None

    inline_bytes = _inline_attachment_bytes(att)

    if inline_bytes is not None:
        data = inline_bytes
    elif url:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.read()
    else:
        data = b""

    display_name = await asyncio.to_thread(
        adapter.save_attachment,
        att_id,
        safe_filename,
        data,
    )

    LOGGER.debug(
        f"{ICONS['comms_outbound']} Downloaded unify attachment {safe_filename} "
        f"(size={len(data)} bytes)",
    )
    return display_name


async def add_unify_message_attachments(
    attachments: list[dict[str, str]],
) -> None:
    """
    Download attachments from Unify console messages and save to Attachments folder.

    Each attachment item should be of the form:
        {"id": str, "filename": str, "url": str}
    or with gs_url for on-demand signed URL generation:
        {"id": str, "filename": str, "gs_url": str}

    If gs_url is present but url is not, a signed URL will be generated
    from Orchestra before downloading.

    All downloads run in parallel, then ingestion (parse/index/embed) runs
    afterward so files are immediately available to the assistant.
    """
    if not attachments:
        return

    from unity.manager_registry import ManagerRegistry

    LOGGER.debug(f"{ICONS['comms_outbound']} Saving unify message attachments...")

    file_manager = ManagerRegistry.get_file_manager()
    adapter = file_manager._adapter

    # Phase 1: Download all files to disk in parallel.
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *(
                _download_single_attachment(session, att, adapter)
                for att in attachments
            ),
            return_exceptions=True,
        )

    saved_display_names: list[str] = []
    for att, result in zip(attachments, results):
        if isinstance(result, BaseException):
            LOGGER.error(
                f"{ICONS['comms_outbound']} Failed to download unify attachment '{att}': {result}",
            )
        elif result is not None:
            saved_display_names.append(result)

    # Phase 2: Ingest all saved files (parse, index, embed) in parallel.
    # Files are already on disk and accessible to the assistant.
    # Gated behind IMPLICIT_INGESTION because the Docling pipeline can
    # consume multiple GB of memory per file and OOM the container before
    # the CodeActActor gets a chance to process the user's request.
    if saved_display_names and SETTINGS.file.IMPLICIT_INGESTION:
        try:
            from unity.file_manager.types.config import FilePipelineConfig

            cfg = FilePipelineConfig()
            cfg.execution.parallel_files = True
            await asyncio.to_thread(
                file_manager.ingest_files,
                saved_display_names,
                config=cfg,
            )
        except Exception as e:
            LOGGER.error(
                f"{ICONS['comms_outbound']} Failed to ingest downloaded attachments: {e}",
            )
