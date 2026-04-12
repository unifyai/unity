from __future__ import annotations

import asyncio
import base64
from datetime import datetime
import mimetypes
import os
from urllib.parse import quote_plus
import uuid

import httpx
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import VoiceResponse

from unity.settings import SETTINGS

from .livekit import ensure_phone_dispatch_rule, make_sip_uri

_RINGTONE_URL = "https://auburn-eagle-6359.twil.io/assets/ring-tone-68676.mp3"


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def get_twilio_client() -> TwilioClient:
    """Build the standard Twilio client."""
    return TwilioClient(
        _required_env("TWILIO_ACCOUNT_SID"),
        _required_env("TWILIO_AUTH_TOKEN"),
    )


def get_twilio_wa_client() -> TwilioClient:
    """Build the WhatsApp Business Twilio client."""
    account_sid = os.environ.get("TWILIO_WA_ACCOUNT_SID", "").strip()
    auth_token = os.environ.get("TWILIO_WA_AUTH_TOKEN", "").strip()
    if account_sid and auth_token:
        return TwilioClient(account_sid, auth_token)
    return get_twilio_client()


def local_public_url() -> str:
    """Resolve the externally reachable base URL for local comms callbacks."""
    public_url = SETTINGS.conversation.LOCAL_COMMS_PUBLIC_URL.strip()
    if public_url:
        return public_url.rstrip("/")
    host = SETTINGS.conversation.LOCAL_COMMS_HOST
    port = SETTINGS.conversation.LOCAL_COMMS_PORT
    return f"http://{host}:{port}"


def validate_signature(
    url: str,
    params: dict[str, str],
    signature: str,
    *,
    whatsapp: bool,
) -> bool:
    """Validate a Twilio webhook signature when a token is configured."""
    token_name = "TWILIO_WA_AUTH_TOKEN" if whatsapp else "TWILIO_AUTH_TOKEN"
    auth_token = os.environ.get(token_name, "").strip()
    if not auth_token:
        return True
    validator = RequestValidator(auth_token)
    return validator.validate(url, params, signature)


def empty_message_response() -> str:
    """Return an empty TwiML messaging response."""
    return str(MessagingResponse())


def create_conference_response(
    conference_name: str,
    *,
    with_status: bool = False,
) -> str:
    """Return TwiML that joins the caller to a named conference."""
    response = VoiceResponse()
    dial = response.dial()
    kwargs = {
        "startConferenceOnEnter": True,
        "endConferenceOnExit": True,
        "muted": False,
        "wait_url": _RINGTONE_URL,
    }
    if with_status:
        kwargs["status_callback"] = (
            f"{local_public_url()}/local/twilio/conference-status"
        )
        kwargs["status_callback_event"] = "end"
    dial.conference(conference_name, **kwargs)
    return str(response)


def build_outbound_call_twiml(twilio_number: str, phone_number: str) -> str:
    """Build TwiML that dials a phone number and reports call status back to Unity."""
    response = VoiceResponse()
    dial = response.dial(caller_id=twilio_number, timeout=15)
    dial.number(
        phone_number,
        status_callback=f"{local_public_url()}/local/twilio/call-status",
        status_callback_event="initiated ringing answered completed",
    )
    return str(response)


async def add_sip_leg_to_conference(
    conference_name: str,
    from_number: str,
    *,
    to_uri: str,
    whatsapp: bool = False,
) -> str:
    """Create the SIP-side participant that connects a conference to LiveKit."""
    client = get_twilio_wa_client() if whatsapp else get_twilio_client()

    def _create():
        call = client.calls.create(
            to=to_uri,
            from_=from_number,
            twiml=create_conference_response(conference_name),
        )
        return call.sid

    return await asyncio.to_thread(_create)


async def send_sms_message(to_number: str, from_number: str, body: str) -> dict:
    """Send an SMS directly via Twilio."""
    client = get_twilio_client()

    def _send():
        msg = client.messages.create(to=to_number, from_=from_number, body=body)
        return {"success": True, "sid": msg.sid}

    return await asyncio.to_thread(_send)


async def send_whatsapp_message(
    to_number: str,
    from_number: str,
    body: str,
    *,
    media_url: str | None = None,
) -> dict:
    """Send a WhatsApp message directly via Twilio."""
    client = get_twilio_wa_client()

    def _send():
        kwargs = {
            "to": f"whatsapp:{to_number}",
            "from_": f"whatsapp:{from_number}",
            "body": body,
            "status_callback": f"{local_public_url()}/local/twilio/whatsapp-status",
        }
        if media_url:
            kwargs["media_url"] = [media_url]
        msg = client.messages.create(**kwargs)
        return {"success": True, "sid": msg.sid, "method": "freeform"}

    return await asyncio.to_thread(_send)


async def start_call(
    to_number: str,
    from_number: str,
    room_name: str,
) -> dict:
    """Initiate an outbound phone call bridged through LiveKit."""
    await ensure_phone_dispatch_rule(from_number, room_name)
    client = get_twilio_client()
    sip_uri = make_sip_uri(from_number)
    callback_url = (
        f"{local_public_url()}/local/twilio/twiml"
        f"?phone_number={quote_plus(to_number)}"
    )

    def _start():
        call = client.calls.create(
            to=sip_uri,
            from_=from_number,
            url=callback_url,
        )
        return {"success": True, "call_sid": call.sid}

    return await asyncio.to_thread(_start)


async def start_whatsapp_call(
    to_number: str,
    from_number: str,
    room_name: str,
) -> dict:
    """Initiate an outbound WhatsApp Business voice call bridged through LiveKit."""
    await ensure_phone_dispatch_rule(from_number, room_name)
    client = get_twilio_wa_client()
    conference_name = (
        f"Unity_WA_{from_number.removeprefix('+')}_"
        f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    )
    sip_uri = make_sip_uri(from_number)

    def _start():
        response = create_conference_response(conference_name)
        user_call = client.calls.create(
            to=f"whatsapp:{to_number}",
            from_=f"whatsapp:{from_number}",
            twiml=response,
            status_callback=f"{local_public_url()}/local/twilio/whatsapp-call-status",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
        )
        client.calls.create(
            to=sip_uri,
            from_=from_number,
            twiml=response,
        )
        return {
            "success": True,
            "method": "direct",
            "conference_name": conference_name,
            "call_sid": user_call.sid,
        }

    return await asyncio.to_thread(_start)


async def fetch_whatsapp_attachments(form_data: dict[str, str]) -> list[dict]:
    """Download WhatsApp media from Twilio and return inline attachments."""
    num_media = int(form_data.get("NumMedia", "0") or "0")
    if num_media == 0:
        return []

    account_sid = (
        os.environ.get("TWILIO_WA_ACCOUNT_SID", "").strip()
        or os.environ.get(
            "TWILIO_ACCOUNT_SID",
            "",
        ).strip()
    )
    auth_token = (
        os.environ.get("TWILIO_WA_AUTH_TOKEN", "").strip()
        or os.environ.get(
            "TWILIO_AUTH_TOKEN",
            "",
        ).strip()
    )
    if not account_sid or not auth_token:
        return []

    attachments: list[dict] = []
    async with httpx.AsyncClient(
        auth=(account_sid, auth_token),
        follow_redirects=True,
    ) as client:
        for index in range(num_media):
            media_url = form_data.get(f"MediaUrl{index}", "")
            content_type = form_data.get(
                f"MediaContentType{index}",
                "application/octet-stream",
            )
            if not media_url:
                continue
            response = await client.get(media_url, timeout=30.0)
            response.raise_for_status()
            extension = mimetypes.guess_extension(content_type) or ""
            attachment_id = str(uuid.uuid4())
            filename = f"whatsapp_media_{attachment_id[:8]}{extension}"
            attachments.append(
                {
                    "id": attachment_id,
                    "filename": filename,
                    "content_base64": base64.b64encode(response.content).decode(
                        "ascii",
                    ),
                    "content_type": content_type,
                    "size_bytes": len(response.content),
                },
            )
    return attachments
