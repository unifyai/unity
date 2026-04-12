from __future__ import annotations

import os
from datetime import datetime, timezone
from urllib.parse import quote_plus

from livekit.api import (
    CreateSIPDispatchRuleRequest,
    EncodedFileOutput,
    GCPUpload,
    LiveKitAPI,
    RoomCompositeEgressRequest,
    SIPDispatchRuleInfo,
    TokenVerifier,
    WebhookConfig,
    WebhookReceiver,
)
from livekit.protocol.sip import (
    ListSIPDispatchRuleRequest,
    ListSIPInboundTrunkRequest,
    SIPDispatchRule,
    SIPDispatchRuleDirect,
)

from unity.settings import SETTINGS


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def get_livekit_api() -> LiveKitAPI:
    """Build a LiveKit API client from the current environment."""
    return LiveKitAPI(
        url=_required_env("LIVEKIT_URL"),
        api_key=_required_env("LIVEKIT_API_KEY"),
        api_secret=_required_env("LIVEKIT_API_SECRET"),
    )


def make_sip_uri(phone_number: str) -> str:
    """Build the SIP URI that bridges a Twilio call into LiveKit."""
    sip_domain = _required_env("LIVEKIT_SIP_URI")
    normalized = phone_number if phone_number.startswith("+") else f"+{phone_number}"
    return f"sip:{normalized}@{sip_domain}"


async def ensure_phone_dispatch_rule(phone_number: str, room_name: str) -> None:
    """Ensure LiveKit routes calls for a phone number into the target room."""
    livekit_api = get_livekit_api()
    try:
        normalized = (
            phone_number if phone_number.startswith("+") else f"+{phone_number}"
        )
        trunks = await livekit_api.sip.list_sip_inbound_trunk(
            ListSIPInboundTrunkRequest(),
        )
        trunk_id = None
        for trunk in trunks.items:
            if normalized in list(trunk.numbers):
                trunk_id = trunk.sip_trunk_id
                break
        if trunk_id is None:
            raise RuntimeError(f"No inbound SIP trunk configured for {normalized}")

        rules = await livekit_api.sip.list_sip_dispatch_rule(
            ListSIPDispatchRuleRequest(),
        )
        for rule in rules.items:
            if trunk_id not in list(rule.trunk_ids):
                continue
            if (
                rule.rule.HasField("dispatch_rule_direct")
                and rule.rule.dispatch_rule_direct.room_name == room_name
            ):
                return
            await livekit_api.sip.delete_sip_dispatch_rule(rule.sip_dispatch_rule_id)

        await livekit_api.sip.create_sip_dispatch_rule(
            CreateSIPDispatchRuleRequest(
                dispatch_rule=SIPDispatchRuleInfo(
                    rule=SIPDispatchRule(
                        dispatch_rule_direct=SIPDispatchRuleDirect(room_name=room_name),
                    ),
                    name=f"Unity_phone_{normalized}",
                    trunk_ids=[trunk_id],
                ),
            ),
        )
    finally:
        await livekit_api.aclose()


def _local_public_url() -> str:
    public_url = SETTINGS.conversation.LOCAL_COMMS_PUBLIC_URL.strip()
    if public_url:
        return public_url.rstrip("/")
    host = SETTINGS.conversation.LOCAL_COMMS_HOST
    port = SETTINGS.conversation.LOCAL_COMMS_PORT
    return f"http://{host}:{port}"


async def start_room_egress(
    room_name: str,
    assistant_id: str | int,
    user_id: str | int = "",
) -> None:
    """Start an audio-only room egress and callback Unity when it completes."""
    gcs_credentials = _required_env("GCP_SA_KEY")
    gcs_bucket = os.environ.get("LIVEKIT_EGRESS_GCS_BUCKET", "").strip()
    if not gcs_bucket:
        raise RuntimeError("LIVEKIT_EGRESS_GCS_BUCKET must be set")

    livekit_api = get_livekit_api()
    try:
        prefix = SETTINGS.DEPLOY_ENV
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        filepath = f"{prefix}/{assistant_id}/{room_name}_{timestamp}.mp3"
        webhook_url = (
            f"{_local_public_url()}/local/livekit/recording-complete"
            f"?assistant_id={quote_plus(str(assistant_id))}"
            f"&user_id={quote_plus(str(user_id))}"
            f"&room_name={quote_plus(room_name)}"
        )
        egress_request = RoomCompositeEgressRequest(
            room_name=room_name,
            audio_only=True,
            file_outputs=[
                EncodedFileOutput(
                    file_type=3,
                    filepath=filepath,
                    gcp=GCPUpload(
                        credentials=gcs_credentials,
                        bucket=gcs_bucket,
                    ),
                ),
            ],
            webhooks=[
                WebhookConfig(
                    url=webhook_url,
                    signing_key=_required_env("LIVEKIT_API_KEY"),
                ),
            ],
        )
        await livekit_api.egress.start_room_composite_egress(egress_request)
    finally:
        await livekit_api.aclose()


def verify_livekit_webhook(body: str, auth_token: str):
    """Verify a LiveKit webhook payload and return the parsed event."""
    receiver = WebhookReceiver(
        TokenVerifier(
            api_key=_required_env("LIVEKIT_API_KEY"),
            api_secret=_required_env("LIVEKIT_API_SECRET"),
        ),
    )
    return receiver.receive(body, auth_token)
