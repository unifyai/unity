import asyncio
import subprocess
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Union

from unity.contact_manager.types.contact import UNASSIGNED
from unity.common.hierarchical_logger import DEFAULT_ICON
from unity.conversation_manager import assistant_jobs
from unity.conversation_manager.events import *
from unity.conversation_manager.domains import managers_utils
from unity.conversation_manager.domains.comms_utils import publish_system_error
from unity.conversation_manager.types import Medium, Mode
from unity.logger import LOGGER
from unity.session_details import SESSION_DETAILS

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager


_AGENT_SERVICE_PID_FILE = Path("/tmp/agent-service.pid")


def _find_agent_service_dir() -> Path | None:
    """Locate the agent-service directory (Docker or local dev)."""
    candidates = [
        Path("/app/agent-service"),
        Path(__file__).resolve().parents[3] / "agent-service",
    ]
    for d in candidates:
        if d.is_dir():
            return d
    return None


def _update_env_file(env_path: Path, key: str, value: str) -> None:
    """Update or add a key=value line in a .env file, preserving other lines."""
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    prefix = f"{key}="
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n")


def _restart_agent_service_with_key(api_key: str) -> None:
    """Kill and restart the agent-service so it picks up the user's API key.

    1. Updates the agent-service .env file (UNIFY_KEY + infrastructure URLs)
    2. Kills the process currently on port 3000
    3. Spawns a new agent-service (inherits os.environ which already has the
       correct UNIFY_KEY from export_to_env())
    4. Writes the new PID to /tmp/agent-service.pid for entrypoint.sh cleanup
    """
    from unity.settings import SETTINGS

    try:
        agent_dir = _find_agent_service_dir()
        if agent_dir is None:
            LOGGER.warning(
                "[agent-service-restart] agent-service directory not found, skipping",
            )
            return

        # 1. Update .env
        env_file = agent_dir / ".env"
        _update_env_file(env_file, "UNIFY_KEY", api_key)
        _update_env_file(env_file, "ORCHESTRA_URL", SETTINGS.ORCHESTRA_URL)
        _update_env_file(env_file, "UNITY_COMMS_URL", SETTINGS.conversation.COMMS_URL)
        LOGGER.info(
            "[agent-service-restart] Updated agent-service .env with user API key"
            " and infrastructure URLs",
        )

        # 2. Kill whatever is listening on port 3000
        subprocess.run(["fuser", "-k", "3000/tcp"], capture_output=True)
        time.sleep(1)
        LOGGER.info(
            "[agent-service-restart] Killed process on port 3000",
        )

        # 3. Restart (same logic as entrypoint.sh)
        compiled = agent_dir / "dist" / "index.js"
        if compiled.exists():
            cmd = ["node", str(compiled)]
        else:
            cmd = ["npx", "ts-node", str(agent_dir / "src" / "index.ts")]

        proc = subprocess.Popen(
            cmd,
            cwd=str(agent_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 4. Write PID file for entrypoint.sh cleanup
        _AGENT_SERVICE_PID_FILE.write_text(str(proc.pid))
        LOGGER.info(
            f"[agent-service-restart] Restarted agent-service (PID: {proc.pid})",
        )
    except Exception as e:
        LOGGER.info(f"[agent-service-restart] Failed (non-fatal): {e}")


def _event_type_to_log_key(event_cls) -> str:
    """Convert an event class name to a log key for icon lookup."""
    name = event_cls.__name__
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _get_sender_name(contact: dict | None, fallback: str = "Unknown") -> str:
    """Get display name from contact dict."""
    if not contact:
        return fallback
    first = contact.get("first_name") or ""
    last = contact.get("surname") or ""
    name = f"{first} {last}".strip()
    return name or fallback


class EventHandler:
    _registry = {}

    @classmethod
    def register(cls, event_cls: list[Event] | Event):
        def wrapper(func):
            events_classes = (
                [event_cls] if not isinstance(event_cls, (list, tuple)) else event_cls
            )
            for e in events_classes:
                cls._registry[e] = func
            return func

        return wrapper

    @classmethod
    def handle_event(cls, event: Event, cm: "ConversationManager", *args, **kwargs):
        event_key = _event_type_to_log_key(event.__class__)
        if (
            hasattr(cm, "_session_logger")
            and not event.__class__.content_logged
            and event.__class__.loggable
        ):
            log_fn = (
                cm._session_logger.info
                if event.__class__.prominent
                else cm._session_logger.debug
            )
            log_fn(
                event_key,
                f"Event: {event.__class__.__name__}",
            )

        if event.__class__.loggable:
            asyncio.create_task(
                managers_utils.queue_operation(
                    managers_utils.publish_bus_events,
                    event,
                ),
            )

        f = cls._registry.get(event.__class__)
        if not f:
            return asyncio.sleep(0)
        return f(event, cm, *args, **kwargs)


@EventHandler.register(Ping)
async def _(event: Ping, cm: "ConversationManager", *args, **kwargs):
    log_str = "Ping received - keeping conversation manager alive"
    cm._session_logger.debug("ping", log_str)


@EventHandler.register((PhoneCallAnswered, WhatsAppCallAnswered))
async def _(
    event: PhoneCallAnswered | WhatsAppCallAnswered,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """
    Forward call answered status to the voice agent subprocess.

    This event arrives from the telephony system (via GCP PubSub) when the contact
    picks up for outbound calls. Forward it unconditionally - if this event arrives,
    we're in a call context by definition.
    """
    await cm.event_broker.publish(
        "app:call:status",
        json.dumps({"type": "call_answered"}),
    )


CallInitEvents = Union[
    PhoneCallReceived,
    PhoneCallSent,
    UnifyMeetReceived,
    WhatsAppCallReceived,
    WhatsAppCallSent,
]


@EventHandler.register(
    (
        PhoneCallReceived,
        PhoneCallSent,
        UnifyMeetReceived,
        WhatsAppCallReceived,
        WhatsAppCallSent,
    ),
)
async def _(event: CallInitEvents, cm: "ConversationManager", *args, **kwargs):
    """
    Handle incoming/outgoing call initiation - spawn voice agent subprocess.
    """
    # Don't start a new call if we're already in voice mode
    if cm.mode.is_voice:
        return

    boss = cm.contact_index.get_contact(contact_id=1)
    if isinstance(event, UnifyMeetReceived):
        contact = boss
    elif isinstance(event, (WhatsAppCallReceived, WhatsAppCallSent)):
        contact = cm.contact_index.get_contact(
            whatsapp_number=event.contact.get("whatsapp_number"),
        )
        if contact is None:
            contact = event.contact
    else:
        contact = cm.contact_index.get_contact(
            phone_number=event.contact["phone_number"],
        )
        if contact is None:
            contact = event.contact

    contact_id = (
        contact.get("contact_id") if contact else event.contact.get("contact_id")
    )
    sender_name = _get_sender_name(contact)

    # Inject stashed context from make_whatsapp_call if a pending permission
    # grant triggered this inbound WhatsApp call.
    if isinstance(event, WhatsAppCallReceived) and contact_id is not None:
        stashed_context = cm._pending_whatsapp_call_contexts.pop(contact_id, None)
        if stashed_context:
            cm.call_manager.initial_notification = stashed_context

    match event:
        case PhoneCallReceived() as e:
            cm.call_manager.conference_name = e.conference_name
            await cm.call_manager.start_call(contact, boss)
            message_content = "<Recvieving Call...>"
            notif_content = f"Call received from {sender_name}"
        case WhatsAppCallReceived() as e:
            cm.call_manager.conference_name = e.conference_name
            await cm.call_manager.start_call(
                contact,
                boss,
                channel="whatsapp_call",
            )
            message_content = "<Receiving WhatsApp Call...>"
            notif_content = f"WhatsApp call received from {sender_name}"
        case WhatsAppCallSent():
            await cm.call_manager.start_call(
                contact,
                boss,
                outbound=True,
                channel="whatsapp_call",
            )
            message_content = "<Sending WhatsApp Call...>"
            notif_content = f"WhatsApp call sent to {sender_name}"
        case PhoneCallSent():
            await cm.call_manager.start_call(contact, boss, outbound=True)
            message_content = "<Sending Call...>"
            notif_content = f"Call sent to {sender_name}"
        case UnifyMeetReceived() as e:
            await cm.call_manager.start_unify_meet(
                contact,
                boss,
                e.room_name,
            )
            message_content = "<Recieving Call...>"
            notif_content = f"Call received from {sender_name}"

    cm.notifications_bar.push_notif("Comms", notif_content, event.timestamp)
    role = "user" if "received" in event.__class__.__name__.lower() else "assistant"
    medium = (
        Medium.UNIFY_MEET
        if isinstance(event, UnifyMeetReceived)
        else (
            Medium.WHATSAPP_CALL
            if isinstance(event, (WhatsAppCallReceived, WhatsAppCallSent))
            else Medium.PHONE_CALL
        )
    )
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=message_content,
        role=role,
        timestamp=event.timestamp,
    )


@EventHandler.register(GoogleMeetReceived)
async def _(
    event: GoogleMeetReceived,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Handle request to join a Google Meet — spawn browser + audio bridge."""
    if (
        cm.mode.is_voice
        or cm.call_manager.has_active_call
        or cm.call_manager.has_active_google_meet
    ):
        return

    boss = cm.contact_index.get_contact(contact_id=1) or {}
    contact = boss

    contact_id = contact.get("contact_id") if contact else 1
    sender_name = _get_sender_name(contact)

    joined = await cm.call_manager.start_google_meet(
        meet_url=event.meet_url,
        contact=contact,
        boss=boss,
    )

    if joined:
        cm.notifications_bar.push_notif(
            "Comms",
            f"Joining Google Meet...",
            event.timestamp,
        )
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=Medium.GOOGLE_MEET,
            message_content="<Joining Google Meet...>",
            role="assistant",
            timestamp=event.timestamp,
        )
    else:
        cm.notifications_bar.push_notif(
            "Comms",
            "Failed to join Google Meet. You may retry by calling join_google_meet again.",
            event.timestamp,
        )
        await cm.request_llm_run(
            delay=0,
            cancel_running=True,
            triggering_contact_id=contact_id,
        )


@EventHandler.register(
    (PhoneCallStarted, UnifyMeetStarted, GoogleMeetStarted, WhatsAppCallStarted),
)
async def _(
    event: (
        PhoneCallStarted | UnifyMeetStarted | GoogleMeetStarted | WhatsAppCallStarted
    ),
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    if isinstance(event, WhatsAppCallStarted):
        cm.mode = Mode.CALL
        whatsapp_number = event.contact.get("whatsapp_number")
        contact = cm.contact_index.get_contact(whatsapp_number=whatsapp_number)
    elif isinstance(event, PhoneCallStarted):
        cm.mode = Mode.CALL
        phone_number = event.contact["phone_number"]
        contact = cm.contact_index.get_contact(phone_number=phone_number)
    elif isinstance(event, GoogleMeetStarted):
        cm.mode = Mode.MEET
        contact_id = event.contact.get("contact_id")
        contact = cm.contact_index.get_contact(contact_id=contact_id)
    else:
        cm.mode = Mode.MEET
        contact_id = event.contact.get("contact_id")
        contact = cm.contact_index.get_contact(contact_id=contact_id)

    if contact is None:
        contact = event.contact

    contact_id = contact.get("contact_id") if contact else 1
    sender_name = _get_sender_name(contact)

    cm.call_manager.call_contact = contact
    if isinstance(event, (PhoneCallStarted, WhatsAppCallStarted)):
        cm.call_manager.call_start_timestamp = event.timestamp
        label = "Phone Call" if isinstance(event, PhoneCallStarted) else "WhatsApp Call"
    elif isinstance(event, GoogleMeetStarted):
        cm.call_manager.google_meet_start_timestamp = event.timestamp
        label = "Google Meet"
    elif isinstance(event, UnifyMeetStarted):
        cm.call_manager.unify_meet_start_timestamp = event.timestamp
        label = "Unify Meet"
    else:
        raise ValueError(f"Unknown event type: {event.__class__.__name__}")

    cm.notifications_bar.push_notif(
        "Comms",
        f"{label} started with {sender_name}",
        timestamp=event.timestamp,
    )
    if isinstance(event, GoogleMeetStarted):
        medium = Medium.GOOGLE_MEET
    elif isinstance(event, PhoneCallStarted):
        medium = Medium.PHONE_CALL
    elif isinstance(event, WhatsAppCallStarted):
        medium = Medium.WHATSAPP_CALL
    else:
        medium = Medium.UNIFY_MEET
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content="<Call Started>",
        timestamp=event.timestamp,
    )
    conv_state = cm.contact_index.get_or_create_conversation(contact_id)
    conv_state.on_call = True

    # Sync meet interaction state that may have been set before the call started.
    # The fast brain starts with all flags as False and relies on guidance delivery,
    # so any state that was already active needs to be pushed now.
    if cm.assistant_screen_share_active and cm.call_manager._socket_server:
        from unity.conversation_manager.medium_scripts.common import (
            _resolve_agent_service_url,
        )

        guidance_text = _MEET_FAST_BRAIN_GUIDANCE[AssistantScreenShareStarted]
        notification_event = FastBrainNotification(
            contact=contact,
            content=guidance_text,
            source="meet_interaction",
            agent_service_url=_resolve_agent_service_url(),
        )
        await cm.call_manager._socket_server.queue_for_clients(
            "app:call:notification",
            notification_event.to_json(),
        )

    # No LLM run here — call guidance is pre-computed via make_call(context=...).
    # The slow brain will be woken later by:
    # - InboundPhoneUtterance (user says something)
    # - ActorResult (action completes)
    # - NotificationInjectedEvent (cross-channel notification)
    # - SMSReceived/EmailReceived while on call


@EventHandler.register(PhoneCallNotAnswered)
async def _(
    event: PhoneCallNotAnswered,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """
    Handle outbound call not answered (no-answer, busy, canceled, failed).

    This event arrives from the telephony system (via GCP PubSub) when the contact
    doesn't pick up for outbound calls. We need to:
    1. Tell the voice agent to stop (if running)
    2. Clean up the call process
    3. Notify the LLM brain so it can react appropriately
    """
    contact = cm.contact_index.get_contact(
        phone_number=event.contact.get("phone_number"),
    )
    if contact is None:
        contact = event.contact

    contact_id = contact.get("contact_id") if contact else 1
    sender_name = _get_sender_name(contact)
    reason = event.reason or "no-answer"

    # Forward stop status to voice agent subprocess
    await cm.event_broker.publish(
        "app:call:status",
        json.dumps({"type": "stop", "reason": f"call_not_answered:{reason}"}),
    )

    # Reset mode if we were in call mode
    if cm.mode.is_voice:
        cm.mode = Mode.TEXT
        cm.call_manager.call_contact = None

    # Clean up the call process
    await cm.call_manager.cleanup_call_proc()

    # Clear session state
    cm.call_manager.conference_name = None
    cm.call_manager.room_name = None
    cm.call_manager.call_start_timestamp = None
    cm.call_manager.unify_meet_start_timestamp = None
    cm.call_manager.google_meet_start_timestamp = None
    cm.call_manager.call_exchange_id = UNASSIGNED
    cm.call_manager.unify_meet_exchange_id = UNASSIGNED
    cm.call_manager.google_meet_exchange_id = UNASSIGNED

    # Build display content
    reason_display = {
        "no-answer": "did not answer",
        "busy": "was busy",
        "canceled": "call was canceled",
        "failed": "call failed",
    }.get(reason, f"not answered ({reason})")

    notif_content = f"Outbound call to {sender_name} {reason_display}"
    cm.notifications_bar.push_notif("Comms", notif_content, event.timestamp)

    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=Medium.PHONE_CALL,
        message_content=f"<Call Not Answered: {reason_display}>",
        role="assistant",
        timestamp=event.timestamp,
    )

    # Trigger LLM run so the brain can decide next steps
    await cm.request_llm_run(
        delay=0,
        cancel_running=True,
        triggering_contact_id=contact_id,
    )


@EventHandler.register(WhatsAppCallNotAnswered)
async def _(
    event: WhatsAppCallNotAnswered,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Handle outbound WhatsApp call not answered — same cleanup as phone."""
    contact = cm.contact_index.get_contact(
        whatsapp_number=event.contact.get("whatsapp_number"),
    )
    if contact is None:
        contact = event.contact

    contact_id = contact.get("contact_id") if contact else 1
    sender_name = _get_sender_name(contact)
    reason = event.reason or "no-answer"

    await cm.event_broker.publish(
        "app:call:status",
        json.dumps({"type": "stop", "reason": f"call_not_answered:{reason}"}),
    )

    if cm.mode.is_voice:
        cm.mode = Mode.TEXT
        cm.call_manager.call_contact = None

    await cm.call_manager.cleanup_call_proc()

    cm.call_manager.conference_name = None
    cm.call_manager.room_name = None
    cm.call_manager.call_start_timestamp = None
    cm.call_manager.unify_meet_start_timestamp = None
    cm.call_manager.call_exchange_id = UNASSIGNED
    cm.call_manager.unify_meet_exchange_id = UNASSIGNED

    reason_display = {
        "no-answer": "did not answer",
        "busy": "was busy",
        "canceled": "call was canceled",
        "failed": "call failed",
    }.get(reason, f"not answered ({reason})")

    notif_content = f"Outbound WhatsApp call to {sender_name} {reason_display}"
    cm.notifications_bar.push_notif("Comms", notif_content, event.timestamp)

    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=Medium.WHATSAPP_CALL,
        message_content=f"<WhatsApp Call Not Answered: {reason_display}>",
        role="assistant",
        timestamp=event.timestamp,
    )

    await cm.request_llm_run(
        delay=0,
        cancel_running=True,
        triggering_contact_id=contact_id,
    )


@EventHandler.register(WhatsAppCallPermissionResponse)
async def _(
    event: WhatsAppCallPermissionResponse,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Handle call permission grant/rejection from a WhatsApp contact."""
    contact = event.contact
    contact_id = contact.get("contact_id") if contact else 1
    sender_name = _get_sender_name(contact)

    has_pending_context = contact_id in cm._pending_whatsapp_call_contexts

    if event.accepted:
        notif_content = f"{sender_name} granted WhatsApp call permission"
    else:
        notif_content = f"{sender_name} rejected WhatsApp call permission"
        cm._pending_whatsapp_call_contexts.pop(contact_id, None)

    cm.notifications_bar.push_notif("Comms", notif_content, event.timestamp)

    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=Medium.WHATSAPP_CALL,
        message_content=f"<Call Permission: {'Accepted' if event.accepted else 'Rejected'}>",
        role="user",
        timestamp=event.timestamp,
    )

    if event.accepted and has_pending_context:
        # Permission granted after we sent a VOICE_CALL_REQUEST template.
        # Place the outbound call directly — no LLM round-trip needed.
        from unity.conversation_manager.domains import comms_utils
        from unity.conversation_manager.domains.call_manager import make_room_name

        context = cm._pending_whatsapp_call_contexts.pop(contact_id)
        cm.call_manager.initial_notification = context

        whatsapp_number = contact.get("whatsapp_number")
        assistant_id = str(SESSION_DETAILS.assistant.agent_id)
        agent_name = SESSION_DETAILS.assistant.name or ""
        room_name = make_room_name(assistant_id, "whatsapp_call")

        cm.call_manager._whatsapp_call_joining = True
        response = await comms_utils.start_whatsapp_call(
            to_number=whatsapp_number,
            agent_name=agent_name,
            room_name=room_name,
        )
        if response.get("success"):
            call_event = WhatsAppCallSent(contact=contact)
            await cm._event_broker.publish(
                "app:comms:whatsapp_call_sent",
                call_event.to_json(),
            )
            return

        cm.call_manager._whatsapp_call_joining = False
        cm.call_manager.initial_notification = None
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=Medium.WHATSAPP_CALL,
            message_content="<WhatsApp Call Failed After Permission Granted>",
            role="assistant",
            timestamp=event.timestamp,
        )

    await cm.request_llm_run(
        delay=0,
        cancel_running=True,
        triggering_contact_id=contact_id,
    )


@EventHandler.register(WhatsAppCallInviteSent)
async def _(
    event: WhatsAppCallInviteSent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Log the invite template send in the conversation thread."""
    contact = event.contact
    contact_id = contact.get("contact_id") if contact else 1
    sender_name = _get_sender_name(contact)

    cm.notifications_bar.push_notif(
        "Comms",
        f"WhatsApp call invite sent to {sender_name}",
        event.timestamp,
    )

    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=Medium.WHATSAPP_CALL,
        message_content="<WhatsApp Call Invite Sent>",
        role="assistant",
        timestamp=event.timestamp,
    )


@EventHandler.register(
    (
        InboundPhoneUtterance,
        InboundUnifyMeetUtterance,
        InboundWhatsAppCallUtterance,
        OutboundPhoneUtterance,
        OutboundUnifyMeetUtterance,
        OutboundWhatsAppCallUtterance,
        InboundGoogleMeetUtterance,
        OutboundGoogleMeetUtterance,
    ),
)
async def _(event: Event, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.debug(
        "event",
        f"Publishing transcript: {event.__class__.__name__}",
    )

    contact_id = event.contact["contact_id"]
    contact = cm.contact_index.get_contact(contact_id=contact_id)
    if contact is None:
        contact = event.contact
    sender_name = _get_sender_name(contact)
    role = "user" if event.__class__.__name__.startswith("Inbound") else "assistant"

    is_whatsapp_call = isinstance(
        event,
        (InboundWhatsAppCallUtterance, OutboundWhatsAppCallUtterance),
    )
    is_google_meet = isinstance(
        event,
        (InboundGoogleMeetUtterance, OutboundGoogleMeetUtterance),
    )
    is_unify_meet = isinstance(
        event,
        (InboundUnifyMeetUtterance, OutboundUnifyMeetUtterance),
    )
    if is_google_meet:
        medium = Medium.GOOGLE_MEET
    elif is_unify_meet:
        medium = Medium.UNIFY_MEET
    elif is_whatsapp_call:
        medium = Medium.WHATSAPP_CALL
    else:
        medium = Medium.PHONE_CALL

    # For diarized Meet utterances, prefer the speaker_label from the event
    if isinstance(event, InboundGoogleMeetUtterance) and event.speaker_label:
        sender_name = event.speaker_label

    message_id = cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=event.content,
        role=role,
    )

    await managers_utils.queue_operation(
        managers_utils.log_message,
        cm,
        event,
        local_message_id=message_id,
    )

    # Reset proactive speech on any utterance (user or assistant).
    await cm.schedule_proactive_speech()

    if role == "user":
        # Link any pending user/webcam screenshot (forwarded from the fast
        # brain via IPC) to this message by stamping it with the message_id.
        cm._claim_pending_user_screenshot(message_id)

        if cm.assistant_screen_share_active:
            await cm.capture_assistant_screenshot(
                event.content,
                message_id,
                cached=True,
            )

        await cm.interject_or_run(
            event.content,
            triggering_contact_id=contact_id,
        )


@EventHandler.register(FastBrainNotification)
async def _(
    event: FastBrainNotification,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    contact_id = event.contact.get("contact_id") if event.contact else None
    contact = (
        cm.contact_index.get_contact(contact_id=contact_id) if contact_id else None
    )
    if contact is None:
        contact = event.contact or {}
    sender_name = _get_sender_name(contact)

    if cm.call_manager.has_active_google_meet:
        medium = Medium.GOOGLE_MEET
    elif cm.mode == Mode.MEET:
        medium = Medium.UNIFY_MEET
    elif cm.call_manager._call_channel == "whatsapp_call":
        medium = Medium.WHATSAPP_CALL
    else:
        medium = Medium.PHONE_CALL
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=event.content,
        role="guidance",
    )

    if event.should_speak:
        await cm.schedule_proactive_speech()


@EventHandler.register(
    (PhoneCallEnded, UnifyMeetEnded, GoogleMeetEnded, WhatsAppCallEnded),
)
async def _(
    event: PhoneCallEnded | UnifyMeetEnded | GoogleMeetEnded | WhatsAppCallEnded,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    # Persist session identifiers in exchange metadata and stash the
    # exchange_id so the async RecordingReady handler can find it.
    if isinstance(event, (PhoneCallEnded, WhatsAppCallEnded)):
        exchange_id = cm.call_manager.call_exchange_id
        if exchange_id != UNASSIGNED and cm.call_manager.conference_name:
            cm.transcript_manager.update_exchange_metadata(
                exchange_id,
                {"conference_name": cm.call_manager.conference_name},
            )
            cm._recording_exchange_ids[cm.call_manager.conference_name] = exchange_id
    elif isinstance(event, GoogleMeetEnded):
        exchange_id = cm.call_manager.google_meet_exchange_id
        if exchange_id != UNASSIGNED and cm.call_manager.room_name:
            cm.transcript_manager.update_exchange_metadata(
                exchange_id,
                {"room_name": cm.call_manager.room_name},
            )
            cm._recording_exchange_ids[cm.call_manager.room_name] = exchange_id
    else:
        exchange_id = cm.call_manager.unify_meet_exchange_id
        if exchange_id != UNASSIGNED and cm.call_manager.room_name:
            cm.transcript_manager.update_exchange_metadata(
                exchange_id,
                {"room_name": cm.call_manager.room_name},
            )
            cm._recording_exchange_ids[cm.call_manager.room_name] = exchange_id

    cm.mode = Mode.TEXT
    cm.call_manager.call_contact = None

    if isinstance(event, (UnifyMeetEnded, GoogleMeetEnded)):
        contact_id = event.contact.get("contact_id")
        contact = cm.contact_index.get_contact(contact_id=contact_id)
    elif isinstance(event, WhatsAppCallEnded):
        contact = cm.contact_index.get_contact(
            whatsapp_number=event.contact.get("whatsapp_number"),
        )
    else:
        contact = cm.contact_index.get_contact(
            phone_number=event.contact["phone_number"],
        )

    if contact is None:
        contact = event.contact

    contact_id = (
        contact.get("contact_id") if contact else event.contact.get("contact_id")
    )
    conv_state = cm.contact_index.get_conversation_state(contact_id)
    if conv_state:
        conv_state.on_call = False

    if isinstance(event, GoogleMeetEnded):
        await cm.call_manager.cleanup_google_meet()
    else:
        await cm.call_manager.cleanup_call_proc()
    await cm.cancel_proactive_speech()
    await _cleanup_computer_sessions(cm)

    # Clear all session state after cleanup.
    cm.call_manager.conference_name = None
    cm.call_manager.room_name = None
    cm.call_manager.call_start_timestamp = None
    cm.call_manager.unify_meet_start_timestamp = None
    cm.call_manager.google_meet_start_timestamp = None
    cm.call_manager.call_exchange_id = UNASSIGNED
    cm.call_manager.unify_meet_exchange_id = UNASSIGNED
    cm.call_manager.google_meet_exchange_id = UNASSIGNED

    sender_name = _get_sender_name(contact)
    if isinstance(event, GoogleMeetEnded):
        label = "Google Meet"
    elif isinstance(event, UnifyMeetEnded):
        label = "Unify Meet"
    elif isinstance(event, WhatsAppCallEnded):
        label = "WhatsApp call"
    else:
        label = "Phone call"
    cm.notifications_bar.push_notif(
        "Comms",
        f"{label} with {sender_name} has ended.",
        event.timestamp,
    )

    await cm.request_llm_run(
        delay=0,
        cancel_running=True,
        triggering_contact_id=contact_id,
    )


async def _cleanup_computer_sessions(cm: "ConversationManager") -> None:
    """Stop in-flight actor sessions and close web browser sessions.

    Called on call end so resource cleanup is deterministic rather than
    relying on the slow brain to remember to call stop/close tools.
    """
    # Stop in-flight actor sessions
    for handle_id, action_data in list(cm.in_flight_actions.items()):
        handle = action_data.get("handle")
        if handle and not handle.done():
            try:
                await handle.stop("Call ended")
            except Exception:
                pass
        stopped = cm.in_flight_actions.pop(handle_id, None)
        if stopped:
            cm.completed_actions[handle_id] = stopped

    # Close active web browser sessions
    cp = cm.computer_primitives
    if cp is not None:
        try:
            active_sessions = cp.web.list_sessions(active_only=True)
            for session in active_sessions:
                try:
                    await session.stop()
                except Exception:
                    pass
        except Exception:
            pass


@EventHandler.register(RecordingReady)
async def _(
    event: RecordingReady,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    name = event.conference_name
    exchange_id = cm._recording_exchange_ids.pop(name, None)
    if exchange_id is not None:
        cm.transcript_manager.update_exchange_metadata(
            exchange_id,
            {"recording_url": event.recording_url},
        )
        cm._session_logger.debug(
            "recording",
            f"{DEFAULT_ICON} [RecordingReady] Stored recording_url on exchange "
            f"{exchange_id} for {name}",
        )
    else:
        cm._session_logger.debug(
            "recording",
            f"{DEFAULT_ICON} [RecordingReady] No exchange_id found for {name}",
        )


@EventHandler.register(
    (
        ActorResponse,
        ActorHandleResponse,
        ActorResult,
        ActorClarificationRequest,
    ),
)
async def _(event, cm: "ConversationManager", *args, **kwargs):
    if isinstance(event, ActorClarificationRequest):
        if event.handle_id in cm.in_flight_actions:
            from unity.common.prompt_helpers import now as prompt_now

            cm.in_flight_actions[event.handle_id]["handle_actions"].append(
                {
                    "action_name": "clarification_request",
                    "query": event.query,
                    "call_id": event.call_id,
                    "timestamp": prompt_now(),
                },
            )
            await cm.request_llm_run()
    elif isinstance(event, ActorHandleResponse):
        # Handle response from an action steering operation.
        # Check both in-flight and completed actions — post-completion asks
        # publish on the same channel after ActorResult has already moved
        # the action to completed_actions.
        handle_data = cm.in_flight_actions.get(
            event.handle_id,
        ) or cm.completed_actions.get(event.handle_id)
        if handle_data:
            handle_actions = handle_data.get("handle_actions", [])
            action_name = event.action_name or "ask"
            expected_action_name = f"{action_name}_{event.handle_id}"

            # Find the pending action and update it with the response.
            for action in reversed(handle_actions):
                if (
                    action.get("action_name") == expected_action_name
                    and action.get("status") == "pending"
                ):
                    action["status"] = "completed"
                    action["response"] = event.response
                    break

            # Wake the brain LLM to process the response
            await cm.request_llm_run()
    else:
        ...


def _push_email_to_all_contacts(
    cm: "ConversationManager",
    event,
    sender_contact: dict | None,
    sender_name: str,
    subject: str,
    body: str,
    email_id: str | None,
    attachments: list[str] | None,
    email_to: list[str],
    email_cc: list[str],
    email_bcc: list[str],
    role: str,
):
    """
    Push an email to ALL contacts involved (sender, to, cc, bcc).

    Emails are pushed to every known contact's thread to ensure no context is
    missing when viewing any contact-specific thread. Each message is tagged
    with `contact_role` to clarify the contact's relationship to the email.

    Args:
        cm: ConversationManager instance
        event: The email event (EmailSent or EmailReceived)
        sender_contact: The sender's contact dict (may be None for external senders)
        sender_name: Display name for the email sender
        subject: Email subject
        body: Email body
        email_id: Email ID for threading
        attachments: List of attachment filenames
        email_to: List of TO recipient email addresses
        email_cc: List of CC recipient email addresses
        email_bcc: List of BCC recipient email addresses
        role: "user" (received) or "assistant" (sent)
    """
    # Track which contact_ids we've already pushed to (avoid duplicates)
    pushed_contact_ids: set[int] = set()

    def _push_to_contact(contact_id: int, contact_role: str):
        """Helper to push email to a contact's thread."""
        if contact_id in pushed_contact_ids:
            return
        pushed_contact_ids.add(contact_id)
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=Medium.EMAIL,
            subject=subject,
            body=body,
            email_id=email_id,
            attachments=attachments,
            timestamp=event.timestamp,
            role=role,
            to=email_to,
            cc=email_cc,
            bcc=email_bcc,
            contact_role=contact_role,
        )

    def _resolve_contact_by_email(email_addr: str) -> dict | None:
        """Look up contact by email address."""
        return cm.contact_index.get_contact(email=email_addr)

    # 1. Push to sender's contact (if known)
    sender_contact_id = sender_contact.get("contact_id") if sender_contact else None
    if sender_contact_id is not None:
        _push_to_contact(sender_contact_id, "sender")

    # 2. Push to all TO recipients
    for email_addr in email_to or []:
        contact = _resolve_contact_by_email(email_addr)
        if contact and contact.get("contact_id"):
            _push_to_contact(contact["contact_id"], "to")

    # 3. Push to all CC recipients
    for email_addr in email_cc or []:
        contact = _resolve_contact_by_email(email_addr)
        if contact and contact.get("contact_id"):
            _push_to_contact(contact["contact_id"], "cc")

    # 4. Push to all BCC recipients
    for email_addr in email_bcc or []:
        contact = _resolve_contact_by_email(email_addr)
        if contact and contact.get("contact_id"):
            _push_to_contact(contact["contact_id"], "bcc")


@EventHandler.register(
    (
        SMSSent,
        SMSReceived,
        WhatsAppSent,
        WhatsAppReceived,
        EmailSent,
        EmailReceived,
        UnifyMessageSent,
        UnifyMessageReceived,
        ApiMessageSent,
        ApiMessageReceived,
    ),
)
async def _(event, cm: "ConversationManager", *args, **kwargs):
    await managers_utils.queue_operation(managers_utils.log_message, cm, event)

    message_content = None
    attachments = None
    tags = None
    notif_content = None

    # Get contact info from ContactManager, fallback to event.contact
    # Note: event.contact may be empty dict for emails to external addresses
    contact_id = event.contact.get("contact_id") if event.contact else None
    contact = cm.contact_index.get_contact(contact_id) if contact_id else None
    if contact is None:
        contact = event.contact or {}

    # contact_id may be None for external recipients not in contacts
    contact_id = contact.get("contact_id") if isinstance(contact, dict) else None
    sender_name = _get_sender_name(contact)

    match event:
        case SMSSent():
            medium = Medium.SMS_MESSAGE
            message_content = event.content
            notif_content = f"SMS sent to {sender_name}"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "sms_sent",
                f"SMS to {sender_name}: {event.content}",
            )
        case SMSReceived():
            medium = Medium.SMS_MESSAGE
            message_content = event.content
            notif_content = f"SMS Received from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "sms_received",
                f"SMS from {sender_name}: {event.content}",
            )
        case WhatsAppSent():
            medium = Medium.WHATSAPP_MESSAGE
            message_content = event.content
            attachments = event.attachments
            notif_content = f"WhatsApp sent to {sender_name}"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "whatsapp_sent",
                f"WhatsApp to {sender_name}: {event.content}",
            )
        case WhatsAppReceived():
            medium = Medium.WHATSAPP_MESSAGE
            message_content = event.content
            attachments = event.attachments
            notif_content = f"WhatsApp received from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "whatsapp_received",
                f"WhatsApp from {sender_name}: {event.content}",
            )
            pending_content = (
                cm._pending_whatsapp_resends.pop(contact_id, None)
                if contact_id is not None
                else None
            )
            if pending_content is not None:
                cm.notifications_bar.push_notif(
                    "comms",
                    (
                        f"Your earlier WhatsApp message to {sender_name} "
                        f"was not delivered verbatim. Original message: "
                        f'"{pending_content}". You can now resend or '
                        f"rework it via send_whatsapp."
                    ),
                    event.timestamp,
                )
        case EmailSent():
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            recipients = ", ".join((event.to or [])[:2])
            if len(event.to or []) > 2:
                recipients += "..."
            cm._session_logger.info(
                "email_sent",
                f"Email to {recipients}\n"
                f"Subject: {event.subject}\n\n"
                f"{event.body}",
            )
            # Email handling is special: push to ALL contacts involved
            email_to = event.to or []
            email_cc = event.cc or []
            email_bcc = event.bcc or []
            # For sent emails, the assistant is the sender
            _push_email_to_all_contacts(
                cm=cm,
                event=event,
                sender_contact=None,  # Assistant is sender, not a contact
                sender_name="You",
                subject=event.subject,
                body=event.body,
                email_id=event.email_id_replied_to,
                attachments=event.attachments,
                email_to=email_to,
                email_cc=email_cc,
                email_bcc=email_bcc,
                role="assistant",
            )
            notif_content = f"Email sent to {', '.join(email_to[:2])}{'...' if len(email_to) > 2 else ''}"
            cm.notifications_bar.push_notif("comms", notif_content, event.timestamp)
            if cm._outbound_suppress_gen != cm._llm_gen:
                await cm.request_llm_run(
                    triggering_contact_id=contact_id,
                )
            return  # Early return - email handling is complete

        case EmailReceived():
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "email_received",
                f"Email from {sender_name}\n"
                f"Subject: {event.subject}\n\n"
                f"{event.body}",
            )
            # Email handling is special: push to ALL contacts involved
            email_to = event.to or []
            email_cc = event.cc or []
            email_bcc = event.bcc or []
            _push_email_to_all_contacts(
                cm=cm,
                event=event,
                sender_contact=contact,  # The contact who sent the email
                sender_name=sender_name,
                subject=event.subject,
                body=event.body,
                email_id=event.email_id,
                attachments=event.attachments,
                email_to=email_to,
                email_cc=email_cc,
                email_bcc=email_bcc,
                role="user",
            )
            notif_content = f"Email Received from {sender_name}"
            cm.notifications_bar.push_notif("comms", notif_content, event.timestamp)
            await cm.cancel_proactive_speech()
            await cm.request_llm_run(
                triggering_contact_id=contact_id,
            )
            return  # Early return - email handling is complete

        case UnifyMessageSent():
            medium = Medium.UNIFY_MESSAGE
            message_content = event.content
            attachments = event.attachments
            notif_content = f"Unify message sent to {sender_name}"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "unify_message_sent",
                f"Message to {sender_name}: {event.content}",
            )
        case UnifyMessageReceived():
            medium = Medium.UNIFY_MESSAGE
            message_content = event.content
            attachments = event.attachments
            notif_content = f"Unify message from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "unify_message_received",
                f"Message from {sender_name}: {event.content}",
            )
        case ApiMessageSent():
            medium = Medium.API_MESSAGE
            message_content = event.content
            attachments = event.attachments
            tags = event.tags
            notif_content = f"API response sent to {sender_name}"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "api_message_sent",
                f"API response to {sender_name}: {event.content}",
            )
        case ApiMessageReceived():
            medium = Medium.API_MESSAGE
            message_content = event.content
            attachments = event.attachments
            tags = event.tags
            notif_content = f"API message from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "api_message_received",
                f"API message from {sender_name}: {event.content}",
            )
            if event.api_message_id:
                cm._pending_api_message_id = event.api_message_id
                cm._pending_api_message_tags = event.tags

    # Non-email messages: push to single contact only
    if contact_id is not None:
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=medium,
            message_content=message_content,
            attachments=attachments,
            timestamp=event.timestamp,
            role=role,
            tags=tags,
        )
    cm.notifications_bar.push_notif("comms", notif_content, event.timestamp)

    if role == "user":
        await cm.cancel_proactive_speech()

    if role == "user" or cm._outbound_suppress_gen != cm._llm_gen:
        await cm.request_llm_run(
            triggering_contact_id=contact_id,
        )


@EventHandler.register(Error)
async def _(event: Error, cm: "ConversationManager", *args, **kwargs):
    """Surface comms errors to the brain via the notification bar.

    When an outbound comms tool fails (send_sms, send_email, make_call, etc.),
    it publishes an Error event. Without a handler, the error is silently
    dropped and the brain never learns about the failure.

    The detailed error context is already pushed into the conversation thread
    by the tool itself (via ``_surface_comms_error``). This handler adds a
    lightweight notification and triggers a follow-up brain turn so the brain
    can see the failure and decide how to recover.
    """
    cm.notifications_bar.push_notif("Error", event.message, event.timestamp)
    await cm.request_llm_run(delay=0)


@EventHandler.register(BackupContactsEvent)
async def _(event: BackupContactsEvent, cm: "ConversationManager", *args, **kwargs):
    """
    Cache contacts from inbound messages for quick lookup.

    This handler is triggered when inbound messages arrive with contact data.
    Contacts are cached in ContactIndex and checked first in get_contact(),
    ensuring contacts from recent inbounds are always available even before
    or during ContactManager initialization.
    """
    if cm.contact_index._contact_manager:
        return
    cm._session_logger.debug(
        "backup_contacts",
        f"Caching {len(event.contacts)} contacts from inbound",
    )
    cm.contact_index.set_fallback_contacts(event.contacts)


@EventHandler.register(UnknownContactCreated)
async def _(event: UnknownContactCreated, cm: "ConversationManager", *args, **kwargs):
    """
    Handle new unknown contact creation from inbound messages.

    When an inbound SMS, email, or call arrives from an unknown sender (not in
    Contacts and not in BlackList), a minimal contact is automatically created
    with should_respond=False. This event notifies the ConversationManager so
    it can inform the boss and seek guidance on how to handle the contact.

    The assistant should use its judgement to decide the best course of action:
    - Inform the boss and ask for guidance
    - If clearly spam, potentially blacklist the contact
    - If legitimate, update contact details and enable responses
    """
    contact = event.contact
    contact_name = (
        contact.get("first_name")
        or contact.get("phone_number")
        or contact.get("email_address")
        or "Unknown"
    )

    cm._session_logger.info(
        "unknown_contact_created",
        f"New unknown contact created: {contact_name} via {event.medium}",
    )

    # Push notification so the assistant is aware
    notif_content = f"New unknown contact from {event.medium}: {contact_name}"
    if event.message_preview:
        notif_content += (
            f" - '{event.message_preview[:50]}...'"
            if len(event.message_preview) > 50
            else f" - '{event.message_preview}'"
        )
    cm.notifications_bar.push_notif("contacts", notif_content, event.timestamp)

    # Trigger LLM run so assistant can decide how to handle
    await cm.request_llm_run(
        triggering_contact_id=contact.get("contact_id"),
    )


async def _startup_sequence(cm: "ConversationManager", medium: str = ""):
    """Run job startup logging.

    VM readiness, desktop session warm-up, and file sync are now driven by
    the ``AssistantDesktopReady`` event handler rather than polling.
    """
    await asyncio.to_thread(
        assistant_jobs.log_job_startup,
        job_name=cm.job_name,
        user_id=cm.user_id,
        assistant_id=cm.assistant_id,
        medium=medium,
    )


@EventHandler.register((StartupEvent))
async def _(event: StartupEvent, cm: "ConversationManager", *args, **kwargs):
    try:
        cm._session_logger.info("startup", "Received startup event")

        # Set demo mode from startup event before initializing managers
        # Demo mode is derived from the presence of a demo_id
        if event.demo_id is not None:
            from unity.settings import SETTINGS

            SETTINGS.DEMO_MODE = True
            SETTINGS.DEMO_ID = event.demo_id
            cm._session_logger.debug(
                "startup",
                f"Demo mode enabled (demo_id={event.demo_id})",
            )

        payload = event.to_dict()["payload"]
        cm.set_details(payload)

        # Restart agent-service with the user's API key.
        # The agent-service is a sibling Node.js process started by entrypoint.sh
        # with the container's original UNIFY_KEY. set_details() + export_to_env()
        # update os.environ in this Python process, but the Node.js process still
        # has the old key. Kill and respawn so auth and LLM billing use the user's key.
        asyncio.create_task(
            asyncio.to_thread(
                _restart_agent_service_with_key,
                SESSION_DETAILS.unify_key,
            ),
        )

        cm.call_manager.set_config(cm.get_call_config())
        try:
            cm.call_manager.start_persistent_worker()
        except Exception as e:
            LOGGER.error(
                "LiveKit worker failed to start, voice calls unavailable: %s",
                e,
            )

        asyncio.create_task(_startup_sequence(cm, medium=event.medium))

        # Manager initialization runs in parallel
        asyncio.create_task(managers_utils.init_conv_manager(cm))
        asyncio.create_task(managers_utils.listen_to_operations(cm))
    except Exception as e:
        cm._session_logger.error("startup", f"Error in startup sequence: {e}")
        traceback.print_exc()
        publish_system_error(
            "The assistant failed to start up. Please try again shortly.",
            error_type="startup_failed",
        )


@EventHandler.register(AssistantUpdateEvent)
async def _(event: AssistantUpdateEvent, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.info("assistant_update", "Received assistant update event")
    payload = event.to_dict()["payload"]
    old_key = SESSION_DETAILS.unify_key
    cm.set_details(payload)

    if SESSION_DETAILS.unify_key != old_key:
        asyncio.create_task(
            asyncio.to_thread(
                _restart_agent_service_with_key,
                SESSION_DETAILS.unify_key,
            ),
        )

    cm.call_manager.set_config(cm.get_call_config())

    # Update contact manager with new assistant/user details
    await managers_utils.queue_operation(
        managers_utils.update_session_contacts,
        cm,
        event.assistant_first_name,
        event.assistant_surname,
        event.assistant_number,
        event.assistant_email,
        event.user_first_name,
        event.user_surname,
        event.user_number,
        event.user_email,
        event.assistant_whatsapp_number,
        event.user_whatsapp_number,
    )


@EventHandler.register(GetChatHistory)
async def _(event: GetChatHistory, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.debug(
        "state_update",
        f"Received chat history ({len(event.chat_history)} messages)",
    )
    cm.chat_history = event.chat_history + cm.chat_history


@EventHandler.register(ActorHandleStarted)
async def _(event: ActorHandleStarted, cm: "ConversationManager", *args, **kwargs):
    pass


@EventHandler.register(NotificationInjectedEvent)
async def _(
    event: NotificationInjectedEvent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm._session_logger.info(
        "notification_injected",
        f"Notification: {event.content[:50]}...",
    )

    cm.notifications_bar.push_notif(
        event.source,
        event.content,
        event.timestamp,
        pinned=event.pinned,
        id=event.interjection_id,
    )

    await cm.schedule_proactive_speech()
    await cm.request_llm_run(delay=0, cancel_running=True)


@EventHandler.register(NotificationUnpinnedEvent)
async def _(
    event: NotificationUnpinnedEvent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm._session_logger.info(
        "notification_unpinned",
        f"Unpinned interjection: {event.interjection_id}",
    )

    cm.notifications_bar.remove_notif(event.interjection_id)


@EventHandler.register(ActorResult)
async def _(event: ActorResult, cm: "ConversationManager", *args, **kwargs):
    action_data = cm.in_flight_actions.get(event.handle_id, {})

    # Log completion in handle_actions before moving to completed_actions.
    from unity.common.prompt_helpers import now as prompt_now

    if action_data and "handle_actions" in action_data:
        action_data["handle_actions"].append(
            {
                "action_name": "act_completed",
                "query": event.result,
                "timestamp": prompt_now(),
            },
        )

    # Move to completed_actions (preserves handle for post-completion ask queries)
    completed = cm.in_flight_actions.pop(event.handle_id, None)
    if completed:
        cm.completed_actions[event.handle_id] = completed
    await cm.request_llm_run()


@EventHandler.register(ActorSessionResponse)
async def _(event: ActorSessionResponse, cm: "ConversationManager", *args, **kwargs):
    """A persistent session completed a turn and is awaiting input.

    This is semantically distinct from ``ActorNotification`` (progress update):
    a response means the actor is *done with this turn* and will not proceed
    until the brain interjects with the next instruction.
    """
    action_data = cm.in_flight_actions.get(event.handle_id, {})

    from unity.common.prompt_helpers import now as prompt_now

    if action_data and "handle_actions" in action_data:
        action_data["handle_actions"].append(
            {
                "action_name": "response",
                "query": event.content,
                "status": "awaiting_input",
                "timestamp": prompt_now(),
            },
        )
    await cm.request_llm_run()


@EventHandler.register(ActorNotification)
async def _(event: ActorNotification, cm: "ConversationManager", *args, **kwargs):
    """A progress notification from an in-flight action.

    Unlike ``ActorResponse``, notifications arrive while the actor is still
    working.  Progress is recorded in the action's history so the slow brain
    sees accumulated progress when it next runs on a legitimate event.

    The fast brain receives actor progress via ``_render_boss_notifications``
    and the ``NotificationReplyEvaluator`` decides whether to speak.
    """
    if event.handle_id in cm.in_flight_actions:
        from unity.common.prompt_helpers import now as prompt_now

        cm.in_flight_actions[event.handle_id]["handle_actions"].append(
            {
                "action_name": "progress",
                "query": event.response,
                "timestamp": prompt_now(),
            },
        )


@EventHandler.register(ComputerActCompleted)
async def _(
    event: ComputerActCompleted,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """A visible computer session's act() call completed somewhere in the
    system. Push a notification and wake the slow brain so it can react."""
    snippet = event.summary[:120] if event.summary else event.instruction[:120]
    cm.notifications_bar.push_notif(
        "Computer",
        f"Computer action executed: {snippet}",
        event.timestamp,
    )
    await cm.request_llm_run()


@EventHandler.register(SyncContacts)
async def _(
    event: SyncContacts,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm._session_logger.info(
        "state_update",
        f"SyncContacts: {event.reason or 'no reason'}",
    )

    async def _sync_contacts():
        try:
            await asyncio.to_thread(cm.contact_manager._sync_required_contacts)
            cm._session_logger.info("state_update", "Contacts synced successfully")
        except Exception as e:
            cm._session_logger.error("state_update", f"Error syncing contacts: {e}")
        cm.notifications_bar.push_notif(
            "System",
            f"Contacts synced: {event.reason or 'manual sync'}",
            event.timestamp,
        )

    await managers_utils.queue_operation(_sync_contacts)


def _recent_conversation_snippet(cm: "ConversationManager", n: int = 4) -> str | None:
    """Extract the last *n* user/assistant messages from the global thread.

    Returns a compact multi-line string or None if no messages are available.
    """
    from unity.conversation_manager.domains.contact_index import CommsMessage

    lines: list[str] = []
    for entry in reversed(cm.contact_index.global_thread):
        if not isinstance(entry.message, CommsMessage):
            continue
        msg = entry.message
        content = getattr(msg, "content", None) or getattr(msg, "body", None) or ""
        content = content.strip()
        if not content or (content.startswith("<") and content.endswith(">")):
            continue
        role = getattr(msg, "role", "user")
        lines.append(f"{role}: {content}")
        if len(lines) >= n:
            break
    if not lines:
        return None
    lines.reverse()
    return "\n".join(lines)


async def _ensure_desktop_session(cm: "ConversationManager") -> None:
    """Create a desktop session in agent-service if one doesn't already exist.

    Sessions are lazy (created on first ``get_session`` call), so this must be
    called explicitly to guarantee the ``/screenshot`` endpoint has an active
    session to fall back to.  ``get_session`` is idempotent — calling it when a
    session already exists returns the cached instance.

    Retries with exponential backoff because the VM's Caddy reverse proxy may
    still be starting up or obtaining its TLS certificate from Let's Encrypt
    even after the Communication service reports the VM as "ready".
    """
    from unity.function_manager.primitives.runtime import ComputerPrimitives
    from unity.manager_registry import ManagerRegistry

    cp = ManagerRegistry.get_instance(ComputerPrimitives)
    if cp is None:
        return

    max_attempts = 12
    base_delay = 5.0
    max_delay = 30.0
    delay = base_delay

    for attempt in range(1, max_attempts + 1):
        try:
            session = await cp.backend.get_session("desktop")
            cm._session_logger.info(
                "desktop_session",
                f"Desktop session ready: {session._session_id}",
            )
            return
        except Exception as e:
            if attempt == max_attempts:
                cm._session_logger.warning(
                    "desktop_session",
                    f"Failed to create desktop session after {max_attempts} attempts: "
                    f"{type(e).__name__}: {e}",
                )
                return
            cm._session_logger.debug(
                "desktop_session",
                f"Attempt {attempt}/{max_attempts} failed ({type(e).__name__}), "
                f"retrying in {delay:.0f}s",
            )
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, max_delay)
            cp.backend.clear_session("desktop")


# --------------------------------------------------------------------------- #
# Desktop Lifecycle Events
# --------------------------------------------------------------------------- #


@EventHandler.register((AssistantDesktopReady,))
async def _(
    event: AssistantDesktopReady,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    from unity.conversation_manager.domains import comms_utils
    from unity.function_manager.primitives.runtime import _vm_ready
    from unity.session_details import SESSION_DETAILS

    desktop_url = event.desktop_url or SESSION_DETAILS.assistant.desktop_url or ""

    cm._session_logger.info(
        "desktop_ready",
        f"VM ready: {event.vm_type} at {desktop_url}",
    )

    if desktop_url:
        SESSION_DETAILS.assistant.desktop_url = desktop_url

    liveview_url = f"{desktop_url.rstrip('/')}/desktop/custom.html"
    await asyncio.to_thread(
        assistant_jobs.update_liveview_url,
        cm.assistant_id,
        cm.user_id,
        liveview_url,
    )

    _vm_ready.set()

    if desktop_url:
        from urllib.parse import urlparse
        from unity.function_manager.primitives.runtime import ComputerPrimitives
        from unity.manager_registry import ManagerRegistry

        cp = ManagerRegistry.get_instance(ComputerPrimitives)
        if cp is not None and cp._backend is not None:
            parsed = urlparse(desktop_url)
            cp._backend.update_container_url(
                f"{parsed.scheme}://{parsed.netloc}/api",
            )

    cm.vm_ready = True
    cm.notifications_bar.push_notif(
        "System",
        "Desktop VM is ready — computer actions are now available.",
        event.timestamp,
    )

    asyncio.ensure_future(_ensure_desktop_session(cm))
    await managers_utils._start_file_sync()

    await cm.event_broker.publish(
        FileSyncComplete.topic,
        FileSyncComplete().to_json(),
    )

    await comms_utils.publish_assistant_desktop_ready(
        desktop_url,
        liveview_url,
        event.vm_type,
    )

    await cm.request_llm_run(delay=0)


@EventHandler.register((FileSyncComplete,))
async def _(
    event: "FileSyncComplete",
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm.file_sync_complete = True
    cm.notifications_bar.push_notif(
        "System",
        "File sync complete — all files from previous sessions are now available on disk.",
        event.timestamp,
    )
    cm._session_logger.debug("file_sync", "File sync complete")
    await cm.request_llm_run(delay=0)


@EventHandler.register((InitializationComplete,))
async def _(
    event: "InitializationComplete",
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm.notifications_bar.push_notif(
        "System",
        (
            "Initialization complete — all actions are now available and "
            "full conversation history has been loaded. Review any earlier "
            "responses you gave during initialization and follow up if "
            "needed (correct, elaborate, or confirm)."
        ),
        event.timestamp,
    )
    cm._session_logger.debug("initialization", "Initialization complete")

    # Notify the fast brain (voice agent) directly via the call manager's
    # IPC socket — the same channel used for meet_interaction and other
    # direct fast brain notifications.  This bypasses the CM's own event
    # broker subscription so the notification reaches only the subprocess.
    if cm.call_manager and cm.call_manager._socket_server:
        fast_brain_notification = FastBrainNotification(
            contact={},
            content=(
                "Initialization complete — all actions are now available. "
                "Full conversation history has been loaded."
            ),
            should_speak=False,
            source="initialization",
        )
        await cm.call_manager._socket_server.queue_for_clients(
            "app:call:notification",
            fast_brain_notification.to_json(),
        )

    await cm.request_llm_run(delay=0)


# --------------------------------------------------------------------------- #
# Meet Interaction Events (screen share / remote control)
# --------------------------------------------------------------------------- #

# Mapping from event class to a human-readable notification for the slow brain.
_MEET_LOG_TYPES: dict[type, str] = {
    AssistantScreenShareStarted: "screen_share",
    AssistantScreenShareStopped: "screen_share_off",
    UserScreenShareStarted: "user_screen_share",
    UserScreenShareStopped: "user_screen_share_off",
    UserWebcamStarted: "webcam_on",
    UserWebcamStopped: "webcam_off",
    UserRemoteControlStarted: "remote_control",
    UserRemoteControlStopped: "remote_control_off",
}

_MEET_INTERACTION_NOTIFICATIONS: dict[type, str] = {
    AssistantScreenShareStarted: "The user enabled assistant screen sharing — they can now see your desktop.",
    AssistantScreenShareStopped: "The user disabled assistant screen sharing — they can no longer see your desktop.",
    UserScreenShareStarted: "The user started sharing their screen with you.",
    UserScreenShareStopped: "The user stopped sharing their screen.",
    UserWebcamStarted: "The user enabled their webcam — you can now see them.",
    UserWebcamStopped: "The user disabled their webcam.",
    UserRemoteControlStarted: "The user took remote control of your desktop — they now have mouse and keyboard control.",
    UserRemoteControlStopped: "The user released remote control of your desktop — you may resume computer actions.",
}

# Direct guidance sent to the fast brain when screen share modes change.
# These carry behavioral instructions so the fast brain knows how to respond
# to visual references immediately, without waiting for the slow brain.
_MEET_FAST_BRAIN_GUIDANCE: dict[type, str] = {
    AssistantScreenShareStarted: (
        "Screen sharing is now ON — the user can see your desktop. "
        "Screenshots are being captured and processed in the background. "
        "If the user references something on screen, acknowledge briefly "
        '("Got it", "I see", "Okay") and wait — visual context will '
        "be processed shortly. Do NOT describe or guess screen contents."
    ),
    AssistantScreenShareStopped: (
        "Screen sharing is now OFF — the user can no longer see your desktop."
    ),
    UserScreenShareStarted: (
        "The user is now sharing their screen with you. Visual context is "
        "being captured in the background. If they reference something on "
        "their screen, acknowledge naturally and wait for the processed "
        "details. Do NOT guess or fabricate what is on their screen."
    ),
    UserScreenShareStopped: ("The user stopped sharing their screen."),
    UserWebcamStarted: (
        "The user enabled their webcam. Visual context is being captured "
        "in the background. If they reference their appearance or something "
        "visible on camera, acknowledge naturally and wait for the processed "
        "details. Do NOT guess or fabricate what you see."
    ),
    UserWebcamStopped: ("The user disabled their webcam."),
    UserRemoteControlStarted: (
        "The user now has remote control of your desktop. Do not perform "
        "any computer actions — wait for them to release control."
    ),
    UserRemoteControlStopped: (
        "The user released remote control. You may resume computer actions."
    ),
}

# State attribute name on the CM for each toggle pair.
_MEET_STATE_FLAGS: dict[type, tuple[str, bool]] = {
    AssistantScreenShareStarted: ("assistant_screen_share_active", True),
    AssistantScreenShareStopped: ("assistant_screen_share_active", False),
    UserScreenShareStarted: ("user_screen_share_active", True),
    UserScreenShareStopped: ("user_screen_share_active", False),
    UserWebcamStarted: ("user_webcam_active", True),
    UserWebcamStopped: ("user_webcam_active", False),
    UserRemoteControlStarted: ("user_remote_control_active", True),
    UserRemoteControlStopped: ("user_remote_control_active", False),
}


@EventHandler.register(
    (
        AssistantScreenShareStarted,
        AssistantScreenShareStopped,
        UserScreenShareStarted,
        UserScreenShareStopped,
        UserWebcamStarted,
        UserWebcamStopped,
        UserRemoteControlStarted,
        UserRemoteControlStopped,
    ),
)
async def _(
    event: (
        AssistantScreenShareStarted
        | AssistantScreenShareStopped
        | UserScreenShareStarted
        | UserScreenShareStopped
        | UserWebcamStarted
        | UserWebcamStopped
        | UserRemoteControlStarted
        | UserRemoteControlStopped
    ),
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    event_name = event.__class__.__name__
    log_type = _MEET_LOG_TYPES.get(event.__class__, "meet_interaction")
    log_msg = f"Event: {event_name}"
    if event.reason == "LiveKit track auto-detected":
        cm._session_logger.debug(log_type, log_msg)
    else:
        cm._session_logger.info(log_type, log_msg)

    # Update state flag on the CM.
    attr, value = _MEET_STATE_FLAGS[event.__class__]
    setattr(cm, attr, value)

    if (
        isinstance(event, (UserWebcamStarted, UserWebcamStopped))
        and not cm.mode.is_voice
    ):
        return

    # Push a notification so the slow brain sees the state change.
    notification_text = _MEET_INTERACTION_NOTIFICATIONS[event.__class__]
    cm.notifications_bar.push_notif("Meet", notification_text, event.timestamp)

    # Send direct guidance to the fast brain so it can immediately adjust
    # its conversational behavior (e.g., acknowledge visual references
    # without hallucinating). This bypasses the slow brain for instant delivery.
    fast_brain_text = _MEET_FAST_BRAIN_GUIDANCE.get(event.__class__)
    if fast_brain_text and cm.mode.is_voice:
        contact = cm.get_active_contact()
        if contact:
            cm._session_logger.debug(
                "call_notification",
                f"Publishing meet interaction reason={event_name}",
            )
            from unity.conversation_manager.medium_scripts.common import (
                _resolve_agent_service_url,
            )

            notification_event = FastBrainNotification(
                contact=contact,
                content=fast_brain_text,
                source="meet_interaction",
                agent_service_url=_resolve_agent_service_url(),
            )
            await cm.event_broker.publish(
                "app:call:notification",
                notification_event.to_json(),
            )

    await cm.schedule_proactive_speech()

    if isinstance(event, AssistantScreenShareStarted):
        asyncio.ensure_future(_ensure_desktop_session(cm))

    # Broadcast remote-control state change to all active CodeActActor loops
    # via the ComputerPrimitives singleton interject queue registry.
    if isinstance(event, (UserRemoteControlStarted, UserRemoteControlStopped)):
        try:
            from unity.function_manager.primitives.runtime import ComputerPrimitives
            from unity.manager_registry import ManagerRegistry

            cp = ManagerRegistry.get_instance(ComputerPrimitives)
            if cp is not None:
                cp.set_user_remote_control(
                    isinstance(event, UserRemoteControlStarted),
                    conversation_context=_recent_conversation_snippet(cm),
                )
        except Exception:
            pass

    if isinstance(event, UserRemoteControlStopped):
        if cm.assistant_screen_share_active:
            await cm.capture_assistant_screenshot("", cached=False)


@EventHandler.register(LogMessageResponse)
async def _(event: LogMessageResponse, cm: "ConversationManager", *args, **kwargs):
    if (
        event.medium in (Medium.PHONE_CALL, Medium.WHATSAPP_CALL)
        and cm.call_manager.call_exchange_id == UNASSIGNED
    ):
        cm.call_manager.call_exchange_id = event.exchange_id
    if (
        event.medium == Medium.UNIFY_MEET
        and cm.call_manager.unify_meet_exchange_id == UNASSIGNED
    ):
        cm.call_manager.unify_meet_exchange_id = event.exchange_id
    if (
        event.medium == Medium.GOOGLE_MEET
        and cm.call_manager.google_meet_exchange_id == UNASSIGNED
    ):
        cm.call_manager.google_meet_exchange_id = event.exchange_id


@EventHandler.register(PreHireMessage)
async def _(event: PreHireMessage, cm: "ConversationManager", *args, **kwargs):
    await managers_utils.queue_operation(managers_utils.log_message, cm, event)


@EventHandler.register(DirectMessageEvent)
async def _(event: DirectMessageEvent, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.info(
        "direct_message",
        f"Direct message: {event.content[:50]}...",
    )

    if cm.mode.is_voice:
        cm._session_logger.info(
            "call_notification",
            "Publishing direct-message notification",
        )
        await cm.event_broker.publish(
            "app:call:notification",
            json.dumps({"content": event.content}),
        )

    contact = cm.get_active_contact()
    contact_id = contact.get("contact_id") if contact else 1
    sender_name = _get_sender_name(contact)

    if cm.call_manager.has_active_google_meet:
        medium = Medium.GOOGLE_MEET
    elif cm.mode == Mode.MEET:
        medium = Medium.UNIFY_MEET
    elif cm.call_manager._call_channel == "whatsapp_call":
        medium = Medium.WHATSAPP_CALL
    else:
        medium = Medium.PHONE_CALL
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=event.content,
        role="assistant",
        timestamp=event.timestamp,
    )
