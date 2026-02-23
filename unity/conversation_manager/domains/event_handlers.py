import asyncio
import traceback
from typing import TYPE_CHECKING, Union

from unity.contact_manager.types.contact import UNASSIGNED
from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON
from unity.conversation_manager import assistant_jobs
from unity.conversation_manager.events import *
from unity.conversation_manager.domains import managers_utils
from unity.conversation_manager.tracing import content_trace_id
from unity.conversation_manager.types import Medium, Mode

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager


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
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                event_key,
                (
                    f"Event: {event.__class__.__name__} "
                    f"(event_id={event_trace.get('event_id', '-')})"
                ),
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


@EventHandler.register(PhoneCallAnswered)
async def _(event: PhoneCallAnswered, cm: "ConversationManager", *args, **kwargs):
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


CallInitEvents = Union[PhoneCallReceived, PhoneCallSent, UnifyMeetReceived]


@EventHandler.register((PhoneCallReceived, PhoneCallSent, UnifyMeetReceived))
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

    match event:
        case PhoneCallReceived() as e:
            cm.call_manager.conference_name = e.conference_name
            await cm.call_manager.start_call(contact, boss)
            message_content = "<Recvieving Call...>"
            notif_content = f"Call received from {sender_name}"
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
        Medium.UNIFY_MEET if isinstance(event, UnifyMeetReceived) else Medium.PHONE_CALL
    )
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=message_content,
        role=role,
        timestamp=event.timestamp,
    )


@EventHandler.register((PhoneCallStarted, UnifyMeetStarted))
async def _(
    event: PhoneCallStarted | UnifyMeetStarted,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    if isinstance(event, PhoneCallStarted):
        cm.mode = Mode.CALL
        phone_number = event.contact["phone_number"]
        contact = cm.contact_index.get_contact(phone_number=phone_number)
    else:
        cm.mode = Mode.MEET
        contact_id = event.contact.get("contact_id")
        contact = cm.contact_index.get_contact(contact_id=contact_id)

    if contact is None:
        contact = event.contact

    contact_id = contact.get("contact_id") if contact else 1
    sender_name = _get_sender_name(contact)

    cm.call_manager.call_contact = contact
    if isinstance(event, PhoneCallStarted):
        cm.call_manager.call_start_timestamp = event.timestamp
    else:
        cm.call_manager.unify_meet_start_timestamp = event.timestamp
    cm.notifications_bar.push_notif(
        "Comms",
        f"Phone Call started with {sender_name}",
        timestamp=event.timestamp,
    )
    medium = (
        Medium.PHONE_CALL if isinstance(event, PhoneCallStarted) else Medium.UNIFY_MEET
    )
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
        guidance_text = _MEET_FAST_BRAIN_GUIDANCE[AssistantScreenShareStarted]
        guidance_event = CallGuidance(
            contact=contact,
            content=guidance_text,
            source="meet_interaction",
        )
        await cm.call_manager._socket_server.queue_for_clients(
            "app:call:call_guidance",
            guidance_event.to_json(),
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
    cm.call_manager.call_exchange_id = UNASSIGNED
    cm.call_manager.unify_meet_exchange_id = UNASSIGNED

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
    await cm.request_llm_run(delay=0, cancel_running=True)


@EventHandler.register(
    (
        InboundPhoneUtterance,
        InboundUnifyMeetUtterance,
        OutboundPhoneUtterance,
        OutboundUnifyMeetUtterance,
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

    is_unify_meet = isinstance(
        event,
        (InboundUnifyMeetUtterance, OutboundUnifyMeetUtterance),
    )
    medium = Medium.UNIFY_MEET if is_unify_meet else Medium.PHONE_CALL
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

    # Outbound utterances signal that the fast brain's generation+TTS cycle
    # completed — clear the suppression flag so proactive speech can resume.
    if role == "assistant":
        cm._fast_brain_active = False

    # Reset proactive speech on any utterance (user or assistant).
    await cm.schedule_proactive_speech()

    if role == "user":
        # Link any pending user/webcam screenshot (forwarded from the fast
        # brain via IPC) to this message by stamping it with the message_id.
        cm._claim_pending_user_screenshot(message_id)

        if cm.assistant_screen_share_active:
            await cm.capture_assistant_screenshot(event.content, message_id)

        await cm.interject_or_run(event.content)


@EventHandler.register(CallGuidance)
async def _(
    event: CallGuidance,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    guidance_id = content_trace_id("guid", event.content or "")
    cm._session_logger.info(
        "call_guidance",
        f"Received guidance guidance_id={guidance_id}: {event.content[:50]}...",
    )
    contact_id = event.contact["contact_id"]
    contact = cm.contact_index.get_contact(contact_id=contact_id)
    if contact is None:
        contact = event.contact
    sender_name = _get_sender_name(contact)

    medium = Medium.UNIFY_MEET if cm.mode == Mode.MEET else Medium.PHONE_CALL
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=event.content,
        role="guidance",
    )

    if event.should_speak:
        await cm.schedule_proactive_speech()


@EventHandler.register((PhoneCallEnded, UnifyMeetEnded))
async def _(
    event: PhoneCallEnded | UnifyMeetEnded,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    # Persist session identifiers in exchange metadata and stash the
    # exchange_id so the async RecordingReady handler can find it.
    if isinstance(event, PhoneCallEnded):
        exchange_id = cm.call_manager.call_exchange_id
        if exchange_id != UNASSIGNED and cm.call_manager.conference_name:
            cm.transcript_manager.update_exchange_metadata(
                exchange_id,
                {"conference_name": cm.call_manager.conference_name},
            )
            cm._recording_exchange_ids[cm.call_manager.conference_name] = exchange_id
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
    if isinstance(event, UnifyMeetEnded):
        contact_id = event.contact.get("contact_id")
        contact = cm.contact_index.get_contact(contact_id=contact_id)
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

    await cm.call_manager.cleanup_call_proc()
    await cm.cancel_proactive_speech()

    # Clear all session state after cleanup.
    cm.call_manager.conference_name = None
    cm.call_manager.room_name = None
    cm.call_manager.call_start_timestamp = None
    cm.call_manager.unify_meet_start_timestamp = None
    cm.call_manager.call_exchange_id = UNASSIGNED
    cm.call_manager.unify_meet_exchange_id = UNASSIGNED

    await cm.request_llm_run(delay=0, cancel_running=True)


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
        LOGGER.debug(
            f"{DEFAULT_ICON} [RecordingReady] Stored recording_url on exchange "
            f"{exchange_id} for {name}",
        )
    else:
        LOGGER.debug(f"{DEFAULT_ICON} [RecordingReady] No exchange_id found for {name}")


@EventHandler.register(
    (
        ActorResponse,
        ActorHandleResponse,
        ActorResult,
        ActorClarificationRequest,
    ),
)
async def _(event, cm: "ConversationManager", *args, **kwargs):
    cm._has_non_forwarded_event = True
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
        if event.handle_id in cm.in_flight_actions:
            handle_data = cm.in_flight_actions[event.handle_id]
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
        EmailSent,
        EmailReceived,
        UnifyMessageSent,
        UnifyMessageReceived,
    ),
)
async def _(event, cm: "ConversationManager", *args, **kwargs):
    await managers_utils.queue_operation(managers_utils.log_message, cm, event)

    message_content = None
    attachments = None
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

    # Flag non-participant comms during voice calls. The fast brain only
    # renders comms from the active call contact; everything else is dropped.
    if getattr(cm.mode, "is_voice", False):
        call_contact_id = (cm.call_manager.call_contact or {}).get("contact_id")
        if contact_id != call_contact_id:
            cm._has_non_forwarded_event = True

    match event:
        case SMSSent():
            medium = Medium.SMS_MESSAGE
            message_content = event.content
            notif_content = f"SMS sent to {sender_name}"
            role = "assistant"
        case SMSReceived():
            medium = Medium.SMS_MESSAGE
            message_content = event.content
            notif_content = f"SMS Received from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "sms_received",
                f"({event_trace.get('event_id', '-')}) "
                f"SMS from {sender_name}: {event.content}",
            )
        case EmailSent():
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
            await cm.request_llm_run(delay=2)
            return  # Early return - email handling is complete

        case EmailReceived():
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "email_received",
                f"({event_trace.get('event_id', '-')}) "
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
            await cm.request_llm_run(delay=2)
            return  # Early return - email handling is complete

        case UnifyMessageSent():
            medium = Medium.UNIFY_MESSAGE
            message_content = event.content
            attachments = event.attachments
            notif_content = f"Unify message sent to {sender_name}"
            role = "assistant"
        case UnifyMessageReceived():
            medium = Medium.UNIFY_MESSAGE
            message_content = event.content
            attachments = event.attachments
            notif_content = f"Unify message from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "unify_message_received",
                f"({event_trace.get('event_id', '-')}) "
                f"Message from {sender_name}: {event.content}",
            )

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
        )
    cm.notifications_bar.push_notif("comms", notif_content, event.timestamp)

    if role == "user":
        await cm.cancel_proactive_speech()

    await cm.request_llm_run(delay=2)


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
    cm._has_non_forwarded_event = True
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

    cm._has_non_forwarded_event = True
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
    await cm.request_llm_run(delay=2)


async def _startup_sequence(cm: "ConversationManager"):
    """Run job startup logging, signal VM readiness, then file sync.

    File sync depends on VM connectivity details that log_job_startup resolves,
    so we must wait for job startup to complete before starting file sync.
    log_job_startup includes _resolve_vm_liveview polling for managed VMs,
    so once it returns the VM is confirmed reachable (or retries exhausted).
    """
    await asyncio.to_thread(
        assistant_jobs.log_job_startup,
        job_name=cm.job_name,
        user_id=cm.user_id,
        assistant_id=cm.assistant_id,
    )
    # Unblock any pending MagnitudeBackend lazy initialization.
    from unity.function_manager.primitives.runtime import _vm_ready

    _vm_ready.set()
    await managers_utils._start_file_sync()


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
            cm._session_logger.info(
                "startup",
                f"Demo mode enabled (demo_id={event.demo_id})",
            )

        payload = event.to_dict()["payload"]
        cm.set_details(payload)
        cm.call_manager.set_config(cm.get_call_config())

        # Job logging + file sync run in sequence (file sync needs VM details from job startup)
        asyncio.create_task(_startup_sequence(cm))

        # Manager initialization runs in parallel
        asyncio.create_task(managers_utils.init_conv_manager(cm))
        asyncio.create_task(managers_utils.listen_to_operations(cm))
    except Exception as e:
        cm._session_logger.error("startup", f"Error in startup sequence: {e}")
        traceback.print_exc()


@EventHandler.register(AssistantUpdateEvent)
async def _(event: AssistantUpdateEvent, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.info("assistant_update", "Received assistant update event")
    payload = event.to_dict()["payload"]
    cm.set_details(payload)
    cm.call_manager.set_config(cm.get_call_config())

    # Update contact manager with new assistant/user details
    await managers_utils.queue_operation(
        managers_utils.update_session_contacts,
        cm,
        event.assistant_name,
        event.assistant_number,
        event.assistant_email,
        event.user_name,
        event.user_number,
        event.user_email,
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
    cm._has_non_forwarded_event = True
    await cm.request_llm_run()


@EventHandler.register(NotificationInjectedEvent)
async def _(
    event: NotificationInjectedEvent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm._has_non_forwarded_event = True
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
    cm._has_non_forwarded_event = True
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
    cm._has_non_forwarded_event = True
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
    working.  Progress is recorded in the action's history.

    The slow brain is woken to decide whether to relay progress via
    ``guide_voice_agent``.  On boss-on-call, the fast brain receives the
    raw event directly via channel forwarding (so guidance is not needed).
    """
    cm._has_non_forwarded_event = True
    if event.handle_id in cm.in_flight_actions:
        from unity.common.prompt_helpers import now as prompt_now

        cm.in_flight_actions[event.handle_id]["handle_actions"].append(
            {
                "action_name": "progress",
                "query": event.response,
                "timestamp": prompt_now(),
            },
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


# --------------------------------------------------------------------------- #
# Meet Interaction Events (screen share / remote control)
# --------------------------------------------------------------------------- #

# Mapping from event class to a human-readable notification for the slow brain.
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
    cm._session_logger.info("meet_interaction", f"{event_name}: {event.reason}")

    # Update state flag on the CM.
    attr, value = _MEET_STATE_FLAGS[event.__class__]
    setattr(cm, attr, value)

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
            guidance_id = content_trace_id("guid", fast_brain_text)
            cm._session_logger.info(
                "call_guidance",
                (
                    f"Publishing meet interaction guidance_id={guidance_id} "
                    f"reason={event_name}"
                ),
            )
            guidance_event = CallGuidance(
                contact=contact,
                content=fast_brain_text,
                source="meet_interaction",
            )
            await cm.event_broker.publish(
                "app:call:call_guidance",
                guidance_event.to_json(),
            )

    # Eagerly initialize the MagnitudeBackend when screen sharing starts so
    # the agent-service has an active session for fast brain screenshot capture.
    # Runs in a thread because MagnitudeBackend.__init__ is synchronous
    # (~1-4s for Chromium cold start).
    if isinstance(event, AssistantScreenShareStarted):

        def _ensure_backend():
            try:
                from unity.function_manager.primitives.runtime import ComputerPrimitives
                from unity.manager_registry import ManagerRegistry

                cp = ManagerRegistry.get_instance(ComputerPrimitives)
                if cp is not None:
                    _ = cp.backend
            except Exception:
                pass

        asyncio.get_event_loop().run_in_executor(None, _ensure_backend)

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

    await cm.request_llm_run()


@EventHandler.register(LogMessageResponse)
async def _(event: LogMessageResponse, cm: "ConversationManager", *args, **kwargs):
    if (
        event.medium == Medium.PHONE_CALL
        and cm.call_manager.call_exchange_id == UNASSIGNED
    ):
        cm.call_manager.call_exchange_id = event.exchange_id
    if (
        event.medium == Medium.UNIFY_MEET
        and cm.call_manager.unify_meet_exchange_id == UNASSIGNED
    ):
        cm.call_manager.unify_meet_exchange_id = event.exchange_id


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
        guidance_id = content_trace_id("guid", event.content or "")
        cm._session_logger.info(
            "call_guidance",
            f"Publishing direct-message guidance guidance_id={guidance_id}",
        )
        await cm.event_broker.publish(
            "app:call:call_guidance",
            json.dumps({"content": event.content}),
        )

    contact = cm.get_active_contact()
    contact_id = contact.get("contact_id") if contact else 1
    sender_name = _get_sender_name(contact)

    medium = Medium.UNIFY_MEET if cm.mode == Mode.MEET else Medium.PHONE_CALL
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=event.content,
        role="assistant",
        timestamp=event.timestamp,
    )
