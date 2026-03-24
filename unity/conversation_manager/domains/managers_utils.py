from datetime import timedelta
import asyncio
import os
from time import perf_counter
from typing import TYPE_CHECKING

import unity

from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON, ICONS
from unity.settings import SETTINGS
from unity.session_details import SESSION_DETAILS
from unity.conversation_manager.metrics import (
    manager_init_total,
    per_manager_init,
)
from unity.common.async_tool_loop import SteerableToolHandle
from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.domains.comms_utils import publish_system_error
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.events import *
from unity.common.prompt_helpers import now as prompt_now
from unity.events.event_bus import EVENT_BUS
from unity.manager_registry import ManagerRegistry
from unity.conversation_manager.types import Medium

if TYPE_CHECKING:
    from unity.actor.base import BaseActor
    from unity.conversation_manager.conversation_manager import ConversationManager

event_broker = get_event_broker()

# Cache for pre-hire exchange ID - used to group all pre-hire messages into one exchange
_pre_hire_exchange_id: int | None = None

# Thought: This entire file could actually be turned into a mixin class


# EVENT BUS
async def get_last_store_chat_history() -> StoreChatHistory:
    bus_events = await EVENT_BUS.search(
        filter='type == "Comms" and payload_cls == "StoreChatHistory"',
        limit=1,
    )
    if len(bus_events):
        return Event.from_bus_event(bus_events[0])
    return None


def _get_sender_name(contact: dict | None) -> str:
    """Extract display name from a contact dict."""
    if not contact:
        return "Unknown"
    first_name = contact.get("first_name", "")
    surname = contact.get("surname", "")
    name = f"{first_name} {surname}".strip()
    return (
        name
        or contact.get("phone_number", "")
        or contact.get("email_address", "")
        or "Unknown"
    )


# Event types that produce push_message calls during hydration.
_MESSAGE_PRODUCING_EVENTS = {
    "SMSReceived",
    "SMSSent",
    "EmailReceived",
    "EmailSent",
    "UnifyMessageReceived",
    "UnifyMessageSent",
    "InboundPhoneUtterance",
    "OutboundPhoneUtterance",
    "InboundUnifyMeetUtterance",
    "OutboundUnifyMeetUtterance",
    "FastBrainNotification",
    "PhoneCallReceived",
    "PhoneCallSent",
    "UnifyMeetReceived",
    "PhoneCallStarted",
    "UnifyMeetStarted",
    "PhoneCallNotAnswered",
    "ApiMessageReceived",
    "ApiMessageSent",
}


async def hydrate_global_thread(cm: "ConversationManager") -> None:
    """Populate the shared global deque from persisted EventBus Comms events.

    Called after initialization to restore conversation state from the previous
    session.  Hydrated (historical) messages are prepended to the global thread
    so that any messages that arrived during initialization keep their correct
    chronological position at the end.
    """
    from unity.conversation_manager.domains.contact_index import ContactIndex

    deque_size = (
        cm.contact_index.global_thread.maxlen or ContactIndex.DEFAULT_GLOBAL_THREAD_SIZE
    )

    bus_events = await EVENT_BUS.search(
        filter='type == "Comms"',
        limit=deque_size,
    )

    if not bus_events:
        LOGGER.info(
            f"{ICONS['managers_worker']} [Hydration] No Comms events found, skipping hydration",
        )
        return

    # Bus events come in descending order (most recent first), reverse for chronological
    bus_events.reverse()

    # Build entries into a buffer via build_message (no append to the live
    # deque), so we can prepend them all at once and preserve chronological
    # ordering relative to any messages that arrived during initialization.
    hydrated_entries: list = []

    restored = 0
    for bus_event in bus_events:
        payload_cls = bus_event.payload_cls
        # Strip module prefix if present (e.g., "unity.conversation_manager.events.SMSReceived")
        if "." in payload_cls:
            payload_cls = payload_cls.rsplit(".", 1)[-1]

        if payload_cls not in _MESSAGE_PRODUCING_EVENTS:
            continue

        try:
            cm_event = Event.from_bus_event(bus_event)
        except Exception:
            continue

        contact = getattr(cm_event, "contact", None) or {}
        contact_id = contact.get("contact_id")
        if contact_id is None:
            continue
        sender_name = _get_sender_name(contact)
        ts = cm_event.timestamp

        entry = None
        match payload_cls:
            # --- SMS ---
            case "SMSReceived":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.SMS_MESSAGE,
                    message_content=cm_event.content,
                    role="user",
                    timestamp=ts,
                )
            case "SMSSent":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.SMS_MESSAGE,
                    message_content=cm_event.content,
                    role="assistant",
                    timestamp=ts,
                )

            # --- Unify Messages ---
            case "UnifyMessageReceived":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.UNIFY_MESSAGE,
                    message_content=cm_event.content,
                    role="user",
                    timestamp=ts,
                    attachments=getattr(cm_event, "attachments", None),
                )
            case "UnifyMessageSent":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.UNIFY_MESSAGE,
                    message_content=cm_event.content,
                    role="assistant",
                    timestamp=ts,
                    attachments=getattr(cm_event, "attachments", None),
                )

            # --- API Messages ---
            case "ApiMessageReceived":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.API_MESSAGE,
                    message_content=cm_event.content,
                    role="user",
                    timestamp=ts,
                    attachments=getattr(cm_event, "attachments", None),
                    tags=getattr(cm_event, "tags", None),
                )
            case "ApiMessageSent":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.API_MESSAGE,
                    message_content=cm_event.content,
                    role="assistant",
                    timestamp=ts,
                    attachments=getattr(cm_event, "attachments", None),
                    tags=getattr(cm_event, "tags", None),
                )

            # --- Email ---
            case "EmailReceived":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.EMAIL,
                    subject=cm_event.subject,
                    body=cm_event.body,
                    email_id=getattr(cm_event, "email_id", None),
                    attachments=getattr(cm_event, "attachments", None),
                    role="user",
                    timestamp=ts,
                    to=getattr(cm_event, "to", None),
                    cc=getattr(cm_event, "cc", None),
                    bcc=getattr(cm_event, "bcc", None),
                    contact_role="sender",
                )
            case "EmailSent":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.EMAIL,
                    subject=cm_event.subject,
                    body=cm_event.body,
                    attachments=getattr(cm_event, "attachments", None),
                    role="assistant",
                    timestamp=ts,
                    to=getattr(cm_event, "to", None),
                    cc=getattr(cm_event, "cc", None),
                    bcc=getattr(cm_event, "bcc", None),
                )

            # --- Phone/Meet utterances ---
            case "InboundPhoneUtterance":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.PHONE_CALL,
                    message_content=cm_event.content,
                    role="user",
                    timestamp=ts,
                )
            case "OutboundPhoneUtterance":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.PHONE_CALL,
                    message_content=cm_event.content,
                    role="assistant",
                    timestamp=ts,
                )
            case "InboundUnifyMeetUtterance":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.UNIFY_MEET,
                    message_content=cm_event.content,
                    role="user",
                    timestamp=ts,
                )
            case "OutboundUnifyMeetUtterance":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.UNIFY_MEET,
                    message_content=cm_event.content,
                    role="assistant",
                    timestamp=ts,
                )

            # --- Fast brain notification ---
            case "FastBrainNotification":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.PHONE_CALL,
                    message_content=cm_event.content,
                    role="guidance",
                    timestamp=ts,
                )

            # --- Call lifecycle ---
            case "PhoneCallReceived":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.PHONE_CALL,
                    message_content="<Receiving Call...>",
                    role="user",
                    timestamp=ts,
                )
            case "PhoneCallSent":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.PHONE_CALL,
                    message_content="<Sending Call...>",
                    role="assistant",
                    timestamp=ts,
                )
            case "UnifyMeetReceived":
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.UNIFY_MEET,
                    message_content="<Receiving Call...>",
                    role="user",
                    timestamp=ts,
                )
            case "PhoneCallStarted" | "UnifyMeetStarted":
                medium = (
                    Medium.UNIFY_MEET
                    if payload_cls == "UnifyMeetStarted"
                    else Medium.PHONE_CALL
                )
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=medium,
                    message_content="<Call Started>",
                    role="user",
                    timestamp=ts,
                )
            case "PhoneCallNotAnswered":
                reason = getattr(cm_event, "reason", "no-answer") or "no-answer"
                reason_display = {
                    "no-answer": "did not answer",
                    "busy": "was busy",
                    "canceled": "call was canceled",
                    "failed": "call failed",
                }.get(reason, f"not answered ({reason})")
                entry = cm.contact_index.build_message(
                    contact_id=contact_id,
                    sender_name=sender_name,
                    thread_name=Medium.PHONE_CALL,
                    message_content=f"<Call Not Answered: {reason_display}>",
                    role="assistant",
                    timestamp=ts,
                )

        if entry is not None:
            hydrated_entries.append(entry)
            restored += 1

    # Prepend hydrated entries so historical messages appear before any
    # messages that arrived during initialization.
    cm.contact_index.prepend_entries(hydrated_entries)

    LOGGER.info(
        f"{ICONS['managers_worker']} [Hydration] Restored {restored} messages from {len(bus_events)} Comms events",
    )


async def publish_bus_events(event):
    try:
        event_name = event.__class__.__name__
        bus_event = event.to_bus_event()
        bus_event.payload.pop("api_key", None)
        bus_event.payload.pop("email_id", None)
        LOGGER.debug(f"{DEFAULT_ICON} Publishing bus event {event_name}")
        await EVENT_BUS.publish(bus_event)
    except Exception as e:
        LOGGER.error(
            f"{ICONS['managers_worker']} [ManagersWorker] Error publishing bus event: {e}",
        )


# ACTOR
async def actor_watch_result(
    handle_id: int,
    handle: SteerableToolHandle,
    *,
    action_type: str = "",
) -> None:
    """Await final result and publish completion (or failure), then cleanup."""
    # await result
    try:
        result = await handle.result()
    except Exception as e:
        result = f"Error getting actor result: {e}"
        LOGGER.error(f"{ICONS['managers_worker']} [ManagersWorker] {result}")
    await event_broker.publish(
        "app:actor:result",
        ActorResult(
            handle_id=handle_id,
            success=False if "Error" in result else True,
            result=result,
            action_type=action_type,
        ).to_json(),
    )


async def actor_watch_notifications(
    handle_id: int,
    handle: SteerableToolHandle,
) -> None:
    """Forward notifications and responses from the handle until it completes.

    The handle's notification queue carries two kinds of messages:

    - **``type="notification"``** — progress updates emitted by ``notify()``
      while the actor is still working.
    - **``type="response"``** — turn-complete signals emitted when a
      persistent session enters its wait state. These mean the actor has
      finished the current turn and is awaiting the next ``interject``.

    Each type is published as a distinct CM event so the brain can tell
    them apart.
    """
    while not handle.done():
        try:
            notif = await asyncio.wait_for(handle.next_notification(), timeout=30)
        except asyncio.TimeoutError:
            continue

        # Determine whether this is a turn-complete response or a progress
        # notification. The loop emits responses with {"type": "response", ...}.
        is_response = isinstance(notif, dict) and notif.get("type") == "response"

        if is_response:
            content = str(notif.get("content", ""))
            await event_broker.publish(
                "app:actor:session_response",
                ActorSessionResponse(
                    handle_id=handle_id,
                    content=content,
                ).to_json(),
            )
        else:
            # Extract a human-friendly message.
            #
            # Contract:
            # - Notifications may be plain strings (already display-ready), OR
            # - Structured dict payloads (recommended: include both "type" and "message").
            #
            # Fallback chain: "message" → "result_summary" → "type" → JSON dump.
            # "result_summary" is checked before "type" because step_complete
            # payloads carry their useful content in that field, not "message".
            msg: str
            if isinstance(notif, dict):
                if notif.get("message") is not None:
                    msg = str(notif.get("message"))
                elif notif.get("result_summary") is not None:
                    msg = str(notif.get("result_summary"))
                elif notif.get("type") is not None:
                    msg = str(notif.get("type"))
                else:
                    try:
                        import json as _json

                        msg = _json.dumps(notif, ensure_ascii=False, default=str)
                    except Exception:
                        msg = str(notif)
            else:
                msg = str(notif)

            completed = (
                bool(notif.get("completed", False))
                if isinstance(notif, dict)
                else False
            )
            await event_broker.publish(
                "app:actor:notification",
                ActorNotification(
                    handle_id=handle_id,
                    response=msg,
                    completed=completed,
                ).to_json(),
            )


async def actor_watch_clarifications(
    handle_id: int,
    handle: SteerableToolHandle,
) -> None:
    """Forward clarifications to CM until handle completes."""
    while not handle.done():
        # await clarification request
        try:
            clar = await asyncio.wait_for(handle.next_clarification(), timeout=30)
        except asyncio.TimeoutError:
            continue

        # get question and call id
        q = clar.get("question") if isinstance(clar, dict) else str(clar)
        call_id = clar.get("call_id") if isinstance(clar, dict) else None

        # publish clarification request
        await event_broker.publish(
            "app:actor:clarification_request",
            ActorClarificationRequest(
                handle_id=handle_id,
                query=q,
                call_id=call_id,
            ).to_json(),
        )


async def log_message(
    cm: "ConversationManager",
    event: Event,
    *,
    local_message_id: int | None = None,
) -> None:
    """Log a message via TranscriptManager."""
    event_name = event.__class__.__name__
    LOGGER.debug(f"{DEFAULT_ICON} publishing transcript {event_name}")
    event_name = event_name.lower()
    if "apimessage" in event_name:
        medium = Medium.API_MESSAGE
    elif "unify" in event_name or "prehire" in event_name:
        medium = Medium.UNIFY_MEET if "meet" in event_name else Medium.UNIFY_MESSAGE
    elif "phone" in event_name:
        medium = Medium.PHONE_CALL
    elif "sms" in event_name:
        medium = Medium.SMS_MESSAGE
    else:
        medium = Medium.EMAIL
    role = "Assistant" if "sent" in event_name or "assistant" in event_name else "User"
    if "prehire" in event_name:
        role = event.role.capitalize()
    if isinstance(event, (EmailSent, EmailReceived)):
        content = event.subject + "\n\n" + event.body
    else:
        content = event.content

    contact_id = None
    if isinstance(event, (PreHireMessage,)):
        # PreHireMessage is always boss context
        contact_id = 1
    elif isinstance(
        event,
        (
            UnifyMessageSent,
            UnifyMessageReceived,
            InboundUnifyMeetUtterance,
            OutboundUnifyMeetUtterance,
            ApiMessageSent,
            ApiMessageReceived,
        ),
    ):
        # Use contact from event - contact_id must be valid, no silent fallback
        evt_contact_id = event.contact.get("contact_id")
        if cm.contact_index.get_contact(contact_id=evt_contact_id):
            contact_id = evt_contact_id
        else:
            LOGGER.warning(
                f"{DEFAULT_ICON} contact_id {evt_contact_id} not in contact_index, "
                f"using contact from event",
            )
            contact_id = evt_contact_id
    elif cm.contact_index.get_contact(contact_id=event.contact["contact_id"]):
        contact_id = event.contact["contact_id"]
    if role == "Assistant":
        sender_id, receiver_ids = 0, [contact_id]
    else:
        sender_id, receiver_ids = contact_id, [0]

    # For emails, resolve to/cc/bcc addresses to contact IDs so that
    # receiver_ids reflects all known recipients.
    if isinstance(event, (EmailSent, EmailReceived)):
        resolved_ids: set[int] = set()
        for addr in (event.to or []) + (event.cc or []) + (event.bcc or []):
            resolved = cm.contact_index.get_contact(email=addr)
            if resolved and resolved.get("contact_id") is not None:
                resolved_ids.add(resolved["contact_id"])
        if resolved_ids:
            if role == "Assistant":
                receiver_ids = sorted(resolved_ids)
            else:
                # Keep assistant (0) plus all resolved recipients
                resolved_ids.add(0)
                receiver_ids = sorted(resolved_ids)

    exchange_id = getattr(event, "exchange_id", UNASSIGNED)

    # For pre-hire messages, reuse the cached exchange_id if available
    # This ensures all messages from a pre-hire chat batch go into the same exchange
    if isinstance(event, PreHireMessage):
        if _pre_hire_exchange_id is not None:
            exchange_id = _pre_hire_exchange_id
        # else: stays UNASSIGNED, will create new exchange
    elif medium == Medium.PHONE_CALL:
        exchange_id = cm.call_manager.call_exchange_id
    elif medium == Medium.UNIFY_MEET:
        exchange_id = cm.call_manager.unify_meet_exchange_id

    call_utterance_timestamp = ""
    # Compute utterance timestamp based on active call type.
    call_start = (
        cm.call_manager.call_start_timestamp
        if medium == Medium.PHONE_CALL
        else (
            cm.call_manager.unify_meet_start_timestamp
            if medium == Medium.UNIFY_MEET
            else None
        )
    )
    if call_start:
        delta = prompt_now(as_string=False) - call_start
        if role == "Assistant":
            delta += timedelta(seconds=2)
        minutes, seconds = divmod(int(delta.total_seconds()), 60)
        call_utterance_timestamp = f"{minutes:02d}.{seconds:02d}"

    # publish transcript on a separate thread
    def _publish_transcript() -> int:
        global _pre_hire_exchange_id
        try:
            nonlocal exchange_id
            LOGGER.debug(
                f"{ICONS['managers_worker']} [ManagersWorker] Logging message: {event.to_dict()}",
            )

            # Extract attachments from event if present (now always list[dict])
            attachments = getattr(event, "attachments", [])

            # Build medium-specific metadata for the transcript record.
            metadata = None
            if isinstance(event, EmailReceived):
                metadata = {
                    "email_id": event.email_id,
                    "to": event.to,
                    "cc": event.cc,
                    "bcc": event.bcc,
                }
            elif isinstance(event, EmailSent):
                metadata = {
                    "email_id_replied_to": event.email_id_replied_to,
                    "to": event.to,
                    "cc": event.cc,
                    "bcc": event.bcc,
                }

            if call_utterance_timestamp:
                metadata = metadata or {}
                metadata["call_utterance_timestamp"] = call_utterance_timestamp

            tm_message_id = None
            if exchange_id == UNASSIGNED:
                msg_data = {
                    "medium": medium,
                    "sender_id": sender_id,
                    "receiver_ids": receiver_ids,
                    "timestamp": event.timestamp,
                    "content": content,
                }
                if attachments:
                    msg_data["attachments"] = attachments
                if metadata:
                    msg_data["metadata"] = metadata
                exchange_id, tm_message_id = (
                    cm.transcript_manager.log_first_message_in_new_exchange(
                        msg_data,
                    )
                )
                # Cache the exchange_id for subsequent pre-hire messages in the batch
                if isinstance(event, PreHireMessage):
                    _pre_hire_exchange_id = exchange_id
                    LOGGER.debug(
                        f"{ICONS['managers_worker']} [ManagersWorker] Cached pre-hire exchange_id: {exchange_id}",
                    )
            else:
                msg_data = {
                    "medium": medium,
                    "sender_id": sender_id,
                    "receiver_ids": receiver_ids,
                    "timestamp": event.timestamp,
                    "content": content,
                    "exchange_id": exchange_id,
                }
                if attachments:
                    msg_data["attachments"] = attachments
                if metadata:
                    msg_data["metadata"] = metadata
                logged_msgs = cm.transcript_manager.log_messages(
                    msg_data,
                    synchronous=True,
                )
                if logged_msgs:
                    tm_message_id = logged_msgs[0].message_id

            if local_message_id is not None and tm_message_id is not None:
                cm._local_to_global_message_ids[local_message_id] = tm_message_id

            LOGGER.debug(
                f"{ICONS['managers_worker']} [ManagersWorker] Logged message: {medium}"
                f" from {sender_id} to {receiver_ids}",
            )
            return exchange_id
        except Exception as e:
            LOGGER.error(
                f"{ICONS['managers_worker']} [ManagersWorker] Error logging message: {e}",
            )

    exchange_id = await asyncio.to_thread(_publish_transcript)

    # publish reply as event envelope
    await event_broker.publish(
        "app:logging:message_logged",
        LogMessageResponse(
            medium=medium,
            exchange_id=exchange_id,
        ).to_json(),
    )
    LOGGER.debug(
        f"{ICONS['managers_worker']} [ManagersWorker] Published exchange_id {exchange_id}",
    )


# Contact updates


async def update_session_contacts(
    cm: "ConversationManager",
    assistant_first_name: str,
    assistant_surname: str,
    assistant_number: str,
    assistant_email: str,
    user_first_name: str,
    user_surname: str,
    user_number: str,
    user_email: str,
) -> None:
    """
    Update the assistant (contact_id=0) and boss (contact_id=1) contacts
    in the ContactManager when session details change.

    Called when an AssistantUpdateEvent is received.

    Note: In demo mode, we skip updating the boss contact (contact_id=1) because
    the user_* fields contain the demoer's details, not the prospect's. The
    prospect's details are either:
    - Set during initialization from demo metadata (prospect_* fields), or
    - Updated dynamically via set_boss_details during the demo
    """
    if cm.contact_manager is None:
        LOGGER.info(
            f"{ICONS['managers_worker']} [ManagersWorker] Cannot update contacts: contact_manager is None",
        )
        return

    async def _update_contact(
        contact_id: int,
        first_name: str,
        surname: str,
        phone_number: str,
        email_address: str,
    ):
        try:
            await asyncio.to_thread(
                cm.contact_manager.update_contact,
                contact_id=contact_id,
                phone_number=phone_number,
                email_address=email_address,
                first_name=first_name,
                surname=surname,
            )
            LOGGER.info(
                f"{ICONS['managers_worker']} [ManagersWorker] Updated contact {contact_id}: {first_name} {surname}",
            )
        except Exception as e:
            LOGGER.error(
                f"{ICONS['managers_worker']} [ManagersWorker] Failed to update contact {contact_id}: {e}",
            )

    await _update_contact(
        0,
        assistant_first_name,
        assistant_surname,
        assistant_number,
        assistant_email,
    )

    # In demo mode:
    # - Skip updating boss contact (contact_id=1) - prospect details come from Orchestra meta
    # - Update demoer contact (contact_id=2) with user_* fields (initially created in _init_managers)
    if SETTINGS.DEMO_MODE:
        LOGGER.info(
            f"{ICONS['managers_worker']} [ManagersWorker] Demo mode: skipping boss contact (contact_id=1), "
            "updating demoer contact (contact_id=2)",
        )
        await _update_contact(2, user_first_name, user_surname, user_number, user_email)
        return

    await _update_contact(1, user_first_name, user_surname, user_number, user_email)


# Queueing operations that need managers

_operations_queue = asyncio.Queue()


async def queue_operation(async_func: callable, *args, **kwargs) -> None:
    """
    Queue an async operation to be executed when managers are initialized.
    The operation will be processed by listen_to_operations().
    """
    await _operations_queue.put((async_func, args, kwargs))


async def wait_for_initialization(
    cm: "ConversationManager",
) -> None:
    """
    Wait for initialization to complete.

    Polls cm.initialized with no timeout. Initialization failures are
    surfaced by init_conv_manager itself (logged errors, pod inactivity
    shutdown). A timeout here would silently kill the operations queue
    processor on slow cold starts, causing queued work to be orphaned.
    """
    while not cm.initialized:
        await asyncio.sleep(0.1)


async def listen_to_operations(cm: "ConversationManager") -> None:
    """
    Worker loop that processes queued operations once initialized.
    Should be started as a background task alongside init_conv_manager.
    """
    # Wait for initialization to complete
    await wait_for_initialization(cm)

    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] Operations listener started, processing queue...",
    )

    # Process operations as they come in
    while True:
        # Wait for next operation (with timeout to allow checking for shutdown)
        try:
            async_func, args, kwargs = await asyncio.wait_for(
                _operations_queue.get(),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            continue

        # Execute the operation
        func_name = getattr(async_func, "__name__", str(async_func))
        try:
            await async_func(*args, **kwargs)
        except Exception as e:
            LOGGER.error(
                f"{ICONS['managers_worker']} [ManagersWorker] Error executing {func_name}: {e}",
            )
        finally:
            _operations_queue.task_done()


# Initialization

_init_lock = asyncio.Lock()


def _init_managers(
    cm: "ConversationManager",
    loop: asyncio.AbstractEventLoop,
    actor: "BaseActor | None" = None,
) -> None:
    """
    Initialize all managers in a separate thread.
    The main event loop is passed for managers that need to schedule async tasks.

    Args:
        cm: The ConversationManager instance to initialize.
        loop: The main event loop for scheduling async tasks.
        actor: Optional pre-instantiated Actor. If provided, used directly instead
            of creating one via ManagerRegistry. Useful for testing with specific
            Actor implementations.
    """
    start_time = perf_counter()

    # 0. Initialize unity (idempotent — SESSION_DETAILS.assistant.agent_id is
    #    already set by the startup handler, so unity.init() reads it for context).
    LOGGER.debug(f"{ICONS['managers_worker']} [ManagersWorker] Initializing unity...")
    local_start_time = perf_counter()
    unity.init()
    _unity_init_dur = perf_counter() - local_start_time
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] Unity initialized in {_unity_init_dur:.2f} seconds",
    )
    per_manager_init.record(_unity_init_dur, {"manager": "unity"})

    # Get API key from SESSION_DETAILS (set by ConversationManager on startup)
    api_key = SESSION_DETAILS.unify_key or None

    # 1. Configure EventBus
    LOGGER.debug(f"{ICONS['managers_worker']} [ManagersWorker] Configuring EventBus...")
    local_start_time = perf_counter()
    if api_key:
        EVENT_BUS._get_logger().session.headers["Authorization"] = f"Bearer {api_key}"
    EVENT_BUS.set_window("Comms", 100)
    _eventbus_dur = perf_counter() - local_start_time
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] EventBus configured in {_eventbus_dur:.2f} seconds",
    )
    per_manager_init.record(_eventbus_dur, {"manager": "event_bus"})

    # 1b. Kick off hydration concurrently — it only needs unity.init() and
    # EventBus config (both done). Runs on the main event loop while the
    # remaining managers initialize in this thread.
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] Starting concurrent hydration...",
    )
    cm._hydration_future = asyncio.run_coroutine_threadsafe(
        hydrate_global_thread(cm),
        loop,
    )

    # 2. Initialize ContactManager (respects SETTINGS.contact.IMPL)
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] Initializing ContactManager...",
    )
    local_start_time = perf_counter()
    cm.contact_manager = ManagerRegistry.get_contact_manager(
        description="production deployment",
    )
    # Wire up ContactManager to ContactIndex for always-fresh contact data
    cm.contact_index.set_contact_manager(cm.contact_manager)
    # In demo mode, ensure the boss contact (contact_id==1) is always visible
    # in active_conversations so the slow brain can use inline details on
    # communication tools (e.g., make_call(contact_id=1, phone_number=...))
    # and set_boss_details to update their record.
    if SETTINGS.DEMO_MODE:
        # Ensure boss (contact_id=1) is visible in active conversations for the brain
        cm.contact_index.get_or_create_conversation(1)
        # Start the boss contact sparse in demo mode; details can be provided
        # later via set_boss_details or demo prospect metadata.
        cm.contact_manager.update_contact(
            contact_id=1,
            first_name="",
            surname="",
            email_address="",
            phone_number="",
            should_respond=True,
        )
        # If we have a demo_id, fetch prospect details from Orchestra and apply
        # them to the boss contact (contact_id=1)
        if SETTINGS.DEMO_ID is not None:
            try:
                from unity.demo_meta import (
                    fetch_demo_meta,
                    apply_prospect_to_boss_contact,
                )

                # Run async fetch_demo_meta on the event loop from this sync context
                future = asyncio.run_coroutine_threadsafe(
                    fetch_demo_meta(SETTINGS.DEMO_ID),
                    loop,
                )
                prospect = future.result(timeout=10.0)  # 10 second timeout
                if prospect and prospect.has_any_details():
                    apply_prospect_to_boss_contact(cm.contact_manager, prospect)
                    LOGGER.info(
                        f"{ICONS['managers_worker']} [ManagersWorker] Applied prospect details from demo_id={SETTINGS.DEMO_ID}",
                    )
            except Exception as e:
                LOGGER.error(
                    f"{ICONS['managers_worker']} [ManagersWorker] Failed to fetch/apply demo prospect details: {e}",
                )

        # Create demoer contact (contact_id=2) with the user's details
        # In demo mode, SESSION_DETAILS.user contains the demoer's info
        # Note: We don't add to active_conversations as the demoer isn't someone
        # the assistant would typically interact with (call/email)
        try:
            demoer_first = SESSION_DETAILS.user.first_name
            demoer_last = SESSION_DETAILS.user.surname
            # Use _create_contact since contact_id=2 doesn't exist yet
            cm.contact_manager._create_contact(
                first_name=demoer_first,
                surname=demoer_last,
                phone_number=SESSION_DETAILS.user.number or "",
                email_address=SESSION_DETAILS.user.email or "",
                should_respond=True,
                is_system=True,
            )
            LOGGER.info(
                f"{ICONS['managers_worker']} [ManagersWorker] Created demoer contact (id=2): {demoer_first} {demoer_last}",
            )
        except Exception as e:
            LOGGER.error(
                f"{ICONS['managers_worker']} [ManagersWorker] Failed to create demoer contact: {e}",
            )
    _contact_dur = perf_counter() - local_start_time
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] ContactManager ({type(cm.contact_manager).__name__}) initialized in "
        f"{_contact_dur:.2f} seconds",
    )
    per_manager_init.record(_contact_dur, {"manager": "contact_manager"})

    # 3. Initialize TranscriptManager (respects SETTINGS.transcript.IMPL)
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] Initializing TranscriptManager...",
    )
    local_start_time = perf_counter()
    cm.transcript_manager = ManagerRegistry.get_transcript_manager(
        description="production deployment",
        contact_manager=cm.contact_manager,
    )
    _transcript_dur = perf_counter() - local_start_time
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] TranscriptManager ({type(cm.transcript_manager).__name__}) initialized in "
        f"{_transcript_dur:.2f} seconds",
    )
    per_manager_init.record(_transcript_dur, {"manager": "transcript_manager"})

    # 4. Configure TranscriptManager logger (only for real implementation)
    # Check hasattr instead of SETTINGS to be defensive against implementation mismatches
    if api_key and hasattr(cm.transcript_manager, "_get_logger"):
        cm.transcript_manager._get_logger().session.headers[
            "Authorization"
        ] = f"Bearer {api_key}"

    # 5. Initialize MemoryManager (optional - respects SETTINGS.memory.ENABLED and IMPL)
    if SETTINGS.memory.ENABLED:
        try:
            from unity.memory_manager.memory_manager import MemoryManager

            LOGGER.info(
                f"{ICONS['managers_worker']} [ManagersWorker] Initializing MemoryManager...",
            )
            local_start_time = perf_counter()
            mem_cfg = MemoryManager.MemoryConfig(
                contacts=SETTINGS.memory.CONTACTS,
                bios=SETTINGS.memory.BIOS,
                rolling_summaries=SETTINGS.memory.ROLLING_SUMMARIES,
                response_policies=SETTINGS.memory.RESPONSE_POLICIES,
                knowledge=SETTINGS.memory.KNOWLEDGE,
                tasks=SETTINGS.memory.TASKS,
            )
            cm.memory_manager = ManagerRegistry.get_memory_manager(
                transcript_manager=cm.transcript_manager,
                contact_manager=cm.contact_manager,
                config=mem_cfg,
                loop=loop,
            )
            _memory_dur = perf_counter() - local_start_time
            LOGGER.info(
                f"{ICONS['managers_worker']} [ManagersWorker] MemoryManager initialized in {_memory_dur:.2f} seconds",
            )
            per_manager_init.record(_memory_dur, {"manager": "memory_manager"})
        except Exception as e:
            LOGGER.warning(
                f"{ICONS['managers_worker']} [ManagersWorker] MemoryManager init failed (degraded): {e}",
            )
    else:
        LOGGER.info(
            f"{ICONS['managers_worker']} [ManagersWorker] MemoryManager disabled (SETTINGS.memory.ENABLED=False)",
        )

    # 6. Initialize ConversationManagerHandle (respects SETTINGS.conversation.IMPL)
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] Initializing ConversationManagerHandle...",
    )
    local_start_time = perf_counter()
    # ConversationManagerHandle has different constructor args for real vs simulated
    if SETTINGS.conversation.IMPL == "simulated":
        cm._conversation_manager_handle = (
            ManagerRegistry.get_conversation_manager_handle(
                description="production deployment",
                assistant_id=SESSION_DETAILS.assistant.agent_id,
                contact_id="1",
            )
        )
    else:
        cm._conversation_manager_handle = (
            ManagerRegistry.get_conversation_manager_handle(
                event_broker=cm.event_broker,
                conversation_id=SESSION_DETAILS.assistant.agent_id,
                contact_id="1",
                transcript_manager=cm.transcript_manager,
                conversation_manager=cm,
            )
        )
    _cmhandle_dur = perf_counter() - local_start_time
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] ConversationManagerHandle ({type(cm._conversation_manager_handle).__name__}) initialized in "
        f"{_cmhandle_dur:.2f} seconds",
    )
    per_manager_init.record(_cmhandle_dur, {"manager": "conversation_manager_handle"})

    # 7. Resolve client customization (org -> team -> user -> assistant cascade)
    LOGGER.debug(
        f"{ICONS['customization']} [ManagersWorker] Resolving customization...",
    )
    local_start_time = perf_counter()
    from unity.customization.clients import resolve as _resolve_customization

    resolved = _resolve_customization(
        org_id=SESSION_DETAILS.org_id,
        team_ids=SESSION_DETAILS.team_ids or None,
        user_id=SESSION_DETAILS.user.id,
        assistant_id=SESSION_DETAILS.assistant.agent_id,
    )
    _resolve_dur = perf_counter() - local_start_time
    LOGGER.info(
        f"{ICONS['customization']} [ManagersWorker] Customization resolved in {_resolve_dur:.2f} seconds",
    )

    # 8. Sync cross-cutting seed data (contacts, guidance, knowledge, secrets, blacklist)
    LOGGER.debug(
        f"{ICONS['customization']} [ManagersWorker] Syncing seed data...",
    )
    local_start_time = perf_counter()
    from unity.customization.seed_sync import sync_all_seed_data

    sync_all_seed_data(resolved)
    _seed_dur = perf_counter() - local_start_time
    LOGGER.info(
        f"{ICONS['customization']} [ManagersWorker] Seed data synced in {_seed_dur:.2f} seconds",
    )

    # 9. Sync custom functions/venvs from client customization
    if resolved.function_dirs or resolved.venv_dirs:
        try:
            LOGGER.debug(
                f"{ICONS['customization']} [ManagersWorker] Syncing custom functions...",
            )
            local_start_time = perf_counter()
            from unity.function_manager.custom_functions import (
                collect_functions_from_directories,
                collect_venvs_from_directories,
            )

            source_fns = collect_functions_from_directories(resolved.function_dirs)
            source_venvs = collect_venvs_from_directories(resolved.venv_dirs)
            fm = ManagerRegistry.get_function_manager()
            if source_fns or source_venvs:
                fm.sync_custom(
                    source_functions=source_fns,
                    source_venvs=source_venvs,
                )
            _func_dur = perf_counter() - local_start_time
            LOGGER.info(
                f"{ICONS['customization']} [ManagersWorker] Custom functions synced in {_func_dur:.2f} seconds",
            )
        except Exception as e:
            LOGGER.warning(
                f"{ICONS['managers_worker']} [ManagersWorker] Custom function sync failed (degraded): {e}",
            )

    # 10. Initialize Actor (use provided actor or create via ManagerRegistry)
    LOGGER.debug(f"{ICONS['managers_worker']} [ManagersWorker] Initializing Actor...")
    try:
        local_start_time = perf_counter()
        if actor is not None:
            # Use pre-instantiated actor (e.g., for testing)
            cm.actor = actor
        else:
            # Create via ManagerRegistry (respects SETTINGS.actor.IMPL)
            from unity.actor.environments import (
                StateManagerEnvironment,
                ComputerEnvironment,
                ActorEnvironment,
            )
            from unity.function_manager.primitives import ComputerPrimitives

            cp = ComputerPrimitives()
            if resolved.config.url_mappings:
                cp.url_mappings = resolved.config.url_mappings

            cm.actor = ManagerRegistry.get_actor(
                description="production deployment",
                environments=[
                    StateManagerEnvironment(),
                    ComputerEnvironment(cp),
                    ActorEnvironment(),
                ],
                resolved=resolved,
            )
        _actor_dur = perf_counter() - local_start_time
        actor_cls = type(cm.actor).__name__
        LOGGER.info(
            f"{ICONS['managers_worker']} [ManagersWorker] Actor ({actor_cls}) initialized in "
            f"{_actor_dur:.2f} seconds",
        )
        per_manager_init.record(_actor_dur, {"manager": "actor"})
    except Exception as e:
        LOGGER.error(
            f"{ICONS['managers_worker']} [ManagersWorker] Error initializing Actor: {e}",
        )

    # 11. Initialize FileManager (eagerly, so the FileRecords context exists
    #     before any file operations or background tasks attempt to use it)
    try:
        LOGGER.info(
            f"{ICONS['managers_worker']} [ManagersWorker] Initializing FileManager...",
        )
        local_start_time = perf_counter()
        fm = ManagerRegistry.get_file_manager()
        # Force the lazy DataManager property to resolve now while ContextVars
        # are correct.  The ingestion pipeline later accesses _data_manager from
        # ThreadPoolExecutor workers where ContextVars may not propagate — eager
        # init avoids the resulting empty-context / double-slash paths.
        _ = fm._data_manager  # noqa: F841
        _file_dur = perf_counter() - local_start_time
        LOGGER.info(
            f"{ICONS['managers_worker']} [ManagersWorker] FileManager initialized in "
            f"{_file_dur:.2f} seconds",
        )
        per_manager_init.record(_file_dur, {"manager": "file_manager"})
    except Exception as e:
        LOGGER.warning(
            f"{ICONS['managers_worker']} [ManagersWorker] FileManager init failed (degraded): {e}",
        )

    # U2: Total manager init duration
    _total_dur = perf_counter() - start_time
    LOGGER.info(
        f"{ICONS['managers_worker']} [ManagersWorker] All managers initialized in {_total_dur:.2f} seconds",
    )
    manager_init_total.record(_total_dur)

    # 12. Eager primitive sync (avoids ~7s cold-start on first execute_function).
    #     Must run after all managers are initialised so the primitive registry
    #     contains every manager's methods (actor, files, contacts, …).
    try:
        LOGGER.debug(
            f"{ICONS['managers_worker']} [ManagersWorker] Syncing primitives...",
        )
        local_start_time = perf_counter()
        _init_fm = ManagerRegistry.get_function_manager()
        _init_fm.sync_primitives()
        _prim_dur = perf_counter() - local_start_time
        LOGGER.info(
            f"{ICONS['managers_worker']} [ManagersWorker] Primitives synced in {_prim_dur:.2f} seconds",
        )
    except Exception as e:
        LOGGER.warning(
            f"{ICONS['managers_worker']} [ManagersWorker] Primitive sync failed (degraded): {e}",
        )

    # 13. Pre-warm embedding columns for all managers (best-effort, avoids
    #     cold-start latency on the first vector search after a fresh hire).
    #     Also explicitly warm the FunctionManager (not in the singleton cache
    #     due to _force_new=True) so Primitives embeddings are ready.
    try:
        LOGGER.debug(
            f"{ICONS['managers_worker']} [ManagersWorker] Warming embedding columns...",
        )
        local_start_time = perf_counter()
        ManagerRegistry.warm_all_embeddings()
        _init_fm.warm_embeddings()
        _warm_dur = perf_counter() - local_start_time
        LOGGER.info(
            f"{ICONS['managers_worker']} [ManagersWorker] Embedding columns warmed in {_warm_dur:.2f} seconds",
        )
    except Exception as e:
        LOGGER.warning(
            f"{ICONS['managers_worker']} [ManagersWorker] Embedding warm-up failed (degraded): {e}",
        )


async def _start_file_sync() -> None:
    """Start file sync with managed VM after managers are initialized.

    This starts rclone-based file synchronization between ~ (assistant home)
    and /home (managed VM) if a desktop_url is configured in SESSION_DETAILS.

    Runs asynchronously and logs success/failure.
    """
    from unity.session_details import SESSION_DETAILS

    # Only sync when a desktop_url is configured
    if not SESSION_DETAILS.assistant.desktop_url:
        LOGGER.debug(
            f"{ICONS['managers_worker']} [ManagersWorker] No desktop_url configured, skipping file sync",
        )
        return

    try:
        from unity.file_manager.managers.local import LocalFileManager

        # Get LocalFileManager singleton (may already exist from manager init)
        local_fm = LocalFileManager()
        adapter = local_fm._adapter

        # Check if adapter supports sync (LocalFileSystemAdapter does)
        if not hasattr(adapter, "start_sync"):
            LOGGER.debug(
                f"{ICONS['managers_worker']} [ManagersWorker] Adapter does not support file sync",
            )
            return

        if adapter._enable_sync:
            LOGGER.debug(
                f"{ICONS['managers_worker']} [ManagersWorker] Starting file sync with managed VM...",
            )
            success = await adapter.start_sync()
            if success:
                LOGGER.debug(
                    f"{ICONS['managers_worker']} [ManagersWorker] File sync started successfully",
                )
            else:
                LOGGER.debug(
                    f"{ICONS['managers_worker']} [ManagersWorker] File sync not enabled or failed to start",
                )
        else:
            LOGGER.debug(
                f"{ICONS['managers_worker']} [ManagersWorker] File sync disabled by configuration",
            )

    except Exception as e:
        # File sync failure should not block manager initialization
        LOGGER.error(
            f"{ICONS['managers_worker']} [ManagersWorker] Failed to start file sync: {e}",
        )
        import traceback

        traceback.print_exc()


async def _register_computer_act_completed_callback(cm: "ConversationManager") -> None:
    """Bridge ``ComputerActCompleted`` events from the in-process EventBUS to the
    CM's ``event_broker`` so both the slow brain and fast brain see them.

    Only publishes when the assistant is actively screen-sharing on a meet.
    """
    from unity.conversation_manager.events import ComputerActCompleted

    async def _on_computer_act_completed(events):  # noqa: ANN001
        if not cm.assistant_screen_share_active:
            return
        for evt in events:
            payload = evt.payload if isinstance(evt.payload, dict) else {}
            cm_event = ComputerActCompleted(
                instruction=payload.get("instruction", ""),
                summary=payload.get("summary", ""),
            )
            await cm.event_broker.publish(
                "app:actor:computer_act_completed",
                cm_event.to_json(),
            )

    try:
        await EVENT_BUS.register_callback(
            event_type="ComputerActCompleted",
            callback=_on_computer_act_completed,
            every_n=1,
        )
    except Exception:
        pass


async def init_conv_manager(
    cm: "ConversationManager",
    *,
    actor: "BaseActor | None" = None,
) -> None:
    """
    Initialize all managers for the ConversationManager.
    All initialization runs in a separate thread (non-blocking).

    Args:
        cm: The ConversationManager instance to initialize.
        actor: Optional pre-instantiated Actor. If provided, used directly instead
            of creating one via ManagerRegistry. Useful for testing with specific
            Actor implementations (e.g., SimulatedActor).
    """
    LOGGER.debug(f"{ICONS['managers_worker']} [ManagersWorker] Processing startup")

    async with _init_lock:
        start_time = perf_counter()
        if cm.initialized:
            LOGGER.info(
                f"{ICONS['managers_worker']} [ManagersWorker] Already initialized, skipping",
            )
            return

        try:
            # Get the main event loop to pass to managers that need it
            loop = asyncio.get_running_loop()

            # Run all manager initialization in a thread (non-blocking).
            # unity.init() inside _init_managers sets Unify ContextVars
            # (CONTEXT_READ/CONTEXT_WRITE) but asyncio.to_thread runs on a
            # copy of the caller's context — changes don't propagate back.
            # Re-apply the context afterwards so any lazily-created managers
            # in the main async context see the correct values.
            await asyncio.to_thread(_init_managers, cm, loop, actor)

            import unify as _unify

            full_ctx = (
                f"{SESSION_DETAILS.user_context}/{SESSION_DETAILS.assistant_context}"
            )
            _unify.set_context(full_ctx, skip_create=True)

            store_chat_history = await get_last_store_chat_history()
            if store_chat_history:
                await cm.event_broker.publish(
                    "app:comms:chat_history",
                    GetChatHistory(
                        chat_history=store_chat_history.chat_history,
                    ).to_json(),
                )

            cm.initialized = True

            # Await the concurrent hydration that was kicked off inside
            # _init_managers right after EventBus config.  In practice it
            # finishes long before this point (hidden behind ContactManager
            # init), so this is effectively a no-op await.
            hydration_future = getattr(cm, "_hydration_future", None)
            if hydration_future is not None:
                try:
                    await asyncio.wrap_future(hydration_future)
                    LOGGER.info(
                        f"{ICONS['managers_worker']} [ManagersWorker] "
                        "Concurrent hydration completed",
                    )
                except Exception as e:
                    LOGGER.error(
                        f"{ICONS['managers_worker']} [ManagersWorker] "
                        f"Global thread hydration failed: {e}",
                    )
                    import traceback

                    traceback.print_exc()
                finally:
                    cm._hydration_future = None

            await _register_computer_act_completed_callback(cm)

            os.environ["UNITY_CM_INITIALIZED"] = "1"

            # Publish initialization complete event.  The registered
            # InitializationComplete handler pushes a notification and
            # triggers a brain turn so it can follow up on deferred requests.
            await event_broker.publish(
                "app:comms:initialization_complete",
                InitializationComplete().to_json(),
            )

            _init_dur = perf_counter() - start_time
            LOGGER.info(
                f"{ICONS['managers_worker']} [ManagersWorker] Initialization complete in {_init_dur:.2f} seconds",
            )

            try:
                from unity.conversation_manager.memory_dump import (
                    write_memory_dump,
                )

                loop = asyncio.get_running_loop()

                def _on_dump_done(fut):
                    try:
                        dump_path = fut.result()
                        if dump_path:
                            LOGGER.info(
                                f"{ICONS['managers_worker']} [ManagersWorker] "
                                f"Startup memory dump written to {dump_path}",
                            )
                    except Exception as exc:
                        LOGGER.warning(
                            f"{ICONS['managers_worker']} [ManagersWorker] "
                            f"Startup memory dump failed: {exc}",
                        )

                fut = loop.run_in_executor(
                    None,
                    write_memory_dump,
                    "startup_memory_dump.txt",
                )
                fut.add_done_callback(_on_dump_done)
            except Exception as exc:
                LOGGER.warning(
                    f"{ICONS['managers_worker']} [ManagersWorker] "
                    f"Startup memory dump failed: {exc}",
                )

        except Exception as e:
            LOGGER.error(
                f"{ICONS['managers_worker']} [ManagersWorker] Error during initialization: {e}",
            )
            publish_system_error(
                "The assistant failed to initialize and may not respond "
                "correctly. Please try again shortly.",
                error_type="init_failed",
            )
            raise
