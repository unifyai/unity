"""
CommsManager: External communications handler for ConversationManager.

This module bridges external communication channels (GCP PubSub for SMS, email,
calls, etc.) to the internal event broker.

Threading Model:
----------------
GCP PubSub uses a thread pool for message callbacks. The `handle_message` method
is called from these background threads, NOT from the asyncio event loop. Therefore:

- `handle_message` uses `asyncio.run_coroutine_threadsafe()` to safely publish
  events to the async event broker from a sync callback context.
- `send_pings` and `start` are async methods that run on the event loop and can
  use direct `await` for async operations.

Testing:
--------
For testing, CommsManager is typically disabled (enable_comms_manager=False) since
there are no real external events to receive. Tests can publish events directly
to the event broker instead.
"""

from __future__ import annotations

import asyncio
from functools import partial
import json
import threading
import time
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from google.cloud import pubsub_v1

from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON, ICONS
from unity.settings import SETTINGS
from unity.conversation_manager.assistant_jobs import mark_job_label
from unity.conversation_manager.domains.comms_utils import (
    add_email_attachments,
    add_unify_message_attachments,
)
from unity.conversation_manager.events import *
from unity.conversation_manager.metrics import pubsub_e2e_latency
from unity.session_details import SESSION_DETAILS
from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.types import Medium

load_dotenv()

# Lock for unknown contact creation to prevent duplicates
_unknown_contact_lock = threading.Lock()

# Lock for startup transition to prevent multiple messages being acked by the same container
_startup_lock = threading.Lock()


if TYPE_CHECKING:
    from unity.conversation_manager.in_memory_event_broker import InMemoryEventBroker

    EventBroker = InMemoryEventBroker


# Subscription IDs
project_id = "responsive-city-458413-a2"
startup_subscription_id = (
    "unity-startup" + ("-staging" if SETTINGS.STAGING else "") + "-sub"
)


def _get_subscription_id() -> str:
    """Build subscription ID from current assistant context."""
    agent_id = SESSION_DETAILS.assistant.agent_id
    staging_suffix = "-staging" if SETTINGS.STAGING and agent_id is not None else ""
    return f"unity-{agent_id}{staging_suffix}-sub"


def _get_local_contact() -> dict:
    """Build local contact dict from current assistant context."""
    return {
        "contact_id": -1,
        "first_name": SESSION_DETAILS.user.first_name,
        "surname": SESSION_DETAILS.user.surname,
        "phone_number": SESSION_DETAILS.user.number,
        "email_address": SESSION_DETAILS.user.email,
    }


# Map subscription IDs to their corresponding event types
events_map: dict[str, Event] = {
    "msg": SMSReceived,
    "email": EmailReceived,
    "unify_message": UnifyMessageReceived,
    "api_message": ApiMessageReceived,
}


def _is_blacklisted(medium: str, contact_detail: str | None) -> bool:
    """
    Check if a contact detail is blacklisted for a given medium.

    This is a fail-open check: returns False on any error to avoid
    blocking legitimate messages due to infrastructure issues.

    Gated by SETTINGS.conversation.BLACKLIST_CHECKS_ENABLED (default False).
    When disabled, returns False immediately without any manager initialization.

    Args:
        medium: The communication medium (e.g., "sms_message", "email", "phone_call")
        contact_detail: The phone number or email address to check

    Returns:
        True if the contact detail is blacklisted, False otherwise
    """
    # Fast path: skip all manager initialization when blacklist checks disabled
    if not SETTINGS.conversation.BLACKLIST_CHECKS_ENABLED:
        return False

    if not contact_detail:
        return False

    try:
        from unity.blacklist_manager import BlackListManager

        blm = BlackListManager()
        result = blm.filter_blacklist(
            filter=f"medium == '{medium}' and contact_detail == '{contact_detail}'",
            limit=1,
        )
        return len(result.get("entries", [])) > 0
    except Exception:
        # Fail-open: don't block messages if blacklist check fails
        return False


def _get_or_create_unknown_contact(
    medium: str,
    contact_detail: str,
) -> dict | None:
    """
    Get an existing contact or create a new unknown contact.

    When an inbound message arrives from an unknown sender (not in Contacts
    and not in BlackList), we create a minimal contact record with:
    - Only the medium field populated (phone_number or email_address)
    - should_respond=False to prevent automatic responses
    - A response_policy guiding the assistant to seek boss guidance

    Uses a lock to prevent duplicate contact creation when multiple
    messages arrive from the same unknown sender simultaneously.

    Gated by SETTINGS.conversation.BLACKLIST_CHECKS_ENABLED (default False).
    When disabled, returns None immediately without any manager initialization.

    Args:
        medium: The communication medium (determines which contact field to set)
        contact_detail: The phone number or email address

    Returns:
        The contact dict (existing or newly created), or None on error
    """
    # Fast path: skip all manager initialization when blacklist checks disabled
    if not SETTINGS.conversation.BLACKLIST_CHECKS_ENABLED:
        return None

    from unity.manager_registry import ManagerRegistry
    from unity.contact_manager.contact_manager import ContactManager

    with _unknown_contact_lock:
        try:
            cm = ManagerRegistry.get_contact_manager()

            # Determine which field to search/set based on medium
            if medium in ("sms_message", "phone_call"):
                field_name = "phone_number"
            elif medium == "email":
                field_name = "email_address"
            else:
                # For unify_message, we don't have external contact details
                return None

            # Check if contact already exists
            result = cm.filter_contacts(
                filter=f"{field_name} == '{contact_detail}'",
                limit=1,
            )
            existing = result.get("contacts", [])
            if existing:
                contact = existing[0]
                return (
                    contact.model_dump() if hasattr(contact, "model_dump") else contact
                )

            # Create new unknown contact
            create_kwargs = {
                field_name: contact_detail,
                "should_respond": False,
                "response_policy": ContactManager.UNKNOWN_INBOUND_RESPONSE_POLICY,
            }
            outcome = cm._create_contact(**create_kwargs)
            new_contact_id = outcome["details"]["contact_id"]

            # Fetch the newly created contact
            contact_info = cm.get_contact_info(new_contact_id)
            new_contact = contact_info.get(new_contact_id)
            return new_contact

        except Exception as e:
            LOGGER.error(f"{DEFAULT_ICON} Error in _get_or_create_unknown_contact: {e}")
            return None


class CommsManager:
    """
    Handles external communications via GCP PubSub.

    Receives events from external channels (SMS, email, calls) and publishes
    them to the internal event broker for ConversationManager to process.
    """

    def __init__(self, event_broker: "EventBroker"):
        self.subscribers: dict = {}
        self.call_proc = None
        self.credentials = None
        # Store reference to event loop for thread-safe publishing from callbacks
        self.loop = asyncio.get_event_loop()
        self.event_broker: "EventBroker" = event_broker

    def _publish_from_callback(self, channel: str, message: str) -> None:
        """
        Publish to event broker from a sync callback (thread-safe).

        This method is called from GCP PubSub callbacks which run in a thread pool,
        NOT from the asyncio event loop. We use run_coroutine_threadsafe to safely
        schedule the async publish on the main event loop.
        """
        asyncio.run_coroutine_threadsafe(
            self.event_broker.publish(channel, message),
            self.loop,
        )

    def _ack_with_latency(self, message, publish_timestamp, topic):
        """Ack the message and record end-to-end Pub/Sub latency if available."""
        if publish_timestamp is not None:
            latency = time.time() - publish_timestamp
            pubsub_e2e_latency.record(latency, {"topic": topic})
        message.ack()

    def handle_message(
        self,
        message: pubsub_v1.types.PubsubMessage,
        subscription_id: str = "",
    ):
        """
        Handle incoming messages from PubSub subscriptions.

        NOTE: This method is called from a GCP PubSub thread pool thread,
        NOT from the asyncio event loop. All async operations must use
        `_publish_from_callback` or `asyncio.run_coroutine_threadsafe`.
        """
        topic = subscription_id.removesuffix("-sub")
        try:
            data = json.loads(message.data.decode("utf-8"))
            thread = data["thread"]
            event = data["event"]
            publish_timestamp = data.get("publish_timestamp")
            LOGGER.debug(
                f"{DEFAULT_ICON} Received message from {thread}: {message.data.decode('utf-8')}",
            )
            if thread in ["startup", "assistant_update"]:
                if thread == "startup":
                    with _startup_lock:
                        # If already assigned, nack the message so another idle container can pick it up
                        if SESSION_DETAILS.assistant.agent_id is not None:
                            LOGGER.debug(
                                f"{DEFAULT_ICON} Already assigned to assistant {SESSION_DETAILS.assistant.agent_id}, "
                                f"nacking startup message for {event.get('assistant_id')}",
                            )
                            message.nack()
                            return

                        # Acknowledge message only when we're sure we're taking it
                        self._ack_with_latency(message, publish_timestamp, topic)

                        # can't use asyncio.to_thread because this code runs on a pubsub
                        # thread pool thread rather than the main event loop
                        threading.Thread(
                            target=mark_job_label,
                            args=(SETTINGS.conversation.JOB_NAME, "running"),
                            daemon=True,
                        ).start()

                        # cancel startup subscription
                        while startup_subscription_id not in self.subscribers:
                            time.sleep(0.1)
                        self.subscribers[startup_subscription_id].cancel()
                        self.subscribers.pop(startup_subscription_id)

                    # Update assistant context and subscribe to the assistant's subscription
                    # Note: Full context is populated by ConversationManager.set_details()
                    # Here we just need to set assistant_id early for subscription
                    SESSION_DETAILS.assistant.agent_id = int(event["assistant_id"])
                    self.subscribe_to_topic(_get_subscription_id(), max_messages=10)
                else:
                    # assistant_update - always ack
                    self._ack_with_latency(message, publish_timestamp, topic)

                # publish
                details = {
                    "api_key": event["api_key"],
                    "medium": event.get("medium", "assistant_update"),
                    "assistant_id": event["assistant_id"],
                    "user_id": event["user_id"],
                    "assistant_first_name": event["assistant_first_name"],
                    "assistant_surname": event["assistant_surname"],
                    "assistant_age": event["assistant_age"],
                    "assistant_nationality": event["assistant_nationality"],
                    "assistant_timezone": event.get("assistant_timezone", ""),
                    "assistant_about": event["assistant_about"],
                    "assistant_number": event["assistant_number"],
                    "assistant_email": event["assistant_email"],
                    "user_first_name": event["user_first_name"],
                    "user_surname": event["user_surname"],
                    "user_number": event["user_number"],
                    "user_email": event["user_email"],
                    "voice_provider": event["voice_provider"],
                    "voice_id": event["voice_id"],
                    "desktop_mode": event.get("desktop_mode", "ubuntu"),
                    "user_desktop_mode": event.get("user_desktop_mode"),
                    "user_desktop_filesys_sync": event.get(
                        "user_desktop_filesys_sync",
                        False,
                    ),
                    "user_desktop_url": event.get("user_desktop_url"),
                    "org_id": event.get("org_id"),
                    "org_name": event.get("org_name", ""),
                    "team_ids": event.get("team_ids") or [],
                    "demo_id": event.get("demo_id"),
                }
                self._publish_from_callback(
                    f"app:comms:{thread}",
                    (
                        StartupEvent(**details)
                        if thread == "startup"
                        else AssistantUpdateEvent(**details)
                    ).to_json(),
                )
            elif thread == "ping":
                self._publish_from_callback(
                    "app:comms:ping",
                    Ping(kind="keepalive").to_json(),
                )
                self._ack_with_latency(message, publish_timestamp, topic)
            elif thread == "unity_system_event":
                system_event_type = event.get("event_type")
                system_message = event.get("message")
                reason = str(system_message) if system_message is not None else ""

                # Map system event types to internal event classes.
                _SYSTEM_EVENT_MAP = {
                    "sync_contacts": lambda r: SyncContacts(
                        reason=r or "Contact sync requested via system event.",
                    ),
                    "assistant_screen_share_started": lambda r: AssistantScreenShareStarted(
                        reason=r or "User enabled assistant screen sharing.",
                    ),
                    "assistant_screen_share_stopped": lambda r: AssistantScreenShareStopped(
                        reason=r or "User disabled assistant screen sharing.",
                    ),
                    "user_screen_share_started": lambda r: UserScreenShareStarted(
                        reason=r or "User started sharing their screen.",
                    ),
                    "user_screen_share_stopped": lambda r: UserScreenShareStopped(
                        reason=r or "User stopped sharing their screen.",
                    ),
                    "user_webcam_started": lambda r: UserWebcamStarted(),
                    "user_webcam_stopped": lambda r: UserWebcamStopped(),
                    "user_remote_control_started": lambda r: UserRemoteControlStarted(
                        reason=r or "User took remote control of assistant desktop.",
                    ),
                    "user_remote_control_stopped": lambda r: UserRemoteControlStopped(
                        reason=r
                        or "User released remote control of assistant desktop.",
                    ),
                    "assistant_desktop_ready": lambda r: AssistantDesktopReady(
                        desktop_url=event.get("desktop_url")
                        or SESSION_DETAILS.assistant.desktop_url
                        or "",
                        vm_type=event.get("vm_type")
                        or SESSION_DETAILS.assistant.desktop_mode,
                    ),
                }

                factory = _SYSTEM_EVENT_MAP.get(system_event_type)
                if factory is not None:
                    evt = factory(reason)
                    self._publish_from_callback(
                        f"app:comms:{system_event_type}",
                        evt.to_json(),
                    )
                self._ack_with_latency(message, publish_timestamp, topic)
            elif thread in events_map:
                # Get contacts for message routing
                contacts = [*event.get("contacts", []), _get_local_contact()]

                # Publish backup contacts for use before ContactManager is initialized
                self._publish_from_callback(
                    "app:comms:backup_contacts",
                    BackupContactsEvent(contacts=contacts).to_json(),
                )

                content = event["body"]
                contact_detail = ""
                medium_for_blacklist = ""

                if thread == "email":
                    content = "Subject: " + event["subject"] + "\n\n" + event["body"]
                    contact_detail = event["from"].split("<")[1][:-1]
                    medium_for_blacklist = Medium.EMAIL

                    # Check blacklist before processing
                    if _is_blacklisted(medium_for_blacklist, contact_detail):
                        LOGGER.debug(
                            f"{DEFAULT_ICON} Ignoring blacklisted email from: {contact_detail}",
                        )
                        self._ack_with_latency(message, publish_timestamp, topic)
                        return

                    # Find or create contact
                    contact = next(
                        (c for c in contacts if c["email_address"] == contact_detail),
                        None,
                    )
                    is_new_unknown = False
                    if contact is None:
                        # Unknown sender - create minimal contact
                        contact = _get_or_create_unknown_contact(
                            medium_for_blacklist,
                            contact_detail,
                        )
                        is_new_unknown = contact is not None

                    if contact is None:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Failed to resolve contact for email from: {contact_detail}",
                        )
                        self._ack_with_latency(message, publish_timestamp, topic)
                        return

                    # Extract attachment metadata for the event
                    attachments = event.get("attachments") or []

                    # Extract to/cc/bcc - normalize to lists
                    def _normalize_recipients(val):
                        if not val:
                            return []
                        if isinstance(val, str):
                            return [val] if val else []
                        return list(val)

                    self._publish_from_callback(
                        f"app:comms:{thread}_message",
                        events_map[thread](
                            subject=event["subject"],
                            body=event["body"],
                            contact=contact,
                            email_id=event["email_id"],
                            attachments=attachments,
                            to=_normalize_recipients(event.get("to")),
                            cc=_normalize_recipients(event.get("cc")),
                            bcc=_normalize_recipients(event.get("bcc")),
                        ).to_json(),
                    )

                    # Publish UnknownContactCreated event if this was a new unknown contact
                    if is_new_unknown:
                        self._publish_from_callback(
                            "app:comms:unknown_contact_created",
                            UnknownContactCreated(
                                contact=contact,
                                medium=medium_for_blacklist,
                                message_preview=(
                                    event["subject"][:100]
                                    if event.get("subject")
                                    else ""
                                ),
                            ).to_json(),
                        )

                    # add attachments (if any) to Attachments using async helper
                    try:
                        if attachments:
                            asyncio.run_coroutine_threadsafe(
                                add_email_attachments(
                                    attachments,
                                    SESSION_DETAILS.assistant.email,
                                    event.get("gmail_message_id", ""),
                                ),
                                self.loop,
                            )
                    except Exception as e:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Failed scheduling attachment download: {e}",
                        )

                elif thread == "unify_message":
                    # contact_id is required - no default to prevent silent privilege escalation
                    # Note: unify_message comes from internal interface, not external unknown senders
                    # so we don't apply blacklist check or unknown contact creation here
                    target_contact_id = event.get("contact_id")
                    if target_contact_id is None:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Error: contact_id is required for unify_message, "
                            "skipping message",
                        )
                        self._ack_with_latency(message, publish_timestamp, topic)
                        return
                    contact = next(
                        (c for c in contacts if c["contact_id"] == target_contact_id),
                        None,
                    )
                    if contact is None:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Error: contact_id {target_contact_id} not found in "
                            f"contacts list, skipping message",
                        )
                        self._ack_with_latency(message, publish_timestamp, topic)
                        return

                    # Extract attachments with full metadata for the event
                    attachments = event.get("attachments") or []

                    self._publish_from_callback(
                        f"app:comms:{thread}_message",
                        events_map[thread](
                            content=content,
                            contact=contact,
                            attachments=attachments,  # Pass full metadata
                        ).to_json(),
                    )

                    # Download attachments (if any) to Attachments using async helper
                    try:
                        if attachments:
                            asyncio.run_coroutine_threadsafe(
                                add_unify_message_attachments(attachments),
                                self.loop,
                            )
                    except Exception as e:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Failed scheduling attachment download: {e}",
                        )

                elif thread == "api_message":
                    target_contact_id = event.get("contact_id", 1)
                    contact = next(
                        (c for c in contacts if c["contact_id"] == target_contact_id),
                        contacts[0] if contacts else {},
                    )
                    api_message_id = event.get("api_message_id", "")
                    attachments = event.get("attachments") or []
                    tags = event.get("tags") or []

                    self._publish_from_callback(
                        f"app:comms:{thread}_message",
                        events_map[thread](
                            content=content,
                            contact=contact,
                            api_message_id=api_message_id,
                            attachments=attachments,
                            tags=tags,
                        ).to_json(),
                    )

                    # Download attachments (if any) to Attachments — reuse the
                    # same helper used by unify_message.
                    try:
                        if attachments:
                            asyncio.run_coroutine_threadsafe(
                                add_unify_message_attachments(attachments),
                                self.loop,
                            )
                    except Exception as e:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Failed scheduling api_message attachment download: {e}",
                        )

                else:
                    # SMS message (thread == "msg")
                    contact_detail = event["from_number"].strip()
                    medium_for_blacklist = Medium.SMS_MESSAGE

                    # Check blacklist before processing
                    if _is_blacklisted(medium_for_blacklist, contact_detail):
                        LOGGER.debug(
                            f"{DEFAULT_ICON} Ignoring blacklisted SMS from: {contact_detail}",
                        )
                        self._ack_with_latency(message, publish_timestamp, topic)
                        return

                    # Find or create contact
                    contact = next(
                        (c for c in contacts if c["phone_number"] == contact_detail),
                        None,
                    )
                    is_new_unknown = False
                    if contact is None:
                        # Unknown sender - create minimal contact
                        contact = _get_or_create_unknown_contact(
                            medium_for_blacklist,
                            contact_detail,
                        )
                        is_new_unknown = contact is not None

                    if contact is None:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Failed to resolve contact for SMS from: {contact_detail}",
                        )
                        self._ack_with_latency(message, publish_timestamp, topic)
                        return

                    self._publish_from_callback(
                        f"app:comms:{thread}_message",
                        events_map[thread](
                            content=content,
                            contact=contact,
                        ).to_json(),
                    )

                    # Publish UnknownContactCreated event if this was a new unknown contact
                    if is_new_unknown:
                        self._publish_from_callback(
                            "app:comms:unknown_contact_created",
                            UnknownContactCreated(
                                contact=contact,
                                medium=medium_for_blacklist,
                                message_preview=content[:100] if content else "",
                            ).to_json(),
                        )

                self._ack_with_latency(message, publish_timestamp, topic)
            elif thread == "log_pre_hire_chats":
                try:
                    contacts = [*event.get("contacts", []), _get_local_contact()]
                    assistant_id = event.get("assistant_id", "")
                    body = event.get("body", []) or []

                    published = 0
                    for item in body:
                        try:
                            role = item.get("role")
                            msg_content = item.get("msg", "")
                            if not isinstance(msg_content, str):
                                msg_content = str(msg_content)

                            payload = PreHireMessage(
                                content=msg_content,
                                role=role,
                                exchange_id=UNASSIGNED,
                            )

                            self._publish_from_callback(
                                "app:comms:pre_hire",
                                payload.to_json(),
                            )
                            published += 1
                        except Exception as inner_e:
                            LOGGER.debug(
                                f"{DEFAULT_ICON} Skipping malformed pre-hire item: {inner_e}",
                            )

                    LOGGER.debug(
                        f"{DEFAULT_ICON} Logged {published} pre-hire chat message(s) for assistant {assistant_id}",
                    )
                    self._ack_with_latency(message, publish_timestamp, topic)
                except Exception as e:
                    LOGGER.error(f"{DEFAULT_ICON} Error processing pre-hire logs: {e}")
                    message.nack()
            elif thread == "recording_ready":
                recording_event = RecordingReady(
                    conference_name=event.get("conference_name", ""),
                    recording_url=event.get("recording_url", ""),
                )
                self._publish_from_callback(
                    "app:comms:recording_ready",
                    recording_event.to_json(),
                )
                self._ack_with_latency(message, publish_timestamp, topic)
            elif "call" in thread or "meet" in thread:
                try:
                    # Get contacts for call routing
                    contacts = [*event.get("contacts", []), _get_local_contact()]

                    # Publish backup contacts for use before ContactManager is initialized
                    self._publish_from_callback(
                        "app:comms:backup_contacts",
                        BackupContactsEvent(contacts=contacts).to_json(),
                    )

                    # Create the event based on the thread
                    if thread == "unify_meet":
                        # unify_meet is internal, no blacklist check needed
                        call_event = UnifyMeetReceived(
                            contact=next(c for c in contacts if c["contact_id"] == 1),
                            room_name=event.get("livekit_room"),
                        )
                        topic = "app:comms:unify_meet_received"
                    elif thread == "call":
                        number = event.get("caller_number", event.get("user_number"))

                        # Check blacklist before processing
                        if _is_blacklisted(Medium.PHONE_CALL, number):
                            LOGGER.debug(
                                f"{DEFAULT_ICON} Ignoring blacklisted call from: {number}",
                            )
                            self._ack_with_latency(message, publish_timestamp, topic)
                            return

                        # Find or create contact
                        contact = next(
                            (c for c in contacts if c["phone_number"] == number),
                            None,
                        )
                        is_new_unknown = False
                        if contact is None:
                            # Unknown caller - create minimal contact
                            contact = _get_or_create_unknown_contact(
                                Medium.PHONE_CALL,
                                number,
                            )
                            is_new_unknown = contact is not None

                        if contact is None:
                            LOGGER.error(
                                f"{DEFAULT_ICON} Failed to resolve contact for call from: {number}",
                            )
                            self._ack_with_latency(message, publish_timestamp, topic)
                            return

                        call_event = PhoneCallReceived(
                            contact=contact,
                            conference_name=event.get("conference_name", ""),
                        )
                        topic = "app:comms:call_received"

                        # Publish UnknownContactCreated event if this was a new unknown contact
                        if is_new_unknown:
                            self._publish_from_callback(
                                "app:comms:unknown_contact_created",
                                UnknownContactCreated(
                                    contact=contact,
                                    medium=Medium.PHONE_CALL,
                                    message_preview="Incoming phone call",
                                ).to_json(),
                            )
                    elif thread == "call_not_answered":
                        # Outbound call was not answered (no-answer, busy, canceled, failed)
                        number = event.get("user_number")
                        call_status = event.get("call_status", "no-answer")
                        contact = next(
                            (c for c in contacts if c["phone_number"] == number),
                            None,
                        )
                        if contact is None:
                            # Fallback to boss contact
                            contact = next(c for c in contacts if c["contact_id"] == 1)
                        call_event = PhoneCallNotAnswered(
                            contact=contact,
                            reason=call_status,
                        )
                        topic = "app:comms:call_not_answered"
                    else:
                        # call_answered - typically from known contacts initiating outbound
                        number = event.get("user_number")
                        contact = next(
                            (c for c in contacts if c["phone_number"] == number),
                            None,
                        )
                        if contact is None:
                            # Fallback to boss contact for answered calls
                            contact = next(c for c in contacts if c["contact_id"] == 1)
                        call_event = PhoneCallAnswered(contact=contact)
                        topic = "app:comms:call_answered"

                    # Publish the event (blocking wait for call events)
                    future = asyncio.run_coroutine_threadsafe(
                        self.event_broker.publish(topic, call_event.to_json()),
                        self.loop,
                    )
                    self._ack_with_latency(message, publish_timestamp, topic)
                    future.result()  # Wait for publish to complete
                except json.JSONDecodeError:
                    LOGGER.error(
                        f"{DEFAULT_ICON} Invalid message format for {thread} event",
                    )
                    self._ack_with_latency(message, publish_timestamp, topic)
                except Exception as e:
                    LOGGER.error(f"{DEFAULT_ICON} Error processing {thread} event: {e}")
                    import traceback

                    traceback.print_exc()
                    self._ack_with_latency(message, publish_timestamp, topic)
            else:
                if thread != "assistant_desktop_ready":
                    LOGGER.error(f"{DEFAULT_ICON} Unknown event type: {thread}")
                self._ack_with_latency(message, publish_timestamp, topic)
        except Exception as e:
            LOGGER.error(f"{DEFAULT_ICON} Error processing message: {e}")
            message.ack()

    def subscribe_to_topic(self, subscription_id: str, max_messages: int | None = None):
        """Subscribe to a specific PubSub topic and process messages."""
        try:
            # Let GCP libraries handle authentication automatically
            if self.credentials:
                subscriber = pubsub_v1.SubscriberClient(credentials=self.credentials)
            else:
                subscriber = pubsub_v1.SubscriberClient()
            subscription_path = subscriber.subscription_path(
                project_id,
                subscription_id,
            )

            LOGGER.debug(
                f"{ICONS['subscription']} Starting subscription to {subscription_path} (max_messages={max_messages})",
            )

            flow_control = (
                pubsub_v1.types.FlowControl(max_messages=max_messages)
                if max_messages
                else pubsub_v1.types.FlowControl()
            )

            callback = partial(self.handle_message, subscription_id=subscription_id)
            streaming_pull_future = subscriber.subscribe(
                subscription_path,
                callback=callback,
                flow_control=flow_control,
            )

            # Store the future for cleanup
            self.subscribers[subscription_id] = streaming_pull_future

        except Exception as e:
            LOGGER.error(
                f"{ICONS['subscription']} Error setting up subscription {subscription_id}: {e}",
            )

    async def start(self):
        """Start all subscriptions and maintain connection to event manager."""
        if SESSION_DETAILS.assistant.agent_id is None:
            # Start the startup subscription with max_messages=1 to prevent batch stealing
            self.subscribe_to_topic(startup_subscription_id, max_messages=1)
            # Label this container as idle so cleanup endpoints can find it
            threading.Thread(
                target=mark_job_label,
                args=(SETTINGS.conversation.JOB_NAME, "idle"),
                daemon=True,
            ).start()
            # Start ping mechanism for idle containers
            asyncio.create_task(self.send_pings())
        else:
            # Start subscription for live assistant
            self.subscribe_to_topic(_get_subscription_id(), max_messages=10)

        # Keep the connection alive
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            LOGGER.debug(f"{ICONS['lifecycle']} Shutting down...")
            # Cleanup subscriptions
            for future in self.subscribers.values():
                future.cancel()

    async def send_pings(self):
        """Send periodic pings to keep the event manager alive while waiting for startup."""
        LOGGER.debug(
            f"{ICONS['subscription']} Starting ping mechanism for idle container...",
        )
        while True:
            try:
                # Send ping to event manager (direct await since we're in async context)
                await self.event_broker.publish(
                    "app:comms:ping",
                    Ping(kind="keepalive").to_json(),
                )

                # Wait 30 seconds before next ping (half the inactivity timeout)
                await asyncio.sleep(30)

                # Check if we've received a startup message (indicated by assistant_id changed)
                if SESSION_DETAILS.assistant.agent_id is not None:
                    LOGGER.debug(
                        f"{ICONS['subscription']} Startup received, stopping ping mechanism",
                    )
                    break

            except Exception as e:
                LOGGER.error(f"{ICONS['subscription']} Error in ping mechanism: {e}")
                await asyncio.sleep(30)  # Continue trying


async def main():
    """Main entry point for the communication manager application."""
    from unity.conversation_manager.event_broker import get_event_broker

    event_broker = get_event_broker()
    manager = CommsManager(event_broker)
    await manager.start()


if __name__ == "__main__":
    asyncio.run(main(), debug=SETTINGS.UNITY_ASYNCIO_DEBUG)
