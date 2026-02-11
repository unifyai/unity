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
import json
import threading
import time
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from google.cloud import pubsub_v1

from unity.settings import SETTINGS
from unity.conversation_manager.domains.comms_utils import (
    add_email_attachments,
    add_unify_message_attachments,
)
from unity.conversation_manager.events import *
from unity.session_details import DEFAULT_ASSISTANT_ID, SESSION_DETAILS
from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.types import Medium

load_dotenv()

# Lock for unknown contact creation to prevent duplicates
_unknown_contact_lock = threading.Lock()


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
    assistant_id = SESSION_DETAILS.assistant.id
    staging_suffix = (
        "-staging"
        if SETTINGS.STAGING and DEFAULT_ASSISTANT_ID not in assistant_id
        else ""
    )
    return f"unity-{assistant_id}{staging_suffix}-sub"


def _get_local_contact() -> dict:
    """Build local contact dict from current assistant context."""
    return {
        "contact_id": -1,
        "first_name": SESSION_DETAILS.user.name,
        "surname": "",
        "phone_number": SESSION_DETAILS.user.number,
        "email_address": SESSION_DETAILS.user.email,
    }


# Map subscription IDs to their corresponding event types
events_map: dict[str, Event] = {
    "msg": SMSReceived,
    "email": EmailReceived,
    "unify_message": UnifyMessageReceived,
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
            print(f"Error in _get_or_create_unknown_contact: {e}")
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

    def handle_message(
        self,
        message: pubsub_v1.types.PubsubMessage,
    ):
        """
        Handle incoming messages from PubSub subscriptions.

        NOTE: This method is called from a GCP PubSub thread pool thread,
        NOT from the asyncio event loop. All async operations must use
        `_publish_from_callback` or `asyncio.run_coroutine_threadsafe`.
        """
        try:
            data = json.loads(message.data.decode("utf-8"))
            thread = data["thread"]
            event = data["event"]
            print(f"Received message from {thread}: {message.data.decode('utf-8')}")
            if thread in ["startup", "assistant_update"]:
                message.ack()
                if thread == "startup":
                    # acknowledge message and cancel startup subscription
                    while startup_subscription_id not in self.subscribers:
                        time.sleep(0.1)
                    self.subscribers[startup_subscription_id].cancel()
                    self.subscribers.pop(startup_subscription_id)

                    # Update assistant context and subscribe to the assistant's subscription
                    # Note: Full context is populated by ConversationManager.set_details()
                    # Here we just need to set assistant_id early for subscription
                    SESSION_DETAILS.assistant.id = event["assistant_id"]
                    self.subscribe_to_topic(_get_subscription_id())

                # publish
                details = {
                    "api_key": event["api_key"],
                    "medium": event.get("medium", "assistant_update"),
                    "assistant_id": event["assistant_id"],
                    "user_id": event["user_id"],
                    "assistant_name": event["assistant_name"],
                    "assistant_age": event["assistant_age"],
                    "assistant_nationality": event["assistant_nationality"],
                    "assistant_timezone": event.get("assistant_timezone", ""),
                    "assistant_about": event["assistant_about"],
                    "assistant_number": event["assistant_number"],
                    "assistant_email": event["assistant_email"],
                    "user_name": event["user_name"],
                    "user_number": event["user_number"],
                    "user_email": event["user_email"],
                    "voice_provider": event["voice_provider"],
                    "voice_id": event["voice_id"],
                    "voice_mode": event["voice_mode"],
                    "desktop_mode": event.get("desktop_mode", "ubuntu"),
                    "desktop_url": event.get("desktop_url"),
                    "user_desktop_mode": event.get("user_desktop_mode"),
                    "user_desktop_filesys_sync": event.get(
                        "user_desktop_filesys_sync",
                        False,
                    ),
                    "user_desktop_url": event.get("user_desktop_url"),
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
            elif thread == "unity_system_event":
                system_event_type = event.get("event_type")
                system_message = event.get("message")
                if system_event_type in ["pause_actor", "resume_actor"]:
                    evt = (
                        ActorPause(
                            reason=(
                                str(system_message)
                                if system_message is not None
                                else "The user has just taken control of the desktop, we're pausing our own actions temporarily."
                            ),
                        )
                        if system_event_type == "pause_actor"
                        else ActorResume(
                            reason=(
                                str(system_message)
                                if system_message is not None
                                else "The user has just handed control of the desktop back to us, we're now continuing our control of the desktop."
                            ),
                        )
                    )
                    self._publish_from_callback(
                        f"app:actor:{system_event_type}",
                        evt.to_json(),
                    )
                elif system_event_type == "sync_contacts":
                    evt = SyncContacts(
                        reason=(
                            str(system_message)
                            if system_message is not None
                            else "Contact sync requested via system event."
                        ),
                    )
                    self._publish_from_callback(
                        f"app:comms:{system_event_type}",
                        evt.to_json(),
                    )
                message.ack()
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
                        print(f"Ignoring blacklisted email from: {contact_detail}")
                        message.ack()
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
                        print(
                            f"Failed to resolve contact for email from: {contact_detail}",
                        )
                        message.ack()
                        return

                    # Extract attachment filenames for the event
                    attachments = event.get("attachments") or []
                    attachment_filenames = [
                        att.get("filename") or f"attachment_{att.get('id', 'unknown')}"
                        for att in attachments
                    ]

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
                            attachments=attachment_filenames,
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

                    # add attachments (if any) to Downloads using async helper
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
                        print(f"Failed scheduling attachment download: {e}")

                elif thread == "unify_message":
                    # contact_id is required - no default to prevent silent privilege escalation
                    # Note: unify_message comes from internal interface, not external unknown senders
                    # so we don't apply blacklist check or unknown contact creation here
                    target_contact_id = event.get("contact_id")
                    if target_contact_id is None:
                        print(
                            "Error: contact_id is required for unify_message, "
                            "skipping message",
                        )
                        message.ack()
                        return
                    contact = next(
                        (c for c in contacts if c["contact_id"] == target_contact_id),
                        None,
                    )
                    if contact is None:
                        print(
                            f"Error: contact_id {target_contact_id} not found in "
                            f"contacts list, skipping message",
                        )
                        message.ack()
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

                    # Download attachments (if any) to Downloads using async helper
                    try:
                        if attachments:
                            asyncio.run_coroutine_threadsafe(
                                add_unify_message_attachments(attachments),
                                self.loop,
                            )
                    except Exception as e:
                        print(f"Failed scheduling attachment download: {e}")

                else:
                    # SMS message (thread == "msg")
                    contact_detail = event["from_number"].strip()
                    medium_for_blacklist = Medium.SMS_MESSAGE

                    # Check blacklist before processing
                    if _is_blacklisted(medium_for_blacklist, contact_detail):
                        print(f"Ignoring blacklisted SMS from: {contact_detail}")
                        message.ack()
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
                        print(
                            f"Failed to resolve contact for SMS from: {contact_detail}",
                        )
                        message.ack()
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

                message.ack()
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
                                metadata={
                                    "source": "pre_hire",
                                    "assistant_id": assistant_id,
                                },
                            )

                            self._publish_from_callback(
                                "app:comms:pre_hire",
                                payload.to_json(),
                            )
                            published += 1
                        except Exception as inner_e:
                            print(f"Skipping malformed pre-hire item: {inner_e}")

                    print(
                        f"Logged {published} pre-hire chat message(s) for assistant {assistant_id}",
                    )
                    message.ack()
                except Exception as e:
                    print(f"Error processing pre-hire logs: {e}")
                    message.nack()
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
                            livekit_agent_name=event.get("livekit_agent_name"),
                            room_name=event.get("livekit_room"),
                        )
                        topic = "app:comms:unify_meet_received"
                    elif thread == "call":
                        number = event.get("caller_number", event.get("user_number"))

                        # Check blacklist before processing
                        if _is_blacklisted(Medium.PHONE_CALL, number):
                            print(f"Ignoring blacklisted call from: {number}")
                            message.ack()
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
                            print(f"Failed to resolve contact for call from: {number}")
                            message.ack()
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
                    message.ack()
                    future.result()  # Wait for publish to complete
                except json.JSONDecodeError:
                    print(f"Invalid message format for {thread} event")
                    message.ack()
                except Exception as e:
                    print(f"Error processing {thread} event: {e}")
                    import traceback

                    traceback.print_exc()
                    message.ack()
            else:
                print(f"Unknown event type: {thread}")
        except Exception as e:
            print(f"Error processing message: {e}")
            message.ack()

    def subscribe_to_topic(self, subscription_id: str):
        # async def subscribe_to_topic(self, subscription_id: str):
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

            print(f"Starting subscription to {subscription_path}")

            streaming_pull_future = subscriber.subscribe(
                subscription_path,
                callback=self.handle_message,
            )

            # Store the future for cleanup
            self.subscribers[subscription_id] = streaming_pull_future

        except Exception as e:
            print(f"Error setting up subscription {subscription_id}: {e}")

    async def start(self):
        """Start all subscriptions and maintain connection to event manager."""
        if SESSION_DETAILS.assistant.id == DEFAULT_ASSISTANT_ID:
            # Start the startup subscription
            self.subscribe_to_topic(startup_subscription_id)
            # Start ping mechanism for idle containers
            asyncio.create_task(self.send_pings())
        else:
            # Start subscription
            self.subscribe_to_topic(_get_subscription_id())

        # Keep the connection alive
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
            # Cleanup subscriptions
            for future in self.subscribers.values():
                future.cancel()

    async def send_pings(self):
        """Send periodic pings to keep the event manager alive while waiting for startup."""
        print("Starting ping mechanism for idle container...")
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
                if SESSION_DETAILS.assistant.id != DEFAULT_ASSISTANT_ID:
                    print("Startup received, stopping ping mechanism")
                    break

            except Exception as e:
                print(f"Error in ping mechanism: {e}")
                await asyncio.sleep(30)  # Continue trying


async def main():
    """Main entry point for the communication manager application."""
    from unity.conversation_manager.event_broker import get_event_broker

    event_broker = get_event_broker()
    manager = CommsManager(event_broker)
    await manager.start()


if __name__ == "__main__":
    asyncio.run(main(), debug=SETTINGS.ASYNCIO_DEBUG)
