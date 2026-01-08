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
import time
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from google.cloud import pubsub_v1

from unity.settings import SETTINGS
from unity.conversation_manager.domains.comms_utils import add_email_attachments
from unity.conversation_manager.events import *
from unity.session_details import DEFAULT_ASSISTANT_ID, SESSION_DETAILS

load_dotenv()

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

                    # Update VNC password and restart x11vnc using UNIFY_KEY (atomic swap)
                    try:
                        import subprocess

                        api_key = event.get("api_key", "") or SESSION_DETAILS.unify_key
                        env = SESSION_DETAILS.get_subprocess_env(UNIFY_KEY=api_key)
                        subprocess.run(
                            ["/bin/bash", "/app/desktop/update_vnc_password.sh"],
                            check=True,
                            env=env,
                        )
                    except Exception as e:
                        print(f"Failed to update VNC password: {e}")

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
                # Publish contacts
                contacts = [*event.get("contacts", []), _get_local_contact()]
                self._publish_from_callback(
                    "app:comms:contacts",
                    GetContactsResponse(contacts=contacts).to_json(),
                )

                content = event["body"]
                topic = ""
                if thread == "email":
                    content = "Subject: " + event["subject"] + "\n\n" + event["body"]
                    topic = event["from"].split("<")[1][:-1]
                    contact = next(c for c in contacts if c["email"] == topic)
                    self._publish_from_callback(
                        f"app:comms:{thread}_message",
                        events_map[thread](
                            subject=event["subject"],
                            body=event["body"],
                            contact=contact,
                            email_id=event["email_id"],
                        ).to_json(),
                    )

                    # add attachments (if any) to Downloads using async helper
                    try:
                        attachments = event.get("attachments") or []
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
                    # Use contact_id from event if provided, otherwise default to boss (1)
                    target_contact_id = event.get("contact_id", 1)
                    contact = next(
                        (c for c in contacts if c["contact_id"] == target_contact_id),
                        None,
                    )
                    if contact is None:
                        print(
                            f"Warning: contact_id {target_contact_id} not found, "
                            f"falling back to boss contact (1)",
                        )
                        contact = next(c for c in contacts if c["contact_id"] == 1)
                    self._publish_from_callback(
                        f"app:comms:{thread}_message",
                        events_map[thread](
                            content=content,
                            contact=contact,
                        ).to_json(),
                    )

                else:
                    topic = event["from_number"].strip()
                    # Put the message in the queue instead of creating a task
                    contact = next(c for c in contacts if c["phone_number"] == topic)
                    self._publish_from_callback(
                        f"app:comms:{thread}_message",
                        events_map[thread](
                            content=content,
                            contact=contact,
                        ).to_json(),
                    )
                message.ack()
            elif thread == "log_pre_hire_chats":
                try:
                    contacts = [*event.get("contacts", []), _get_local_contact()]
                    self._publish_from_callback(
                        "app:comms:contacts",
                        GetContactsResponse(contacts=contacts).to_json(),
                    )
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
                                exchange_id=0,
                                metadata={
                                    "source": "pre_hire",
                                    "assistant_id": assistant_id,
                                },
                            )

                            self._publish_from_callback(
                                "app:managers:input",
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
                    # Publish contacts
                    contacts = [*event.get("contacts", []), _get_local_contact()]
                    self._publish_from_callback(
                        "app:comms:contacts",
                        GetContactsResponse(contacts=contacts).to_json(),
                    )

                    # Create the event based on the thread
                    if thread == "unify_meet":
                        event = UnifyMeetReceived(
                            contact=next(c for c in contacts if c["contact_id"] == 1),
                            agent_name=event.get("agent_name"),
                            room_name=event.get("livekit_room"),
                        )
                        topic = "app:comms:unify_meet_received"
                    elif thread == "call":
                        number = event.get("caller_number", event.get("user_number"))
                        contact = next(
                            c for c in contacts if c["phone_number"] == number
                        )
                        event = PhoneCallReceived(
                            contact=contact,
                            conference_name=event.get("conference_name", ""),
                        )
                        topic = "app:comms:call_received"
                    else:
                        number = event.get("user_number")
                        contact = next(
                            c for c in contacts if c["phone_number"] == number
                        )
                        event = PhoneCallAnswered(contact=contact)
                        topic = "app:comms:call_answered"

                    # Publish the event (blocking wait for call events)
                    future = asyncio.run_coroutine_threadsafe(
                        self.event_broker.publish(topic, event.to_json()),
                        self.loop,
                    )
                    message.ack()
                    future.result()  # Wait for publish to complete
                except json.JSONDecodeError:
                    print(f"Invalid message format for {thread} event")
                    message.ack()
                except Exception as e:
                    print(f"Error processing {thread} event: {e}")
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
