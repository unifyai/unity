from __future__ import annotations

import asyncio
import threading
from datetime import datetime
import os
from typing import Optional

import redis.asyncio as redis

import unity
from unity.contact_manager.contact_manager import ContactManager
from unity.events.event_bus import EVENT_BUS
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.conversation_manager_2.new_events import (
    CreateContactInput,
    Event,
    GetBusEventsInput,
    GetBusEventsOutput,
    ManagersStartupInput,
    LogMessageInput,
    GetContactsInput,
    LogMessageOutput,
    GetContactsOutput,
    ManagersStartupOutput,
    PublishBusEvent,
)


class ManagersWorker:
    """
    Background worker that subscribes to Redis Pub/Sub events from ConversationManager
    and handles ContactManager, TranscriptManager, and Conductor operations.

    Runs as a separate async task parallel to the conversation manager.
    Uses Pub/Sub with an internal queue to ensure FIFO ordering.
    """

    def __init__(self, event_broker: Optional[redis.Redis] = None):
        self._event_broker = event_broker

        # Pub/Sub channels
        self._subscribe_channel = "app:managers:input"
        self._publish_channel = "app:managers:output"

        # Internal queue for ordered processing
        self._message_queue: asyncio.Queue = asyncio.Queue()

        # Managers (initialized on startup message)
        self._contact_manager: Optional[ContactManager] = None
        self._transcript_manager: Optional[TranscriptManager] = None

        # State flags
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._stop_event = threading.Event()

    # ──────────────────────────────────────────────────────────────────
    # Message handlers
    # ──────────────────────────────────────────────────────────────────

    async def _startup(self, payload: dict) -> None:
        """
        Initialize all managers and configure them.
        This is the first message processed, blocking all subsequent messages.

        Note: Environment variables are already set by ConversationManagerState.set_details()
        when the StartupEvent arrives, so we don't duplicate that logic here.
        """
        print("[ManagersWorker] Processing startup")

        async with self._init_lock:
            if self._initialized:
                print("[ManagersWorker] Already initialized, skipping")
                return

            try:
                # 0. Initialize unity
                print("[ManagersWorker] Initializing unity...")
                if not unity.ASSISTANT:
                    unity.init(
                        assistant_id=int(
                            payload.get("agent_id", "0").replace(
                                "default-assistant-",
                                "",
                            ),
                        ),
                        default_assistant={
                            "user_id": "default-user",
                            "created_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat(),
                            "surname": "",
                            "weekly_limit": None,
                            "max_parallel": None,
                            "profile_photo": None,
                            "country": None,
                            "user_last_name": "",
                            **payload,
                        },
                    )
                print("[ManagersWorker] Unity initialized")

                # Assumes UNIFY_KEY is already in environment from set_details()
                api_key = os.environ.get("UNIFY_KEY")

                # 1. Configure EventBus
                print("[ManagersWorker] Configuring EventBus...")
                if api_key:
                    EVENT_BUS._get_logger().session.headers[
                        "Authorization"
                    ] = f"Bearer {api_key}"
                # event_bus auto-pinning registration
                EVENT_BUS.set_window("Comms", 50)
                EVENT_BUS.register_auto_pin(
                    event_type="Comms",
                    open_predicate=lambda e: e.payload.get("role", "")
                    == "tool_use start",
                    close_predicate=lambda e: e.payload.get("role", "")
                    == "tool_use end",
                    key_fn=lambda e: e.payload.get("handle_id", ""),
                )
                bus_events_task = asyncio.create_task(self._get_bus_events())
                print("[ManagersWorker] EventBus configured")

                # 2. Initialize ContactManager and get contacts
                print("[ManagersWorker] Initializing ContactManager...")
                self._contact_manager = ContactManager()
                contacts_task = asyncio.create_task(self._get_contacts())
                await asyncio.gather(bus_events_task, contacts_task)
                print("[ManagersWorker] ContactManager initialized")

                # 3. Initialize TranscriptManager with ContactManager
                print("[ManagersWorker] Initializing TranscriptManager...")
                self._transcript_manager = TranscriptManager(
                    contact_manager=self._contact_manager,
                )
                print("[ManagersWorker] TranscriptManager initialized")

                # 4. Configure TranscriptManager logger with auth header
                if api_key:
                    self._transcript_manager._get_logger().session.headers[
                        "Authorization"
                    ] = f"Bearer {api_key}"
                    print("[ManagersWorker] TranscriptManager logger configured")

                # TODO: Initialize other managers (Conductor, etc.) here

                self._initialized = True
                print("[ManagersWorker] Initialization complete")

            except Exception as e:
                print(f"[ManagersWorker] Error during initialization: {e}")

            await self._event_broker.publish(
                self._publish_channel,
                ManagersStartupOutput(initialized=self._initialized).to_json(),
            )

    async def _get_bus_events(self) -> None:
        """Get events from EventBus."""
        bus_events = await EVENT_BUS.search(filter='type == "Comms"', limit=50)
        await self._event_broker.publish(
            self._publish_channel,
            GetBusEventsOutput(
                events=[Event.from_bus_event(e).to_dict() for e in bus_events][::-1],
            ).to_json(),
        )

    async def _publish_bus_event(self, event: Event) -> None:
        """Publish an event to the EventBus."""
        while not self._initialized:
            await asyncio.sleep(1)
            print("[ManagersWorker] Not initialized yet, cannot publish bus event")
        event_dict = event.to_dict()["payload"]["event"]
        bus_event = Event.from_dict(event_dict).to_bus_event()
        bus_event.payload.pop("api_key", None)
        bus_event.payload.pop("message_id", None)
        print("Publishing bus event", bus_event)
        await EVENT_BUS.publish(bus_event)

    async def _log_message(self, event: LogMessageInput) -> None:
        """Log a message via TranscriptManager."""
        if not self._transcript_manager:
            print("[ManagersWorker] Not initialized, cannot log message")
            return

        try:
            print(f"[ManagersWorker] Logging message: {event.to_dict()}")
            medium = event.medium or "unify_message"
            sender_id = int(event.sender_id)
            receiver_ids = [int(r) for r in (event.receiver_ids or [])]
            content = event.content
            timestamp = event.timestamp
            exchange_id = event.exchange_id
            call_utterance_timestamp = event.call_utterance_timestamp
            call_url = event.call_url
            metadata = getattr(event, "metadata", None)

            # Log the message
            messages = self._transcript_manager.log_messages(
                {
                    "medium": medium,
                    "sender_id": sender_id,
                    "receiver_ids": receiver_ids,
                    "timestamp": timestamp,
                    "content": content,
                    "exchange_id": exchange_id,
                    "call_utterance_timestamp": call_utterance_timestamp,
                    "call_url": call_url,
                    "_metadata": metadata,
                },
                synchronous=True,
            )

            message = messages[0] if messages else None
            print(
                f"[ManagersWorker] Logged message: {medium}"
                f" from {sender_id} to {receiver_ids}",
            )

            # Publish reply as Event envelope
            if message:
                await self._event_broker.publish(
                    self._publish_channel,
                    LogMessageOutput(
                        medium=medium,
                        exchange_id=message.exchange_id,
                    ).to_json(),
                )
                print(f"[ManagersWorker] Published exchange_id {message.exchange_id}")

        except Exception as e:
            print(f"[ManagersWorker] Error logging message: {e}")

    async def _get_contacts(self) -> None:
        """Fetch all contacts and publish back."""
        if not self._contact_manager:
            print("[ManagersWorker] Not initialized, cannot get contacts")
            return

        try:
            # Get all contacts from ContactManager and convert to dict
            rows = self._contact_manager._filter_contacts()
            contacts = [c.model_dump() for c in rows]

            # Publish reply as Event envelope
            await self._event_broker.publish(
                self._publish_channel,
                GetContactsOutput(contacts=contacts).to_json(),
            )

            print(f"[ManagersWorker] Fetched {len(contacts)} contacts")

            return contacts

        except Exception as e:
            print(f"[ManagersWorker] Error fetching contacts: {e}")

    async def _create_contact(self, contact: dict) -> None:
        """Create a contact in the ContactManager."""
        if not self._contact_manager:
            print("[ManagersWorker] Not initialized, cannot create contact")
            return

        try:
            await self._contact_manager._create_contact(
                first_name=contact["first_name"],
                surname=contact["surname"],
                phone_number=contact["phone_number"],
                email_address=contact["email_address"],
            )
            print(f"[ManagersWorker] Created contact: {contact}")

            # return back the list of updated contacts
            await self._get_contacts()
        except Exception as e:
            print(f"[ManagersWorker] Error creating contact: {e}")

    # ──────────────────────────────────────────────────────────────────
    # Message processing
    # ──────────────────────────────────────────────────────────────────

    async def _process_message(self, event: Event) -> None:
        """Process a single Event from the queue."""
        # Route to handlers using isinstance
        if isinstance(event, ManagersStartupInput):
            asyncio.create_task(self._startup(event.to_dict()["payload"]))
        elif isinstance(event, GetBusEventsInput):
            asyncio.create_task(self._get_bus_events())
        elif isinstance(event, PublishBusEvent):
            asyncio.create_task(self._publish_bus_event(event))
        elif isinstance(event, LogMessageInput):
            asyncio.create_task(self._log_message(event))
        elif isinstance(event, GetContactsInput):
            asyncio.create_task(self._get_contacts())
        elif isinstance(event, CreateContactInput):
            asyncio.create_task(self._create_contact(event.to_dict()["payload"]))
        else:
            print(f"[ManagersWorker] Unknown event: {event.to_dict()['event_name']}")

    async def _queue_processor(self) -> None:
        """Worker task that processes messages from the queue in FIFO order."""
        print("[ManagersWorker] Queue processor started")

        while not self._stop_event.is_set():
            try:
                # Wait for message with timeout to check stop event
                try:
                    msg = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Process message
                await self._process_message(msg)

            except Exception as e:
                print(f"[ManagersWorker] Error in queue processor: {e}")

        print("[ManagersWorker] Queue processor stopped")

    # ──────────────────────────────────────────────────────────────────
    # Main event loop (Pub/Sub listener)
    # ──────────────────────────────────────────────────────────────────

    async def wait_for_events(self) -> None:
        """
        Subscribe to Redis Pub/Sub and enqueue messages.
        A separate task processes the queue to ensure ordering.
        """
        if self._event_broker is None:
            raise RuntimeError(
                "[ManagersWorker] _event_broker must be set before wait_for_events()",
            )
        print("[ManagersWorker] Flag", self._initialized)
        print("[ManagersWorker] Starting to wait for events")
        print(f"[ManagersWorker] Subscribe channel: {self._subscribe_channel}")
        print(f"[ManagersWorker] Publish channel: {self._publish_channel}")

        # Start queue processor task
        processor_task = asyncio.create_task(self._queue_processor())

        try:
            async with self._event_broker.pubsub() as pubsub:
                await pubsub.subscribe(self._subscribe_channel)
                print(f"[ManagersWorker] Subscribed to {self._subscribe_channel}")

                while not self._stop_event.is_set():
                    try:
                        msg = await pubsub.get_message(
                            timeout=2,
                            ignore_subscribe_messages=True,
                        )

                        # Parse Event from JSON envelope and enqueue
                        if msg is not None:
                            try:
                                event = Event.from_json(msg["data"])  # type: ignore[arg-type]
                                print(
                                    f"[ManagersWorker] Enqueued event: {event.to_dict()['event_name']}",
                                )
                                await self._message_queue.put(event)
                            except Exception as parse_err:
                                print(
                                    f"[ManagersWorker] Failed to parse Event from message: {parse_err}",
                                )

                    except Exception as e:
                        print(f"[ManagersWorker] Error receiving message: {e}")
                        await asyncio.sleep(1)

        finally:
            # Stop processor
            self._stop_event.set()
            await self._event_broker.aclose()
            await processor_task
            print("[ManagersWorker] Worker stopped")

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()
