from __future__ import annotations

import asyncio
from datetime import datetime
import os
import logging
from typing import Optional

import redis.asyncio as redis

import unity
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import UNASSIGNED
from unity.conversation_manager_2.new_events import (
    Event,
    StartupEvent,
    LogMessageInput,
    GetContactsInput,
    LogMessageOutput,
    GetContactsOutput,
)


class ManagersWorker:
    """
    Background worker that subscribes to Redis Pub/Sub events from ConversationManager
    and handles ContactManager, TranscriptManager, and Conductor operations.

    Runs as a separate async task parallel to the conversation manager.
    Uses Pub/Sub with an internal queue to ensure FIFO ordering.
    """

    def __init__(self, event_broker: redis.Redis):
        self._event_broker = event_broker

        # Pub/Sub channels
        self._subscribe_channel = "app:managers:input"
        self._publish_channel = "app:managers:output"

        # Setup logger
        self._logger = logging.getLogger("ManagersWorker")

        # Internal queue for ordered processing
        self._message_queue: asyncio.Queue = asyncio.Queue()

        # Managers (initialized on startup message)
        self._contact_manager: Optional[ContactManager] = None
        self._transcript_manager: Optional[TranscriptManager] = None

        # State flags
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    async def _initialize_unity(self) -> None:
        assistant_id = os.environ.get("ASSISTANT_ID", "0")
        unity.init(
            assistant_id=int(
                assistant_id.replace("default-assistant-", ""),
            ),
            default_assistant=dict(
                user_id="default-user",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                agent_id=assistant_id,
                first_name=os.environ.get("ASSISTANT_NAME", ""),
                surname="",
                age=os.environ.get("ASSISTANT_AGE", ""),
                region=os.environ.get("ASSISTANT_REGION", ""),
                about=os.environ.get("ASSISTANT_ABOUT", ""),
                phone=os.environ.get("ASSISTANT_NUMBER", ""),
                email=os.environ.get("ASSISTANT_EMAIL", ""),
                user_phone=os.environ.get("USER_NUMBER", ""),
                user_whatsapp_number=os.environ.get("USER_WHATSAPP_NUMBER", ""),
                assistant_whatsapp_number=os.environ.get("ASSISTANT_NUMBER", ""),
                api_key=os.environ.get("UNIFY_KEY"),
                weekly_limit=None,
                max_parallel=None,
                profile_photo=None,
                country=None,
                voice_id=None,
                voice_provider="cartesia",
                user_last_name="",
            ),
        )

    # ──────────────────────────────────────────────────────────────────
    # Message handlers
    # ──────────────────────────────────────────────────────────────────

    async def _startup(self) -> None:
        """
        Initialize all managers and configure them.
        This is the first message processed, blocking all subsequent messages.

        Note: Environment variables are already set by ConversationManager.set_details()
        when the StartupEvent arrives, so we don't duplicate that logic here.
        """
        self._logger.info("Processing startup")

        async with self._init_lock:
            if self._initialized:
                self._logger.info("Already initialized, skipping")
                return

            try:
                # 0. Initialize unity
                self._logger.info("Initializing unity...")
                if not unity.ASSISTANT:
                    await self._initialize_unity()
                self._logger.info("Unity initialized")

                # 1. Initialize ContactManager
                self._logger.info("Initializing ContactManager...")
                self._contact_manager = ContactManager()
                self._logger.info("ContactManager initialized")

                # 2. Initialize TranscriptManager with ContactManager
                self._logger.info("Initializing TranscriptManager...")
                self._transcript_manager = TranscriptManager(
                    contact_manager=self._contact_manager
                )
                self._logger.info("TranscriptManager initialized")

                # 3. Configure TranscriptManager logger with auth header
                # Assumes UNIFY_KEY is already in environment from set_details()
                api_key = os.environ.get("UNIFY_KEY")
                if api_key:
                    self._transcript_manager._get_logger().session.headers[
                        "Authorization"
                    ] = f"Bearer {api_key}"
                    self._logger.info("TranscriptManager logger configured")

                # TODO: Initialize other managers (Conductor, etc.) here

                self._initialized = True
                self._logger.info("Initialization complete")

            except Exception as e:
                self._logger.error(f"Error during initialization: {e}", exc_info=True)

    async def _log_message(self, evt: LogMessageInput) -> None:
        """Log a message via TranscriptManager."""
        if not self._initialized:
            self._logger.warning("Not initialized, cannot log message")
            await self._startup()

        try:
            medium = evt.medium or "unify_chat"
            sender_id = int(evt.sender_id)
            receiver_ids = [int(r) for r in (evt.receiver_ids or [])]
            content = evt.content
            timestamp = evt.timestamp
            exchange_id = getattr(evt, "exchange_id", UNASSIGNED)
            metadata = getattr(evt, "metadata", None)

            # Log the message
            messages = self._transcript_manager.log_messages(
                {
                    "medium": medium,
                    "sender_id": sender_id,
                    "receiver_ids": receiver_ids,
                    "timestamp": timestamp,
                    "content": content,
                    "exchange_id": exchange_id,
                    "_metadata": metadata,
                },
                synchronous=True,
            )

            message = messages[0] if messages else None
            self._logger.debug(
                f"Logged message: {medium} from {sender_id} to {receiver_ids}"
            )

            # Publish reply as Event envelope
            if message:
                await self._event_broker.publish(
                    self._publish_channel,
                    LogMessageOutput(exchange_id=message.exchange_id).to_json(),
                )
                self._logger.debug(f"Published exchange_id {message.exchange_id}")

        except Exception as e:
            self._logger.error(f"Error logging message: {e}", exc_info=True)

    async def _get_contacts(self) -> None:
        """Fetch all contacts and publish back."""
        if not self._initialized:
            self._logger.warning("Not initialized, cannot get contacts")
            await self._startup()

        try:
            # Get all contacts from ContactManager and convert to dict
            rows = self._contact_manager._filter_contacts()
            contacts = [
                {
                    "id": str(c.contact_id),
                    "name": f"{c.first_name or ''} {c.surname or ''}".strip(),
                    "number": c.phone_number,
                    "email": c.email_address,
                }
                for c in rows
            ]

            # Publish reply as Event envelope
            await self._event_broker.publish(
                self._publish_channel,
                GetContactsOutput(contacts=contacts).to_json(),
            )

            self._logger.debug(f"Fetched {len(contacts)} contacts")

        except Exception as e:
            self._logger.error(f"Error fetching contacts: {e}", exc_info=True)

    # ──────────────────────────────────────────────────────────────────
    # Message processing
    # ──────────────────────────────────────────────────────────────────

    async def _process_message(self, event: Event) -> None:
        """Process a single Event from the queue."""
        # Route to handlers using isinstance
        if isinstance(event, StartupEvent):
            await self._startup()
        elif isinstance(event, LogMessageInput):
            await self._log_message(event)
        elif isinstance(event, GetContactsInput):
            await self._get_contacts()
        else:
            self._logger.warning(f"Unknown event: {event.to_dict()['event_name']}")

    async def _queue_processor(self) -> None:
        """Worker task that processes messages from the queue in FIFO order."""
        self._logger.info("Queue processor started")

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
                self._logger.error(f"Error in queue processor: {e}", exc_info=True)

        self._logger.info("Queue processor stopped")

    # ──────────────────────────────────────────────────────────────────
    # Main event loop (Pub/Sub listener)
    # ──────────────────────────────────────────────────────────────────

    async def wait_for_events(self) -> None:
        """
        Subscribe to Redis Pub/Sub and enqueue messages.
        A separate task processes the queue to ensure ordering.
        """
        self._logger.info("Starting to wait for events")
        self._logger.info(f"Subscribe channel: {self._subscribe_channel}")
        self._logger.info(f"Publish channel: {self._publish_channel}")

        # Start queue processor task
        processor_task = asyncio.create_task(self._queue_processor())

        try:
            async with self._event_broker.pubsub() as pubsub:
                await pubsub.subscribe(self._subscribe_channel)
                self._logger.info(f"Subscribed to {self._subscribe_channel}")

                while not self._stop_event.is_set():
                    try:
                        msg = await pubsub.get_message(
                            timeout=2, ignore_subscribe_messages=True
                        )

                        # Parse Event from JSON envelope and enqueue
                        if msg is not None:
                            try:
                                event = Event.from_json(msg["data"])  # type: ignore[arg-type]
                                await self._message_queue.put(event)
                                self._logger.debug(
                                    f"Enqueued event: {event.to_dict()['event_name']}"
                                )
                            except Exception:
                                self._logger.error(
                                    "Failed to parse Event from message", exc_info=True
                                )

                    except Exception as e:
                        self._logger.error(
                            f"Error receiving message: {e}", exc_info=True
                        )
                        await asyncio.sleep(1)

        finally:
            # Stop processor
            self._stop_event.set()
            await processor_task
            self._logger.info("Worker stopped")

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()
