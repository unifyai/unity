from __future__ import annotations

import asyncio
import json
import os
import logging
from typing import Any, Dict, Optional

import redis.asyncio as redis

from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import UNASSIGNED


class ManagersWorker:
    """
    Background worker that subscribes to Redis Pub/Sub events from ConversationManager
    and handles ContactManager, TranscriptManager, and Conductor operations.

    Runs as a separate async task parallel to the conversation manager.
    Uses Pub/Sub with an internal queue to ensure FIFO ordering.
    """

    def __init__(self, event_broker: redis.Redis, job_name: str):
        self.event_broker = event_broker
        self.job_name = job_name

        # Pub/Sub channels
        self.subscribe_channel = f"managers:{job_name}:requests"
        self.reply_channel = f"managers:{job_name}:replies"

        # Setup logger
        self.logger = logging.getLogger("ManagersWorker")

        # Internal queue for ordered processing
        self.message_queue: asyncio.Queue = asyncio.Queue()

        # Managers (initialized on startup message)
        self._contact_manager: Optional[ContactManager] = None
        self._transcript_manager: Optional[TranscriptManager] = None

        # State flags
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    # ──────────────────────────────────────────────────────────────────
    # Message handlers
    # ──────────────────────────────────────────────────────────────────

    async def _startup(self, payload: Dict[str, Any]) -> None:
        """
        Initialize all managers and configure them.
        This is the first message processed, blocking all subsequent messages.

        Note: Environment variables are already set by ConversationManager.set_details()
        when the StartupEvent arrives, so we don't duplicate that logic here.
        """
        self.logger.info("Processing startup")

        async with self._init_lock:
            if self._initialized:
                self.logger.info("Already initialized, skipping")
                return

            try:
                # 1. Initialize ContactManager
                self.logger.info("Initializing ContactManager...")
                self._contact_manager = ContactManager()
                self.logger.info("ContactManager initialized")

                # 2. Initialize TranscriptManager with ContactManager
                self.logger.info("Initializing TranscriptManager...")
                self._transcript_manager = TranscriptManager(
                    contact_manager=self._contact_manager
                )
                self.logger.info("TranscriptManager initialized")

                # 3. Configure TranscriptManager logger with auth header
                # Assumes UNIFY_KEY is already in environment from set_details()
                api_key = os.environ.get("UNIFY_KEY")
                if api_key:
                    self._transcript_manager._get_logger().session.headers[
                        "Authorization"
                    ] = f"Bearer {api_key}"
                    self.logger.info("TranscriptManager logger configured")

                # TODO: Initialize other managers (Conductor, etc.) here

                self._initialized = True
                self.logger.info("Initialization complete")

            except Exception as e:
                self.logger.error(f"Error during initialization: {e}", exc_info=True)
                raise

    async def _log_message(self, payload: Dict[str, Any]) -> None:
        """Log a message via TranscriptManager."""
        if not self._initialized:
            self.logger.warning("Not initialized, cannot log message")
            return

        try:
            medium = payload.get("medium", "unify_chat")
            sender_id = int(payload["sender_id"])
            receiver_ids = [int(r) for r in payload.get("receiver_ids", [])]
            content = payload["content"]
            timestamp = payload["timestamp"]
            exchange_id = payload.get("exchange_id", UNASSIGNED)
            metadata = payload.get("metadata")

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
            self.logger.debug(
                f"Logged message: {medium} from {sender_id} to {receiver_ids}"
            )

            # Publish reply with exchange_id
            if message:
                await self.event_broker.publish(
                    self.reply_channel,
                    json.dumps(
                        {
                            "type": "log_message_reply",
                            "exchange_id": message.exchange_id,
                        }
                    ),
                )
                self.logger.debug(f"Published exchange_id {message.exchange_id}")

        except Exception as e:
            self.logger.error(f"Error logging message: {e}", exc_info=True)

    async def _get_contacts(self, payload: Dict[str, Any]) -> None:
        """Fetch all contacts and publish back."""
        if not self._initialized:
            self.logger.warning("Not initialized, cannot get contacts")
            return

        try:
            # Get all contacts from ContactManager and convert to dict
            rows = self._contact_manager._filter_contacts()
            contacts = [
                {
                    "id": str(c.contact_id),
                    "name": f"{c.first_name or ''} {c.surname or ''}".strip(),
                    "phone": c.phone_number,
                    "email": c.email_address,
                }
                for c in rows
            ]

            # Publish reply
            await self.event_broker.publish(
                self.reply_channel,
                json.dumps(
                    {
                        "type": "get_contacts_reply",
                        "contacts": contacts,
                    }
                ),
            )

            self.logger.debug(f"Fetched {len(contacts)} contacts")

        except Exception as e:
            self.logger.error(f"Error fetching contacts: {e}", exc_info=True)

    # ──────────────────────────────────────────────────────────────────
    # Message processing
    # ──────────────────────────────────────────────────────────────────

    async def _process_message(self, msg_data: Dict[str, Any]) -> None:
        """Process a single message from the queue."""
        message_type = msg_data.get("type")
        payload = msg_data.get("payload", {})

        # Route to handlers
        if message_type == "startup":
            await self._startup(payload)
        elif message_type == "log_message":
            await self._log_message(payload)
        elif message_type == "get_contacts":
            await self._get_contacts(payload)
        else:
            self.logger.warning(f"Unknown message type: {message_type}")

    async def _queue_processor(self) -> None:
        """Worker task that processes messages from the queue in FIFO order."""
        self.logger.info("Queue processor started")

        while not self._stop_event.is_set():
            try:
                # Wait for message with timeout to check stop event
                try:
                    msg = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Process message
                await self._process_message(msg)

            except Exception as e:
                self.logger.error(f"Error in queue processor: {e}", exc_info=True)

        self.logger.info("Queue processor stopped")

    # ──────────────────────────────────────────────────────────────────
    # Main event loop (Pub/Sub listener)
    # ──────────────────────────────────────────────────────────────────

    async def wait_for_events(self) -> None:
        """
        Subscribe to Redis Pub/Sub and enqueue messages.
        A separate task processes the queue to ensure ordering.
        """
        self.logger.info("Starting to wait for events")
        self.logger.info(f"Subscribe channel: {self.subscribe_channel}")
        self.logger.info(f"Reply channel: {self.reply_channel}")

        # Start queue processor task
        processor_task = asyncio.create_task(self._queue_processor())

        try:
            async with self.event_broker.pubsub() as pubsub:
                await pubsub.subscribe(self.subscribe_channel)
                self.logger.info(f"Subscribed to {self.subscribe_channel}")

                while not self._stop_event.is_set():
                    try:
                        msg = await pubsub.get_message(
                            timeout=2, ignore_subscribe_messages=True
                        )

                        if msg and msg["type"] == "message":
                            # Decode and enqueue
                            data = json.loads(msg["data"])
                            await self.message_queue.put(data)
                            self.logger.debug(f"Enqueued message: {data.get('type')}")

                    except Exception as e:
                        self.logger.error(
                            f"Error receiving message: {e}", exc_info=True
                        )
                        await asyncio.sleep(1)

        finally:
            # Stop processor
            self._stop_event.set()
            await processor_task
            self.logger.info("Worker stopped")

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()
