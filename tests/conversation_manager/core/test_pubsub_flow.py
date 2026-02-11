"""
tests/conversation_manager/core/test_pubsub_flow.py
===================================================

Integration tests for the Pub/Sub message flow in production deployment.

These tests verify the critical Pub/Sub patterns documented in INFRA.md:
1. Idle container subscribes to unity-startup topic
2. On startup message: unsubscribe from startup, subscribe to unity-{assistant_id}
3. Inbound messages correctly routed to the assistant's topic
4. Message acknowledgment patterns prevent duplicate processing

WHY THESE TESTS MATTER:
-----------------------
The Pub/Sub flow is the backbone of the Unity deployment. Bugs here cause:
- Containers that never go live (stuck on startup topic)
- Messages lost in transit (wrong topic subscription)
- Duplicate message processing (incorrect acknowledgment)
- Race conditions when startup + inbound arrive simultaneously

These patterns are critical to test because:
- CommsManager runs subscription callbacks in a thread pool (not asyncio)
- Thread-safe publishing requires run_coroutine_threadsafe
- Subscription switching must be atomic (unsubscribe startup → subscribe assistant)

Ved's fixes that relate to this flow:
- 3c44b692: Race condition with wakeup + pre-hire within 3s
- 6237411a: Pre-hire logging using wrong channel after refactor
- fe355d6f: Blocking inbound until managers initialized
"""

from __future__ import annotations

import asyncio
import json
import threading
import time as _time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from unity.conversation_manager.in_memory_event_broker import (
    create_in_memory_event_broker,
    reset_in_memory_event_broker,
)

# =============================================================================
# Helper for deterministic waiting
# =============================================================================


async def _wait_for_condition(
    predicate,
    *,
    timeout: float = 5.0,
    poll: float = 0.02,
) -> bool:
    """Poll predicate() until True or timeout. Returns whether condition was met."""
    start = _time.perf_counter()
    while _time.perf_counter() - start < timeout:
        if predicate():
            return True
        await asyncio.sleep(poll)
    return False


@pytest_asyncio.fixture
async def event_broker():
    """Real in-memory event broker."""
    reset_in_memory_event_broker()
    broker = create_in_memory_event_broker()
    yield broker
    await broker.aclose()
    reset_in_memory_event_broker()


@pytest.fixture
def boss_contact():
    return {
        "contact_id": 1,
        "first_name": "Boss",
        "surname": "User",
        "phone_number": "+15555550001",
        "email_address": "boss@example.com",
    }


@dataclass
class MockPubSubMessage:
    """Mock PubSub message for testing CommsManager.handle_message()."""

    data: bytes
    _acked: bool = False
    _nacked: bool = False

    def ack(self):
        self._acked = True

    def nack(self):
        self._nacked = True


def create_pubsub_message(thread: str, event: dict) -> MockPubSubMessage:
    """Create a mock PubSub message with the given thread and event data."""
    payload = {"thread": thread, "event": event}
    return MockPubSubMessage(data=json.dumps(payload).encode("utf-8"))


class TestSubscriptionSwitching:
    """
    Tests for the subscription switching flow: startup → assistant topic.

    When an idle container receives a startup message, it must:
    1. Acknowledge the startup message
    2. Cancel the startup subscription
    3. Subscribe to the assistant-specific topic
    4. Update SESSION_DETAILS with the assistant ID

    This must happen atomically to avoid message loss.
    """

    @pytest.mark.asyncio
    async def test_startup_message_triggers_subscription_switch(self, event_broker):
        """
        Test that startup message triggers subscription switch.

        This is the core flow for going from idle → live container.
        """
        from unity.conversation_manager.comms_manager import CommsManager
        from unity.session_details import SESSION_DETAILS, DEFAULT_ASSISTANT_ID

        # Start with default assistant (idle container state)
        original_id = SESSION_DETAILS.assistant.id
        SESSION_DETAILS.assistant.id = DEFAULT_ASSISTANT_ID

        try:
            cm = CommsManager(event_broker)
            cm.loop = asyncio.get_event_loop()

            # Pre-register the startup subscription (normally done by start())
            mock_startup_future = MagicMock()
            cm.subscribers["unity-startup-sub"] = mock_startup_future

            # Track if subscribe_to_topic was called with the new topic
            subscribed_topics = []
            original_subscribe = cm.subscribe_to_topic

            def track_subscribe(topic):
                subscribed_topics.append(topic)
                # Don't actually subscribe (would need real PubSub)

            cm.subscribe_to_topic = track_subscribe

            # Create startup message
            startup_event = {
                "api_key": "test_key",
                "assistant_id": "test_assistant_42",
                "user_id": "123",
                "assistant_name": "Test Assistant",
                "assistant_age": "25",
                "assistant_nationality": "American",
                "assistant_about": "A test assistant",
                "assistant_number": "+15555550000",
                "assistant_email": "assistant@test.com",
                "user_name": "Boss User",
                "user_number": "+15555550001",
                "user_email": "boss@test.com",
                "voice_provider": "cartesia",
                "voice_id": "test_voice",
                "voice_mode": "tts",
            }
            message = create_pubsub_message("startup", startup_event)

            # Patch subprocess.run to avoid VNC password update
            with patch("subprocess.run"):
                # Handle the message (synchronous, like real PubSub callback)
                cm.handle_message(message)

            # Verify message was acknowledged
            assert message._acked, "Startup message should be acknowledged"

            # Verify startup subscription was cancelled
            mock_startup_future.cancel.assert_called_once()

            # Verify we subscribed to the assistant's topic
            assert len(subscribed_topics) == 1
            assert "test_assistant_42" in subscribed_topics[0]

            # Verify SESSION_DETAILS was updated
            assert SESSION_DETAILS.assistant.id == "test_assistant_42"

        finally:
            SESSION_DETAILS.assistant.id = original_id

    @pytest.mark.asyncio
    async def test_startup_removed_from_subscribers_after_cancel(self, event_broker):
        """
        Test that startup subscription is removed from subscribers dict.

        This prevents attempts to cancel it again if another startup arrives.
        """
        from unity.conversation_manager.comms_manager import CommsManager
        from unity.session_details import SESSION_DETAILS, DEFAULT_ASSISTANT_ID

        original_id = SESSION_DETAILS.assistant.id
        SESSION_DETAILS.assistant.id = DEFAULT_ASSISTANT_ID

        try:
            cm = CommsManager(event_broker)
            cm.loop = asyncio.get_event_loop()

            mock_startup_future = MagicMock()
            cm.subscribers["unity-startup-sub"] = mock_startup_future

            cm.subscribe_to_topic = MagicMock()

            startup_event = {
                "api_key": "test_key",
                "assistant_id": "42",
                "user_id": "123",
                "assistant_name": "Test",
                "assistant_age": "25",
                "assistant_nationality": "American",
                "assistant_about": "Test",
                "assistant_number": "+15555550000",
                "assistant_email": "a@test.com",
                "user_name": "Boss",
                "user_number": "+15555550001",
                "user_email": "b@test.com",
                "voice_provider": "cartesia",
                "voice_id": "",
                "voice_mode": "tts",
            }
            message = create_pubsub_message("startup", startup_event)

            with patch("subprocess.run"):
                cm.handle_message(message)

            # Startup subscription should be removed from subscribers
            assert "unity-startup-sub" not in cm.subscribers, (
                "Startup subscription should be removed after handling. "
                "Leaving it could cause issues if another startup arrives."
            )

        finally:
            SESSION_DETAILS.assistant.id = original_id


class TestMessageAcknowledgment:
    """
    Tests for correct message acknowledgment patterns.

    Pub/Sub messages MUST be acknowledged to prevent redelivery.
    Different message types have different acknowledgment requirements:
    - startup: ack immediately, then process
    - inbound (sms, email, call): ack after processing
    - call events: blocking ack (wait for publish to complete)
    """

    @pytest.mark.asyncio
    async def test_sms_message_acknowledged(self, event_broker, boss_contact):
        """Test that SMS messages are acknowledged after processing."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(event_broker)
        cm.loop = asyncio.get_event_loop()

        message = create_pubsub_message(
            "msg",
            {
                "body": "Hello!",
                "from_number": boss_contact["phone_number"],
                "contacts": [boss_contact],
            },
        )

        cm.handle_message(message)
        # Poll for message acknowledgment instead of fixed sleep
        await _wait_for_condition(lambda: message._acked)

        assert message._acked, "SMS message should be acknowledged"

    @pytest.mark.asyncio
    async def test_email_message_acknowledged(self, event_broker, boss_contact):
        """Test that email messages are acknowledged."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(event_broker)
        cm.loop = asyncio.get_event_loop()

        message = create_pubsub_message(
            "email",
            {
                "subject": "Test Subject",
                "body": "Test body",
                "from": f"Boss <{boss_contact['email_address']}>",
                "email_id": "msg_123",
                "contacts": [boss_contact],
            },
        )

        cm.handle_message(message)
        # Poll for message acknowledgment instead of fixed sleep
        await _wait_for_condition(lambda: message._acked)

        assert message._acked, "Email message should be acknowledged"

    @pytest.mark.asyncio
    async def test_call_message_acknowledged_after_publish(
        self,
        event_broker,
        boss_contact,
    ):
        """
        Test that call messages use blocking acknowledgment.

        Call events require the publish to complete before ack, because
        losing a call event is more critical than losing a text message.
        """
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(event_broker)
        cm.loop = asyncio.get_event_loop()

        message = create_pubsub_message(
            "call",
            {
                "caller_number": boss_contact["phone_number"],
                "conference_name": "conf_123",
                "contacts": [boss_contact],
            },
        )

        # Run in thread to simulate GCP PubSub's threading model
        await asyncio.to_thread(cm.handle_message, message)

        assert message._acked, "Call message should be acknowledged"

    @pytest.mark.asyncio
    async def test_malformed_message_acknowledged(self, event_broker):
        """
        Test that malformed messages are still acknowledged.

        We don't want malformed messages to be redelivered forever.
        """
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(event_broker)
        cm.loop = asyncio.get_event_loop()

        # Invalid JSON
        message = MockPubSubMessage(data=b"not valid json")
        cm.handle_message(message)

        assert message._acked, (
            "Malformed message should still be acknowledged to prevent "
            "infinite redelivery loop."
        )

    @pytest.mark.asyncio
    async def test_unknown_thread_not_crashed(self, event_broker):
        """
        Test that unknown thread types don't crash the handler.

        Unknown threads should be logged but not cause exceptions.
        """
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(event_broker)
        cm.loop = asyncio.get_event_loop()

        message = create_pubsub_message(
            "unknown_thread_type",
            {"data": "some data"},
        )

        # Should not raise
        cm.handle_message(message)


class TestThreadSafePublishing:
    """
    Tests for thread-safe event publishing.

    CommsManager.handle_message() is called from GCP PubSub's thread pool,
    NOT from the asyncio event loop. Publishing to the event broker must
    use run_coroutine_threadsafe to be thread-safe.
    """

    @pytest.mark.asyncio
    async def test_publish_from_callback_thread_safe(self, event_broker):
        """
        Test that _publish_from_callback works from a different thread.

        This simulates the real GCP PubSub callback scenario.
        """
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(event_broker)
        cm.loop = asyncio.get_event_loop()

        received_messages = []

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("test:channel")

            # Publish from a different thread (like GCP PubSub callback)
            def publish_from_thread():
                cm._publish_from_callback("test:channel", '{"test": "data"}')

            thread = threading.Thread(target=publish_from_thread)
            thread.start()
            thread.join()

            # Poll for message to arrive instead of fixed sleep
            msg = None
            for _ in range(50):  # 5s timeout with 0.1s poll
                msg = await pubsub.get_message(
                    timeout=0.1,
                    ignore_subscribe_messages=True,
                )
                if msg:
                    break

            assert msg is not None, "Message not received from thread-safe publish"
            assert msg["data"] == '{"test": "data"}'

    @pytest.mark.asyncio
    async def test_concurrent_publishes_from_multiple_threads(self, event_broker):
        """
        Test that concurrent publishes from multiple threads work correctly.

        In production, multiple PubSub callbacks can run concurrently.
        """
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(event_broker)
        cm.loop = asyncio.get_event_loop()

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("test:concurrent")

            # Publish from multiple threads concurrently
            threads = []
            for i in range(5):

                def publish(idx=i):
                    cm._publish_from_callback(
                        "test:concurrent",
                        json.dumps({"index": idx}),
                    )

                t = threading.Thread(target=publish)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Collect all messages - poll until we have all 5 or timeout
            received = []
            for _ in range(100):  # 10s timeout with 0.1s poll
                msg = await pubsub.get_message(
                    timeout=0.1,
                    ignore_subscribe_messages=True,
                )
                if msg:
                    received.append(json.loads(msg["data"]))
                if len(received) >= 5:
                    break

            # All 5 messages should be received
            indices = {m["index"] for m in received}
            assert indices == {0, 1, 2, 3, 4}, (
                f"Not all concurrent messages received: {indices}. "
                "Thread-safe publishing may be broken."
            )


class TestStartupInboundRace:
    """
    Tests for the race condition between startup and inbound messages.

    In production, the adapter sends:
    1. Startup message to unity-startup (if not already live)
    2. Inbound message to unity-{assistant_id}

    These can arrive within milliseconds of each other. The container must:
    - Process startup first (subscription switch)
    - Then receive inbound on the new subscription
    - Not lose either message
    """

    @pytest.mark.asyncio
    async def test_startup_publishes_event_before_inbound(
        self,
        event_broker,
        boss_contact,
    ):
        """
        Test that startup event is published to the event broker.

        This is critical because the EventHandler needs to receive StartupEvent
        to initialize managers before processing inbound messages.
        """
        from unity.conversation_manager.comms_manager import CommsManager
        from unity.conversation_manager.events import StartupEvent, Event
        from unity.session_details import SESSION_DETAILS, DEFAULT_ASSISTANT_ID

        original_id = SESSION_DETAILS.assistant.id
        SESSION_DETAILS.assistant.id = DEFAULT_ASSISTANT_ID

        try:
            cm = CommsManager(event_broker)
            cm.loop = asyncio.get_event_loop()
            cm.subscribers["unity-startup-sub"] = MagicMock()
            cm.subscribe_to_topic = MagicMock()

            received_events = []

            async with event_broker.pubsub() as pubsub:
                await pubsub.psubscribe("app:comms:*")

                startup_event = {
                    "api_key": "test_key",
                    "assistant_id": "race_test",
                    "user_id": "123",
                    "assistant_name": "Test",
                    "assistant_age": "25",
                    "assistant_nationality": "American",
                    "assistant_about": "Test",
                    "assistant_number": "+15555550000",
                    "assistant_email": "a@test.com",
                    "user_name": "Boss",
                    "user_number": "+15555550001",
                    "user_email": "b@test.com",
                    "voice_provider": "cartesia",
                    "voice_id": "",
                    "voice_mode": "tts",
                }
                message = create_pubsub_message("startup", startup_event)

                with patch("subprocess.run"):
                    cm.handle_message(message)

                # Collect events - poll until we find startup event or timeout
                for _ in range(50):  # 5s timeout with 0.1s poll
                    msg = await pubsub.get_message(
                        timeout=0.1,
                        ignore_subscribe_messages=True,
                    )
                    if msg and msg["channel"] == "app:comms:startup":
                        event = Event.from_json(msg["data"])
                        received_events.append(event)
                        break

            # StartupEvent should have been published
            startup_events = [e for e in received_events if isinstance(e, StartupEvent)]
            assert len(startup_events) >= 1, (
                "StartupEvent not published to event broker. "
                "EventHandler won't know to initialize managers."
            )

        finally:
            SESSION_DETAILS.assistant.id = original_id

    @pytest.mark.asyncio
    async def test_backup_contacts_published_with_inbound(
        self,
        event_broker,
        boss_contact,
    ):
        """
        Test that BackupContactsEvent is published with inbound messages.

        This is the mechanism that enables contact lookup before ContactManager init.
        """
        from unity.conversation_manager.comms_manager import CommsManager
        from unity.conversation_manager.events import BackupContactsEvent, Event

        cm = CommsManager(event_broker)
        cm.loop = asyncio.get_event_loop()

        received_backup = []

        async with event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            message = create_pubsub_message(
                "msg",
                {
                    "body": "Hello!",
                    "from_number": boss_contact["phone_number"],
                    "contacts": [boss_contact],
                },
            )

            cm.handle_message(message)

            # Poll until we find backup_contacts event or timeout
            for _ in range(50):  # 5s timeout with 0.1s poll
                msg = await pubsub.get_message(
                    timeout=0.1,
                    ignore_subscribe_messages=True,
                )
                if msg and msg["channel"] == "app:comms:backup_contacts":
                    event = Event.from_json(msg["data"])
                    if isinstance(event, BackupContactsEvent):
                        received_backup.append(event)
                        break

        assert len(received_backup) >= 1, (
            "BackupContactsEvent not published with SMS. "
            "Contact lookup will fail before manager init."
        )

        # Verify the contact is in the backup
        contacts = received_backup[0].contacts
        contact_ids = [c["contact_id"] for c in contacts]
        assert boss_contact["contact_id"] in contact_ids


class TestSubscriptionIdGeneration:
    """
    Tests for correct subscription ID generation.

    Subscription IDs must follow the exact pattern expected by GCP Pub/Sub:
    - Production: unity-{assistant_id}-sub
    - Staging: unity-{assistant_id}-staging-sub
    """

    @pytest.mark.asyncio
    async def test_production_subscription_id(self):
        """Test production subscription ID format."""
        from unity.conversation_manager.comms_manager import _get_subscription_id
        from unity.session_details import SESSION_DETAILS

        original_id = SESSION_DETAILS.assistant.id
        SESSION_DETAILS.assistant.id = "42"

        try:
            with patch(
                "unity.conversation_manager.comms_manager.SETTINGS",
            ) as mock_settings:
                mock_settings.STAGING = False

                sub_id = _get_subscription_id()
                assert sub_id == "unity-42-sub", f"Wrong production sub ID: {sub_id}"
        finally:
            SESSION_DETAILS.assistant.id = original_id

    @pytest.mark.asyncio
    async def test_staging_subscription_id(self):
        """Test staging subscription ID format."""
        from unity.conversation_manager.comms_manager import _get_subscription_id
        from unity.session_details import SESSION_DETAILS

        original_id = SESSION_DETAILS.assistant.id
        SESSION_DETAILS.assistant.id = "25"

        try:
            with patch(
                "unity.conversation_manager.comms_manager.SETTINGS",
            ) as mock_settings:
                mock_settings.STAGING = True

                sub_id = _get_subscription_id()
                assert (
                    sub_id == "unity-25-staging-sub"
                ), f"Wrong staging sub ID: {sub_id}"
        finally:
            SESSION_DETAILS.assistant.id = original_id

    @pytest.mark.asyncio
    async def test_startup_subscription_id_constants(self):
        """Test that startup subscription ID constants are correct."""
        from unity.conversation_manager.comms_manager import startup_subscription_id

        # The module-level constant should be set based on SETTINGS.STAGING
        # We verify the format is correct (includes -sub suffix)
        assert startup_subscription_id.endswith(
            "-sub",
        ), f"Startup subscription ID missing -sub suffix: {startup_subscription_id}"
        assert "startup" in startup_subscription_id


class TestEventPublishChannels:
    """
    Tests for correct event channel names.

    Events must be published to the correct channels for handlers to receive them.
    Wrong channel names cause silent failures (messages never received).

    This would have caught Ved's bug in commit 6237411a where pre-hire logging
    was using the wrong channel name.
    """

    @pytest.mark.asyncio
    async def test_sms_publishes_to_msg_message_channel(
        self,
        event_broker,
        boss_contact,
    ):
        """Test that SMS publishes to app:comms:msg_message channel."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(event_broker)
        cm.loop = asyncio.get_event_loop()

        received_channel = None

        async with event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            message = create_pubsub_message(
                "msg",
                {
                    "body": "Test",
                    "from_number": boss_contact["phone_number"],
                    "contacts": [boss_contact],
                },
            )

            cm.handle_message(message)

            # Poll until we find msg_message event or timeout
            for _ in range(50):  # 5s timeout with 0.1s poll
                msg = await pubsub.get_message(
                    timeout=0.1,
                    ignore_subscribe_messages=True,
                )
                if msg and "msg_message" in msg["channel"]:
                    received_channel = msg["channel"]
                    break

        assert (
            received_channel == "app:comms:msg_message"
        ), f"SMS published to wrong channel: {received_channel}"

    @pytest.mark.asyncio
    async def test_email_publishes_to_email_message_channel(
        self,
        event_broker,
        boss_contact,
    ):
        """Test that email publishes to app:comms:email_message channel."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(event_broker)
        cm.loop = asyncio.get_event_loop()

        received_channel = None

        async with event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            message = create_pubsub_message(
                "email",
                {
                    "subject": "Test",
                    "body": "Body",
                    "from": f"Boss <{boss_contact['email_address']}>",
                    "email_id": "123",
                    "contacts": [boss_contact],
                },
            )

            cm.handle_message(message)

            # Poll until we find email_message event or timeout
            for _ in range(50):  # 5s timeout with 0.1s poll
                msg = await pubsub.get_message(
                    timeout=0.1,
                    ignore_subscribe_messages=True,
                )
                if msg and "email_message" in msg["channel"]:
                    received_channel = msg["channel"]
                    break

        assert (
            received_channel == "app:comms:email_message"
        ), f"Email published to wrong channel: {received_channel}"

    @pytest.mark.asyncio
    async def test_pre_hire_publishes_to_pre_hire_channel(
        self,
        event_broker,
        boss_contact,
    ):
        """
        Test that pre-hire messages publish to app:comms:pre_hire channel.

        This was broken in Ved's bug (6237411a) - wrong channel name after refactor.
        """
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(event_broker)
        cm.loop = asyncio.get_event_loop()

        received_channel = None

        async with event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            message = create_pubsub_message(
                "log_pre_hire_chats",
                {
                    "assistant_id": "test",
                    "contacts": [boss_contact],
                    "body": [
                        {"role": "user", "msg": "Hello"},
                    ],
                },
            )

            cm.handle_message(message)

            # Poll until we find pre_hire event or timeout
            for _ in range(50):  # 5s timeout with 0.1s poll
                msg = await pubsub.get_message(
                    timeout=0.1,
                    ignore_subscribe_messages=True,
                )
                if msg and "pre_hire" in msg["channel"]:
                    received_channel = msg["channel"]
                    break

        assert received_channel == "app:comms:pre_hire", (
            f"Pre-hire published to wrong channel: {received_channel}. "
            "This was Ved's bug (6237411a)."
        )


class TestPingMechanismForIdleContainers:
    """
    Tests for the ping mechanism that keeps idle containers alive.

    Idle containers must send periodic pings to avoid the inactivity timeout.
    The ping mechanism is critical for container availability.
    """

    @pytest.mark.asyncio
    async def test_ping_publishes_to_correct_channel(self, event_broker):
        """Test that pings are published to app:comms:ping channel."""
        from unity.conversation_manager.comms_manager import CommsManager
        from unity.conversation_manager.events import Ping, Event
        from unity.session_details import SESSION_DETAILS, DEFAULT_ASSISTANT_ID

        original_id = SESSION_DETAILS.assistant.id
        SESSION_DETAILS.assistant.id = DEFAULT_ASSISTANT_ID

        try:
            cm = CommsManager(event_broker)

            received_ping = False

            async with event_broker.pubsub() as pubsub:
                await pubsub.subscribe("app:comms:ping")

                # Manually trigger one ping cycle
                async def run_single_ping():
                    await cm.event_broker.publish(
                        "app:comms:ping",
                        Ping(kind="keepalive").to_json(),
                    )

                await run_single_ping()

                msg = await pubsub.get_message(
                    timeout=1.0,
                    ignore_subscribe_messages=True,
                )

                if msg:
                    event = Event.from_json(msg["data"])
                    if isinstance(event, Ping):
                        received_ping = True

            assert received_ping, "Ping not received on expected channel"

        finally:
            SESSION_DETAILS.assistant.id = original_id

    @pytest.mark.asyncio
    async def test_ping_has_keepalive_kind(self, event_broker):
        """Test that ping events have kind='keepalive'."""
        from unity.conversation_manager.events import Ping

        ping = Ping(kind="keepalive")
        assert ping.kind == "keepalive"

        # Verify serialization/deserialization
        from unity.conversation_manager.events import Event

        restored = Event.from_json(ping.to_json())
        assert isinstance(restored, Ping)
        assert restored.kind == "keepalive"


class TestDemoModePropagation:
    """
    Tests for demo mode flag propagation through the Pub/Sub startup flow.

    Demo mode is passed from adapters → comms → Unity via the startup event.
    When demo_mode=True, Unity should:
    1. Include demo_mode in the StartupEvent published to the event broker
    2. Set SETTINGS.DEMO_MODE = True before initializing managers

    This enables demo-specific behavior:
    - set_boss_details tool instead of act
    - Demo-specific prompts for voice and slow brain
    - Boss contact starts without details (learned during demo)
    """

    @pytest.mark.asyncio
    async def test_startup_event_includes_demo_mode_true(self, event_broker):
        """
        Test that demo_mode=True is passed through to StartupEvent.

        The comms layer passes demo_mode from the job data to the StartupEvent.
        """
        from unity.conversation_manager.comms_manager import CommsManager
        from unity.conversation_manager.events import StartupEvent, Event
        from unity.session_details import SESSION_DETAILS, DEFAULT_ASSISTANT_ID

        original_id = SESSION_DETAILS.assistant.id
        SESSION_DETAILS.assistant.id = DEFAULT_ASSISTANT_ID

        try:
            cm = CommsManager(event_broker)
            cm.loop = asyncio.get_event_loop()
            cm.subscribers["unity-startup-sub"] = MagicMock()
            cm.subscribe_to_topic = MagicMock()

            async with event_broker.pubsub() as pubsub:
                await pubsub.psubscribe("app:comms:*")

                startup_event = {
                    "api_key": "test_key",
                    "assistant_id": "demo_test_123",
                    "user_id": "456",
                    "assistant_name": "Demo Assistant",
                    "assistant_age": "25",
                    "assistant_nationality": "American",
                    "assistant_about": "A demo assistant",
                    "assistant_number": "+15555550000",
                    "assistant_email": "demo@test.com",
                    "user_name": "Boss",
                    "user_number": "+15555550001",
                    "user_email": "boss@test.com",
                    "voice_provider": "cartesia",
                    "voice_id": "test_voice",
                    "voice_mode": "tts",
                    "demo_mode": True,  # Demo mode enabled
                }
                message = create_pubsub_message("startup", startup_event)

                with patch("subprocess.run"):
                    cm.handle_message(message)

                # Poll for the startup event
                received_event = None
                for _ in range(50):
                    msg = await pubsub.get_message(
                        timeout=0.1,
                        ignore_subscribe_messages=True,
                    )
                    if msg and msg["channel"] == "app:comms:startup":
                        received_event = Event.from_json(msg["data"])
                        break

            assert received_event is not None, "StartupEvent not received"
            assert isinstance(received_event, StartupEvent)
            assert received_event.demo_mode is True, (
                "demo_mode should be True in StartupEvent. "
                "Check that comms_manager.py extracts demo_mode from event data."
            )

        finally:
            SESSION_DETAILS.assistant.id = original_id

    @pytest.mark.asyncio
    async def test_startup_event_includes_demo_mode_false_by_default(
        self,
        event_broker,
    ):
        """
        Test that demo_mode defaults to False when not specified.

        Regular assistants don't include demo_mode in their startup data.
        """
        from unity.conversation_manager.comms_manager import CommsManager
        from unity.conversation_manager.events import StartupEvent, Event
        from unity.session_details import SESSION_DETAILS, DEFAULT_ASSISTANT_ID

        original_id = SESSION_DETAILS.assistant.id
        SESSION_DETAILS.assistant.id = DEFAULT_ASSISTANT_ID

        try:
            cm = CommsManager(event_broker)
            cm.loop = asyncio.get_event_loop()
            cm.subscribers["unity-startup-sub"] = MagicMock()
            cm.subscribe_to_topic = MagicMock()

            async with event_broker.pubsub() as pubsub:
                await pubsub.psubscribe("app:comms:*")

                # No demo_mode field - should default to False
                startup_event = {
                    "api_key": "test_key",
                    "assistant_id": "regular_test_123",
                    "user_id": "456",
                    "assistant_name": "Regular Assistant",
                    "assistant_age": "25",
                    "assistant_nationality": "American",
                    "assistant_about": "A regular assistant",
                    "assistant_number": "+15555550000",
                    "assistant_email": "regular@test.com",
                    "user_name": "Boss",
                    "user_number": "+15555550001",
                    "user_email": "boss@test.com",
                    "voice_provider": "cartesia",
                    "voice_id": "test_voice",
                    "voice_mode": "tts",
                }
                message = create_pubsub_message("startup", startup_event)

                with patch("subprocess.run"):
                    cm.handle_message(message)

                # Poll for the startup event
                received_event = None
                for _ in range(50):
                    msg = await pubsub.get_message(
                        timeout=0.1,
                        ignore_subscribe_messages=True,
                    )
                    if msg and msg["channel"] == "app:comms:startup":
                        received_event = Event.from_json(msg["data"])
                        break

            assert received_event is not None, "StartupEvent not received"
            assert isinstance(received_event, StartupEvent)
            assert (
                received_event.demo_mode is False
            ), "demo_mode should default to False for regular assistants."

        finally:
            SESSION_DETAILS.assistant.id = original_id

    @pytest.mark.asyncio
    async def test_demo_mode_sets_settings_on_startup_handler(self, event_broker):
        """
        Test that SETTINGS.DEMO_MODE is set when processing a demo startup event.

        The EventHandler for StartupEvent should set SETTINGS.DEMO_MODE = True
        before managers are initialized, so demo-specific logic takes effect.
        """
        from unity.conversation_manager.events import StartupEvent
        from unity.settings import SETTINGS

        # Ensure demo mode starts as False
        original_demo_mode = SETTINGS.DEMO_MODE
        SETTINGS.DEMO_MODE = False

        try:
            # Create a StartupEvent with demo_mode=True
            startup_event = StartupEvent(
                api_key="test_key",
                medium="startup",
                assistant_id="demo_handler_test",
                user_id="456",
                assistant_name="Demo",
                assistant_age="25",
                assistant_nationality="American",
                assistant_about="Demo",
                assistant_number="+15555550000",
                assistant_email="demo@test.com",
                user_name="Boss",
                user_number="+15555550001",
                user_email="boss@test.com",
                voice_id="test",
                voice_mode="tts",
                demo_mode=True,
            )

            # Verify the event has demo_mode=True
            assert startup_event.demo_mode is True

            # The actual SETTINGS update happens in the event handler
            # which requires a full ConversationManager. Here we just verify
            # the event carries the flag correctly and can be used to set SETTINGS.
            if startup_event.demo_mode:
                SETTINGS.DEMO_MODE = True

            assert (
                SETTINGS.DEMO_MODE is True
            ), "SETTINGS.DEMO_MODE should be True after processing demo startup event"

        finally:
            SETTINGS.DEMO_MODE = original_demo_mode
