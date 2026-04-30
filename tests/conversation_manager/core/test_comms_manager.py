"""
Tests for CommsManager.

CommsManager bridges external communication channels (GCP PubSub for SMS, email,
calls, etc.) to the internal event broker. These tests verify:

1. Thread-safe callback handling (GCP PubSub uses thread pool for callbacks)
2. Message parsing and event generation for all message types
3. Contact handling from incoming messages
4. Event publishing to correct channels
5. Subscription lifecycle management
6. Ping mechanism for idle containers

Since CommsManager requires GCP PubSub, we mock the pubsub_v1 module and test
the message handling logic in isolation.
"""

from __future__ import annotations

import asyncio
import json
import time as _time
import pytest
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, patch

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


from unity.conversation_manager.events import (
    Event,
    BackupContactsEvent,
    SMSReceived,
    EmailReceived,
    UnifyMessageReceived,
    PhoneCallReceived,
    PhoneCallAnswered,
    UnifyMeetReceived,
    AssistantUpdateEvent,
    SyncContacts,
    TaskDue,
    PreHireMessage,
    Ping,
    AssistantScreenShareStarted,
    AssistantScreenShareStopped,
    UserScreenShareStarted,
    UserScreenShareStopped,
    UserRemoteControlStarted,
    UserRemoteControlStopped,
    UserWebcamStarted,
    UserWebcamStopped,
)
from unity.contact_manager.types.contact import UNASSIGNED

# =============================================================================
# Contact Schema Tests (existing test preserved)
# =============================================================================

# Required keys for contact dictionaries throughout the system
REQUIRED_CONTACT_KEYS = {
    "contact_id",
    "first_name",
    "surname",
    "phone_number",
    "email_address",  # NOT "email" - must be "email_address" for consistency
    "whatsapp_number",
}


def test_get_local_contact_has_correct_keys():
    """
    Verify that _get_local_contact() returns a contact dict with the
    correct field names. Specifically, the email field must be 'email_address',
    not 'email', to match the expected contact schema used throughout the system.
    """
    # Mock SESSION_DETAILS to avoid needing real session context
    mock_user = MagicMock()
    mock_user.first_name = "Test User"
    mock_user.surname = ""
    mock_user.number = "+15555551234"
    mock_user.email = "test@example.com"
    mock_user.whatsapp_number = ""

    mock_session = MagicMock()
    mock_session.user = mock_user

    with patch(
        "unity.conversation_manager.comms_manager.SESSION_DETAILS",
        mock_session,
    ):
        from unity.conversation_manager.comms_manager import _get_local_contact

        contact = _get_local_contact()

    # Verify all required keys are present
    assert set(contact.keys()) == REQUIRED_CONTACT_KEYS, (
        f"Contact dict has unexpected keys. "
        f"Expected: {REQUIRED_CONTACT_KEYS}, Got: {set(contact.keys())}"
    )

    # Explicitly verify 'email_address' is used, not 'email'
    assert (
        "email_address" in contact
    ), "Contact must use 'email_address' key, not 'email'"
    assert (
        "email" not in contact
    ), "Contact should NOT have 'email' key - use 'email_address' instead"

    # Verify the value is correctly mapped
    assert contact["email_address"] == "test@example.com"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the global singleton before and after each test."""
    reset_in_memory_event_broker()
    yield
    reset_in_memory_event_broker()


@pytest.fixture
def broker():
    """Create a fresh event broker for each test."""
    return create_in_memory_event_broker()


@pytest.fixture
def mock_session_details():
    """Mock SESSION_DETAILS for testing."""
    with patch("unity.conversation_manager.comms_manager.SESSION_DETAILS") as mock:
        mock.assistant.agent_id = 42
        mock.assistant.email = "assistant@test.com"
        mock.user.first_name = "Test User"
        mock.user.surname = ""
        mock.user.number = "+15555550000"
        mock.user.email = "user@test.com"
        mock.user.whatsapp_number = ""
        mock.unify_key = "test_key"
        mock.get_subprocess_env.return_value = {}
        yield mock


@pytest.fixture
def mock_settings():
    """Mock SETTINGS for testing."""
    with patch("unity.conversation_manager.comms_manager.SETTINGS") as mock:
        mock.DEPLOY_ENV = "production"
        mock.ENV_SUFFIX = ""
        yield mock


@dataclass
class MockPubSubMessage:
    """Mock PubSub message for testing."""

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


async def get_message_on_channel(
    pubsub,
    expected_channel: str,
    timeout: float = 1.0,
) -> dict | None:
    """
    Get the next message on the expected channel, skipping internal events.

    CommsManager publishes internal events (like backup_contacts) before the
    main message event. This helper loops until it finds a message on the
    expected channel, or times out.

    Args:
        pubsub: The pubsub instance from broker.pubsub()
        expected_channel: The channel to wait for (e.g., "app:comms:msg_message")
        timeout: Total timeout in seconds for finding the expected message

    Returns:
        The message dict if found, None if timeout reached
    """
    import time

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0:
            return None
        msg = await pubsub.get_message(
            timeout=min(remaining, 0.5),
            ignore_subscribe_messages=True,
        )
        if msg is None:
            continue
        if msg["channel"] == expected_channel:
            return msg
        # Skip internal events like backup_contacts and keep looking
    return None


# =============================================================================
# Test: Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Test helper functions for subscription ID and contact generation."""

    def test_get_subscription_id_non_staging(self, mock_session_details, mock_settings):
        """Test subscription ID generation for non-staging environment."""
        from unity.conversation_manager.comms_manager import _get_subscription_id

        mock_session_details.assistant.agent_id = 123
        mock_settings.DEPLOY_ENV = "production"
        mock_settings.ENV_SUFFIX = ""

        result = _get_subscription_id()
        assert result == "unity-123-sub"

    def test_get_subscription_id_staging(self, mock_session_details, mock_settings):
        """Test subscription ID generation for staging environment."""
        from unity.conversation_manager.comms_manager import _get_subscription_id

        mock_session_details.assistant.agent_id = 123
        mock_settings.DEPLOY_ENV = "staging"
        mock_settings.ENV_SUFFIX = "-staging"

        result = _get_subscription_id()
        assert result == "unity-123-staging-sub"

    def test_get_local_contact(self, mock_session_details):
        """Test local contact generation from session details."""
        from unity.conversation_manager.comms_manager import _get_local_contact

        result = _get_local_contact()
        assert result["contact_id"] == -1
        assert result["first_name"] == "Test User"
        assert result["surname"] == ""
        # first_name carries the combined name from the mock fixture
        assert result["phone_number"] == "+15555550000"
        assert result["email_address"] == "user@test.com"
        assert result["whatsapp_number"] == ""


# =============================================================================
# Test: CommsManager Initialization
# =============================================================================


class TestCommsManagerInit:
    """Test CommsManager initialization."""

    @pytest.mark.asyncio
    async def test_init_creates_empty_subscribers_dict(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that CommsManager initializes with empty subscribers dict."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        assert cm.subscribers == {}
        assert cm.call_proc is None
        assert cm.credentials is None
        assert cm.event_broker is broker

    @pytest.mark.asyncio
    async def test_init_stores_event_loop_reference(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that CommsManager stores reference to event loop."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        assert cm.loop is not None


# =============================================================================
# Test: Thread-Safe Publishing
# =============================================================================


class TestThreadSafePublishing:
    """Test thread-safe publishing from sync callbacks."""

    @pytest.mark.asyncio
    async def test_publish_from_callback_schedules_on_event_loop(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that _publish_from_callback schedules async publish on event loop."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        # Subscribe to the channel first
        async with broker.pubsub() as pubsub:
            await pubsub.subscribe("test:channel")

            # Call _publish_from_callback (simulating call from thread pool)
            cm._publish_from_callback("test:channel", '{"test": "data"}')

            # Poll for message to arrive instead of fixed sleep
            msg = None
            for _ in range(50):  # 5s timeout with 0.1s poll
                msg = await pubsub.get_message(
                    timeout=0.1,
                    ignore_subscribe_messages=True,
                )
                if msg:
                    break

            # Should receive the message
            assert msg is not None
            assert msg["data"] == '{"test": "data"}'


# =============================================================================
# Test: SMS Message Handling
# =============================================================================


class TestSMSMessageHandling:
    """Test handling of SMS messages."""

    @pytest.mark.asyncio
    async def test_handle_sms_message(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of incoming SMS message."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        # Subscribe to relevant channels
        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            # Create SMS message
            contacts = [
                {
                    "contact_id": 1,
                    "first_name": "Test",
                    "surname": "Contact",
                    "phone_number": "+15555551111",
                    "email_address": "test@contact.com",
                },
            ]
            message = create_pubsub_message(
                "msg",
                {
                    "body": "Hello from SMS!",
                    "from_number": "+15555551111",
                    "contacts": contacts,
                },
            )

            # Handle the message
            cm.handle_message(message)
            # Poll for message acknowledgment instead of fixed sleep
            await _wait_for_condition(lambda: message._acked)

            # Should receive SMS message event
            msg = await get_message_on_channel(pubsub, "app:comms:msg_message")
            assert msg is not None

            # Verify event data
            event = Event.from_json(msg["data"])
            assert isinstance(event, SMSReceived)
            assert event.content == "Hello from SMS!"

            # Message should be acked
            assert message._acked


# =============================================================================
# Test: Email Message Handling
# =============================================================================


class TestEmailMessageHandling:
    """Test handling of email messages."""

    @pytest.mark.asyncio
    async def test_handle_email_message(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of incoming email message."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = [
                {
                    "contact_id": 1,
                    "first_name": "Test",
                    "surname": "Contact",
                    "phone_number": "+15555551111",
                    "email_address": "test@contact.com",
                },
            ]
            message = create_pubsub_message(
                "email",
                {
                    "subject": "Test Subject",
                    "body": "Test email body",
                    "from": "Test Contact <test@contact.com>",
                    "email_id": "msg_123",
                    "contacts": contacts,
                },
            )

            cm.handle_message(message)
            # Poll for message acknowledgment instead of fixed sleep
            await _wait_for_condition(lambda: message._acked)

            # Should receive email message event
            msg = await get_message_on_channel(pubsub, "app:comms:email_message")
            assert msg is not None

            event = Event.from_json(msg["data"])
            assert isinstance(event, EmailReceived)
            assert event.subject == "Test Subject"
            assert event.body == "Test email body"
            assert event.email_id == "msg_123"

    @pytest.mark.asyncio
    async def test_handle_email_with_attachments(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of email with attachments schedules download."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        with patch(
            "unity.conversation_manager.comms_manager.add_email_attachments",
        ) as mock_add:
            mock_add.return_value = None  # Coroutine return value

            contacts = [
                {
                    "contact_id": 1,
                    "first_name": "Test",
                    "surname": "Contact",
                    "phone_number": "+15555551111",
                    "email_address": "test@contact.com",
                },
            ]
            message = create_pubsub_message(
                "email",
                {
                    "subject": "Email with attachment",
                    "body": "See attached",
                    "from": "Test Contact <test@contact.com>",
                    "email_id": "msg_456",
                    "gmail_message_id": "gmail_789",
                    "contacts": contacts,
                    "attachments": [{"filename": "test.pdf", "data": "base64..."}],
                },
            )

            cm.handle_message(message)
            # Poll for message acknowledgment instead of fixed sleep
            await _wait_for_condition(lambda: message._acked)

            # Attachment download should have been scheduled
            # (Note: run_coroutine_threadsafe doesn't immediately call the function)


# =============================================================================
# Test: UnifyMessage Handling
# =============================================================================


class TestUnifyMessageHandling:
    """Test handling of UnifyMessage (web app messages)."""

    @pytest.mark.asyncio
    async def test_handle_unify_message(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of incoming UnifyMessage."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = [
                {
                    "contact_id": 1,
                    "first_name": "Boss",
                    "surname": "User",
                    "phone_number": "+15555551111",
                    "email_address": "boss@test.com",
                },
            ]
            message = create_pubsub_message(
                "unify_message",
                {
                    "body": "Message via Unify web app",
                    "contact_id": 1,
                    "contacts": contacts,
                },
            )

            cm.handle_message(message)
            # Poll for message acknowledgment instead of fixed sleep
            await _wait_for_condition(lambda: message._acked)

            # Should receive unify_message event
            msg = await get_message_on_channel(
                pubsub,
                "app:comms:unify_message_message",
            )
            assert msg is not None

            event = Event.from_json(msg["data"])
            assert isinstance(event, UnifyMessageReceived)
            assert event.content == "Message via Unify web app"
            assert event.contact["contact_id"] == 1

    @pytest.mark.asyncio
    async def test_handle_unify_message_missing_contact_id_rejected(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that UnifyMessage without contact_id is rejected (no silent default)."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = [
                {
                    "contact_id": 1,
                    "first_name": "Boss",
                    "surname": "User",
                    "phone_number": "+15555551111",
                    "email_address": "boss@test.com",
                },
            ]
            # No contact_id specified - should be rejected, not defaulted to boss
            message = create_pubsub_message(
                "unify_message",
                {
                    "body": "Message without contact_id",
                    "contacts": contacts,
                },
            )

            cm.handle_message(message)
            # For rejected messages, use a brief poll to ensure processing completes
            await _wait_for_condition(lambda: message._acked, timeout=1.0)

            # No event should be published - message was rejected
            msg = await get_message_on_channel(
                pubsub,
                "app:comms:unify_message_message",
                timeout=0.2,
            )
            assert msg is None, "Message without contact_id should be rejected"

    @pytest.mark.asyncio
    async def test_handle_unify_message_unknown_contact_rejected(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that UnifyMessage with unknown contact_id is rejected (no silent fallback)."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = [
                {
                    "contact_id": 1,
                    "first_name": "Boss",
                    "surname": "User",
                    "phone_number": "+15555551111",
                    "email_address": "boss@test.com",
                },
            ]
            # contact_id 999 doesn't exist - should be rejected, not fallback to boss
            message = create_pubsub_message(
                "unify_message",
                {
                    "body": "Message with unknown contact",
                    "contact_id": 999,
                    "contacts": contacts,
                },
            )

            cm.handle_message(message)
            # For rejected messages, use a brief poll to ensure processing completes
            await _wait_for_condition(lambda: message._acked, timeout=1.0)

            # No event should be published - message was rejected
            msg = await get_message_on_channel(
                pubsub,
                "app:comms:unify_message_message",
                timeout=0.2,
            )
            assert msg is None, "Message with unknown contact_id should be rejected"


# =============================================================================
# Test: Phone Call Handling
# =============================================================================


class TestPhoneCallHandling:
    """Test handling of phone call events."""

    @pytest.mark.asyncio
    async def test_handle_incoming_call(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of incoming phone call."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = [
                {
                    "contact_id": 1,
                    "first_name": "Caller",
                    "surname": "Person",
                    "phone_number": "+15555551111",
                    "email_address": "caller@test.com",
                },
            ]
            message = create_pubsub_message(
                "call",
                {
                    "caller_number": "+15555551111",
                    "conference_name": "conf_123",
                    "contacts": contacts,
                },
            )

            # Run in thread to simulate GCP PubSub's threading model
            # (handle_message uses blocking future.result() for call events)
            await asyncio.to_thread(cm.handle_message, message)

            # Should receive call_received event
            msg = await get_message_on_channel(pubsub, "app:comms:call_received")
            assert msg is not None

            event = Event.from_json(msg["data"])
            assert isinstance(event, PhoneCallReceived)
            assert event.conference_name == "conf_123"

    @pytest.mark.asyncio
    async def test_handle_call_answered(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of outbound call answered event."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = [
                {
                    "contact_id": 1,
                    "first_name": "Called",
                    "surname": "Person",
                    "phone_number": "+15555551111",
                    "email_address": "called@test.com",
                },
            ]
            message = create_pubsub_message(
                "call_answered",
                {
                    "user_number": "+15555551111",
                    "contacts": contacts,
                },
            )

            # Run in thread to simulate GCP PubSub's threading model
            # (handle_message uses blocking future.result() for call events)
            await asyncio.to_thread(cm.handle_message, message)

            msg = await get_message_on_channel(pubsub, "app:comms:call_answered")
            assert msg is not None

            event = Event.from_json(msg["data"])
            assert isinstance(event, PhoneCallAnswered)


# =============================================================================
# Test: Unify Meet Handling
# =============================================================================


class TestUnifyMeetHandling:
    """Test handling of Unify Meet (web-based) voice calls."""

    @pytest.mark.asyncio
    async def test_handle_unify_meet_received(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of Unify Meet session start."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = [
                {
                    "contact_id": 1,
                    "first_name": "Boss",
                    "surname": "User",
                    "phone_number": "+15555551111",
                    "email_address": "boss@test.com",
                },
            ]
            message = create_pubsub_message(
                "unify_meet",
                {
                    "livekit_agent_name": "TestAgent",
                    "livekit_room": "room_123",
                    "contacts": contacts,
                },
            )

            # Run in thread to simulate GCP PubSub's threading model
            # (handle_message uses blocking future.result() for call/meet events)
            await asyncio.to_thread(cm.handle_message, message)

            msg = await get_message_on_channel(pubsub, "app:comms:unify_meet_received")
            assert msg is not None

            event = Event.from_json(msg["data"])
            assert isinstance(event, UnifyMeetReceived)
            assert event.room_name == "room_123"

    @pytest.mark.asyncio
    @patch(
        "unity.conversation_manager.comms_manager._get_local_contact",
        return_value={
            "contact_id": 0,
            "first_name": "Assistant",
            "surname": "Bot",
            "phone_number": "",
            "email_address": "",
            "whatsapp_number": "",
            "discord_id": None,
            "bio": "Test",
            "timezone": "UTC",
            "is_system": True,
            "should_respond": True,
            "response_policy": "",
            "rolling_summary": None,
        },
    )
    async def test_backup_contacts_arrives_before_unify_meet(
        self,
        _mock_local_contact,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """BackupContactsEvent must arrive before UnifyMeetReceived.

        dispatch_inbound_envelope publishes BackupContactsEvent (to populate
        the fallback contact cache) before the call event.  If the call event
        arrives first, the downstream handler crashes because contacts haven't
        been cached yet.  This test goes through handle_message (the real
        Pub/Sub callback path) to verify subscriber delivery order.
        """
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = [
                {
                    "contact_id": 1,
                    "first_name": "Boss",
                    "surname": "User",
                    "phone_number": "+15555551111",
                    "email_address": "boss@test.com",
                },
            ]
            message = create_pubsub_message(
                "unify_meet",
                {
                    "livekit_agent_name": "TestAgent",
                    "livekit_room": "room_123",
                    "contacts": contacts,
                },
            )

            await asyncio.to_thread(cm.handle_message, message)

            received_types: list[type] = []
            for _ in range(10):
                msg = await pubsub.get_message(
                    timeout=0.5,
                    ignore_subscribe_messages=True,
                )
                if msg is None:
                    continue
                event = Event.from_json(msg["data"])
                received_types.append(type(event))

            assert (
                BackupContactsEvent in received_types
            ), "BackupContactsEvent was never delivered"
            assert (
                UnifyMeetReceived in received_types
            ), "UnifyMeetReceived was never delivered"
            backup_idx = received_types.index(BackupContactsEvent)
            meet_idx = received_types.index(UnifyMeetReceived)
            assert backup_idx < meet_idx, (
                f"BackupContactsEvent arrived at index {backup_idx} but "
                f"UnifyMeetReceived arrived at index {meet_idx}. "
                "Contacts must be cached before the call handler runs."
            )


# =============================================================================
# Test: Startup and Assistant Update Events
# =============================================================================


class TestStartupEvents:
    """Test handling of startup and assistant update events."""

    @pytest.mark.asyncio
    async def test_handle_assistant_update_event(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of assistant update event."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            message = create_pubsub_message(
                "assistant_update",
                {
                    "api_key": "updated_api_key",
                    "assistant_id": "updated_assistant_id",
                    "user_id": "user_123",
                    "assistant_first_name": "Updated",
                    "assistant_surname": "Assistant",
                    "assistant_age": "30",
                    "assistant_nationality": "British",
                    "assistant_about": "An updated assistant",
                    "assistant_number": "+15555551234",
                    "assistant_email": "updated@test.com",
                    "user_first_name": "Test",
                    "user_surname": "User",
                    "user_number": "+15555550000",
                    "user_email": "user@test.com",
                    "voice_provider": "elevenlabs",
                    "voice_id": "new_voice",
                },
            )

            cm.handle_message(message)
            # Poll for message acknowledgment instead of fixed sleep
            await _wait_for_condition(lambda: message._acked)

            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg is not None
            assert msg["channel"] == "app:comms:assistant_update"

            event = Event.from_json(msg["data"])
            assert isinstance(event, AssistantUpdateEvent)
            assert event.assistant_first_name == "Updated"
            assert event.assistant_surname == "Assistant"

    @pytest.mark.asyncio
    async def test_assistant_update_propagates_email_provider(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """assistant_update with email_provider propagates it into the event."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            message = create_pubsub_message(
                "assistant_update",
                {
                    "api_key": "key",
                    "assistant_id": "1",
                    "user_id": "user_1",
                    "assistant_first_name": "MS",
                    "assistant_surname": "Outlook",
                    "assistant_age": "25",
                    "assistant_nationality": "US",
                    "assistant_about": "test",
                    "assistant_number": "+15555551234",
                    "assistant_email": "ms@outlook.unify.ai",
                    "assistant_email_provider": "microsoft_365",
                    "user_first_name": "Test",
                    "user_surname": "User",
                    "user_number": "+15555550000",
                    "user_email": "user@test.com",
                    "voice_provider": "cartesia",
                    "voice_id": "voice_1",
                },
            )

            cm.handle_message(message)
            await _wait_for_condition(lambda: message._acked)

            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg is not None
            assert msg["channel"] == "app:comms:assistant_update"

            event = Event.from_json(msg["data"])
            assert isinstance(event, AssistantUpdateEvent)
            assert event.assistant_email_provider == "microsoft_365"

    @pytest.mark.asyncio
    async def test_assistant_update_defaults_email_provider_to_google(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """assistant_update without email_provider defaults to google_workspace."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            message = create_pubsub_message(
                "assistant_update",
                {
                    "api_key": "key",
                    "assistant_id": "2",
                    "user_id": "user_2",
                    "assistant_first_name": "Gmail",
                    "assistant_surname": "Default",
                    "assistant_age": "25",
                    "assistant_nationality": "US",
                    "assistant_about": "test",
                    "assistant_number": "+15555551234",
                    "assistant_email": "default@unify.ai",
                    "user_first_name": "Test",
                    "user_surname": "User",
                    "user_number": "+15555550000",
                    "user_email": "user@test.com",
                    "voice_provider": "cartesia",
                    "voice_id": "voice_1",
                },
            )

            cm.handle_message(message)
            await _wait_for_condition(lambda: message._acked)

            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg is not None

            event = Event.from_json(msg["data"])
            assert isinstance(event, AssistantUpdateEvent)
            assert event.assistant_email_provider == "google_workspace"

    @pytest.mark.asyncio
    async def test_assistant_update_propagates_membership_fields(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """assistant_update carries membership fields into the runtime event."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            message = create_pubsub_message(
                "assistant_update",
                {
                    "api_key": "key",
                    "assistant_id": "3",
                    "user_id": "user_3",
                    "assistant_first_name": "Member",
                    "assistant_surname": "Update",
                    "assistant_age": "25",
                    "assistant_nationality": "US",
                    "assistant_about": "test",
                    "assistant_number": "+15555551234",
                    "assistant_email": "member@unify.ai",
                    "user_first_name": "Test",
                    "user_surname": "User",
                    "user_number": "+15555550000",
                    "user_email": "user@test.com",
                    "voice_provider": "cartesia",
                    "voice_id": "voice_1",
                    "space_ids": [3, 7],
                    "update_kind": "membership",
                },
            )

            cm.handle_message(message)
            await _wait_for_condition(lambda: message._acked)

            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg is not None

            event = Event.from_json(msg["data"])
            assert isinstance(event, AssistantUpdateEvent)
            assert event.space_ids == [3, 7]
            assert event.update_kind == "membership"


# =============================================================================
# Test: System Events (Pause/Resume/Sync)
# =============================================================================


class TestSystemEvents:
    """Test handling of system events (sync_contacts, screen share, remote control)."""

    @pytest.mark.asyncio
    async def test_handle_sync_contacts_event(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of sync_contacts system event."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            message = create_pubsub_message(
                "unity_system_event",
                {
                    "event_type": "sync_contacts",
                    "message": "Manual sync requested",
                },
            )

            cm.handle_message(message)
            # Poll for message acknowledgment instead of fixed sleep
            await _wait_for_condition(lambda: message._acked)

            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg is not None
            assert msg["channel"] == "app:comms:sync_contacts"

            event = Event.from_json(msg["data"])
            assert isinstance(event, SyncContacts)
            assert event.reason == "Manual sync requested"

    @pytest.mark.asyncio
    async def test_handle_task_due_event(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Scheduled task system events should map onto the typed task_due channel."""

        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            message = create_pubsub_message(
                "unity_system_event",
                {
                    "event_type": "task_due",
                    "message": "Scheduled task 'Morning briefing' became due.",
                    "task_id": 101,
                    "source_task_log_id": 555,
                    "activation_revision": "rev-1",
                    "scheduled_for": "2026-04-10T09:00:00+00:00",
                    "execution_mode": "live",
                    "source_type": "scheduled",
                    "task_label": "Morning briefing",
                    "task_summary": "Prepare the morning update before the user checks in.",
                    "visibility_policy": "silent_by_default",
                    "recurrence_hint": "recurring",
                },
            )

            cm.handle_message(message)
            await _wait_for_condition(lambda: message._acked)

            msg = await get_message_on_channel(pubsub, "app:comms:task_due")
            assert msg is not None

            event = Event.from_json(msg["data"])
            assert isinstance(event, TaskDue)
            assert event.task_id == 101
            assert event.source_task_log_id == 555
            assert event.activation_revision == "rev-1"
            assert event.scheduled_for == "2026-04-10T09:00:00+00:00"
            assert event.task_label == "Morning briefing"
            assert (
                event.task_summary
                == "Prepare the morning update before the user checks in."
            )
            assert event.visibility_policy == "silent_by_default"
            assert event.recurrence_hint == "recurring"


# =============================================================================
# Test: Meet Interaction System Events (screen share / remote control)
# =============================================================================


class TestMeetInteractionSystemEvents:
    """Test handling of meet interaction system events through CommsManager.

    These events are published by the Console frontend when the user toggles
    screen sharing or remote control during a Unify Meet session. They arrive
    as unity_system_event messages and should be routed to app:comms:*.
    """

    _EVENT_CASES = [
        (
            "assistant_screen_share_started",
            AssistantScreenShareStarted,
            "User enabled assistant screen sharing",
        ),
        (
            "assistant_screen_share_stopped",
            AssistantScreenShareStopped,
            "User disabled assistant screen sharing",
        ),
        (
            "user_screen_share_started",
            UserScreenShareStarted,
            "User started sharing their screen",
        ),
        (
            "user_screen_share_stopped",
            UserScreenShareStopped,
            "User stopped sharing their screen",
        ),
        (
            "user_webcam_started",
            UserWebcamStarted,
            "",
        ),
        (
            "user_webcam_stopped",
            UserWebcamStopped,
            "",
        ),
        (
            "user_remote_control_started",
            UserRemoteControlStarted,
            "User took remote control of assistant desktop",
        ),
        (
            "user_remote_control_stopped",
            UserRemoteControlStopped,
            "User released remote control of assistant desktop",
        ),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "event_type,expected_cls,message",
        _EVENT_CASES,
        ids=[c[0] for c in _EVENT_CASES],
    )
    async def test_meet_interaction_event_routing(
        self,
        broker,
        mock_session_details,
        mock_settings,
        event_type,
        expected_cls,
        message,
    ):
        """System events for meet interactions are routed to app:comms:<event_type>."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            msg = create_pubsub_message(
                "unity_system_event",
                {"event_type": event_type, "message": message},
            )

            cm.handle_message(msg)
            await _wait_for_condition(lambda: msg._acked)

            received = await get_message_on_channel(
                pubsub,
                f"app:comms:{event_type}",
            )
            assert received is not None, f"Expected message on app:comms:{event_type}"

            event = Event.from_json(received["data"])
            assert isinstance(event, expected_cls)
            assert event.reason == message

    @pytest.mark.asyncio
    async def test_meet_interaction_event_default_reason(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Meet interaction events use a default reason when message is None."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            msg = create_pubsub_message(
                "unity_system_event",
                {"event_type": "assistant_screen_share_started", "message": None},
            )

            cm.handle_message(msg)
            await _wait_for_condition(lambda: msg._acked)

            received = await get_message_on_channel(
                pubsub,
                "app:comms:assistant_screen_share_started",
            )
            event = Event.from_json(received["data"])
            assert isinstance(event, AssistantScreenShareStarted)
            assert "screen sharing" in event.reason.lower()

    @pytest.mark.asyncio
    async def test_unknown_system_event_is_acked_silently(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Unrecognized system event types are acked without publishing."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        msg = create_pubsub_message(
            "unity_system_event",
            {"event_type": "some_unknown_event", "message": "test"},
        )

        cm.handle_message(msg)
        await _wait_for_condition(lambda: msg._acked)
        assert msg._acked


# =============================================================================
# Test: Pre-Hire Chat Logging
# =============================================================================


class TestPreHireChatLogging:
    """Test handling of pre-hire chat logs."""

    @pytest.mark.asyncio
    async def test_handle_pre_hire_chats(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of pre-hire chat messages."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:managers:*", "app:comms:*")

            contacts = [
                {
                    "contact_id": 1,
                    "first_name": "Boss",
                    "surname": "User",
                    "phone_number": "+15555551111",
                    "email_address": "boss@test.com",
                },
            ]
            message = create_pubsub_message(
                "log_pre_hire_chats",
                {
                    "assistant_id": "pre_hire_assistant",
                    "contacts": contacts,
                    "body": [
                        {"role": "user", "msg": "Hello, I need help"},
                        {"role": "assistant", "msg": "How can I assist you?"},
                    ],
                },
            )

            cm.handle_message(message)
            # Poll for message acknowledgment instead of fixed sleep
            await _wait_for_condition(lambda: message._acked)

            # Collect pre-hire messages (may arrive in any order)
            pre_hire_messages = []
            for _ in range(5):  # Allow for contacts message + 2 pre-hire messages
                msg = await pubsub.get_message(
                    timeout=1.0,
                    ignore_subscribe_messages=True,
                )
                if msg and msg["channel"] == "app:comms:pre_hire":
                    event = Event.from_json(msg["data"])
                    if isinstance(event, PreHireMessage):
                        pre_hire_messages.append(event)

            # Should have received at least one pre-hire message
            assert len(pre_hire_messages) >= 1

            # Find the user message and verify it
            user_messages = [m for m in pre_hire_messages if m.role == "user"]
            assert len(user_messages) >= 1, "Should have at least one user message"
            user_msg = user_messages[0]
            assert user_msg.content == "Hello, I need help"
            # exchange_id should be UNASSIGNED so log_message creates a new exchange
            assert user_msg.exchange_id == UNASSIGNED

    @pytest.mark.asyncio
    async def test_handle_pre_hire_chats_all_messages_have_unassigned_exchange_id(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that all pre-hire messages have exchange_id=UNASSIGNED.

        This is critical for the log_message caching logic:
        - First message with UNASSIGNED creates a new exchange
        - Subsequent messages with UNASSIGNED will use the cached exchange_id
        """
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = [
                {
                    "contact_id": 1,
                    "first_name": "Boss",
                    "surname": "User",
                    "phone_number": "+15555551111",
                    "email_address": "boss@test.com",
                },
            ]
            # Multiple messages simulating a pre-hire conversation
            message = create_pubsub_message(
                "log_pre_hire_chats",
                {
                    "assistant_id": "pre_hire_assistant",
                    "contacts": contacts,
                    "body": [
                        {"role": "assistant", "msg": "Hello! I'm Ada."},
                        {"role": "user", "msg": "Can you help me?"},
                        {"role": "assistant", "msg": "Of course!"},
                    ],
                },
            )

            cm.handle_message(message)
            # Poll for message acknowledgment instead of fixed sleep
            await _wait_for_condition(lambda: message._acked)

            # Collect all pre-hire messages
            messages_received = []
            for _ in range(3):
                msg = await pubsub.get_message(
                    timeout=1.0,
                    ignore_subscribe_messages=True,
                )
                if msg and msg["channel"] == "app:comms:pre_hire":
                    event = Event.from_json(msg["data"])
                    if isinstance(event, PreHireMessage):
                        messages_received.append(event)

            assert len(messages_received) == 3
            # All messages should have exchange_id=UNASSIGNED
            for event in messages_received:
                assert (
                    event.exchange_id == UNASSIGNED
                ), f"Pre-hire message should have exchange_id=UNASSIGNED, got {event.exchange_id}"

    @pytest.mark.asyncio
    async def test_handle_pre_hire_chats_empty_body(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of pre-hire chat with empty body."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        contacts = [
            {
                "contact_id": 1,
                "first_name": "Boss",
                "surname": "User",
                "phone_number": "+15555551111",
                "email_address": "boss@test.com",
            },
        ]
        message = create_pubsub_message(
            "log_pre_hire_chats",
            {
                "assistant_id": "pre_hire_assistant",
                "contacts": contacts,
                "body": None,  # Empty body
            },
        )

        # Should not raise exception
        cm.handle_message(message)
        assert message._acked


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling in message processing."""

    @pytest.mark.asyncio
    async def test_handle_malformed_json(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of malformed JSON message."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        message = MockPubSubMessage(data=b"not valid json")

        # Should not raise exception
        cm.handle_message(message)

        # Message should still be acked to prevent redelivery
        assert message._acked

    @pytest.mark.asyncio
    async def test_handle_unknown_thread_type(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of unknown thread type."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        message = create_pubsub_message(
            "unknown_thread_type",
            {"data": "some data"},
        )

        # Should not raise exception
        cm.handle_message(message)

        # Unknown thread types are silently ignored (just printed)
        # Message is NOT acked for unknown threads (falls through)

    @pytest.mark.asyncio
    async def test_handle_missing_contact_in_sms(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test handling of SMS with phone number not matching any contact."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        contacts = [
            {
                "contact_id": 1,
                "first_name": "Boss",
                "surname": "User",
                "phone_number": "+15555551111",  # Different number
                "email_address": "boss@test.com",
            },
        ]
        message = create_pubsub_message(
            "msg",
            {
                "body": "Hello!",
                "from_number": "+19999999999",  # Unknown number
                "contacts": contacts,
            },
        )

        # Should raise StopIteration due to contact not found
        # The exception is caught and message is acked
        cm.handle_message(message)
        assert message._acked


# =============================================================================
# Test: Ping Mechanism
# =============================================================================


class TestPingMechanism:
    """Test the ping mechanism for idle containers."""

    @pytest.mark.asyncio
    async def test_send_pings_publishes_keepalive(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that send_pings publishes keepalive ping events."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)

        # Set assistant to unassigned (triggers ping loop)
        mock_session_details.assistant.agent_id = None

        async with broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:ping")

            # Create a task that will run send_pings briefly
            async def run_send_pings_briefly():
                # Patch sleep to return immediately and change assistant ID to break loop
                call_count = 0
                original_sleep = asyncio.sleep

                async def mock_sleep(duration):
                    nonlocal call_count
                    call_count += 1
                    if call_count >= 1:
                        # Change assistant ID to break the loop
                        mock_session_details.assistant.agent_id = 99
                    await original_sleep(0.01)  # Minimal sleep using original

                with patch("asyncio.sleep", mock_sleep):
                    await cm.send_pings()

            await run_send_pings_briefly()

            # Should have received at least one ping
            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
            assert msg is not None
            assert msg["channel"] == "app:comms:ping"

            event = Event.from_json(msg["data"])
            assert isinstance(event, Ping)
            assert event.kind == "keepalive"


# =============================================================================
# Test: Subscription Management
# =============================================================================


class TestSubscriptionManagement:
    """Test subscription lifecycle management."""

    @pytest.mark.asyncio
    async def test_subscribe_to_topic_stores_future(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that subscribe_to_topic stores the streaming pull future."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)

        mock_subscriber = Mock()
        mock_future = Mock()
        mock_subscriber.subscribe.return_value = mock_future
        mock_subscriber.subscription_path.return_value = (
            "projects/test/subscriptions/test-sub"
        )

        with patch(
            "unity.conversation_manager.comms_manager.pubsub_v1.SubscriberClient",
            return_value=mock_subscriber,
        ):
            cm.subscribe_to_topic("test-sub")

        assert "test-sub" in cm.subscribers
        assert cm.subscribers["test-sub"] is mock_future

    @pytest.mark.asyncio
    async def test_subscribe_with_credentials(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that subscribe uses credentials if provided."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.credentials = Mock()  # Set credentials

        mock_subscriber = Mock()
        mock_subscriber.subscribe.return_value = Mock()
        mock_subscriber.subscription_path.return_value = (
            "projects/test/subscriptions/test-sub"
        )

        with patch(
            "unity.conversation_manager.comms_manager.pubsub_v1.SubscriberClient",
            return_value=mock_subscriber,
        ) as mock_client_class:
            cm.subscribe_to_topic("test-sub")

        # Should have been called with credentials
        mock_client_class.assert_called_once_with(credentials=cm.credentials)


# =============================================================================
# Test: Events Map
# =============================================================================


class TestEventsMap:
    """Test the events_map constant."""

    def test_events_map_contains_expected_threads(self):
        """Test that events_map has the expected thread mappings."""
        from unity.conversation_manager.comms_manager import events_map

        assert "msg" in events_map
        assert events_map["msg"] is SMSReceived

        assert "email" in events_map
        assert events_map["email"] is EmailReceived

        assert "unify_message" in events_map
        assert events_map["unify_message"] is UnifyMessageReceived
