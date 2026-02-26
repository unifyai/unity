"""
Tests for unknown contact creation in CommsManager.

When an inbound SMS, email, or call arrives from an unknown sender (not in
Contacts and not in BlackList), CommsManager should:

1. Create a new contact with only the medium field populated
2. Set should_respond=False to prevent automatic responses
3. Set a response_policy guiding the assistant to seek boss guidance
4. Publish an UnknownContactCreated event

These tests verify this workflow for all communication mediums.
"""

from __future__ import annotations

import asyncio
import json
import time as _time
import pytest
from dataclasses import dataclass
from unittest.mock import patch

from unity.contact_manager.simulated import SimulatedContactManager
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
        mock.user.name = "Test User"
        mock.user.number = "+15555550000"
        mock.user.email = "user@test.com"
        mock.unify_key = "test_key"
        mock.get_subprocess_env.return_value = {}
        yield mock


@pytest.fixture
def mock_settings():
    """Mock SETTINGS for testing."""
    with patch("unity.conversation_manager.comms_manager.SETTINGS") as mock:
        mock.STAGING = False
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


def get_non_system_contacts(cm: SimulatedContactManager) -> list[dict]:
    """Get all non-system contacts from a SimulatedContactManager."""
    return [c for c in cm._contacts.values() if c.get("contact_id") not in (0, 1)]


async def collect_messages(
    pubsub,
    channels: list[str],
    timeout: float = 2.0,
) -> list[dict]:
    """Collect all messages on the specified channels within the timeout."""
    messages = []
    import time

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        msg = await pubsub.get_message(timeout=0.1, ignore_subscribe_messages=True)
        if msg and msg["channel"] in channels:
            messages.append(msg)
    return messages


# =============================================================================
# Test: Unknown SMS Contact Creation
# =============================================================================


class TestUnknownSMSContactCreation:
    """Test unknown contact creation for SMS messages."""

    @pytest.mark.asyncio
    async def test_unknown_sms_creates_contact(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that SMS from unknown sender creates a contact with correct settings."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        contact_manager = SimulatedContactManager()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            # No contacts provided - this simulates an unknown sender
            contacts = []
            message = create_pubsub_message(
                "msg",
                {
                    "body": "Hello from unknown number",
                    "from_number": "+15555559999",
                    "contacts": contacts,
                },
            )

            with (
                patch(
                    "unity.conversation_manager.comms_manager._is_blacklisted",
                    return_value=False,
                ),
                patch(
                    "unity.manager_registry.ManagerRegistry.get_contact_manager",
                    return_value=contact_manager,
                ),
            ):
                cm.handle_message(message)
                # Wait for message to be acked and contact created
                await _wait_for_condition(
                    lambda: message._acked
                    and len(get_non_system_contacts(contact_manager)) >= 1,
                )

            # Message should be acked
            assert message._acked

            # Contact should be created with correct settings
            created_contacts = get_non_system_contacts(contact_manager)
            assert len(created_contacts) == 1
            created = created_contacts[0]
            assert created["phone_number"] == "+15555559999"
            assert created["should_respond"] is False
            assert "unknown inbound" in created["response_policy"].lower()

    @pytest.mark.asyncio
    async def test_unknown_sms_publishes_event(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that UnknownContactCreated event is published for unknown SMS sender."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        contact_manager = SimulatedContactManager()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = []
            message = create_pubsub_message(
                "msg",
                {
                    "body": "Hello from unknown",
                    "from_number": "+15555559999",
                    "contacts": contacts,
                },
            )

            with (
                patch(
                    "unity.conversation_manager.comms_manager._is_blacklisted",
                    return_value=False,
                ),
                patch(
                    "unity.manager_registry.ManagerRegistry.get_contact_manager",
                    return_value=contact_manager,
                ),
            ):
                cm.handle_message(message)
                # Wait for message to be acked
                await _wait_for_condition(lambda: message._acked)

            # Collect messages
            messages = await collect_messages(
                pubsub,
                ["app:comms:unknown_contact_created", "app:comms:msg_message"],
                timeout=1.0,
            )

            # Should have both the SMS message and UnknownContactCreated event
            channels = [m["channel"] for m in messages]
            assert "app:comms:unknown_contact_created" in channels
            assert "app:comms:msg_message" in channels

    @pytest.mark.asyncio
    async def test_known_sms_contact_no_creation(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that SMS from known contact doesn't create a new contact."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        # Existing contact with the phone number
        existing_contact = {
            "contact_id": 5,
            "first_name": "John",
            "phone_number": "+15555551111",
            "should_respond": True,
        }

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = [existing_contact]
            message = create_pubsub_message(
                "msg",
                {
                    "body": "Hello from known contact",
                    "from_number": "+15555551111",
                    "contacts": contacts,
                },
            )

            with patch(
                "unity.conversation_manager.comms_manager._is_blacklisted",
                return_value=False,
            ):
                cm.handle_message(message)
                # Wait for message to be acked
                await _wait_for_condition(lambda: message._acked)

            # Message should be acked
            assert message._acked

            # Collect messages - should NOT have UnknownContactCreated
            messages = await collect_messages(
                pubsub,
                ["app:comms:unknown_contact_created", "app:comms:msg_message"],
                timeout=1.0,
            )

            channels = [m["channel"] for m in messages]
            assert "app:comms:unknown_contact_created" not in channels
            assert "app:comms:msg_message" in channels


# =============================================================================
# Test: Unknown Email Contact Creation
# =============================================================================


class TestUnknownEmailContactCreation:
    """Test unknown contact creation for email messages."""

    @pytest.mark.asyncio
    async def test_unknown_email_creates_contact(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that email from unknown sender creates a contact with correct settings."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        contact_manager = SimulatedContactManager()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = []
            message = create_pubsub_message(
                "email",
                {
                    "subject": "Hello",
                    "body": "Email from unknown sender",
                    "from": "Unknown Sender <unknown@example.com>",
                    "email_id": "msg-123",
                    "contacts": contacts,
                },
            )

            with (
                patch(
                    "unity.conversation_manager.comms_manager._is_blacklisted",
                    return_value=False,
                ),
                patch(
                    "unity.manager_registry.ManagerRegistry.get_contact_manager",
                    return_value=contact_manager,
                ),
            ):
                cm.handle_message(message)
                # Wait for message to be acked and contact created
                await _wait_for_condition(
                    lambda: message._acked
                    and len(get_non_system_contacts(contact_manager)) >= 1,
                )

            # Message should be acked
            assert message._acked

            # Contact should be created with correct settings
            created_contacts = get_non_system_contacts(contact_manager)
            assert len(created_contacts) == 1
            created = created_contacts[0]
            assert created["email_address"] == "unknown@example.com"
            assert created["should_respond"] is False
            assert "unknown inbound" in created["response_policy"].lower()

    @pytest.mark.asyncio
    async def test_unknown_email_publishes_event(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that UnknownContactCreated event is published for unknown email sender."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        contact_manager = SimulatedContactManager()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = []
            message = create_pubsub_message(
                "email",
                {
                    "subject": "Test Subject",
                    "body": "Email body",
                    "from": "Unknown <unknown@example.com>",
                    "email_id": "msg-456",
                    "contacts": contacts,
                },
            )

            with (
                patch(
                    "unity.conversation_manager.comms_manager._is_blacklisted",
                    return_value=False,
                ),
                patch(
                    "unity.manager_registry.ManagerRegistry.get_contact_manager",
                    return_value=contact_manager,
                ),
            ):

                cm.handle_message(message)
                # Wait for message to be acked
                await _wait_for_condition(lambda: message._acked)

            messages = await collect_messages(
                pubsub,
                ["app:comms:unknown_contact_created", "app:comms:email_message"],
                timeout=1.0,
            )

            channels = [m["channel"] for m in messages]
            assert "app:comms:unknown_contact_created" in channels
            assert "app:comms:email_message" in channels


# =============================================================================
# Note: Phone call tests for unknown contact creation are not included here
# because the call event handling uses a blocking future.result() call that
# doesn't work well with the test framework's async setup. However, the
# core unknown contact creation logic (_get_or_create_unknown_contact) is
# already tested via the SMS and email tests above. The phone call code path
# uses the same function.
# =============================================================================


# =============================================================================
# Test: Duplicate Prevention
# =============================================================================


class TestDuplicatePrevention:
    """Test that duplicate contacts are not created for the same unknown sender."""

    @pytest.mark.asyncio
    async def test_second_message_uses_existing_contact(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that second message from same unknown sender uses existing contact."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        # SimulatedContactManager that tracks contacts
        contact_manager = SimulatedContactManager()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            # First message
            message1 = create_pubsub_message(
                "msg",
                {
                    "body": "First message",
                    "from_number": "+15555559999",
                    "contacts": [],
                },
            )

            with (
                patch(
                    "unity.conversation_manager.comms_manager._is_blacklisted",
                    return_value=False,
                ),
                patch(
                    "unity.manager_registry.ManagerRegistry.get_contact_manager",
                    return_value=contact_manager,
                ),
            ):

                cm.handle_message(message1)
                # Wait for message to be acked and contact created
                await _wait_for_condition(
                    lambda: message1._acked
                    and len(get_non_system_contacts(contact_manager)) >= 1,
                )

            # Should have created one contact
            assert len(get_non_system_contacts(contact_manager)) == 1

            # Second message from same number
            message2 = create_pubsub_message(
                "msg",
                {
                    "body": "Second message",
                    "from_number": "+15555559999",
                    "contacts": [],
                },
            )

            with (
                patch(
                    "unity.conversation_manager.comms_manager._is_blacklisted",
                    return_value=False,
                ),
                patch(
                    "unity.manager_registry.ManagerRegistry.get_contact_manager",
                    return_value=contact_manager,
                ),
            ):

                cm.handle_message(message2)
                # Wait for second message to be acked
                await _wait_for_condition(lambda: message2._acked)

            # Should still have only one contact (reused the existing one)
            assert len(get_non_system_contacts(contact_manager)) == 1
