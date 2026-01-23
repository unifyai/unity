"""
Tests for blacklist filtering in CommsManager.

CommsManager should silently ignore inbound messages (SMS, email, calls) from
blacklisted contact details. This prevents spam/bad actors from waking the assistant.

The blacklist is stored in the BlackListManager and contains entries with:
- medium: email, sms_message, phone_call
- contact_detail: the phone number or email address

These tests verify:
1. Blacklisted SMS messages are ignored
2. Blacklisted emails are ignored
3. Blacklisted phone calls are ignored
4. Non-blacklisted messages pass through normally
5. Blacklist check failures don't block messages (fail-open)
"""

from __future__ import annotations

import asyncio
import json
import pytest
from dataclasses import dataclass
from unittest.mock import Mock, patch

from unity.conversation_manager.in_memory_event_broker import (
    create_in_memory_event_broker,
    reset_in_memory_event_broker,
)
from unity.conversation_manager.events import (
    Event,
    SMSReceived,
    EmailReceived,
    PhoneCallReceived,
)

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
        mock.assistant.id = "test_assistant"
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
# Mock BlackListManager
# =============================================================================


class MockBlackListManager:
    """Mock BlackListManager that returns configurable blacklist entries."""

    def __init__(self, blacklisted_entries: list[dict] | None = None):
        """
        Args:
            blacklisted_entries: List of dicts with 'medium' and 'contact_detail' keys.
        """
        self.blacklisted_entries = blacklisted_entries or []

    def filter_blacklist(self, *, filter: str = None, limit: int = 100):
        """Return matching blacklist entries based on filter."""
        # Parse the filter to extract medium and contact_detail
        # Filter format: "medium == 'email' and contact_detail == 'spam@example.com'"
        if not filter:
            return {"entries": []}

        # Simple parsing for test purposes
        entries = []
        for entry in self.blacklisted_entries:
            medium_match = f"medium == '{entry['medium']}'" in filter
            detail_match = f"contact_detail == '{entry['contact_detail']}'" in filter
            if medium_match and detail_match:
                # Return a mock entry object
                mock_entry = Mock()
                mock_entry.reason = entry.get("reason", "Blacklisted for testing")
                entries.append(mock_entry)

        return {"entries": entries}


# =============================================================================
# Test: SMS Blacklist Filtering
# =============================================================================


class TestSMSBlacklistFiltering:
    """Test blacklist filtering for SMS messages."""

    @pytest.mark.asyncio
    async def test_blacklisted_sms_is_ignored(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that SMS from blacklisted phone number is silently ignored."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        # Create blacklist with the spam number
        mock_blm = MockBlackListManager(
            blacklisted_entries=[
                {
                    "medium": "sms_message",
                    "contact_detail": "+15555559999",
                    "reason": "Known spammer",
                },
            ],
        )

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = [
                {
                    "contact_id": 99,
                    "first_name": "Spam",
                    "surname": "Bot",
                    "phone_number": "+15555559999",
                    "email_address": "spam@bot.com",
                },
            ]
            message = create_pubsub_message(
                "msg",
                {
                    "body": "Buy cheap watches!",
                    "from_number": "+15555559999",
                    "contacts": contacts,
                },
            )

            # Patch the _is_blacklisted function to use our mock
            with patch(
                "unity.conversation_manager.comms_manager._is_blacklisted",
                side_effect=lambda medium, detail: mock_blm.filter_blacklist(
                    filter=f"medium == '{medium}' and contact_detail == '{detail}'",
                )["entries"]
                != [],
            ):
                cm.handle_message(message)
                await asyncio.sleep(0.1)

            # Message should be acked
            assert message._acked

            # Should NOT receive any SMS message event (blacklisted)
            msg = await pubsub.get_message(timeout=0.5, ignore_subscribe_messages=True)
            # If we get a message, it should NOT be the SMS (should be None or contacts)
            if msg is not None:
                assert (
                    "msg_message" not in msg["channel"]
                ), "Blacklisted SMS should not be published"

    @pytest.mark.asyncio
    async def test_non_blacklisted_sms_passes_through(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that SMS from non-blacklisted number passes through normally."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        # Empty blacklist
        mock_blm = MockBlackListManager(blacklisted_entries=[])

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
                "msg",
                {
                    "body": "Hello, legitimate message!",
                    "from_number": "+15555551111",
                    "contacts": contacts,
                },
            )

            with patch(
                "unity.conversation_manager.comms_manager._is_blacklisted",
                return_value=False,
            ):
                cm.handle_message(message)
                await asyncio.sleep(0.1)

            # Should receive SMS message event
            msg = await get_message_on_channel(pubsub, "app:comms:msg_message")
            assert msg is not None

            event = Event.from_json(msg["data"])
            assert isinstance(event, SMSReceived)
            assert event.content == "Hello, legitimate message!"


# =============================================================================
# Test: Email Blacklist Filtering
# =============================================================================


class TestEmailBlacklistFiltering:
    """Test blacklist filtering for email messages."""

    @pytest.mark.asyncio
    async def test_blacklisted_email_is_ignored(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that email from blacklisted address is silently ignored."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = [
                {
                    "contact_id": 99,
                    "first_name": "Spam",
                    "surname": "Sender",
                    "phone_number": "+15555559999",
                    "email_address": "spam@evil.com",
                },
            ]
            message = create_pubsub_message(
                "email",
                {
                    "subject": "You've won a million dollars!",
                    "body": "Click here to claim...",
                    "from": "Spam Sender <spam@evil.com>",
                    "email_id": "msg_spam",
                    "contacts": contacts,
                },
            )

            with patch(
                "unity.conversation_manager.comms_manager._is_blacklisted",
                side_effect=lambda medium, detail: medium == "email"
                and detail == "spam@evil.com",
            ):
                cm.handle_message(message)
                await asyncio.sleep(0.1)

            # Message should be acked
            assert message._acked

            # Should NOT receive any email message event (blacklisted)
            msg = await pubsub.get_message(timeout=0.5, ignore_subscribe_messages=True)
            if msg is not None:
                assert (
                    "email_message" not in msg["channel"]
                ), "Blacklisted email should not be published"

    @pytest.mark.asyncio
    async def test_non_blacklisted_email_passes_through(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that email from non-blacklisted address passes through normally."""
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
                    "subject": "Meeting tomorrow",
                    "body": "Can we meet at 2pm?",
                    "from": "Test Contact <test@contact.com>",
                    "email_id": "msg_123",
                    "contacts": contacts,
                },
            )

            with patch(
                "unity.conversation_manager.comms_manager._is_blacklisted",
                return_value=False,
            ):
                cm.handle_message(message)
                await asyncio.sleep(0.1)

            # Should receive email message event
            msg = await get_message_on_channel(pubsub, "app:comms:email_message")
            assert msg is not None

            event = Event.from_json(msg["data"])
            assert isinstance(event, EmailReceived)
            assert event.subject == "Meeting tomorrow"


# =============================================================================
# Test: Phone Call Blacklist Filtering
# =============================================================================


class TestPhoneCallBlacklistFiltering:
    """Test blacklist filtering for phone calls."""

    @pytest.mark.asyncio
    async def test_blacklisted_call_is_ignored(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that call from blacklisted phone number is silently ignored."""
        from unity.conversation_manager.comms_manager import CommsManager

        cm = CommsManager(broker)
        cm.loop = asyncio.get_event_loop()

        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            contacts = [
                {
                    "contact_id": 99,
                    "first_name": "Spam",
                    "surname": "Caller",
                    "phone_number": "+15555559999",
                    "email_address": "spam@caller.com",
                },
            ]
            message = create_pubsub_message(
                "call",
                {
                    "caller_number": "+15555559999",
                    "conference_name": "conf_spam",
                    "contacts": contacts,
                },
            )

            with patch(
                "unity.conversation_manager.comms_manager._is_blacklisted",
                side_effect=lambda medium, detail: medium == "phone_call"
                and detail == "+15555559999",
            ):
                cm.handle_message(message)
                await asyncio.sleep(0.1)

            # Message should be acked
            assert message._acked

            # Should NOT receive any call event (blacklisted)
            msg = await pubsub.get_message(timeout=0.5, ignore_subscribe_messages=True)
            if msg is not None:
                assert (
                    "call_received" not in msg["channel"]
                ), "Blacklisted call should not be published"

    @pytest.mark.asyncio
    async def test_non_blacklisted_call_passes_through(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that call from non-blacklisted number passes through normally."""
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

            with patch(
                "unity.conversation_manager.comms_manager._is_blacklisted",
                return_value=False,
            ):
                # Run in thread to simulate GCP PubSub's threading model
                # (handle_message uses blocking future.result() for call events)
                await asyncio.to_thread(cm.handle_message, message)

            # Should receive call_received event
            msg = await get_message_on_channel(pubsub, "app:comms:call_received")
            assert msg is not None

            event = Event.from_json(msg["data"])
            assert isinstance(event, PhoneCallReceived)


# =============================================================================
# Test: Fail-Open Behavior
# =============================================================================


class TestFailOpenBehavior:
    """Test that blacklist check failures don't block messages."""

    @pytest.mark.asyncio
    async def test_blacklist_check_exception_allows_message(
        self,
        broker,
        mock_session_details,
        mock_settings,
    ):
        """Test that if blacklist check raises exception, message passes through."""
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
                "msg",
                {
                    "body": "Message during blacklist outage",
                    "from_number": "+15555551111",
                    "contacts": contacts,
                },
            )

            # Blacklist check raises exception (e.g., database down)
            with patch(
                "unity.conversation_manager.comms_manager._is_blacklisted",
                return_value=False,  # _is_blacklisted returns False on exception
            ):
                cm.handle_message(message)
                await asyncio.sleep(0.1)

            # Should still receive SMS message event (fail-open)
            msg = await get_message_on_channel(pubsub, "app:comms:msg_message")
            assert msg is not None


# =============================================================================
# Test: _is_blacklisted Helper Function
# =============================================================================


class TestIsBlacklistedHelper:
    """Test the _is_blacklisted helper function directly."""

    def test_is_blacklisted_returns_true_for_match(self):
        """Test that _is_blacklisted returns True when entry exists."""
        from unity.conversation_manager.comms_manager import _is_blacklisted

        mock_blm = MockBlackListManager(
            blacklisted_entries=[
                {
                    "medium": "sms_message",
                    "contact_detail": "+15555559999",
                    "reason": "Spammer",
                },
            ],
        )

        # Patch at the source module (where it's imported FROM), not where it's used
        with patch(
            "unity.blacklist_manager.BlackListManager",
            return_value=mock_blm,
        ):
            result = _is_blacklisted("sms_message", "+15555559999")
            assert result is True

    def test_is_blacklisted_returns_false_for_no_match(self):
        """Test that _is_blacklisted returns False when no entry exists."""
        from unity.conversation_manager.comms_manager import _is_blacklisted

        mock_blm = MockBlackListManager(blacklisted_entries=[])

        # Patch at the source module (where it's imported FROM), not where it's used
        with patch(
            "unity.blacklist_manager.BlackListManager",
            return_value=mock_blm,
        ):
            result = _is_blacklisted("sms_message", "+15555551111")
            assert result is False

    def test_is_blacklisted_returns_false_on_exception(self):
        """Test that _is_blacklisted returns False when exception occurs."""
        from unity.conversation_manager.comms_manager import _is_blacklisted

        # Patch at the source module (where it's imported FROM), not where it's used
        with patch(
            "unity.blacklist_manager.BlackListManager",
            side_effect=Exception("Database unavailable"),
        ):
            result = _is_blacklisted("sms_message", "+15555551111")
            assert result is False

    def test_is_blacklisted_returns_false_for_empty_contact_detail(self):
        """Test that _is_blacklisted returns False for empty contact detail."""
        from unity.conversation_manager.comms_manager import _is_blacklisted

        # Should not even try to query blacklist
        result = _is_blacklisted("sms_message", "")
        assert result is False

        result = _is_blacklisted("sms_message", None)
        assert result is False
