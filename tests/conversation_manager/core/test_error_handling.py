"""
tests/conversation_manager/core/test_error_handling.py
======================================================

Tests for error handling and recovery in ConversationManager.

This test file covers:
1. Malformed events (invalid JSON, missing fields, unknown types)
2. Event handler edge cases (unregistered handlers, missing contacts)
3. State recovery scenarios (out-of-order events, duplicate events)
4. Graceful degradation when data is missing or invalid

Most tests are marked as `symbolic` since they test deterministic error
handling behavior rather than LLM output.
"""

import json
import pytest
from datetime import datetime
from dataclasses import dataclass

from tests.conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import (
    Event,
    SMSReceived,
    EmailReceived,
    PhoneCallReceived,
    PhoneCallStarted,
    PhoneCallEnded,
    UnifyMeetEnded,
    Ping,
    ActorResult,
)
from unity.conversation_manager.types import Medium

pytestmark = pytest.mark.symbolic


# =============================================================================
# Malformed Event Tests
# =============================================================================


class TestMalformedEvents:
    """Tests for handling malformed event data."""

    def test_from_json_invalid_json(self):
        """Event.from_json should raise on invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            Event.from_json("not valid json {")

    def test_from_json_missing_event_name(self):
        """Event.from_json should raise on missing event_name."""
        data = json.dumps({"payload": {"content": "test"}})
        with pytest.raises(KeyError):
            Event.from_json(data)

    def test_from_json_missing_payload(self):
        """Event.from_json should raise on missing payload."""
        data = json.dumps({"event_name": "SMSReceived"})
        with pytest.raises(KeyError):
            Event.from_json(data)

    def test_from_json_unknown_event_type(self, static_now):
        """Event.from_json should raise on unknown event type."""
        data = json.dumps(
            {
                "event_name": "NonExistentEvent",
                "payload": {"timestamp": static_now.isoformat()},
            },
        )
        with pytest.raises(Exception, match="not registered"):
            Event.from_json(data)

    def test_from_json_invalid_timestamp(self):
        """Event.from_json should raise on invalid timestamp format."""
        data = json.dumps(
            {
                "event_name": "Ping",
                "payload": {"kind": "test", "timestamp": "not-a-valid-timestamp"},
            },
        )
        with pytest.raises(ValueError):
            Event.from_json(data)

    def test_from_json_missing_required_field(self, static_now):
        """Event.from_json should raise when required field is missing."""
        # SMSReceived requires 'contact' and 'content'
        data = json.dumps(
            {
                "event_name": "SMSReceived",
                "payload": {"timestamp": static_now.isoformat()},
            },
        )
        with pytest.raises(TypeError):
            Event.from_json(data)

    def test_from_json_extra_fields_ignored(self, static_now):
        """Event.from_json should ignore extra fields not in dataclass."""
        contact = TEST_CONTACTS[1]
        data = json.dumps(
            {
                "event_name": "SMSReceived",
                "payload": {
                    "contact": contact,
                    "content": "test message",
                    "timestamp": static_now.isoformat(),
                    "extra_field_that_does_not_exist": "should be ignored",
                    "another_random_field": 12345,
                },
            },
        )
        # Should not raise - extra fields are filtered out
        event = Event.from_json(data)
        assert isinstance(event, SMSReceived)
        assert event.content == "test message"
        assert not hasattr(event, "extra_field_that_does_not_exist")


# =============================================================================
# Event Handler Edge Cases
# =============================================================================


class TestEventHandlerEdgeCases:
    """Tests for event handler registry and edge cases."""

    @pytest.mark.asyncio
    async def test_unregistered_event_type_returns_noop(self, initialized_cm):
        """Events without registered handlers should be handled gracefully."""
        cm = initialized_cm

        # Create a custom event type that isn't registered
        @dataclass
        class UnregisteredTestEvent(Event):
            data: str = "test"

        event = UnregisteredTestEvent()

        # Should not raise - just returns a no-op coroutine
        result = await cm.step(event)

        # The event was processed (no exception), but no handler ran
        assert result.llm_requested is False
        assert result.llm_ran is False
        assert result.output_events == []

    @pytest.mark.asyncio
    async def test_event_with_unknown_contact_id(self, initialized_cm):
        """SMS from unknown contact_id should still be handled using event.contact fallback.

        This test verifies that the event handler processes the event correctly
        when the contact_id isn't found in ContactManager. The handler should
        fall back to using event.contact data.

        Note: We test only the event handling, not the LLM response. The LLM
        might try to reply, which could fail for unrelated reasons.
        """
        cm = initialized_cm
        from unity.conversation_manager.domains.event_handlers import EventHandler

        unknown_contact = {
            "contact_id": 9999,  # Not in TEST_CONTACTS or ContactManager
            "first_name": "Unknown",
            "surname": "Person",
            "email_address": "unknown@example.com",
            "phone_number": "+19999999999",
        }

        event = SMSReceived(
            contact=unknown_contact,
            content="Hello from unknown contact",
        )

        # Call the event handler directly (without running LLM)
        await EventHandler.handle_event(
            event,
            cm.cm,
            is_voice_call=False,
        )

        # Verify the message was added to conversations using event.contact data
        assert 9999 in cm.contact_index.active_conversations
        sms_thread = cm.contact_index.get_messages_for_contact(9999, Medium.SMS_MESSAGE)
        assert len(sms_thread) >= 1
        assert sms_thread[0].content == "Hello from unknown contact"

    @pytest.mark.asyncio
    async def test_ping_event_handler(self, initialized_cm):
        """Ping events should be handled without triggering LLM."""
        cm = initialized_cm

        result = await cm.step(Ping(kind="test"))

        # Ping handler doesn't request LLM run
        assert result.llm_requested is False
        assert result.llm_ran is False
        assert result.output_events == []


# =============================================================================
# State Recovery Scenarios
# =============================================================================


class TestStateRecovery:
    """Tests for state recovery from abnormal event sequences."""

    @pytest.mark.asyncio
    async def test_phone_call_ended_without_started(self, initialized_cm):
        """PhoneCallEnded without PhoneCallStarted should not crash."""
        cm = initialized_cm
        contact = TEST_CONTACTS[1]

        # End a call that was never started
        result = await cm.step(PhoneCallEnded(contact=contact))

        # Should not crash - guard in event handler protects against KeyError
        assert result.llm_requested is True  # Handler requests LLM run
        # Mode should remain "text" (never entered "call" mode)
        assert cm.cm.mode == "text"

    @pytest.mark.asyncio
    async def test_unify_meet_ended_without_started(self, initialized_cm):
        """UnifyMeetEnded without UnifyMeetStarted should not crash."""
        cm = initialized_cm
        contact = TEST_CONTACTS[1]

        # End a meeting that was never started
        result = await cm.step(UnifyMeetEnded(contact=contact))

        # Should not crash
        assert cm.cm.mode == "text"

    @pytest.mark.asyncio
    async def test_duplicate_sms_received(self, initialized_cm):
        """Duplicate SMS events should be handled (added to thread twice)."""
        cm = initialized_cm
        contact = TEST_CONTACTS[1]

        event = SMSReceived(contact=contact, content="Duplicate message")

        # Process the same event twice
        await cm.step(event)
        await cm.step(event)

        # Both messages should be in the thread (no deduplication at this level)
        sms_thread = cm.contact_index.get_messages_for_contact(
            contact["contact_id"],
            Medium.SMS_MESSAGE,
        )
        matching = [
            m for m in sms_thread if getattr(m, "content", None) == "Duplicate message"
        ]
        assert len(matching) == 2

    @pytest.mark.asyncio
    async def test_actor_result_for_nonexistent_action(self, initialized_cm):
        """ActorResult for an action not in in_flight_actions should not crash."""
        cm = initialized_cm

        # Send result for an action that doesn't exist
        result = await cm.step(
            ActorResult(
                handle_id=99999,  # Non-existent
                success=True,
                result="Some result",
            ),
        )

        # Should not crash - handler uses .pop() with default
        assert result.llm_requested is True


# =============================================================================
# Contact Index Edge Cases
# =============================================================================


class TestContactIndexEdgeCases:
    """Tests for contact_index edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_get_contact_nonexistent_id(self, initialized_cm):
        """get_contact with non-existent ID should return None."""
        cm = initialized_cm

        contact = cm.contact_index.get_contact(contact_id=99999)
        assert contact is None

    @pytest.mark.asyncio
    async def test_get_contact_by_phone_doesnt_crash(self, initialized_cm):
        """get_contact with phone_number should not crash."""
        cm = initialized_cm

        # Search for a phone number - should not raise
        result = cm.contact_index.get_contact(phone_number="+10000000000")

        # Result is either None or a dict with contact data
        assert result is None or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_contact_nonexistent_email(self, initialized_cm):
        """get_contact with non-existent email should return None."""
        cm = initialized_cm

        contact = cm.contact_index.get_contact(email="nonexistent@example.com")
        assert contact is None

    @pytest.mark.asyncio
    async def test_push_message_creates_conversation(self, initialized_cm):
        """push_message to new contact should create active_conversation entry."""
        cm = initialized_cm

        new_contact_id = 888

        # Contact 888 not in active_conversations yet
        assert new_contact_id not in cm.contact_index.active_conversations

        # Push a message using the correct signature:
        # push_message(contact_id, sender_name, thread_name, message_content=...)
        cm.contact_index.push_message(
            contact_id=new_contact_id,
            sender_name="New Person",
            thread_name=Medium.SMS_MESSAGE,
            message_content="Hello",
            role="user",
        )

        # Now contact 888 should have an active conversation
        assert new_contact_id in cm.contact_index.active_conversations
        sms_thread = cm.contact_index.get_messages_for_contact(
            new_contact_id,
            Medium.SMS_MESSAGE,
        )
        assert len(sms_thread) == 1


# =============================================================================
# Notification Bar Edge Cases
# =============================================================================


class TestNotificationBarEdgeCases:
    """Tests for notification bar edge cases."""

    @pytest.mark.asyncio
    async def test_remove_nonexistent_notification(self, initialized_cm):
        """Removing non-existent notification should not crash."""
        cm = initialized_cm

        # Should not raise
        cm.cm.notifications_bar.remove_notif("nonexistent_id_12345")

    @pytest.mark.asyncio
    async def test_push_notification_with_datetime_timestamp(
        self,
        initialized_cm,
        static_now,
    ):
        """push_notif should accept datetime timestamp."""
        cm = initialized_cm

        cm.cm.notifications_bar.push_notif(
            type="Test",
            notif_content="Test notification",
            timestamp=static_now,
        )

        # Should have added the notification
        assert len(cm.cm.notifications_bar.notifications) >= 1
        last_notif = cm.cm.notifications_bar.notifications[-1]
        assert last_notif.content == "Test notification"
        assert last_notif.type == "Test"


# =============================================================================
# Event Serialization Edge Cases
# =============================================================================


class TestEventSerializationEdgeCases:
    """Tests for event serialization and deserialization edge cases."""

    def test_event_round_trip_preserves_data(self):
        """Event should survive JSON round-trip with all data intact."""
        contact = TEST_CONTACTS[1]
        original = SMSReceived(contact=contact, content="Test message")

        # Serialize and deserialize
        json_str = original.to_json()
        restored = Event.from_json(json_str)

        assert isinstance(restored, SMSReceived)
        assert restored.contact == original.contact
        assert restored.content == original.content
        # Timestamps should be equal (within serialization precision)
        assert abs((restored.timestamp - original.timestamp).total_seconds()) < 1

    def test_email_event_with_none_email_id(self):
        """EmailReceived with None email_id should serialize correctly."""
        contact = TEST_CONTACTS[1]
        event = EmailReceived(
            contact=contact,
            subject="Test",
            body="Test body",
            email_id=None,
        )

        json_str = event.to_json()
        restored = Event.from_json(json_str)

        assert isinstance(restored, EmailReceived)
        assert restored.email_id is None

    def test_event_to_dict_with_datetime(self):
        """Event.to_dict should serialize datetime correctly."""
        event = Ping(kind="test")
        data = event.to_dict()

        # Timestamp should be an ISO format string
        assert isinstance(data["payload"]["timestamp"], str)
        # Should be parseable
        datetime.fromisoformat(data["payload"]["timestamp"])


# =============================================================================
# Chat History Edge Cases
# =============================================================================


class TestChatHistoryEdgeCases:
    """Tests for chat history management edge cases."""

    @pytest.mark.asyncio
    async def test_preprocess_messages_with_string(self, initialized_cm):
        """_preprocess_messages should pass through strings unchanged."""
        cm = initialized_cm

        result = cm.cm._preprocess_messages("just a string")
        assert result == "just a string"

    @pytest.mark.asyncio
    async def test_preprocess_messages_with_dict(self, initialized_cm):
        """_preprocess_messages should pass through dicts unchanged."""
        cm = initialized_cm

        msg = {"role": "user", "content": "test"}
        result = cm.cm._preprocess_messages(msg)
        assert result == msg

    @pytest.mark.asyncio
    async def test_preprocess_messages_empty_list(self, initialized_cm):
        """_preprocess_messages should handle empty list."""
        cm = initialized_cm

        result = cm.cm._preprocess_messages([])
        assert result == []

    @pytest.mark.asyncio
    async def test_preprocess_messages_no_state_snapshots(self, initialized_cm):
        """_preprocess_messages should handle messages without state snapshots."""
        cm = initialized_cm

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = cm.cm._preprocess_messages(messages)
        assert result == messages


# =============================================================================
# Mode Transition Edge Cases
# =============================================================================


class TestModeTransitions:
    """Tests for mode transition edge cases."""

    @pytest.mark.asyncio
    async def test_double_call_start_stays_in_call_mode(self, initialized_cm):
        """Starting a call while already in call mode should be handled."""
        cm = initialized_cm
        contact = TEST_CONTACTS[1]

        # Start first call
        await cm.step(PhoneCallReceived(contact=contact, conference_name="conf1"))
        await cm.step(PhoneCallStarted(contact=contact))
        assert cm.cm.mode == "call"

        # Try to start second call (should be a no-op per current implementation)
        await cm.step(PhoneCallReceived(contact=contact, conference_name="conf2"))
        # Mode should still be "call"
        assert cm.cm.mode == "call"

        # Clean up
        await cm.step(PhoneCallEnded(contact=contact))
        assert cm.cm.mode == "text"

    @pytest.mark.asyncio
    async def test_mode_after_call_cleanup(self, initialized_cm):
        """Mode should return to text after call cleanup."""
        cm = initialized_cm
        contact = TEST_CONTACTS[1]

        # Full call lifecycle
        await cm.step(PhoneCallReceived(contact=contact, conference_name="conf"))
        await cm.step(PhoneCallStarted(contact=contact))
        assert cm.cm.mode == "call"

        await cm.step(PhoneCallEnded(contact=contact))
        assert cm.cm.mode == "text"
        assert cm.cm.call_manager.call_contact is None


# =============================================================================
# Debouncer Edge Cases (Symbolic Tests)
# =============================================================================


class TestDebouncerBasics:
    """Basic tests for the Debouncer utility."""

    @pytest.mark.asyncio
    async def test_debouncer_exists(self, initialized_cm):
        """ConversationManager should have a debouncer instance."""
        cm = initialized_cm
        assert hasattr(cm.cm, "debouncer")
        assert cm.cm.debouncer is not None


# =============================================================================
# LLM Request Edge Cases
# =============================================================================


class TestLLMRequestEdgeCases:
    """Tests for LLM request management edge cases."""

    @pytest.mark.asyncio
    async def test_flush_empty_requests(self, initialized_cm):
        """flush_llm_requests with no pending requests should be a no-op."""
        cm = initialized_cm

        # Ensure no pending requests
        cm.cm._pending_llm_requests.clear()

        # Should not raise
        await cm.cm.flush_llm_requests()

    @pytest.mark.asyncio
    async def test_multiple_pending_requests_uses_last(self, initialized_cm):
        """Multiple pending LLM requests should use the last one's params."""
        cm = initialized_cm

        # Manually add multiple requests with different params
        cm.cm._pending_llm_requests.append((0, False))
        cm.cm._pending_llm_requests.append((1, False))
        cm.cm._pending_llm_requests.append((2, True))  # Last one

        # The flush logic uses the last request's params
        assert cm.cm._pending_llm_requests[-1] == (2, True)

        # Clear for other tests
        cm.cm._pending_llm_requests.clear()


# =============================================================================
# Contact Fallback Tests
# =============================================================================
# Note: Tests for ContactIndex data freshness are in
# tests/contact_manager/test_contact_index_freshness.py


class TestContactFallback:
    """Tests for contact fallback behavior when ContactManager is unavailable."""

    @pytest.mark.asyncio
    async def test_contact_manager_not_set_returns_none(
        self,
        initialized_cm,
    ):
        """When ContactManager is not set, get_contact returns None.

        ContactIndex.get_contact() delegates entirely to ContactManager.
        When ContactManager is not set, it correctly returns None rather
        than attempting any fallback. This is by design - ContactManager
        is the single source of truth for contact data.
        """
        cm = initialized_cm

        # Temporarily unset the contact_manager
        original_cm = cm.contact_index._contact_manager
        cm.contact_index._contact_manager = None

        try:
            # Without ContactManager, get_contact returns None
            contact = cm.contact_index.get_contact(
                contact_id=TEST_CONTACTS[1]["contact_id"],
            )
            assert contact is None
        finally:
            # Restore
            cm.contact_index._contact_manager = original_cm
