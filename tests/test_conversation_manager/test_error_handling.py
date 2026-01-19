"""
tests/test_conversation_manager/test_error_handling.py
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

from tests.test_conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import (
    Event,
    SMSReceived,
    EmailReceived,
    PhoneCallReceived,
    PhoneCallStarted,
    PhoneCallEnded,
    UnifyMeetEnded,
    Ping,
    GetContactsResponse,
    ActorResult,
)

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

    def test_from_json_unknown_event_type(self):
        """Event.from_json should raise on unknown event type."""
        data = json.dumps(
            {
                "event_name": "NonExistentEvent",
                "payload": {"timestamp": datetime.now().isoformat()},
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

    def test_from_json_missing_required_field(self):
        """Event.from_json should raise when required field is missing."""
        # SMSReceived requires 'contact' and 'content'
        data = json.dumps(
            {
                "event_name": "SMSReceived",
                "payload": {"timestamp": datetime.now().isoformat()},
            },
        )
        with pytest.raises(TypeError):
            Event.from_json(data)

    def test_from_json_extra_fields_ignored(self):
        """Event.from_json should ignore extra fields not in dataclass."""
        contact = TEST_CONTACTS[1]
        data = json.dumps(
            {
                "event_name": "SMSReceived",
                "payload": {
                    "contact": contact,
                    "content": "test message",
                    "timestamp": datetime.now().isoformat(),
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
        sms_thread = list(
            cm.contact_index.active_conversations[9999].threads["sms"],
        )
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
        sms_thread = list(
            cm.contact_index.active_conversations[contact["contact_id"]].threads["sms"],
        )
        matching = [
            m for m in sms_thread if getattr(m, "content", None) == "Duplicate message"
        ]
        assert len(matching) == 2

    @pytest.mark.asyncio
    async def test_phone_call_started_without_received(self, initialized_cm):
        """PhoneCallStarted without PhoneCallReceived should still work."""
        cm = initialized_cm
        contact = TEST_CONTACTS[1]

        # Skip PhoneCallReceived, go straight to Started
        result = await cm.step(PhoneCallStarted(contact=contact))

        # Should set mode to "call" even without prior Received event
        assert cm.cm.mode == "call"
        assert result.llm_requested is True

        # Clean up
        await cm.step(PhoneCallEnded(contact=contact))

    @pytest.mark.asyncio
    async def test_get_contacts_response_adds_contacts(self, initialized_cm):
        """GetContactsResponse should add/update contacts in the index."""
        cm = initialized_cm

        original_count = len(cm.contact_index.contacts)

        # Send new contact
        new_contacts = [
            {
                "contact_id": 100,
                "first_name": "New",
                "surname": "Contact",
                "email_address": "new@test.com",
                "phone_number": "+15555550100",
            },
        ]
        await cm.step(GetContactsResponse(contacts=new_contacts))

        # Contact 100 should now exist (added to existing contacts)
        assert 100 in cm.contact_index.contacts
        assert cm.contact_index.contacts[100].first_name == "New"
        # Original contacts should still exist
        assert len(cm.contact_index.contacts) == original_count + 1

    @pytest.mark.asyncio
    async def test_actor_result_for_nonexistent_task(self, initialized_cm):
        """ActorResult for a task not in active_tasks should not crash."""
        cm = initialized_cm

        # Send result for a task that doesn't exist
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

        new_contact = {
            "contact_id": 888,
            "first_name": "New",
            "surname": "Person",
            "email_address": "new@example.com",
            "phone_number": "+18888888888",
        }

        # Contact 888 not in active_conversations yet
        assert 888 not in cm.contact_index.active_conversations

        # Push a message
        cm.contact_index.push_message(
            new_contact,
            "sms",
            message_content="Hello",
            role="user",
        )

        # Now contact 888 should have an active conversation
        assert 888 in cm.contact_index.active_conversations
        sms_thread = list(cm.contact_index.active_conversations[888].threads["sms"])
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
    async def test_push_notification_with_datetime_timestamp(self, initialized_cm):
        """push_notif should accept datetime timestamp."""
        cm = initialized_cm

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        cm.cm.notifications_bar.push_notif(
            type="Test",
            notif_content="Test notification",
            timestamp=now,
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
# Contact Data Freshness Tests
# =============================================================================


class TestContactDataFreshness:
    """Tests verifying that contact_index always returns fresh data from ContactManager.

    These tests verify that when contacts are created or updated outside of the
    ConversationManager context (e.g., by the Actor via ContactManager), the
    contact_index.get_contact() method returns the up-to-date data.

    This is critical because:
    1. The Actor might create new contacts during task execution
    2. Contact details might be updated via ContactManager.update()
    3. The contact_index local cache can become stale

    The fix ensures contact_index always queries ContactManager (which has an
    auto-syncing cache backed by the database) for fresh data.
    """

    @pytest.mark.asyncio
    async def test_contact_created_via_contact_manager_is_visible(self, initialized_cm):
        """Contacts created via ContactManager should be immediately visible via contact_index."""
        cm = initialized_cm

        # Create a unique email to avoid conflicts
        unique_email = f"actor.created.{id(self)}@example.com"

        # Create contact directly via ContactManager (simulating Actor behavior)
        result = cm.cm.contact_manager._create_contact(
            first_name="ActorCreated",
            surname="Contact",
            email_address=unique_email,
            phone_number=f"+1500555{id(self) % 10000:04d}",
        )

        # Get the assigned contact_id from the result
        new_contact_id = result["details"]["contact_id"]

        # contact_index.get_contact() should find it via ContactManager
        contact = cm.contact_index.get_contact(contact_id=new_contact_id)
        assert contact is not None
        assert contact["first_name"] == "ActorCreated"

        # Also verify search by email works
        contact_by_email = cm.contact_index.get_contact(email=unique_email)
        assert contact_by_email is not None
        assert contact_by_email["first_name"] == "ActorCreated"

    @pytest.mark.asyncio
    async def test_contact_updated_via_contact_manager_reflects_changes(
        self,
        initialized_cm,
    ):
        """Updates to contacts via ContactManager should be reflected in contact_index."""
        cm = initialized_cm

        # Use system contact 0 (assistant) which always exists in ContactManager
        contact_id = 0

        # Get original data
        original = cm.contact_index.get_contact(contact_id=contact_id)
        assert original is not None
        original_bio = original.get("bio")

        # Update the contact directly via ContactManager (simulating Actor behavior)
        new_bio = f"Updated bio at {id(self)}"
        cm.cm.contact_manager.update_contact(
            contact_id=contact_id,
            bio=new_bio,
        )

        # contact_index.get_contact() should return the updated data
        updated = cm.contact_index.get_contact(contact_id=contact_id)
        assert updated is not None
        assert updated["bio"] == new_bio

        # Restore original
        cm.cm.contact_manager.update_contact(
            contact_id=contact_id,
            bio=original_bio or "",
        )

    @pytest.mark.asyncio
    async def test_stale_local_cache_is_bypassed(self, initialized_cm):
        """Even if local cache has old data, get_contact returns fresh data."""
        cm = initialized_cm

        # Use system contact 0 which exists in both local cache and ContactManager
        contact_id = 0

        # Manually set stale data in local cache
        if contact_id in cm.contact_index.contacts:
            cm.contact_index.contacts[contact_id].bio = "STALE_BIO_DATA"

        # Update via ContactManager to have different "fresh" data
        fresh_bio = f"FreshBio_{id(self)}"
        cm.cm.contact_manager.update_contact(
            contact_id=contact_id,
            bio=fresh_bio,
        )

        # get_contact should return fresh data, not stale cache
        result = cm.contact_index.get_contact(contact_id=contact_id)
        assert result is not None
        assert result["bio"] == fresh_bio

    @pytest.mark.asyncio
    async def test_contact_manager_not_set_falls_back_to_local_cache(
        self,
        initialized_cm,
    ):
        """When ContactManager is not set, should fall back to local cache."""
        cm = initialized_cm

        # Temporarily unset the contact_manager
        original_cm = cm.contact_index._contact_manager
        cm.contact_index._contact_manager = None

        try:
            # Should still work using local cache for contacts in TEST_CONTACTS
            contact = cm.contact_index.get_contact(
                contact_id=TEST_CONTACTS[1]["contact_id"],
            )
            # Should find it in local cache
            assert contact is not None
            assert contact["first_name"] == TEST_CONTACTS[1]["first_name"]
        finally:
            # Restore
            cm.contact_index._contact_manager = original_cm

    @pytest.mark.asyncio
    async def test_event_with_unknown_contact_uses_event_contact_fallback(
        self,
        initialized_cm,
    ):
        """SMS from unknown contact should use event.contact as fallback."""
        cm = initialized_cm

        # This contact doesn't exist anywhere - not in local cache or ContactManager
        unknown_contact = {
            "contact_id": 99999,
            "first_name": "CompletelyUnknown",
            "surname": "Person",
            "email_address": "completely.unknown@example.com",
            "phone_number": "+19999999999",
        }

        # Step should not raise - it should use event.contact as fallback
        result = await cm.step(
            SMSReceived(
                contact=unknown_contact,
                content="Hello from completely unknown contact",
            ),
        )

        # The handler should have run successfully
        assert result.llm_requested is True

        # The message should be in the conversation (using event.contact data)
        assert 99999 in cm.contact_index.active_conversations
        sms_thread = list(
            cm.contact_index.active_conversations[99999].threads["sms"],
        )
        assert len(sms_thread) >= 1
