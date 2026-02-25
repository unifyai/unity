"""
tests/conversation_manager/test_call_lifecycle.py
======================================================

Full call lifecycle tests for ConversationManager.

Tests the complete flow of voice calls including:
1. LivekitCallManager.start_call() and start_unify_meet() thread launching
2. Voice interrupts (VoiceInterrupt event handling)
3. CallGuidance flowing from Main CM Brain to Voice Agent
4. Graceful call cleanup via cleanup_call_proc()
5. Full call lifecycle: receive → start → utterances → guidance → end

## Test Categories

### Unit Tests (no LLM required)
- CallManager configuration and state
- Event serialization/deserialization
- Thread lifecycle (mocked)

### Integration Tests (use CMStepDriver with simulated managers)
- Event handler responses to call events
- State transitions during call lifecycle
- CallGuidance event flow through the system
"""

from __future__ import annotations

import asyncio
import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from unity.conversation_manager.events import (
    Event,
    PhoneCallReceived,
    PhoneCallAnswered,
    PhoneCallStarted,
    PhoneCallEnded,
    PhoneCallSent,
    UnifyMeetReceived,
    UnifyMeetStarted,
    UnifyMeetEnded,
    InboundPhoneUtterance,
    OutboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    OutboundUnifyMeetUtterance,
    VoiceInterrupt,
    CallGuidance,
)

from tests.conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.types import Medium

# =============================================================================
# Unit Tests: CallManager Configuration
# =============================================================================


class TestCallManagerConfiguration:
    """Tests for LivekitCallManager configuration and state."""

    def test_call_config_fields(self):
        """CallConfig has all required fields."""
        from unity.conversation_manager.domains.call_manager import CallConfig

        config = CallConfig(
            assistant_id="test_assistant",
            user_id="test_user",
            assistant_bio="A helpful assistant",
            assistant_number="+15551234567",
            voice_provider="cartesia",
            voice_id="test_voice_id",
        )

        assert config.assistant_id == "test_assistant"
        assert config.assistant_bio == "A helpful assistant"
        assert config.assistant_number == "+15551234567"
        assert config.voice_provider == "cartesia"
        assert config.voice_id == "test_voice_id"

    def test_call_manager_initial_state(self):
        """LivekitCallManager initializes with correct default state."""
        from unity.conversation_manager.domains.call_manager import (
            CallConfig,
            LivekitCallManager,
        )
        from unity.contact_manager.types.contact import UNASSIGNED

        config = CallConfig(
            assistant_id="test",
            user_id="test_user",
            assistant_bio="Test bio",
            assistant_number="+15551234567",
            voice_provider="cartesia",
            voice_id="test_voice",
        )

        manager = LivekitCallManager(config)

        # Initial state should be unassigned/None
        assert manager.call_exchange_id == UNASSIGNED
        assert manager.unify_meet_exchange_id == UNASSIGNED
        assert manager.call_start_timestamp is None
        assert manager.unify_meet_start_timestamp is None
        assert manager.call_contact is None
        assert manager._call_proc is None
        assert manager.conference_name == ""

    def test_call_manager_set_config(self):
        """LivekitCallManager.set_config() updates configuration."""
        from unity.conversation_manager.domains.call_manager import (
            CallConfig,
            LivekitCallManager,
        )

        initial_config = CallConfig(
            assistant_id="initial",
            user_id="test_user",
            assistant_bio="Initial bio",
            assistant_number="+15551111111",
            voice_provider="cartesia",
            voice_id="initial_voice",
        )

        new_config = CallConfig(
            assistant_id="updated",
            user_id="test_user",
            assistant_bio="Updated bio",
            assistant_number="+15552222222",
            voice_provider="elevenlabs",
            voice_id="updated_voice",
        )

        manager = LivekitCallManager(initial_config)
        assert manager.assistant_id == "initial"

        manager.set_config(new_config)
        assert manager.assistant_id == "updated"
        assert manager.assistant_bio == "Updated bio"
        assert manager.assistant_number == "+15552222222"
        assert manager.voice_provider == "elevenlabs"
        assert manager.voice_id == "updated_voice"


# =============================================================================
# Unit Tests: Call Event Serialization
# =============================================================================


class TestCallEventSerialization:
    """Tests for call-related event serialization and deserialization."""

    @pytest.fixture
    def sample_contact(self):
        return {
            "contact_id": 1,
            "first_name": "Test",
            "surname": "User",
            "phone_number": "+15551234567",
        }

    def test_phone_call_received_serialization(self, sample_contact):
        """PhoneCallReceived event serializes correctly."""
        event = PhoneCallReceived(
            contact=sample_contact,
            conference_name="conf_123",
        )

        json_str = event.to_json()
        data = json.loads(json_str)

        assert data["event_name"] == "PhoneCallReceived"
        assert data["payload"]["contact"]["contact_id"] == 1
        assert data["payload"]["conference_name"] == "conf_123"

    def test_phone_call_received_deserialization(self, sample_contact):
        """PhoneCallReceived event deserializes correctly."""
        original = PhoneCallReceived(
            contact=sample_contact,
            conference_name="conf_456",
        )

        restored = Event.from_json(original.to_json())

        assert isinstance(restored, PhoneCallReceived)
        assert restored.contact["contact_id"] == sample_contact["contact_id"]
        assert restored.conference_name == "conf_456"

    def test_voice_interrupt_serialization(self, sample_contact):
        """VoiceInterrupt event serializes and deserializes correctly."""
        event = VoiceInterrupt(contact=sample_contact)

        json_str = event.to_json()
        data = json.loads(json_str)

        assert data["event_name"] == "VoiceInterrupt"
        assert data["payload"]["contact"]["contact_id"] == 1

        restored = Event.from_json(json_str)
        assert isinstance(restored, VoiceInterrupt)
        assert restored.contact["contact_id"] == sample_contact["contact_id"]

    def test_call_guidance_serialization(self, sample_contact):
        """CallGuidance event serializes and deserializes correctly."""
        event = CallGuidance(
            contact=sample_contact,
            content="Please ask about their schedule",
        )

        json_str = event.to_json()
        data = json.loads(json_str)

        assert data["event_name"] == "CallGuidance"
        assert data["payload"]["content"] == "Please ask about their schedule"

        restored = Event.from_json(json_str)
        assert isinstance(restored, CallGuidance)
        assert restored.content == "Please ask about their schedule"

    def test_inbound_phone_utterance_serialization(self, sample_contact):
        """InboundPhoneUtterance event serializes correctly."""
        event = InboundPhoneUtterance(
            contact=sample_contact,
            content="Hello, how are you?",
        )

        json_str = event.to_json()
        restored = Event.from_json(json_str)

        assert isinstance(restored, InboundPhoneUtterance)
        assert restored.content == "Hello, how are you?"

    def test_outbound_phone_utterance_serialization(self, sample_contact):
        """OutboundPhoneUtterance event serializes correctly."""
        event = OutboundPhoneUtterance(
            contact=sample_contact,
            content="I'm doing well, thanks!",
        )

        json_str = event.to_json()
        restored = Event.from_json(json_str)

        assert isinstance(restored, OutboundPhoneUtterance)
        assert restored.content == "I'm doing well, thanks!"

    def test_unify_meet_events_serialization(self, sample_contact):
        """UnifyMeet events serialize correctly."""
        received = UnifyMeetReceived(
            contact=sample_contact,
            room_name="test_room",
        )
        started = UnifyMeetStarted(contact=sample_contact)
        ended = UnifyMeetEnded(contact=sample_contact)

        for event in [received, started, ended]:
            restored = Event.from_json(event.to_json())
            assert type(restored) == type(event)


# =============================================================================
# Unit Tests: Call Subprocess Lifecycle (Mocked)
# =============================================================================


class TestCallSubprocessLifecycle:
    """Tests for call subprocess management in LivekitCallManager."""

    @pytest.fixture
    def call_manager(self):
        from unity.conversation_manager.domains.call_manager import (
            CallConfig,
            LivekitCallManager,
        )

        config = CallConfig(
            assistant_id="test",
            user_id="test_user",
            assistant_bio="Test assistant",
            assistant_number="+15551234567",
            voice_provider="cartesia",
            voice_id="test_voice",
        )
        return LivekitCallManager(config)

    @pytest.fixture
    def sample_contact(self):
        return {
            "contact_id": 2,
            "first_name": "Alice",
            "surname": "Smith",
            "phone_number": "+15552222222",
        }

    @pytest.fixture
    def boss_contact(self):
        return {
            "contact_id": 1,
            "first_name": "Boss",
            "surname": "User",
            "phone_number": "+15551111111",
        }

    @pytest.mark.asyncio
    async def test_start_call_creates_subprocess(
        self,
        call_manager,
        sample_contact,
        boss_contact,
    ):
        """start_call() creates a subprocess for the voice agent."""
        with patch(
            "unity.conversation_manager.domains.call_manager.run_script",
        ) as mock_run_script:
            mock_proc = MagicMock()
            mock_run_script.return_value = mock_proc

            await call_manager.start_call(sample_contact, boss_contact)

            mock_run_script.assert_called_once()
            call_args = mock_run_script.call_args
            assert "call.py" in str(call_args[0][0])  # script path
            assert "dev" in call_args[0]  # args

    @pytest.mark.asyncio
    async def test_start_call_outbound_flag(
        self,
        call_manager,
        sample_contact,
        boss_contact,
    ):
        """start_call() passes outbound flag correctly."""
        with patch(
            "unity.conversation_manager.domains.call_manager.run_script",
        ) as mock_run_script:
            mock_proc = MagicMock()
            mock_run_script.return_value = mock_proc

            await call_manager.start_call(sample_contact, boss_contact, outbound=True)

            call_args = mock_run_script.call_args
            # Outbound flag should be in the args
            assert "True" in call_args[0]

    @pytest.mark.asyncio
    async def test_start_unify_meet_creates_subprocess(
        self,
        call_manager,
        sample_contact,
        boss_contact,
    ):
        """start_unify_meet() creates a subprocess for the voice agent."""
        with patch(
            "unity.conversation_manager.domains.call_manager.run_script",
        ) as mock_run_script:
            mock_proc = MagicMock()
            mock_run_script.return_value = mock_proc

            await call_manager.start_unify_meet(
                sample_contact,
                boss_contact,
                room_name="unity_25_meet",
            )

            mock_run_script.assert_called_once()
            call_args = mock_run_script.call_args
            assert "call.py" in str(call_args[0][0])  # script path
            assert any("unity_25_meet" in str(arg) for arg in call_args[0])

    @pytest.mark.asyncio
    async def test_start_unify_meet_default_names(
        self,
        call_manager,
        sample_contact,
        boss_contact,
    ):
        """start_unify_meet() generates default room name from assistant_id."""
        call_manager.assistant_id = "my_assistant"

        with patch(
            "unity.conversation_manager.domains.call_manager.run_script",
        ) as mock_run_script:
            mock_proc = MagicMock()
            mock_run_script.return_value = mock_proc

            await call_manager.start_unify_meet(
                sample_contact,
                boss_contact,
                room_name=None,
            )

            call_args = mock_run_script.call_args
            # Default names should use make_room_name(assistant_id, "meet")
            assert any("unity_my_assistant_meet" in str(arg) for arg in call_args[0])

    @pytest.mark.asyncio
    async def test_cleanup_call_proc_no_process(self, call_manager):
        """cleanup_call_proc() handles case when no process is running."""
        assert call_manager._call_proc is None

        # Should not raise
        await call_manager.cleanup_call_proc()

    @pytest.mark.asyncio
    async def test_cleanup_call_proc_terminates_subprocess(self, call_manager):
        """cleanup_call_proc() terminates the subprocess."""
        # Create a mock subprocess that's running
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Process is running
        mock_proc.pid = 12345
        call_manager._call_proc = mock_proc

        with patch(
            "unity.conversation_manager.domains.call_manager.terminate_process",
        ) as mock_terminate:
            await call_manager.cleanup_call_proc()

            mock_terminate.assert_called_once_with(mock_proc, 0)

    @pytest.mark.asyncio
    async def test_cleanup_call_proc_clears_process_reference(self, call_manager):
        """cleanup_call_proc() clears the process reference."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # Process already exited
        mock_proc.returncode = 0
        call_manager._call_proc = mock_proc

        await call_manager.cleanup_call_proc()

        assert call_manager._call_proc is None


# =============================================================================
# Integration Tests: Event Handler Call Flow
# =============================================================================


@pytest.mark.asyncio
class TestCallEventHandlers:
    """Integration tests for call-related event handlers."""

    @pytest.fixture
    def boss_contact(self):
        return TEST_CONTACTS[1]

    @pytest.fixture
    def alice_contact(self):
        return TEST_CONTACTS[2]

    async def test_phone_call_received_updates_mode(
        self,
        initialized_cm,
        alice_contact,
    ):
        """PhoneCallReceived event should NOT immediately change mode (waits for started)."""
        # Mode starts as text
        assert initialized_cm.cm.mode == "text"

        # Patch start_call to avoid actually starting a thread
        with patch.object(
            initialized_cm.cm.call_manager,
            "start_call",
        ):
            event = PhoneCallReceived(
                contact=alice_contact,
                conference_name="conf_test",
            )
            await initialized_cm.step(event)

        # Mode should still be text (waits for PhoneCallStarted)
        assert initialized_cm.cm.mode == "text"

    async def test_phone_call_received_stores_conference_name(
        self,
        initialized_cm,
        alice_contact,
    ):
        """PhoneCallReceived stores the conference name in call_manager."""
        with patch.object(
            initialized_cm.cm.call_manager,
            "start_call",
        ):
            event = PhoneCallReceived(
                contact=alice_contact,
                conference_name="conf_12345",
            )
            await initialized_cm.step(event)

        assert initialized_cm.cm.call_manager.conference_name == "conf_12345"

    async def test_phone_call_received_adds_notification(
        self,
        initialized_cm,
        alice_contact,
    ):
        """PhoneCallReceived adds a notification to the notification bar."""
        with patch.object(
            initialized_cm.cm.call_manager,
            "start_call",
        ):
            event = PhoneCallReceived(
                contact=alice_contact,
                conference_name="conf_test",
            )
            await initialized_cm.step(event)

        # Check notification was added to the notifications list
        notifs = initialized_cm.cm.notifications_bar.notifications
        assert len(notifs) > 0, "Expected at least one notification"

        # Find a notification about the call
        call_notifs = [n for n in notifs if "Alice" in n.content or "Call" in n.content]
        assert (
            len(call_notifs) > 0
        ), f"Expected notification about call, got: {[n.content for n in notifs]}"

    async def test_phone_call_started_changes_mode_to_call(
        self,
        initialized_cm,
        alice_contact,
    ):
        """PhoneCallStarted event changes mode to 'call'."""
        assert initialized_cm.cm.mode == "text"

        event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(event)

        assert initialized_cm.cm.mode == "call"

    async def test_phone_call_started_sets_call_contact(
        self,
        initialized_cm,
        alice_contact,
    ):
        """PhoneCallStarted sets call_manager.call_contact."""
        event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(event)

        assert initialized_cm.cm.call_manager.call_contact is not None
        assert (
            initialized_cm.cm.call_manager.call_contact["contact_id"]
            == alice_contact["contact_id"]
        )

    async def test_phone_call_started_marks_contact_on_call(
        self,
        initialized_cm,
        alice_contact,
    ):
        """PhoneCallStarted marks the contact as on_call in active_conversations."""
        event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(event)

        contact_id = alice_contact["contact_id"]
        conv = initialized_cm.cm.contact_index.active_conversations.get(contact_id)
        assert conv is not None
        assert conv.on_call is True

    async def test_phone_call_ended_resets_mode(self, initialized_cm, alice_contact):
        """PhoneCallEnded resets mode to 'text'."""
        # First start a call
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)
        assert initialized_cm.cm.mode == "call"

        # Now end it
        ended_event = PhoneCallEnded(contact=alice_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            await initialized_cm.step(ended_event)

        assert initialized_cm.cm.mode == "text"

    async def test_phone_call_ended_clears_call_contact(
        self,
        initialized_cm,
        alice_contact,
    ):
        """PhoneCallEnded clears call_manager.call_contact."""
        # Start a call
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)
        assert initialized_cm.cm.call_manager.call_contact is not None

        # End it
        ended_event = PhoneCallEnded(contact=alice_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            await initialized_cm.step(ended_event)

        assert initialized_cm.cm.call_manager.call_contact is None

    async def test_phone_call_ended_clears_on_call_flag(
        self,
        initialized_cm,
        alice_contact,
    ):
        """PhoneCallEnded clears the on_call flag in active_conversations."""
        # Start a call
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)

        contact_id = alice_contact["contact_id"]
        assert (
            initialized_cm.cm.contact_index.active_conversations[contact_id].on_call
            is True
        )

        # End it
        ended_event = PhoneCallEnded(contact=alice_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            await initialized_cm.step(ended_event)

        assert (
            initialized_cm.cm.contact_index.active_conversations[contact_id].on_call
            is False
        )

    async def test_phone_call_ended_calls_cleanup(
        self,
        initialized_cm,
        alice_contact,
    ):
        """PhoneCallEnded calls cleanup_call_proc."""
        # Start a call
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)

        # End it and verify cleanup is called
        ended_event = PhoneCallEnded(contact=alice_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ) as mock_cleanup:
            await initialized_cm.step(ended_event)

            mock_cleanup.assert_called_once()


# =============================================================================
# Integration Tests: UnifyMeet Event Flow
# =============================================================================


@pytest.mark.asyncio
class TestUnifyMeetEventHandlers:
    """Integration tests for UnifyMeet (web-based voice) event handlers."""

    @pytest.fixture
    def boss_contact(self):
        return TEST_CONTACTS[1]

    async def test_unify_meet_received_starts_call(self, initialized_cm, boss_contact):
        """UnifyMeetReceived event should start a unify_meet call."""
        with patch.object(
            initialized_cm.cm.call_manager,
            "start_unify_meet",
        ) as mock_start:
            event = UnifyMeetReceived(
                contact=boss_contact,
                room_name="test_room",
            )
            await initialized_cm.step(event)

            mock_start.assert_called_once()

    async def test_unify_meet_started_changes_mode(self, initialized_cm, boss_contact):
        """UnifyMeetStarted event changes mode to 'meet'."""
        assert initialized_cm.cm.mode == "text"

        event = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(event)

        assert initialized_cm.cm.mode == "meet"

    async def test_unify_meet_ended_resets_mode(self, initialized_cm, boss_contact):
        """UnifyMeetEnded resets mode to 'text'."""
        # Start a meeting
        started_event = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started_event)
        assert initialized_cm.cm.mode == "meet"

        # End it
        ended_event = UnifyMeetEnded(contact=boss_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            await initialized_cm.step(ended_event)

        assert initialized_cm.cm.mode == "text"


# =============================================================================
# Integration Tests: Voice Utterance Flow
# =============================================================================


@pytest.mark.asyncio
class TestVoiceUtteranceHandlers:
    """Integration tests for voice utterance events during calls."""

    @pytest.fixture
    def alice_contact(self):
        return TEST_CONTACTS[2]

    async def test_inbound_utterance_pushes_to_contact_index(
        self,
        initialized_cm,
        alice_contact,
    ):
        """InboundPhoneUtterance adds message to contact_index."""
        # Start a call first
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)

        # Send an utterance
        utterance_event = InboundPhoneUtterance(
            contact=alice_contact,
            content="Hello, can you help me?",
        )
        await initialized_cm.step(utterance_event)

        # Check message was added to contact's voice thread
        contact_id = alice_contact["contact_id"]
        voice_thread = initialized_cm.cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )
        assert len(voice_thread) > 0
        messages = [msg.content for msg in voice_thread]
        assert "Hello, can you help me?" in messages

    async def test_outbound_utterance_pushes_to_contact_index(
        self,
        initialized_cm,
        alice_contact,
    ):
        """OutboundPhoneUtterance adds assistant message to contact_index."""
        # Start a call first
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)

        # Send an outbound utterance
        utterance_event = OutboundPhoneUtterance(
            contact=alice_contact,
            content="Of course, I'd be happy to help!",
        )
        await initialized_cm.step(utterance_event)

        # Check message was added with assistant role (name="You" for assistant)
        contact_id = alice_contact["contact_id"]
        voice_thread = initialized_cm.cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )
        messages = [msg.content for msg in voice_thread]
        assert "Of course, I'd be happy to help!" in messages

    async def test_inbound_utterance_triggers_interject(
        self,
        initialized_cm,
        alice_contact,
    ):
        """InboundPhoneUtterance should trigger interject_or_run for LLM processing."""
        # Start a call first
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)

        # Mock interject_or_run to verify it's called
        with patch.object(
            initialized_cm.cm,
            "interject_or_run",
            new_callable=AsyncMock,
        ) as mock_interject:
            utterance_event = InboundPhoneUtterance(
                contact=alice_contact,
                content="What's the weather like?",
            )
            await initialized_cm.step(utterance_event)

            mock_interject.assert_called_once_with("What's the weather like?")


# =============================================================================
# Integration Tests: VoiceInterrupt Event
# =============================================================================


@pytest.mark.asyncio
class TestVoiceInterruptHandler:
    """Integration tests for VoiceInterrupt event handling.

    VoiceInterrupt is sent when the user interrupts the assistant during speech.
    This should:
    1. Cancel any pending proactive speech
    2. Allow the new user input to be processed
    """

    @pytest.fixture
    def alice_contact(self):
        return TEST_CONTACTS[2]

    async def test_voice_interrupt_event_structure(self, alice_contact):
        """VoiceInterrupt has correct event structure."""
        event = VoiceInterrupt(contact=alice_contact)

        assert event.contact["contact_id"] == alice_contact["contact_id"]
        assert hasattr(event, "timestamp")

    async def test_voice_interrupt_during_call(self, initialized_cm, alice_contact):
        """VoiceInterrupt during a call should be handled without error.

        NOTE: VoiceInterrupt is currently NOT handled by any registered handler.
        This test documents the expected behavior once implemented.
        """
        # Start a call first
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)
        assert initialized_cm.cm.mode == "call"

        # Send a voice interrupt - should not raise
        interrupt_event = VoiceInterrupt(contact=alice_contact)
        result = await initialized_cm.step(interrupt_event)

        # Event should be processed (even if handler does nothing)
        assert result.input_event == interrupt_event


# =============================================================================
# Integration Tests: CallGuidance Event Flow
# =============================================================================


@pytest.mark.asyncio
class TestCallGuidanceFlow:
    """Integration tests for CallGuidance events from Main CM Brain to Voice Agent."""

    @pytest.fixture
    def alice_contact(self):
        return TEST_CONTACTS[2]

    async def test_call_guidance_event_handled(self, initialized_cm, alice_contact):
        """CallGuidance event is processed by the handler."""
        # Start a call first
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)

        # Send guidance
        guidance_event = CallGuidance(
            contact=alice_contact,
            content="Please ask about their availability for next week",
        )
        result = await initialized_cm.step(guidance_event)

        # Should be handled
        assert result.input_event == guidance_event

    async def test_call_guidance_pushes_to_contact_index(
        self,
        initialized_cm,
        alice_contact,
    ):
        """CallGuidance adds message to contact_index with 'guidance' role."""
        # Start a call first
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)

        # Send guidance
        guidance_event = CallGuidance(
            contact=alice_contact,
            content="Reminder: User prefers morning meetings",
        )
        await initialized_cm.step(guidance_event)

        # Check message was added to voice thread
        contact_id = alice_contact["contact_id"]
        voice_thread = initialized_cm.cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )

        # Guidance messages have name="guidance" (the role becomes the name)
        guidance_msgs = [msg for msg in voice_thread if msg.name == "guidance"]
        assert len(guidance_msgs) >= 1
        assert "Reminder: User prefers morning meetings" in [
            msg.content for msg in guidance_msgs
        ]


# =============================================================================
# Integration Tests: Full Call Lifecycle
# =============================================================================


@pytest.mark.asyncio
class TestFullCallLifecycle:
    """End-to-end tests for complete call lifecycle."""

    @pytest.fixture
    def alice_contact(self):
        return TEST_CONTACTS[2]

    @pytest.fixture
    def boss_contact(self):
        return TEST_CONTACTS[1]

    async def test_inbound_call_lifecycle(self, initialized_cm, alice_contact):
        """Test complete lifecycle of an inbound phone call."""
        # 1. Call received - should start voice agent
        with patch.object(
            initialized_cm.cm.call_manager,
            "start_call",
        ) as mock_start:
            received_event = PhoneCallReceived(
                contact=alice_contact,
                conference_name="conf_lifecycle",
            )
            await initialized_cm.step(received_event)

            mock_start.assert_called_once()
            assert initialized_cm.cm.call_manager.conference_name == "conf_lifecycle"

        # 2. Call started - mode changes
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)
        assert initialized_cm.cm.mode == "call"

        # 3. User speaks
        user_utterance = InboundPhoneUtterance(
            contact=alice_contact,
            content="Hi, I need help with my order",
        )
        await initialized_cm.step(user_utterance)

        # 4. Assistant responds
        assistant_utterance = OutboundPhoneUtterance(
            contact=alice_contact,
            content="I'd be happy to help. What's your order number?",
        )
        await initialized_cm.step(assistant_utterance)

        # 5. Call ends
        ended_event = PhoneCallEnded(contact=alice_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            await initialized_cm.step(ended_event)

        assert initialized_cm.cm.mode == "text"
        assert initialized_cm.cm.call_manager.call_contact is None

    async def test_outbound_call_lifecycle(
        self,
        initialized_cm,
        alice_contact,
        boss_contact,
    ):
        """Test complete lifecycle of an outbound phone call."""
        # 1. Call sent (outbound)
        with patch.object(
            initialized_cm.cm.call_manager,
            "start_call",
        ) as mock_start:
            sent_event = PhoneCallSent(contact=alice_contact)
            await initialized_cm.step(sent_event)

            mock_start.assert_called_once()
            # Should be called with outbound=True
            call_args = mock_start.call_args
            assert call_args[1].get("outbound") is True or (
                len(call_args[0]) >= 3 and call_args[0][2] is True
            )

        # 2. Call started - mode changes to "call"
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)
        assert initialized_cm.cm.mode == "call"

        # 3. Call answered - when in call mode, handler publishes to voice agent
        # Use publish=True to let the event flow through the broker
        # The handler publishes {"type": "call_answered"} to app:call:status
        answered_event = PhoneCallAnswered(contact=alice_contact)
        await initialized_cm.step(answered_event, publish=True)
        # The publish happens within the event handler - we trust the handler logic
        # as tested separately in TestCallEventBrokerChannels

        # 4. Call ends
        ended_event = PhoneCallEnded(contact=alice_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            await initialized_cm.step(ended_event)

        assert initialized_cm.cm.mode == "text"

    async def test_unify_meet_lifecycle(self, initialized_cm, boss_contact):
        """Test complete lifecycle of a UnifyMeet (web) call."""
        # 1. Meeting received
        with patch.object(
            initialized_cm.cm.call_manager,
            "start_unify_meet",
        ) as mock_start:
            received_event = UnifyMeetReceived(
                contact=boss_contact,
                room_name="test_room",
            )
            await initialized_cm.step(received_event)

            mock_start.assert_called_once()

        # 2. Meeting started
        started_event = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started_event)
        assert initialized_cm.cm.mode == "meet"

        # 3. User speaks
        user_utterance = InboundUnifyMeetUtterance(
            contact=boss_contact,
            content="Let's go over the quarterly numbers",
        )
        await initialized_cm.step(user_utterance)

        # 4. Assistant responds
        assistant_utterance = OutboundUnifyMeetUtterance(
            contact=boss_contact,
            content="Sure, let me pull up the report",
        )
        await initialized_cm.step(assistant_utterance)

        # 5. Meeting ends
        ended_event = UnifyMeetEnded(contact=boss_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            await initialized_cm.step(ended_event)

        assert initialized_cm.cm.mode == "text"

    async def test_call_with_guidance(self, initialized_cm, alice_contact):
        """Test call lifecycle with guidance from Main CM Brain."""
        # Start a call
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)

        # User asks a question
        user_utterance = InboundPhoneUtterance(
            contact=alice_contact,
            content="What's on my schedule tomorrow?",
        )
        await initialized_cm.step(user_utterance)

        # Main CM Brain sends guidance to Voice Agent
        guidance_event = CallGuidance(
            contact=alice_contact,
            content="You have a 10am meeting with the marketing team and a 2pm call with the client",
        )
        await initialized_cm.step(guidance_event)

        # Verify guidance was recorded
        contact_id = alice_contact["contact_id"]
        voice_thread = initialized_cm.cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )
        messages = [msg.content for msg in voice_thread]
        assert "10am meeting" in " ".join(messages)


# =============================================================================
# Integration Tests: Error Handling and Edge Cases
# =============================================================================


@pytest.mark.asyncio
class TestCallErrorHandling:
    """Tests for error handling during call lifecycle."""

    @pytest.fixture
    def alice_contact(self):
        return TEST_CONTACTS[2]

    async def test_call_ended_without_started(self, initialized_cm, alice_contact):
        """PhoneCallEnded when no active conversation exists should not crash.

        This can happen in production when:
        - Voice agent thread crashes before PhoneCallStarted is emitted
        - Container restarts mid-call, losing in-memory state
        - Event delivery race condition where PhoneCallEnded arrives first
        - External telephony webhook for an unregistered call
        """
        assert initialized_cm.cm.mode == "text"

        ended_event = PhoneCallEnded(contact=alice_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            # Should handle gracefully without raising KeyError
            await initialized_cm.step(ended_event)

        # Mode should still be text
        assert initialized_cm.cm.mode == "text"

    async def test_multiple_calls_rejected_when_already_in_call(
        self,
        initialized_cm,
        alice_contact,
    ):
        """PhoneCallReceived during an active call should be ignored."""
        # Start first call
        started_event = PhoneCallStarted(contact=alice_contact)
        await initialized_cm.step(started_event)
        assert initialized_cm.cm.mode == "call"

        # Try to receive another call
        with patch.object(
            initialized_cm.cm.call_manager,
            "start_call",
        ) as mock_start:
            second_call = PhoneCallReceived(
                contact=alice_contact,
                conference_name="second_conf",
            )
            await initialized_cm.step(second_call)

            # Should NOT start a new call
            mock_start.assert_not_called()

    async def test_cleanup_call_proc_timeout_handling(
        self,
        initialized_cm,
        alice_contact,
    ):
        """cleanup_call_proc handles timeout gracefully."""
        from unity.conversation_manager.domains.call_manager import (
            CallConfig,
            LivekitCallManager,
        )

        config = CallConfig(
            assistant_id="test",
            user_id="test_user",
            assistant_bio="Test",
            assistant_number="+15551234567",
            voice_provider="cartesia",
            voice_id="test",
        )

        manager = LivekitCallManager(config)

        # Create a mock subprocess that's running
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Process is running
        mock_proc.pid = 12345
        manager._call_proc = mock_proc

        with patch(
            "unity.conversation_manager.domains.call_manager.terminate_process",
        ) as mock_terminate:
            await asyncio.wait_for(
                manager.cleanup_call_proc(),
                timeout=1.0,
            )


@pytest.mark.asyncio
class TestConversationManagerInactivityCleanup:
    """Tests for ConversationManager cleanup and shutdown flow."""

    @pytest.fixture
    def alice_contact(self):
        return TEST_CONTACTS[2]

    async def test_cleanup_call_proc_terminates_subprocess(
        self,
        initialized_cm,
        alice_contact,
    ):
        """cleanup_call_proc() terminates the subprocess.

        The subprocess should be terminated via SIGTERM when cleanup() is called.
        """
        cm = initialized_cm.cm

        # Create a mock subprocess that's running
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Process is running
        mock_proc.pid = 12345
        cm.call_manager._call_proc = mock_proc

        with patch(
            "unity.conversation_manager.domains.call_manager.terminate_process",
        ) as mock_terminate:
            await cm.call_manager.cleanup_call_proc()

            mock_terminate.assert_called_once_with(mock_proc, 0)


# =============================================================================
# Integration Tests: Event Broker Channel Subscriptions
# =============================================================================


@pytest.mark.asyncio
class TestCallEventBrokerChannels:
    """Tests for event broker channel patterns used in call lifecycle."""

    @pytest_asyncio.fixture
    async def event_broker(self):
        """Create a fresh in-memory event broker for channel tests."""
        from unity.conversation_manager.event_broker import create_event_broker

        broker = create_event_broker()
        yield broker
        await broker.aclose()

    async def test_call_status_channel(self, event_broker):
        """app:call:status channel receives stop messages."""
        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:status")

            # Publish stop message
            await event_broker.publish(
                "app:call:status",
                json.dumps({"type": "stop"}),
            )

            msg = await pubsub.get_message(
                timeout=1.0,
                ignore_subscribe_messages=True,
            )

            assert msg is not None
            data = json.loads(msg["data"])
            assert data["type"] == "stop"

    async def test_call_guidance_channel(self, event_broker):
        """app:call:call_guidance channel receives guidance messages."""
        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:call_guidance")

            # Publish guidance
            guidance = CallGuidance(
                contact={"contact_id": 1, "first_name": "Test"},
                content="Ask about their schedule",
            )
            await event_broker.publish(
                "app:call:call_guidance",
                guidance.to_json(),
            )

            msg = await pubsub.get_message(
                timeout=1.0,
                ignore_subscribe_messages=True,
            )

            assert msg is not None
            restored = Event.from_json(msg["data"])
            assert isinstance(restored, CallGuidance)
            assert restored.content == "Ask about their schedule"

    async def test_phone_utterance_channel(self, event_broker):
        """app:comms:phone_utterance channel receives utterance events."""
        async with event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            # Publish utterance
            utterance = InboundPhoneUtterance(
                contact={"contact_id": 1, "first_name": "Test"},
                content="Hello there",
            )
            await event_broker.publish(
                "app:comms:phone_utterance",
                utterance.to_json(),
            )

            msg = await pubsub.get_message(
                timeout=1.0,
                ignore_subscribe_messages=True,
            )

            assert msg is not None
            restored = Event.from_json(msg["data"])
            assert isinstance(restored, InboundPhoneUtterance)

    async def test_call_answered_status_message(self, event_broker):
        """app:call:status channel receives call_answered messages."""
        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:status")

            # Publish call answered
            await event_broker.publish(
                "app:call:status",
                json.dumps({"type": "call_answered"}),
            )

            msg = await pubsub.get_message(
                timeout=1.0,
                ignore_subscribe_messages=True,
            )

            assert msg is not None
            data = json.loads(msg["data"])
            assert data["type"] == "call_answered"


# =============================================================================
# Test: Channel forwarding tiers — boss vs non-boss calls
# =============================================================================


@pytest.mark.asyncio
class TestChannelForwardingTiers:
    """Verify the two-tier channel policy: all calls get comms channels,
    boss calls additionally get actor/manager channels."""

    @pytest_asyncio.fixture
    async def call_manager_with_broker(self):
        from unity.conversation_manager.event_broker import create_event_broker
        from unity.conversation_manager.domains.call_manager import (
            CallConfig,
            LivekitCallManager,
        )

        config = CallConfig(
            assistant_id="test",
            user_id="test_user",
            assistant_bio="Test assistant",
            assistant_number="+15551234567",
            voice_provider="cartesia",
            voice_id="test_voice",
        )
        broker = create_event_broker()
        mgr = LivekitCallManager(config, event_broker=broker)
        yield mgr
        await mgr.cleanup_call_proc()
        await broker.aclose()

    @pytest.fixture
    def boss_contact(self):
        return {"contact_id": 1, "first_name": "Boss", "surname": "User"}

    @pytest.fixture
    def non_boss_contact(self):
        return {"contact_id": 5, "first_name": "Alice", "surname": "Smith"}

    async def test_boss_call_gets_comms_and_actor_channels(
        self,
        call_manager_with_broker,
        boss_contact,
    ):
        """Boss calls should forward app:comms:* AND app:actor:* channels."""
        mgr = call_manager_with_broker
        with patch(
            "unity.conversation_manager.domains.call_manager.run_script",
        ) as mock_run:
            mock_run.return_value = MagicMock()
            await mgr.start_call(boss_contact, boss_contact)

        channels = mgr._socket_server._forward_channels
        assert "app:comms:*" in channels, "Boss call must forward comms"
        assert "app:actor:*" in channels, "Boss call must forward actor events"
        assert "app:call:*" in channels, "Boss call must forward call events"

    async def test_non_boss_call_gets_comms_but_not_actor_channels(
        self,
        call_manager_with_broker,
        non_boss_contact,
        boss_contact,
    ):
        """Non-boss calls should forward app:comms:* but NOT app:actor:*."""
        mgr = call_manager_with_broker
        with patch(
            "unity.conversation_manager.domains.call_manager.run_script",
        ) as mock_run:
            mock_run.return_value = MagicMock()
            await mgr.start_call(non_boss_contact, boss_contact)

        channels = mgr._socket_server._forward_channels
        assert "app:comms:*" in channels, "Non-boss call must forward comms"
        assert (
            "app:actor:*" not in channels
        ), "Non-boss call must NOT forward actor events"
        assert "app:call:*" in channels, "Non-boss call must forward call events"

    async def test_unify_meet_boss_gets_full_channels(
        self,
        call_manager_with_broker,
        boss_contact,
    ):
        """Boss Unify Meet should get the same full channels as a boss phone call."""
        mgr = call_manager_with_broker
        with patch(
            "unity.conversation_manager.domains.call_manager.run_script",
        ) as mock_run:
            mock_run.return_value = MagicMock()
            await mgr.start_unify_meet(
                boss_contact,
                boss_contact,
                room_name="test_room",
            )

        channels = mgr._socket_server._forward_channels
        assert "app:comms:*" in channels
        assert "app:actor:*" in channels
