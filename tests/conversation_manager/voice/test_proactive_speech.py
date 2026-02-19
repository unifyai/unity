"""
tests/conversation_manager/test_proactive_speech.py
=========================================================

Integration tests for the proactive speech system in ConversationManager.

Tests cover:
1. schedule_proactive_speech() - scheduling behavior, mode restrictions
2. cancel_proactive_speech() - cancellation logic, edge cases
3. _proactive_speech_loop() - main loop behavior, decision flow
4. ProactiveSpeech.decide() - LLM decision making with real model calls
5. Event handler integration - verifying handlers reset/cancel proactive speech
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field

import pytest

from unity.contact_manager.simulated import SimulatedContactManager
from unity.common.prompt_helpers import PromptParts
from unity.conversation_manager.domains.proactive_speech import (
    ProactiveDecision,
    ProactiveSpeech,
)
from unity.conversation_manager.types import Medium, Mode

# =============================================================================
# Mock Helpers
# =============================================================================


def _default_prompt_parts() -> PromptParts:
    pp = PromptParts()
    pp.add("You are a helpful assistant.")
    return pp


@dataclass
class MockBrainSpec:
    """Mock brain spec matching the real BrainSpec's PromptParts-typed system_prompt."""

    system_prompt: PromptParts = field(default_factory=_default_prompt_parts)


def _make_noop_coro():
    """Create a no-op coroutine that can be used with asyncio.create_task."""

    async def _noop():
        pass

    return _noop()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_session_logger():
    """Create a mock session logger."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.debug = MagicMock()
    logger.error = MagicMock()
    return logger


@pytest.fixture
def mock_event_broker():
    """Create a mock event broker."""
    broker = MagicMock()
    broker.publish = AsyncMock(return_value=0)
    return broker


@pytest.fixture
def sample_contacts():
    """Standard test contacts."""
    return [
        {
            "contact_id": 0,
            "first_name": "Test",
            "surname": "Assistant",
            "email_address": "assistant@test.com",
            "phone_number": "+15555551234",
        },
        {
            "contact_id": 1,
            "first_name": "Boss",
            "surname": "User",
            "email_address": "boss@test.com",
            "phone_number": "+15555551111",
        },
    ]


@pytest.fixture
def mock_cm(mock_session_logger, mock_event_broker, sample_contacts):
    """Create a mock ConversationManager with minimal state for proactive speech tests."""
    from unity.conversation_manager.domains.contact_index import ContactIndex

    cm = MagicMock()
    cm._session_logger = mock_session_logger
    cm.event_broker = mock_event_broker
    cm.mode = "call"  # Default to voice mode where proactive speech is active
    cm._proactive_speech_task = None
    cm.assistant_screen_share_active = False

    # Create SimulatedContactManager and populate with sample contacts
    contact_manager = SimulatedContactManager()
    for contact_data in sample_contacts:
        contact_manager.update_contact(
            contact_id=contact_data["contact_id"],
            first_name=contact_data.get("first_name"),
            surname=contact_data.get("surname"),
            email_address=contact_data.get("email_address"),
            phone_number=contact_data.get("phone_number"),
        )

    # Set up contact index with SimulatedContactManager
    cm.contact_index = ContactIndex()
    cm.contact_index.set_contact_manager(contact_manager)

    # Set up proactive speech instance
    cm.proactive_speech = ProactiveSpeech()

    # Mock get_active_contact to return boss contact
    cm.get_active_contact = MagicMock(return_value=sample_contacts[1])

    # Mock get_recent_voice_transcript
    cm.get_recent_voice_transcript = MagicMock(return_value=([], None))

    return cm


# =============================================================================
# 1. schedule_proactive_speech() Tests
# =============================================================================


@pytest.mark.asyncio
class TestScheduleProactiveSpeech:
    """Tests for the schedule_proactive_speech() method."""

    async def test_schedule_only_in_call_mode(self, mock_cm):
        """Proactive speech only schedules in 'call' mode."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        # Test call mode - should create task
        mock_cm.mode = Mode.CALL
        mock_cm.cancel_proactive_speech = AsyncMock()

        # The key issue: schedule_proactive_speech calls self._proactive_speech_loop()
        # which looks up on the mock instance. We need to set it on the mock directly.
        async def mock_loop(*args, **kwargs):
            await asyncio.sleep(100)  # Long sleep so task stays alive

        mock_cm._proactive_speech_loop = mock_loop

        await ConversationManager.schedule_proactive_speech(mock_cm)

        # Should have called cancel first and created a task
        mock_cm.cancel_proactive_speech.assert_called_once()
        assert mock_cm._proactive_speech_task is not None

        # Clean up the task
        mock_cm._proactive_speech_task.cancel()

    async def test_schedule_only_in_unify_meet_mode(self, mock_cm):
        """Proactive speech schedules in 'meet' mode."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = Mode.MEET
        mock_cm.cancel_proactive_speech = AsyncMock()

        async def mock_loop(*args, **kwargs):
            await asyncio.sleep(100)

        mock_cm._proactive_speech_loop = mock_loop

        await ConversationManager.schedule_proactive_speech(mock_cm)

        mock_cm.cancel_proactive_speech.assert_called_once()
        assert mock_cm._proactive_speech_task is not None

        # Clean up
        mock_cm._proactive_speech_task.cancel()

    async def test_schedule_skipped_in_text_mode(self, mock_cm):
        """Proactive speech does NOT schedule in 'text' mode."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = Mode.TEXT
        mock_cm.cancel_proactive_speech = AsyncMock()

        await ConversationManager.schedule_proactive_speech(mock_cm)

        # Should have called cancel (to clean up any existing task) but NOT created new task
        mock_cm.cancel_proactive_speech.assert_called_once()
        # Task should not be created for text mode
        assert mock_cm._proactive_speech_task is None

    async def test_schedule_cancels_existing_task(self, mock_cm):
        """schedule_proactive_speech cancels any existing task first."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        # Create a fake existing task
        existing_task = MagicMock()
        existing_task.done = MagicMock(return_value=False)
        existing_task.cancel = MagicMock()
        mock_cm._proactive_speech_task = existing_task
        mock_cm.cancel_proactive_speech = AsyncMock()

        mock_cm.mode = Mode.CALL

        async def mock_loop(*args, **kwargs):
            await asyncio.sleep(100)

        mock_cm._proactive_speech_loop = mock_loop

        await ConversationManager.schedule_proactive_speech(mock_cm)

        mock_cm.cancel_proactive_speech.assert_called_once()

        # Clean up
        if mock_cm._proactive_speech_task:
            mock_cm._proactive_speech_task.cancel()


# =============================================================================
# 4. cancel_proactive_speech() Tests
# =============================================================================


@pytest.mark.asyncio
class TestCancelProactiveSpeech:
    """Tests for the cancel_proactive_speech() method."""

    async def test_cancel_does_nothing_if_no_task(self, mock_cm):
        """cancel_proactive_speech is a no-op if no task exists."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm._proactive_speech_task = None

        # Should not raise
        await ConversationManager.cancel_proactive_speech(mock_cm)

        assert mock_cm._proactive_speech_task is None

    async def test_cancel_does_nothing_if_task_done(self, mock_cm):
        """cancel_proactive_speech is a no-op if task is already done."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        done_task = MagicMock()
        done_task.done = MagicMock(return_value=True)
        mock_cm._proactive_speech_task = done_task

        await ConversationManager.cancel_proactive_speech(mock_cm)

        # Should not have called cancel on done task
        done_task.cancel.assert_not_called()

    async def test_cancel_cancels_running_task(self, mock_cm):
        """cancel_proactive_speech cancels a running task."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        # Create a real async task that we can cancel
        async def slow_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(slow_task())
        mock_cm._proactive_speech_task = task

        await ConversationManager.cancel_proactive_speech(mock_cm)

        assert task.cancelled() or task.done()
        assert mock_cm._proactive_speech_task is None


# =============================================================================
# 3. _proactive_speech_loop() Tests
# =============================================================================


@pytest.mark.asyncio
class TestProactiveSpeechLoop:
    """Tests for the _proactive_speech_loop() method."""

    async def test_loop_goes_dormant_when_should_not_speak(self, mock_cm):
        """When the LLM decides should_speak=False, the loop exits without
        rescheduling -- it goes dormant until the next utterance event."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = Mode.CALL

        async def mock_decide(*args, **kwargs):
            return ProactiveDecision(should_speak=False)

        mock_cm.proactive_speech.decide = mock_decide
        mock_cm.schedule_proactive_speech = AsyncMock()

        with (
            patch("asyncio.sleep", new=AsyncMock()),
            patch(
                "unity.conversation_manager.conversation_manager.build_brain_spec",
                return_value=MockBrainSpec(),
            ),
        ):
            await ConversationManager._proactive_speech_loop(mock_cm)

        # Loop should NOT have published any guidance
        mock_cm.event_broker.publish.assert_not_called()
        # Loop should NOT have self-rescheduled
        mock_cm.schedule_proactive_speech.assert_not_called()

    async def test_loop_publishes_guidance_when_should_speak(self, mock_cm):
        """The loop publishes a CallGuidance event with should_speak=True and
        response_text so the fast brain speaks it via session.say()."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = Mode.CALL

        async def mock_decide(*args, **kwargs):
            return ProactiveDecision(
                should_speak=True,
                delay=0,
                content="Still with you!",
            )

        mock_cm.proactive_speech.decide = mock_decide

        with (
            patch("asyncio.sleep", new=AsyncMock()),
            patch(
                "unity.conversation_manager.conversation_manager.build_brain_spec",
                return_value=MockBrainSpec(),
            ),
        ):
            await ConversationManager._proactive_speech_loop(mock_cm)

        # Should have published to call_guidance channel
        mock_cm.event_broker.publish.assert_called()
        call_args = mock_cm.event_broker.publish.call_args_list

        # Find the call_guidance publish
        guidance_call = None
        for call in call_args:
            if "app:call:call_guidance" in str(call):
                guidance_call = call
                break

        assert guidance_call is not None
        channel, message = guidance_call.args
        assert channel == "app:call:call_guidance"

        # Proactive speech publishes a CallGuidance event (not raw JSON)
        data = json.loads(message)
        payload = data["payload"]
        assert payload["content"] == "Still with you!"
        assert payload["response_text"] == "Still with you!"
        assert payload["should_speak"] is True

    async def test_loop_records_message_in_contact_index(self, mock_cm):
        """The loop records the proactive message in contact_index."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = Mode.CALL

        async def mock_decide(*args, **kwargs):
            return ProactiveDecision(
                should_speak=True,
                delay=0,
                content="Are you still there?",
            )

        mock_cm.proactive_speech.decide = mock_decide

        with (
            patch("asyncio.sleep", new=AsyncMock()),
            patch(
                "unity.conversation_manager.conversation_manager.build_brain_spec",
                return_value=MockBrainSpec(),
            ),
        ):
            await ConversationManager._proactive_speech_loop(mock_cm)

        # Should have recorded the message
        contact = mock_cm.get_active_contact()
        contact_id = contact["contact_id"]

        voice_thread = mock_cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )
        proactive_msg = None
        for msg in voice_thread:
            if "Are you still there?" in (msg.content or ""):
                proactive_msg = msg
                break

        assert proactive_msg is not None


# =============================================================================
# 4. ProactiveSpeech.decide() Integration Tests (Real LLM)
# =============================================================================


@pytest.mark.asyncio
class TestProactiveSpeechDecideIntegration:
    """Integration tests for ProactiveSpeech.decide() with real LLM calls."""

    async def test_decide_returns_valid_decision(self):
        """decide() returns a valid ProactiveDecision."""
        ps = ProactiveSpeech()

        chat_history = [
            {"role": "user", "content": "Can you help me with something?"},
            {"role": "assistant", "content": "Of course! What do you need help with?"},
        ]
        system_prompt = "You are a helpful assistant."

        decision = await ps.decide(
            chat_history=chat_history,
            system_prompt=system_prompt,
        )

        assert isinstance(decision, ProactiveDecision)
        assert isinstance(decision.should_speak, bool)
        assert isinstance(decision.delay, int)

    async def test_decide_does_not_speak_after_question(self):
        """decide() should NOT speak when assistant just asked a question."""
        ps = ProactiveSpeech()

        chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ]
        system_prompt = "You are a helpful assistant."

        decision = await ps.decide(
            chat_history=chat_history,
            system_prompt=system_prompt,
        )

        # The assistant just asked a question; the user is likely thinking.
        assert decision.should_speak is False

    async def test_decide_handles_empty_history(self):
        """decide() handles empty chat history gracefully."""
        ps = ProactiveSpeech()

        decision = await ps.decide(
            chat_history=[],
            system_prompt="You are a helpful assistant.",
        )

        assert isinstance(decision, ProactiveDecision)

    async def test_decide_returns_safe_default_on_error(self):
        """decide() returns should_speak=False when the LLM call fails."""
        ps = ProactiveSpeech()

        # Force an error by patching new_llm_client to raise
        with patch(
            "unity.conversation_manager.domains.proactive_speech.new_llm_client",
            side_effect=RuntimeError("connection failed"),
        ):
            decision = await ps.decide(
                chat_history=[{"role": "user", "content": "Hello"}],
                system_prompt="Test",
            )

        assert decision.should_speak is False


# =============================================================================
# 7. Event Handler Integration Tests
# =============================================================================


@pytest.mark.asyncio
class TestEventHandlerProactiveSpeechIntegration:
    """Tests verifying event handlers properly reset/cancel proactive speech."""

    async def test_call_guidance_should_speak_resets_proactive(self, mock_cm):
        """CallGuidance with should_speak=True resets (reschedules) proactive speech.

        Regression test for a coordination gap: when slow-brain guidance is
        dispatched with should_speak=True, it will be spoken via session.say()
        and can take many seconds of TTS playback. During that window, the
        proactive speech timer (reset only by OutboundUtterance events) can
        fire and produce stale filler that contradicts the just-delivered
        content.

        The fix: the CallGuidance handler must reset the proactive timer when
        should_speak=True, preventing stale proactive speech from queueing
        during TTS playback of substantive guidance.
        """
        from unity.conversation_manager.events import CallGuidance
        from unity.conversation_manager.domains.event_handlers import EventHandler

        mock_cm.schedule_proactive_speech = AsyncMock()

        event = CallGuidance(
            contact={"contact_id": 1, "first_name": "Boss", "surname": "User"},
            content="I found several backend engineer openings at OpenAI.",
            response_text="I found several backend engineer openings at OpenAI.",
            should_speak=True,
            source="slow_brain",
        )

        await EventHandler.handle_event(event, mock_cm, is_voice_call=False)

        mock_cm.schedule_proactive_speech.assert_called_once()

    async def test_call_guidance_notify_only_does_not_reset_proactive(self, mock_cm):
        """CallGuidance with should_speak=False does NOT reset proactive speech.

        Notify-only guidance is silently injected into the fast brain's context.
        The user is still in silence, so proactive speech may still be warranted.
        """
        from unity.conversation_manager.events import CallGuidance
        from unity.conversation_manager.domains.event_handlers import EventHandler

        mock_cm.schedule_proactive_speech = AsyncMock()

        event = CallGuidance(
            contact={"contact_id": 1, "first_name": "Boss", "surname": "User"},
            content="Checking your contacts for Bob.",
            should_speak=False,
            source="slow_brain",
        )

        await EventHandler.handle_event(event, mock_cm, is_voice_call=False)

        mock_cm.schedule_proactive_speech.assert_not_called()

    async def test_inbound_phone_utterance_resets_proactive(self, mock_cm):
        """InboundPhoneUtterance event resets (reschedules) proactive speech."""
        from unity.conversation_manager.events import InboundPhoneUtterance
        from unity.conversation_manager.domains.event_handlers import EventHandler

        mock_cm.schedule_proactive_speech = AsyncMock()
        mock_cm.interject_or_run = AsyncMock()

        event = InboundPhoneUtterance(
            contact={"contact_id": 1, "first_name": "Boss", "surname": "User"},
            content="Hello?",
        )

        await EventHandler.handle_event(event, mock_cm, is_voice_call=False)

        mock_cm.schedule_proactive_speech.assert_called_once()

    async def test_outbound_phone_utterance_resets_proactive(self, mock_cm):
        """OutboundPhoneUtterance event resets (reschedules) proactive speech.

        This is the indirect path that restarts the cycle after the fast brain
        speaks proactive content via TTS.
        """
        from unity.conversation_manager.events import OutboundPhoneUtterance
        from unity.conversation_manager.domains.event_handlers import EventHandler

        mock_cm.schedule_proactive_speech = AsyncMock()

        event = OutboundPhoneUtterance(
            contact={"contact_id": 1, "first_name": "Boss", "surname": "User"},
            content="Sure, give me a moment.",
        )

        await EventHandler.handle_event(event, mock_cm, is_voice_call=False)

        mock_cm.schedule_proactive_speech.assert_called_once()

    async def test_phone_call_ended_cancels_proactive(self, mock_cm):
        """PhoneCallEnded event cancels proactive speech."""
        from unity.conversation_manager.events import PhoneCallEnded
        from unity.conversation_manager.domains.event_handlers import EventHandler

        mock_cm.cancel_proactive_speech = AsyncMock()
        mock_cm.request_llm_run = AsyncMock()
        mock_cm.call_manager.cleanup_call_proc = AsyncMock()
        mock_cm.call_manager.call_contact = None
        mock_cm.call_manager.conference_name = None

        # Set up an active conversation for the contact (required by the handler)
        mock_cm.contact_index.push_message(
            contact_id=1,
            sender_name="System",
            thread_name=Medium.PHONE_CALL,
            message_content="<Call Started>",
            role="system",
        )
        # Set on_call flag
        mock_cm.contact_index.active_conversations[1].on_call = True

        event = PhoneCallEnded(
            contact={
                "contact_id": 1,
                "first_name": "Boss",
                "surname": "User",
                "phone_number": "+15555551111",
            },
        )

        await EventHandler.handle_event(event, mock_cm, is_voice_call=False)

        mock_cm.cancel_proactive_speech.assert_called_once()

    async def test_unify_meet_ended_cancels_proactive(self, mock_cm):
        """UnifyMeetEnded event cancels proactive speech."""
        from unity.conversation_manager.events import UnifyMeetEnded
        from unity.conversation_manager.domains.event_handlers import EventHandler

        mock_cm.cancel_proactive_speech = AsyncMock()
        mock_cm.request_llm_run = AsyncMock()
        mock_cm.call_manager.cleanup_call_proc = AsyncMock()
        mock_cm.call_manager.call_contact = None

        # Set up an active conversation for the contact (required by the handler)
        mock_cm.contact_index.push_message(
            contact_id=1,
            sender_name="System",
            thread_name=Medium.UNIFY_MEET,
            message_content="<Call Started>",
            role="system",
        )
        # Set on_call flag
        mock_cm.contact_index.active_conversations[1].on_call = True

        event = UnifyMeetEnded(
            contact={"contact_id": 1, "first_name": "Boss", "surname": "User"},
        )

        await EventHandler.handle_event(event, mock_cm, is_voice_call=False)

        mock_cm.cancel_proactive_speech.assert_called_once()

    async def test_sms_received_cancels_proactive(self, mock_cm):
        """SMSReceived event cancels proactive speech."""
        from unity.conversation_manager.events import SMSReceived
        from unity.conversation_manager.domains.event_handlers import EventHandler
        from unity.conversation_manager.domains import managers_utils

        mock_cm.cancel_proactive_speech = AsyncMock()
        mock_cm.request_llm_run = AsyncMock()

        # Mock managers_utils to avoid async queue issues
        with patch.object(managers_utils, "queue_operation", new=AsyncMock()):
            event = SMSReceived(
                contact={
                    "contact_id": 1,
                    "first_name": "Boss",
                    "surname": "User",
                    "phone_number": "+15555551111",
                },
                content="Hey there!",
            )

            await EventHandler.handle_event(event, mock_cm, is_voice_call=False)

        mock_cm.cancel_proactive_speech.assert_called_once()


# =============================================================================
# 8. End-to-End Integration Tests (using initialized_cm fixture)
# =============================================================================


@pytest.mark.asyncio
class TestProactiveSpeechE2E:
    """End-to-end tests using the real ConversationManager fixture."""

    async def test_proactive_speech_not_scheduled_in_text_mode(
        self,
        initialized_cm,
    ):
        """In text mode, proactive speech should not be scheduled."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        cm = initialized_cm.cm

        # Restore the real schedule_proactive_speech (test fixtures mock it to no-op)
        cm.schedule_proactive_speech = (
            lambda: ConversationManager.schedule_proactive_speech(cm)
        )

        # Ensure we're in text mode
        assert cm.mode == "text"

        # Schedule should be a no-op in text mode
        await cm.schedule_proactive_speech()

        # Task should not exist (text mode doesn't schedule)
        assert cm._proactive_speech_task is None

    async def test_proactive_speech_scheduled_in_call_mode(
        self,
        initialized_cm,
    ):
        """In call mode, proactive speech should be scheduled."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        cm = initialized_cm.cm

        # Restore the real schedule_proactive_speech (test fixtures mock it to no-op)
        cm.schedule_proactive_speech = (
            lambda: ConversationManager.schedule_proactive_speech(cm)
        )

        # Switch to call mode
        cm.mode = Mode.CALL

        # Schedule proactive speech
        await cm.schedule_proactive_speech()

        # Task should be created
        assert cm._proactive_speech_task is not None

        # Clean up
        await cm.cancel_proactive_speech()

    async def test_cancel_proactive_speech_clears_task(
        self,
        initialized_cm,
    ):
        """cancel_proactive_speech should clear the task."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        cm = initialized_cm.cm

        # Restore the real schedule_proactive_speech (test fixtures mock it to no-op)
        cm.schedule_proactive_speech = (
            lambda: ConversationManager.schedule_proactive_speech(cm)
        )

        # Switch to call mode and schedule
        cm.mode = Mode.CALL
        await cm.schedule_proactive_speech()

        assert cm._proactive_speech_task is not None

        # Cancel
        await cm.cancel_proactive_speech()

        # Task should be cleared
        assert cm._proactive_speech_task is None


# =============================================================================
# 9. Additional Blind Spot Tests
# =============================================================================


@pytest.mark.asyncio
class TestProactiveSpeechBlindSpots:
    """Tests for previously untested edge cases and blind spots."""

    # -------------------------------------------------------------------------
    # Test: Medium.UNIFY_MEET in the loop
    # -------------------------------------------------------------------------

    async def test_loop_records_message_with_unify_meet_medium(self, mock_cm):
        """In MEET mode, proactive messages should use Medium.UNIFY_MEET."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = Mode.MEET  # Key: test MEET mode specifically

        async def mock_decide(*args, **kwargs):
            return ProactiveDecision(
                should_speak=True,
                delay=0,
                content="Still here for the meeting!",
            )

        mock_cm.proactive_speech.decide = mock_decide

        with (
            patch("asyncio.sleep", new=AsyncMock()),
            patch(
                "unity.conversation_manager.conversation_manager.build_brain_spec",
                return_value=MockBrainSpec(),
            ),
        ):
            await ConversationManager._proactive_speech_loop(mock_cm)

        # Verify the message was recorded with UNIFY_MEET medium
        contact = mock_cm.get_active_contact()
        contact_id = contact["contact_id"]

        # Should use UNIFY_MEET, not PHONE_CALL
        meet_thread = mock_cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.UNIFY_MEET,
        )
        phone_thread = mock_cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )

        # Find the proactive message in the meet thread
        proactive_msg = None
        for msg in meet_thread:
            if "Still here for the meeting!" in (msg.content or ""):
                proactive_msg = msg
                break

        assert proactive_msg is not None, (
            f"Expected proactive message in UNIFY_MEET thread. "
            f"Meet thread: {meet_thread}, Phone thread: {phone_thread}"
        )

    # -------------------------------------------------------------------------
    # Test: InboundUnifyMeetUtterance cancels proactive speech
    # -------------------------------------------------------------------------

    async def test_inbound_unify_meet_utterance_resets_proactive(self, mock_cm):
        """InboundUnifyMeetUtterance event resets (reschedules) proactive speech."""
        from unity.conversation_manager.events import InboundUnifyMeetUtterance
        from unity.conversation_manager.domains.event_handlers import EventHandler

        mock_cm.schedule_proactive_speech = AsyncMock()
        mock_cm.interject_or_run = AsyncMock()

        event = InboundUnifyMeetUtterance(
            contact={"contact_id": 1, "first_name": "Boss", "surname": "User"},
            content="Hello from the meeting!",
        )

        await EventHandler.handle_event(event, mock_cm, is_voice_call=False)

        mock_cm.schedule_proactive_speech.assert_called_once()

    # -------------------------------------------------------------------------
    # Test: EmailReceived cancels proactive speech
    # -------------------------------------------------------------------------

    async def test_email_received_cancels_proactive(self, mock_cm):
        """EmailReceived event cancels proactive speech."""
        from unity.conversation_manager.events import EmailReceived
        from unity.conversation_manager.domains.event_handlers import EventHandler
        from unity.conversation_manager.domains import managers_utils

        mock_cm.cancel_proactive_speech = AsyncMock()
        mock_cm.request_llm_run = AsyncMock()

        # Mock managers_utils to avoid async queue issues
        with patch.object(managers_utils, "queue_operation", new=AsyncMock()):
            event = EmailReceived(
                contact={
                    "contact_id": 1,
                    "first_name": "Boss",
                    "surname": "User",
                    "email_address": "boss@test.com",
                },
                subject="Test Subject",
                body="Test email body",
                email_id="email_123",
            )

            await EventHandler.handle_event(event, mock_cm, is_voice_call=False)

        mock_cm.cancel_proactive_speech.assert_called_once()

    # -------------------------------------------------------------------------
    # Test: Recursion protection in cancel
    # -------------------------------------------------------------------------

    async def test_cancel_does_not_cancel_self_from_inside_task(self, mock_cm):
        """cancel_proactive_speech should not cancel if called from inside the task."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        cancel_actually_called = False

        async def inner_test():
            nonlocal cancel_actually_called
            # Set the current task as the proactive speech task
            mock_cm._proactive_speech_task = asyncio.current_task()

            # Now call cancel - it should detect we're inside the task and not cancel
            await ConversationManager.cancel_proactive_speech(mock_cm)

            # If we reach here without the task being cancelled, the recursion
            # protection worked
            cancel_actually_called = True

        task = asyncio.create_task(inner_test())
        await task

        # The task should have completed normally (not been cancelled)
        assert cancel_actually_called
        assert not task.cancelled()

    # -------------------------------------------------------------------------
    # Test: Loop does not self-reschedule after speaking
    # -------------------------------------------------------------------------

    async def test_loop_does_not_self_reschedule(self, mock_cm):
        """After speaking, the loop exits without calling schedule_proactive_speech.

        The cycle restarts via the indirect path: the fast brain speaks the
        guidance, producing an OutboundUtterance event whose handler calls
        schedule_proactive_speech().
        """
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = Mode.CALL

        async def mock_decide(*args, **kwargs):
            return ProactiveDecision(
                should_speak=True,
                delay=0,
                content="Still with you!",
            )

        mock_cm.proactive_speech.decide = mock_decide
        mock_cm.schedule_proactive_speech = AsyncMock()

        with (
            patch("asyncio.sleep", new=AsyncMock()),
            patch(
                "unity.conversation_manager.conversation_manager.build_brain_spec",
                return_value=MockBrainSpec(),
            ),
        ):
            await ConversationManager._proactive_speech_loop(mock_cm)

        mock_cm.schedule_proactive_speech.assert_not_called()


@pytest.mark.asyncio
class TestProactiveSpeechLLMBehavior:
    """Integration tests for LLM decision behavior with specific scenarios."""

    # -------------------------------------------------------------------------
    # Test: User asked to wait - LLM should respect longer threshold
    # -------------------------------------------------------------------------

    async def test_decide_respects_user_asked_to_wait(self):
        """When user explicitly asked to wait, LLM should NOT speak."""
        ps = ProactiveSpeech()

        chat_history = [
            {"role": "assistant", "content": "How can I help you today?"},
            {"role": "user", "content": "Hold on a moment, I need to find something."},
        ]
        system_prompt = "You are a helpful assistant."

        decision = await ps.decide(
            chat_history=chat_history,
            system_prompt=system_prompt,
        )

        assert (
            decision.should_speak is False
        ), f"User asked to wait - should NOT speak. Decision: {decision}"

    # -------------------------------------------------------------------------
    # Test: Previous proactive messages - LLM should vary responses
    # -------------------------------------------------------------------------

    async def test_decide_varies_content_from_previous_proactive(self):
        """LLM should generate different content from previous proactive messages."""
        ps = ProactiveSpeech()

        chat_history = [
            {"role": "assistant", "content": "How can I help you today?"},
            {"role": "user", "content": "I need to check something, one moment."},
            {"role": "assistant", "content": "Still with you, take your time."},
        ]
        system_prompt = "You are a helpful assistant."

        decision = await ps.decide(
            chat_history=chat_history,
            system_prompt=system_prompt,
        )

        if decision.should_speak and decision.content:
            assert (
                decision.content.lower() != "still with you, take your time."
            ), f"LLM should vary content. Got same as before: {decision.content}"

    # -------------------------------------------------------------------------
    # Test: Conversation is closing - should not speak
    # -------------------------------------------------------------------------

    async def test_decide_does_not_speak_during_goodbye(self):
        """LLM should not speak proactively when conversation is closing."""
        ps = ProactiveSpeech()

        chat_history = [
            {"role": "user", "content": "Thanks for your help, goodbye!"},
            {"role": "assistant", "content": "You're welcome! Have a great day!"},
        ]
        system_prompt = "You are a helpful assistant."

        decision = await ps.decide(
            chat_history=chat_history,
            system_prompt=system_prompt,
        )

        assert (
            decision.should_speak is False
        ), f"Should NOT speak during goodbye. Decision: {decision}"


@pytest.mark.asyncio
class TestProactiveSpeechMediumScriptIntegration:
    """Tests verifying medium scripts properly handle proactive speech."""

    def test_tts_call_subscribes_to_call_guidance(self):
        """call.py (TTS mode) should subscribe to app:call:call_guidance."""
        import inspect
        from unity.conversation_manager.medium_scripts import call

        # Read the source to verify subscription
        source = inspect.getsource(call)
        assert (
            "app:call:call_guidance" in source
        ), "call.py (TTS mode) should subscribe to app:call:call_guidance"
        assert "on_guidance" in source, "call.py should have an on_guidance callback"

    def test_sts_call_subscribes_to_call_guidance(self):
        """sts_call.py (STS/fast mode) should subscribe to app:call:call_guidance."""
        import inspect
        from unity.conversation_manager.medium_scripts import sts_call

        # Read the source to verify subscription
        source = inspect.getsource(sts_call)
        assert (
            "app:call:call_guidance" in source
        ), "sts_call.py (STS mode) should subscribe to app:call:call_guidance"
        assert (
            "on_guidance" in source
        ), "sts_call.py should have an on_guidance callback"

    def test_tts_and_sts_guidance_handlers_have_same_interface(self):
        """Both TTS and STS guidance handlers should handle the same payload structure."""
        import inspect
        from unity.conversation_manager.medium_scripts import call, sts_call

        # Both should extract content from payload
        tts_source = inspect.getsource(call)
        sts_source = inspect.getsource(sts_call)

        # Both should handle payload.get("content")
        assert (
            'payload.get("content"' in tts_source
            or "payload.get('content'" in tts_source
        ), "TTS call.py should extract content from payload"
        assert (
            'payload.get("content"' in sts_source
            or "payload.get('content'" in sts_source
        ), "STS sts_call.py should extract content from payload"
