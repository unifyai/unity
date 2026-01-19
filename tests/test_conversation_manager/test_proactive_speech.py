"""
tests/test_conversation_manager/test_proactive_speech.py
=========================================================

Integration tests for the proactive speech system in ConversationManager.

Tests cover:
1. schedule_proactive_speech() - scheduling behavior, mode restrictions
2. cancel_proactive_speech() - cancellation logic, edge cases
3. _proactive_speech_loop() - main loop behavior, decision flow, adaptive wait
4. ProactiveSpeech.decide() - LLM decision making with real model calls
5. Event handler integration - verifying handlers cancel proactive speech appropriately

## Test Categories

### Unit Tests (mocked dependencies)
- Schedule/cancel mechanics
- Mode restrictions
- Task lifecycle management

### Integration Tests (real LLM, mocked timers)
- Decision making with conversation context
- Adaptive wait logic
- Event broker publishing

### Eval Tests (real LLM)
- Decision quality with realistic transcripts
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

import pytest

from unity.conversation_manager.domains.proactive_speech import (
    ProactiveDecision,
    ProactiveSpeech,
)


# =============================================================================
# Mock Helpers
# =============================================================================


@dataclass
class MockBrainSpec:
    """Mock brain spec for testing."""

    system_prompt: str = "You are a helpful assistant."


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

    # Set up contact index with sample contacts
    cm.contact_index = ContactIndex()
    cm.contact_index.set_contacts(sample_contacts)

    # Set up proactive speech instance
    cm.proactive_speech = ProactiveSpeech()

    # Mock get_active_contact
    cm.get_active_contact = MagicMock(
        return_value=cm.contact_index.get_contact(contact_id=1),
    )

    # Mock get_recent_voice_transcript
    cm.get_recent_voice_transcript = MagicMock(return_value=([], None))

    return cm


# =============================================================================
# 1. ProactiveDecision Model Tests
# =============================================================================


class TestProactiveDecisionModel:
    """Tests for the ProactiveDecision Pydantic model."""

    def test_decision_with_all_fields(self):
        """ProactiveDecision accepts all fields."""
        decision = ProactiveDecision(
            should_speak=True,
            delay=3,
            content="Still working on that for you.",
        )
        assert decision.should_speak is True
        assert decision.delay == 3
        assert decision.content == "Still working on that for you."

    def test_decision_with_defaults(self):
        """ProactiveDecision has sensible defaults."""
        decision = ProactiveDecision(should_speak=False)
        assert decision.should_speak is False
        assert decision.delay == 5  # Default delay
        assert decision.content is None

    def test_decision_serialization(self):
        """ProactiveDecision can be serialized to JSON."""
        decision = ProactiveDecision(
            should_speak=True,
            delay=2,
            content="Hello?",
        )
        json_str = decision.model_dump_json()
        data = json.loads(json_str)

        assert data["should_speak"] is True
        assert data["delay"] == 2
        assert data["content"] == "Hello?"

    def test_decision_deserialization(self):
        """ProactiveDecision can be deserialized from JSON."""
        json_str = '{"should_speak": true, "delay": 4, "content": "Are you there?"}'
        decision = ProactiveDecision.model_validate_json(json_str)

        assert decision.should_speak is True
        assert decision.delay == 4
        assert decision.content == "Are you there?"


# =============================================================================
# 2. ProactiveSpeech Class Tests
# =============================================================================


class TestProactiveSpeechClass:
    """Tests for the ProactiveSpeech decision-making class."""

    def test_default_model_is_fast(self):
        """ProactiveSpeech uses a fast model for low-latency decisions."""
        ps = ProactiveSpeech()
        # Should use a flash/mini model for quick responses
        assert "flash" in ps.model.lower() or "mini" in ps.model.lower()

    def test_custom_model_accepted(self):
        """ProactiveSpeech accepts a custom model."""
        ps = ProactiveSpeech(model="gpt-5-mini@openai")
        assert ps.model == "gpt-5-mini@openai"


# =============================================================================
# 3. schedule_proactive_speech() Tests
# =============================================================================


@pytest.mark.asyncio
class TestScheduleProactiveSpeech:
    """Tests for the schedule_proactive_speech() method."""

    async def test_schedule_only_in_call_mode(self, mock_cm):
        """Proactive speech only schedules in 'call' mode."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        # Test call mode - should create task
        mock_cm.mode = "call"
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
        """Proactive speech schedules in 'unify_meet' mode."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = "unify_meet"
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

        mock_cm.mode = "text"
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

        mock_cm.mode = "call"

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
# 5. _proactive_speech_loop() Tests
# =============================================================================


@pytest.mark.asyncio
class TestProactiveSpeechLoop:
    """Tests for the _proactive_speech_loop() method."""

    async def test_loop_waits_initial_10_seconds(self, mock_cm):
        """The loop waits 10 seconds initially before checking."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = "call"

        async def mock_decide(*args, **kwargs):
            return ProactiveDecision(should_speak=False)

        mock_cm.proactive_speech.decide = mock_decide
        mock_cm.schedule_proactive_speech = AsyncMock()
        mock_cm.schedule_proactive_speech.side_effect = asyncio.CancelledError()

        sleep_calls = []

        async def track_sleep(duration):
            sleep_calls.append(duration)

        with (
            patch("asyncio.sleep", new=track_sleep),
            patch(
                "unity.conversation_manager.conversation_manager.build_brain_spec",
                return_value=MockBrainSpec(),
            ),
        ):
            try:
                await ConversationManager._proactive_speech_loop(
                    mock_cm,
                    skip_initial_wait=False,
                )
            except asyncio.CancelledError:
                pass

        # Should have waited 10 seconds initially
        assert 10 in sleep_calls

    async def test_loop_skips_initial_wait_when_requested(self, mock_cm):
        """The loop skips initial wait when skip_initial_wait=True."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = "call"

        async def mock_decide(*args, **kwargs):
            return ProactiveDecision(should_speak=False)

        mock_cm.proactive_speech.decide = mock_decide
        mock_cm.schedule_proactive_speech = AsyncMock()

        sleep_calls = []

        async def track_sleep(duration):
            sleep_calls.append(duration)
            if len(sleep_calls) >= 2:
                raise asyncio.CancelledError()

        with (
            patch("asyncio.sleep", new=track_sleep),
            patch(
                "unity.conversation_manager.conversation_manager.build_brain_spec",
                return_value=MockBrainSpec(),
            ),
        ):
            try:
                await ConversationManager._proactive_speech_loop(
                    mock_cm,
                    skip_initial_wait=True,
                )
            except asyncio.CancelledError:
                pass

        # Should NOT have 10 second initial wait
        assert 10 not in sleep_calls

    async def test_loop_calculates_elapsed_time(self, mock_cm):
        """The loop calculates elapsed time from last message timestamp."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = "call"

        # Set up a fixed "now" time and a transcript timestamp 15 seconds before
        fixed_now = datetime(2025, 1, 1, 12, 0, 15)
        past_time = datetime(2025, 1, 1, 12, 0, 0)  # 15 seconds before fixed_now

        mock_cm.get_recent_voice_transcript = MagicMock(
            return_value=(
                [{"role": "user", "content": "Hello"}],
                past_time,
            ),
        )

        captured_elapsed = None

        async def mock_decide(chat_history, system_prompt, elapsed_seconds=0):
            nonlocal captured_elapsed
            captured_elapsed = elapsed_seconds
            return ProactiveDecision(should_speak=False)

        mock_cm.proactive_speech.decide = mock_decide
        mock_cm.schedule_proactive_speech = AsyncMock()
        mock_cm.schedule_proactive_speech.side_effect = asyncio.CancelledError()

        with (
            patch("asyncio.sleep", new=AsyncMock()),
            patch(
                "unity.conversation_manager.conversation_manager.build_brain_spec",
                return_value=MockBrainSpec(),
            ),
            patch(
                "unity.conversation_manager.conversation_manager.prompt_now",
                return_value=fixed_now,
            ),
        ):
            try:
                await ConversationManager._proactive_speech_loop(
                    mock_cm,
                    skip_initial_wait=True,
                )
            except asyncio.CancelledError:
                pass

        # Should have calculated exactly 15 seconds elapsed
        assert captured_elapsed is not None
        assert captured_elapsed == 15.0

    async def test_loop_publishes_guidance_when_should_speak(self, mock_cm):
        """The loop publishes call_guidance when decision says to speak."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = "call"

        async def mock_decide(*args, **kwargs):
            return ProactiveDecision(
                should_speak=True,
                delay=0,  # No delay for test
                content="Still with you!",
            )

        mock_cm.proactive_speech.decide = mock_decide
        mock_cm.schedule_proactive_speech = AsyncMock()
        mock_cm.schedule_proactive_speech.side_effect = asyncio.CancelledError()

        with (
            patch("asyncio.sleep", new=AsyncMock()),
            patch(
                "unity.conversation_manager.conversation_manager.build_brain_spec",
                return_value=MockBrainSpec(),
            ),
        ):
            try:
                await ConversationManager._proactive_speech_loop(
                    mock_cm,
                    skip_initial_wait=True,
                )
            except asyncio.CancelledError:
                pass

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
        data = json.loads(message)
        assert data["content"] == "Still with you!"

    async def test_loop_records_message_in_contact_index(self, mock_cm):
        """The loop records the proactive message in contact_index."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = "call"

        async def mock_decide(*args, **kwargs):
            return ProactiveDecision(
                should_speak=True,
                delay=0,
                content="Are you still there?",
            )

        mock_cm.proactive_speech.decide = mock_decide
        mock_cm.schedule_proactive_speech = AsyncMock()
        mock_cm.schedule_proactive_speech.side_effect = asyncio.CancelledError()

        with (
            patch("asyncio.sleep", new=AsyncMock()),
            patch(
                "unity.conversation_manager.conversation_manager.build_brain_spec",
                return_value=MockBrainSpec(),
            ),
        ):
            try:
                await ConversationManager._proactive_speech_loop(
                    mock_cm,
                    skip_initial_wait=True,
                )
            except asyncio.CancelledError:
                pass

        # Should have recorded the message
        # Check the active contact has a voice thread with the message
        contact = mock_cm.get_active_contact()
        active_contact = mock_cm.contact_index.active_conversations.get(
            contact["contact_id"],
        )
        assert active_contact is not None

        voice_thread = active_contact.threads.get("voice", [])
        # Find the proactive message
        proactive_msg = None
        for msg in voice_thread:
            if "Are you still there?" in (msg.content or ""):
                proactive_msg = msg
                break

        assert proactive_msg is not None

    async def test_loop_reschedules_with_adaptive_wait(self, mock_cm):
        """The loop reschedules with adaptive wait when not speaking."""
        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm.mode = "call"

        # Set up fixed timestamps: elapsed time = 5s (< 10s threshold)
        # This should result in wait_time = min(12 - 5, 7) = 7 seconds
        fixed_now = datetime(2025, 1, 1, 12, 0, 5)
        past_time = datetime(2025, 1, 1, 12, 0, 0)  # 5 seconds before

        mock_cm.get_recent_voice_transcript = MagicMock(
            return_value=(
                [{"role": "user", "content": "Hello"}],
                past_time,
            ),
        )

        async def mock_decide(*args, **kwargs):
            return ProactiveDecision(should_speak=False, delay=3)

        mock_cm.proactive_speech.decide = mock_decide

        schedule_called_with_skip = None

        async def mock_schedule(skip_initial_wait=False):
            nonlocal schedule_called_with_skip
            schedule_called_with_skip = skip_initial_wait
            raise asyncio.CancelledError()

        mock_cm.schedule_proactive_speech = mock_schedule

        sleep_durations = []

        async def track_sleep(duration):
            sleep_durations.append(duration)

        with (
            patch("asyncio.sleep", new=track_sleep),
            patch(
                "unity.conversation_manager.conversation_manager.build_brain_spec",
                return_value=MockBrainSpec(),
            ),
            patch(
                "unity.conversation_manager.conversation_manager.prompt_now",
                return_value=fixed_now,
            ),
        ):
            try:
                await ConversationManager._proactive_speech_loop(
                    mock_cm,
                    skip_initial_wait=True,
                )
            except asyncio.CancelledError:
                pass

        # Should have rescheduled with skip_initial_wait=True
        assert schedule_called_with_skip is True

        # Should have adaptive wait of 7 seconds (min(12 - 5, 7) = 7)
        assert 7 in sleep_durations


# =============================================================================
# 6. ProactiveSpeech.decide() Integration Tests (Real LLM)
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
            elapsed_seconds=5,
        )

        assert isinstance(decision, ProactiveDecision)
        assert isinstance(decision.should_speak, bool)
        assert isinstance(decision.delay, int)

    async def test_decide_respects_short_elapsed_time(self):
        """decide() should NOT speak when elapsed time is very short."""
        ps = ProactiveSpeech()

        chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ]
        system_prompt = "You are a helpful assistant."

        decision = await ps.decide(
            chat_history=chat_history,
            system_prompt=system_prompt,
            elapsed_seconds=3,  # Very short - should not speak
        )

        # Per the prompt, < 10s should always be should_speak=false
        assert decision.should_speak is False

    async def test_decide_speaks_on_long_silence(self):
        """decide() should speak when elapsed time is very long after assistant's question."""
        ps = ProactiveSpeech()

        # Scenario: Assistant asked a question, user hasn't responded for 35 seconds
        # This is clearly awkward silence that needs to be filled
        chat_history = [
            {
                "role": "assistant",
                "content": "What time works best for your appointment?",
            },
        ]
        system_prompt = "You are a helpful assistant scheduling an appointment."

        decision = await ps.decide(
            chat_history=chat_history,
            system_prompt=system_prompt,
            elapsed_seconds=35,  # Very long - should speak
        )

        # Per the prompt, >= 30s should ABSOLUTELY speak
        assert decision.should_speak is True
        assert decision.content is not None
        assert len(decision.content) > 0

    async def test_decide_handles_empty_history(self):
        """decide() handles empty chat history gracefully."""
        ps = ProactiveSpeech()

        decision = await ps.decide(
            chat_history=[],
            system_prompt="You are a helpful assistant.",
            elapsed_seconds=15,
        )

        assert isinstance(decision, ProactiveDecision)

    async def test_decide_handles_exception_gracefully(self):
        """decide() returns should_speak=False on exception."""
        ps = ProactiveSpeech(model="nonexistent-model@fake-provider")

        # This should fail but return a safe default
        decision = await ps.decide(
            chat_history=[{"role": "user", "content": "Hello"}],
            system_prompt="Test",
            elapsed_seconds=15,
        )

        # Should return safe default on error
        assert decision.should_speak is False


# =============================================================================
# 7. Event Handler Integration Tests
# =============================================================================


@pytest.mark.asyncio
class TestEventHandlerProactiveSpeechIntegration:
    """Tests verifying event handlers properly cancel proactive speech."""

    async def test_inbound_phone_utterance_cancels_proactive(self, mock_cm):
        """InboundPhoneUtterance event cancels proactive speech."""
        from unity.conversation_manager.events import InboundPhoneUtterance
        from unity.conversation_manager.domains.event_handlers import EventHandler

        mock_cm.cancel_proactive_speech = AsyncMock()
        mock_cm.interject_or_run = AsyncMock()

        event = InboundPhoneUtterance(
            contact={"contact_id": 1, "first_name": "Boss", "surname": "User"},
            content="Hello?",
        )

        await EventHandler.handle_event(event, mock_cm, is_voice_call=False)

        mock_cm.cancel_proactive_speech.assert_called_once()

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
        contact = mock_cm.contact_index.get_contact(contact_id=1)
        mock_cm.contact_index.push_message(
            contact,
            "voice",
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
        contact = mock_cm.contact_index.get_contact(contact_id=1)
        mock_cm.contact_index.push_message(
            contact,
            "voice",
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
            lambda **kw: ConversationManager.schedule_proactive_speech(
                cm,
                **kw,
            )
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
            lambda **kw: ConversationManager.schedule_proactive_speech(
                cm,
                **kw,
            )
        )

        # Switch to call mode
        cm.mode = "call"

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
            lambda **kw: ConversationManager.schedule_proactive_speech(
                cm,
                **kw,
            )
        )

        # Switch to call mode and schedule
        cm.mode = "call"
        await cm.schedule_proactive_speech()

        assert cm._proactive_speech_task is not None

        # Cancel
        await cm.cancel_proactive_speech()

        # Task should be cleared
        assert cm._proactive_speech_task is None
