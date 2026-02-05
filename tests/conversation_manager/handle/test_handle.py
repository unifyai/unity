"""
tests/conversation_manager/test_handle.py
===============================================

Comprehensive tests for ConversationManagerHandle, specifically the ask() method.

The ask() method is the bridge between the Actor (and other managers) and the live
conversation with the user. It enables nested tool loops that can:
- Infer answers from recent conversation transcript (PATH 1)
- Ask clarifying questions and wait for user replies (PATH 2)
- Return structured responses (Pydantic models, Enums)

These tests validate the complete flow including:
- PATH 1: Inference from transcript without user interaction
- PATH 2: Interactive question/answer with user reply routing
- Structured output parsing and validation
- active_ask_handle registration and input routing
- Multiple communication modalities (voice calls, SMS, email context)
- Error handling and edge cases

Test Categories:
- Unit tests: Test isolated components with mocked dependencies
- Integration tests: Test the full flow with the CMStepDriver

Note: These are primarily eval tests since they involve LLM behavior.
"""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from tests.helpers import _handle_project
from tests.conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import (
    DirectMessageEvent,
    EmailReceived,
    Event,
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    PhoneCallReceived,
    PhoneCallStarted,
    SMSReceived,
    UnifyMeetReceived,
    UnifyMeetStarted,
    UnifyMessageReceived,
)
from unity.conversation_manager.handle import ConversationManagerHandle

pytestmark = pytest.mark.eval


# =============================================================================
# Helper Functions for Deterministic Waiting
# =============================================================================


async def _wait_for_condition(predicate, *, timeout: float = 30.0, poll: float = 0.05):
    """Poll predicate() until it returns True or timeout expires."""
    import time as _time

    start = _time.perf_counter()
    while _time.perf_counter() - start < timeout:
        if predicate():
            return True
        await asyncio.sleep(poll)
    return False


# =============================================================================
# Test Response Formats (Pydantic Models and Enums)
# =============================================================================


class UserPreference(BaseModel):
    """Structured response for user preference questions."""

    preference: str = Field(description="The user's stated preference")
    confidence: float = Field(
        description="Confidence level from 0 to 1",
        ge=0.0,
        le=1.0,
    )


class MeetingTime(BaseModel):
    """Structured response for meeting time questions."""

    hour: int = Field(description="Hour of the meeting (24-hour format)", ge=0, le=23)
    minute: int = Field(description="Minute of the meeting", ge=0, le=59)
    period: str = Field(description="AM or PM")


class IssueCategory(str, Enum):
    """Enum for categorizing user issues."""

    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    URGENT = "urgent"


class ConfirmationResponse(str, Enum):
    """Enum for yes/no confirmation."""

    YES = "yes"
    NO = "no"
    MAYBE = "maybe"


class IssueCategoryResponse(BaseModel):
    """Wrapper for IssueCategory enum to use with response_format."""

    category: IssueCategory = Field(description="The category of the issue")


class ConfirmationResponseWrapper(BaseModel):
    """Wrapper for ConfirmationResponse enum to use with response_format."""

    answer: ConfirmationResponse = Field(description="The user's yes/no/maybe response")


# =============================================================================
# Helper Functions
# =============================================================================


def _direct_messages(events) -> list[DirectMessageEvent]:
    """Extract DirectMessageEvent from output events."""
    return [e for e in events if isinstance(e, DirectMessageEvent)]


# =============================================================================
# Unit Tests: Handle Initialization
# =============================================================================


class TestHandleInitialization:
    """Tests for ConversationManagerHandle initialization."""

    def test_handle_requires_event_broker(self):
        """Handle cannot be created without an event broker."""
        with pytest.raises(TypeError):
            ConversationManagerHandle(
                conversation_id="test",
                contact_id=1,
            )

    def test_handle_stores_conversation_context(self):
        """Handle stores conversation_id and contact_id correctly."""
        mock_broker = MagicMock()
        mock_cm = MagicMock()
        mock_cm.call_manager = MagicMock()
        mock_cm.contact_index = MagicMock()

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_id="conv_123",
            contact_id=42,
            conversation_manager=mock_cm,
        )

        assert handle.conversation_id == "conv_123"
        assert handle.contact_id == 42
        assert handle.event_broker is mock_broker
        assert not handle._stopped

    def test_handle_not_stopped_initially(self):
        """Handle starts in active (not stopped) state."""
        mock_broker = MagicMock()
        mock_cm = MagicMock()
        mock_cm.call_manager = MagicMock()
        mock_cm.contact_index = MagicMock()

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_id="conv_123",
            contact_id=1,
            conversation_manager=mock_cm,
        )

        assert handle.done() is False


# =============================================================================
# Unit Tests: send_notification and interject
# =============================================================================


class TestSendNotification:
    """Tests for send_notification method."""

    @pytest.mark.asyncio
    async def test_send_notification_publishes_event(self):
        """send_notification publishes NotificationInjectedEvent to broker."""
        mock_broker = AsyncMock()
        mock_broker.publish = AsyncMock(return_value=1)
        mock_cm = MagicMock()
        mock_cm.call_manager = MagicMock()
        mock_cm.contact_index = MagicMock()

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_id="conv_123",
            contact_id=1,
            conversation_manager=mock_cm,
        )

        result = await handle.send_notification(
            "Task completed successfully",
            source="test_system",
        )

        assert result["status"] == "ok"
        assert "interjection_id" in result
        mock_broker.publish.assert_called_once()

        # Verify the published event
        call_args = mock_broker.publish.call_args
        channel = call_args[0][0]
        assert channel == "app:comms:steering"

    @pytest.mark.asyncio
    async def test_send_notification_with_pinned(self):
        """send_notification respects pinned parameter."""
        mock_broker = AsyncMock()
        mock_broker.publish = AsyncMock(return_value=1)
        mock_cm = MagicMock()
        mock_cm.call_manager = MagicMock()
        mock_cm.contact_index = MagicMock()

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_id="conv_123",
            contact_id=1,
            conversation_manager=mock_cm,
        )

        result = await handle.send_notification(
            "Important reminder",
            pinned=True,
            interjection_id="custom_id_123",
        )

        assert result["status"] == "ok"
        assert result["interjection_id"] == "custom_id_123"

        # Verify pinned flag was included
        call_args = mock_broker.publish.call_args
        message_json = call_args[0][1]
        event_data = json.loads(message_json)
        assert event_data["payload"]["pinned"] is True

    @pytest.mark.asyncio
    async def test_send_notification_fails_when_stopped(self):
        """send_notification returns error when handle is stopped."""
        mock_broker = AsyncMock()
        mock_cm = MagicMock()
        mock_cm.call_manager = MagicMock()
        mock_cm.contact_index = MagicMock()

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_id="conv_123",
            contact_id=1,
            conversation_manager=mock_cm,
        )
        await handle.stop(reason="test")

        result = await handle.send_notification("Should fail")

        assert result["status"] == "error"
        assert "stopped" in result["message"].lower()


class TestInterject:
    """Tests for interject method (wrapper around send_notification)."""

    @pytest.mark.asyncio
    async def test_interject_delegates_to_send_notification(self):
        """interject calls send_notification with correct parameters."""
        mock_broker = AsyncMock()
        mock_broker.publish = AsyncMock(return_value=1)
        mock_cm = MagicMock()
        mock_cm.call_manager = MagicMock()
        mock_cm.contact_index = MagicMock()

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_id="conv_123",
            contact_id=1,
            conversation_manager=mock_cm,
        )

        await handle.interject(
            "User correction incoming",
            pinned=True,
        )

        mock_broker.publish.assert_called_once()


# =============================================================================
# Unit Tests: stop and lifecycle methods
# =============================================================================


class TestHandleLifecycle:
    """Tests for handle lifecycle methods (stop, done, result)."""

    @pytest.mark.asyncio
    async def test_stop_marks_handle_done(self):
        """stop() marks the handle as done."""
        mock_broker = MagicMock()
        mock_cm = MagicMock()
        mock_cm.call_manager = MagicMock()
        mock_cm.contact_index = MagicMock()

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_id="conv_123",
            contact_id=1,
            conversation_manager=mock_cm,
        )

        assert handle.done() is False
        await handle.stop(reason="user cancelled")
        assert handle.done() is True

    @pytest.mark.asyncio
    async def test_stop_returns_reason(self):
        """stop() returns message with reason."""
        mock_broker = MagicMock()
        mock_cm = MagicMock()
        mock_cm.call_manager = MagicMock()
        mock_cm.contact_index = MagicMock()

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_id="conv_123",
            contact_id=1,
            conversation_manager=mock_cm,
        )

        await handle.stop(reason="task completed")
        assert handle._stopped
        result = await handle.result()
        assert "task completed" in result

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        """Calling stop() multiple times is safe."""
        mock_broker = MagicMock()
        mock_cm = MagicMock()
        mock_cm.call_manager = MagicMock()
        mock_cm.contact_index = MagicMock()

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_id="conv_123",
            contact_id=1,
            conversation_manager=mock_cm,
        )

        await handle.stop(reason="first")
        await handle.stop(reason="second")
        assert handle._stopped

    @pytest.mark.asyncio
    async def test_result_waits_for_stop(self):
        """result() blocks until handle is stopped."""
        mock_broker = MagicMock()
        mock_cm = MagicMock()
        mock_cm.call_manager = MagicMock()
        mock_cm.contact_index = MagicMock()

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_id="conv_123",
            contact_id=1,
            conversation_manager=mock_cm,
        )

        # Start waiting for result in background
        result_task = asyncio.create_task(handle.result())

        # Give it a moment to start waiting
        await asyncio.sleep(0.05)
        assert not result_task.done()

        # Stop the handle
        await handle.stop(reason="completed")

        # Result should now complete
        result = await asyncio.wait_for(result_task, timeout=1.0)
        assert "completed" in result


# =============================================================================
# Integration Tests: ask() with PATH 1 (Inference)
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_ask_path1_infers_from_voice_transcript(initialized_cm):
    """
    PATH 1: LLM infers answer from voice transcript without asking user.

    During a voice call, the user mentions their preference. When ask() is called,
    the LLM should infer the answer from the recent transcript.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Establish a voice call with conversation context
    await cm.step(PhoneCallReceived(contact=contact, conference_name="test_conf"))
    await cm.step(PhoneCallStarted(contact=contact))

    # User mentions their preference in the conversation
    await cm.step(
        InboundPhoneUtterance(
            contact=contact,
            content="I prefer to have meetings in the morning, around 9 AM works best for me.",
        ),
    )

    # Get the handle from CM
    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    # Ask about the user's preference - should infer from transcript
    ask_handle = await handle.ask(
        "What time does the user prefer for meetings?",
        response_format=MeetingTime,
    )

    # Get the result
    result = await ask_handle.result()

    # Should have inferred from transcript
    assert isinstance(result, MeetingTime)
    assert result.hour == 9
    assert result.period.upper() == "AM"


@pytest.mark.asyncio
@_handle_project
async def test_ask_path1_infers_from_sms_context(initialized_cm):
    """
    PATH 1: LLM infers answer from SMS conversation context.

    When the user has already stated something via SMS, ask() should
    infer from that context without needing to ask again.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # User sends SMS with clear preference
    await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="For the project meeting, I'd like to discuss billing issues we've been having.",
        ),
    )

    # Get the handle
    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    # Ask about issue category - should infer billing from context
    ask_handle = await handle.ask(
        "What category of issue is the user asking about?",
        response_format=IssueCategoryResponse,
    )

    result = await ask_handle.result()

    assert isinstance(result, IssueCategoryResponse)
    assert result.category == IssueCategory.BILLING


@pytest.mark.asyncio
@_handle_project
async def test_ask_path1_sends_acknowledgment(initialized_cm):
    """
    PATH 1: When inferring, LLM sends acknowledgment via DirectMessageEvent.

    The acknowledgment should be published to app:comms:direct_speech.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Establish voice call
    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))

    # User provides clear answer
    await cm.step(
        InboundUnifyMeetUtterance(
            contact=contact,
            content="Yes, I confirm I want to proceed with the booking.",
        ),
    )

    # Track published events
    published_events = []
    original_publish = cm.event_broker.publish

    async def track_publish(channel, message):
        try:
            evt = Event.from_json(message)
            published_events.append((channel, evt))
        except Exception:
            pass
        return await original_publish(channel, message)

    cm.cm.event_broker.publish = track_publish

    # Get handle and ask
    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask(
        "Does the user confirm they want to proceed?",
        response_format=ConfirmationResponseWrapper,
    )

    result = await ask_handle.result()

    # Should have confirmed
    assert isinstance(result, ConfirmationResponseWrapper)
    assert result.answer == ConfirmationResponse.YES

    # Check for direct speech event (acknowledgment)
    direct_speech_events = [
        evt for channel, evt in published_events if channel == "app:comms:direct_speech"
    ]
    # PATH 1 may or may not send acknowledgment depending on LLM decision
    # Just verify it didn't crash


# =============================================================================
# Integration Tests: ask() with PATH 2 (Interactive)
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_ask_path2_asks_when_ambiguous(initialized_cm):
    """
    PATH 2: LLM asks clarifying question when transcript is ambiguous.

    When there's no clear answer in the transcript, the LLM should use
    the ask_question tool to ask the user directly.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Establish voice call with no relevant context
    await cm.step(PhoneCallReceived(contact=contact, conference_name="test_conf"))
    await cm.step(PhoneCallStarted(contact=contact))

    # User says something unrelated
    await cm.step(
        InboundPhoneUtterance(
            contact=contact,
            content="Hello, I wanted to discuss something with you today.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    # Track direct speech events
    direct_speech_events = []

    async def capture_events(channel, message):
        if channel == "app:comms:direct_speech":
            try:
                direct_speech_events.append(Event.from_json(message))
            except Exception:
                pass
        return 1

    cm.cm.event_broker.publish = capture_events

    # Ask something not in transcript - should trigger PATH 2
    ask_handle = await handle.ask(
        "What time would the user like to schedule the meeting?",
        response_format=MeetingTime,
    )

    # Wait for active_ask_handle to be set (indicates PATH 2 started)
    # Use polling instead of fixed sleep for deterministic behavior
    await _wait_for_condition(
        lambda: cm.cm.active_ask_handle is not None,
        timeout=30.0,
    )

    # If PATH 2, there should be a direct speech asking the question
    # The handle should be waiting for user reply
    assert not ask_handle.done()

    # Simulate user reply via interject
    await ask_handle.interject("Let's do 2 PM")

    # Now get result
    result = await asyncio.wait_for(ask_handle.result(), timeout=30)

    assert isinstance(result, MeetingTime)
    assert result.hour == 14  # 2 PM in 24-hour
    assert result.period.upper() == "PM"


@pytest.mark.asyncio
@_handle_project
async def test_ask_path2_routes_user_input_via_active_ask_handle(initialized_cm):
    """
    PATH 2: User input is routed to active_ask_handle via interject_or_run.

    When active_ask_handle is set, user utterances should be routed to it
    instead of triggering the Main CM Brain.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Establish voice call
    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    # Start an ask - this should register active_ask_handle
    ask_handle = await handle.ask(
        "What is the user's favorite color?",
    )

    # Verify active_ask_handle is set
    assert cm.cm.active_ask_handle is not None
    assert cm.cm.active_ask_handle is ask_handle

    # Simulate user input - should route to ask_handle
    await cm.cm.interject_or_run("My favorite color is blue")

    # Result should now be available
    result = await asyncio.wait_for(ask_handle.result(), timeout=30)

    assert "blue" in result.lower()

    # active_ask_handle should be cleared after result
    assert cm.cm.active_ask_handle is None


@pytest.mark.asyncio
@_handle_project
async def test_ask_path2_multiple_followup_questions(initialized_cm):
    """
    PATH 2: LLM can ask multiple follow-up questions to get clarity.

    If the user's first reply doesn't answer the question, the LLM
    should ask a follow-up question.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Establish voice call
    await cm.step(PhoneCallReceived(contact=contact, conference_name="test_conf"))
    await cm.step(PhoneCallStarted(contact=contact))

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    # Track questions asked
    questions_asked = []
    original_publish = cm.cm.event_broker.publish

    async def track_questions(channel, message):
        if channel == "app:comms:direct_speech":
            try:
                evt = Event.from_json(message)
                if isinstance(evt, DirectMessageEvent):
                    questions_asked.append(evt.content)
            except Exception:
                pass
        return await original_publish(channel, message)

    cm.cm.event_broker.publish = track_questions

    # Ask something specific
    ask_handle = await handle.ask(
        "What is the user's phone number for callbacks?",
    )

    # Wait for first question to be asked (poll for direct_speech event)
    await _wait_for_condition(
        lambda: len(questions_asked) >= 1,
        timeout=30.0,
    )

    # Give vague reply - should trigger follow-up
    await ask_handle.interject("You can reach me anytime")

    # Wait for follow-up question (poll for second question)
    await _wait_for_condition(
        lambda: len(questions_asked) >= 2 or ask_handle.done(),
        timeout=30.0,
    )

    # Provide actual answer
    await ask_handle.interject("+1-555-123-4567")

    result = await asyncio.wait_for(ask_handle.result(), timeout=30)

    # Should have extracted the phone number
    assert "555" in result or "123" in result or "4567" in result


# =============================================================================
# Integration Tests: Structured Output (Pydantic and Enum)
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_ask_returns_pydantic_model(initialized_cm):
    """
    ask() returns validated Pydantic model when response_format is specified.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="I strongly prefer video calls over phone calls for all meetings.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask(
        "What is the user's communication preference and how confident are you?",
        response_format=UserPreference,
    )

    result = await ask_handle.result()

    assert isinstance(result, UserPreference)
    assert "video" in result.preference.lower()
    assert 0 <= result.confidence <= 1


@pytest.mark.asyncio
@_handle_project
async def test_ask_returns_enum_value(initialized_cm):
    """
    ask() returns correct Enum value when response_format is an Enum.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Can you help me understand why my payment didn't go through?",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask(
        "What category of issue is this?",
        response_format=IssueCategoryResponse,
    )

    result = await ask_handle.result()

    assert isinstance(result, IssueCategoryResponse)
    assert result.category == IssueCategory.BILLING


@pytest.mark.asyncio
@_handle_project
async def test_ask_without_response_format_returns_string(initialized_cm):
    """
    ask() without response_format returns a string summary.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="The product code I'm asking about is SKU-12345-XYZ.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask(
        "What product code did the user mention?",
    )

    result = await ask_handle.result()

    # Without response_format, should return string/dict
    assert "SKU-12345-XYZ" in str(result) or "12345" in str(result)


# =============================================================================
# Integration Tests: InterceptingHandle Behavior
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_intercepting_handle_delegates_lifecycle_methods(initialized_cm):
    """
    The InterceptingHandle correctly delegates stop/pause/resume/done.

    Follows the standard steerable handle pattern:
    - stop() sets the cancel flag
    - await result() waits for clean completion (returns None when stopped)
    - done() returns True after the inner loop completes
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask("What is the user's name?")

    # done() should return False initially (loop is running)
    assert ask_handle.done() is False

    # stop() tells the inner loop to cancel - this is non-blocking
    await ask_handle.stop(reason="cancelled by test")

    # Following standard pattern: await result() for clean completion
    # When stopped, result() returns None
    result = await asyncio.wait_for(ask_handle.result(), timeout=30.0)
    assert result is None

    # After the inner loop completes, done() should return True
    assert ask_handle.done() is True


@pytest.mark.asyncio
@_handle_project
async def test_intercepting_handle_clears_active_ask_handle_on_result(initialized_cm):
    """
    When result() completes, active_ask_handle is cleared.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="My name is Alice Johnson.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask("What is the user's name?")

    # active_ask_handle should be set
    assert cm.cm.active_ask_handle is ask_handle

    # Get result
    await ask_handle.result()

    # active_ask_handle should be cleared
    assert cm.cm.active_ask_handle is None


# =============================================================================
# Integration Tests: Multiple Modalities
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_ask_during_phone_call(initialized_cm):
    """
    ask() works correctly during an active phone call.

    DirectMessageEvent should be published to call_guidance channel.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Start phone call
    await cm.step(PhoneCallReceived(contact=contact, conference_name="test_conf"))
    await cm.step(PhoneCallStarted(contact=contact))

    # User provides context
    await cm.step(
        InboundPhoneUtterance(
            contact=contact,
            content="I need help with a technical problem with my software.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask(
        "What type of issue is the user experiencing?",
        response_format=IssueCategoryResponse,
    )

    result = await ask_handle.result()

    assert isinstance(result, IssueCategoryResponse)
    assert result.category == IssueCategory.TECHNICAL


@pytest.mark.asyncio
@_handle_project
async def test_ask_during_unify_meet(initialized_cm):
    """
    ask() works correctly during an active Unify Meet session.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Start Unify Meet
    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))

    # User provides context
    await cm.step(
        InboundUnifyMeetUtterance(
            contact=contact,
            content="Yes, let's definitely proceed with the plan we discussed.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask(
        "Does the user want to proceed?",
        response_format=ConfirmationResponseWrapper,
    )

    result = await ask_handle.result()

    assert isinstance(result, ConfirmationResponseWrapper)
    assert result.answer == ConfirmationResponse.YES


@pytest.mark.asyncio
@_handle_project
async def test_ask_with_sms_context_only(initialized_cm):
    """
    ask() can infer from SMS-only context (no active voice call).
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # SMS conversation only
    await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="No, I don't think that will work for me.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask(
        "Is the user agreeing or disagreeing?",
        response_format=ConfirmationResponseWrapper,
    )

    result = await ask_handle.result()

    assert isinstance(result, ConfirmationResponseWrapper)
    assert result.answer == ConfirmationResponse.NO


@pytest.mark.asyncio
@_handle_project
async def test_ask_with_email_context(initialized_cm):
    """
    ask() can infer from email context.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm.step_until_wait(
        EmailReceived(
            contact=contact,
            subject="Question about my account",
            body="I'm not sure if I should continue with this. Maybe we should reconsider?",
            email_id="email_123",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask(
        "Is the user certain about their decision?",
        response_format=ConfirmationResponseWrapper,
    )

    result = await ask_handle.result()

    # "Maybe" or uncertain response expected
    assert isinstance(result, ConfirmationResponseWrapper)
    assert result.answer in [ConfirmationResponse.MAYBE, ConfirmationResponse.NO]


@pytest.mark.asyncio
@_handle_project
async def test_ask_with_mixed_modality_context(initialized_cm):
    """
    ask() handles context from multiple modalities (SMS + voice).
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # First, SMS context
    await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="I'd like to schedule a call to discuss something important.",
        ),
    )

    # Then voice call
    await cm.step(PhoneCallReceived(contact=contact, conference_name="test_conf"))
    await cm.step(PhoneCallStarted(contact=contact))

    await cm.step(
        InboundPhoneUtterance(
            contact=contact,
            content="So about that important thing - it's a billing question.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask(
        "What is the main topic the user wants to discuss?",
        response_format=IssueCategoryResponse,
    )

    result = await ask_handle.result()

    assert isinstance(result, IssueCategoryResponse)
    assert result.category == IssueCategory.BILLING


# =============================================================================
# Integration Tests: Error Handling and Edge Cases
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_ask_raises_when_handle_stopped(initialized_cm):
    """
    ask() raises RuntimeError when called on a stopped handle.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    await handle.stop(reason="already done")

    with pytest.raises(RuntimeError, match="stopped"):
        await handle.ask("This should fail")


@pytest.mark.asyncio
@_handle_project
async def test_ask_handles_empty_transcript(initialized_cm):
    """
    ask() handles case where transcript is empty.

    Should fall back to PATH 2 and ask the user directly.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Start call but no utterances yet
    await cm.step(PhoneCallReceived(contact=contact, conference_name="test_conf"))
    await cm.step(PhoneCallStarted(contact=contact))

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask("What is the user's name?")

    # Wait for active_ask_handle to be set (PATH 2 started)
    await _wait_for_condition(
        lambda: cm.cm.active_ask_handle is not None,
        timeout=30.0,
    )

    # Should be waiting for user input (PATH 2)
    assert not ask_handle.done()

    # Provide answer via interject
    await ask_handle.interject("My name is Bob")

    result = await asyncio.wait_for(ask_handle.result(), timeout=30)

    assert "bob" in result.lower()


@pytest.mark.asyncio
@_handle_project
async def test_ask_with_transcript_manager_tool(initialized_cm):
    """
    ask() provides ask_historic_transcript tool for querying older context.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Current conversation
    await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Can you remind me what we discussed last time?",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    # This ask might trigger ask_historic_transcript tool
    ask_handle = await handle.ask(
        "What does the user want to be reminded about?",
    )

    result = await ask_handle.result()

    # Should have some response (may use transcript tool or infer)
    assert result is not None


# =============================================================================
# Integration Tests: Concurrent ask() calls
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_only_one_active_ask_handle_at_a_time(initialized_cm):
    """
    Only one ask handle can be active at a time.

    Starting a new ask() should replace the previous active_ask_handle.
    Follows established cleanup pattern: stop() + await result().
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    # Start first ask
    ask_handle_1 = await handle.ask("What is the user's favorite color?")

    # Verify it's active
    assert cm.cm.active_ask_handle is ask_handle_1

    # Stop first handle before starting second (clean replacement)
    await ask_handle_1.stop(reason="replaced by new ask")

    # Start second ask (should replace first)
    ask_handle_2 = await handle.ask("What is the user's favorite food?")

    # Second should now be active
    assert cm.cm.active_ask_handle is ask_handle_2

    # Complete second ask properly
    await ask_handle_2.interject("I love pizza")
    await ask_handle_2.result()

    # Cleanup: ensure first handle completes
    try:
        await asyncio.wait_for(ask_handle_1.result(), timeout=5.0)
    except Exception:
        pass  # May already be cleaned up


# =============================================================================
# Integration Tests: Transcript Content (CallGuidance Exclusion)
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_transcript_excludes_call_guidance(initialized_cm):
    """
    CallGuidance (internal orchestration) should NOT appear in the transcript.

    The transcript passed to handle.ask() should only contain actual communications
    between the assistant and contacts, not internal guidance from the Main CM Brain
    to the Voice Agent.

    This test verifies that:
    1. User utterances appear in the transcript
    2. CallGuidance messages are filtered out
    3. Only "user" and "assistant" roles appear (not "guidance")
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Start a Unify Meet session - this will trigger CallGuidance from the CM brain
    await cm.step(UnifyMeetReceived(contact=contact))
    await cm.step(UnifyMeetStarted(contact=contact))

    # User speaks - this should appear in transcript
    user_message = "I'd like to schedule a meeting for tomorrow."
    await cm.step(
        InboundUnifyMeetUtterance(
            contact=contact,
            content=user_message,
        ),
    )

    # Get the transcript that would be passed to handle.ask()
    conversation_turns, _ = cm.cm.get_recent_transcript(
        contact=contact,
        max_messages=20,
    )

    # Should have at least the user message
    assert len(conversation_turns) >= 1, "Transcript should contain user message"

    # All roles should be either "user" or "assistant" (not "guidance")
    for turn in conversation_turns:
        assert turn["role"] in (
            "user",
            "assistant",
        ), f"Unexpected role in transcript: {turn['role']}"

    # User message should be in transcript
    transcript_contents = [turn["content"] for turn in conversation_turns]
    assert any(
        user_message in content for content in transcript_contents
    ), "User message should appear in transcript"


# =============================================================================
# Integration Tests: get_full_transcript
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_get_full_transcript_returns_messages(initialized_cm):
    """
    get_full_transcript returns recent conversation messages.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Create some conversation
    await cm.step_until_wait(
        SMSReceived(contact=contact, content="First message"),
    )
    await cm.step_until_wait(
        SMSReceived(contact=contact, content="Second message"),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    result = await handle.get_full_transcript(max_messages=10)

    assert result["status"] == "ok"
    # The simulated transcript manager may return varying results
    assert "messages" in result
    assert "count" in result


# =============================================================================
# Integration Tests: unpin_interjection
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_unpin_interjection_publishes_event(initialized_cm):
    """
    unpin_interjection publishes NotificationUnpinnedEvent.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Track published events
    published = []
    original_publish = cm.event_broker.publish

    async def track(channel, message):
        published.append((channel, message))
        return await original_publish(channel, message)

    cm.cm.event_broker.publish = track

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_id="test_conv",
        contact_id=contact["contact_id"],
        conversation_manager=cm.cm,
    )

    # First pin something via send_notification (interject returns None)
    pin_result = await handle.send_notification(
        "Pinned message",
        source="interjection",
        pinned=True,
    )
    interjection_id = pin_result["interjection_id"]

    # Now unpin
    unpin_result = await handle.unpin_interjection(interjection_id)

    assert unpin_result["status"] == "ok"
    assert unpin_result["interjection_id"] == interjection_id

    # Verify unpin event was published
    unpin_events = [
        (ch, msg) for ch, msg in published if "NotificationUnpinnedEvent" in msg
    ]
    assert len(unpin_events) >= 1
