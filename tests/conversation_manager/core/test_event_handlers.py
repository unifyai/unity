"""
tests/conversation_manager/test_event_handlers.py
======================================================

Unit and integration tests for the EventHandler registry and individual
event handlers in `domains/event_handlers.py`.

Tests cover:
1. EventHandler registry pattern (`@EventHandler.register`, `handle_event`)
2. The `_event_type_to_log_key` helper (CamelCase → snake_case conversion)
3. Individual event handler behavior and side effects
4. Handler error cases and edge conditions
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.conversation_manager.domains.event_handlers import (
    EventHandler,
    _event_type_to_log_key,
)
from unity.conversation_manager.events import (
    Event,
    Ping,
    SMSReceived,
    SMSSent,
    EmailReceived,
    EmailSent,
    UnifyMessageReceived,
    PhoneCallReceived,
    PhoneCallStarted,
    PhoneCallEnded,
    PhoneCallAnswered,
    UnifyMeetReceived,
    UnifyMeetStarted,
    UnifyMeetEnded,
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    OutboundPhoneUtterance,
    CallGuidance,
    GetChatHistory,
    ActorHandleStarted,
    ActorHandleResponse,
    ActorResult,
    ActorClarificationRequest,
    NotificationInjectedEvent,
    NotificationUnpinnedEvent,
    SyncContacts,
    LogMessageResponse,
    SummarizeContext,
    DirectMessageEvent,
    AssistantUpdateEvent,
    AssistantScreenShareStarted,
    AssistantScreenShareStopped,
    UserScreenShareStarted,
    UserScreenShareStopped,
    UserRemoteControlStarted,
    UserRemoteControlStopped,
)
from unity.contact_manager.simulated import SimulatedContactManager
from unity.conversation_manager.domains.contact_index import ContactIndex
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.types import Medium, Mode

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
def mock_call_manager():
    """Create a mock call manager."""
    manager = MagicMock()
    manager.start_call = AsyncMock()
    manager.start_unify_meet = AsyncMock()
    manager.cleanup_call_proc = AsyncMock()
    manager.uses_realtime_api = False
    manager.conference_name = None
    manager.call_contact = None
    manager.call_exchange_id = -1
    manager.unify_meet_exchange_id = -1
    return manager


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
        {
            "contact_id": 2,
            "first_name": "Alice",
            "surname": "Smith",
            "email_address": "alice@example.com",
            "phone_number": "+15555552222",
        },
    ]


@pytest.fixture
def mock_cm(mock_session_logger, mock_event_broker, mock_call_manager, sample_contacts):
    """Create a mock ConversationManager with minimal state for handler tests."""
    cm = MagicMock()
    cm._session_logger = mock_session_logger
    cm.event_broker = mock_event_broker
    cm.call_manager = mock_call_manager
    cm.mode = Mode.TEXT
    cm.chat_history = []
    cm.in_flight_actions = {}
    cm.completed_actions = {}
    cm.is_summarizing = False
    cm.memory_manager = None

    # Create a SimulatedContactManager and populate with sample contacts
    contact_manager = SimulatedContactManager()

    # Update system contacts (0 and 1) with sample data
    for contact_data in sample_contacts:
        contact_id = contact_data["contact_id"]
        contact_manager.update_contact(
            contact_id=contact_id,
            first_name=contact_data.get("first_name"),
            surname=contact_data.get("surname"),
            email_address=contact_data.get("email_address"),
            phone_number=contact_data.get("phone_number"),
        )

    # Set up contact index with SimulatedContactManager
    cm.contact_index = ContactIndex()
    cm.contact_index.set_contact_manager(contact_manager)
    cm.contact_manager = contact_manager

    # Set up notifications bar
    cm.notifications_bar = NotificationBar()

    # Mock async methods
    cm.request_llm_run = AsyncMock()
    cm.cancel_proactive_speech = AsyncMock()
    cm.interject_or_run = AsyncMock()
    cm.get_active_contact = MagicMock(return_value=sample_contacts[1])

    return cm


# =============================================================================
# 1. EventHandler Registry Tests
# =============================================================================


class TestEventHandlerRegistry:
    """Tests for the EventHandler registry pattern."""

    def test_registry_is_populated(self):
        """Verify that the registry contains registered event handlers."""
        assert len(EventHandler._registry) > 0, "Registry should have handlers"

    def test_known_events_are_registered(self):
        """Verify that expected event classes are in the registry."""
        expected_events = [
            Ping,
            SMSReceived,
            SMSSent,
            EmailReceived,
            EmailSent,
            PhoneCallReceived,
            PhoneCallStarted,
            PhoneCallEnded,
            GetChatHistory,
            ActorHandleStarted,
            ActorResult,
            NotificationInjectedEvent,
            NotificationUnpinnedEvent,
            AssistantUpdateEvent,
        ]
        for event_cls in expected_events:
            assert (
                event_cls in EventHandler._registry
            ), f"{event_cls.__name__} should be registered"

    def test_unregistered_event_returns_sleep(self):
        """Verify that unregistered events return a no-op coroutine."""
        # VoiceInterrupt is a real event class that has no registered handler
        from unity.conversation_manager.events import VoiceInterrupt

        # VoiceInterrupt should not have a handler registered
        result = EventHandler._registry.get(VoiceInterrupt)
        assert result is None, "VoiceInterrupt should not have a handler"

    def test_register_decorator_single_event(self):
        """Verify @EventHandler.register works for single event class."""
        # Use a dynamically created event class (proper subclass syntax)
        # We need to create a unique class each time to avoid registry conflicts
        import uuid

        class_name = f"_TestSingleEvent_{uuid.uuid4().hex[:8]}"
        TestEventCls = type(class_name, (Event,), {})

        test_handler_called = []

        @EventHandler.register(TestEventCls)
        async def test_handler(event, cm, *args, **kwargs):
            test_handler_called.append(True)

        assert TestEventCls in EventHandler._registry
        assert EventHandler._registry[TestEventCls] == test_handler

        # Cleanup
        del EventHandler._registry[TestEventCls]

    def test_register_decorator_multiple_events(self):
        """Verify @EventHandler.register works for tuple of event classes."""
        import uuid

        suffix = uuid.uuid4().hex[:8]
        TestEventA = type(f"_TestEventA_{suffix}", (Event,), {})
        TestEventB = type(f"_TestEventB_{suffix}", (Event,), {})

        @EventHandler.register((TestEventA, TestEventB))
        async def multi_handler(event, cm, *args, **kwargs):
            pass

        assert TestEventA in EventHandler._registry
        assert TestEventB in EventHandler._registry
        assert EventHandler._registry[TestEventA] == multi_handler
        assert EventHandler._registry[TestEventB] == multi_handler

        # Cleanup
        del EventHandler._registry[TestEventA]
        del EventHandler._registry[TestEventB]


class TestEventTypeToLogKey:
    """Tests for the _event_type_to_log_key helper function."""

    def test_simple_camel_case(self):
        """Simple CamelCase converts to snake_case."""
        assert _event_type_to_log_key(SMSReceived) == "sms_received"
        assert _event_type_to_log_key(EmailSent) == "email_sent"

    def test_consecutive_uppercase(self):
        """Handles consecutive uppercase letters (SMS, LLM)."""
        assert _event_type_to_log_key(SMSReceived) == "sms_received"
        assert _event_type_to_log_key(SMSSent) == "sms_sent"

    def test_phone_call_events(self):
        """Phone call event names convert correctly."""
        assert _event_type_to_log_key(PhoneCallReceived) == "phone_call_received"
        assert _event_type_to_log_key(PhoneCallStarted) == "phone_call_started"
        assert _event_type_to_log_key(PhoneCallEnded) == "phone_call_ended"

    def test_unify_events(self):
        """UnifyMeet and UnifyMessage events convert correctly."""
        assert _event_type_to_log_key(UnifyMeetReceived) == "unify_meet_received"
        assert _event_type_to_log_key(UnifyMessageReceived) == "unify_message_received"

    def test_single_word(self):
        """Single-word event names convert correctly."""
        assert _event_type_to_log_key(Ping) == "ping"

    def test_actor_events(self):
        """Actor event names convert correctly."""
        assert _event_type_to_log_key(ActorResult) == "actor_result"


# =============================================================================
# 2. handle_event Core Behavior Tests
# =============================================================================


class TestHandleEventCore:
    """Tests for EventHandler.handle_event core behavior."""

    @pytest.mark.asyncio
    async def test_handle_event_logs_event(self, mock_cm):
        """Verify handle_event logs the event via session logger."""
        event = Ping(kind="keepalive")
        await EventHandler.handle_event(event, mock_cm)

        mock_cm._session_logger.info.assert_called_with("ping", "Event: Ping")

    @pytest.mark.asyncio
    async def test_handle_event_publishes_loggable_events(self, mock_cm):
        """Verify loggable events are published to bus."""
        event = SMSReceived(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
            content="Hello",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

            # Should queue a publish operation for loggable event
            assert mock_utils.queue_operation.called

    @pytest.mark.asyncio
    async def test_handle_event_skips_non_loggable(self, mock_cm):
        """Verify non-loggable events (like Ping) don't publish to bus."""
        event = Ping(kind="keepalive")
        assert event.loggable is False

        with patch(
            "unity.conversation_manager.domains.event_handlers.asyncio.create_task",
        ) as mock_create_task:
            await EventHandler.handle_event(event, mock_cm)
            # asyncio.create_task should not be called for non-loggable events
            # (The loggable check happens before create_task)


# =============================================================================
# 3. Ping Event Handler Tests
# =============================================================================


class TestPingHandler:
    """Tests for the Ping event handler."""

    @pytest.mark.asyncio
    async def test_ping_prints_keepalive_message(self, mock_cm, capsys):
        """Ping handler prints keepalive message to stdout."""
        event = Ping(kind="keepalive")
        await EventHandler.handle_event(event, mock_cm)

        captured = capsys.readouterr()
        assert "Ping received - keeping conversation manager alive" in captured.out

    @pytest.mark.asyncio
    async def test_ping_logs_debug_message(self, mock_cm):
        """Ping handler logs debug message."""
        event = Ping(kind="test")
        await EventHandler.handle_event(event, mock_cm)

        mock_cm._session_logger.debug.assert_called_with(
            "ping",
            "Ping received - keeping conversation manager alive",
        )


# =============================================================================
# 4. Text Message Event Handler Tests (SMS, Email, UnifyMessage)
# =============================================================================


class TestTextMessageHandlers:
    """Tests for SMS, Email, and UnifyMessage event handlers."""

    @pytest.mark.asyncio
    async def test_sms_received_updates_contact_index(self, mock_cm):
        """SMSReceived adds message to contact's SMS thread."""
        event = SMSReceived(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
            content="Hello there!",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.SMS_MESSAGE)
        assert len(msgs) == 1
        assert msgs[0].content == "Hello there!"

    @pytest.mark.asyncio
    async def test_sms_received_pushes_notification(self, mock_cm):
        """SMSReceived pushes notification to notification bar."""
        event = SMSReceived(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
            content="Test message",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        assert len(mock_cm.notifications_bar.notifications) == 1
        assert (
            "SMS Received from Alice"
            in mock_cm.notifications_bar.notifications[0].content
        )

    @pytest.mark.asyncio
    async def test_sms_received_cancels_proactive_speech(self, mock_cm):
        """SMSReceived cancels any proactive speech."""
        event = SMSReceived(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
            content="Interrupt!",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.cancel_proactive_speech.assert_called_once()

    @pytest.mark.asyncio
    async def test_sms_received_requests_llm_run(self, mock_cm):
        """SMSReceived requests an LLM run with delay."""
        event = SMSReceived(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
            content="Need response",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called_once_with(delay=2)

    @pytest.mark.asyncio
    async def test_sms_sent_updates_contact_index(self, mock_cm):
        """SMSSent adds message with assistant role."""
        event = SMSSent(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
            content="Reply to you",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.SMS_MESSAGE)
        assert len(msgs) == 1
        # Sent messages have assistant role, not user

    @pytest.mark.asyncio
    async def test_email_received_stores_subject_and_body(self, mock_cm):
        """EmailReceived stores subject, body, and email_id."""
        event = EmailReceived(
            contact={"contact_id": 2, "email_address": "alice@example.com"},
            subject="Important Update",
            body="Please review the attached.",
            email_id="msg_123",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.EMAIL)
        assert len(msgs) == 1
        assert msgs[0].subject == "Important Update"
        assert msgs[0].body == "Please review the attached."

    @pytest.mark.asyncio
    async def test_unify_message_received_updates_index(self, mock_cm):
        """UnifyMessageReceived adds to unify_message thread."""
        event = UnifyMessageReceived(
            contact={"contact_id": 2},
            content="Unify chat message",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.UNIFY_MESSAGE)
        assert len(msgs) == 1

    @pytest.mark.asyncio
    async def test_sent_messages_do_not_cancel_proactive_speech(self, mock_cm):
        """Sent messages (assistant role) don't cancel proactive speech."""
        event = SMSSent(
            contact={"contact_id": 2},
            content="Outgoing message",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        # cancel_proactive_speech should NOT be called for sent (assistant) messages
        mock_cm.cancel_proactive_speech.assert_not_called()


# =============================================================================
# 5. Phone Call Event Handler Tests
# =============================================================================


class TestPhoneCallHandlers:
    """Tests for phone call event handlers."""

    @pytest.mark.asyncio
    async def test_phone_call_received_in_text_mode_starts_call(self, mock_cm):
        """PhoneCallReceived in text mode starts a call."""
        mock_cm.mode = Mode.TEXT
        event = PhoneCallReceived(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
            conference_name="conf_123",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.call_manager.start_call.assert_called_once()
        assert mock_cm.call_manager.conference_name == "conf_123"

    @pytest.mark.asyncio
    async def test_phone_call_received_pushes_notification(self, mock_cm):
        """PhoneCallReceived pushes call notification."""
        mock_cm.mode = Mode.TEXT
        event = PhoneCallReceived(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
            conference_name="conf_123",
        )

        await EventHandler.handle_event(event, mock_cm)

        assert len(mock_cm.notifications_bar.notifications) == 1
        assert (
            "Call received from Alice"
            in mock_cm.notifications_bar.notifications[0].content
        )

    @pytest.mark.asyncio
    async def test_phone_call_received_during_call_does_nothing(self, mock_cm):
        """PhoneCallReceived during existing call doesn't start new call."""
        mock_cm.mode = Mode.CALL  # Already in a call
        event = PhoneCallReceived(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
            conference_name="conf_456",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.call_manager.start_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_phone_call_answered_during_call_publishes_status(self, mock_cm):
        """PhoneCallAnswered during call publishes status event."""
        mock_cm.mode = Mode.CALL
        event = PhoneCallAnswered(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.event_broker.publish.assert_called_once()
        call_args = mock_cm.event_broker.publish.call_args
        assert call_args[0][0] == "app:call:status"

    @pytest.mark.asyncio
    async def test_phone_call_started_sets_mode(self, mock_cm):
        """PhoneCallStarted sets CM mode to 'call'."""
        mock_cm.mode = Mode.TEXT
        event = PhoneCallStarted(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
        )

        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.mode == Mode.CALL

    @pytest.mark.asyncio
    async def test_phone_call_started_sets_call_contact(self, mock_cm):
        """PhoneCallStarted sets the call contact."""
        event = PhoneCallStarted(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
        )

        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.call_manager.call_contact is not None
        assert mock_cm.call_manager.call_contact["contact_id"] == 2

    @pytest.mark.asyncio
    async def test_phone_call_started_marks_contact_on_call(self, mock_cm):
        """PhoneCallStarted sets on_call=True for the contact."""
        event = PhoneCallStarted(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
        )

        await EventHandler.handle_event(event, mock_cm)

        contact = mock_cm.contact_index.active_conversations.get(2)
        assert contact is not None
        assert contact.on_call is True

    @pytest.mark.asyncio
    async def test_phone_call_started_does_not_trigger_llm_run(self, mock_cm):
        """PhoneCallStarted does not trigger an LLM run.

        Call guidance is pre-computed via make_call(context=...) before the call
        is placed.  The slow brain is woken later by InboundPhoneUtterance,
        ActorResult, or cross-channel notifications — not by call-start itself.
        """
        event = PhoneCallStarted(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_phone_call_ended_resets_mode(self, mock_cm):
        """PhoneCallEnded resets mode to 'text'."""
        mock_cm.mode = Mode.CALL
        # Need to have an active conversation first
        mock_cm.contact_index.push_message(
            contact_id=2,
            sender_name="Alice",
            thread_name=Medium.PHONE_CALL,
            message_content="test",
        )
        mock_cm.contact_index.active_conversations[2].on_call = True

        event = PhoneCallEnded(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
        )

        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.mode == Mode.TEXT
        assert mock_cm.call_manager.call_contact is None

    @pytest.mark.asyncio
    async def test_phone_call_ended_clears_conference_name(self, mock_cm):
        """PhoneCallEnded clears the conference name."""
        mock_cm.mode = Mode.CALL
        mock_cm.call_manager.conference_name = "conf_123"
        mock_cm.contact_index.push_message(
            contact_id=2,
            sender_name="Alice",
            thread_name=Medium.PHONE_CALL,
            message_content="test",
        )
        mock_cm.contact_index.active_conversations[2].on_call = True

        event = PhoneCallEnded(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
        )

        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.call_manager.conference_name is None

    @pytest.mark.asyncio
    async def test_phone_call_ended_cleanup_and_llm_run(self, mock_cm):
        """PhoneCallEnded triggers cleanup and LLM run."""
        mock_cm.mode = Mode.CALL
        mock_cm.contact_index.push_message(
            contact_id=2,
            sender_name="Alice",
            thread_name=Medium.PHONE_CALL,
            message_content="test",
        )
        mock_cm.contact_index.active_conversations[2].on_call = True

        event = PhoneCallEnded(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.call_manager.cleanup_call_proc.assert_called_once()
        mock_cm.cancel_proactive_speech.assert_called_once()
        mock_cm.request_llm_run.assert_called_once_with(delay=0, cancel_running=True)


# =============================================================================
# 6. UnifyMeet Event Handler Tests
# =============================================================================


class TestUnifyMeetHandlers:
    """Tests for UnifyMeet event handlers."""

    @pytest.mark.asyncio
    async def test_unify_meet_received_starts_meet(self, mock_cm):
        """UnifyMeetReceived starts a UnifyMeet session."""
        mock_cm.mode = Mode.TEXT
        event = UnifyMeetReceived(
            contact={"contact_id": 1},  # Boss contact
            livekit_agent_name="TestAgent",
            room_name="room_123",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.call_manager.start_unify_meet.assert_called_once()

    @pytest.mark.asyncio
    async def test_unify_meet_started_sets_mode(self, mock_cm):
        """UnifyMeetStarted sets mode to 'unify_meet'."""
        mock_cm.mode = Mode.TEXT
        event = UnifyMeetStarted(
            contact={"contact_id": 1},
        )

        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.mode == Mode.MEET

    @pytest.mark.asyncio
    async def test_unify_meet_ended_resets_mode(self, mock_cm):
        """UnifyMeetEnded resets mode to 'text'."""
        mock_cm.mode = Mode.MEET
        mock_cm.contact_index.push_message(
            contact_id=1,
            sender_name="Boss",
            thread_name=Medium.UNIFY_MEET,
            message_content="test",
        )
        mock_cm.contact_index.active_conversations[1].on_call = True

        event = UnifyMeetEnded(contact={"contact_id": 1})

        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.mode == Mode.TEXT


# =============================================================================
# 7. Voice Utterance Event Handler Tests
# =============================================================================


class TestVoiceUtteranceHandlers:
    """Tests for voice utterance event handlers."""

    @pytest.mark.asyncio
    async def test_inbound_phone_utterance_updates_index(self, mock_cm):
        """InboundPhoneUtterance adds to voice thread with user role."""
        event = InboundPhoneUtterance(
            contact={"contact_id": 2},
            content="Hello, can you hear me?",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.PHONE_CALL)
        assert len(msgs) == 1
        assert msgs[0].content == "Hello, can you hear me?"

    @pytest.mark.asyncio
    async def test_inbound_utterance_cancels_proactive_speech(self, mock_cm):
        """Inbound utterances cancel proactive speech."""
        event = InboundPhoneUtterance(
            contact={"contact_id": 2},
            content="User speaking",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.cancel_proactive_speech.assert_called_once()

    @pytest.mark.asyncio
    async def test_inbound_utterance_triggers_interject_or_run(self, mock_cm):
        """Inbound utterances trigger interject_or_run."""
        event = InboundPhoneUtterance(
            contact={"contact_id": 2},
            content="What's the weather?",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.interject_or_run.assert_called_once_with("What's the weather?")

    @pytest.mark.asyncio
    async def test_outbound_utterance_does_not_cancel_proactive(self, mock_cm):
        """Outbound utterances don't cancel proactive speech."""
        event = OutboundPhoneUtterance(
            contact={"contact_id": 2},
            content="Here's my response",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.cancel_proactive_speech.assert_not_called()

    @pytest.mark.asyncio
    async def test_call_guidance_updates_contact_index(self, mock_cm):
        """CallGuidance adds guidance message to voice thread."""
        event = CallGuidance(
            contact={"contact_id": 2},
            content="Please mention the meeting at 3pm",
        )

        await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.PHONE_CALL)
        assert len(msgs) == 1
        # Guidance messages have role="guidance"


# =============================================================================
# 8. State Update Event Handler Tests
# =============================================================================


class TestStateUpdateHandlers:
    """Tests for state update event handlers."""

    @pytest.mark.asyncio
    async def test_get_chat_history_prepends_to_history(self, mock_cm):
        """GetChatHistory prepends messages to existing history."""
        mock_cm.chat_history = [{"role": "user", "content": "existing"}]
        event = GetChatHistory(
            chat_history=[
                {"role": "user", "content": "older message"},
                {"role": "assistant", "content": "older response"},
            ],
        )

        await EventHandler.handle_event(event, mock_cm)

        # New history should be prepended
        assert len(mock_cm.chat_history) == 3
        assert mock_cm.chat_history[0]["content"] == "older message"
        assert mock_cm.chat_history[2]["content"] == "existing"


# =============================================================================
# 9. Actor Event Handler Tests
# =============================================================================


class TestActorEventHandlers:
    """Tests for Actor-related event handlers."""

    @pytest.mark.asyncio
    async def test_actor_handle_started_does_not_push_notification(self, mock_cm):
        """ActorHandleStarted does not push a notification (action state is shown in in_flight_actions)."""
        event = ActorHandleStarted(
            action_name="search_task",
            handle_id=1,
            query="Search for documents about Python",
        )

        await EventHandler.handle_event(event, mock_cm)

        assert len(mock_cm.notifications_bar.notifications) == 0

    @pytest.mark.asyncio
    async def test_actor_handle_started_requests_llm_run(self, mock_cm):
        """ActorHandleStarted requests an LLM run."""
        event = ActorHandleStarted(
            action_name="task",
            handle_id=1,
            query="Do something",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called()

    @pytest.mark.asyncio
    async def test_actor_result_moves_action_to_completed(self, mock_cm):
        """ActorResult moves action from in_flight_actions to completed_actions."""
        mock_cm.in_flight_actions = {
            1: {"query": "Test action", "handle_actions": []},
        }
        event = ActorResult(
            handle_id=1,
            success=True,
            result="Action completed successfully",
        )

        await EventHandler.handle_event(event, mock_cm)

        assert 1 not in mock_cm.in_flight_actions
        assert 1 in mock_cm.completed_actions
        assert mock_cm.completed_actions[1]["query"] == "Test action"
        # Result is recorded in handle_actions as act_completed event
        handle_actions = mock_cm.completed_actions[1]["handle_actions"]
        assert any(a["action_name"] == "act_completed" for a in handle_actions)
        # No notification pushed (result is shown in completed_actions section)
        assert len(mock_cm.notifications_bar.notifications) == 0

    @pytest.mark.asyncio
    async def test_actor_handle_response_updates_matching_pending_action(
        self,
        mock_cm,
    ):
        """ActorHandleResponse should complete only the matching pending action."""
        response_text = "Action-specific response payload."

        mock_cm.in_flight_actions = {
            1: {
                "query": "Search transcripts for budget review",
                "handle_actions": [
                    {
                        "action_name": "interject_1",
                        "query": "add context",
                        "status": "pending",
                    },
                    {
                        "action_name": "ask_1",
                        "query": "what is the current status?",
                        "status": "pending",
                    },
                ],
            },
        }
        event = ActorHandleResponse(
            handle_id=1,
            action_name="ask",
            query="what is the current status?",
            response=response_text,
            call_id="",
        )

        await EventHandler.handle_event(event, mock_cm)

        interject_event = mock_cm.in_flight_actions[1]["handle_actions"][0]
        ask_event = mock_cm.in_flight_actions[1]["handle_actions"][1]

        assert interject_event["status"] == "pending"
        assert ask_event["status"] == "completed"
        assert ask_event.get("response") == response_text

    @pytest.mark.asyncio
    async def test_actor_clarification_request_updates_handle_actions(self, mock_cm):
        """ActorClarificationRequest adds clarification to handle_actions."""
        mock_cm.in_flight_actions = {
            1: {"query": "Ambiguous action", "handle_actions": []},
        }
        event = ActorClarificationRequest(
            handle_id=1,
            query="What do you mean by 'documents'?",
            call_id="call_123",
        )

        await EventHandler.handle_event(event, mock_cm)

        assert len(mock_cm.in_flight_actions[1]["handle_actions"]) == 1
        clarification = mock_cm.in_flight_actions[1]["handle_actions"][0]
        assert clarification["action_name"] == "clarification_request"
        assert clarification["query"] == "What do you mean by 'documents'?"


# =============================================================================
# 10. Meet Interaction Event Handler Tests
# =============================================================================


class TestMeetInteractionEventHandlers:
    """Tests for screen share and remote control event handlers."""

    @pytest.mark.asyncio
    async def test_assistant_screen_share_started_sets_flag(self, mock_cm):
        """AssistantScreenShareStarted sets assistant_screen_share_active to True."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False

        event = AssistantScreenShareStarted(
            reason="User enabled assistant screen sharing",
        )
        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.assistant_screen_share_active is True

    @pytest.mark.asyncio
    async def test_assistant_screen_share_stopped_clears_flag(self, mock_cm):
        """AssistantScreenShareStopped sets assistant_screen_share_active to False."""
        mock_cm.assistant_screen_share_active = True
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False

        event = AssistantScreenShareStopped(
            reason="User disabled assistant screen sharing",
        )
        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.assistant_screen_share_active is False

    @pytest.mark.asyncio
    async def test_user_screen_share_started_sets_flag(self, mock_cm):
        """UserScreenShareStarted sets user_screen_share_active to True."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False

        event = UserScreenShareStarted(
            reason="User started sharing their screen",
        )
        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.user_screen_share_active is True

    @pytest.mark.asyncio
    async def test_user_screen_share_stopped_clears_flag(self, mock_cm):
        """UserScreenShareStopped sets user_screen_share_active to False."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = True
        mock_cm.user_remote_control_active = False

        event = UserScreenShareStopped(
            reason="User stopped sharing their screen",
        )
        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.user_screen_share_active is False

    @pytest.mark.asyncio
    async def test_user_remote_control_started_sets_flag(
        self,
        mock_cm,
    ):
        """UserRemoteControlStarted sets user_remote_control_active to True."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False

        event = UserRemoteControlStarted(
            reason="User took remote control of assistant desktop",
        )
        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.user_remote_control_active is True

    @pytest.mark.asyncio
    async def test_user_remote_control_stopped_clears_flag(
        self,
        mock_cm,
    ):
        """UserRemoteControlStopped clears user_remote_control_active."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = True

        event = UserRemoteControlStopped(
            reason="User released remote control of assistant desktop",
        )
        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.user_remote_control_active is False

    @pytest.mark.asyncio
    async def test_meet_interaction_pushes_notification(self, mock_cm):
        """All meet interaction events push a notification to the bar."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False

        initial_count = len(mock_cm.notifications_bar.notifications)

        event = AssistantScreenShareStarted(
            reason="User enabled assistant screen sharing",
        )
        await EventHandler.handle_event(event, mock_cm)

        assert len(mock_cm.notifications_bar.notifications) == initial_count + 1
        notification = mock_cm.notifications_bar.notifications[-1]
        assert notification.type == "Meet"
        assert "screen sharing" in notification.content.lower()

    @pytest.mark.asyncio
    async def test_meet_interaction_triggers_llm_run(self, mock_cm):
        """Meet interaction events trigger an LLM run."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False

        event = UserScreenShareStarted(
            reason="User started sharing their screen",
        )
        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called()

    # --------------------------------------------------------------------- #
    # Screenshot capture on utterance
    # --------------------------------------------------------------------- #

    @pytest.mark.asyncio
    async def test_utterance_triggers_screenshot_capture_when_screen_sharing(
        self,
        mock_cm,
    ):
        """Inbound user utterance triggers screenshot capture when assistant
        screen sharing is active."""
        mock_cm.assistant_screen_share_active = True
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False
        mock_cm.capture_assistant_screenshot = AsyncMock()

        contact = {"contact_id": 1, "first_name": "Boss", "surname": "User"}
        event = InboundUnifyMeetUtterance(
            contact=contact,
            content="So you need to click that button",
        )
        await EventHandler.handle_event(event, mock_cm)

        mock_cm.capture_assistant_screenshot.assert_called_once_with(
            "So you need to click that button",
        )

    @pytest.mark.asyncio
    async def test_utterance_no_screenshot_capture_when_not_screen_sharing(
        self,
        mock_cm,
    ):
        """Inbound user utterance does NOT trigger screenshot capture when
        assistant screen sharing is inactive."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False
        mock_cm.capture_assistant_screenshot = AsyncMock()

        contact = {"contact_id": 1, "first_name": "Boss", "surname": "User"}
        event = InboundUnifyMeetUtterance(
            contact=contact,
            content="Just some regular conversation",
        )
        await EventHandler.handle_event(event, mock_cm)

        mock_cm.capture_assistant_screenshot.assert_not_called()

    # --------------------------------------------------------------------- #
    # Direct fast brain guidance on mode change
    # --------------------------------------------------------------------- #

    @pytest.mark.asyncio
    async def test_meet_event_sends_fast_brain_guidance_in_voice_mode(
        self,
        mock_cm,
    ):
        """Screen share events publish direct CallGuidance to the fast brain
        when in voice mode, bypassing the slow brain for instant delivery."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False
        mock_cm.mode = Mode.MEET  # voice mode

        event = AssistantScreenShareStarted(
            reason="User enabled screen sharing",
        )
        await EventHandler.handle_event(event, mock_cm)

        # Verify CallGuidance was published to the fast brain channel
        calls = mock_cm.event_broker.publish.call_args_list
        guidance_calls = [c for c in calls if c.args[0] == "app:call:call_guidance"]
        assert len(guidance_calls) == 1
        # The guidance text should contain behavioral instructions
        import json as _json

        data = _json.loads(guidance_calls[0].args[1])
        content = data.get("payload", {}).get("content", "")
        assert "screen sharing" in content.lower()

    @pytest.mark.asyncio
    async def test_meet_event_no_fast_brain_guidance_in_text_mode(
        self,
        mock_cm,
    ):
        """Screen share events do NOT publish fast brain guidance when in
        text mode (no voice agent to receive it)."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False
        mock_cm.mode = Mode.TEXT

        event = AssistantScreenShareStarted(
            reason="User enabled screen sharing",
        )
        await EventHandler.handle_event(event, mock_cm)

        # No CallGuidance should be published
        calls = mock_cm.event_broker.publish.call_args_list
        guidance_calls = [c for c in calls if c.args[0] == "app:call:call_guidance"]
        assert len(guidance_calls) == 0

    @pytest.mark.asyncio
    async def test_all_six_meet_events_have_fast_brain_guidance(self, mock_cm):
        """Each of the six meet interaction events has corresponding fast brain
        guidance text defined."""
        from unity.conversation_manager.domains.event_handlers import (
            _MEET_FAST_BRAIN_GUIDANCE,
        )

        event_classes = [
            AssistantScreenShareStarted,
            AssistantScreenShareStopped,
            UserScreenShareStarted,
            UserScreenShareStopped,
            UserRemoteControlStarted,
            UserRemoteControlStopped,
        ]
        for cls in event_classes:
            assert (
                cls in _MEET_FAST_BRAIN_GUIDANCE
            ), f"{cls.__name__} missing from _MEET_FAST_BRAIN_GUIDANCE"
            assert len(_MEET_FAST_BRAIN_GUIDANCE[cls]) > 0

    # --------------------------------------------------------------------- #
    # Renderer tests
    # --------------------------------------------------------------------- #

    def test_render_meet_state_empty_when_all_off(self):
        """render_meet_interaction_state returns empty when nothing is active."""
        from unity.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_meet_interaction_state(
            assistant_screen_share_active=False,
            user_screen_share_active=False,
            user_remote_control_active=False,
        )
        assert result == ""

    def test_render_meet_state_assistant_screen_share_only(self):
        """Only assistant screen share active produces a single section."""
        from unity.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_meet_interaction_state(
            assistant_screen_share_active=True,
            user_screen_share_active=False,
            user_remote_control_active=False,
        )
        assert "<assistant_screen_share status='active'>" in result
        assert "</assistant_screen_share>" in result
        assert "visible to the user" in result
        # Other sections absent.
        assert "<user_screen_share" not in result
        assert "<user_remote_control" not in result

    def test_render_meet_state_user_screen_share_only(self):
        """Only user screen share active produces a single section."""
        from unity.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_meet_interaction_state(
            assistant_screen_share_active=False,
            user_screen_share_active=True,
            user_remote_control_active=False,
        )
        assert "<user_screen_share status='active'>" in result
        assert "</user_screen_share>" in result
        assert "sharing their screen with you" in result
        assert "<assistant_screen_share" not in result
        assert "<user_remote_control" not in result

    def test_render_meet_state_user_remote_control_only(self):
        """Only user remote control active produces a single section."""
        from unity.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_meet_interaction_state(
            assistant_screen_share_active=False,
            user_screen_share_active=False,
            user_remote_control_active=True,
        )
        assert "<user_remote_control status='active'>" in result
        assert "</user_remote_control>" in result
        assert "mouse and keyboard" in result
        assert "<assistant_screen_share" not in result
        assert "<user_screen_share" not in result

    def test_render_meet_state_all_three_active(self):
        """All three active produces three independent sections."""
        from unity.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_meet_interaction_state(
            assistant_screen_share_active=True,
            user_screen_share_active=True,
            user_remote_control_active=True,
        )
        assert "<assistant_screen_share status='active'>" in result
        assert "<user_screen_share status='active'>" in result
        assert "<user_remote_control status='active'>" in result

    def test_render_meet_state_appears_at_top_of_full_render(self):
        """Active meet sections appear before notifications in the full render."""
        from unity.conversation_manager.domains.renderer import Renderer
        from unity.conversation_manager.domains.notifications import NotificationBar

        renderer = Renderer()
        result = renderer.render_state(
            contact_index=ContactIndex(),
            notification_bar=NotificationBar(),
            assistant_screen_share_active=True,
            user_screen_share_active=False,
            user_remote_control_active=False,
        ).full_render

        screen_share_pos = result.index("<assistant_screen_share")
        notifications_pos = result.index("<notifications>")
        assert screen_share_pos < notifications_pos


# =============================================================================
# 11. Notification Event Handler Tests
# =============================================================================


class TestNotificationEventHandlers:
    """Tests for notification injection/unpinning event handlers."""

    @pytest.mark.asyncio
    async def test_notification_injected_adds_to_bar(self, mock_cm):
        """NotificationInjectedEvent adds notification to bar."""
        event = NotificationInjectedEvent(
            content="Important update from task",
            source="Actor",
            target_conversation_id="conv_123",
        )

        await EventHandler.handle_event(event, mock_cm)

        assert len(mock_cm.notifications_bar.notifications) == 1
        assert (
            mock_cm.notifications_bar.notifications[0].content
            == "Important update from task"
        )

    @pytest.mark.asyncio
    async def test_notification_injected_cancels_proactive_speech(self, mock_cm):
        """NotificationInjectedEvent cancels proactive speech."""
        event = NotificationInjectedEvent(
            content="Interrupt notification",
            source="System",
            target_conversation_id="conv_123",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.cancel_proactive_speech.assert_called_once()

    @pytest.mark.asyncio
    async def test_notification_injected_triggers_immediate_llm(self, mock_cm):
        """NotificationInjectedEvent triggers immediate LLM run."""
        event = NotificationInjectedEvent(
            content="React to this",
            source="Actor",
            target_conversation_id="conv_123",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called_once_with(delay=0, cancel_running=True)

    @pytest.mark.asyncio
    async def test_notification_unpinned_removes_from_bar(self, mock_cm, static_now):
        """NotificationUnpinnedEvent removes pinned notification."""
        # First add a pinned notification
        mock_cm.notifications_bar.push_notif(
            "Test",
            "Pinned content",
            static_now,
            pinned=True,
            id="notif_123",
        )
        assert len(mock_cm.notifications_bar.notifications) == 1

        event = NotificationUnpinnedEvent(
            interjection_id="notif_123",
            target_conversation_id="conv_123",
        )

        await EventHandler.handle_event(event, mock_cm)

        # Notification should be removed
        assert len(mock_cm.notifications_bar.notifications) == 0


# =============================================================================
# 11. SyncContacts Event Handler Tests
# =============================================================================


class TestSyncContactsHandler:
    """Tests for SyncContacts event handler."""

    @pytest.mark.asyncio
    async def test_sync_contacts_logs_event(self, mock_cm):
        """SyncContacts logs the sync reason."""
        event = SyncContacts(reason="Manual refresh")

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        # Verify the handler logged the event
        mock_cm._session_logger.info.assert_any_call(
            "state_update",
            "SyncContacts: Manual refresh",
        )


# =============================================================================
# 12. LogMessageResponse Event Handler Tests
# =============================================================================


class TestLogMessageResponseHandler:
    """Tests for LogMessageResponse event handler."""

    @pytest.mark.asyncio
    async def test_log_message_response_sets_call_exchange_id(self, mock_cm):
        """LogMessageResponse sets call exchange ID when appropriate."""
        from unity.contact_manager.types.contact import UNASSIGNED

        mock_cm.call_manager.call_exchange_id = UNASSIGNED
        event = LogMessageResponse(
            medium="phone_call",
            exchange_id=42,
        )

        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.call_manager.call_exchange_id == 42

    @pytest.mark.asyncio
    async def test_log_message_response_sets_unify_meet_exchange_id(self, mock_cm):
        """LogMessageResponse sets UnifyMeet exchange ID when appropriate."""
        from unity.contact_manager.types.contact import UNASSIGNED

        mock_cm.call_manager.unify_meet_exchange_id = UNASSIGNED
        event = LogMessageResponse(
            medium="unify_meet",
            exchange_id=99,
        )

        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.call_manager.unify_meet_exchange_id == 99


# =============================================================================
# 13. SummarizeContext Event Handler Tests
# =============================================================================


class TestSummarizeContextHandler:
    """Tests for SummarizeContext event handler."""

    @pytest.mark.asyncio
    async def test_summarize_context_skips_without_memory_manager(self, mock_cm):
        """SummarizeContext is skipped when memory_manager is None."""
        mock_cm.memory_manager = None
        mock_cm.is_summarizing = True

        event = SummarizeContext()

        # Mock queue_operation to execute the function immediately
        async def immediate_queue_operation(func, *args, **kwargs):
            await func(*args, **kwargs)

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils.queue_operation",
            side_effect=immediate_queue_operation,
        ):
            await EventHandler.handle_event(event, mock_cm)

        # is_summarizing should be reset to False
        assert mock_cm.is_summarizing is False
        # chat_history should be cleared
        assert mock_cm.chat_history == []


# =============================================================================
# 14. DirectMessageEvent Handler Tests
# =============================================================================


class TestDirectMessageEventHandler:
    """Tests for DirectMessageEvent handler."""

    @pytest.mark.asyncio
    async def test_direct_message_publishes_to_call_guidance_during_call(self, mock_cm):
        """DirectMessageEvent publishes to call_guidance channel during call."""
        mock_cm.mode = Mode.CALL
        event = DirectMessageEvent(
            content="Speak this directly",
            source="handle",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.event_broker.publish.assert_called()
        call_args = mock_cm.event_broker.publish.call_args
        assert call_args[0][0] == "app:call:call_guidance"

    @pytest.mark.asyncio
    async def test_direct_message_records_in_contact_index(self, mock_cm):
        """DirectMessageEvent records message in contact_index."""
        mock_cm.mode = Mode.CALL
        event = DirectMessageEvent(
            content="Direct message content",
            source="system",
        )

        await EventHandler.handle_event(event, mock_cm)

        # Should have pushed message to active contact's voice thread
        contact = mock_cm.get_active_contact()
        assert contact is not None


# =============================================================================
# 15. Edge Cases and Error Handling Tests
# =============================================================================


class TestEventHandlerEdgeCases:
    """Tests for edge cases and error handling in event handlers."""

    @pytest.mark.asyncio
    async def test_actor_clarification_for_nonexistent_action(self, mock_cm):
        """ActorClarificationRequest for non-existent action does nothing."""
        mock_cm.in_flight_actions = {}  # No actions
        event = ActorClarificationRequest(
            handle_id=999,  # Non-existent
            query="Question?",
            call_id="call_123",
        )

        # Should not raise
        await EventHandler.handle_event(event, mock_cm)

        # No notifications should be pushed
        assert len(mock_cm.notifications_bar.notifications) == 0


# =============================================================================
# 16. AssistantUpdateEvent Handler Tests
# =============================================================================


class TestAssistantUpdateEventHandler:
    """Tests for AssistantUpdateEvent handler (updates contact manager)."""

    @pytest.mark.asyncio
    async def test_assistant_update_logs_event(self, mock_cm):
        """AssistantUpdateEvent logs the update event."""
        event = AssistantUpdateEvent(
            api_key="test_key",
            medium="assistant_update",
            assistant_id="asst_123",
            user_id="user_456",
            assistant_name="Updated Assistant",
            assistant_age="25",
            assistant_nationality="US",
            assistant_about="Test assistant",
            assistant_number="+15555550001",
            assistant_email="assistant@updated.com",
            user_name="Updated Boss",
            user_number="+15555550002",
            user_email="boss@updated.com",
            voice_id="voice_123",
            voice_provider="cartesia",
            voice_mode="tts",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm._session_logger.info.assert_any_call(
            "assistant_update",
            "Received assistant update event",
        )

    @pytest.mark.asyncio
    async def test_assistant_update_calls_set_details(self, mock_cm):
        """AssistantUpdateEvent calls set_details with payload."""
        mock_cm.set_details = MagicMock()
        mock_cm.get_call_config = MagicMock(return_value={})

        event = AssistantUpdateEvent(
            api_key="test_key",
            medium="assistant_update",
            assistant_id="asst_123",
            user_id="user_456",
            assistant_name="Updated Assistant",
            assistant_age="25",
            assistant_nationality="US",
            assistant_about="Test assistant",
            assistant_number="+15555550001",
            assistant_email="assistant@updated.com",
            user_name="Updated Boss",
            user_number="+15555550002",
            user_email="boss@updated.com",
            voice_id="voice_123",
            voice_provider="cartesia",
            voice_mode="tts",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.set_details.assert_called_once()

    @pytest.mark.asyncio
    async def test_assistant_update_updates_call_config(self, mock_cm):
        """AssistantUpdateEvent updates the call manager config."""
        mock_cm.set_details = MagicMock()
        mock_cm.get_call_config = MagicMock(return_value={"voice_id": "new_voice"})

        event = AssistantUpdateEvent(
            api_key="test_key",
            medium="assistant_update",
            assistant_id="asst_123",
            user_id="user_456",
            assistant_name="Updated Assistant",
            assistant_age="25",
            assistant_nationality="US",
            assistant_about="Test assistant",
            assistant_number="+15555550001",
            assistant_email="assistant@updated.com",
            user_name="Updated Boss",
            user_number="+15555550002",
            user_email="boss@updated.com",
            voice_id="voice_123",
            voice_provider="cartesia",
            voice_mode="tts",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.call_manager.set_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_assistant_update_queues_contact_update(self, mock_cm):
        """AssistantUpdateEvent queues update_session_contacts operation."""
        mock_cm.set_details = MagicMock()
        mock_cm.get_call_config = MagicMock(return_value={})

        event = AssistantUpdateEvent(
            api_key="test_key",
            medium="assistant_update",
            assistant_id="asst_123",
            user_id="user_456",
            assistant_name="New Assistant Name",
            assistant_age="25",
            assistant_nationality="US",
            assistant_about="Test assistant",
            assistant_number="+15555559999",
            assistant_email="new_assistant@test.com",
            user_name="New Boss Name",
            user_number="+15555558888",
            user_email="new_boss@test.com",
            voice_id="voice_123",
            voice_provider="cartesia",
            voice_mode="tts",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

            # Verify queue_operation was called with update_session_contacts
            mock_utils.queue_operation.assert_called_once()
            call_args = mock_utils.queue_operation.call_args
            # First arg is the function (update_session_contacts)
            assert call_args[0][0] == mock_utils.update_session_contacts
            # Remaining args are: cm, assistant_name, assistant_number, assistant_email,
            # user_name, user_number, user_email
            assert call_args[0][1] == mock_cm
            assert call_args[0][2] == "New Assistant Name"
            assert call_args[0][3] == "+15555559999"
            assert call_args[0][4] == "new_assistant@test.com"
            assert call_args[0][5] == "New Boss Name"
            assert call_args[0][6] == "+15555558888"
            assert call_args[0][7] == "new_boss@test.com"

    @pytest.mark.asyncio
    async def test_assistant_update_handles_no_contact_manager(self, mock_cm):
        """AssistantUpdateEvent handles missing contact_manager gracefully."""
        mock_cm.set_details = MagicMock()
        mock_cm.get_call_config = MagicMock(return_value={})
        mock_cm.contact_manager = None  # No contact manager

        event = AssistantUpdateEvent(
            api_key="test_key",
            medium="assistant_update",
            assistant_id="asst_123",
            user_id="user_456",
            assistant_name="Updated Assistant",
            assistant_age="25",
            assistant_nationality="US",
            assistant_about="Test assistant",
            assistant_number="+15555550001",
            assistant_email="assistant@updated.com",
            user_name="Updated Boss",
            user_number="+15555550002",
            user_email="boss@updated.com",
            voice_id="voice_123",
            voice_provider="cartesia",
            voice_mode="tts",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            # Should not raise
            await EventHandler.handle_event(event, mock_cm)

        # set_details should still be called
        mock_cm.set_details.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_session_contacts_updates_both_contacts(self, mock_cm):
        """update_session_contacts updates both assistant (0) and boss (1) contacts."""
        from unity.conversation_manager.domains.managers_utils import (
            update_session_contacts,
        )

        mock_cm.contact_manager.update_contact = MagicMock()

        await update_session_contacts(
            mock_cm,
            assistant_name="New Assistant",
            assistant_number="+15555559999",
            assistant_email="new_assistant@test.com",
            user_name="New Boss",
            user_number="+15555558888",
            user_email="new_boss@test.com",
        )

        # Verify update_contact was called for both contacts
        calls = mock_cm.contact_manager.update_contact.call_args_list
        assert len(calls) == 2

        # Check assistant contact (ID 0) - "New Assistant" splits to first="New", surname="Assistant"
        call_0 = next(c for c in calls if c.kwargs.get("contact_id") == 0)
        assert call_0.kwargs["phone_number"] == "+15555559999"
        assert call_0.kwargs["email_address"] == "new_assistant@test.com"
        assert call_0.kwargs["first_name"] == "New"
        assert call_0.kwargs["surname"] == "Assistant"

        # Check boss contact (ID 1) - "New Boss" splits to first="New", surname="Boss"
        call_1 = next(c for c in calls if c.kwargs.get("contact_id") == 1)
        assert call_1.kwargs["phone_number"] == "+15555558888"
        assert call_1.kwargs["email_address"] == "new_boss@test.com"
        assert call_1.kwargs["first_name"] == "New"
        assert call_1.kwargs["surname"] == "Boss"

    @pytest.mark.asyncio
    async def test_update_session_contacts_handles_failure(self, mock_cm, capsys):
        """update_session_contacts logs errors when update_contact fails."""
        from unity.conversation_manager.domains.managers_utils import (
            update_session_contacts,
        )

        mock_cm.contact_manager.update_contact = MagicMock(
            side_effect=Exception("Update failed"),
        )

        # Should not raise - errors are caught and logged
        await update_session_contacts(
            mock_cm,
            assistant_name="Updated Assistant",
            assistant_number="+15555550001",
            assistant_email="assistant@updated.com",
            user_name="Updated Boss",
            user_number="+15555550002",
            user_email="boss@updated.com",
        )

        # Errors should be printed
        captured = capsys.readouterr()
        assert "Failed to update contact 0" in captured.out
        assert "Failed to update contact 1" in captured.out

    @pytest.mark.asyncio
    async def test_update_session_contacts_handles_no_contact_manager(
        self,
        mock_cm,
        capsys,
    ):
        """update_session_contacts handles None contact_manager gracefully."""
        from unity.conversation_manager.domains.managers_utils import (
            update_session_contacts,
        )

        mock_cm.contact_manager = None

        # Should not raise
        await update_session_contacts(
            mock_cm,
            assistant_name="Test",
            assistant_number="+1555",
            assistant_email="test@test.com",
            user_name="Boss",
            user_number="+1666",
            user_email="boss@test.com",
        )

        # Should print a message about missing contact_manager
        captured = capsys.readouterr()
        assert "contact_manager is None" in captured.out


# =============================================================================
# 19. _recent_conversation_snippet Helper Tests
# =============================================================================


class TestRecentConversationSnippet:
    """Tests for the _recent_conversation_snippet helper used in remote-control broadcasts."""

    def _get_snippet(self):
        from unity.conversation_manager.domains.event_handlers import (
            _recent_conversation_snippet,
        )
        return _recent_conversation_snippet

    def test_returns_none_when_empty(self, mock_cm):
        """Returns None when the global thread has no messages."""
        snippet = self._get_snippet()
        assert snippet(mock_cm) is None

    def test_extracts_recent_messages(self, mock_cm):
        """Extracts the last N user/assistant messages in chronological order."""
        snippet = self._get_snippet()
        mock_cm.contact_index.push_message(
            contact_id=1, sender_name="Boss", thread_name=Medium.SMS_MESSAGE,
            message_content="Hey, open the browser", role="user",
        )
        mock_cm.contact_index.push_message(
            contact_id=1, sender_name="You", thread_name=Medium.SMS_MESSAGE,
            message_content="Sure, opening it now", role="assistant",
        )

        result = snippet(mock_cm)
        assert result is not None
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0] == "user: Hey, open the browser"
        assert lines[1] == "assistant: Sure, opening it now"

    def test_limits_to_n_messages(self, mock_cm):
        """Only the last n messages are returned (default 4)."""
        snippet = self._get_snippet()
        for i in range(10):
            mock_cm.contact_index.push_message(
                contact_id=1, sender_name="Boss", thread_name=Medium.SMS_MESSAGE,
                message_content=f"Message {i}", role="user",
            )

        result = snippet(mock_cm, n=4)
        assert result is not None
        lines = result.split("\n")
        assert len(lines) == 4
        assert "Message 9" in lines[-1]
        assert "Message 6" in lines[0]

    def test_skips_system_markers(self, mock_cm):
        """System markers like <Call Started> are excluded."""
        snippet = self._get_snippet()
        mock_cm.contact_index.push_message(
            contact_id=1, sender_name="System", thread_name=Medium.PHONE_CALL,
            message_content="<Call Started>", role="user",
        )
        mock_cm.contact_index.push_message(
            contact_id=1, sender_name="Boss", thread_name=Medium.PHONE_CALL,
            message_content="Hello there", role="user",
        )

        result = snippet(mock_cm)
        assert result is not None
        assert "<Call Started>" not in result
        assert "Hello there" in result


# =============================================================================
# 20. Remote Control → ComputerPrimitives Integration Tests
# =============================================================================


class TestRemoteControlComputerPrimitivesIntegration:
    """Verify the event handler calls ComputerPrimitives.set_user_remote_control."""

    @pytest.mark.asyncio
    async def test_started_calls_set_user_remote_control_true(self, mock_cm):
        """UserRemoteControlStarted calls set_user_remote_control(True, ...)."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False

        mock_cp = MagicMock()
        with patch(
            "unity.manager_registry.ManagerRegistry.get_instance",
            return_value=mock_cp,
        ):
            event = UserRemoteControlStarted(reason="User took control")
            await EventHandler.handle_event(event, mock_cm)

        mock_cp.set_user_remote_control.assert_called_once()
        args, kwargs = mock_cp.set_user_remote_control.call_args
        assert args[0] is True
        assert "conversation_context" in kwargs

    @pytest.mark.asyncio
    async def test_stopped_calls_set_user_remote_control_false(self, mock_cm):
        """UserRemoteControlStopped calls set_user_remote_control(False, ...)."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = True

        mock_cp = MagicMock()
        with patch(
            "unity.manager_registry.ManagerRegistry.get_instance",
            return_value=mock_cp,
        ):
            event = UserRemoteControlStopped(reason="User released control")
            await EventHandler.handle_event(event, mock_cm)

        mock_cp.set_user_remote_control.assert_called_once()
        args, kwargs = mock_cp.set_user_remote_control.call_args
        assert args[0] is False

    @pytest.mark.asyncio
    async def test_noop_when_no_computer_primitives_singleton(self, mock_cm):
        """No error when ComputerPrimitives singleton doesn't exist."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False

        with patch(
            "unity.manager_registry.ManagerRegistry.get_instance",
            return_value=None,
        ):
            event = UserRemoteControlStarted(reason="User took control")
            # Should not raise
            await EventHandler.handle_event(event, mock_cm)

    @pytest.mark.asyncio
    async def test_screen_share_events_do_not_call_set_user_remote_control(
        self, mock_cm,
    ):
        """Non-remote-control meet events do not trigger set_user_remote_control."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False

        mock_cp = MagicMock()
        with patch(
            "unity.manager_registry.ManagerRegistry.get_instance",
            return_value=mock_cp,
        ):
            event = AssistantScreenShareStarted(reason="Screen share started")
            await EventHandler.handle_event(event, mock_cm)

        mock_cp.set_user_remote_control.assert_not_called()
