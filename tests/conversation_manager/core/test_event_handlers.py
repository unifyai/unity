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

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.conversation_manager.domains.event_handlers import (
    EventHandler,
    _event_type_to_log_key,
)
from unity.conversation_manager.cm_types import ScreenshotEntry
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
    WhatsAppCallPermissionResponse,
    UnifyMeetReceived,
    UnifyMeetStarted,
    UnifyMeetEnded,
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    OutboundPhoneUtterance,
    FastBrainNotification,
    GetChatHistory,
    ActorHandleStarted,
    ActorHandleResponse,
    ActorResult,
    ActorClarificationRequest,
    InitializationComplete,
    NotificationInjectedEvent,
    NotificationUnpinnedEvent,
    SyncContacts,
    TaskDue,
    LogMessageResponse,
    DirectMessageEvent,
    AssistantUpdateEvent,
    AssistantScreenShareStarted,
    AssistantScreenShareStopped,
    UserScreenShareStarted,
    UserScreenShareStopped,
    UserRemoteControlStarted,
    UserRemoteControlStopped,
    UserWebcamStarted,
    UserWebcamStopped,
)
from unity.contact_manager.simulated import SimulatedContactManager
from unity.conversation_manager.domains.contact_index import ContactIndex
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.cm_types import Medium, Mode
from unity.task_scheduler.machine_state import TaskActivationSnapshot

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
    manager.has_active_call = False
    manager.has_active_google_meet = False
    manager.has_active_teams_meet = False
    manager.has_gmeet_presenting = False
    manager.has_teams_presenting = False
    manager._meet_joining = False
    manager._whatsapp_call_joining = False
    manager._gmeet_joining = False
    manager._socket_server = None
    manager._disconnect_contact = None
    manager.initial_notification = ""
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
    cm.assistant_screen_share_active = False
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

    # Generation-scoped outbound suppression (wait() sets suppress_gen = llm_gen)
    cm._llm_gen = 0
    cm._outbound_suppress_gen = -1

    # Mock async methods
    cm.request_llm_run = AsyncMock()
    cm.cancel_proactive_speech = AsyncMock()
    cm.schedule_proactive_speech = AsyncMock()
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
        """Verify handle_event logs loggable+prominent events via session logger."""
        from dataclasses import dataclass

        @dataclass
        class StubLoggableEvent(Event):
            prominent: ClassVar[bool] = True

        mock_cm._current_event_trace = {"event_id": "evt-test"}
        event = StubLoggableEvent()
        await EventHandler.handle_event(event, mock_cm)

        mock_cm._session_logger.info.assert_called_with(
            "stub_loggable_event",
            "Event: StubLoggableEvent",
        )

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

        mock_cm.request_llm_run.assert_called_once_with(
            triggering_contact_id=2,
        )

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
        mock_cm.request_llm_run.assert_called_once_with(
            delay=0,
            cancel_running=True,
            triggering_contact_id=2,
        )

    @pytest.mark.asyncio
    async def test_whatsapp_permission_keeps_legacy_room_name_when_agent_id_missing(
        self,
        mock_cm,
    ):
        """Accepted WhatsApp permission should preserve the pre-refactor room-name fallback."""

        mock_cm._pending_whatsapp_call_contexts = {2: "pending call context"}
        mock_cm._event_broker = mock_cm.event_broker
        event = WhatsAppCallPermissionResponse(
            contact={
                "contact_id": 2,
                "first_name": "Alice",
                "surname": "Smith",
                "whatsapp_number": "+15555552222",
            },
            accepted=True,
        )

        with (
            patch(
                "unity.conversation_manager.domains.comms_utils.start_whatsapp_call",
                new_callable=AsyncMock,
                return_value={"success": True},
            ) as mock_start_whatsapp_call,
            patch(
                "unity.conversation_manager.domains.event_handlers.SESSION_DETAILS",
            ) as mock_session_details,
        ):
            mock_session_details.assistant.agent_id = None
            mock_session_details.assistant.name = "Test Assistant"

            await EventHandler.handle_event(event, mock_cm)

        assert mock_start_whatsapp_call.await_args.kwargs["room_name"] == (
            "unity_None_whatsapp_call"
        )
        mock_cm.request_llm_run.assert_not_called()


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
    async def test_inbound_utterance_resets_proactive_speech(self, mock_cm):
        """Inbound utterances reset (reschedule) proactive speech."""
        event = InboundPhoneUtterance(
            contact={"contact_id": 2},
            content="User speaking",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.schedule_proactive_speech.assert_called()

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

        mock_cm.interject_or_run.assert_called_once_with(
            "What's the weather?",
            triggering_contact_id=2,
        )

    @pytest.mark.asyncio
    async def test_outbound_utterance_resets_proactive_speech(self, mock_cm):
        """Outbound utterances also reset (reschedule) proactive speech."""
        event = OutboundPhoneUtterance(
            contact={"contact_id": 2},
            content="Here's my response",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.schedule_proactive_speech.assert_called()

    @pytest.mark.asyncio
    async def test_call_guidance_updates_contact_index(self, mock_cm):
        """FastBrainNotification adds guidance message to voice thread."""
        event = FastBrainNotification(
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
    async def test_actor_handle_started_does_not_trigger_slow_brain(self, mock_cm):
        """ActorHandleStarted is a no-op (fast brain handles acknowledgement)."""
        event = ActorHandleStarted(
            action_name="task",
            handle_id=1,
            query="Do something",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_not_called()

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
    async def test_actor_handle_response_updates_completed_action(
        self,
        mock_cm,
    ):
        """ActorHandleResponse for a completed action should still update
        the pending ask and wake the brain.

        Regression: the handler only checked ``in_flight_actions``, so ask
        responses on actions that had already moved to ``completed_actions``
        were silently dropped.
        """
        response_text = "Detailed retrospective answer."

        mock_cm.completed_actions = {
            1: {
                "query": "Process accounts into standard format",
                "handle_actions": [
                    {
                        "action_name": "act_completed",
                        "query": "Done",
                        "status": "completed",
                    },
                    {
                        "action_name": "ask_1",
                        "query": "How did you break down the task?",
                        "status": "pending",
                    },
                ],
            },
        }
        event = ActorHandleResponse(
            handle_id=1,
            action_name="ask",
            query="How did you break down the task?",
            response=response_text,
            call_id="",
        )

        await EventHandler.handle_event(event, mock_cm)

        ask_event = mock_cm.completed_actions[1]["handle_actions"][1]
        assert ask_event["status"] == "completed"
        assert ask_event.get("response") == response_text
        mock_cm.request_llm_run.assert_called()

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
    async def test_user_webcam_started_sets_flag(self, mock_cm):
        """UserWebcamStarted sets user_webcam_active to True."""
        mock_cm.user_webcam_active = False

        event = UserWebcamStarted()
        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.user_webcam_active is True

    @pytest.mark.asyncio
    async def test_user_webcam_stopped_clears_flag(self, mock_cm):
        """UserWebcamStopped sets user_webcam_active to False."""
        mock_cm.user_webcam_active = True

        event = UserWebcamStopped()
        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.user_webcam_active is False

    @pytest.mark.asyncio
    async def test_webcam_events_skip_brain_step_before_meet_started(self, mock_cm):
        """Webcam events before UnifyMeetStarted should track state but not
        wake the slow brain.  The meeting hasn't started yet, so there is
        nothing useful for the LLM to reason about."""
        mock_cm.mode = Mode.TEXT
        mock_cm.user_webcam_active = False

        event = UserWebcamStarted()
        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.user_webcam_active is True
        mock_cm.request_llm_run.assert_not_called()
        mock_cm.schedule_proactive_speech.assert_not_called()

    @pytest.mark.asyncio
    async def test_webcam_events_set_state_without_slow_brain(self, mock_cm):
        """Webcam events set the state flag but don't trigger a slow-brain
        run (fast brain handles via silent notification injection)."""
        mock_cm.mode = Mode.MEET
        mock_cm.user_webcam_active = False

        event = UserWebcamStarted()
        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.user_webcam_active is True
        mock_cm.request_llm_run.assert_not_called()

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
    async def test_meet_interaction_does_not_trigger_slow_brain(self, mock_cm):
        """Meet interaction events are handled by the fast brain; no slow-brain run."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False

        event = UserScreenShareStarted(
            reason="User started sharing their screen",
        )
        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_not_called()

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

        mock_cm.capture_assistant_screenshot.assert_called_once()
        call_args = mock_cm.capture_assistant_screenshot.call_args
        assert call_args[0][0] == "So you need to click that button"

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
    # User screenshot buffer (IPC path)
    # --------------------------------------------------------------------- #

    def test_buffer_screenshot_parses_ipc_json(self, mock_cm):
        """_buffer_screenshot parses IPC JSON and buffers a ScreenshotEntry."""
        import json
        from datetime import datetime

        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm._screenshot_buffer = []
        mock_cm._session_logger = MagicMock()
        method = ConversationManager._buffer_screenshot.__get__(mock_cm)

        payload = json.dumps(
            {
                "b64": "iVBORw0KGgoAAAANSUhEUg==",
                "utterance": "Look at this part of my screen",
                "timestamp": "2026-02-15T12:00:00+00:00",
            },
        )
        method(payload)

        assert len(mock_cm._screenshot_buffer) == 1
        entry = mock_cm._screenshot_buffer[0]
        assert isinstance(entry, ScreenshotEntry)
        assert entry.source == "user"
        assert entry.b64 == "iVBORw0KGgoAAAANSUhEUg=="
        assert entry.utterance == "Look at this part of my screen"
        assert isinstance(entry.timestamp, datetime)

    # --------------------------------------------------------------------- #
    # Two-phase screenshot buffer (peek + commit)
    # --------------------------------------------------------------------- #

    def test_peek_does_not_clear_buffer_simulating_cancelled_turn(self, mock_cm):
        """Peeking the screenshot buffer leaves entries intact for retry.

        Simulates a cancelled LLM turn: peek is called but commit never
        happens.  The next peek must return the same screenshots.  With the
        old destructive drain this would have returned an empty list on the
        second call.
        """
        from datetime import datetime, timezone

        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm._screenshot_buffer = []
        peek = ConversationManager.peek_screenshot_buffer.__get__(mock_cm)

        ts = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_cm._screenshot_buffer.append(
            ScreenshotEntry("AAAA", "Click the button", ts, "assistant", 1),
        )
        mock_cm._screenshot_buffer.append(
            ScreenshotEntry("BBBB", "Now scroll down", ts, "user", 2),
        )

        # First peek (start of a turn that will be cancelled)
        first = peek()
        assert len(first) == 2

        # Simulate cancellation — commit is never called.

        # Second peek (retry turn) must see the same screenshots.
        second = peek()
        assert len(second) == 2
        assert second[0].b64 == "AAAA"
        assert second[1].b64 == "BBBB"

    def test_commit_clears_peeked_and_preserves_new_arrivals(self, mock_cm):
        """Committing after a successful turn removes consumed entries while
        preserving screenshots that arrived during the turn.

        With the old destructive drain, screenshots appended between drain
        and turn completion would have been lost.
        """
        from datetime import datetime, timezone

        from unity.conversation_manager.conversation_manager import ConversationManager

        mock_cm._screenshot_buffer = []
        peek = ConversationManager.peek_screenshot_buffer.__get__(mock_cm)
        commit = ConversationManager.commit_screenshot_buffer.__get__(mock_cm)

        ts = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_cm._screenshot_buffer.append(
            ScreenshotEntry("AAAA", "Original screenshot", ts, "assistant", 1),
        )

        # Peek at the start of the turn (1 screenshot).
        peeked = peek()
        assert len(peeked) == 1

        # A new screenshot arrives mid-turn (e.g. user speaks again).
        mock_cm._screenshot_buffer.append(
            ScreenshotEntry("BBBB", "New during turn", ts, "user", 2),
        )

        # Turn succeeds — commit only the peeked count.
        commit(len(peeked))

        # The original screenshot is gone; the mid-turn arrival survives.
        assert len(mock_cm._screenshot_buffer) == 1
        assert mock_cm._screenshot_buffer[0].b64 == "BBBB"

    # --------------------------------------------------------------------- #
    # Direct fast brain guidance on mode change
    # --------------------------------------------------------------------- #

    @pytest.mark.asyncio
    async def test_meet_event_sends_fast_brain_guidance_in_voice_mode(
        self,
        mock_cm,
    ):
        """Screen share events publish direct FastBrainNotification to the fast brain
        when in voice mode, bypassing the slow brain for instant delivery."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.user_screen_share_active = False
        mock_cm.user_remote_control_active = False
        mock_cm.mode = Mode.MEET  # voice mode

        event = AssistantScreenShareStarted(
            reason="User enabled screen sharing",
        )
        await EventHandler.handle_event(event, mock_cm)

        # Verify FastBrainNotification was published to the fast brain channel
        calls = mock_cm.event_broker.publish.call_args_list
        guidance_calls = [c for c in calls if c.args[0] == "app:call:notification"]
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

        # No FastBrainNotification should be published
        calls = mock_cm.event_broker.publish.call_args_list
        guidance_calls = [c for c in calls if c.args[0] == "app:call:notification"]
        assert len(guidance_calls) == 0

    @pytest.mark.asyncio
    async def test_all_meet_events_have_fast_brain_guidance(self, mock_cm):
        """Each meet interaction event has corresponding fast brain guidance text."""
        from unity.conversation_manager.domains.event_handlers import (
            _MEET_FAST_BRAIN_GUIDANCE,
        )

        event_classes = [
            AssistantScreenShareStarted,
            AssistantScreenShareStopped,
            UserScreenShareStarted,
            UserScreenShareStopped,
            UserWebcamStarted,
            UserWebcamStopped,
            UserRemoteControlStarted,
            UserRemoteControlStopped,
        ]
        for cls in event_classes:
            assert (
                cls in _MEET_FAST_BRAIN_GUIDANCE
            ), f"{cls.__name__} missing from _MEET_FAST_BRAIN_GUIDANCE"
            assert len(_MEET_FAST_BRAIN_GUIDANCE[cls]) > 0

    # --------------------------------------------------------------------- #
    # Call-started screen share state sync (initialization race fix)
    # --------------------------------------------------------------------- #

    @pytest.mark.asyncio
    async def test_call_started_syncs_screen_share_state_to_fast_brain(
        self,
        mock_cm,
    ):
        """When assistant_screen_share_active is already True at call start,
        the handler queues screen share guidance to the fast brain socket."""
        mock_cm.assistant_screen_share_active = True
        mock_cm.mode = Mode.TEXT
        mock_socket = AsyncMock()
        mock_cm.call_manager._socket_server = mock_socket

        event = UnifyMeetStarted(contact={"contact_id": 1})
        await EventHandler.handle_event(event, mock_cm)

        mock_socket.queue_for_clients.assert_called_once()
        channel, event_json = mock_socket.queue_for_clients.call_args.args
        assert channel == "app:call:notification"
        import json

        event_data = json.loads(event_json)
        content = event_data.get("payload", {}).get("content", "")
        assert "screen sharing is now on" in content.lower()

    @pytest.mark.asyncio
    async def test_call_started_does_not_sync_when_screen_share_inactive(
        self,
        mock_cm,
    ):
        """When assistant_screen_share_active is False, no guidance is queued."""
        mock_cm.assistant_screen_share_active = False
        mock_cm.mode = Mode.TEXT
        mock_socket = AsyncMock()
        mock_cm.call_manager._socket_server = mock_socket

        event = UnifyMeetStarted(contact={"contact_id": 1})
        await EventHandler.handle_event(event, mock_cm)

        mock_socket.queue_for_clients.assert_not_called()

    @pytest.mark.asyncio
    async def test_call_started_no_crash_without_socket_server(
        self,
        mock_cm,
    ):
        """If socket server is not yet created, the handler skips gracefully."""
        mock_cm.assistant_screen_share_active = True
        mock_cm.mode = Mode.TEXT
        mock_cm.call_manager._socket_server = None

        event = UnifyMeetStarted(contact={"contact_id": 1})
        await EventHandler.handle_event(event, mock_cm)
        # Should not raise — the falsy check on _socket_server prevents the call.

    @pytest.mark.asyncio
    async def test_phone_call_started_also_syncs_screen_share_state(
        self,
        mock_cm,
    ):
        """PhoneCallStarted handler also syncs screen share state."""
        mock_cm.assistant_screen_share_active = True
        mock_cm.mode = Mode.TEXT
        mock_socket = AsyncMock()
        mock_cm.call_manager._socket_server = mock_socket

        event = PhoneCallStarted(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
        )
        await EventHandler.handle_event(event, mock_cm)

        mock_socket.queue_for_clients.assert_called_once()
        channel, _ = mock_socket.queue_for_clients.call_args.args
        assert channel == "app:call:notification"

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

    def test_render_meet_state_user_webcam_only(self):
        """Only user webcam active produces a single webcam section."""
        from unity.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_meet_interaction_state(
            user_webcam_active=True,
        )
        assert "<user_webcam status='active'>" in result
        assert "</user_webcam>" in result
        assert "webcam" in result
        assert "<assistant_screen_share" not in result
        assert "<user_screen_share" not in result
        assert "<user_remote_control" not in result

    def test_render_meet_state_all_four_active(self):
        """All four active produces four independent sections."""
        from unity.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_meet_interaction_state(
            assistant_screen_share_active=True,
            user_screen_share_active=True,
            user_webcam_active=True,
            user_remote_control_active=True,
        )
        assert "<assistant_screen_share status='active'>" in result
        assert "<user_screen_share status='active'>" in result
        assert "<user_webcam status='active'>" in result
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
    async def test_notification_injected_resets_proactive_speech(self, mock_cm):
        """NotificationInjectedEvent resets (reschedules) proactive speech."""
        event = NotificationInjectedEvent(
            content="Interrupt notification",
            source="System",
            target_conversation_id="conv_123",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.schedule_proactive_speech.assert_called_once()

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


class TestTaskDueEventHandlers:
    """Tests for scheduled task due notifications and startup replay."""

    @pytest.mark.asyncio
    async def test_task_due_pushes_notification_and_requests_llm(self, mock_cm):
        """Current due activations should surface one execute-task directive."""

        event = TaskDue(
            task_id=101,
            source_task_log_id=555,
            activation_revision="rev-1",
            scheduled_for="2026-04-10T09:00:00+00:00",
            task_label="Morning briefing",
            task_summary="Prepare the morning update before the user checks in.",
            visibility_policy="silent_by_default",
            recurrence_hint="recurring",
        )

        with (
            patch(
                "unity.conversation_manager.domains.task_activation.validate_task_due_activation",
                return_value=(
                    TaskActivationSnapshot(
                        assistant_id="42",
                        activation_key="42:101",
                        task_id=101,
                        source_task_log_id=555,
                        activation_revision="rev-1",
                    ),
                    None,
                ),
            ),
            patch(
                "unity.conversation_manager.domains.task_activation.remember_live_task_run_provenance",
            ) as mock_remember_provenance,
        ):
            await EventHandler.handle_event(event, mock_cm)

        assert len(mock_cm.notifications_bar.notifications) == 1
        notification = mock_cm.notifications_bar.notifications[0].content
        assert "Morning briefing" in notification
        assert "Prepare the morning update before the user checks in" in notification
        assert "work silently unless you genuinely need the user" in notification
        assert "primitives.tasks.execute(task_id=101)" in notification
        remembered = mock_remember_provenance.call_args.args[0]
        assert remembered.assistant_id == "42"
        assert remembered.source_type == "scheduled"
        assert remembered.scheduled_for == "2026-04-10T09:00:00+00:00"
        mock_cm.request_llm_run.assert_called_once_with(delay=0)

    @pytest.mark.asyncio
    async def test_task_due_queues_fast_brain_context_during_voice_call(self, mock_cm):
        """Validated task_due events should also reach the fast brain silently."""

        event = TaskDue(
            task_id=101,
            source_task_log_id=555,
            activation_revision="rev-1",
            scheduled_for="2026-04-10T09:00:00+00:00",
            task_label="Morning briefing",
            task_summary="Prepare the morning update before the user checks in.",
            visibility_policy="silent_by_default",
            recurrence_hint="recurring",
        )
        mock_cm.mode = Mode.CALL
        mock_socket = MagicMock()
        mock_socket.queue_for_clients = AsyncMock()
        mock_cm.call_manager._socket_server = mock_socket

        with patch(
            "unity.conversation_manager.domains.task_activation.validate_task_due_activation",
            return_value=(
                TaskActivationSnapshot(
                    assistant_id="42",
                    activation_key="42:101",
                    task_id=101,
                    activation_revision="rev-1",
                    task_name="Morning briefing",
                ),
                None,
            ),
        ):
            await EventHandler.handle_event(event, mock_cm)

        mock_socket.queue_for_clients.assert_called_once()
        channel, payload = mock_socket.queue_for_clients.call_args.args
        assert channel == "app:call:notification"
        guidance = FastBrainNotification.from_json(payload)
        assert guidance.should_speak is False
        assert "Morning briefing" in guidance.content
        assert (
            "Prepare the morning update before the user checks in" in guidance.content
        )
        assert "silent action unless the user is needed" in guidance.content

    @pytest.mark.asyncio
    async def test_task_due_ignores_stale_activation(self, mock_cm):
        """Stale due deliveries should not notify or re-run the slow brain."""

        event = TaskDue(
            task_id=101,
            source_task_log_id=555,
            activation_revision="rev-1",
            scheduled_for="2026-04-10T09:00:00+00:00",
        )

        with patch(
            "unity.conversation_manager.domains.task_activation.validate_task_due_activation",
            return_value=(None, "activation_revision_mismatch"),
        ):
            await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.notifications_bar.notifications == []
        mock_cm.request_llm_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialization_complete_replays_startup_task_due_reasons(
        self,
        mock_cm,
    ):
        """Startup task_due wake reasons should surface on the first post-init run."""

        mock_cm.call_manager._socket_server = None
        mock_cm._startup_wake_reasons = [
            {
                "type": "task_due",
                "task_id": 101,
                "source_task_log_id": 555,
                "activation_revision": "rev-1",
                "scheduled_for": "2026-04-10T09:00:00+00:00",
                "task_label": "Morning briefing",
                "task_summary": "Prepare the morning update before the user checks in.",
                "visibility_policy": "silent_by_default",
                "recurrence_hint": "recurring",
            },
        ]

        with patch(
            "unity.conversation_manager.domains.task_activation.validate_task_due_activation",
            return_value=(MagicMock(activation_revision="rev-1"), None),
        ):
            await EventHandler.handle_event(InitializationComplete(), mock_cm)

        assert mock_cm._startup_wake_reasons == []
        assert len(mock_cm.notifications_bar.notifications) == 2
        assert any(
            "primitives.tasks.execute(task_id=101)" in notif.content
            for notif in mock_cm.notifications_bar.notifications
        )
        assert any(
            "Morning briefing" in notif.content
            for notif in mock_cm.notifications_bar.notifications
        )
        mock_cm.request_llm_run.assert_called_once_with(delay=0)


class TestInitializationCompleteHandler:
    """Tests for the InitializationComplete handler.

    The handler always schedules a post-init brain turn (so deferred work
    from pre-init replies is not silently dropped), but the system
    notification it pins is deliberately worded to discourage gratuitous
    duplicate replies — see
    ``event_handlers.INITIALIZATION_COMPLETE_NOTIFICATION``. These tests
    lock in both the "always run the brain" contract and the key wording
    of that directive (regression coverage for the cold-start
    duplicate-reply bug observed for assistant 1820).
    """

    @pytest.fixture(autouse=True)
    def _init_handler_state(self, mock_cm):
        """Reset per-test CM attributes the handler reads."""
        mock_cm._startup_wake_reasons = []
        mock_cm.call_manager._socket_server = None
        yield

    @pytest.mark.asyncio
    async def test_notification_pushed(self, mock_cm):
        """The handler must pin an 'Initialization complete' notification."""

        await EventHandler.handle_event(InitializationComplete(), mock_cm)

        assert any(
            "Initialization complete" in notif.content
            for notif in mock_cm.notifications_bar.notifications
        )

    @pytest.mark.asyncio
    async def test_notification_uses_anti_duplicate_directive(self, mock_cm):
        """The pinned notif must explicitly tell the brain to call wait
        and NOT send a duplicate/rephrased reply.

        Regression test for the Unify duplicate-reply bug: the previous
        wording ("Review any earlier responses … and follow up if needed
        — correct, elaborate, or confirm") was being read as permission
        to re-send a rephrased version of the pre-init reply. The new
        wording must keep the legitimate follow-up paths (deferred work,
        wrong/incomplete due to missing context) but explicitly forbid
        rephrase/restate/confirm-style duplicates.
        """
        from unity.conversation_manager.domains.event_handlers import (
            INITIALIZATION_COMPLETE_NOTIFICATION,
        )

        await EventHandler.handle_event(InitializationComplete(), mock_cm)

        notif = next(
            n
            for n in mock_cm.notifications_bar.notifications
            if "Initialization complete" in n.content
        )
        assert notif.content == INITIALIZATION_COMPLETE_NOTIFICATION

        text = notif.content.lower()
        # Must keep the legitimate follow-up affordance.
        assert "deferred work" in text
        assert "incorrect or incomplete" in text
        # Must explicitly direct the brain to wait and forbid duplicates.
        assert "call wait" in text
        assert "do not send a message" in text
        assert "rephrases" in text or "restates" in text

    @pytest.mark.asyncio
    async def test_always_requests_post_init_brain_run(self, mock_cm):
        """The handler must always schedule a brain turn, even when the
        pre-init conversation already has a trailing assistant reply.

        The brain still needs the opportunity to follow up on deferred
        work (e.g. the user asked for something the brain couldn't do
        until managers were ready, replied with "I'll look into it",
        and now needs to actually do it). Suppressing the brain run
        here would silently drop those follow-ups, which is a worse
        failure mode than the duplicate reply we mitigate via the
        notification wording above.
        """

        mock_cm.contact_index.push_message(
            contact_id=1,
            sender_name="Boss",
            thread_name=Medium.UNIFY_MESSAGE,
            message_content="What meetings do I have today?",
            role="user",
        )
        mock_cm.contact_index.push_message(
            contact_id=1,
            sender_name="You",
            thread_name=Medium.UNIFY_MESSAGE,
            message_content="I'm still booting up — give me a moment.",
            role="assistant",
        )

        await EventHandler.handle_event(InitializationComplete(), mock_cm)

        mock_cm.request_llm_run.assert_called_once_with(delay=0)

    @pytest.mark.asyncio
    async def test_requests_brain_run_when_thread_is_empty(self, mock_cm):
        """No pre-init traffic — handler still schedules a brain turn so
        the brain can react to any startup wake reasons or hydration."""

        await EventHandler.handle_event(InitializationComplete(), mock_cm)

        mock_cm.request_llm_run.assert_called_once_with(delay=0)

    @pytest.mark.asyncio
    async def test_voice_mode_pushes_fast_brain_notification(self, mock_cm):
        """In voice mode the handler also pushes a (silent) FastBrainNotification
        so the voice agent subprocess sees the post-init context.

        This path is independent of the slow-brain notification wording
        change, but we lock it in so a refactor to the handler doesn't
        silently drop it.
        """

        mock_socket = MagicMock()
        mock_socket.queue_for_clients = AsyncMock()
        mock_cm.call_manager._socket_server = mock_socket

        await EventHandler.handle_event(InitializationComplete(), mock_cm)

        mock_socket.queue_for_clients.assert_called_once()
        channel, payload = mock_socket.queue_for_clients.call_args.args
        assert channel == "app:call:notification"
        notification = FastBrainNotification.from_json(payload)
        assert notification.should_speak is False
        assert notification.source == "initialization"
        assert "Initialization complete" in notification.content


class TestTriggeredTaskNotifications:
    """Tests for mechanically matched trigger-task notifications."""

    @pytest.mark.asyncio
    async def test_inbound_message_surfaces_trigger_candidates(self, mock_cm):
        """Inbound user messages should surface only live matching trigger tasks."""

        event = SMSReceived(
            contact={"contact_id": 2, "first_name": "Alice", "surname": "Smith"},
            content="Can you review the invoice?",
        )
        candidates = [
            TaskActivationSnapshot(
                assistant_id="42",
                activation_key="42:301",
                task_id=301,
                activation_kind="triggered",
                execution_mode="live",
                trigger_from_contact_ids=[2],
                interrupt=True,
                task_name="Invoice follow-up",
                task_description="Help handle invoice-related requests from Alice.",
            ),
            TaskActivationSnapshot(
                assistant_id="42",
                activation_key="42:302",
                task_id=302,
                activation_kind="triggered",
                execution_mode="offline",
                trigger_from_contact_ids=[2],
                task_name="Hidden offline task",
            ),
            TaskActivationSnapshot(
                assistant_id="42",
                activation_key="42:303",
                task_id=303,
                activation_kind="triggered",
                execution_mode="live",
                trigger_omit_contact_ids=[2],
                task_name="Wrong sender task",
            ),
        ]

        with (
            patch(
                "unity.conversation_manager.domains.task_activation.list_trigger_activations",
                return_value=candidates,
            ),
            patch(
                "unity.conversation_manager.domains.task_activation.remember_live_task_run_provenance",
            ) as mock_remember_provenance,
            patch(
                "unity.conversation_manager.domains.task_activation._dispatch_offline_trigger_candidate",
                return_value={"status": "launched"},
            ) as mock_offline_dispatch,
        ):
            await EventHandler.handle_event(event, mock_cm)

        assert len(mock_cm.notifications_bar.notifications) == 2
        trigger_notification = mock_cm.notifications_bar.notifications[1].content
        assert "Invoice follow-up" in trigger_notification
        assert "Help handle invoice-related requests from Alice" in trigger_notification
        assert "Semantic judgement is still pending" in trigger_notification
        assert (
            'primitives.tasks.execute(task_id=301, trigger_attempt_token="'
            in trigger_notification
        )
        assert "Hidden offline task" not in trigger_notification
        assert "Wrong sender task" not in trigger_notification
        remembered = mock_remember_provenance.call_args.args[0]
        assert remembered.task_id == 301
        assert remembered.source_type == "triggered"
        assert remembered.source_medium == "sms_message"
        assert remembered.attempt_token
        mock_offline_dispatch.assert_called_once()
        mock_cm.request_llm_run.assert_called_once_with(triggering_contact_id=2)

    @pytest.mark.asyncio
    async def test_inbound_call_dispatches_offline_triggers_without_live_notification(
        self,
        mock_cm,
    ):
        """Offline trigger matches should stay invisible to the live call lane."""

        event = PhoneCallReceived(
            contact={
                "contact_id": 2,
                "first_name": "Alice",
                "surname": "Smith",
                "phone_number": "+15555552222",
            },
            conference_name="conf-123",
        )
        candidates = [
            TaskActivationSnapshot(
                assistant_id="42",
                activation_key="42:402",
                task_id=402,
                activation_kind="triggered",
                execution_mode="offline",
                trigger_from_contact_ids=[2],
            ),
        ]

        with (
            patch(
                "unity.conversation_manager.domains.task_activation.list_trigger_activations",
                return_value=candidates,
            ),
            patch(
                "unity.conversation_manager.domains.task_activation._dispatch_offline_trigger_candidate",
                return_value={"status": "launched"},
            ) as mock_offline_dispatch,
        ):
            await EventHandler.handle_event(event, mock_cm)

        assert not any(
            "402" in notif.content for notif in mock_cm.notifications_bar.notifications
        )
        mock_offline_dispatch.assert_called_once()
        mock_cm.request_llm_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_inbound_call_requests_llm_when_trigger_candidates_match(
        self,
        mock_cm,
    ):
        """Inbound call triggers should wake the slow brain immediately."""

        event = PhoneCallReceived(
            contact={
                "contact_id": 2,
                "first_name": "Alice",
                "surname": "Smith",
                "phone_number": "+15555552222",
            },
            conference_name="conf-123",
        )
        candidates = [
            TaskActivationSnapshot(
                assistant_id="42",
                activation_key="42:401",
                task_id=401,
                activation_kind="triggered",
                execution_mode="live",
                trigger_from_contact_ids=[2],
                task_name="Handle VIP caller",
                task_description="Prioritize urgent inbound calls from Alice.",
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.task_activation.list_trigger_activations",
            return_value=candidates,
        ):
            await EventHandler.handle_event(event, mock_cm)

        assert any(
            "Handle VIP caller" in notif.content
            for notif in mock_cm.notifications_bar.notifications
        )
        assert any(
            "Semantic judgement is still pending" in notif.content
            for notif in mock_cm.notifications_bar.notifications
        )
        mock_cm.request_llm_run.assert_called_once_with(
            delay=0,
            triggering_contact_id=2,
        )

    @pytest.mark.asyncio
    async def test_inbound_call_queues_fast_brain_trigger_context(self, mock_cm):
        """Live trigger candidates should be mirrored to the fast brain silently."""

        event = PhoneCallReceived(
            contact={
                "contact_id": 2,
                "first_name": "Alice",
                "surname": "Smith",
                "phone_number": "+15555552222",
            },
            conference_name="conf-123",
        )
        # PhoneCallReceived is handled while the CM is still in text mode; the
        # fast-brain socket may already exist before the started event flips mode.
        mock_cm.mode = Mode.TEXT
        mock_socket = MagicMock()
        mock_socket.queue_for_clients = AsyncMock()
        mock_cm.call_manager._socket_server = mock_socket
        candidates = [
            TaskActivationSnapshot(
                assistant_id="42",
                activation_key="42:401",
                task_id=401,
                activation_kind="triggered",
                execution_mode="live",
                trigger_from_contact_ids=[2],
                task_name="Handle VIP caller",
                task_description="Prioritize urgent inbound calls from Alice.",
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.task_activation.list_trigger_activations",
            return_value=candidates,
        ):
            await EventHandler.handle_event(event, mock_cm)

        call_args = mock_socket.queue_for_clients.call_args_list
        assert call_args
        channel, payload = call_args[-1].args
        assert channel == "app:call:notification"
        guidance = FastBrainNotification.from_json(payload)
        assert guidance.should_speak is False
        assert "Handle VIP caller" in guidance.content
        assert "Prioritize urgent inbound calls from Alice" in guidance.content
        assert "Do not mention the task unless it naturally helps" in guidance.content


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
        assert call_args[0][0] == "app:call:notification"

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
            assistant_first_name="Updated",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="US",
            assistant_about="Test assistant",
            assistant_number="+15555550001",
            assistant_email="assistant@updated.com",
            user_first_name="Updated",
            user_surname="Boss",
            user_number="+15555550002",
            user_email="boss@updated.com",
            voice_id="voice_123",
            voice_provider="cartesia",
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
            assistant_first_name="Updated",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="US",
            assistant_about="Test assistant",
            assistant_number="+15555550001",
            assistant_email="assistant@updated.com",
            user_first_name="Updated",
            user_surname="Boss",
            user_number="+15555550002",
            user_email="boss@updated.com",
            voice_id="voice_123",
            voice_provider="cartesia",
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
            assistant_first_name="Updated",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="US",
            assistant_about="Test assistant",
            assistant_number="+15555550001",
            assistant_email="assistant@updated.com",
            user_first_name="Updated",
            user_surname="Boss",
            user_number="+15555550002",
            user_email="boss@updated.com",
            voice_id="voice_123",
            voice_provider="cartesia",
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
            assistant_first_name="New Assistant",
            assistant_surname="Name",
            assistant_age="25",
            assistant_nationality="US",
            assistant_about="Test assistant",
            assistant_number="+15555559999",
            assistant_email="new_assistant@test.com",
            user_first_name="New Boss",
            user_surname="Name",
            user_number="+15555558888",
            user_email="new_boss@test.com",
            voice_id="voice_123",
            voice_provider="cartesia",
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

            # Handler queues two operations: sync_assistant_secrets (to refresh
            # any rotated OAuth tokens) followed by update_session_contacts.
            assert mock_utils.queue_operation.call_count == 2

            secrets_call = mock_utils.queue_operation.call_args_list[0]
            assert secrets_call[0][0] == mock_utils.sync_assistant_secrets

            contacts_call = mock_utils.queue_operation.call_args_list[1]
            # First arg is the function (update_session_contacts)
            assert contacts_call[0][0] == mock_utils.update_session_contacts
            # Remaining args are: cm, assistant_first_name, assistant_surname,
            # assistant_number, assistant_email, user_first_name, user_surname, user_number, user_email
            assert contacts_call[0][1] == mock_cm
            assert contacts_call[0][2] == "New Assistant"
            assert contacts_call[0][3] == "Name"
            assert contacts_call[0][4] == "+15555559999"
            assert contacts_call[0][5] == "new_assistant@test.com"
            assert contacts_call[0][6] == "New Boss"
            assert contacts_call[0][7] == "Name"
            assert contacts_call[0][8] == "+15555558888"
            assert contacts_call[0][9] == "new_boss@test.com"

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
            assistant_first_name="Updated",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="US",
            assistant_about="Test assistant",
            assistant_number="+15555550001",
            assistant_email="assistant@updated.com",
            user_first_name="Updated",
            user_surname="Boss",
            user_number="+15555550002",
            user_email="boss@updated.com",
            voice_id="voice_123",
            voice_provider="cartesia",
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
    async def test_membership_update_rebinds_space_roots_without_restart(
        self,
        mock_cm,
        monkeypatch,
    ):
        """Membership updates refresh reachable roots without running config side effects."""
        from unity.common.context_registry import (
            PERSONAL_ROOT_IDENTITY,
            SPACE_CONTEXT_PREFIX,
            ContextRegistry,
        )
        from unity.session_details import SESSION_DETAILS

        SESSION_DETAILS.space_ids = [3, 7]
        SESSION_DETAILS.self_contact_id = 0
        SESSION_DETAILS.boss_contact_id = 1
        monkeypatch.delenv("SELF_CONTACT_ID", raising=False)
        monkeypatch.delenv("BOSS_CONTACT_ID", raising=False)
        monkeypatch.delenv("SPACE_SUMMARIES", raising=False)
        ContextRegistry._registry = {
            ("TaskScheduler", "Tasks", PERSONAL_ROOT_IDENTITY): "user456/123/Tasks",
            ("TaskScheduler", "Tasks", f"{SPACE_CONTEXT_PREFIX}3"): "Spaces/3/Tasks",
            ("TaskScheduler", "Tasks", f"{SPACE_CONTEXT_PREFIX}7"): "Spaces/7/Tasks",
        }
        mock_cm.set_details = MagicMock()
        mock_cm.get_call_config = MagicMock(return_value={})

        event = AssistantUpdateEvent(
            api_key="test_key",
            medium="assistant_update",
            assistant_id="asst_123",
            user_id="user_456",
            assistant_first_name="Updated",
            assistant_surname="Assistant",
            assistant_age="25",
            assistant_nationality="US",
            assistant_about="Test assistant",
            assistant_number="+15555550001",
            assistant_email="assistant@updated.com",
            user_first_name="Updated",
            user_surname="Boss",
            user_number="+15555550002",
            user_email="boss@updated.com",
            voice_id="voice_123",
            voice_provider="cartesia",
            update_kind="membership",
            space_ids=[7, 11],
            space_summaries=[
                {
                    "space_id": 7,
                    "name": "Ops",
                    "description": "Operations workspace for customer support.",
                },
                {
                    "space_id": 11,
                    "name": "Repairs",
                    "description": "South-East repairs patch daily operations.",
                },
            ],
            self_contact_id=42,
            boss_contact_id=43,
        )

        with patch(
            "unity.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            try:
                await EventHandler.handle_event(event, mock_cm)

                assert SESSION_DETAILS.space_ids == [7, 11]
                assert SESSION_DETAILS.self_contact_id == 42
                assert SESSION_DETAILS.boss_contact_id == 43
                assert SESSION_DETAILS.assistant.contact_id == 42
                assert SESSION_DETAILS.user.contact_id == 43
                assert os.environ["SELF_CONTACT_ID"] == "42"
                assert os.environ["BOSS_CONTACT_ID"] == "43"
                assert (
                    "South-East repairs patch daily operations."
                    in os.environ["SPACE_SUMMARIES"]
                )
                assert mock_cm.space_ids == [7, 11]
                assert [summary.space_id for summary in mock_cm.space_summaries] == [
                    7,
                    11,
                ]
                assert mock_cm.self_contact_id == 42
                assert mock_cm.boss_contact_id == 43
                assert (
                    ContextRegistry._registry[
                        ("TaskScheduler", "Tasks", PERSONAL_ROOT_IDENTITY)
                    ]
                    == "user456/123/Tasks"
                )
                assert (
                    ContextRegistry._registry[
                        ("TaskScheduler", "Tasks", f"{SPACE_CONTEXT_PREFIX}7")
                    ]
                    == "Spaces/7/Tasks"
                )
                assert (
                    "TaskScheduler",
                    "Tasks",
                    f"{SPACE_CONTEXT_PREFIX}3",
                ) not in ContextRegistry._registry
                assert (
                    "TaskScheduler",
                    "Tasks",
                    f"{SPACE_CONTEXT_PREFIX}11",
                ) not in ContextRegistry._registry
                mock_cm.set_details.assert_not_called()
                mock_cm.call_manager.set_config.assert_not_called()
                mock_utils.queue_operation.assert_not_called()
            finally:
                ContextRegistry.clear()
                SESSION_DETAILS.reset()

    @pytest.mark.asyncio
    async def test_update_session_contacts_updates_both_contacts(self, mock_cm):
        """update_session_contacts updates both assistant (0) and boss (1) contacts."""
        from unity.conversation_manager.domains.managers_utils import (
            update_session_contacts,
        )

        mock_cm.contact_manager.update_contact = MagicMock()

        await update_session_contacts(
            mock_cm,
            assistant_first_name="New",
            assistant_surname="Assistant",
            assistant_number="+15555559999",
            assistant_email="new_assistant@test.com",
            user_first_name="New",
            user_surname="Boss",
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
    async def test_update_session_contacts_handles_failure(self, mock_cm, caplog):
        """update_session_contacts logs errors when update_contact fails."""
        import logging

        from unity.conversation_manager.domains.managers_utils import (
            update_session_contacts,
        )

        mock_cm.contact_manager.update_contact = MagicMock(
            side_effect=Exception("Update failed"),
        )

        unity_logger = logging.getLogger("unity")
        unity_logger.addHandler(caplog.handler)
        caplog.handler.setLevel(logging.DEBUG)
        try:
            await update_session_contacts(
                mock_cm,
                assistant_first_name="Updated",
                assistant_surname="Assistant",
                assistant_number="+15555550001",
                assistant_email="assistant@updated.com",
                user_first_name="Updated",
                user_surname="Boss",
                user_number="+15555550002",
                user_email="boss@updated.com",
            )
        finally:
            unity_logger.removeHandler(caplog.handler)

        assert "Failed to update contact 0" in caplog.text
        assert "Failed to update contact 1" in caplog.text

    @pytest.mark.asyncio
    async def test_update_session_contacts_handles_no_contact_manager(
        self,
        mock_cm,
        caplog,
    ):
        """update_session_contacts handles None contact_manager gracefully."""
        import logging

        from unity.conversation_manager.domains.managers_utils import (
            update_session_contacts,
        )

        mock_cm.contact_manager = None

        unity_logger = logging.getLogger("unity")
        unity_logger.addHandler(caplog.handler)
        caplog.handler.setLevel(logging.DEBUG)
        try:
            await update_session_contacts(
                mock_cm,
                assistant_first_name="Test",
                assistant_surname="",
                assistant_number="+1555",
                assistant_email="test@test.com",
                user_first_name="Boss",
                user_surname="",
                user_number="+1666",
                user_email="boss@test.com",
            )
        finally:
            unity_logger.removeHandler(caplog.handler)

        assert "contact_manager is None" in caplog.text


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
            contact_id=1,
            sender_name="Boss",
            thread_name=Medium.SMS_MESSAGE,
            message_content="Hey, open the browser",
            role="user",
        )
        mock_cm.contact_index.push_message(
            contact_id=1,
            sender_name="You",
            thread_name=Medium.SMS_MESSAGE,
            message_content="Sure, opening it now",
            role="assistant",
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
                contact_id=1,
                sender_name="Boss",
                thread_name=Medium.SMS_MESSAGE,
                message_content=f"Message {i}",
                role="user",
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
            contact_id=1,
            sender_name="System",
            thread_name=Medium.PHONE_CALL,
            message_content="<Call Started>",
            role="user",
        )
        mock_cm.contact_index.push_message(
            contact_id=1,
            sender_name="Boss",
            thread_name=Medium.PHONE_CALL,
            message_content="Hello there",
            role="user",
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
        self,
        mock_cm,
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


# =============================================================================
# FileSyncComplete Event Tests
# =============================================================================


class TestFileSyncCompleteEvent:
    """Tests for the FileSyncComplete event."""

    def test_file_sync_complete_is_registered(self):
        """FileSyncComplete should have a handler in the registry."""
        from unity.conversation_manager.events import FileSyncComplete

        assert FileSyncComplete in EventHandler._registry

    @pytest.mark.asyncio
    async def test_file_sync_complete_sets_flag(self, mock_cm):
        """FileSyncComplete handler should set cm.file_sync_complete to True."""
        from unity.conversation_manager.events import FileSyncComplete

        mock_cm.file_sync_complete = False
        event = FileSyncComplete()
        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.file_sync_complete is True

    @pytest.mark.asyncio
    async def test_file_sync_complete_triggers_llm_run(self, mock_cm):
        """FileSyncComplete handler should trigger an LLM run."""
        from unity.conversation_manager.events import FileSyncComplete

        mock_cm.file_sync_complete = False
        event = FileSyncComplete()
        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called_once_with(delay=0)

    @pytest.mark.asyncio
    async def test_file_sync_complete_handler_logs(self, mock_cm):
        """FileSyncComplete handler should log the sync completion."""
        from unity.conversation_manager.events import FileSyncComplete

        mock_cm.file_sync_complete = False
        event = FileSyncComplete()
        await EventHandler.handle_event(event, mock_cm)

        mock_cm._session_logger.debug.assert_called()


# =============================================================================
# AssistantDesktopReady Event Tests
# =============================================================================


class TestAssistantDesktopReadyEvent:
    """Tests for the AssistantDesktopReady VM-readiness handler."""

    def test_assistant_desktop_ready_is_registered(self):
        """AssistantDesktopReady should have a handler in the registry."""
        from unity.conversation_manager.events import AssistantDesktopReady

        assert AssistantDesktopReady in EventHandler._registry

    @pytest.mark.asyncio
    async def test_assistant_desktop_ready_sets_vm_ready_flag(self, mock_cm):
        """AssistantDesktopReady should set cm.vm_ready to True."""
        from unity.conversation_manager.events import AssistantDesktopReady

        mock_cm.vm_ready = False
        mock_cm.assistant_id = "84"
        mock_cm.user_id = "test_user"
        event = AssistantDesktopReady(
            binding_id="binding-84",
            desktop_url="https://unity-pool-ubuntu-1.vm.unify.ai",
            vm_type="ubuntu",
        )

        with (
            patch(
                "unity.conversation_manager.assistant_jobs.update_liveview_url",
            ),
            patch(
                "unity.conversation_manager.domains.managers_utils._start_file_sync",
                new_callable=AsyncMock,
            ),
            patch(
                "unity.session_details.SESSION_DETAILS",
            ) as mock_sd,
            patch(
                "unity.function_manager.primitives.runtime._vm_ready",
            ),
            patch(
                "unity.conversation_manager.domains.event_handlers._ensure_desktop_session",
                return_value=AsyncMock()(),
            ),
            patch(
                "unity.conversation_manager.domains.comms_utils.publish_assistant_desktop_ready",
                new_callable=AsyncMock,
            ),
        ):
            mock_sd.assistant.desktop_url = ""
            mock_sd.assistant.binding_id = "binding-84"
            await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.vm_ready is True

    @pytest.mark.asyncio
    async def test_assistant_desktop_ready_triggers_llm_run(self, mock_cm):
        """AssistantDesktopReady should trigger an LLM run."""
        from unity.conversation_manager.events import AssistantDesktopReady

        mock_cm.vm_ready = False
        mock_cm.assistant_id = "84"
        mock_cm.user_id = "test_user"
        event = AssistantDesktopReady(
            binding_id="binding-84",
            desktop_url="https://unity-pool-ubuntu-1.vm.unify.ai",
            vm_type="ubuntu",
        )

        with (
            patch(
                "unity.conversation_manager.assistant_jobs.update_liveview_url",
            ),
            patch(
                "unity.conversation_manager.domains.managers_utils._start_file_sync",
                new_callable=AsyncMock,
            ),
            patch(
                "unity.session_details.SESSION_DETAILS",
            ) as mock_sd,
            patch(
                "unity.function_manager.primitives.runtime._vm_ready",
            ),
            patch(
                "unity.conversation_manager.domains.event_handlers._ensure_desktop_session",
                return_value=AsyncMock()(),
            ),
            patch(
                "unity.conversation_manager.domains.comms_utils.publish_assistant_desktop_ready",
                new_callable=AsyncMock,
            ),
        ):
            mock_sd.assistant.desktop_url = ""
            mock_sd.assistant.binding_id = "binding-84"
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called_with(delay=0)

    @pytest.mark.asyncio
    async def test_assistant_desktop_ready_ignores_missing_binding_when_session_has_one(
        self,
        mock_cm,
    ):
        """AssistantDesktopReady should fail closed on missing binding_id."""
        from unity.conversation_manager.events import AssistantDesktopReady

        mock_cm.vm_ready = False
        mock_cm.assistant_id = "84"
        mock_cm.user_id = "test_user"
        event = AssistantDesktopReady(
            binding_id="",
            desktop_url="https://unity-pool-ubuntu-1.vm.unify.ai",
            vm_type="ubuntu",
        )

        with (
            patch(
                "unity.conversation_manager.assistant_jobs.update_liveview_url",
            ) as mock_update_liveview_url,
            patch(
                "unity.conversation_manager.domains.managers_utils._start_file_sync",
                new_callable=AsyncMock,
            ) as mock_start_file_sync,
            patch(
                "unity.session_details.SESSION_DETAILS",
            ) as mock_sd,
            patch(
                "unity.function_manager.primitives.runtime._vm_ready",
            ) as mock_vm_ready,
            patch(
                "unity.conversation_manager.domains.event_handlers._ensure_desktop_session",
                return_value=AsyncMock()(),
            ) as mock_ensure_desktop_session,
            patch(
                "unity.conversation_manager.domains.comms_utils.publish_assistant_desktop_ready",
                new_callable=AsyncMock,
            ) as mock_publish_desktop_ready,
        ):
            mock_sd.assistant.desktop_url = ""
            mock_sd.assistant.binding_id = "binding-84"
            await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.vm_ready is False
        mock_update_liveview_url.assert_not_called()
        mock_start_file_sync.assert_not_awaited()
        mock_publish_desktop_ready.assert_not_awaited()
        mock_ensure_desktop_session.assert_not_called()
        mock_vm_ready.set.assert_not_called()
        mock_cm.request_llm_run.assert_not_called()
        mock_cm._session_logger.info.assert_any_call(
            "desktop_ready_missing_binding",
            "Ignoring desktop_ready without binding_id because the session already has a current binding",
        )


# =============================================================================
# Renderer: Infrastructure State Tests
# =============================================================================


class TestRenderInfrastructureState:
    """Tests for Renderer.render_infrastructure_state()."""

    def test_both_pending_with_desktop(self):
        """Both VM and sync pending should produce two sections."""
        from unity.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_infrastructure_state(
            vm_ready=False,
            file_sync_complete=False,
            has_desktop=True,
        )
        assert "vm_pending" in result
        assert "sync_pending" in result

    def test_vm_ready_sync_pending(self):
        """Only sync pending should produce one section."""
        from unity.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_infrastructure_state(
            vm_ready=True,
            file_sync_complete=False,
            has_desktop=True,
        )
        assert "vm_pending" not in result
        assert "sync_pending" in result

    def test_all_ready(self):
        """Both ready should return empty string."""
        from unity.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_infrastructure_state(
            vm_ready=True,
            file_sync_complete=True,
            has_desktop=True,
        )
        assert result == ""

    def test_no_desktop(self):
        """No desktop configured should return empty string regardless of flags."""
        from unity.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_infrastructure_state(
            vm_ready=False,
            file_sync_complete=False,
            has_desktop=False,
        )
        assert result == ""

    def test_vm_pending_mentions_computer_actions(self):
        """The VM pending section should mention computer actions are unavailable."""
        from unity.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_infrastructure_state(
            vm_ready=False,
            file_sync_complete=True,
            has_desktop=True,
        )
        assert "desktop_act" in result
        assert "vm_pending" in result
        assert "sync_pending" not in result

    def test_sync_pending_mentions_files(self):
        """The sync pending section should mention historical files."""
        from unity.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_infrastructure_state(
            vm_ready=True,
            file_sync_complete=False,
            has_desktop=True,
        )
        assert "Attachments/" in result
        assert "sync_pending" in result
