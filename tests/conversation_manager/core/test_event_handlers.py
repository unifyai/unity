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
from typing import ClassVar
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from types import SimpleNamespace

from tests.helpers import _handle_project
from unify.conversation_manager.domains.event_handlers import (
    EventHandler,
    OPEN_SLOW_BRAIN_TURN_NOTIFICATION,
    _MEET_STATE_FLAGS,
    _event_type_to_log_key,
)
from unify.conversation_manager.cm_types import ScreenshotEntry
from unify.conversation_manager.events import (
    Event,
    Ping,
    SMSReceived,
    SMSSent,
    WhatsAppSent,
    EmailReceived,
    EmailSent,
    UnifyMessageReceived,
    UnifyMessageSent,
    PhoneCallReceived,
    PhoneCallStarted,
    PhoneCallEnded,
    PhoneCallAnswered,
    WhatsAppCallInviteSent,
    WhatsAppCallPermissionResponse,
    UnifyMeetReceived,
    UnifyMeetStarted,
    UnifyMeetEnded,
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    OutboundPhoneUtterance,
    FastBrainNotification,
    AssistantTurnInjected,
    ProactiveSpeechControl,
    ActionStopRequested,
    GetChatHistory,
    ActorHandleStarted,
    ActorHandleResponse,
    ActorResult,
    ActorNotification,
    ActorClarificationRequest,
    InitializationComplete,
    OpenSlowBrainTurn,
    NotificationInjectedEvent,
    NotificationUnpinnedEvent,
    SyncContacts,
    TaskDue,
    TaskTriggerRequested,
    LogMessageResponse,
    DirectMessageEvent,
    AssistantUpdateEvent,
    AssistantPresenceObserved,
    AssistantScreenShareStarted,
    AssistantScreenShareStopped,
    UserScreenShareStarted,
    UserScreenShareStopped,
    UserRemoteControlStarted,
    UserRemoteControlStopped,
    UserWebcamStarted,
    UserWebcamStopped,
    UserFilesysAccessStarted,
    UserFilesysAccessStopped,
)
from unify.contact_manager.simulated import SimulatedContactManager
from unify.conversation_manager.domains.contact_index import ContactIndex
from unify.conversation_manager.domains.notifications import NotificationBar
from unify.conversation_manager.medium_scripts.common import TRACK_AUTODETECT_REASON
from unify.conversation_manager.cm_types import Medium, Mode
from unify.task_scheduler.machine_state import TaskExecutionSnapshot

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
    manager.cleanup_google_meet = AsyncMock()
    manager.cleanup_teams_meet = AsyncMock()
    manager.set_hang_up_gate = AsyncMock()
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
    manager.pending_opener = ""
    manager.pending_briefing = ""
    manager.active_call_briefing = ""
    manager.conference_name = None
    manager.call_contact = None
    manager.call_exchange_id = -1
    manager.unify_meet_exchange_id = -1
    manager.unify_meet_call_session_id = ""
    manager.refresh_persistent_worker_after_key_change = AsyncMock()
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
    cm._frontend_reported_meet_surfaces = set()
    cm.memory_manager = None

    # Real viewer-set behaviour: the handler derives
    # ``assistant_screen_share_active`` from this set, so a MagicMock's
    # auto-returned truthy stub would make every assertion about the flag
    # meaningless.
    from unify.conversation_manager.conversation_manager import ConversationManager

    cm._assistant_screen_share_viewers = set()
    cm.assistant_screen_share_viewer_key = (
        ConversationManager.assistant_screen_share_viewer_key
    )
    cm.note_assistant_screen_share_viewer = (
        ConversationManager.note_assistant_screen_share_viewer.__get__(
            cm,
            ConversationManager,
        )
    )
    cm.drop_assistant_screen_share_viewers = (
        ConversationManager.drop_assistant_screen_share_viewers.__get__(
            cm,
            ConversationManager,
        )
    )
    cm.assistant_desktop_watched_from_call = (
        ConversationManager.assistant_desktop_watched_from_call.__get__(
            cm,
            ConversationManager,
        )
    )
    cm.drop_stale_call_screen_share_viewers = (
        ConversationManager.drop_stale_call_screen_share_viewers.__get__(
            cm,
            ConversationManager,
        )
    )

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

    # LLM generation counter (commissioning duplicate guard, tool execution metadata)
    cm._llm_gen = 0

    # Mock async methods
    cm.request_llm_run = AsyncMock()
    cm.cancel_proactive_speech = AsyncMock()
    cm.schedule_proactive_speech = AsyncMock()
    cm.handle_voice_user_turn = AsyncMock()
    cm.get_active_contact = MagicMock(return_value=sample_contacts[1])
    cm.is_coordinator = False
    cm.coordinator_onboarding_active = False
    cm._refresh_coordinator_onboarding_state = AsyncMock()
    cm.learning_demo_storage_wake_armed = False

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
            AssistantPresenceObserved,
            TaskTriggerRequested,
        ]
        for event_cls in expected_events:
            assert (
                event_cls in EventHandler._registry
            ), f"{event_cls.__name__} should be registered"

    def test_unregistered_event_returns_sleep(self):
        """Verify that unregistered events return a no-op coroutine."""
        import uuid

        # A freshly-created event class has no registered handler.
        unregistered_cls = type(
            f"_UnregisteredEvent_{uuid.uuid4().hex[:8]}",
            (Event,),
            {},
        )
        result = EventHandler._registry.get(unregistered_cls)
        assert result is None, "A fresh event class should have no handler"

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
            "unify.conversation_manager.domains.event_handlers.managers_utils",
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
            "unify.conversation_manager.domains.event_handlers.asyncio.create_task",
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
            "unify.conversation_manager.domains.event_handlers.managers_utils",
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
            "unify.conversation_manager.domains.event_handlers.managers_utils",
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
            "unify.conversation_manager.domains.event_handlers.managers_utils",
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
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called_once_with(
            triggering_contact_id=2,
            credit_gate_reply_context={"medium": "sms_message", "contact_id": 2},
        )

    @pytest.mark.asyncio
    async def test_sms_sent_updates_contact_index(self, mock_cm):
        """SMSSent adds message with assistant role."""
        event = SMSSent(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
            content="Reply to you",
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.SMS_MESSAGE)
        assert len(msgs) == 1
        # Sent messages have assistant role, not user

    @pytest.mark.asyncio
    async def test_whatsapp_template_sent_records_delivered_template(self, mock_cm):
        """Template fallback history shows what WhatsApp actually delivered."""
        event = WhatsAppSent(
            contact={"contact_id": 2, "first_name": "Alice", "surname": "Smith"},
            content="The clue is Blade Runner.",
            via_template=True,
            delivered_content=(
                "Hello Alice, this is T-W1N from Unify. I have a message for you. "
                "Reply here and I'll share the details!"
            ),
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(
            2,
            Medium.WHATSAPP_MESSAGE,
        )
        assert len(msgs) == 1
        assert "Hello Alice, this is T-W1N from Unify" in msgs[0].content
        assert "The clue is Blade Runner." in msgs[0].content
        assert "template fallback" in msgs[0].content
        assert (
            "WhatsApp template fallback sent"
            in mock_cm.notifications_bar.notifications[0].content
        )

    @pytest.mark.asyncio
    async def test_email_received_stores_subject_and_body(self, mock_cm):
        """EmailReceived stores subject, body, and email_id."""
        event = EmailReceived(
            contact={"contact_id": 2, "email_address": "alice@example.com"},
            subject="Important Update",
            body="Please review the attached.",
            email_id="msg_123",
            thread_id="gmail-thread-123",
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.EMAIL)
        assert len(msgs) == 1
        assert msgs[0].subject == "Important Update"
        assert msgs[0].body == "Please review the attached."
        assert msgs[0].thread_id == "gmail-thread-123"

    @pytest.mark.asyncio
    async def test_unify_message_received_updates_index(self, mock_cm):
        """UnifyMessageReceived adds to unify_message thread."""
        event = UnifyMessageReceived(
            contact={"contact_id": 2},
            content="Unify chat message",
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
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
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        # cancel_proactive_speech should NOT be called for sent (assistant) messages
        mock_cm.cancel_proactive_speech.assert_not_called()


class TestOutboundSentWakePolicy:
    """Outbound *Sent events honor suppress_slow_brain_wake on the event."""

    @pytest.mark.asyncio
    async def test_email_sent_with_suppress_flag_skips_llm_run(self, mock_cm):
        event = EmailSent(
            contact={"contact_id": 2, "email_address": "alice@example.com"},
            subject="Update",
            body="Please review.",
            to=["alice@example.com"],
            suppress_slow_brain_wake=True,
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_not_called()
        assert len(mock_cm.notifications_bar.notifications) == 1

    @pytest.mark.asyncio
    async def test_email_sent_without_suppress_flag_wakes_slow_brain(self, mock_cm):
        event = EmailSent(
            contact={"contact_id": 2, "email_address": "alice@example.com"},
            subject="Update",
            body="Please review.",
            to=["alice@example.com"],
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called_once_with(triggering_contact_id=2)

    @pytest.mark.asyncio
    async def test_unify_message_sent_with_suppress_flag_skips_llm_run(self, mock_cm):
        event = UnifyMessageSent(
            contact={"contact_id": 2},
            content="Onboarding reply",
            suppress_slow_brain_wake=True,
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_unify_message_sent_without_suppress_flag_wakes_slow_brain(
        self,
        mock_cm,
    ):
        event = UnifyMessageSent(
            contact={"contact_id": 2},
            content="Actor follow-up",
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called_once_with(
            triggering_contact_id=2,
            credit_gate_reply_context=None,
        )


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

        Call guidance is pre-computed via make_call(opener=...) before the call
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
            triggering_contact_id=2,
        )

    @pytest.mark.asyncio
    async def test_phone_call_ended_preserves_in_flight_actions(self, mock_cm):
        """PhoneCallEnded leaves action lifecycle decisions to the slow brain."""
        mock_cm.mode = Mode.CALL
        mock_cm.contact_index.push_message(
            contact_id=2,
            sender_name="Alice",
            thread_name=Medium.PHONE_CALL,
            message_content="test",
        )
        mock_cm.contact_index.active_conversations[2].on_call = True

        handle = MagicMock()
        handle.done.return_value = False
        handle.stop = AsyncMock()
        mock_cm.in_flight_actions = {
            42: {
                "handle": handle,
                "instruction": "Research this and let me know when it is done",
                "handle_actions": [],
            },
        }

        event = PhoneCallEnded(
            contact={"contact_id": 2, "phone_number": "+15555552222"},
        )

        await EventHandler.handle_event(event, mock_cm)

        handle.stop.assert_not_called()
        assert 42 in mock_cm.in_flight_actions
        assert 42 not in mock_cm.completed_actions
        mock_cm.request_llm_run.assert_called_once_with(
            delay=0,
            triggering_contact_id=2,
        )

    @pytest.mark.asyncio
    async def test_whatsapp_permission_keeps_legacy_room_name_when_agent_id_missing(
        self,
        mock_cm,
    ):
        """Accepted WhatsApp permission should preserve the pre-refactor room-name fallback."""

        mock_cm._pending_whatsapp_call_openers = {
            2: {"opener": "pending call context", "briefing": ""},
        }
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
                "unify.conversation_manager.domains.comms_utils.start_whatsapp_call",
                new_callable=AsyncMock,
                return_value={"success": True},
            ) as mock_start_whatsapp_call,
            patch(
                "unify.conversation_manager.domains.event_handlers.SESSION_DETAILS",
            ) as mock_session_details,
        ):
            mock_session_details.assistant.agent_id = None
            mock_session_details.assistant.name = "Test Assistant"

            await EventHandler.handle_event(event, mock_cm)

        assert mock_start_whatsapp_call.await_args.kwargs["room_name"] == (
            "unity_None_whatsapp_call"
        )
        mock_cm.request_llm_run.assert_not_called()
        mock_cm.event_broker.publish.assert_awaited_once_with(
            "app:comms:whatsapp_call_sent",
            ANY,
        )

    @pytest.mark.asyncio
    async def test_whatsapp_call_invite_is_permission_request(self, mock_cm):
        event = WhatsAppCallInviteSent(
            contact={
                "contact_id": 2,
                "first_name": "Alice",
                "surname": "Smith",
            },
        )

        await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.WHATSAPP_CALL)
        assert msgs[-1].content == (
            "<WhatsApp Call Permission Request Sent: waiting for the user to allow calls>"
        )
        assert (
            "permission request sent"
            in mock_cm.notifications_bar.notifications[-1].content
        )

    @pytest.mark.asyncio
    async def test_whatsapp_permission_acceptance_says_calling_now(self, mock_cm):
        event = WhatsAppCallPermissionResponse(
            contact={
                "contact_id": 2,
                "first_name": "Alice",
                "surname": "Smith",
                "whatsapp_number": "+15555552222",
            },
            accepted=True,
        )

        await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.WHATSAPP_CALL)
        assert msgs[-1].content == "<WhatsApp Call Permission Granted: calling now>"
        assert "calling now" in mock_cm.notifications_bar.notifications[-1].content

    @pytest.mark.asyncio
    async def test_whatsapp_permission_acceptance_uses_persisted_opener(
        self,
        mock_cm,
    ):
        mock_cm._pending_whatsapp_call_openers = {}
        mock_cm.assistant_whatsapp_number = "+15550000000"
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
                "unify.conversation_manager.domains.comms_utils.get_pending_whatsapp_call_intent",
                new_callable=AsyncMock,
                return_value={"context": "Persisted call briefing"},
            ) as mock_get_intent,
            patch(
                "unify.conversation_manager.domains.comms_utils.clear_pending_whatsapp_call_intent",
                new_callable=AsyncMock,
            ) as mock_clear_intent,
            patch(
                "unify.conversation_manager.domains.comms_utils.start_whatsapp_call",
                new_callable=AsyncMock,
                return_value={"success": True},
            ) as mock_start_whatsapp_call,
            patch(
                "unify.conversation_manager.domains.event_handlers.SESSION_DETAILS",
            ) as mock_session_details,
        ):
            mock_session_details.assistant.agent_id = 7
            mock_session_details.assistant.name = "Test Assistant"
            mock_session_details.assistant.whatsapp_number = "+15550000000"

            await EventHandler.handle_event(event, mock_cm)

        mock_get_intent.assert_awaited_once_with(
            pool_number="+15550000000",
            contact_number="+15555552222",
        )
        mock_start_whatsapp_call.assert_awaited_once()
        assert mock_cm.call_manager.pending_opener == "Persisted call briefing"
        mock_clear_intent.assert_awaited_once_with(
            pool_number="+15550000000",
            contact_number="+15555552222",
        )
        mock_cm.event_broker.publish.assert_awaited_once_with(
            "app:comms:whatsapp_call_sent",
            ANY,
        )

    @pytest.mark.asyncio
    async def test_whatsapp_permission_acceptance_unpacks_stashed_context(
        self,
        mock_cm,
    ):
        """The per-contact stash is a dict; its fields must be unpacked onto
        the call manager, never assigned wholesale to ``pending_opener`` —
        ``start_call`` strips the opener and crashes on a dict."""
        mock_cm._pending_whatsapp_call_openers = {
            2: {
                "opener": "Call opener line.",
                "briefing": "Full task design.",
                "hang_up_gate": "End once the quiz answer is confirmed.",
            },
        }
        mock_cm.assistant_whatsapp_number = "+15550000000"
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
                "unify.conversation_manager.domains.comms_utils.start_whatsapp_call",
                new_callable=AsyncMock,
                return_value={"success": True},
            ),
            patch(
                "unify.conversation_manager.domains.comms_utils.clear_pending_whatsapp_call_intent",
                new_callable=AsyncMock,
            ),
            patch(
                "unify.conversation_manager.domains.event_handlers.SESSION_DETAILS",
            ) as mock_session_details,
        ):
            mock_session_details.assistant.agent_id = 7
            mock_session_details.assistant.name = "Test Assistant"

            await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.call_manager.pending_opener == "Call opener line."
        assert mock_cm.call_manager.pending_briefing == "Full task design."
        assert (
            mock_cm.call_manager.pending_hang_up_gate
            == "End once the quiz answer is confirmed."
        )

    @pytest.mark.asyncio
    async def test_whatsapp_permission_acceptance_tags_onboarding_outbound(
        self,
        mock_cm,
    ):
        """Permission-granted auto-calls must consume pending onboarding metadata."""
        from unify.conversation_manager.conversation_manager import (
            ConversationManager,
        )
        from unify.conversation_manager.events import WhatsAppCallSent

        mock_cm._pending_whatsapp_call_openers = {
            2: {"opener": "Call opener line.", "briefing": "Full task design."},
        }
        mock_cm._pending_onboarding_outbound = {
            "onboarding_trigger_step_id": "whatsapp-call-reference",
            "onboarding_reply_step_id": "whatsapp-call",
            "onboarding_request_id": "req-1",
            "onboarding_origin_event_id": "evt-1",
            "channel": "whatsapp_call",
            "tool_name": "make_whatsapp_call_to_boss",
            "expires_at": 1_000_000.0,
        }
        mock_cm.loop.time.return_value = 0.0
        mock_cm.consume_pending_onboarding_outbound = (
            ConversationManager.consume_pending_onboarding_outbound.__get__(
                mock_cm,
                ConversationManager,
            )
        )
        mock_cm.build_whatsapp_call_sent_event = (
            ConversationManager.build_whatsapp_call_sent_event.__get__(
                mock_cm,
                ConversationManager,
            )
        )
        mock_cm.assistant_whatsapp_number = "+15550000000"
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
                "unify.conversation_manager.domains.comms_utils.start_whatsapp_call",
                new_callable=AsyncMock,
                return_value={"success": True},
            ),
            patch(
                "unify.conversation_manager.domains.event_handlers.SESSION_DETAILS",
            ) as mock_session_details,
        ):
            mock_session_details.assistant.agent_id = 7
            mock_session_details.assistant.name = "Test Assistant"

            await EventHandler.handle_event(event, mock_cm)

        published_event = WhatsAppCallSent.from_json(
            mock_cm.event_broker.publish.await_args.args[1],
        )
        assert published_event.onboarding_trigger_step_id == "whatsapp-call-reference"
        assert published_event.onboarding_reply_step_id == "whatsapp-call"
        assert mock_cm._pending_onboarding_outbound is None

    @pytest.mark.asyncio
    async def test_whatsapp_permission_unknown_does_not_clear_pending_opener(
        self,
        mock_cm,
    ):
        mock_cm._pending_whatsapp_call_openers = {
            2: {"opener": "Call opener line.", "briefing": "Full task design."},
        }
        event = WhatsAppCallPermissionResponse(
            contact={
                "contact_id": 2,
                "first_name": "Alice",
                "surname": "Smith",
                "whatsapp_number": "+15555552222",
            },
            accepted=False,
            status="unknown_interaction",
        )

        await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.WHATSAPP_CALL)
        assert msgs[-1].content == (
            "<WhatsApp Call Permission Unknown: waiting for reconciliation>"
        )
        assert "did not include" in mock_cm.notifications_bar.notifications[-1].content
        assert mock_cm._pending_whatsapp_call_openers[2] == {
            "opener": "Call opener line.",
            "briefing": "Full task design.",
        }
        mock_cm.request_llm_run.assert_called_once()


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
            call_session_id="session-123",
            opening_config={
                "mode": "simulated",
                "simulated_utterance": "Hi, I'm T-W1N.",
                "source": "twin_onboarding_intro",
            },
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.call_manager.start_unify_meet.assert_called_once()
        assert (
            mock_cm.call_manager.start_unify_meet.await_args.kwargs["call_session_id"]
            == "session-123"
        )
        assert mock_cm.call_manager.start_unify_meet.await_args.kwargs[
            "opening_config"
        ] == {
            "mode": "simulated",
            "simulated_utterance": "Hi, I'm T-W1N.",
            "source": "twin_onboarding_intro",
        }

    @pytest.mark.asyncio
    async def test_unify_meet_started_sets_mode(self, mock_cm):
        """UnifyMeetStarted sets mode to 'unify_meet'."""
        mock_cm.mode = Mode.TEXT
        mock_cm.call_manager.unify_meet_call_session_id = "session-123"
        event = UnifyMeetStarted(
            contact={"contact_id": 1},
            call_session_id="session-123",
        )

        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.mode == Mode.MEET
        assert mock_cm.call_manager.unify_meet_call_session_id == "session-123"

    @pytest.mark.asyncio
    async def test_unify_meet_started_ignores_stale_session(self, mock_cm):
        """UnifyMeetStarted from an old agent job does not replace the active session."""
        mock_cm.mode = Mode.TEXT
        mock_cm.call_manager.unify_meet_call_session_id = "current-session"
        event = UnifyMeetStarted(
            contact={"contact_id": 1},
            call_session_id="old-session",
        )

        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.mode == Mode.TEXT
        mock_cm.call_manager.cleanup_call_proc.assert_not_called()

    @pytest.mark.asyncio
    async def test_unify_meet_ended_resets_mode(self, mock_cm):
        """UnifyMeetEnded resets mode to 'text'."""
        mock_cm.mode = Mode.MEET
        mock_cm.call_manager.unify_meet_call_session_id = "current-session"
        mock_cm.contact_index.push_message(
            contact_id=1,
            sender_name="Boss",
            thread_name=Medium.UNIFY_MEET,
            message_content="test",
        )
        mock_cm.contact_index.active_conversations[1].on_call = True

        event = UnifyMeetEnded(
            contact={"contact_id": 1},
            call_session_id="current-session",
        )

        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.mode == Mode.TEXT

    @pytest.mark.asyncio
    async def test_unify_meet_ended_ignores_stale_session(self, mock_cm):
        """UnifyMeetEnded from an old agent job does not tear down the current call."""
        mock_cm.mode = Mode.MEET
        mock_cm.call_manager.unify_meet_call_session_id = "current-session"
        event = UnifyMeetEnded(contact={"contact_id": 1}, call_session_id="old-session")

        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.mode == Mode.MEET
        mock_cm.call_manager.cleanup_call_proc.assert_not_called()

    @pytest.mark.asyncio
    async def test_unify_meet_received_resets_meet_surfaces(self, mock_cm):
        """A new call starts with its shared surfaces closed."""
        mock_cm.mode = Mode.TEXT
        event = UnifyMeetReceived(
            contact={"contact_id": 1},
            room_name="room_123",
            call_session_id="session-123",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.reset_meet_surfaces.assert_called_once()

    @pytest.mark.asyncio
    async def test_roster_refresh_leaves_meet_surfaces_alone(self, mock_cm):
        """A mid-call roster refresh is not a new call.

        Late joins arrive as a fresh dispatch for the session already running,
        so resetting here would close a share while it is still on screen.
        """
        mock_cm.mode = Mode.MEET
        mock_cm.call_manager.has_active_call = True
        mock_cm.call_manager.unify_meet_call_session_id = "session-123"
        mock_cm.call_manager.refresh_unify_meet_roster = AsyncMock()
        event = UnifyMeetReceived(
            contact={"contact_id": 1},
            room_name="room_123",
            call_session_id="session-123",
            participants=[{"kind": "human", "user_id": "u1"}],
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.call_manager.refresh_unify_meet_roster.assert_awaited_once()
        mock_cm.reset_meet_surfaces.assert_not_called()

    @pytest.mark.asyncio
    async def test_unify_meet_ended_resets_meet_surfaces(self, mock_cm):
        """Hanging up closes the shared surfaces the call opened."""
        mock_cm.mode = Mode.MEET
        mock_cm.call_manager.unify_meet_call_session_id = "current-session"
        mock_cm.contact_index.push_message(
            contact_id=1,
            sender_name="Boss",
            thread_name=Medium.UNIFY_MEET,
            message_content="test",
        )

        await EventHandler.handle_event(
            UnifyMeetEnded(
                contact={"contact_id": 1},
                call_session_id="current-session",
            ),
            mock_cm,
        )

        mock_cm.reset_meet_surfaces.assert_called_once()


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
            "unify.conversation_manager.domains.event_handlers.managers_utils",
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
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.schedule_proactive_speech.assert_called()

    @pytest.mark.asyncio
    async def test_inbound_utterance_does_not_trigger_voice_user_turn_handler(
        self,
        mock_cm,
    ):
        """Inbound utterances are logged; slow brain runs after fast brain completes."""
        event = InboundPhoneUtterance(
            contact={"contact_id": 2},
            content="What's the weather?",
            turn_id=3,
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.handle_voice_user_turn.assert_not_called()

    @pytest.mark.asyncio
    async def test_fast_brain_turn_completed_triggers_voice_user_turn_handler(
        self,
        mock_cm,
    ):
        """Slow brain runs after the Voice Agent finishes a user turn."""
        from unify.conversation_manager.events import (
            FAST_BRAIN_TURN_SMALLTALK,
            FastBrainTurnCompleted,
        )

        event = FastBrainTurnCompleted(
            contact={"contact_id": 2},
            turn_id=7,
            user_content="How are you?",
            classification=FAST_BRAIN_TURN_SMALLTALK,
            intended_speech="Doing great, thanks!",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.handle_voice_user_turn.assert_called_once_with(
            "How are you?",
            triggering_contact_id=2,
            turn_id=7,
        )
        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.PHONE_CALL)
        assert len(msgs) == 1
        assert "Classification: SMALLTALK" in msgs[0].content
        assert "Doing great, thanks!" in msgs[0].content

    @pytest.mark.asyncio
    async def test_outbound_utterance_resets_proactive_speech(self, mock_cm):
        """Outbound utterances also reset (reschedule) proactive speech."""
        event = OutboundPhoneUtterance(
            contact={"contact_id": 2},
            content="Here's my response",
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.schedule_proactive_speech.assert_called()

    @pytest.mark.asyncio
    async def test_outbound_utterance_clears_matching_inflight_overlay(self, mock_cm):
        """When the real spoken utterance lands, the render-only in-flight overlay
        for that line is cleared (full or truncated-prefix match) and the
        speech-delivered signal (used to gate a deferred hang-up) is set."""
        import asyncio

        mock_cm._inflight_voice_speech = "The next step is to click Trigger email."
        mock_cm._inflight_speech_delivered = asyncio.Event()

        # A truncated prefix (barge-in) still matches and clears it.
        event = OutboundPhoneUtterance(
            contact={"contact_id": 2},
            content="The next step is to click",
        )
        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        assert mock_cm._inflight_voice_speech == ""
        assert mock_cm._inflight_speech_delivered.is_set()

    @pytest.mark.asyncio
    async def test_outbound_filler_does_not_clear_unrelated_inflight(self, mock_cm):
        """A short buffer filler that is not this line must not clear the overlay."""
        mock_cm._inflight_voice_speech = "The next step is to click Trigger email."

        event = OutboundPhoneUtterance(content="One moment.", contact={"contact_id": 2})
        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        assert (
            mock_cm._inflight_voice_speech == "The next step is to click Trigger email."
        )

    @pytest.mark.asyncio
    async def test_call_guidance_updates_contact_index(self, mock_cm):
        """FastBrainNotification adds guidance message to voice thread."""
        event = FastBrainNotification(
            contact={"contact_id": 2},
            message="Please mention the meeting at 3pm",
        )

        await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.PHONE_CALL)
        assert len(msgs) == 1
        # Guidance messages have role="guidance"

    @pytest.mark.asyncio
    async def test_voice_interrupt_records_unheard_remainder(self, mock_cm):
        """VoiceInterrupt records a guidance note naming the unheard remainder so
        the slow brain knows it was cut off."""
        from unify.conversation_manager.events import VoiceInterrupt

        event = VoiceInterrupt(
            contact={"contact_id": 2},
            spoken_prefix="I've sent the clue to your",
            unheard_remainder="email — reply with your guess.",
        )

        await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.PHONE_CALL)
        assert len(msgs) == 1
        # Recorded as an internal GuidanceMessage (no role; content carries the note).
        assert "email — reply with your guess." in msgs[0].content
        assert "interrupted" in msgs[0].content.lower()

    @pytest.mark.asyncio
    async def test_voice_interrupt_without_remainder_is_noop(self, mock_cm):
        """A VoiceInterrupt with no unheard remainder records nothing."""
        from unify.conversation_manager.events import VoiceInterrupt

        event = VoiceInterrupt(contact={"contact_id": 2})

        await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.PHONE_CALL)
        assert len(msgs) == 0

    @pytest.mark.asyncio
    async def test_fast_brain_turn_completed_skips_silence(self, mock_cm):
        from unify.conversation_manager.events import (
            FAST_BRAIN_TURN_SILENCE,
            FastBrainTurnCompleted,
        )

        event = FastBrainTurnCompleted(
            contact={"contact_id": 2},
            turn_id=7,
            user_content="Okay.",
            classification=FAST_BRAIN_TURN_SILENCE,
            intended_speech="",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.handle_voice_user_turn.assert_not_called()
        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.PHONE_CALL)
        assert len(msgs) == 0

    @pytest.mark.asyncio
    async def test_fast_brain_turn_completed_runs_for_undecided(self, mock_cm):
        """The third gate an undecided turn has to clear.

        The handler drops ``silence`` outright (above). ``undecided`` is silent
        too but must NOT be dropped: nothing was spoken, so the slow brain is the
        only thing left that can answer a turn that was in fact this
        assistant's, and it is also the only thing that can tell.
        """
        from unify.conversation_manager.events import (
            FAST_BRAIN_TURN_UNDECIDED,
            FastBrainTurnCompleted,
        )

        event = FastBrainTurnCompleted(
            contact={"contact_id": 2},
            turn_id=7,
            user_content="A-DA, where did we land on pricing?",
            classification=FAST_BRAIN_TURN_UNDECIDED,
            intended_speech="",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.handle_voice_user_turn.assert_called_once()
        # The guidance note is what tells the slow brain nothing was said; a
        # dropped note would leave it continuing a line that never existed.
        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.PHONE_CALL)
        assert len(msgs) == 1
        assert "NOTHING was said aloud" in msgs[0].content

    @pytest.mark.asyncio
    async def test_fast_brain_turn_completed_skips_empty_user_content(self, mock_cm):
        from unify.conversation_manager.events import (
            FAST_BRAIN_TURN_DEFER,
            FastBrainTurnCompleted,
        )

        event = FastBrainTurnCompleted(
            contact={"contact_id": 2},
            turn_id=7,
            user_content="   ",
            classification=FAST_BRAIN_TURN_DEFER,
            intended_speech="One moment.",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.handle_voice_user_turn.assert_not_called()

    @pytest.mark.asyncio
    async def test_assistant_turn_injection_updates_history_without_user_turn(
        self,
        mock_cm,
    ):
        """Injected assistant turns update history but do not wake a user turn."""
        mock_socket = MagicMock()
        mock_socket.queue_for_clients = AsyncMock()
        mock_cm.call_manager._socket_server = mock_socket
        event = AssistantTurnInjected(
            contact={"contact_id": 2},
            content="I just gave the onboarding intro.",
            source="test",
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        msgs = mock_cm.contact_index.get_messages_for_contact(2, Medium.PHONE_CALL)
        assert len(msgs) == 1
        assert msgs[0].role == "assistant"
        assert msgs[0].content == "I just gave the onboarding intro."
        mock_cm.handle_voice_user_turn.assert_not_called()
        mock_cm.request_llm_run.assert_not_called()
        mock_cm.schedule_proactive_speech.assert_not_called()
        mock_socket.queue_for_clients.assert_called_once()
        channel, payload = mock_socket.queue_for_clients.call_args.args
        assert channel == "app:call:notification"
        forwarded = AssistantTurnInjected.from_json(payload)
        assert forwarded.content == "I just gave the onboarding intro."

    @pytest.mark.asyncio
    async def test_assistant_turn_injection_can_schedule_proactive_speech(
        self,
        mock_cm,
    ):
        event = AssistantTurnInjected(
            contact={"contact_id": 2},
            content="I just gave the onboarding intro.",
            source="test",
            schedule_proactive=True,
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.schedule_proactive_speech.assert_called_once()

    @pytest.mark.asyncio
    async def test_proactive_speech_control_toggles_cm_gate(self, mock_cm):
        mock_cm.set_proactive_speech_enabled = AsyncMock()
        disable_event = ProactiveSpeechControl(enabled=False, source="test")
        enable_event = ProactiveSpeechControl(
            enabled=True,
            source="test",
            schedule_now=True,
        )

        await EventHandler.handle_event(disable_event, mock_cm)
        mock_cm.set_proactive_speech_enabled.assert_called_once_with(False)
        mock_cm.schedule_proactive_speech.assert_not_called()

        await EventHandler.handle_event(enable_event, mock_cm)
        assert mock_cm.set_proactive_speech_enabled.call_args_list[-1].args == (True,)
        mock_cm.schedule_proactive_speech.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_stop_requested_stops_matching_calling_id(self, mock_cm):
        mock_cm.stop_in_flight_action_by_calling_id = AsyncMock(return_value=True)
        event = ActionStopRequested(
            calling_id="call-xyz",
            reason="Stopped from Console Actions pane.",
            source="console",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.stop_in_flight_action_by_calling_id.assert_awaited_once_with(
            "call-xyz",
            reason="Stopped from Console Actions pane.",
        )


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
            1: {"query": "Test action", "action_type": "act", "handle_actions": []},
        }
        event = ActorResult(
            handle_id=1,
            success=True,
            result="Action completed successfully",
            action_type="act",
        )

        await EventHandler.handle_event(event, mock_cm)

        assert 1 not in mock_cm.in_flight_actions
        assert 1 in mock_cm.completed_actions
        assert mock_cm.completed_actions[1]["query"] == "Test action"
        # Result is recorded in handle_actions as act_completed event
        handle_actions = mock_cm.completed_actions[1]["handle_actions"]
        completion = next(
            a for a in handle_actions if a["action_name"] == "act_completed"
        )
        assert completion["success"] is True
        assert completion["action_type"] == "act"
        assert completion["result"] == "Action completed successfully"
        # No notification pushed (result is shown in completed_actions section)
        assert len(mock_cm.notifications_bar.notifications) == 0

    @pytest.mark.asyncio
    async def test_actor_result_failure_records_error_context(self, mock_cm):
        """ActorResult failure stores error context before completion handoff."""
        mock_cm.in_flight_actions = {
            7: {
                "query": "Fix coordinator memberships",
                "action_type": "act",
                "handle_actions": [],
            },
        }
        event = ActorResult(
            handle_id=7,
            success=False,
            result={"error_kind": "permission_denied"},
            error="Coordinator role required",
            action_type="act",
        )

        await EventHandler.handle_event(event, mock_cm)

        assert 7 not in mock_cm.in_flight_actions
        completed = mock_cm.completed_actions[7]
        completion = next(
            a for a in completed["handle_actions"] if a["action_name"] == "act_failed"
        )
        assert completion["success"] is False
        assert completion["action_type"] == "act"
        assert completion["error"] == "Coordinator role required"
        assert completion["result"] == {"error_kind": "permission_denied"}

    @pytest.mark.asyncio
    async def test_actor_notification_wakes_brain_when_armed_and_kind_matches(
        self,
        mock_cm,
    ):
        """storage_review_complete + learning-demo flag armed -> wake the brain."""
        mock_cm.in_flight_actions = {
            1: {"query": "Split Friday's dinner", "handle_actions": []},
        }
        mock_cm.learning_demo_storage_wake_armed = True
        event = ActorNotification(
            handle_id=1,
            response="Stored: bill-split rule, Sam fact, split_dinner_bill skill.",
            kind="storage_review_complete",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_actor_notification_no_wake_when_flag_not_armed(self, mock_cm):
        """Outside the learning-demo window the wake is gated off, even though
        the notification kind matches -- global behavior stays unchanged."""
        mock_cm.in_flight_actions = {
            1: {"query": "Split Friday's dinner", "handle_actions": []},
        }
        mock_cm.learning_demo_storage_wake_armed = False
        event = ActorNotification(
            handle_id=1,
            response="Stored: bill-split rule, Sam fact, split_dinner_bill skill.",
            kind="storage_review_complete",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_actor_notification_no_wake_for_other_kinds_even_when_armed(
        self,
        mock_cm,
    ):
        """The wake is specific to storage_review_complete, not any notification."""
        mock_cm.in_flight_actions = {
            1: {"query": "Split Friday's dinner", "handle_actions": []},
        }
        mock_cm.learning_demo_storage_wake_armed = True
        event = ActorNotification(
            handle_id=1,
            response="Still working...",
            kind="progress",
        )

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_actor_notification_records_late_notification_in_completed_actions(
        self,
        mock_cm,
    ):
        """A notification arriving after the handle moved to completed_actions
        (e.g. StorageCheck Phase 2 finishing after ActorResult already fired)
        must be recorded, not silently dropped."""
        mock_cm.in_flight_actions = {}
        mock_cm.completed_actions = {
            1: {"query": "Split Friday's dinner", "handle_actions": []},
        }
        event = ActorNotification(
            handle_id=1,
            response="Saved: bill-split rule, Sam fact, split_dinner_bill skill.",
            kind="storage_review_complete",
        )

        await EventHandler.handle_event(event, mock_cm)

        handle_actions = mock_cm.completed_actions[1]["handle_actions"]
        assert any(
            a["action_name"] == "progress" and "Saved" in a["query"]
            for a in handle_actions
        )

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


def _reset_surfaces_cm(surfaces, viewers, dropped=None, claimed=None):
    """Minimal stand-in for ``reset_meet_surfaces``, with a real viewer set.

    ``assistant_screen_share_active`` is derived from the viewer set, so the two
    are seeded together — a stub with the flag set and no viewers is a state the
    real object cannot reach.
    """
    from unify.conversation_manager.conversation_manager import ConversationManager

    def _drop(source):
        if dropped is not None:
            dropped.append(source)
        return 0

    cm = SimpleNamespace(
        _frontend_reported_meet_surfaces=set(
            surfaces if claimed is None else claimed,
        ),
        _assistant_screen_share_viewers=set(viewers),
        drop_unpaired_screenshots=_drop,
        **{name: True for name in surfaces},
    )
    cm.drop_assistant_screen_share_viewers = (
        ConversationManager.drop_assistant_screen_share_viewers.__get__(
            cm,
            ConversationManager,
        )
    )
    return cm


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
    async def test_a_call_share_is_one_switch_anyone_on_the_call_can_flip(
        self,
        mock_cm,
    ):
        """A call shows the desktop to everyone, so it is not a tally of people.

        The desktop goes up on the stage of a call every participant is watching,
        which makes it one piece of shared state rather than a view each person
        holds. Counted per person, the second participant to reach for the switch
        would be turning off a share they never started: the discard would miss,
        the desktop would stay up, and the control would do nothing.
        """
        call = "call:sess-1"
        for user_id in ("user-1", "user-2"):
            await EventHandler.handle_event(
                AssistantScreenShareStarted(
                    viewer_user_id=user_id,
                    viewer_source=call,
                ),
                mock_cm,
            )
        assert mock_cm.assistant_screen_share_active is True
        # The second press is the same switch, not a second thing to turn off,
        # and a live share is not announced to the assistant twice.
        assert len(mock_cm.notifications_bar.notifications) == 1

        await EventHandler.handle_event(
            AssistantScreenShareStopped(viewer_user_id="user-2", viewer_source=call),
            mock_cm,
        )
        assert mock_cm.assistant_screen_share_active is False

    @pytest.mark.asyncio
    async def test_a_stop_from_one_surface_leaves_another_surface_watching(
        self,
        mock_cm,
    ):
        """The Desktop tab and a call are separate viewers of one desktop."""
        await EventHandler.handle_event(
            AssistantScreenShareStarted(
                viewer_user_id="user-1",
                viewer_source="desktop_pane",
            ),
            mock_cm,
        )
        await EventHandler.handle_event(
            AssistantScreenShareStarted(
                viewer_user_id="user-1",
                viewer_source="call:sess-1",
            ),
            mock_cm,
        )

        # Leaving the call closes only what the call owned.
        assert mock_cm.drop_assistant_screen_share_viewers("call:sess-1") is True
        assert mock_cm.assistant_screen_share_active is True

        await EventHandler.handle_event(
            AssistantScreenShareStopped(
                viewer_user_id="user-1",
                viewer_source="desktop_pane",
            ),
            mock_cm,
        )
        assert mock_cm.assistant_screen_share_active is False

    @pytest.mark.asyncio
    async def test_a_client_that_sends_no_viewer_still_opens_and_closes(
        self,
        mock_cm,
    ):
        """A Console older than viewer tracking must not pin the desktop open."""
        await EventHandler.handle_event(AssistantScreenShareStarted(), mock_cm)
        assert mock_cm.assistant_screen_share_active is True

        await EventHandler.handle_event(AssistantScreenShareStopped(), mock_cm)
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
    async def test_track_autodetect_drops_once_frontend_owns_surface(self, mock_cm):
        """A frontend report silences track-inferred events for that surface.

        Both sources describe the same screen share, so without this the
        assistant is told twice that sharing began.
        """
        mock_cm.mode = Mode.MEET
        mock_cm.user_screen_share_active = False

        await EventHandler.handle_event(
            UserScreenShareStarted(reason="User started sharing their screen"),
            mock_cm,
        )
        count_after_frontend = len(mock_cm.notifications_bar.notifications)

        await EventHandler.handle_event(
            UserScreenShareStarted(reason=TRACK_AUTODETECT_REASON),
            mock_cm,
        )

        assert len(mock_cm.notifications_bar.notifications) == count_after_frontend
        assert mock_cm.user_screen_share_active is True

    @pytest.mark.asyncio
    async def test_stale_track_event_cannot_flip_frontend_owned_flag(self, mock_cm):
        """A late track unsubscribe must not undo a fresh frontend report.

        LiveKit can hold a camera track until room teardown, long after the user
        switched it off and back on in the UI.
        """
        mock_cm.mode = Mode.MEET
        mock_cm.user_webcam_active = False

        await EventHandler.handle_event(UserWebcamStarted(), mock_cm)
        assert mock_cm.user_webcam_active is True

        await EventHandler.handle_event(
            UserWebcamStopped(reason=TRACK_AUTODETECT_REASON),
            mock_cm,
        )

        assert mock_cm.user_webcam_active is True

    @pytest.mark.asyncio
    async def test_track_autodetect_applies_without_a_frontend(self, mock_cm):
        """With no frontend reporting, track inference still drives the state.

        This is the LiveKit Agents Playground, which has no Console to announce
        that a developer started sharing.
        """
        mock_cm.mode = Mode.MEET
        mock_cm.user_screen_share_active = False
        initial_count = len(mock_cm.notifications_bar.notifications)

        await EventHandler.handle_event(
            UserScreenShareStarted(reason=TRACK_AUTODETECT_REASON),
            mock_cm,
        )

        assert mock_cm.user_screen_share_active is True
        assert len(mock_cm.notifications_bar.notifications) == initial_count + 1

    @pytest.mark.asyncio
    async def test_repeated_frontend_event_does_not_renotify(self, mock_cm):
        """Restating the current state carries no new information."""
        mock_cm.mode = Mode.MEET
        mock_cm.user_screen_share_active = False

        await EventHandler.handle_event(
            UserScreenShareStarted(reason="User started sharing their screen"),
            mock_cm,
        )
        count_after_first = len(mock_cm.notifications_bar.notifications)

        await EventHandler.handle_event(
            UserScreenShareStarted(reason="User started sharing their screen"),
            mock_cm,
        )

        assert len(mock_cm.notifications_bar.notifications) == count_after_first

    def test_every_meet_surface_is_scoped(self):
        """Every handled surface is classified as call- or desktop-scoped.

        A call boundary closes one group and must not touch the other, so a new
        surface that joins ``_MEET_STATE_FLAGS`` without picking a side would
        otherwise be silently treated as desktop-scoped — never reset, and
        leaking across calls exactly as before.
        """
        from unify.conversation_manager.conversation_manager import (
            CALL_SCOPED_MEET_SURFACES,
            DESKTOP_SCOPED_MEET_SURFACES,
        )

        handled = {attr for attr, _ in _MEET_STATE_FLAGS.values()}
        call_scoped = set(CALL_SCOPED_MEET_SURFACES)
        desktop_scoped = set(DESKTOP_SCOPED_MEET_SURFACES)

        assert call_scoped | desktop_scoped == handled
        assert call_scoped.isdisjoint(desktop_scoped)

    def test_reset_meet_surfaces_spares_the_assistant_desktop(self):
        """A call boundary closes the call's surfaces and only those.

        The Console's Desktop tab opens the assistant's desktop with no call in
        sight, so clearing it on a call boundary would tell the assistant nobody
        is watching while that pane is still open — and hand back control the
        user still holds.
        """
        from unify.conversation_manager.conversation_manager import (
            CALL_SCOPED_MEET_SURFACES,
            DESKTOP_SCOPED_MEET_SURFACES,
            ConversationManager,
        )

        every_surface = (*CALL_SCOPED_MEET_SURFACES, *DESKTOP_SCOPED_MEET_SURFACES)
        cm = _reset_surfaces_cm(every_surface, viewers={"desktop_pane:user-1"})

        ConversationManager.reset_meet_surfaces(cm)  # type: ignore[arg-type]

        for name in CALL_SCOPED_MEET_SURFACES:
            assert getattr(cm, name) is False, name
        for name in DESKTOP_SCOPED_MEET_SURFACES:
            assert getattr(cm, name) is True, name
        assert cm._frontend_reported_meet_surfaces == set(DESKTOP_SCOPED_MEET_SURFACES)
        assert cm._assistant_screen_share_viewers == {"desktop_pane:user-1"}

    def test_reset_meet_surfaces_closes_the_desktop_the_call_was_watching(self):
        """With only call viewers, the boundary does close the desktop.

        The counterpart to sparing the Desktop tab: viewers that belonged to the
        call go with it, so a share nobody kept open outside the call does not
        leak into the next one — and remote control cannot stay held by someone
        who is no longer even watching.
        """
        from unify.conversation_manager.conversation_manager import (
            CALL_SCOPED_MEET_SURFACES,
            DESKTOP_SCOPED_MEET_SURFACES,
            ConversationManager,
        )

        every_surface = (*CALL_SCOPED_MEET_SURFACES, *DESKTOP_SCOPED_MEET_SURFACES)
        cm = _reset_surfaces_cm(
            every_surface,
            viewers={"call:sess-1:user-1", "call:sess-1:user-2"},
        )

        ConversationManager.reset_meet_surfaces(cm)  # type: ignore[arg-type]

        assert cm._assistant_screen_share_viewers == set()
        assert cm.assistant_screen_share_active is False
        assert cm.user_remote_control_active is False
        assert cm._frontend_reported_meet_surfaces == set()

    def test_reset_meet_surfaces_drops_the_frames_those_surfaces_fed(self):
        """Closing a surface takes its unpaired frames with it.

        A flag saying nobody is sharing while the buffer still offers that screen
        as current is the same stale-visual bug in a subtler form: the next turn
        describes a screen that has been taken down.
        """
        from unify.conversation_manager.conversation_manager import (
            CALL_SCOPED_MEET_SURFACES,
            DESKTOP_SCOPED_MEET_SURFACES,
            ConversationManager,
        )

        dropped: list[str] = []
        every_surface = (*CALL_SCOPED_MEET_SURFACES, *DESKTOP_SCOPED_MEET_SURFACES)
        # A Desktop tab is still open, so the assistant's own surface survives
        # the boundary and keeps its frames.
        cm = _reset_surfaces_cm(
            every_surface,
            viewers={"desktop_pane:user-1"},
            dropped=dropped,
            claimed=set(),
        )

        ConversationManager.reset_meet_surfaces(cm)  # type: ignore[arg-type]

        expected = [
            source
            for sources in CALL_SCOPED_MEET_SURFACES.values()
            for source in sources
        ]
        assert sorted(dropped) == sorted(expected)
        # The assistant's own frames belong to a surface that survives the call.
        assert "assistant" not in dropped

    @pytest.mark.asyncio
    async def test_frontend_surface_claim_does_not_outlive_its_call(self, mock_cm):
        """A 1:1 call's frontend claim must not silence the next call's tracks.

        Surface ownership is per call: the Console speaks for the surfaces it
        renders, and an org call has no Console reporting on either of them.
        Carrying the claim forward left the assistant capturing frames from a
        share it never registered as having started.
        """
        from unify.conversation_manager.conversation_manager import (
            ConversationManager,
        )

        mock_cm.reset_meet_surfaces = lambda: ConversationManager.reset_meet_surfaces(
            mock_cm,
        )
        mock_cm.mode = Mode.MEET
        mock_cm.user_screen_share_active = False
        mock_cm.call_manager.unify_meet_call_session_id = "call-one"
        mock_cm.contact_index.push_message(
            contact_id=1,
            sender_name="Boss",
            thread_name=Medium.UNIFY_MEET,
            message_content="test",
        )

        # Call one: the Console reports the share, claiming the surface.
        await EventHandler.handle_event(
            UserScreenShareStarted(reason="User started sharing their screen"),
            mock_cm,
        )
        assert "user_screen_share_active" in mock_cm._frontend_reported_meet_surfaces

        await EventHandler.handle_event(
            UnifyMeetEnded(contact={"contact_id": 1}, call_session_id="call-one"),
            mock_cm,
        )
        await EventHandler.handle_event(
            UnifyMeetReceived(
                contact={"contact_id": 1},
                room_name="room_two",
                call_session_id="call-two",
            ),
            mock_cm,
        )

        # Call two has no frontend on the surface, so the track speaks for it.
        count_before = len(mock_cm.notifications_bar.notifications)
        await EventHandler.handle_event(
            UserScreenShareStarted(reason=TRACK_AUTODETECT_REASON),
            mock_cm,
        )

        assert mock_cm.user_screen_share_active is True
        assert len(mock_cm.notifications_bar.notifications) == count_before + 1

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

    @pytest.mark.asyncio
    async def test_assistant_presence_observed_is_noop(self, mock_cm):
        """AssistantPresenceObserved keeps the runtime warm without waking the brain."""
        event = AssistantPresenceObserved(
            reason="selection",
            source="assistant_profile",
            page_visibility="visible",
            occurred_at="2026-06-15T21:00:00.000Z",
        )
        await EventHandler.handle_event(event, mock_cm)

        mock_cm._session_logger.debug.assert_any_call(
            "assistant_presence_observed",
            "Assistant presence observed from assistant_profile.",
        )
        mock_cm.request_llm_run.assert_not_called()
        mock_cm.schedule_proactive_speech.assert_not_called()

    @pytest.mark.asyncio
    async def test_assistant_presence_observed_records_console_guidance(self, mock_cm):
        """The heartbeat hands Console's orientation text to the runtime.

        Delivery rides on presence because a Console must be running to
        publish it; once recorded, the runtime keeps the text for the session.
        """
        event = AssistantPresenceObserved(
            reason="keepwarm",
            source="assistant_profile",
            console_guidance_version="guidance-v2",
            console_guidance_brief="Console knowledge\nBrief.",
            console_guidance_full="Console knowledge\nFull.",
            console_action_catalogue="- `section:chat` — Chat",
        )
        await EventHandler.handle_event(event, mock_cm)

        mock_cm.record_console_presence.assert_called_once_with(
            version="guidance-v2",
            brief="Console knowledge\nBrief.",
            full="Console knowledge\nFull.",
            actions="- `section:chat` — Chat",
        )

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

        from unify.conversation_manager.conversation_manager import ConversationManager

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

    def _buffer(self, mock_cm):
        from unify.conversation_manager.conversation_manager import ConversationManager

        mock_cm._screenshot_buffer = []
        mock_cm._session_logger = MagicMock()
        return ConversationManager._buffer_screenshot.__get__(mock_cm)

    def test_unpaired_frames_from_one_source_collapse(self, mock_cm):
        """A shared screen left up must not pile a reel of frames into one turn.

        Unpaired frames arrive every few seconds for as long as somebody presents,
        and each one appended would land in the same state message and be
        registered as its own image. Only the newest describes what is on screen.
        """
        import json

        method = self._buffer(mock_cm)
        for i in range(5):
            method(
                json.dumps(
                    {
                        "b64": f"frame-{i}",
                        "utterance": "",
                        "source": "google_meet",
                        "attribution": "Ada",
                    },
                ),
            )

        assert len(mock_cm._screenshot_buffer) == 1
        assert mock_cm._screenshot_buffer[0].b64 == "frame-4"
        assert mock_cm._screenshot_buffer[0].attribution == "Ada"

    def test_frames_paired_with_speech_always_accumulate(self, mock_cm):
        """Those are evidence for a specific thing somebody said, not "now"."""
        import json

        method = self._buffer(mock_cm)
        for text in ("what is this?", "and this bit?"):
            method(
                json.dumps(
                    {
                        "b64": f"frame-{text}",
                        "utterance": text,
                        "source": "google_meet",
                    },
                ),
            )

        assert [e.utterance for e in mock_cm._screenshot_buffer] == [
            "what is this?",
            "and this bit?",
        ]

    def test_collapsing_does_not_cross_sources(self, mock_cm):
        """The assistant's screen and a shared screen are different pictures."""
        import json

        method = self._buffer(mock_cm)
        method(json.dumps({"b64": "meet", "utterance": "", "source": "google_meet"}))
        method(json.dumps({"b64": "desk", "utterance": "", "source": "assistant"}))

        assert [e.source for e in mock_cm._screenshot_buffer] == [
            "google_meet",
            "assistant",
        ]

    def test_an_unpaired_frame_does_not_overwrite_a_paired_one(self, mock_cm):
        """Losing the frame tied to a question would lose the question's answer."""
        import json

        method = self._buffer(mock_cm)
        method(
            json.dumps(
                {"b64": "asked", "utterance": "what is this?", "source": "google_meet"},
            ),
        )
        method(json.dumps({"b64": "ambient", "utterance": "", "source": "google_meet"}))

        assert [e.b64 for e in mock_cm._screenshot_buffer] == ["asked", "ambient"]

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

        from unify.conversation_manager.conversation_manager import ConversationManager

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

        from unify.conversation_manager.conversation_manager import ConversationManager

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
        import json as _json

        from unify.conversation_manager.medium_scripts.common import (
            CALL_DESKTOP_SHARE_SURFACE,
        )

        calls = mock_cm.event_broker.publish.call_args_list
        payloads = [
            _json.loads(c.args[1]).get("payload", {})
            for c in calls
            if c.args[0] == "app:call:notification"
        ]

        # Two publishes with two different jobs. The guidance tells the assistant
        # how to behave and is suppressed when the state has not moved; the state
        # sync tells the room what to mount and is restated every time, because
        # each client's copy can only be corrected by being told again.
        spoken = [p for p in payloads if p.get("message")]
        assert len(spoken) == 1
        assert "screen sharing" in spoken[0]["message"].lower()

        synced = [
            p
            for p in payloads
            if CALL_DESKTOP_SHARE_SURFACE in (p.get("meet_surface_state") or {})
        ]
        assert len(synced) == 1
        assert synced[0]["message"] == ""

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
        from unify.conversation_manager.domains.event_handlers import (
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
        content = event_data.get("payload", {}).get("message", "")
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
        from unify.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_meet_interaction_state(
            assistant_screen_share_active=False,
            user_screen_share_active=False,
            user_remote_control_active=False,
        )
        assert result == ""

    def test_render_meet_state_assistant_screen_share_only(self):
        """Only assistant screen share active produces a single section."""
        from unify.conversation_manager.domains.renderer import Renderer

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
        from unify.conversation_manager.domains.renderer import Renderer

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
        from unify.conversation_manager.domains.renderer import Renderer

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
        from unify.conversation_manager.domains.renderer import Renderer

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
        from unify.conversation_manager.domains.renderer import Renderer

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
        from unify.conversation_manager.domains.renderer import Renderer
        from unify.conversation_manager.domains.notifications import NotificationBar

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

        mock_cm.request_llm_run.assert_called_once_with(delay=0)

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
    async def test_task_due_executes_scheduler_without_requesting_llm(self, mock_cm):
        """Current due activations should start the scheduler execution path."""

        event = TaskDue(
            task_id=101,
            source_task_log_id=555,
            revision="rev-1",
            scheduled_for="2026-04-10T09:00:00+00:00",
            task_label="Morning briefing",
            task_summary="Prepare the morning update before the user checks in.",
            visibility_policy="silent_by_default",
            recurrence_hint="recurring",
        )

        with (
            patch(
                "unify.conversation_manager.domains.task_execution.validate_task_due_execution",
                return_value=(
                    TaskExecutionSnapshot(
                        assistant_id="42",
                        run_key="42:101",
                        task_id=101,
                        source_task_log_id=555,
                        revision="rev-1",
                    ),
                    None,
                ),
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.remember_live_task_run_provenance",
            ) as mock_remember_provenance,
            patch(
                "unify.conversation_manager.domains.task_execution._start_live_task_due_execution",
                new=AsyncMock(return_value=7),
            ) as mock_start_execution,
        ):
            await EventHandler.handle_event(event, mock_cm)

        assert len(mock_cm.notifications_bar.notifications) == 1
        notification = mock_cm.notifications_bar.notifications[0].content
        assert "Morning briefing" in notification
        assert "Prepare the morning update before the user checks in" in notification
        assert "work silently unless you genuinely need the user" in notification
        assert "started automatically" in notification
        remembered = mock_remember_provenance.call_args.args[0]
        assert remembered.assistant_id == "42"
        assert remembered.wake.value == "scheduled"
        assert remembered.scheduled_for == "2026-04-10T09:00:00+00:00"
        mock_start_execution.assert_awaited_once()
        mock_cm.request_llm_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_offline_activation_rejects_task_due_delivery(self, mock_cm):
        """A task_due wake for an offline activation must never reach the CM lane.

        Offline runs execute as dedicated one-shot Kubernetes Jobs (hosted) or
        local-scheduler subprocesses; a stale ``task_due`` delivery for a task
        that switched to offline execution is rejected as stale by activation
        validation.
        """

        from unify.conversation_manager.domains.task_execution import (
            _handle_task_due_event,
        )

        event = TaskDue(
            task_id=9,
            source_task_log_id=555,
            revision="rev-1",
            scheduled_for="2026-04-10T09:00:00+00:00",
            task_label="GTM stargazer poll",
        )
        offline_activation = TaskExecutionSnapshot(
            assistant_id="42",
            run_key="42:9",
            task_id=9,
            source_task_log_id=555,
            wake="scheduled",
            delivery="offline",
            revision="rev-1",
            scheduled_for="2026-04-10T09:00:00+00:00",
        )

        with (
            patch(
                "unify.task_scheduler.machine_state.get_open_task_execution",
                return_value=offline_activation,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution._start_live_task_due_execution",
            ) as mock_start_execution,
        ):
            should_request_llm = await _handle_task_due_event(event, mock_cm)

        assert should_request_llm is False
        mock_start_execution.assert_not_called()

    @pytest.mark.asyncio
    async def test_task_due_requires_scheduled_for(self):
        """scheduled_for is part of every task_due delivery's identity."""

        from unify.conversation_manager.domains.task_execution import (
            _task_due_event_from_wake_reason,
        )

        wake_reason = {
            "type": "task_due",
            "task_id": 9,
            "source_task_log_id": 555,
            "revision": "rev-1",
            "scheduled_for": "",
            "wake": "scheduled",
        }
        assert _task_due_event_from_wake_reason(wake_reason) is None

    @pytest.mark.asyncio
    async def test_task_due_start_executes_scheduler_with_scheduled_reason(
        self,
        mock_cm,
    ):
        """Direct due-task start should preserve scheduled activation provenance."""

        from unify.common.task_execution_context import current_task_execution_delegate
        from unify.conversation_manager.domains.task_execution import (
            _start_live_task_due_execution,
        )
        from unify.task_scheduler.types.activated_by import ActivatedBy

        event = TaskDue(
            task_id=101,
            source_task_log_id=555,
            revision="rev-1",
            scheduled_for="2026-04-10T09:00:00+00:00",
            task_label="Morning briefing",
        )
        activation = TaskExecutionSnapshot(
            assistant_id="42",
            run_key="42:101",
            task_id=101,
            source_task_log_id=555,
            revision="rev-1",
            task_name="Morning briefing",
        )
        mock_cm.actor = MagicMock()
        captured: dict[str, object] = {}
        fake_handle = MagicMock()

        async def _execute(**kwargs):
            captured.update(kwargs)
            captured["delegate"] = current_task_execution_delegate.get()
            return fake_handle

        fake_scheduler = MagicMock()
        fake_scheduler.execute = AsyncMock(side_effect=_execute)
        fake_scheduler._get_task_or_raise = MagicMock(
            return_value=SimpleNamespace(
                description="Deliver the overnight briefing summary unprompted.",
            ),
        )

        async def _noop(*args, **kwargs):
            return None

        with (
            patch(
                "unify.conversation_manager.domains.task_execution.ManagerRegistry.get_task_scheduler",
                return_value=fake_scheduler,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_result",
                new=_noop,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_notifications",
                new=_noop,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_clarifications",
                new=_noop,
            ),
        ):
            handle_id = await _start_live_task_due_execution(event, mock_cm, activation)

        assert handle_id in mock_cm.in_flight_actions
        assert captured["task_id"] == 101
        assert captured["_activated_by"] == ActivatedBy.schedule
        assert captured["delegate"] is not None
        assert (
            mock_cm.in_flight_actions[handle_id]["task_description"]
            == "Deliver the overnight briefing summary unprompted."
        )

    @pytest.mark.asyncio
    @_handle_project
    async def test_task_due_start_uses_real_scheduler_delegate_contract(
        self,
        mock_cm,
        monkeypatch,
    ):
        """Due-task startup should exercise the real scheduler/delegate boundary."""

        from unify.actor.simulated import SimulatedActor
        from unify.conversation_manager.domains.task_execution import (
            _start_live_task_due_execution,
        )
        from unify.task_scheduler import task_scheduler as task_scheduler_module
        from unify.task_scheduler.machine_state import (
            TaskRunProvenance,
            TaskRunReference,
            remember_live_task_run_provenance,
        )
        from unify.task_scheduler.task_scheduler import TaskScheduler
        from unify.task_scheduler.types.repetition import Frequency, RepeatPattern
        from unify.task_scheduler.types.schedule import Schedule

        calls: list[dict] = []
        actor = SimulatedActor(steps=0)
        original_act = actor.act

        async def _spy_act(*args, **kwargs):
            calls.append(kwargs)
            return await original_act(*args, **kwargs)

        actor.act = _spy_act  # type: ignore[method-assign]
        scheduler = TaskScheduler(actor=actor)
        task_id = scheduler._create_task(
            name="Scheduled integration report",
            description="Prepare the scheduled report.",
            schedule=Schedule(start_at="2026-04-10T09:00:00+00:00"),
            repeat=[RepeatPattern(frequency=Frequency.DAILY)],
        )["details"]["task_id"]
        # The real scheduler resolves the activation's source task by its
        # log id (the delegate contract this test exercises), so use the
        # actual scheduled instance's log id rather than a fabricated value.
        # The log id lives on the store row (``unisdk.Log.id``), not on the
        # sanitized ``Task`` returned by ``_filter_tasks``.
        source_task_log_id = None
        for _ctx in scheduler._read_task_contexts():
            _rows = scheduler._store_for_task_context(_ctx).get_rows(
                filter=f"task_id == {task_id}",
                return_ids_only=False,
            )
            if _rows:
                source_task_log_id = int(_rows[0].id)
                break
        assert source_task_log_id is not None
        event = TaskDue(
            task_id=task_id,
            source_task_log_id=source_task_log_id,
            revision="rev-1",
            scheduled_for="2026-04-10T09:00:00+00:00",
            task_label="Scheduled integration report",
        )
        activation = TaskExecutionSnapshot(
            assistant_id="42",
            run_key=f"42:{task_id}",
            task_id=task_id,
            source_task_log_id=source_task_log_id,
            revision="rev-1",
            task_name="Scheduled integration report",
        )
        mock_cm.actor = actor

        monkeypatch.setattr(
            task_scheduler_module.SESSION_DETAILS.assistant,
            "agent_id",
            42,
        )
        remember_live_task_run_provenance(
            TaskRunProvenance(
                assistant_id="42",
                task_id=task_id,
                wake="scheduled",
                delivery="live",
                source_task_log_id=source_task_log_id,
                revision="rev-1",
                scheduled_for="2026-04-10T09:00:00+00:00",
                task_name="Scheduled integration report",
            ),
        )
        monkeypatch.setattr(
            "unify.task_scheduler.active_task.create_or_adopt_live_task_run",
            lambda provenance: TaskRunReference(
                assistant_id=provenance.assistant_id,
                run_key="live:scheduled:42:0:rev-1:2026-04-10T09:00:00+00:00",
            ),
        )
        monkeypatch.setattr(
            "unify.task_scheduler.active_task.update_task_run_record",
            lambda *args, **kwargs: None,
        )

        async def _noop(*args, **kwargs):
            return None

        with (
            patch(
                "unify.conversation_manager.domains.task_execution.ManagerRegistry.get_task_scheduler",
                return_value=scheduler,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_result",
                new=_noop,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_notifications",
                new=_noop,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_clarifications",
                new=_noop,
            ),
        ):
            handle_id = await _start_live_task_due_execution(event, mock_cm, activation)

        assert handle_id in mock_cm.in_flight_actions
        assert calls
        assert calls[0]["guidelines"] is not None
        assert calls[0]["persist"] is False
        assert (
            mock_cm.in_flight_actions[handle_id]["task_description"]
            == "Prepare the scheduled report."
        )

        # ``Tasks`` is definition-only: one row per ``task_id`` for the whole
        # series, with run state living in ``Tasks/Executions``. Starting an
        # occurrence must therefore leave exactly one definition row, still armed
        # -- a run that disarmed its own series would silently stop recurring.
        # This previously asserted a ``status`` field, which the arming-flag
        # migration removed from the model altogether.
        rows = scheduler._filter_tasks(filter=f"task_id == {task_id}")
        assert len(rows) == 1
        assert rows[0].task_id == task_id
        assert rows[0].enabled is True

    @pytest.mark.asyncio
    async def test_task_due_start_failure_surfaces_error_without_llm_prompt(
        self,
        mock_cm,
    ):
        from unify.conversation_manager.domains.task_execution import (
            _handle_task_due_event,
        )

        event = TaskDue(
            task_id=101,
            source_task_log_id=555,
            revision="rev-1",
            scheduled_for="2026-04-10T09:00:00+00:00",
            task_label="Morning briefing",
        )

        with (
            patch(
                "unify.conversation_manager.domains.task_execution.validate_task_due_execution",
                return_value=(
                    TaskExecutionSnapshot(
                        assistant_id="42",
                        run_key="42:101",
                        task_id=101,
                        source_task_log_id=555,
                        revision="rev-1",
                        task_name="Morning briefing",
                    ),
                    None,
                ),
            ),
            patch(
                "unify.conversation_manager.domains.task_execution._start_live_task_due_execution",
                new=AsyncMock(side_effect=RuntimeError("delegate mismatch")),
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.publish_system_error",
            ) as mock_publish_system_error,
            patch(
                "unify.conversation_manager.domains.task_execution."
                "update_task_run_record",
            ) as mock_record_failure,
        ):
            should_request_llm = await _handle_task_due_event(event, mock_cm)

        # A turn is requested so the assistant can say the run did not
        # happen. Returning False here left the notification written below
        # unread until some unrelated later turn, and a run the user was
        # waiting on vanished with nothing said by any channel.
        assert should_request_llm is True
        assert len(mock_cm.notifications_bar.notifications) == 1
        notification = mock_cm.notifications_bar.notifications[0].content
        assert "failed to start through TaskScheduler.execute" in notification
        assert "delegate mismatch" in notification
        assert "started automatically" not in notification

        # The occurrence is terminalized with the real reason. Left open, it
        # keeps its seat as the definition's head and projection never mints
        # a successor -- one failure to start would end the series.
        mock_record_failure.assert_called_once()
        reference, entries = mock_record_failure.call_args.args
        assert reference.run_key == "42:101"
        assert reference.source_task_log_id == 555
        assert entries["state"] == "failed"
        assert "delegate mismatch" in entries["error"]
        assert entries["completed_at"]
        mock_publish_system_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_task_due_queues_fast_brain_context_during_voice_call(self, mock_cm):
        """Validated task_due events should also reach the fast brain silently."""

        event = TaskDue(
            task_id=101,
            source_task_log_id=555,
            revision="rev-1",
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

        with (
            patch(
                "unify.conversation_manager.domains.task_execution.validate_task_due_execution",
                return_value=(
                    TaskExecutionSnapshot(
                        assistant_id="42",
                        run_key="42:101",
                        task_id=101,
                        revision="rev-1",
                        task_name="Morning briefing",
                    ),
                    None,
                ),
            ),
            patch(
                "unify.conversation_manager.domains.task_execution._start_live_task_due_execution",
                new=AsyncMock(return_value=7),
            ),
        ):
            await EventHandler.handle_event(event, mock_cm)

        mock_socket.queue_for_clients.assert_called_once()
        channel, payload = mock_socket.queue_for_clients.call_args.args
        assert channel == "app:call:notification"
        guidance = FastBrainNotification.from_json(payload)
        assert guidance.should_speak is False
        assert "Morning briefing" in guidance.message
        assert (
            "Prepare the morning update before the user checks in" in guidance.message
        )
        assert "silent action unless the user is needed" in guidance.message

    @pytest.mark.asyncio
    async def test_task_due_ignores_stale_activation(self, mock_cm):
        """Stale due deliveries should not notify or re-run the slow brain."""

        event = TaskDue(
            task_id=101,
            source_task_log_id=555,
            revision="rev-1",
            scheduled_for="2026-04-10T09:00:00+00:00",
        )

        with patch(
            "unify.conversation_manager.domains.task_execution.validate_task_due_execution",
            return_value=(None, "revision_mismatch"),
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
                "revision": "rev-1",
                "scheduled_for": "2026-04-10T09:00:00+00:00",
                "task_label": "Morning briefing",
                "task_summary": "Prepare the morning update before the user checks in.",
                "visibility_policy": "silent_by_default",
                "recurrence_hint": "recurring",
            },
        ]

        with (
            patch(
                "unify.conversation_manager.domains.task_execution.validate_task_due_execution",
                return_value=(MagicMock(revision="rev-1"), None),
            ),
            patch(
                "unify.conversation_manager.domains.task_execution._start_live_task_due_execution",
                new=AsyncMock(return_value=7),
            ),
        ):
            await EventHandler.handle_event(InitializationComplete(), mock_cm)

        assert mock_cm._startup_wake_reasons == []
        assert len(mock_cm.notifications_bar.notifications) == 2
        assert any(
            "started automatically" in notif.content
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
        # Default to the nothing-restored boot; tests that pin the
        # history-loaded wording set a positive count explicitly.
        mock_cm._hydrated_history_count = 0
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
        from unify.conversation_manager.domains.event_handlers import (
            INITIALIZATION_COMPLETE_NOTIFICATION,
        )

        # Hydration restored history, so the notification may claim it.
        mock_cm._hydrated_history_count = 3
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
    async def test_notification_is_truthful_when_nothing_was_restored(
        self,
        mock_cm,
    ):
        """A boot whose hydration restored nothing must not claim that
        history was loaded.

        Regression test for the cold-boot amnesia incident (2026-08-22):
        on a deployment without a persisted Comms stream, the rebooted CM
        announced "full conversation history has been loaded" over an
        empty thread render, and the brain went hunting through email and
        desktop for context the notification told it already had.
        """
        from unify.conversation_manager.domains.event_handlers import (
            INITIALIZATION_COMPLETE_NO_HISTORY_NOTIFICATION,
        )

        await EventHandler.handle_event(InitializationComplete(), mock_cm)

        notif = next(
            n
            for n in mock_cm.notifications_bar.notifications
            if "Initialization complete" in n.content
        )
        assert notif.content == INITIALIZATION_COMPLETE_NO_HISTORY_NOTIFICATION

        text = notif.content.lower()
        assert "history has been loaded" not in text
        assert "no prior conversation history" in text
        # The anti-duplicate directive survives in both variants.
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
        assert "Initialization complete" in notification.message


class TestTriggeredTaskNotifications:
    """Tests for mechanically matched trigger-task notifications."""

    @pytest.mark.asyncio
    async def test_rest_task_trigger_starts_task_with_explicit_provenance(
        self,
        mock_cm,
    ):
        mock_cm.actor = object()
        fake_scheduler = MagicMock()
        fake_scheduler.execute = AsyncMock(return_value=object())

        event = TaskTriggerRequested(
            task_id=301,
            source_task_log_id=9001,
            source_ref="req-abc",
            task_label="Review report",
            task_summary="Review the weekly report.",
        )

        with (
            patch(
                "unify.conversation_manager.domains.task_execution.ManagerRegistry.get_task_scheduler",
                return_value=fake_scheduler,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution._register_live_task_handle",
                new_callable=AsyncMock,
                return_value=77,
            ) as register_handle,
            patch(
                "unify.conversation_manager.domains.task_execution._current_task_assistant_id",
                return_value="42",
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.get_open_task_execution",
                return_value=None,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.remember_live_task_run_provenance",
            ) as remember_provenance,
        ):
            await EventHandler.handle_event(event, mock_cm)

        fake_scheduler.execute.assert_awaited_once()
        assert fake_scheduler.execute.await_args.kwargs["task_id"] == 301
        assert (
            str(fake_scheduler.execute.await_args.kwargs["_activated_by"]) == "explicit"
        )
        register_handle.assert_awaited_once()
        provenance = remember_provenance.call_args.args[0]
        assert provenance.assistant_id == "42"
        assert provenance.task_id == 301
        assert provenance.wake.value == "explicit"
        assert provenance.source_task_log_id == 9001
        assert provenance.source_ref == "req-abc"
        mock_cm.request_llm_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_rest_task_trigger_start_passes_task_description_through(
        self,
        mock_cm,
    ):
        """REST-triggered task registration must carry the authored task
        description through to completed-actions rendering, so the brain
        sees delivery intent (e.g. "deliver unprompted") next to the result.
        """

        from types import SimpleNamespace

        from unify.conversation_manager.domains.renderer import Renderer
        from unify.conversation_manager.domains.task_execution import (
            _start_live_task_trigger_execution,
        )

        mock_cm.actor = MagicMock()
        fake_task = SimpleNamespace(
            description=(
                "Deliver this summary unprompted to Yusha via task "
                "completion delivery."
            ),
        )
        fake_scheduler = MagicMock()
        fake_scheduler.execute = AsyncMock(return_value=MagicMock())
        fake_scheduler._get_task_or_raise = MagicMock(return_value=fake_task)

        event = TaskTriggerRequested(
            task_id=301,
            source_task_log_id=9001,
            source_ref="req-abc",
            task_label="Review report",
            task_summary="Review the weekly report.",
        )

        async def _noop(*args, **kwargs):
            return None

        with (
            patch(
                "unify.conversation_manager.domains.task_execution.ManagerRegistry.get_task_scheduler",
                return_value=fake_scheduler,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_result",
                new=_noop,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_notifications",
                new=_noop,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_clarifications",
                new=_noop,
            ),
        ):
            handle_id = await _start_live_task_trigger_execution(event, mock_cm)

        fake_scheduler._get_task_or_raise.assert_called_once_with(301)
        assert (
            mock_cm.in_flight_actions[handle_id]["task_description"]
            == "Deliver this summary unprompted to Yusha via task completion delivery."
        )

        # Simulate the handle reaching completion and confirm the rendered
        # <completed_actions> block carries both tags the brain relies on.
        completed_actions = {
            handle_id: {
                **mock_cm.in_flight_actions[handle_id],
                "handle_actions": [
                    {
                        "action_name": "act_completed",
                        "query": "Report reviewed.",
                        "success": True,
                        "result": "Report reviewed.",
                    },
                ],
            },
        }
        rendered = Renderer().render_completed_actions(completed_actions)
        assert "<original_request>" in rendered
        assert (
            "<task_description>Deliver this summary unprompted to Yusha via "
            "task completion delivery.</task_description>" in rendered
        )

    @pytest.mark.asyncio
    async def test_rest_task_trigger_start_tolerates_missing_task_lookup(
        self,
        mock_cm,
    ):
        """A failed description lookup must not block task-trigger startup."""

        from unify.conversation_manager.domains.task_execution import (
            _start_live_task_trigger_execution,
        )

        mock_cm.actor = MagicMock()
        fake_scheduler = MagicMock()
        fake_scheduler.execute = AsyncMock(return_value=MagicMock())
        fake_scheduler._get_task_or_raise = MagicMock(
            side_effect=ValueError("No task found with id=301"),
        )

        event = TaskTriggerRequested(
            task_id=301,
            source_task_log_id=9001,
            source_ref="req-abc",
            task_label="Review report",
            task_summary="Review the weekly report.",
        )

        async def _noop(*args, **kwargs):
            return None

        with (
            patch(
                "unify.conversation_manager.domains.task_execution.ManagerRegistry.get_task_scheduler",
                return_value=fake_scheduler,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_result",
                new=_noop,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_notifications",
                new=_noop,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_clarifications",
                new=_noop,
            ),
        ):
            handle_id = await _start_live_task_trigger_execution(event, mock_cm)

        assert handle_id in mock_cm.in_flight_actions
        assert "task_description" not in mock_cm.in_flight_actions[handle_id]

    @pytest.mark.asyncio
    async def test_inbound_message_surfaces_trigger_candidates(self, mock_cm):
        """Inbound user messages should surface only live matching trigger tasks."""

        event = SMSReceived(
            contact={"contact_id": 2, "first_name": "Alice", "surname": "Smith"},
            content="Can you review the invoice?",
        )
        candidates = [
            TaskExecutionSnapshot(
                assistant_id="42",
                run_key="42:301",
                task_id=301,
                wake="triggered",
                delivery="live",
                trigger_from_contact_ids=[2],
                task_name="Invoice follow-up",
                task_summary="Help handle invoice-related requests from Alice.",
            ),
            TaskExecutionSnapshot(
                assistant_id="42",
                run_key="42:302",
                task_id=302,
                wake="triggered",
                delivery="offline",
                trigger_from_contact_ids=[2],
                task_name="Hidden offline task",
            ),
            TaskExecutionSnapshot(
                assistant_id="42",
                run_key="42:303",
                task_id=303,
                wake="triggered",
                delivery="live",
                trigger_omit_contact_ids=[2],
                task_name="Wrong sender task",
            ),
        ]

        with (
            patch(
                "unify.conversation_manager.domains.task_execution.list_trigger_executions",
                return_value=candidates,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.remember_live_task_run_provenance",
            ) as mock_remember_provenance,
            patch(
                "unify.conversation_manager.domains.task_execution._dispatch_offline_trigger_candidate",
                return_value={"status": "launched"},
            ) as mock_offline_dispatch,
            patch(
                "unify.settings.SETTINGS.task.LOCAL_SCHEDULER_ENABLED",
                False,
            ),
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
        assert remembered.wake.value == "triggered"
        assert remembered.source_medium == "sms_message"
        assert remembered.attempt_token
        mock_offline_dispatch.assert_called_once()
        mock_cm.request_llm_run.assert_called_once_with(
            triggering_contact_id=2,
            credit_gate_reply_context={"medium": "sms_message", "contact_id": 2},
        )

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
            TaskExecutionSnapshot(
                assistant_id="42",
                run_key="42:402",
                task_id=402,
                wake="triggered",
                delivery="offline",
                trigger_from_contact_ids=[2],
            ),
        ]

        with (
            patch(
                "unify.conversation_manager.domains.task_execution.list_trigger_executions",
                return_value=candidates,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution._dispatch_offline_trigger_candidate",
                return_value={"status": "launched"},
            ) as mock_offline_dispatch,
            patch(
                "unify.settings.SETTINGS.task.LOCAL_SCHEDULER_ENABLED",
                False,
            ),
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
            TaskExecutionSnapshot(
                assistant_id="42",
                run_key="42:401",
                task_id=401,
                wake="triggered",
                delivery="live",
                trigger_from_contact_ids=[2],
                task_name="Handle VIP caller",
                task_summary="Prioritize urgent inbound calls from Alice.",
            ),
        ]

        with patch(
            "unify.conversation_manager.domains.task_execution.list_trigger_executions",
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
            TaskExecutionSnapshot(
                assistant_id="42",
                run_key="42:401",
                task_id=401,
                wake="triggered",
                delivery="live",
                trigger_from_contact_ids=[2],
                task_name="Handle VIP caller",
                task_summary="Prioritize urgent inbound calls from Alice.",
            ),
        ]

        with patch(
            "unify.conversation_manager.domains.task_execution.list_trigger_executions",
            return_value=candidates,
        ):
            await EventHandler.handle_event(event, mock_cm)

        call_args = mock_socket.queue_for_clients.call_args_list
        assert call_args
        channel, payload = call_args[-1].args
        assert channel == "app:call:notification"
        guidance = FastBrainNotification.from_json(payload)
        assert guidance.should_speak is False
        assert "Handle VIP caller" in guidance.message
        assert "Prioritize urgent inbound calls from Alice" in guidance.message
        assert "Do not mention the task unless it naturally helps" in guidance.message


class TestProviderEventDispatchNotifications:
    """Tests for the live provider-event dispatch call site.

    This is the exact path behind the incident where a live task completed
    correctly but the reactive brain never saw the task's own authored
    description (only a generic dispatch string) next to the result.
    """

    @pytest.mark.asyncio
    async def test_provider_event_dispatch_registers_task_description(
        self,
        mock_cm,
    ):
        """A successful live start must carry outcome.description through to
        the registered handle so completed-actions rendering surfaces it.
        """

        from datetime import datetime, timezone

        from unify.conversation_manager.domains.renderer import Renderer
        from unify.conversation_manager.domains.task_execution import (
            _handle_provider_event_dispatch_requested_event,
        )
        from unify.conversation_manager.events import ProviderEventDispatchRequested
        from unify.task_scheduler.provider_event_dispatch import (
            LiveProviderEventDispatchOutcome,
        )

        mock_cm.actor = MagicMock()
        fake_handle = MagicMock()
        outcome = LiveProviderEventDispatchOutcome(
            operation_id="op-1",
            run_id=4242,
            run_key="run-key-1",
            captured_task_revision=3,
            status="started",
            fencing_token=7,
            adopted_only=False,
            description=(
                "Deliver this summary unprompted to Yusha via task "
                "completion delivery."
            ),
        )

        event = ProviderEventDispatchRequested(
            operation_id="op-1",
            run_id=4242,
            run_key="run-key-1",
            assistant_id="assistant-123",
            task_id=101,
            binding_id="binding-1",
            receipt_id="receipt-1",
            accepted_revision="rev-123",
            event_context_ref="blob://binding-1/receipt-1",
            issued_at=datetime.now(timezone.utc).isoformat(),
        )

        async def _noop(*args, **kwargs):
            return None

        with (
            patch(
                "unify.task_scheduler.provider_event_execution.handle_provider_event_live_dispatch",
                new=AsyncMock(return_value=(outcome, fake_handle)),
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_result",
                new=_noop,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_notifications",
                new=_noop,
            ),
            patch(
                "unify.conversation_manager.domains.task_execution.managers_utils.actor_watch_clarifications",
                new=_noop,
            ),
        ):
            should_request_llm = await _handle_provider_event_dispatch_requested_event(
                event,
                mock_cm,
            )

        assert should_request_llm is False
        assert len(mock_cm.in_flight_actions) == 1
        handle_id = next(iter(mock_cm.in_flight_actions))
        assert (
            mock_cm.in_flight_actions[handle_id]["task_description"]
            == "Deliver this summary unprompted to Yusha via task completion delivery."
        )

        # Simulate the handle reaching completion and confirm the rendered
        # <completed_actions> block carries both tags the brain relies on.
        completed_actions = {
            handle_id: {
                **mock_cm.in_flight_actions[handle_id],
                "handle_actions": [
                    {
                        "action_name": "act_completed",
                        "query": "Provider event completed.",
                        "success": True,
                        "result": "Provider event completed.",
                    },
                ],
            },
        }
        rendered = Renderer().render_completed_actions(completed_actions)
        assert "<original_request>" in rendered
        assert (
            "<task_description>Deliver this summary unprompted to Yusha via "
            "task completion delivery.</task_description>" in rendered
        )

    @pytest.mark.asyncio
    async def test_provider_event_dispatch_adopted_only_skips_handle_registration(
        self,
        mock_cm,
    ):
        """Adopt-only / already-terminal claims return handle=None and must
        never reach _register_live_task_handle, regardless of description.
        """

        from datetime import datetime, timezone

        from unify.conversation_manager.domains.task_execution import (
            _handle_provider_event_dispatch_requested_event,
        )
        from unify.conversation_manager.events import ProviderEventDispatchRequested
        from unify.task_scheduler.provider_event_dispatch import (
            LiveProviderEventDispatchOutcome,
        )

        mock_cm.actor = MagicMock()
        outcome = LiveProviderEventDispatchOutcome(
            operation_id="op-2",
            run_id=4243,
            run_key="run-key-2",
            captured_task_revision=3,
            status="adopted",
            fencing_token=7,
            adopted_only=True,
            description="Some other task's authored description.",
        )

        event = ProviderEventDispatchRequested(
            operation_id="op-2",
            run_id=4243,
            run_key="run-key-2",
            assistant_id="assistant-123",
            task_id=102,
            binding_id="binding-2",
            receipt_id="receipt-2",
            accepted_revision="rev-124",
            event_context_ref="blob://binding-2/receipt-2",
            issued_at=datetime.now(timezone.utc).isoformat(),
        )

        with patch(
            "unify.task_scheduler.provider_event_execution.handle_provider_event_live_dispatch",
            new=AsyncMock(return_value=(outcome, None)),
        ):
            should_request_llm = await _handle_provider_event_dispatch_requested_event(
                event,
                mock_cm,
            )

        assert should_request_llm is False
        assert mock_cm.in_flight_actions == {}


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
            "unify.conversation_manager.domains.event_handlers.managers_utils",
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
        from unify.contact_manager.types.contact import UNASSIGNED

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
        from unify.contact_manager.types.contact import UNASSIGNED

        mock_cm.call_manager.unify_meet_exchange_id = UNASSIGNED
        event = LogMessageResponse(
            medium="unify_meet",
            exchange_id=99,
            destination="team:11",
        )

        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.call_manager.unify_meet_exchange_id == 99
        # Cached with the id, because ids are root-local: the hangup and
        # recording writes address the exchange by (id, destination).
        assert mock_cm.call_manager.call_exchange_destination == "team:11"


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
            is_coordinator=True,
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            mock_utils.update_session_contacts = AsyncMock()
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
            is_coordinator=True,
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            mock_utils.update_session_contacts = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.set_details.assert_called_once()
        assert mock_cm.set_details.call_args.args[0]["is_coordinator"] is True

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
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            mock_utils.update_session_contacts = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.call_manager.set_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_assistant_update_queues_contact_update(self, mock_cm):
        """AssistantUpdateEvent syncs live contacts before later sends."""
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
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            mock_utils.update_session_contacts = AsyncMock()
            await EventHandler.handle_event(event, mock_cm)

            # Secret sync can stay queued, but contact identity must be current
            # before the next onboarding send observes ContactManager state.
            assert mock_utils.queue_operation.call_count == 1

            secrets_call = mock_utils.queue_operation.call_args_list[0]
            assert secrets_call[0][0] == mock_utils.sync_assistant_secrets

            mock_utils.update_session_contacts.assert_awaited_once()
            contacts_call = mock_utils.update_session_contacts.await_args
            assert contacts_call.args[0] == mock_cm
            assert contacts_call.args[1] == "New Assistant"
            assert contacts_call.args[2] == "Name"
            assert contacts_call.args[3] == "+15555559999"
            assert contacts_call.args[4] == "new_assistant@test.com"
            assert contacts_call.args[5] == "New Boss"
            assert contacts_call.args[6] == "Name"
            assert contacts_call.args[7] == "+15555558888"
            assert contacts_call.args[8] == "new_boss@test.com"

    @pytest.mark.asyncio
    async def test_assistant_update_syncs_boss_whatsapp_with_nullable_metadata(
        self,
        mock_cm,
    ):
        """Sparse assistant metadata must not block the boss WhatsApp sync."""
        mock_cm.set_details = MagicMock()
        mock_cm.get_call_config = MagicMock(return_value={})
        mock_cm.contact_manager.update_contact = MagicMock()

        event = AssistantUpdateEvent(
            api_key="test_key",
            medium="assistant_update",
            assistant_id="asst_123",
            user_id="user_456",
            assistant_first_name="T-W1N",
            assistant_surname=None,
            assistant_age="",
            assistant_nationality="",
            assistant_about="",
            assistant_number="",
            assistant_email="assistant@test.com",
            assistant_whatsapp_number="",
            user_first_name="Daniel",
            user_surname="Lenton",
            user_number="",
            user_email="dan@unify.ai",
            user_whatsapp_number="+4915237826557",
            voice_id=None,
            voice_provider=None,
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils.queue_operation",
            new_callable=AsyncMock,
        ):
            await EventHandler.handle_event(event, mock_cm)

        boss_call = next(
            call
            for call in mock_cm.contact_manager.update_contact.call_args_list
            if call.kwargs.get("contact_id") == 1
        )
        assert boss_call.kwargs["first_name"] == "Daniel"
        assert boss_call.kwargs["surname"] == "Lenton"
        assert boss_call.kwargs["email_address"] == "dan@unify.ai"
        assert boss_call.kwargs["whatsapp_number"] == "+4915237826557"

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
            "unify.conversation_manager.domains.event_handlers.managers_utils",
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
        from unify.common.context_registry import (
            PERSONAL_ROOT_IDENTITY,
            TEAM_CONTEXT_PREFIX,
            ContextRegistry,
        )
        from unify.session_details import SESSION_DETAILS

        SESSION_DETAILS.team_ids = [3, 7]
        SESSION_DETAILS.self_contact_id = 0
        SESSION_DETAILS.boss_contact_id = 1
        monkeypatch.delenv("SELF_CONTACT_ID", raising=False)
        monkeypatch.delenv("BOSS_CONTACT_ID", raising=False)
        monkeypatch.delenv("TEAM_SUMMARIES", raising=False)
        ContextRegistry._registry = {
            ("TaskScheduler", "Tasks", PERSONAL_ROOT_IDENTITY): "user456/123/Tasks",
            ("TaskScheduler", "Tasks", f"{TEAM_CONTEXT_PREFIX}3"): "Teams/3/Tasks",
            ("TaskScheduler", "Tasks", f"{TEAM_CONTEXT_PREFIX}7"): "Teams/7/Tasks",
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
            team_ids=[7, 11],
            team_summaries=[
                {
                    "team_id": 7,
                    "name": "Ops",
                    "description": "Operations workspace for customer support.",
                },
                {
                    "team_id": 11,
                    "name": "Repairs",
                    "description": "South-East repairs patch daily operations.",
                },
            ],
            self_contact_id=42,
            boss_contact_id=43,
        )

        with patch(
            "unify.conversation_manager.domains.event_handlers.managers_utils",
        ) as mock_utils:
            mock_utils.queue_operation = AsyncMock()
            try:
                await EventHandler.handle_event(event, mock_cm)

                assert SESSION_DETAILS.team_ids == [7, 11]
                assert SESSION_DETAILS.self_contact_id == 42
                assert SESSION_DETAILS.boss_contact_id == 43
                assert os.environ["SELF_CONTACT_ID"] == "42"
                assert os.environ["BOSS_CONTACT_ID"] == "43"
                assert (
                    "South-East repairs patch daily operations."
                    in os.environ["TEAM_SUMMARIES"]
                )
                assert mock_cm.team_ids == [7, 11]
                assert [summary.team_id for summary in mock_cm.team_summaries] == [
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
                        ("TaskScheduler", "Tasks", f"{TEAM_CONTEXT_PREFIX}7")
                    ]
                    == "Teams/7/Tasks"
                )
                assert (
                    "TaskScheduler",
                    "Tasks",
                    f"{TEAM_CONTEXT_PREFIX}3",
                ) not in ContextRegistry._registry
                assert (
                    "TaskScheduler",
                    "Tasks",
                    f"{TEAM_CONTEXT_PREFIX}11",
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
        from unify.conversation_manager.domains.managers_utils import (
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
            assistant_whatsapp_number="+15555557777",
            user_whatsapp_number="+4915237826557",
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
        assert call_0.kwargs["whatsapp_number"] == "+15555557777"

        # Check boss contact (ID 1) - "New Boss" splits to first="New", surname="Boss"
        call_1 = next(c for c in calls if c.kwargs.get("contact_id") == 1)
        assert call_1.kwargs["phone_number"] == "+15555558888"
        assert call_1.kwargs["email_address"] == "new_boss@test.com"
        assert call_1.kwargs["first_name"] == "New"
        assert call_1.kwargs["surname"] == "Boss"
        assert call_1.kwargs["whatsapp_number"] == "+4915237826557"

    @pytest.mark.asyncio
    async def test_update_session_contacts_handles_failure(self, mock_cm, caplog):
        """update_session_contacts logs errors when update_contact fails."""
        import logging

        from unify.conversation_manager.domains.managers_utils import (
            update_session_contacts,
        )

        mock_cm.contact_manager.update_contact = MagicMock(
            side_effect=Exception("Update failed"),
        )

        unity_logger = logging.getLogger("unify")
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

        from unify.conversation_manager.domains.managers_utils import (
            update_session_contacts,
        )

        mock_cm.contact_manager = None

        unity_logger = logging.getLogger("unify")
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
        from unify.conversation_manager.domains.event_handlers import (
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
            "unify.manager_registry.ManagerRegistry.get_instance",
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
            "unify.manager_registry.ManagerRegistry.get_instance",
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
            "unify.manager_registry.ManagerRegistry.get_instance",
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
            "unify.manager_registry.ManagerRegistry.get_instance",
            return_value=mock_cp,
        ):
            event = AssistantScreenShareStarted(reason="Screen share started")
            await EventHandler.handle_event(event, mock_cm)

        mock_cp.set_user_remote_control.assert_not_called()


# =============================================================================
# OpenSlowBrainTurn Event Tests
# =============================================================================


class TestOpenSlowBrainTurnEvent:
    """Tests for the OpenSlowBrainTurn event."""

    def test_open_slow_brain_turn_is_registered(self):
        """OpenSlowBrainTurn should have a handler in the registry."""
        assert OpenSlowBrainTurn in EventHandler._registry

    @pytest.mark.asyncio
    async def test_open_slow_brain_turn_pushes_notification(self, mock_cm):
        """OpenSlowBrainTurn handler should push the follow-on notification."""
        event = OpenSlowBrainTurn(
            origin_run_id="llmrun-000001",
            previous_tools=["act"],
        )
        await EventHandler.handle_event(event, mock_cm)

        notifications = mock_cm.notifications_bar.notifications
        assert any(
            OPEN_SLOW_BRAIN_TURN_NOTIFICATION in n.content for n in notifications
        )

    @pytest.mark.asyncio
    async def test_open_slow_brain_turn_triggers_llm_run(self, mock_cm):
        """OpenSlowBrainTurn handler should trigger an LLM run."""
        event = OpenSlowBrainTurn(
            origin_run_id="llmrun-000001",
            previous_tools=["act"],
        )
        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called_once_with(delay=0)


# =============================================================================
# FileSyncComplete Event Tests
# =============================================================================


class TestFileSyncCompleteEvent:
    """Tests for the FileSyncComplete event."""

    def test_file_sync_complete_is_registered(self):
        """FileSyncComplete should have a handler in the registry."""
        from unify.conversation_manager.events import FileSyncComplete

        assert FileSyncComplete in EventHandler._registry

    @pytest.mark.asyncio
    async def test_file_sync_complete_sets_flag(self, mock_cm):
        """FileSyncComplete handler should set cm.file_sync_complete to True."""
        from unify.conversation_manager.events import FileSyncComplete

        mock_cm.file_sync_complete = False
        event = FileSyncComplete()
        await EventHandler.handle_event(event, mock_cm)

        assert mock_cm.file_sync_complete is True

    @pytest.mark.asyncio
    async def test_file_sync_complete_triggers_llm_run(self, mock_cm):
        """FileSyncComplete handler should trigger an LLM run."""
        from unify.conversation_manager.events import FileSyncComplete

        mock_cm.file_sync_complete = False
        event = FileSyncComplete()
        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called_once_with(delay=0)

    @pytest.mark.asyncio
    async def test_file_sync_complete_handler_logs(self, mock_cm):
        """FileSyncComplete handler should log the sync completion."""
        from unify.conversation_manager.events import FileSyncComplete

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
        from unify.conversation_manager.events import AssistantDesktopReady

        assert AssistantDesktopReady in EventHandler._registry

    @pytest.mark.asyncio
    async def test_assistant_desktop_ready_sets_vm_ready_flag(self, mock_cm):
        """AssistantDesktopReady should set cm.vm_ready to True."""
        from unify.conversation_manager.events import AssistantDesktopReady

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
                "unify.conversation_manager.assistant_jobs.update_liveview_url",
            ),
            patch(
                "unify.conversation_manager.domains.managers_utils._start_file_sync",
                new_callable=AsyncMock,
            ),
            patch(
                "unify.session_details.SESSION_DETAILS",
            ) as mock_sd,
            patch(
                "unify.function_manager.primitives.runtime._vm_ready",
            ),
            patch(
                "unify.conversation_manager.domains.event_handlers._ensure_desktop_session",
                return_value=AsyncMock()(),
            ),
            patch(
                "unify.conversation_manager.domains.comms_utils.publish_assistant_desktop_ready",
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
        from unify.conversation_manager.events import AssistantDesktopReady

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
                "unify.conversation_manager.assistant_jobs.update_liveview_url",
            ),
            patch(
                "unify.conversation_manager.domains.managers_utils._start_file_sync",
                new_callable=AsyncMock,
            ),
            patch(
                "unify.session_details.SESSION_DETAILS",
            ) as mock_sd,
            patch(
                "unify.function_manager.primitives.runtime._vm_ready",
            ),
            patch(
                "unify.conversation_manager.domains.event_handlers._ensure_desktop_session",
                return_value=AsyncMock()(),
            ),
            patch(
                "unify.conversation_manager.domains.comms_utils.publish_assistant_desktop_ready",
                new_callable=AsyncMock,
            ),
        ):
            mock_sd.assistant.desktop_url = ""
            mock_sd.assistant.binding_id = "binding-84"
            await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called_with(delay=0)

    @pytest.mark.asyncio
    async def test_assistant_desktop_ready_relays_desktop_secret_when_present(
        self,
        mock_cm,
    ):
        """A desktop_secret on the event must reach the liveview PATCH and the
        Console SSE payload as ``liveview_password``."""
        from unify.conversation_manager.events import AssistantDesktopReady

        mock_cm.vm_ready = False
        mock_cm.assistant_id = "84"
        mock_cm.user_id = "test_user"
        event = AssistantDesktopReady(
            binding_id="binding-84",
            desktop_url="https://unity-pool-ubuntu-1.vm.unify.ai",
            vm_type="ubuntu",
            desktop_secret="binding-secret",  # pragma: allowlist secret
        )

        with (
            patch(
                "unify.conversation_manager.assistant_jobs.update_liveview_url",
            ) as mock_update_liveview_url,
            patch(
                "unify.conversation_manager.domains.managers_utils._start_file_sync",
                new_callable=AsyncMock,
            ),
            patch(
                "unify.session_details.SESSION_DETAILS",
            ) as mock_sd,
            patch(
                "unify.function_manager.primitives.runtime._vm_ready",
            ),
            patch(
                "unify.conversation_manager.domains.event_handlers._ensure_desktop_session",
                return_value=AsyncMock()(),
            ),
            patch(
                "unify.conversation_manager.domains.comms_utils.publish_assistant_desktop_ready",
                new_callable=AsyncMock,
            ) as mock_publish_desktop_ready,
        ):
            mock_sd.assistant.desktop_url = ""
            mock_sd.assistant.binding_id = "binding-84"
            await EventHandler.handle_event(event, mock_cm)

        assert mock_update_liveview_url.call_args.args[-1] == "binding-secret"
        mock_publish_desktop_ready.assert_awaited_once()
        _, publish_kwargs = mock_publish_desktop_ready.call_args
        assert publish_kwargs.get("liveview_password") == "binding-secret"

    @pytest.mark.asyncio
    async def test_assistant_desktop_ready_omits_liveview_password_without_secret(
        self,
        mock_cm,
    ):
        """No secret on the event (old binding) -- relay ``None``, not an
        empty string, so downstream layers treat it as absent."""
        from unify.conversation_manager.events import AssistantDesktopReady

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
                "unify.conversation_manager.assistant_jobs.update_liveview_url",
            ) as mock_update_liveview_url,
            patch(
                "unify.conversation_manager.domains.managers_utils._start_file_sync",
                new_callable=AsyncMock,
            ),
            patch(
                "unify.session_details.SESSION_DETAILS",
            ) as mock_sd,
            patch(
                "unify.function_manager.primitives.runtime._vm_ready",
            ),
            patch(
                "unify.conversation_manager.domains.event_handlers._ensure_desktop_session",
                return_value=AsyncMock()(),
            ),
            patch(
                "unify.conversation_manager.domains.comms_utils.publish_assistant_desktop_ready",
                new_callable=AsyncMock,
            ) as mock_publish_desktop_ready,
        ):
            mock_sd.assistant.desktop_url = ""
            mock_sd.assistant.binding_id = "binding-84"
            await EventHandler.handle_event(event, mock_cm)

        assert mock_update_liveview_url.call_args.args[-1] is None
        _, publish_kwargs = mock_publish_desktop_ready.call_args
        assert publish_kwargs.get("liveview_password") is None

    @pytest.mark.asyncio
    async def test_assistant_desktop_ready_ignores_missing_binding_when_session_has_one(
        self,
        mock_cm,
    ):
        """AssistantDesktopReady should fail closed on missing binding_id."""
        from unify.conversation_manager.events import AssistantDesktopReady

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
                "unify.conversation_manager.assistant_jobs.update_liveview_url",
            ) as mock_update_liveview_url,
            patch(
                "unify.conversation_manager.domains.managers_utils._start_file_sync",
                new_callable=AsyncMock,
            ) as mock_start_file_sync,
            patch(
                "unify.session_details.SESSION_DETAILS",
            ) as mock_sd,
            patch(
                "unify.function_manager.primitives.runtime._vm_ready",
            ) as mock_vm_ready,
            patch(
                "unify.conversation_manager.domains.event_handlers._ensure_desktop_session",
                return_value=AsyncMock()(),
            ) as mock_ensure_desktop_session,
            patch(
                "unify.conversation_manager.domains.comms_utils.publish_assistant_desktop_ready",
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
        from unify.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_infrastructure_state(
            vm_ready=False,
            file_sync_complete=False,
            has_desktop=True,
        )
        assert "vm_pending" in result
        assert "sync_pending" in result

    def test_vm_ready_sync_pending(self):
        """Only sync pending should produce one section."""
        from unify.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_infrastructure_state(
            vm_ready=True,
            file_sync_complete=False,
            has_desktop=True,
        )
        assert "vm_pending" not in result
        assert "sync_pending" in result

    def test_all_ready(self):
        """Both ready should return empty string."""
        from unify.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_infrastructure_state(
            vm_ready=True,
            file_sync_complete=True,
            has_desktop=True,
        )
        assert result == ""

    def test_no_desktop(self):
        """No desktop configured should return empty string regardless of flags."""
        from unify.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_infrastructure_state(
            vm_ready=False,
            file_sync_complete=False,
            has_desktop=False,
        )
        assert result == ""

    def test_vm_pending_mentions_computer_actions(self):
        """The VM pending section should mention computer actions are unavailable."""
        from unify.conversation_manager.domains.renderer import Renderer

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
        from unify.conversation_manager.domains.renderer import Renderer

        result = Renderer.render_infrastructure_state(
            vm_ready=True,
            file_sync_complete=False,
            has_desktop=True,
        )
        assert "Attachments/" in result
        assert "sync_pending" in result


# =============================================================================
# User Filesystem-Access consent handler
# =============================================================================


class TestUserFilesysAccessHandler:
    """The handler applies live consent changes to the ComputerPrimitives.

    Started → grant the desktop owner's link; stopped → revoke it (passing a
    conversation snapshot). Resolution goes through ManagerRegistry, so we patch
    it to a recording stub.
    """

    @pytest.mark.asyncio
    async def test_started_grants_for_user(self, mock_cm):
        cp = MagicMock()
        with patch(
            "unify.manager_registry.ManagerRegistry.get_instance",
            return_value=cp,
        ):
            await EventHandler.handle_event(
                UserFilesysAccessStarted(user_id="42"),
                mock_cm,
            )
        cp.grant_user_filesys_access.assert_called_once_with("42")
        cp.revoke_user_filesys_access.assert_not_called()

    @pytest.mark.asyncio
    async def test_stopped_revokes_with_conversation_context(self, mock_cm):
        cp = MagicMock()
        with patch(
            "unify.manager_registry.ManagerRegistry.get_instance",
            return_value=cp,
        ):
            await EventHandler.handle_event(
                UserFilesysAccessStopped(user_id="42"),
                mock_cm,
            )
        cp.grant_user_filesys_access.assert_not_called()
        cp.revoke_user_filesys_access.assert_called_once()
        args, kwargs = cp.revoke_user_filesys_access.call_args
        assert args[0] == "42"
        assert "conversation_context" in kwargs

    @pytest.mark.asyncio
    async def test_empty_user_id_is_ignored(self, mock_cm):
        cp = MagicMock()
        with patch(
            "unify.manager_registry.ManagerRegistry.get_instance",
            return_value=cp,
        ) as get_instance:
            await EventHandler.handle_event(
                UserFilesysAccessStarted(user_id=""),
                mock_cm,
            )
        get_instance.assert_not_called()
        cp.grant_user_filesys_access.assert_not_called()
        cp.revoke_user_filesys_access.assert_not_called()

    @pytest.mark.asyncio
    async def test_absent_primitive_does_not_crash(self, mock_cm):
        with patch(
            "unify.manager_registry.ManagerRegistry.get_instance",
            return_value=None,
        ):
            # Must return cleanly even when no ComputerPrimitives is registered.
            await EventHandler.handle_event(
                UserFilesysAccessStopped(user_id="42"),
                mock_cm,
            )
