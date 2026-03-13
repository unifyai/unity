"""
tests/conversation_manager/core/test_session_hydration.py
=========================================================

Tests for global thread hydration from EventBus on wakeup.

Verifies that hydrate_global_thread() correctly reconstructs the shared
global deque from persisted Comms events, making session boundaries
invisible to the brain. All tests are symbolic — the EventBus search
is mocked to return synthetic events.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.conversation_manager.domains.contact_index import (
    ContactIndex,
    EmailMessage,
    GuidanceMessage,
    Message,
    UnifyMessage,
)
from unity.conversation_manager.domains.managers_utils import hydrate_global_thread
from unity.conversation_manager.events import (
    FastBrainNotification,
    EmailReceived,
    EmailSent,
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    OutboundPhoneUtterance,
    PhoneCallNotAnswered,
    PhoneCallReceived,
    PhoneCallStarted,
    SMSReceived,
    SMSSent,
    UnifyMessageReceived,
    UnifyMessageSent,
)
from unity.conversation_manager.types import Medium

# =============================================================================
# Helpers
# =============================================================================

BASE_TIME = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
ALICE = {
    "contact_id": 2,
    "first_name": "Alice",
    "surname": "Smith",
    "phone_number": "+15555552222",
}
BOB = {
    "contact_id": 3,
    "first_name": "Bob",
    "surname": "Jones",
    "email_address": "bob@example.com",
}


def _make_bus_events(cm_events):
    """Convert CM events to bus events (as EventBus.search would return them).

    Returns events in descending timestamp order (newest first), matching
    the real EventBus.search() behavior.
    """
    bus_events = []
    for ev in cm_events:
        bus_event = ev.to_bus_event()
        # Simulate the email_id stripping done by publish_bus_events
        bus_event.payload.pop("email_id", None)
        bus_events.append(bus_event)
    # Descending order (newest first)
    bus_events.reverse()
    return bus_events


def _make_mock_cm():
    """Create a minimal mock CM with a real ContactIndex."""
    cm = MagicMock()
    cm.contact_index = ContactIndex()
    return cm


# =============================================================================
# SMS Hydration
# =============================================================================


class TestSMSHydration:

    @pytest.mark.asyncio
    async def test_sms_received_restored(self):
        """SMSReceived events restore as user messages in the SMS thread."""
        cm = _make_mock_cm()
        events = [
            SMSReceived(contact=ALICE, content="Hello!", timestamp=BASE_TIME),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        msgs = cm.contact_index.get_messages_for_contact(2, Medium.SMS_MESSAGE)
        assert len(msgs) == 1
        assert isinstance(msgs[0], Message)
        assert msgs[0].content == "Hello!"
        assert msgs[0].role == "user"

    @pytest.mark.asyncio
    async def test_sms_sent_restored(self):
        """SMSSent events restore as assistant messages."""
        cm = _make_mock_cm()
        events = [
            SMSSent(contact=ALICE, content="Got it!", timestamp=BASE_TIME),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        msgs = cm.contact_index.get_messages_for_contact(2, Medium.SMS_MESSAGE)
        assert len(msgs) == 1
        assert msgs[0].role == "assistant"
        assert msgs[0].name == "You"


# =============================================================================
# Email Hydration
# =============================================================================


class TestEmailHydration:

    @pytest.mark.asyncio
    async def test_email_received_restored(self):
        """EmailReceived events restore with subject, body, and recipients."""
        cm = _make_mock_cm()
        events = [
            EmailReceived(
                contact=BOB,
                subject="Meeting Notes",
                body="Here are the notes from today.",
                email_id="msg_123",
                to=["assistant@unify.ai"],
                cc=["alice@example.com"],
                timestamp=BASE_TIME,
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        msgs = cm.contact_index.get_messages_for_contact(3, Medium.EMAIL)
        assert len(msgs) == 1
        assert isinstance(msgs[0], EmailMessage)
        assert msgs[0].subject == "Meeting Notes"
        assert msgs[0].body == "Here are the notes from today."
        assert msgs[0].role == "user"

    @pytest.mark.asyncio
    async def test_email_sent_restored(self):
        """EmailSent events restore as assistant emails."""
        cm = _make_mock_cm()
        events = [
            EmailSent(
                contact=BOB,
                subject="Re: Meeting Notes",
                body="Thanks for sharing!",
                to=["bob@example.com"],
                timestamp=BASE_TIME,
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        msgs = cm.contact_index.get_messages_for_contact(3, Medium.EMAIL)
        assert len(msgs) == 1
        assert msgs[0].role == "assistant"
        assert msgs[0].name == "You"


# =============================================================================
# Voice / Call Hydration
# =============================================================================


class TestVoiceHydration:

    @pytest.mark.asyncio
    async def test_phone_utterances_restored(self):
        """Inbound and outbound phone utterances restore correctly."""
        cm = _make_mock_cm()
        events = [
            InboundPhoneUtterance(
                contact=ALICE,
                content="Hi there",
                timestamp=BASE_TIME,
            ),
            OutboundPhoneUtterance(
                contact=ALICE,
                content="Hello!",
                timestamp=BASE_TIME + timedelta(seconds=5),
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        msgs = cm.contact_index.get_messages_for_contact(2, Medium.PHONE_CALL)
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[0].content == "Hi there"
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "Hello!"

    @pytest.mark.asyncio
    async def test_unify_meet_utterances_restored(self):
        """Unify Meet utterances restore to the UNIFY_MEET medium."""
        cm = _make_mock_cm()
        events = [
            InboundUnifyMeetUtterance(
                contact=ALICE,
                content="Can you hear me?",
                timestamp=BASE_TIME,
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        msgs = cm.contact_index.get_messages_for_contact(2, Medium.UNIFY_MEET)
        assert len(msgs) == 1
        assert msgs[0].content == "Can you hear me?"

    @pytest.mark.asyncio
    async def test_call_guidance_restored(self):
        """FastBrainNotification events restore as guidance-role messages."""
        cm = _make_mock_cm()
        events = [
            FastBrainNotification(
                contact=ALICE,
                content="Mention the 3pm meeting",
                timestamp=BASE_TIME,
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        msgs = cm.contact_index.get_messages_for_contact(2, Medium.PHONE_CALL)
        assert len(msgs) == 1
        assert isinstance(msgs[0], GuidanceMessage)
        assert msgs[0].content == "Mention the 3pm meeting"

    @pytest.mark.asyncio
    async def test_call_lifecycle_events_restored(self):
        """Call lifecycle events (received, started, not answered) restore correctly."""
        cm = _make_mock_cm()
        events = [
            PhoneCallReceived(
                contact=ALICE,
                timestamp=BASE_TIME,
            ),
            PhoneCallStarted(
                contact=ALICE,
                timestamp=BASE_TIME + timedelta(seconds=2),
            ),
            PhoneCallNotAnswered(
                contact=ALICE,
                reason="busy",
                timestamp=BASE_TIME + timedelta(seconds=5),
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        msgs = cm.contact_index.get_messages_for_contact(2, Medium.PHONE_CALL)
        assert len(msgs) == 3
        assert "<Receiving Call...>" in msgs[0].content
        assert "<Call Started>" in msgs[1].content
        assert "was busy" in msgs[2].content


# =============================================================================
# Unify Message Hydration
# =============================================================================


class TestUnifyMessageHydration:

    @pytest.mark.asyncio
    async def test_unify_messages_restored(self):
        """UnifyMessage events restore with content and attachments."""
        cm = _make_mock_cm()
        events = [
            UnifyMessageReceived(
                contact=ALICE,
                content="Check this file",
                attachments=[{"id": "a1", "filename": "report.pdf"}],
                timestamp=BASE_TIME,
            ),
            UnifyMessageSent(
                contact=ALICE,
                content="Got it, reviewing now",
                timestamp=BASE_TIME + timedelta(seconds=10),
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        msgs = cm.contact_index.get_messages_for_contact(2, Medium.UNIFY_MESSAGE)
        assert len(msgs) == 2
        assert isinstance(msgs[0], UnifyMessage)
        assert msgs[0].role == "user"
        assert msgs[0].content == "Check this file"
        assert msgs[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_hydration_collects_unify_attachment_metadata(self):
        """Inbound UnifyMessage attachments are collected for deferred download."""
        cm = _make_mock_cm()
        att = [
            {
                "id": "att-1",
                "filename": "report.pdf",
                "gs_url": "gs://bucket/att-1_report.pdf",
            },
        ]
        events = [
            UnifyMessageReceived(
                contact=ALICE,
                content="See attached",
                attachments=att,
                timestamp=BASE_TIME,
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        pending = cm._pending_hydration_attachments
        assert len(pending) == 1
        kind, atts, extra = pending[0]
        assert kind == "unify"
        assert atts == att
        assert extra == {}

    @pytest.mark.asyncio
    async def test_hydration_collects_email_attachment_metadata(self):
        """Inbound EmailReceived attachments are collected with email-specific metadata.

        Note: email_id is stripped from bus events during publish_bus_events(),
        so gmail_message_id will be empty after hydration. The important thing
        is that the attachment metadata itself is collected.
        """
        cm = _make_mock_cm()
        att = [{"id": "att-email-1", "filename": "invoice.pdf"}]
        events = [
            EmailReceived(
                contact=BOB,
                subject="Invoice",
                body="Please review",
                attachments=att,
                email_id="gmail-msg-123",
                timestamp=BASE_TIME,
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        pending = cm._pending_hydration_attachments
        assert len(pending) == 1
        kind, atts, extra = pending[0]
        assert kind == "email"
        assert atts == att
        # email_id is stripped from bus events by publish_bus_events, so
        # gmail_message_id is empty after hydration — this is expected
        assert "gmail_message_id" in extra

    @pytest.mark.asyncio
    async def test_hydration_skips_outbound_attachments(self):
        """Outbound messages (Sent events) don't trigger attachment collection."""
        cm = _make_mock_cm()
        events = [
            UnifyMessageSent(
                contact=ALICE,
                content="Here's the result",
                attachments=[{"id": "out-1", "filename": "output.xlsx"}],
                timestamp=BASE_TIME,
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        pending = cm._pending_hydration_attachments
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_hydration_collects_multiple_attachments_across_messages(self):
        """Multiple inbound messages with attachments are all collected."""
        cm = _make_mock_cm()
        events = [
            UnifyMessageReceived(
                contact=ALICE,
                content="File 1",
                attachments=[{"id": "a1", "filename": "f1.pdf"}],
                timestamp=BASE_TIME,
            ),
            UnifyMessageReceived(
                contact=ALICE,
                content="File 2",
                attachments=[
                    {"id": "a2", "filename": "f2.pdf"},
                    {"id": "a3", "filename": "f3.xlsx"},
                ],
                timestamp=BASE_TIME + timedelta(minutes=5),
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        pending = cm._pending_hydration_attachments
        assert len(pending) == 2
        assert len(pending[0][1]) == 1
        assert len(pending[1][1]) == 2


# =============================================================================
# Cross-Cutting Concerns
# =============================================================================


class TestHydrationCrossCutting:

    @pytest.mark.asyncio
    async def test_chronological_order_preserved(self):
        """Messages from multiple contacts and mediums maintain chronological order."""
        cm = _make_mock_cm()
        events = [
            SMSReceived(contact=ALICE, content="msg_1", timestamp=BASE_TIME),
            SMSSent(
                contact=ALICE,
                content="msg_2",
                timestamp=BASE_TIME + timedelta(minutes=1),
            ),
            SMSReceived(
                contact=BOB,
                content="msg_3",
                timestamp=BASE_TIME + timedelta(minutes=2),
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        assert len(cm.contact_index.global_thread) == 3
        contents = [e.message.content for e in cm.contact_index.global_thread]
        assert contents == ["msg_1", "msg_2", "msg_3"]

    @pytest.mark.asyncio
    async def test_multi_contact_multi_medium(self):
        """Events across contacts and mediums all hydrate correctly."""
        cm = _make_mock_cm()
        events = [
            SMSReceived(contact=ALICE, content="sms from alice", timestamp=BASE_TIME),
            EmailReceived(
                contact=BOB,
                subject="Hi",
                body="Email body",
                timestamp=BASE_TIME + timedelta(minutes=1),
            ),
            InboundPhoneUtterance(
                contact=ALICE,
                content="voice from alice",
                timestamp=BASE_TIME + timedelta(minutes=2),
            ),
        ]

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=_make_bus_events(events))
            await hydrate_global_thread(cm)

        # Alice has SMS and phone
        alice_sms = cm.contact_index.get_messages_for_contact(2, Medium.SMS_MESSAGE)
        alice_phone = cm.contact_index.get_messages_for_contact(2, Medium.PHONE_CALL)
        assert len(alice_sms) == 1
        assert len(alice_phone) == 1

        # Bob has email
        bob_email = cm.contact_index.get_messages_for_contact(3, Medium.EMAIL)
        assert len(bob_email) == 1

        # Active contacts derived from global thread
        active = cm.contact_index.get_active_contact_ids()
        assert active == {2, 3}

    @pytest.mark.asyncio
    async def test_non_message_events_skipped(self):
        """Events that don't produce messages (e.g. ActorResult) are skipped."""
        from unity.events.event_bus import Event as BusEvent

        cm = _make_mock_cm()
        # Simulate a mix: one real message + one non-message event
        sms = SMSReceived(contact=ALICE, content="Hello", timestamp=BASE_TIME)
        bus_events = _make_bus_events([sms])

        # Insert a non-message bus event
        non_msg = BusEvent(
            type="Comms",
            payload_cls="ActorResult",
            payload={"handle_id": 1, "success": True, "result": "done"},
            timestamp=BASE_TIME.isoformat(),
        )
        bus_events.insert(0, non_msg)

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=bus_events)
            await hydrate_global_thread(cm)

        # Only the SMS should be in the deque
        assert len(cm.contact_index.global_thread) == 1

    @pytest.mark.asyncio
    async def test_empty_bus_is_noop(self):
        """No events in EventBus means empty deque (clean start)."""
        cm = _make_mock_cm()

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=[])
            await hydrate_global_thread(cm)

        assert len(cm.contact_index.global_thread) == 0

    @pytest.mark.asyncio
    async def test_missing_contact_id_skipped(self):
        """Events with no contact_id in the payload are skipped gracefully."""
        cm = _make_mock_cm()
        event = SMSReceived(contact={}, content="no id", timestamp=BASE_TIME)
        bus_events = _make_bus_events([event])

        with patch(
            "unity.conversation_manager.domains.managers_utils.EVENT_BUS",
        ) as mock_bus:
            mock_bus.search = AsyncMock(return_value=bus_events)
            await hydrate_global_thread(cm)

        assert len(cm.contact_index.global_thread) == 0
