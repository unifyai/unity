"""
tests/conversation_manager/core/test_renderer.py
================================================

Unit tests for the Renderer class in `domains/renderer.py`.

These are symbolic tests that verify rendering logic without invoking the LLM.
"""

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from unity.conversation_manager.domains.contact_index import (
    ContactIndex,
    EmailMessage,
    Message,
    UnifyMessage,
)
from unity.conversation_manager.domains.notifications import (
    NotificationBar,
    Notification,
)
from unity.conversation_manager.domains.renderer import (
    Renderer,
    SnapshotState,
    MessageElement,
    NotificationElement,
    ActionElement,
    compute_snapshot_diff,
    _get_assistant_email_role,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def renderer():
    """Create a Renderer instance."""
    return Renderer()


@pytest.fixture
def sample_received_email():
    """Create a sample received email where assistant is in To."""
    return EmailMessage(
        name="Alice Smith",
        subject="Project Update",
        body="Here's the latest update on the project.",
        email_id="CAKx7fQ_test@mail.gmail.com",
        timestamp=datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc),
        role="user",
        attachments=[],
        to=["assistant@unify.ai"],
        cc=["bob@example.com"],
        bcc=[],
        contact_role="sender",
    )


@pytest.fixture
def sample_sent_email():
    """Create a sample sent email from the assistant."""
    return EmailMessage(
        name="You",
        subject="Re: Project Update",
        body="Thanks for the update!",
        email_id="CAKx7fQ_test@mail.gmail.com",
        timestamp=datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc),
        role="assistant",
        attachments=[],
        to=["alice@example.com"],
        cc=["bob@example.com"],
        bcc=[],
        contact_role="to",
    )


# =============================================================================
# Tests for _get_assistant_email_role
# =============================================================================


class TestGetAssistantEmailRole:
    """Tests for the _get_assistant_email_role helper function."""

    def test_assistant_is_direct_recipient_to(self):
        """When assistant's email is in To field, returns 'direct recipient'."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="user",
            to=["assistant@unify.ai", "other@example.com"],
            cc=[],
            bcc=[],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            result = _get_assistant_email_role(email)
            assert result == "You were a direct recipient (To)"

    def test_assistant_is_cc_recipient(self):
        """When assistant's email is in Cc field, returns 'CC'd'."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="user",
            to=["bob@example.com"],
            cc=["assistant@unify.ai"],
            bcc=[],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            result = _get_assistant_email_role(email)
            assert result == "You were CC'd"

    def test_assistant_is_bcc_recipient(self):
        """When assistant's email is in Bcc field, returns 'BCC'd'."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="user",
            to=["bob@example.com"],
            cc=[],
            bcc=["assistant@unify.ai"],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            result = _get_assistant_email_role(email)
            assert result == "You were BCC'd"

    def test_assistant_sent_email(self):
        """When assistant sent the email (role=assistant), returns 'sent'."""
        email = EmailMessage(
            name="You",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="assistant",
            to=["alice@example.com"],
            cc=[],
            bcc=[],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            result = _get_assistant_email_role(email)
            assert result == "You sent this email"

    def test_assistant_not_in_email(self):
        """When assistant's email is not in any field, returns None."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="user",
            to=["bob@example.com"],
            cc=["charlie@example.com"],
            bcc=[],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            result = _get_assistant_email_role(email)
            assert result is None

    def test_case_insensitive_email_matching(self):
        """Email matching should be case-insensitive."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="user",
            to=["ASSISTANT@UNIFY.AI"],  # Uppercase
            cc=[],
            bcc=[],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"  # Lowercase
            result = _get_assistant_email_role(email)
            assert result == "You were a direct recipient (To)"

    def test_no_assistant_email_configured(self):
        """When assistant email is not configured, returns None."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="user",
            to=["assistant@unify.ai"],
            cc=[],
            bcc=[],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = None
            result = _get_assistant_email_role(email)
            assert result is None


# =============================================================================
# Tests for Renderer.render_message with email assistant role
# =============================================================================


class TestRendererEmailAssistantRole:
    """Tests for email rendering with assistant role context."""

    def test_render_email_shows_assistant_role_when_direct_recipient(self, renderer):
        """Rendered email includes '[Your role: ...]' when assistant is To recipient."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Important Update",
            body="Please review this.",
            email_id="test123@mail.gmail.com",
            timestamp=datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc),
            role="user",
            to=["assistant@unify.ai"],
            cc=[],
            bcc=[],
            contact_role="sender",
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            # Use a timestamp before the message to mark it as NEW
            last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
            result = renderer.render_message(email, last_snapshot)

            # Should contain assistant role line
            assert "[Your role: You were a direct recipient (To)]" in result
            # Should also contain contact role line
            assert "[Context: This contact SENT this email]" in result

    def test_render_email_shows_assistant_role_when_sender(self, renderer):
        """Rendered email includes '[Your role: You sent this email]' for outgoing."""
        email = EmailMessage(
            name="You",
            subject="Re: Important Update",
            body="Got it, thanks!",
            email_id="test123@mail.gmail.com",
            timestamp=datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc),
            role="assistant",
            to=["alice@example.com"],
            cc=[],
            bcc=[],
            contact_role="to",
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
            result = renderer.render_message(email, last_snapshot)

            # Should contain assistant role line for sent email
            assert "[Your role: You sent this email]" in result

    def test_render_email_no_assistant_role_when_not_involved(self, renderer):
        """Rendered email does not include assistant role when not in email."""
        email = EmailMessage(
            name="Alice Smith",
            subject="FYI",
            body="Forwarding this for reference.",
            email_id="test456@mail.gmail.com",
            timestamp=datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc),
            role="user",
            to=["bob@example.com"],  # Not assistant
            cc=["charlie@example.com"],  # Not assistant
            bcc=[],
            contact_role="sender",
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
            result = renderer.render_message(email, last_snapshot)

            # Should NOT contain assistant role line
            assert "[Your role:" not in result
            # But should still contain contact role line
            assert "[Context: This contact SENT this email]" in result


# =============================================================================
# Tests for SMS/Simple Message Rendering
# =============================================================================


class TestRendererSimpleMessage:
    """Tests for simple Message rendering (SMS, phone call utterances)."""

    def test_render_incoming_sms_shows_contact_name(self, renderer):
        """Incoming SMS shows contact's name."""
        message = Message(
            name="Alice Smith",
            content="Hey, can you call me back?",
            timestamp=datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc),
            role="user",
        )
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
        result = renderer.render_message(message, last_snapshot)

        assert "[Alice Smith @" in result
        assert "Hey, can you call me back?" in result
        assert "**NEW**" in result  # Message is newer than last_snapshot

    def test_render_outgoing_sms_shows_you(self, renderer):
        """Outgoing SMS shows 'You' as the sender."""
        message = Message(
            name="You",
            content="Sure, I'll call you in 5 minutes.",
            timestamp=datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc),
            role="assistant",
        )
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
        result = renderer.render_message(message, last_snapshot)

        assert "[You @" in result
        assert "Sure, I'll call you in 5 minutes." in result

    def test_render_old_message_no_new_marker(self, renderer):
        """Messages older than last_snapshot don't have **NEW** marker."""
        message = Message(
            name="Alice Smith",
            content="Old message",
            timestamp=datetime(2025, 6, 13, 10, 0, 0, tzinfo=timezone.utc),
            role="user",
        )
        # last_snapshot is AFTER the message timestamp
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
        result = renderer.render_message(message, last_snapshot)

        assert "**NEW**" not in result
        assert "[Alice Smith @" in result


# =============================================================================
# Tests for UnifyMessage Rendering
# =============================================================================


class TestRendererUnifyMessage:
    """Tests for UnifyMessage rendering (Unify console chat)."""

    def test_render_incoming_unify_message_shows_contact_name(self, renderer):
        """Incoming UnifyMessage shows contact's name."""
        message = UnifyMessage(
            name="Boss",
            content="Please send the report to Alice.",
            timestamp=datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc),
            role="user",
            attachments=[],
        )
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
        result = renderer.render_message(message, last_snapshot)

        assert "[Boss @" in result
        assert "Please send the report to Alice." in result
        assert "**NEW**" in result

    def test_render_outgoing_unify_message_shows_you(self, renderer):
        """Outgoing UnifyMessage shows 'You' as the sender."""
        message = UnifyMessage(
            name="You",
            content="Done, I've sent the report.",
            timestamp=datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc),
            role="assistant",
            attachments=[],
        )
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
        result = renderer.render_message(message, last_snapshot)

        assert "[You @" in result
        assert "Done, I've sent the report." in result

    def test_render_incoming_unify_message_with_attachments(self, renderer):
        """Incoming UnifyMessage attachments show as auto-downloaded."""
        message = UnifyMessage(
            name="Boss",
            content="Here's the document.",
            timestamp=datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc),
            role="user",
            attachments=["report.pdf", "data.xlsx"],
        )
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
        result = renderer.render_message(message, last_snapshot)

        assert "Here's the document." in result
        assert "[Attachments:" in result
        assert "report.pdf (auto-downloaded to Downloads/report.pdf)" in result
        assert "data.xlsx (auto-downloaded to Downloads/data.xlsx)" in result

    def test_render_outgoing_unify_message_with_attachments(self, renderer):
        """Outgoing UnifyMessage attachments show as 'attached'."""
        message = UnifyMessage(
            name="You",
            content="Here's the analysis.",
            timestamp=datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc),
            role="assistant",
            attachments=["analysis.pdf"],
        )
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
        result = renderer.render_message(message, last_snapshot)

        assert "Here's the analysis." in result
        assert "[Attachments:" in result
        assert "analysis.pdf (attached)" in result
        # Should NOT say "auto-downloaded"
        assert "auto-downloaded" not in result


# =============================================================================
# Tests for SnapshotState and Incremental Diff
# =============================================================================


class TestSnapshotState:
    """Tests for SnapshotState identity tracking."""

    def test_message_ids_returns_identity_tuples(self):
        """message_ids() returns set of (contact_id, thread, index, timestamp) tuples."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc)

        snapshot = SnapshotState(
            full_render="<test>",
            messages=[
                MessageElement(
                    contact_id=1,
                    thread_name="global",
                    index_in_thread=0,
                    timestamp=ts1,
                    rendered="[User @ ...]: Hello",
                ),
                MessageElement(
                    contact_id=1,
                    thread_name="global",
                    index_in_thread=1,
                    timestamp=ts2,
                    rendered="[You @ ...]: Hi there",
                ),
            ],
        )

        ids = snapshot.message_ids()
        assert len(ids) == 2
        assert (1, "global", 0, ts1) in ids
        assert (1, "global", 1, ts2) in ids

    def test_notification_ids_returns_identity_tuples(self):
        """notification_ids() returns set of (timestamp, content_hash, pinned) tuples."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)

        snapshot = SnapshotState(
            full_render="<test>",
            notifications=[
                NotificationElement(
                    timestamp=ts1,
                    content_hash=hash("Task completed"),
                    pinned=False,
                    rendered="[Task Notification @ ...] Task completed",
                ),
            ],
        )

        ids = snapshot.notification_ids()
        assert len(ids) == 1
        assert (ts1, hash("Task completed"), False) in ids

    def test_action_states_returns_handle_to_state_dict(self):
        """action_states() returns dict mapping handle_id to (status, history_count)."""
        snapshot = SnapshotState(
            full_render="<test>",
            actions=[
                ActionElement(
                    handle_id=0,
                    query="search contacts",
                    status="executing",
                    history_count=2,
                    rendered="<action id='0'>...",
                ),
                ActionElement(
                    handle_id=1,
                    query="send email",
                    status="paused",
                    history_count=0,
                    rendered="<action id='1'>...",
                ),
            ],
        )

        states = snapshot.action_states()
        assert states[0] == ("executing", 2)
        assert states[1] == ("paused", 0)


class TestComputeSnapshotDiff:
    """Tests for compute_snapshot_diff incremental diff computation."""

    def test_diff_returns_full_render_when_old_is_none(self):
        """When old_snapshot is None, returns full new snapshot."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        new_snapshot = SnapshotState(
            full_render="<full_state>content</full_state>",
            messages=[
                MessageElement(
                    contact_id=1,
                    thread_name="global",
                    index_in_thread=0,
                    timestamp=ts1,
                    rendered="[User @ ...]: Hello",
                ),
            ],
        )

        diff = compute_snapshot_diff(None, new_snapshot)
        assert diff == "<full_state>content</full_state>"

    def test_diff_returns_empty_when_nothing_changed(self):
        """When snapshots are identical, returns empty string."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)

        old_snapshot = SnapshotState(
            full_render="<state>same</state>",
            messages=[
                MessageElement(
                    contact_id=1,
                    thread_name="global",
                    index_in_thread=0,
                    timestamp=ts1,
                    rendered="[User @ ...]: Hello",
                ),
            ],
        )
        new_snapshot = SnapshotState(
            full_render="<state>same</state>",
            messages=[
                MessageElement(
                    contact_id=1,
                    thread_name="global",
                    index_in_thread=0,
                    timestamp=ts1,
                    rendered="[User @ ...]: Hello",
                ),
            ],
        )

        diff = compute_snapshot_diff(old_snapshot, new_snapshot)
        assert diff == ""

    def test_diff_includes_new_messages(self):
        """New messages are included in <new_messages> section."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc)

        old_snapshot = SnapshotState(
            full_render="<state>old</state>",
            messages=[
                MessageElement(
                    contact_id=1,
                    thread_name="global",
                    index_in_thread=0,
                    timestamp=ts1,
                    rendered="[User @ ...]: Hello",
                ),
            ],
        )
        new_snapshot = SnapshotState(
            full_render="<state>new</state>",
            messages=[
                MessageElement(
                    contact_id=1,
                    thread_name="global",
                    index_in_thread=0,
                    timestamp=ts1,
                    rendered="[User @ ...]: Hello",
                ),
                MessageElement(
                    contact_id=1,
                    thread_name="global",
                    index_in_thread=1,
                    timestamp=ts2,
                    rendered="[User @ ...]: Please help me",
                ),
            ],
        )

        diff = compute_snapshot_diff(old_snapshot, new_snapshot)
        assert "<new_messages>" in diff
        assert "[User @ ...]: Please help me" in diff
        assert "[User @ ...]: Hello" not in diff  # Old message not in diff

    def test_diff_includes_new_notifications(self):
        """New notifications are included in <new_notifications> section."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc)

        old_snapshot = SnapshotState(
            full_render="<state>old</state>",
            notifications=[
                NotificationElement(
                    timestamp=ts1,
                    content_hash=hash("Task started"),
                    pinned=False,
                    rendered="[Task Notification] Task started",
                ),
            ],
        )
        new_snapshot = SnapshotState(
            full_render="<state>new</state>",
            notifications=[
                NotificationElement(
                    timestamp=ts1,
                    content_hash=hash("Task started"),
                    pinned=False,
                    rendered="[Task Notification] Task started",
                ),
                NotificationElement(
                    timestamp=ts2,
                    content_hash=hash("Task completed"),
                    pinned=False,
                    rendered="[Task Notification] Task completed",
                ),
            ],
        )

        diff = compute_snapshot_diff(old_snapshot, new_snapshot)
        assert "<new_notifications>" in diff
        assert "Task completed" in diff
        assert "Task started" not in diff  # Old notification not in diff

    def test_diff_includes_action_state_changes(self):
        """Action state changes are included in <action_updates> section."""
        old_snapshot = SnapshotState(
            full_render="<state>old</state>",
            actions=[
                ActionElement(
                    handle_id=0,
                    query="search contacts",
                    status="executing",
                    history_count=0,
                    rendered="<action id='0' status='executing'>...",
                ),
            ],
        )
        new_snapshot = SnapshotState(
            full_render="<state>new</state>",
            actions=[
                ActionElement(
                    handle_id=0,
                    query="search contacts",
                    status="executing",
                    history_count=1,  # History count changed (new event)
                    rendered="<action id='0' status='executing'>new history event...",
                ),
            ],
        )

        diff = compute_snapshot_diff(old_snapshot, new_snapshot)
        assert "<action_updates>" in diff
        assert "new history event" in diff

    def test_diff_includes_new_actions(self):
        """New actions (not in old snapshot) are included in diff."""
        old_snapshot = SnapshotState(
            full_render="<state>old</state>",
            actions=[],
        )
        new_snapshot = SnapshotState(
            full_render="<state>new</state>",
            actions=[
                ActionElement(
                    handle_id=0,
                    query="search contacts",
                    status="executing",
                    history_count=0,
                    rendered="<action id='0'>search contacts...",
                ),
            ],
        )

        diff = compute_snapshot_diff(old_snapshot, new_snapshot)
        assert "<action_updates>" in diff
        assert "search contacts" in diff

    def test_diff_tracks_notification_pinned_state_change(self):
        """Notification pinned state change is detected as a new notification."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)

        old_snapshot = SnapshotState(
            full_render="<state>old</state>",
            notifications=[
                NotificationElement(
                    timestamp=ts1,
                    content_hash=hash("Important reminder"),
                    pinned=False,  # Not pinned
                    rendered="[Notification] Important reminder",
                ),
            ],
        )
        new_snapshot = SnapshotState(
            full_render="<state>new</state>",
            notifications=[
                NotificationElement(
                    timestamp=ts1,
                    content_hash=hash("Important reminder"),
                    pinned=True,  # Now pinned
                    rendered="[PINNED][Notification] Important reminder",
                ),
            ],
        )

        diff = compute_snapshot_diff(old_snapshot, new_snapshot)
        # The pinned=True version has a different identity tuple
        assert "<new_notifications>" in diff
        assert "[PINNED]" in diff


class TestRenderStateWithTracking:
    """Tests for render_state_with_tracking method."""

    @pytest.fixture
    def contact_index(self):
        """Create a ContactIndex with a conversation."""
        ci = ContactIndex()
        ci._fallback_contacts[1] = {
            "contact_id": 1,
            "first_name": "Alice",
            "surname": "Smith",
        }
        return ci

    @pytest.fixture
    def notification_bar(self):
        """Create a NotificationBar."""
        return NotificationBar()

    def test_returns_snapshot_state_with_full_render(
        self,
        renderer,
        contact_index,
        notification_bar,
    ):
        """render_state_with_tracking returns SnapshotState with full_render."""
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)

        result = renderer.render_state_with_tracking(
            contact_index,
            notification_bar,
            in_flight_actions={},
            last_snapshot=last_snapshot,
        )

        assert isinstance(result, SnapshotState)
        assert result.full_render is not None
        assert "<notifications>" in result.full_render
        assert "<in_flight_actions>" in result.full_render
        assert "<active_conversations>" in result.full_render

    def test_tracks_messages_in_conversation(
        self,
        renderer,
        contact_index,
        notification_bar,
    ):
        """Messages in conversations are tracked with identity."""
        from unity.conversation_manager.types import Medium

        # Add a message to the conversation
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        contact_index.push_message(
            contact_id=1,
            sender_name="Alice",
            thread_name=Medium.SMS_MESSAGE,
            message_content="Hello there!",
            timestamp=ts1,
            role="user",
        )

        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)

        result = renderer.render_state_with_tracking(
            contact_index,
            notification_bar,
            in_flight_actions={},
            last_snapshot=last_snapshot,
        )

        # Should have tracked the message
        assert len(result.messages) >= 1
        msg = next(m for m in result.messages if "Hello there!" in m.rendered)
        assert msg.contact_id == 1
        assert msg.timestamp == ts1

    def test_tracks_notifications(self, renderer, contact_index, notification_bar):
        """Notifications are tracked with identity."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        notification_bar.notifications.append(
            Notification(
                type="task",
                content="Action completed successfully",
                timestamp=ts1,
                pinned=False,
            ),
        )

        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)

        result = renderer.render_state_with_tracking(
            contact_index,
            notification_bar,
            in_flight_actions={},
            last_snapshot=last_snapshot,
        )

        assert len(result.notifications) == 1
        notif = result.notifications[0]
        assert notif.timestamp == ts1
        assert notif.pinned is False
        assert "Action completed" in notif.rendered

    def test_tracks_in_flight_actions(self, renderer, contact_index, notification_bar):
        """In-flight actions are tracked with identity."""
        mock_handle = MagicMock()
        mock_handle._pause_event = MagicMock()
        mock_handle._pause_event.is_set.return_value = True  # Not paused

        in_flight_actions = {
            0: {
                "handle": mock_handle,
                "query": "Search for Alice's email",
                "handle_actions": [
                    {"action_name": "interject_0", "query": "also check phone"},
                ],
            },
        }

        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)

        result = renderer.render_state_with_tracking(
            contact_index,
            notification_bar,
            in_flight_actions=in_flight_actions,
            last_snapshot=last_snapshot,
        )

        assert len(result.actions) == 1
        action = result.actions[0]
        assert action.handle_id == 0
        assert action.query == "Search for Alice's email"
        assert action.status == "executing"
        assert action.history_count == 1
