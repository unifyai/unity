"""
tests/test_conversation_manager/test_utils.py
=============================================

Unit tests for ConversationManager utility functions and helpers.

These are pure unit tests that don't require Redis, LLM calls, or async infrastructure.
They test the building blocks used by the conversation manager.
"""

from datetime import datetime, timedelta

import pytest

from unity.conversation_manager.task_actions import (
    OPERATION_MAP,
    STEERING_OPERATIONS,
    ParsedActionName,
    build_action_name,
    derive_short_name,
    get_steering_operation,
    is_dynamic_action,
    iter_available_actions_for_task,
    parse_action_name,
    safe_call_id_suffix,
)
from unity.conversation_manager.domains.notifications import (
    Notification,
    NotificationBar,
)
from unity.conversation_manager.domains.renderer import Renderer
from unity.conversation_manager.domains.contact_index import (
    Contact,
    Message,
    EmailMessage,
)


# =============================================================================
# task_actions.py tests
# =============================================================================


class TestDeriveShortName:
    """Tests for derive_short_name function."""

    def test_basic_query(self):
        """Simple query extracts first 4 words."""
        result = derive_short_name("List all contacts in the database")
        assert result == "list_all_contacts_in"

    def test_removes_punctuation(self):
        """Punctuation is stripped from the query."""
        result = derive_short_name("What's the weather?")
        assert result == "whats_the_weather"

    def test_lowercases_words(self):
        """All words are lowercased."""
        result = derive_short_name("Find IMPORTANT Documents NOW")
        assert result == "find_important_documents_now"

    def test_max_words_limit(self):
        """Respects max_words parameter."""
        result = derive_short_name("one two three four five six", max_words=2)
        assert result == "one_two"

    def test_empty_query(self):
        """Empty query returns 'task'."""
        result = derive_short_name("")
        assert result == "task"

    def test_only_punctuation(self):
        """Query with only punctuation returns 'task'."""
        result = derive_short_name("!@#$%^&*()")
        assert result == "task"

    def test_collapses_double_underscores(self):
        """Double underscores in result are collapsed to single."""
        # This could happen if a word boundary creates __ somehow
        # The function sanitizes this
        result = derive_short_name("test  query")  # double space
        assert "__" not in result

    def test_numbers_preserved(self):
        """Numbers in query are preserved."""
        result = derive_short_name("Send 100 emails to 50 people")
        assert result == "send_100_emails_to"


class TestSafeCallIdSuffix:
    """Tests for safe_call_id_suffix function."""

    def test_empty_call_id(self):
        """Empty call_id returns '0'."""
        assert safe_call_id_suffix("") == "0"

    def test_none_call_id(self):
        """None call_id returns '0'."""
        assert safe_call_id_suffix(None) == "0"

    def test_replaces_dashes(self):
        """Dashes are replaced with underscores."""
        result = safe_call_id_suffix("abc-def-ghi")
        assert "-" not in result
        assert "_" in result

    def test_takes_last_8_chars(self):
        """Only last 8 characters are returned."""
        result = safe_call_id_suffix("0123456789abcdef")
        assert result == "89abcdef"
        assert len(result) == 8

    def test_short_call_id(self):
        """Short call_id is returned as-is (with dash replacement)."""
        result = safe_call_id_suffix("abc")
        assert result == "abc"

    def test_collapses_double_underscores(self):
        """Double underscores are collapsed."""
        result = safe_call_id_suffix("test--id")
        assert "__" not in result


class TestBuildActionName:
    """Tests for build_action_name function."""

    def test_basic_action_name(self):
        """Builds basic action name with __ delimiter."""
        result = build_action_name("ask", "list_contacts", 0)
        assert result == "ask_list_contacts__0"

    def test_with_call_id_suffix(self):
        """Includes call_id_suffix with __ delimiter."""
        result = build_action_name("answer_clarification", "task", 0, "abc123")
        assert result == "answer_clarification_task__0__abc123"

    def test_different_handle_ids(self):
        """Different handle IDs produce different names."""
        result1 = build_action_name("stop", "search", 0)
        result2 = build_action_name("stop", "search", 1)
        assert result1 != result2
        assert "__0" in result1
        assert "__1" in result2

    def test_all_operations(self):
        """All steering operations produce valid names."""
        for op in STEERING_OPERATIONS:
            result = build_action_name(op.name, "test_task", 5)
            assert result.startswith(f"{op.name}_")
            assert "__5" in result


class TestParseActionName:
    """Tests for parse_action_name function."""

    def test_basic_parsing(self):
        """Parses basic action name correctly."""
        result = parse_action_name("ask_list_contacts__0")
        assert result.operation == "ask"
        assert result.handle_id == 0
        assert result.call_id_suffix is None

    def test_parse_with_call_id(self):
        """Parses action name with call_id_suffix."""
        result = parse_action_name("answer_clarification_task__0__abc123")
        assert result.operation == "answer_clarification"
        assert result.handle_id == 0
        assert result.call_id_suffix == "abc123"

    def test_parse_with_numeric_call_id(self):
        """Correctly parses when call_id_suffix contains digits."""
        # This is the key test - the bug we fixed
        result = parse_action_name(
            "answer_clarification_list_all_contacts__0__tion_123",
        )
        assert result.operation == "answer_clarification"
        assert result.handle_id == 0
        assert result.call_id_suffix == "tion_123"

    def test_different_operations(self):
        """Parses different operation types."""
        test_cases = [
            ("stop_search_web__1", "stop", 1),
            ("interject_do_task__2", "interject", 2),
            ("pause_long_task__0", "pause", 0),
            ("resume_paused_task__3", "resume", 3),
        ]
        for action_name, expected_op, expected_id in test_cases:
            result = parse_action_name(action_name)
            assert result.operation == expected_op
            assert result.handle_id == expected_id

    def test_invalid_action_name(self):
        """Handles invalid action names gracefully."""
        result = parse_action_name("not_a_valid_action")
        assert result.handle_id == 0

    def test_no_delimiter(self):
        """Handles action names without __ delimiter."""
        result = parse_action_name("ask_something_0")
        # Should still extract operation
        assert result.operation == "ask"

    def test_roundtrip(self):
        """build_action_name and parse_action_name are inverses."""
        for op in STEERING_OPERATIONS:
            if op.requires_clarification:
                original = build_action_name(op.name, "test_task", 7, "suffix")
            else:
                original = build_action_name(op.name, "test_task", 7)
            parsed = parse_action_name(original)
            assert parsed.operation == op.name
            assert parsed.handle_id == 7


class TestIsDynamicAction:
    """Tests for is_dynamic_action function."""

    def test_dynamic_actions(self):
        """Recognizes dynamic task actions."""
        dynamic_names = [
            "ask_something__0",
            "stop_task__1",
            "interject_other__2",
            "pause_this__0",
            "resume_that__1",
            "answer_clarification_query__0__suffix",
        ]
        for name in dynamic_names:
            assert is_dynamic_action(name), f"{name} should be dynamic"

    def test_static_actions(self):
        """Rejects static/built-in actions."""
        static_names = [
            "send_sms",
            "send_email",
            "make_call",
            "start_task",
            "wait",
            "unknown_action",
        ]
        for name in static_names:
            assert not is_dynamic_action(name), f"{name} should not be dynamic"


class TestIterAvailableActionsForTask:
    """Tests for iter_available_actions_for_task function."""

    def test_basic_actions(self):
        """Generates standard actions for a task."""
        actions = iter_available_actions_for_task(0, "List contacts")
        action_names = [a[0] for a in actions]

        assert any("ask_" in n for n in action_names)
        assert any("stop_" in n for n in action_names)
        assert any("interject_" in n for n in action_names)
        assert any("pause_" in n for n in action_names)
        assert any("resume_" in n for n in action_names)

    def test_no_answer_clarification_without_pending(self):
        """No answer_clarification without pending clarifications."""
        actions = iter_available_actions_for_task(
            0,
            "Task",
            pending_clarifications=None,
        )
        action_names = [a[0] for a in actions]

        assert not any("answer_clarification" in n for n in action_names)

    def test_answer_clarification_with_pending(self):
        """Generates answer_clarification for pending clarifications."""
        pending = [{"call_id": "test_call_123"}]
        actions = iter_available_actions_for_task(
            0,
            "Task",
            pending_clarifications=pending,
        )
        action_names = [a[0] for a in actions]

        assert any("answer_clarification" in n for n in action_names)

    def test_actions_have_descriptions(self):
        """All actions have non-empty descriptions."""
        actions = iter_available_actions_for_task(0, "Test task")
        for name, description in actions:
            assert len(description) > 0, f"{name} should have a description"


class TestGetSteeringOperation:
    """Tests for get_steering_operation function."""

    def test_valid_operations(self):
        """Returns SteeringOperation for valid names."""
        for op in STEERING_OPERATIONS:
            result = get_steering_operation(op.name)
            assert result is not None
            assert result.name == op.name

    def test_invalid_operation(self):
        """Returns None for invalid names."""
        assert get_steering_operation("invalid") is None
        assert get_steering_operation("") is None


class TestSteeringOperationDocstring:
    """Tests for SteeringOperation.get_docstring method."""

    def test_operations_have_docstrings(self):
        """All operations have docstrings from SteerableToolHandle."""
        for op in STEERING_OPERATIONS:
            # Some might have empty docstrings if method doesn't exist
            # but the method should not raise
            docstring = op.get_docstring()
            assert isinstance(docstring, str)


class TestParsedActionNameProperties:
    """Tests for ParsedActionName dataclass."""

    def test_steering_operation_property(self):
        """steering_operation property returns correct operation."""
        parsed = ParsedActionName("ask", 0, None)
        assert parsed.steering_operation is not None
        assert parsed.steering_operation.name == "ask"

    def test_steering_operation_invalid(self):
        """steering_operation returns None for invalid operation."""
        parsed = ParsedActionName("invalid", 0, None)
        assert parsed.steering_operation is None


# =============================================================================
# domains/notifications.py tests
# =============================================================================


class TestNotification:
    """Tests for Notification dataclass."""

    def test_basic_notification(self):
        """Creates notification with required fields."""
        ts = datetime.now()
        n = Notification(type="Test", content="Hello", timestamp=ts)
        assert n.type == "Test"
        assert n.content == "Hello"
        assert n.timestamp == ts
        assert n.pinned is False
        assert n.interjection_id is None

    def test_pinned_notification(self):
        """Creates pinned notification."""
        ts = datetime.now()
        n = Notification(
            type="Alert",
            content="Important",
            timestamp=ts,
            pinned=True,
            interjection_id="inj_123",
        )
        assert n.pinned is True
        assert n.interjection_id == "inj_123"


class TestNotificationBar:
    """Tests for NotificationBar class."""

    def test_empty_bar(self):
        """New bar starts empty."""
        bar = NotificationBar()
        assert len(bar.notifications) == 0

    def test_push_notif(self):
        """Can push notifications."""
        bar = NotificationBar()
        ts = datetime.now()
        bar.push_notif("Comms", "SMS received", ts)
        assert len(bar.notifications) == 1
        assert bar.notifications[0].type == "Comms"
        assert bar.notifications[0].content == "SMS received"

    def test_push_multiple(self):
        """Can push multiple notifications."""
        bar = NotificationBar()
        ts = datetime.now()
        bar.push_notif("Type1", "Content1", ts)
        bar.push_notif("Type2", "Content2", ts)
        bar.push_notif("Type3", "Content3", ts)
        assert len(bar.notifications) == 3

    def test_push_pinned(self):
        """Can push pinned notifications."""
        bar = NotificationBar()
        ts = datetime.now()
        bar.push_notif("Alert", "Urgent", ts, pinned=True, id="alert_1")
        assert bar.notifications[0].pinned is True
        assert bar.notifications[0].interjection_id == "alert_1"


# =============================================================================
# domains/renderer.py tests
# =============================================================================


class TestRenderer:
    """Tests for Renderer class."""

    @pytest.fixture
    def renderer(self):
        return Renderer()

    @pytest.fixture
    def sample_contact(self):
        return Contact(
            contact_id=1,
            first_name="John",
            surname="Doe",
            is_boss=True,
            phone_number="+15551234567",
            email_address="john@example.com",
            bio="Test bio",
            rolling_summary="Test summary",
            response_policy="Be polite",
            threads={},
            on_call=False,
        )

    def test_render_active_tasks_empty(self, renderer):
        """Renders empty active tasks."""
        result = renderer.render_active_tasks({})
        assert "No active tasks" in result
        assert "<active_tasks>" in result
        assert "</active_tasks>" in result

    def test_render_active_tasks_with_task(self, renderer):
        """Renders active tasks with task data."""
        tasks = {
            0: {
                "query": "List all contacts",
                "handle_actions": [],
            },
        }
        result = renderer.render_active_tasks(tasks)
        assert "<task id='0'" in result
        assert "List all contacts" in result
        assert "ask_" in result
        assert "stop_" in result

    def test_render_active_tasks_with_clarification(self, renderer):
        """Renders tasks with pending clarifications."""
        tasks = {
            0: {
                "query": "Do something",
                "handle_actions": [
                    {
                        "action_name": "clarification_request",
                        "query": "Need more info?",
                        "call_id": "call_123",
                    },
                ],
            },
        }
        result = renderer.render_active_tasks(tasks)
        assert "answer_clarification" in result
        assert "clarification_request" in result
        assert "Need more info?" in result

    def test_render_notification_bar_empty(self, renderer):
        """Renders empty notification bar."""
        bar = NotificationBar()
        last_snapshot = datetime.now() - timedelta(hours=1)
        result = renderer.render_notification_bar(bar, last_snapshot)
        assert "<notifications>" in result
        assert "</notifications>" in result

    def test_render_notification_bar_with_notifications(self, renderer):
        """Renders notification bar with notifications."""
        bar = NotificationBar()
        ts = datetime.now()
        bar.push_notif("Comms", "SMS received from John", ts)
        last_snapshot = datetime.now() - timedelta(seconds=10)
        result = renderer.render_notification_bar(bar, last_snapshot)
        assert "SMS received from John" in result

    def test_render_message(self, renderer):
        """Renders a message with NEW tag for new messages."""
        old_snapshot = datetime.now() - timedelta(hours=1)
        msg = Message(
            name="John",
            content="Hello!",
            timestamp=datetime.now(),
        )
        result = renderer.render_message(msg, old_snapshot)
        assert "**NEW**" in result
        assert "John" in result
        assert "Hello!" in result

    def test_render_message_not_new(self, renderer):
        """Old messages don't have NEW tag."""
        recent_snapshot = datetime.now() + timedelta(seconds=10)
        msg = Message(
            name="John",
            content="Hello!",
            timestamp=datetime.now(),
        )
        result = renderer.render_message(msg, recent_snapshot)
        assert "**NEW**" not in result

    def test_render_email_message(self, renderer):
        """Renders email message with subject and body."""
        old_snapshot = datetime.now() - timedelta(hours=1)
        msg = EmailMessage(
            name="Jane",
            subject="Important Update",
            body="Please review the attached document.",
            email_id="email_456",
            timestamp=datetime.now(),
        )
        result = renderer.render_message(msg, old_snapshot)
        assert "Subject: Important Update" in result
        assert "Email ID: email_456" in result
        assert "Please review" in result

    def test_render_contact(self, renderer, sample_contact):
        """Renders contact with all details."""
        last_snapshot = datetime.now()
        result = renderer.render_contact(sample_contact, last_snapshot=last_snapshot)
        assert 'contact_id="1"' in result
        assert 'first_name="John"' in result
        assert 'surname="Doe"' in result
        assert 'is_boss="True"' in result
        assert "<bio>Test bio</bio>" in result


# =============================================================================
# Integration-style tests for action name format
# =============================================================================


class TestActionNameFormatIntegration:
    """Integration tests ensuring action names work end-to-end."""

    def test_build_parse_roundtrip_all_operations(self):
        """All operations roundtrip correctly through build and parse."""
        test_queries = [
            "List all contacts",
            "Search for documents",
            "Send an email to John",
            "What's the weather in NYC?",
        ]

        for query in test_queries:
            short_name = derive_short_name(query)
            for op in STEERING_OPERATIONS:
                if op.requires_clarification:
                    suffix = safe_call_id_suffix("test-call-id-123")
                    action_name = build_action_name(op.name, short_name, 42, suffix)
                else:
                    action_name = build_action_name(op.name, short_name, 42)

                parsed = parse_action_name(action_name)
                assert parsed.operation == op.name
                assert parsed.handle_id == 42
                assert is_dynamic_action(action_name)

    def test_action_names_in_iter_match_parser(self):
        """Actions from iter_available_actions_for_task parse correctly."""
        pending = [{"call_id": "clarification-uuid-12345"}]
        actions = iter_available_actions_for_task(
            handle_id=3,
            query="Do something important",
            pending_clarifications=pending,
        )

        for action_name, _ in actions:
            parsed = parse_action_name(action_name)
            assert parsed.handle_id == 3
            assert parsed.operation in OPERATION_MAP

    def test_numeric_suffix_does_not_confuse_parser(self):
        """Call ID suffixes with numbers don't confuse the parser."""
        # This tests the specific bug that was fixed
        test_cases = [
            ("answer_clarification_task__0__123", 0, "123"),
            ("answer_clarification_task__5__abc_789", 5, "abc_789"),
            ("answer_clarification_list_contacts__10__suffix_999", 10, "suffix_999"),
        ]
        for action_name, expected_handle, expected_suffix in test_cases:
            parsed = parse_action_name(action_name)
            assert parsed.handle_id == expected_handle, f"Failed for {action_name}"
            assert parsed.call_id_suffix == expected_suffix, f"Failed for {action_name}"
