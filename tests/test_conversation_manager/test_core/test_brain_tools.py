"""
tests/test_conversation_manager/test_brain_tools.py
====================================================

Unit tests for ConversationManager brain tools.

Tests cover:
- ConversationManagerBrainTools (read-only inspection tools)
- ConversationManagerBrainActionTools (side-effecting action tools)

These tests verify the tool implementations directly, testing:
- Tool method signatures and return types
- Tool docstrings (important for LLM understanding)
- Dynamic tool generation for task steering
- Integration with ConversationManager state
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.conversation_manager.domains.brain_tools import (
    ConversationManagerBrainTools,
)
from unity.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
    _get_or_create_contact,
)
from unity.conversation_manager.domains.notifications import (
    NotificationBar,
)
from unity.conversation_manager.domains.contact_index import (
    ContactIndex,
)
from unity.conversation_manager.task_actions import (
    STEERING_OPERATIONS,
    parse_action_name,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_cm():
    """Create a minimal mock ConversationManager for testing."""
    cm = MagicMock()
    cm.mode = "text"
    cm.contact_index = ContactIndex()
    cm.active_tasks = {}
    cm.notifications_bar = NotificationBar()
    cm.chat_history = []
    cm.assistant_number = "+15555550000"
    cm.assistant_email = "assistant@test.com"
    return cm


@pytest.fixture
def brain_tools(mock_cm):
    """Create ConversationManagerBrainTools instance."""
    return ConversationManagerBrainTools(mock_cm)


@pytest.fixture
def brain_action_tools(mock_cm):
    """Create ConversationManagerBrainActionTools instance."""
    # Patch the event broker to avoid actual pubsub
    with patch(
        "unity.conversation_manager.domains.brain_action_tools.get_event_broker",
    ) as mock_broker:
        mock_broker.return_value = MagicMock()
        mock_broker.return_value.publish = AsyncMock()
        tools = ConversationManagerBrainActionTools(mock_cm)
        yield tools


@pytest.fixture
def sample_contacts():
    """Sample contacts for testing."""
    return [
        {
            "contact_id": 1,
            "first_name": "Alice",
            "surname": "Smith",
            "phone_number": "+15551111111",
            "email_address": "alice@example.com",
        },
        {
            "contact_id": 2,
            "first_name": "Bob",
            "surname": "Johnson",
            "phone_number": "+15552222222",
            "email_address": "bob@example.com",
        },
    ]


# =============================================================================
# ConversationManagerBrainTools Tests
# =============================================================================


class TestCmGetMode:
    """Tests for cm_get_mode tool."""

    def test_returns_text_mode(self, brain_tools, mock_cm):
        """Returns 'text' when CM is in text mode."""
        mock_cm.mode = "text"
        assert brain_tools.cm_get_mode() == "text"

    def test_returns_call_mode(self, brain_tools, mock_cm):
        """Returns 'call' when CM is in call mode."""
        mock_cm.mode = "call"
        assert brain_tools.cm_get_mode() == "call"

    def test_returns_unify_meet_mode(self, brain_tools, mock_cm):
        """Returns 'unify_meet' when CM is in unify_meet mode."""
        mock_cm.mode = "unify_meet"
        assert brain_tools.cm_get_mode() == "unify_meet"

    def test_converts_mode_to_string(self, brain_tools, mock_cm):
        """Converts mode to string regardless of type."""
        # Mode could be an enum or other type
        mock_cm.mode = MagicMock(__str__=lambda self: "custom_mode")
        result = brain_tools.cm_get_mode()
        assert isinstance(result, str)


class TestCmGetContact:
    """Tests for cm_get_contact tool."""

    def test_returns_contact_by_id(self, brain_tools, mock_cm, sample_contacts):
        """Returns contact when found by ID."""
        mock_cm.contact_index.set_contacts(sample_contacts)
        result = brain_tools.cm_get_contact(1)
        assert result is not None
        assert result["contact_id"] == 1
        assert result["first_name"] == "Alice"

    def test_returns_none_for_unknown_id(self, brain_tools, mock_cm, sample_contacts):
        """Returns None when contact not found."""
        mock_cm.contact_index.set_contacts(sample_contacts)
        result = brain_tools.cm_get_contact(999)
        assert result is None

    def test_excludes_threads_from_contact(self, brain_tools, mock_cm, sample_contacts):
        """Contact summary excludes thread data for efficiency."""
        mock_cm.contact_index.set_contacts(sample_contacts)
        result = brain_tools.cm_get_contact(1)
        # get_contact uses model_dump(exclude={"threads", "global_thread"})
        assert "threads" not in result
        assert "global_thread" not in result


class TestCmListActiveTasks:
    """Tests for cm_list_active_tasks tool."""

    def test_returns_empty_list_when_no_tasks(self, brain_tools, mock_cm):
        """Returns empty list when no active tasks."""
        mock_cm.active_tasks = {}
        result = brain_tools.cm_list_active_tasks()
        assert result == []

    def test_returns_task_summary(self, brain_tools, mock_cm):
        """Returns summary for each active task."""
        mock_cm.active_tasks = {
            0: {"query": "Search for contacts", "handle_actions": []},
            1: {"query": "Send an email", "handle_actions": [{"action": "test"}]},
        }
        result = brain_tools.cm_list_active_tasks()
        assert len(result) == 2
        assert result[0]["handle_id"] == 0
        assert result[0]["query"] == "Search for contacts"
        assert result[0]["num_handle_actions"] == 0
        assert result[1]["handle_id"] == 1
        assert result[1]["query"] == "Send an email"
        assert result[1]["num_handle_actions"] == 1

    def test_handles_none_active_tasks(self, brain_tools, mock_cm):
        """Handles None active_tasks gracefully."""
        mock_cm.active_tasks = None
        result = brain_tools.cm_list_active_tasks()
        assert result == []

    def test_handles_none_handle_actions(self, brain_tools, mock_cm):
        """Handles None handle_actions in task data."""
        mock_cm.active_tasks = {
            0: {"query": "Task", "handle_actions": None},
        }
        result = brain_tools.cm_list_active_tasks()
        assert result[0]["num_handle_actions"] == 0


class TestCmListNotifications:
    """Tests for cm_list_notifications tool."""

    def test_returns_empty_list_when_no_notifications(self, brain_tools, mock_cm):
        """Returns empty list when no notifications."""
        result = brain_tools.cm_list_notifications()
        assert result == []

    def test_returns_all_notifications(self, brain_tools, mock_cm):
        """Returns all notifications when pinned_only=False."""
        ts = datetime.now()
        mock_cm.notifications_bar.push_notif("Type1", "Content1", ts)
        mock_cm.notifications_bar.push_notif("Type2", "Content2", ts, pinned=True)
        result = brain_tools.cm_list_notifications()
        assert len(result) == 2

    def test_filters_pinned_only(self, brain_tools, mock_cm):
        """Returns only pinned notifications when pinned_only=True."""
        ts = datetime.now()
        mock_cm.notifications_bar.push_notif("Regular", "Not pinned", ts)
        mock_cm.notifications_bar.push_notif("Pinned", "Important", ts, pinned=True)
        result = brain_tools.cm_list_notifications(pinned_only=True)
        assert len(result) == 1
        assert result[0]["content"] == "Important"

    def test_converts_timestamp_to_isoformat(self, brain_tools, mock_cm):
        """Converts datetime timestamps to ISO format strings."""
        ts = datetime(2024, 1, 15, 10, 30, 0)
        mock_cm.notifications_bar.push_notif("Test", "Content", ts)
        result = brain_tools.cm_list_notifications()
        assert result[0]["timestamp"] == "2024-01-15T10:30:00"


class TestBrainToolsAsTools:
    """Tests for as_tools method."""

    def test_returns_dict_of_callables(self, brain_tools):
        """Returns dictionary mapping names to callable methods."""
        tools = brain_tools.as_tools()
        assert isinstance(tools, dict)
        assert all(callable(fn) for fn in tools.values())

    def test_contains_all_brain_tools(self, brain_tools):
        """Contains all expected brain tools."""
        tools = brain_tools.as_tools()
        expected = {
            "cm_get_mode",
            "cm_get_contact",
            "cm_list_active_tasks",
            "cm_list_notifications",
        }
        assert set(tools.keys()) == expected

    def test_tools_are_bound_methods(self, brain_tools):
        """Tools are bound to the BrainTools instance."""
        tools = brain_tools.as_tools()
        # Calling through the dict should work
        assert tools["cm_get_mode"]() == "text"


# =============================================================================
# ConversationManagerBrainActionTools Tests
# =============================================================================


class TestActionToolsAsTools:
    """Tests for action tools as_tools method."""

    def test_returns_dict_of_callables(self, brain_action_tools):
        """Returns dictionary mapping names to callable methods."""
        tools = brain_action_tools.as_tools()
        assert isinstance(tools, dict)
        assert all(callable(fn) for fn in tools.values())

    def test_contains_all_action_tools(self, brain_action_tools):
        """Contains all expected action tools."""
        tools = brain_action_tools.as_tools()
        expected = {
            "send_sms",
            "send_unify_message",
            "send_email",
            "make_call",
            "act",
            "wait",
        }
        assert set(tools.keys()) == expected


class TestWaitTool:
    """Tests for wait tool."""

    @pytest.mark.asyncio
    async def test_returns_waiting_status(self, brain_action_tools):
        """Wait tool returns waiting status."""
        result = await brain_action_tools.wait()
        assert result == {"status": "waiting"}

    def test_has_docstring(self, brain_action_tools):
        """Wait tool has descriptive docstring."""
        assert brain_action_tools.wait.__doc__ is not None
        assert "Wait" in brain_action_tools.wait.__doc__


class TestSendSmsTool:
    """Tests for send_sms tool."""

    @pytest.mark.asyncio
    async def test_requires_contact_id_or_details(self, brain_action_tools):
        """Raises error if neither contact_id nor contact_details provided."""
        with pytest.raises(ValueError, match="Either contact_id or details"):
            await brain_action_tools.send_sms(content="Hello")

    @pytest.mark.asyncio
    async def test_has_docstring(self, brain_action_tools):
        """Send SMS tool has descriptive docstring."""
        assert brain_action_tools.send_sms.__doc__ is not None
        assert "SMS" in brain_action_tools.send_sms.__doc__

    @pytest.mark.asyncio
    async def test_returns_error_for_contact_without_phone(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Returns error when contact has no phone number."""
        # Set up contact without phone number
        contact_without_phone = {
            "contact_id": 5,
            "first_name": "NoPhone",
            "surname": "Person",
            "email_address": "nophone@example.com",
            "should_respond": True,
            # No phone_number field
        }
        mock_cm.contact_index.set_contacts([contact_without_phone])

        result = await brain_action_tools.send_sms(
            contact_id=5,
            content="Hello",
        )

        assert result["status"] == "error"
        assert "does not have" in result["error"]
        assert "phone" in result["error"].lower()


class TestSendUnifyMessageTool:
    """Tests for send_unify_message tool."""

    def test_has_docstring(self, brain_action_tools):
        """Send Unify message tool has descriptive docstring."""
        assert brain_action_tools.send_unify_message.__doc__ is not None
        assert "Unify" in brain_action_tools.send_unify_message.__doc__


class TestSendEmailTool:
    """Tests for send_email tool."""

    @pytest.mark.asyncio
    async def test_requires_contact_id_or_details(self, brain_action_tools):
        """Raises error if neither contact_id nor contact_details provided."""
        with pytest.raises(ValueError, match="Either contact_id or details"):
            await brain_action_tools.send_email(subject="Test", body="Body")

    def test_has_docstring(self, brain_action_tools):
        """Send email tool has descriptive docstring."""
        assert brain_action_tools.send_email.__doc__ is not None
        assert "email" in brain_action_tools.send_email.__doc__.lower()

    @pytest.mark.asyncio
    async def test_returns_error_for_contact_without_email(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Returns error when contact has no email address."""
        # Set up contact without email address
        contact_without_email = {
            "contact_id": 4,
            "first_name": "NoEmail",
            "surname": "Person",
            "phone_number": "+15555554444",
            "should_respond": True,
            # No email_address field
        }
        mock_cm.contact_index.set_contacts([contact_without_email])

        result = await brain_action_tools.send_email(
            contact_id=4,
            subject="Test",
            body="Hello",
        )

        assert result["status"] == "error"
        assert "does not have" in result["error"]
        assert "email" in result["error"].lower()


class TestMakeCallTool:
    """Tests for make_call tool."""

    @pytest.mark.asyncio
    async def test_requires_contact_id_or_details(self, brain_action_tools):
        """Raises error if neither contact_id nor contact_details provided."""
        with pytest.raises(ValueError, match="Either contact_id or details"):
            await brain_action_tools.make_call()

    def test_has_docstring(self, brain_action_tools):
        """Make call tool has descriptive docstring."""
        assert brain_action_tools.make_call.__doc__ is not None
        assert "call" in brain_action_tools.make_call.__doc__.lower()

    @pytest.mark.asyncio
    async def test_returns_error_for_contact_without_phone(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Returns error when contact has no phone number."""
        # Set up contact without phone number
        contact_without_phone = {
            "contact_id": 5,
            "first_name": "NoPhone",
            "surname": "Person",
            "email_address": "nophone@example.com",
            "should_respond": True,
            # No phone_number field
        }
        mock_cm.contact_index.set_contacts([contact_without_phone])

        result = await brain_action_tools.make_call(contact_id=5)

        assert result["status"] == "error"
        assert "does not have" in result["error"]
        assert "phone" in result["error"].lower()


class TestActTool:
    """Tests for act tool."""

    def test_has_docstring(self, brain_action_tools):
        """Act tool has descriptive docstring."""
        assert brain_action_tools.act.__doc__ is not None
        assert (
            "Engage" in brain_action_tools.act.__doc__
            or "act" in brain_action_tools.act.__doc__.lower()
        )

    def test_docstring_describes_capabilities(self, brain_action_tools):
        """Act tool docstring describes its capabilities."""
        doc = brain_action_tools.act.__doc__
        # Should mention key capabilities
        assert "Retrieval" in doc or "search" in doc.lower()


# =============================================================================
# Dynamic Task Steering Tools Tests
# =============================================================================


class TestBuildTaskSteeringTools:
    """Tests for build_task_steering_tools method."""

    def test_returns_empty_dict_when_no_active_tasks(self, brain_action_tools, mock_cm):
        """Returns empty dict when no active tasks."""
        mock_cm.active_tasks = {}
        tools = brain_action_tools.build_task_steering_tools()
        assert tools == {}

    def test_generates_tools_for_active_task(self, brain_action_tools, mock_cm):
        """Generates steering tools for each active task."""
        mock_cm.active_tasks = {
            0: {
                "query": "List all contacts",
                "handle": MagicMock(),
                "handle_actions": [],
            },
        }
        tools = brain_action_tools.build_task_steering_tools()
        # Should have tools for ask, stop, interject, pause, resume
        # (but NOT answer_clarification without pending clarifications)
        non_clar_ops = [
            op for op in STEERING_OPERATIONS if not op.requires_clarification
        ]
        assert len(tools) >= len(non_clar_ops)

    def test_tool_names_follow_expected_format(self, brain_action_tools, mock_cm):
        """Tool names follow the build_action_name format."""
        mock_cm.active_tasks = {
            0: {
                "query": "Search web",
                "handle": MagicMock(),
                "handle_actions": [],
            },
        }
        tools = brain_action_tools.build_task_steering_tools()
        for name in tools.keys():
            # Should be parseable
            parsed = parse_action_name(name)
            assert parsed.operation in [op.name for op in STEERING_OPERATIONS]
            assert parsed.handle_id == 0

    def test_generates_answer_clarification_when_pending(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Generates answer_clarification tool when pending clarifications exist."""
        mock_cm.active_tasks = {
            0: {
                "query": "Do something",
                "handle": MagicMock(),
                "handle_actions": [
                    {
                        "action_name": "clarification_request",
                        "query": "Need more info?",
                        "call_id": "call_123",
                    },
                ],
            },
        }
        tools = brain_action_tools.build_task_steering_tools()
        answer_tools = [n for n in tools.keys() if "answer_clarification" in n]
        assert len(answer_tools) == 1

    def test_no_answer_clarification_without_pending(self, brain_action_tools, mock_cm):
        """No answer_clarification tool when no pending clarifications."""
        mock_cm.active_tasks = {
            0: {
                "query": "Do something",
                "handle": MagicMock(),
                "handle_actions": [],  # No pending clarifications
            },
        }
        tools = brain_action_tools.build_task_steering_tools()
        answer_tools = [n for n in tools.keys() if "answer_clarification" in n]
        assert len(answer_tools) == 0

    def test_skips_answered_clarifications(self, brain_action_tools, mock_cm):
        """Does not generate tool for already answered clarifications."""
        mock_cm.active_tasks = {
            0: {
                "query": "Task",
                "handle": MagicMock(),
                "handle_actions": [
                    {
                        "action_name": "clarification_request",
                        "query": "Need info?",
                        "call_id": "call_answered",
                        "response": "Here's the answer",  # Already answered
                    },
                ],
            },
        }
        tools = brain_action_tools.build_task_steering_tools()
        answer_tools = [n for n in tools.keys() if "answer_clarification" in n]
        assert len(answer_tools) == 0

    def test_handles_multiple_tasks(self, brain_action_tools, mock_cm):
        """Generates tools for multiple active tasks."""
        mock_cm.active_tasks = {
            0: {
                "query": "Task one",
                "handle": MagicMock(),
                "handle_actions": [],
            },
            1: {
                "query": "Task two",
                "handle": MagicMock(),
                "handle_actions": [],
            },
        }
        tools = brain_action_tools.build_task_steering_tools()
        # Should have steering tools for both tasks
        task0_tools = [n for n in tools.keys() if "__0" in n]
        task1_tools = [n for n in tools.keys() if "__1" in n]
        assert len(task0_tools) > 0
        assert len(task1_tools) > 0

    def test_steering_tools_have_docstrings(self, brain_action_tools, mock_cm):
        """Generated steering tools have docstrings."""
        mock_cm.active_tasks = {
            0: {
                "query": "Test task",
                "handle": MagicMock(),
                "handle_actions": [],
            },
        }
        tools = brain_action_tools.build_task_steering_tools()
        for name, fn in tools.items():
            assert fn.__doc__ is not None, f"{name} should have docstring"
            assert "Test task" in fn.__doc__, f"{name} docstring should mention task"

    def test_handles_none_active_tasks(self, brain_action_tools, mock_cm):
        """Handles None active_tasks gracefully."""
        mock_cm.active_tasks = None
        tools = brain_action_tools.build_task_steering_tools()
        assert tools == {}


class TestMakeSteeringTool:
    """Tests for _make_steering_tool method."""

    @pytest.mark.asyncio
    async def test_ask_operation_calls_handle_ask(self, brain_action_tools, mock_cm):
        """Ask operation calls handle.ask with parameter."""
        mock_handle = MagicMock()
        mock_ask_handle = MagicMock()
        mock_ask_handle.result = AsyncMock(return_value="Answer")
        mock_handle.ask = AsyncMock(return_value=mock_ask_handle)

        mock_cm.active_tasks = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="ask",
            param_name="query",
            docstring="Ask a question",
            query="Test",
        )
        result = await tool(query="What is the status?")
        mock_handle.ask.assert_called_once()
        assert result["status"] == "ok"
        assert result["operation"] == "ask"

    @pytest.mark.asyncio
    async def test_stop_operation_calls_handle_stop(self, brain_action_tools, mock_cm):
        """Stop operation calls handle.stop and removes task."""
        mock_handle = MagicMock()

        mock_cm.active_tasks = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="stop",
            param_name="reason",
            docstring="Stop the task",
            query="Test",
        )
        result = await tool(reason="No longer needed")
        mock_handle.stop.assert_called_once_with(reason="No longer needed")
        assert result["operation"] == "stop"
        assert 0 not in mock_cm.active_tasks  # Task should be removed

    @pytest.mark.asyncio
    async def test_interject_operation_calls_handle_interject(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Interject operation calls handle.interject."""
        mock_handle = MagicMock()
        mock_handle.interject = AsyncMock()

        mock_cm.active_tasks = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="interject",
            param_name="message",
            docstring="Interject a message",
            query="Test",
        )
        result = await tool(message="Important update")
        mock_handle.interject.assert_called_once()
        assert result["operation"] == "interject"

    @pytest.mark.asyncio
    async def test_pause_operation_calls_handle_pause(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Pause operation calls handle.pause."""
        mock_handle = MagicMock()
        mock_handle.pause = AsyncMock()

        mock_cm.active_tasks = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="pause",
            param_name="",
            docstring="Pause the task",
            query="Test",
        )
        result = await tool()
        mock_handle.pause.assert_called_once()
        assert result["operation"] == "pause"

    @pytest.mark.asyncio
    async def test_resume_operation_calls_handle_resume(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Resume operation calls handle.resume."""
        mock_handle = MagicMock()
        mock_handle.resume = AsyncMock()

        mock_cm.active_tasks = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="resume",
            param_name="",
            docstring="Resume the task",
            query="Test",
        )
        result = await tool()
        mock_handle.resume.assert_called_once()
        assert result["operation"] == "resume"

    @pytest.mark.asyncio
    async def test_answer_clarification_calls_handle_method(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Answer clarification calls handle.answer_clarification."""
        mock_handle = MagicMock()
        mock_handle.answer_clarification = AsyncMock()

        mock_cm.active_tasks = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="answer_clarification",
            param_name="answer",
            docstring="Answer clarification",
            query="Test",
            call_id="call_123",
        )
        result = await tool(answer="Here is the answer")
        mock_handle.answer_clarification.assert_called_once_with(
            "call_123",
            "Here is the answer",
        )
        assert result["operation"] == "answer_clarification"

    @pytest.mark.asyncio
    async def test_records_intervention_in_handle_actions(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Steering operations record intervention in handle_actions."""
        mock_handle = MagicMock()
        mock_handle.pause = AsyncMock()

        mock_cm.active_tasks = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="pause",
            param_name="",
            docstring="Pause",
            query="Test",
        )
        await tool()

        actions = mock_cm.active_tasks[0]["handle_actions"]
        assert len(actions) == 1
        assert actions[0]["action_name"] == "pause_0"

    @pytest.mark.asyncio
    async def test_handles_operation_errors(self, brain_action_tools, mock_cm):
        """Handles errors in steering operations gracefully."""
        mock_handle = MagicMock()
        mock_handle.pause = AsyncMock(side_effect=RuntimeError("Test error"))

        mock_cm.active_tasks = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="pause",
            param_name="",
            docstring="Pause",
            query="Test",
        )
        result = await tool()
        assert "Error" in result["result"]


# =============================================================================
# _get_or_create_contact Helper Tests
# =============================================================================


class TestGetOrCreateContact:
    """Tests for _get_or_create_contact helper function."""

    @pytest.mark.asyncio
    async def test_raises_without_contact_id_or_details(self, mock_cm):
        """Raises ValueError if neither contact_id nor details provided."""
        with pytest.raises(ValueError, match="Either contact_id or details"):
            await _get_or_create_contact(mock_cm)

    @pytest.mark.asyncio
    async def test_retrieves_by_contact_id(self, mock_cm, sample_contacts):
        """Retrieves contact by ID when provided."""
        mock_cm.contact_index.set_contacts(sample_contacts)
        result = await _get_or_create_contact(mock_cm, contact_id=1)
        assert result is not None
        assert result["contact_id"] == 1

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_contact_id(self, mock_cm, sample_contacts):
        """Returns None when contact_id not found."""
        mock_cm.contact_index.set_contacts(sample_contacts)
        result = await _get_or_create_contact(mock_cm, contact_id=999)
        assert result is None


# =============================================================================
# Tool Docstring Quality Tests
# =============================================================================


class TestToolDocstrings:
    """Tests verifying tool docstrings are informative for LLM usage."""

    def test_brain_tools_have_docstrings(self, brain_tools):
        """All brain tools have docstrings."""
        tools = brain_tools.as_tools()
        for name, fn in tools.items():
            assert fn.__doc__ is not None, f"{name} missing docstring"
            assert len(fn.__doc__) > 10, f"{name} docstring too short"

    def test_action_tools_have_docstrings(self, brain_action_tools):
        """All action tools have docstrings."""
        tools = brain_action_tools.as_tools()
        for name, fn in tools.items():
            assert fn.__doc__ is not None, f"{name} missing docstring"
            assert len(fn.__doc__) > 10, f"{name} docstring too short"

    def test_send_sms_docstring_mentions_contact(self, brain_action_tools):
        """send_sms docstring mentions contact parameters."""
        doc = brain_action_tools.send_sms.__doc__
        assert "contact_id" in doc
        assert "contact_details" in doc.lower() or "details" in doc.lower()

    def test_send_email_docstring_mentions_parameters(self, brain_action_tools):
        """send_email docstring mentions subject and body."""
        doc = brain_action_tools.send_email.__doc__
        assert "subject" in doc.lower()
        assert "body" in doc.lower()

    def test_act_docstring_is_comprehensive(self, brain_action_tools):
        """act tool has comprehensive docstring explaining capabilities."""
        doc = brain_action_tools.act.__doc__
        assert len(doc) > 100, "act docstring should be comprehensive"

    def test_wait_docstring_explains_when_to_use(self, brain_action_tools):
        """wait tool docstring explains when to use it."""
        doc = brain_action_tools.wait.__doc__
        assert "PREFER" in doc or "prefer" in doc.lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestBrainToolsIntegration:
    """Integration tests for brain tools working together."""

    def test_brain_and_action_tools_have_distinct_names(
        self,
        brain_tools,
        brain_action_tools,
    ):
        """Brain tools and action tools have non-overlapping names."""
        brain_names = set(brain_tools.as_tools().keys())
        action_names = set(brain_action_tools.as_tools().keys())
        overlap = brain_names & action_names
        assert len(overlap) == 0, f"Overlapping tool names: {overlap}"

    def test_steering_tools_distinct_from_static_tools(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Dynamic steering tools don't overlap with static action tools."""
        mock_cm.active_tasks = {
            0: {
                "query": "Test",
                "handle": MagicMock(),
                "handle_actions": [],
            },
        }
        static_names = set(brain_action_tools.as_tools().keys())
        steering_names = set(brain_action_tools.build_task_steering_tools().keys())
        overlap = static_names & steering_names
        assert len(overlap) == 0, f"Overlapping tool names: {overlap}"
