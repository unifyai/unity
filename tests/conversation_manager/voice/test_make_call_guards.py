"""Tests for make_call safety guards.

Guard (a): If the LLM produces more than one make_call in a single turn, ALL
instances are rejected without execution (exclusive_tools mechanism).

Guard (b): make_call is never exposed as a tool when a voice call or meet is
already in progress (either cm.mode.is_voice or a subprocess is running).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)
from unity.conversation_manager.domains.contact_index import ContactIndex
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.types import Mode
from unity.common.single_shot import (
    single_shot_tool_decision,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_cm(*, mode=Mode.TEXT, has_call_proc=False, assistant_number="+15550000"):
    cm = MagicMock()
    cm.mode = mode
    cm.contact_index = ContactIndex()
    cm.in_flight_actions = {}
    cm.completed_actions = {}
    cm.notifications_bar = NotificationBar()
    cm.chat_history = []
    cm.assistant_number = assistant_number
    cm.assistant_email = "assistant@test.com"
    cm.call_manager = MagicMock()
    cm.call_manager._call_proc = MagicMock() if has_call_proc else None
    cm.call_manager.has_active_call = has_call_proc
    cm.call_manager.has_active_google_meet = False
    cm.call_manager._whatsapp_call_joining = False
    cm.computer_fast_path_eligible = False
    return cm


# ===========================================================================
# Guard (b): make_call not exposed during active voice sessions
# ===========================================================================


class TestMakeCallNotExposedDuringVoice:

    def test_make_call_exposed_in_text_mode(self):
        """make_call is available when in TEXT mode with no active call."""
        cm = _make_mock_cm(mode=Mode.TEXT)
        with patch(
            "unity.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            tools_instance = ConversationManagerBrainActionTools(cm)
            tools = tools_instance.as_tools()
        assert "make_call" in tools

    def test_make_call_hidden_during_phone_call(self):
        """make_call is not available when cm.mode is CALL."""
        cm = _make_mock_cm(mode=Mode.CALL)
        with patch(
            "unity.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            tools_instance = ConversationManagerBrainActionTools(cm)
            tools = tools_instance.as_tools()
        assert "make_call" not in tools

    def test_make_call_hidden_during_meet(self):
        """make_call is not available when cm.mode is MEET."""
        cm = _make_mock_cm(mode=Mode.MEET)
        with patch(
            "unity.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            tools_instance = ConversationManagerBrainActionTools(cm)
            tools = tools_instance.as_tools()
        assert "make_call" not in tools

    def test_make_call_hidden_when_subprocess_running(self):
        """make_call is not available when a call subprocess is already running,
        even if cm.mode hasn't transitioned to voice yet."""
        cm = _make_mock_cm(mode=Mode.TEXT, has_call_proc=True)
        with patch(
            "unity.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            tools_instance = ConversationManagerBrainActionTools(cm)
            tools = tools_instance.as_tools()
        assert "make_call" not in tools

    def test_other_tools_unaffected_during_voice(self):
        """send_sms and send_email remain available during voice calls."""
        cm = _make_mock_cm(mode=Mode.CALL)
        with patch(
            "unity.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            tools_instance = ConversationManagerBrainActionTools(cm)
            tools = tools_instance.as_tools()
        assert "send_sms" in tools
        assert "send_email" in tools

    def test_make_call_hidden_without_assistant_number(self):
        """make_call is not available when assistant has no phone number."""
        cm = _make_mock_cm(mode=Mode.TEXT, assistant_number="")
        with patch(
            "unity.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            tools_instance = ConversationManagerBrainActionTools(cm)
            tools = tools_instance.as_tools()
        assert "make_call" not in tools


# ===========================================================================
# Guard (a): exclusive_tools rejects duplicate make_call in single turn
# ===========================================================================


@pytest.mark.asyncio
class TestExclusiveToolsRejectsDuplicateMakeCall:

    async def test_single_make_call_executes_normally(self):
        """A single make_call in a turn executes without issue."""
        call_count = 0

        async def make_call(*, contact_id: int) -> dict:
            """Place a phone call to a contact.

            Parameters
            ----------
            contact_id : int
                The contact to call.
            """
            nonlocal call_count
            call_count += 1
            return {"status": "ok"}

        async def wait() -> dict:
            """Wait for a response."""
            return {"status": "waiting"}

        client = MagicMock()
        client.messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {
                            "name": "make_call",
                            "arguments": '{"contact_id": 1}',
                        },
                    },
                ],
            },
        ]
        client.generate = AsyncMock()

        result = await single_shot_tool_decision(
            client,
            "Call Alice",
            {"make_call": make_call, "wait": wait},
            exclusive_tools={"make_call"},
        )

        assert call_count == 1
        assert len(result.tools) == 1
        assert result.tools[0].name == "make_call"
        assert result.tools[0].result == {"status": "ok"}

    async def test_duplicate_make_call_rejects_all(self):
        """Two make_call calls in a single turn are both rejected."""
        call_count = 0

        async def make_call(*, contact_id: int) -> dict:
            """Place a phone call to a contact.

            Parameters
            ----------
            contact_id : int
                The contact to call.
            """
            nonlocal call_count
            call_count += 1
            return {"status": "ok"}

        client = MagicMock()
        client.messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {
                            "name": "make_call",
                            "arguments": '{"contact_id": 1}',
                        },
                    },
                    {
                        "type": "function",
                        "id": "call_2",
                        "function": {
                            "name": "make_call",
                            "arguments": '{"contact_id": 2}',
                        },
                    },
                ],
            },
        ]
        client.generate = AsyncMock()

        result = await single_shot_tool_decision(
            client,
            "Call Alice and Bob",
            {"make_call": make_call},
            exclusive_tools={"make_call"},
        )

        assert call_count == 0, "Neither make_call should have executed"
        assert len(result.tools) == 2
        for tool_exec in result.tools:
            assert tool_exec.name == "make_call"
            assert "error" in tool_exec.result

    async def test_non_exclusive_tools_execute_alongside_rejected_exclusive(self):
        """Other tools still execute normally even when an exclusive tool is rejected."""
        call_count = 0
        sms_count = 0

        async def make_call(*, contact_id: int) -> dict:
            """Place a phone call.

            Parameters
            ----------
            contact_id : int
                The contact to call.
            """
            nonlocal call_count
            call_count += 1
            return {"status": "ok"}

        async def send_sms(*, contact_id: int, message: str) -> dict:
            """Send an SMS message.

            Parameters
            ----------
            contact_id : int
                The contact to message.
            message : str
                The message content.
            """
            nonlocal sms_count
            sms_count += 1
            return {"status": "sent"}

        client = MagicMock()
        client.messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {
                            "name": "make_call",
                            "arguments": '{"contact_id": 1}',
                        },
                    },
                    {
                        "type": "function",
                        "id": "call_2",
                        "function": {
                            "name": "make_call",
                            "arguments": '{"contact_id": 2}',
                        },
                    },
                    {
                        "type": "function",
                        "id": "sms_1",
                        "function": {
                            "name": "send_sms",
                            "arguments": '{"contact_id": 3, "message": "hi"}',
                        },
                    },
                ],
            },
        ]
        client.generate = AsyncMock()

        result = await single_shot_tool_decision(
            client,
            "Call Alice and Bob, and text Charlie",
            {"make_call": make_call, "send_sms": send_sms},
            exclusive_tools={"make_call"},
        )

        assert call_count == 0, "make_call should not have executed"
        assert sms_count == 1, "send_sms should have executed normally"
        make_call_results = [t for t in result.tools if t.name == "make_call"]
        sms_results = [t for t in result.tools if t.name == "send_sms"]
        assert len(make_call_results) == 2
        assert all("error" in t.result for t in make_call_results)
        assert len(sms_results) == 1
        assert sms_results[0].result == {"status": "sent"}

    async def test_exclusive_tools_none_disables_check(self):
        """When exclusive_tools is None, duplicate tools execute normally."""
        call_count = 0

        async def make_call(*, contact_id: int) -> dict:
            """Place a phone call.

            Parameters
            ----------
            contact_id : int
                The contact to call.
            """
            nonlocal call_count
            call_count += 1
            return {"status": "ok"}

        client = MagicMock()
        client.messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {
                            "name": "make_call",
                            "arguments": '{"contact_id": 1}',
                        },
                    },
                    {
                        "type": "function",
                        "id": "call_2",
                        "function": {
                            "name": "make_call",
                            "arguments": '{"contact_id": 2}',
                        },
                    },
                ],
            },
        ]
        client.generate = AsyncMock()

        result = await single_shot_tool_decision(
            client,
            "Call Alice and Bob",
            {"make_call": make_call},
            exclusive_tools=None,
        )

        assert call_count == 2, "Both calls should execute without exclusive_tools"
        assert len(result.tools) == 2
        assert all(t.result == {"status": "ok"} for t in result.tools)
