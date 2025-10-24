import pytest
import unify
import json

from unittest.mock import patch
from tests.helpers import _handle_project
from datetime import datetime

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.contact_manager.contact_manager import ContactManager
from unity.common._async_tool.semantic_cache import _Config
from unity.common._async_tool import semantic_cache as sc
from unity.transcript_manager.types.message import Message


@pytest.fixture(autouse=True)
def _patch_semantic_cache_config(monkeypatch):
    class _DynamicConfig(_Config):
        # Raise threshold to ensure cache is always hit during the test
        threshold = 0.5

        @property
        def context(self):
            return f"{unify.get_active_context()['write']}/SemanticCache"

    monkeypatch.setattr(
        "unity.common._async_tool.semantic_cache._CONFIG",
        _DynamicConfig(),
    )


def _count_tool_calls_in_reasoning(reasoning_steps) -> tuple[int, set[str]]:
    """Count the number of tool calls in the reasoning steps."""
    tool_call_count = 0
    tool_names = set()
    for step in reasoning_steps:
        if step.get("role") == "tool":
            if step.get("name") == "semantic_search":
                continue
            tool_call_count += 1
            tool_names.add(step.get("name"))
    return tool_call_count, tool_names


@pytest.mark.asyncio
@_handle_project
async def test_semantic_cache_exact_match():
    cm = ContactManager()
    first_contact = cm._create_contact(first_name="John", surname="Doe")
    second_contact = cm._create_contact(first_name="Bob", surname="Alice")
    fc_id = first_contact["details"]["contact_id"]
    sc_id = second_contact["details"]["contact_id"]

    tm = TranscriptManager()
    tm.log_messages(
        Message(
            medium="email",
            sender_id=fc_id,
            receiver_ids=[sc_id],
            timestamp=datetime.now(),
            content="Hey there!",
            synchronous=True,
        ),
    )

    with patch(
        "unity.transcript_manager.transcript_manager.is_semantic_cache_enabled",
        return_value=True,
    ):
        handle = await tm.ask(
            "Is there any contact with name John? if so, please provide the latest message by John "
            "Provide only the message content if any in the JSON format of {message: <message_content>}",
            _return_reasoning_steps=True,
        )
        res, reasoning_steps_first = await handle.result()
        total_calls_first, _ = _count_tool_calls_in_reasoning(reasoning_steps_first)
        assert json.loads(res)["message"] == "Hey there!"

        sc._SEMANTIC_CACHE_SAVER.wait()

        second_handle = await tm.ask(
            "Is there any contact with name John? if so, please provide the latest message by John "
            "Provide only the message content if any in the JSON format of {message: <message_content>}",
            _return_reasoning_steps=True,
        )
        second_res, reasoning_steps_second = await second_handle.result()
        assert json.loads(second_res)["message"] == "Hey there!"

        # Reasoning should take less steps as no tools should be called
        total_calls_second, tools_called = _count_tool_calls_in_reasoning(
            reasoning_steps_second,
        )
        assert total_calls_second < total_calls_first
        # No tool calls should be made
        assert "search_messages" not in tools_called
        assert "filter_messages" not in tools_called


@pytest.mark.asyncio
@_handle_project
async def test_semantic_cache_no_exact_match():
    cm = ContactManager()
    first_contact = cm._create_contact(first_name="John", surname="Doe")
    second_contact = cm._create_contact(first_name="Bob", surname="Alice")
    fc_id = first_contact["details"]["contact_id"]
    sc_id = second_contact["details"]["contact_id"]

    tm = TranscriptManager()
    tm.log_messages(
        Message(
            medium="email",
            sender_id=fc_id,
            receiver_ids=[sc_id],
            timestamp=datetime.now(),
            content="Hey there!",
            synchronous=True,
        ),
    )

    with patch(
        "unity.transcript_manager.transcript_manager.is_semantic_cache_enabled",
        return_value=True,
    ):
        handle = await tm.ask(
            "Is there any contact with name John? if so, please provide the latest message by John "
            "Provide only the message content if any in the JSON format of {message: <message_content>}",
            _return_reasoning_steps=True,
        )
        res, reasoning_steps_first = await handle.result()
        total_calls_first, _ = _count_tool_calls_in_reasoning(reasoning_steps_first)
        assert json.loads(res)["message"] == "Hey there!"

        sc._SEMANTIC_CACHE_SAVER.wait()

        second_handle = await tm.ask(
            "Please provide the latest message by John "
            "Provide only the message content if any in the JSON format of {message: <message_content>}",
            _return_reasoning_steps=True,
        )
        second_res, reasoning_steps_second = await second_handle.result()
        assert json.loads(second_res)["message"] == "Hey there!"

        # Reasoning should take less steps as no tools should be called
        total_calls_second, tools_called = _count_tool_calls_in_reasoning(
            reasoning_steps_second,
        )
        assert total_calls_second < total_calls_first
        # No tool calls should be made
        assert "search_messages" not in tools_called
        assert "filter_messages" not in tools_called


@pytest.mark.asyncio
@_handle_project
async def test_semantic_cache_similar_query_benefit():
    cm = ContactManager()
    first_contact = cm._create_contact(first_name="John", surname="Doe")
    second_contact = cm._create_contact(first_name="Bob", surname="Alice")
    fc_id = first_contact["details"]["contact_id"]
    sc_id = second_contact["details"]["contact_id"]

    tm = TranscriptManager()
    tm.log_messages(
        Message(
            medium="email",
            sender_id=fc_id,
            receiver_ids=[sc_id],
            timestamp=datetime.now(),
            content="Hey there!",
            synchronous=True,
        ),
    )

    tm.log_messages(
        Message(
            medium="email",
            sender_id=sc_id,
            receiver_ids=[fc_id],
            timestamp=datetime.now(),
            content="This is Bob!",
            synchronous=True,
        ),
    )

    with patch(
        "unity.transcript_manager.transcript_manager.is_semantic_cache_enabled",
        return_value=True,
    ):
        handle = await tm.ask(
            "Do I have a contact named John? can you provide the latest message by John? "
            "Provide only the message content if any in the JSON format of {message: <message_content>}",
            _return_reasoning_steps=True,
        )
        res, reasoning_steps_first = await handle.result()
        total_calls_first, _ = _count_tool_calls_in_reasoning(reasoning_steps_first)
        assert json.loads(res)["message"] == "Hey there!"

        sc._SEMANTIC_CACHE_SAVER.wait()

        second_handle = await tm.ask(
            "Can you provide the latest message by Bob? "
            "Provide only the message content if any in the JSON format of {message: <message_content>}",
            _return_reasoning_steps=True,
        )
        second_res, reasoning_steps_second = await second_handle.result()
        assert json.loads(second_res)["message"] == "This is Bob!"

        # Reasoning should take less or equal steps
        total_calls_second, _ = _count_tool_calls_in_reasoning(reasoning_steps_second)
        assert total_calls_second <= total_calls_first
