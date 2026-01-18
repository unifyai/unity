"""
tests/test_conversation_manager/test_multi_turn.py
==================================================

Tests for multi-turn conversation memory.

These tests verify that the ConversationManager maintains context across
multiple user messages - i.e., the assistant can recall information from
earlier turns in the conversation.

Unlike test_comms.py (single-turn: one message in, one response out),
these tests send multiple messages and verify the assistant reasons
over the full conversation history.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import (
    SMSReceived,
    SMSSent,
    UnifyMessageReceived,
    UnifyMessageSent,
)

pytestmark = pytest.mark.eval


def _only(events, typ):
    return [e for e in events if isinstance(e, typ)]


def _get_one(events, typ):
    """Get exactly one event of the given type."""
    matches = _only(events, typ)
    assert len(matches) == 1, f"Expected 1 {typ.__name__}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
#  Multi-turn tests: UnifyMessage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_two_turn_recall(initialized_cm):
    """
    Two-turn conversation: user mentions a word, then asks assistant to recall it.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Turn 1: User mentions a unique identifier
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="My order number is ABC-9876. Please confirm you have it.",
        ),
    )
    msg1 = _get_one(result1.output_events, UnifyMessageSent)
    assert msg1.content

    # Turn 2: User asks assistant to recall it
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="What was my order number?",
        ),
    )
    msg2 = _get_one(result2.output_events, UnifyMessageSent)

    # Assistant should recall the order number
    assert (
        "ABC-9876" in msg2.content or "abc-9876" in msg2.content.lower()
    ), f"Assistant should recall order number 'ABC-9876', got: {msg2.content}"


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_three_turn_recall(initialized_cm):
    """
    Three-turn conversation: verify context persists across multiple exchanges.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Turn 1: First piece of info
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="My name is Jordan. Please acknowledge.",
        ),
    )
    _get_one(result1.output_events, UnifyMessageSent)

    # Turn 2: Second piece of info
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="I live in Seattle. Please acknowledge.",
        ),
    )
    _get_one(result2.output_events, UnifyMessageSent)

    # Turn 3: Ask about both
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="What is my name and where do I live?",
        ),
    )
    msg3 = _get_one(result3.output_events, UnifyMessageSent)
    content_lower = msg3.content.lower()

    assert "jordan" in content_lower, f"Should recall 'Jordan': {msg3.content}"
    assert "seattle" in content_lower, f"Should recall 'Seattle': {msg3.content}"


# ---------------------------------------------------------------------------
#  Multi-turn tests: SMS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_sms_two_turn_recall(initialized_cm):
    """
    Two-turn SMS conversation: user gives info, then asks for recall.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Turn 1: User provides info
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Remember this code: DELTA-42. Acknowledge please.",
        ),
    )
    msg1 = _get_one(result1.output_events, SMSSent)
    assert msg1.content

    # Turn 2: User asks for recall
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="What was the code I gave you?",
        ),
    )
    msg2 = _get_one(result2.output_events, SMSSent)

    assert (
        "DELTA-42" in msg2.content or "delta-42" in msg2.content.lower()
    ), f"Assistant should recall code 'DELTA-42', got: {msg2.content}"
