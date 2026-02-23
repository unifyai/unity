"""
tests/conversation_manager/test_multi_turn.py
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
from tests.conversation_manager.cm_helpers import (
    assert_content_contains,
    get_exactly_one,
)
from tests.conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import (
    SMSReceived,
    SMSSent,
    UnifyMessageReceived,
    UnifyMessageSent,
)

pytestmark = pytest.mark.eval


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
    contact = TEST_CONTACTS[0]

    # Turn 1: User mentions a unique identifier
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="My order number is ABC-9876. Please confirm you have it.",
        ),
    )
    msg1 = get_exactly_one(result1.output_events, UnifyMessageSent)
    assert msg1.content

    # Turn 2: User asks assistant to recall it
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="What was my order number?",
        ),
    )
    msg2 = get_exactly_one(result2.output_events, UnifyMessageSent)

    # Assistant should recall the order number
    assert_content_contains(
        msg2.content,
        "ABC-9876",
        "Assistant should recall order number from previous turn",
        cm=cm,
        result=result2,
    )


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_three_turn_recall(initialized_cm):
    """
    Three-turn conversation: verify context persists across multiple exchanges.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    # Turn 1: First piece of info (use non-identity info to avoid conflict with contact metadata)
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="My favorite color is blue. Please acknowledge.",
        ),
    )
    get_exactly_one(result1.output_events, UnifyMessageSent)

    # Turn 2: Second piece of info
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="I live in Seattle. Please acknowledge.",
        ),
    )
    get_exactly_one(result2.output_events, UnifyMessageSent)

    # Turn 3: Ask about both
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=contact,
            content="What is my favorite color and where do I live?",
        ),
    )
    msg3 = get_exactly_one(result3.output_events, UnifyMessageSent)

    assert_content_contains(
        msg3.content,
        "blue",
        "Assistant should recall favorite color from earlier turns",
        cm=cm,
        result=result3,
    )
    assert_content_contains(
        msg3.content,
        "Seattle",
        "Assistant should recall location from earlier turns",
        cm=cm,
        result=result3,
    )


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
    contact = TEST_CONTACTS[0]

    # Turn 1: User provides info
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="Remember this code: DELTA-42. Acknowledge please.",
        ),
    )
    msg1 = get_exactly_one(result1.output_events, SMSSent)
    assert msg1.content

    # Turn 2: User asks for recall
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=contact,
            content="What was the code I gave you?",
        ),
    )
    msg2 = get_exactly_one(result2.output_events, SMSSent)

    assert_content_contains(
        msg2.content,
        "DELTA-42",
        "Assistant should recall code from previous turn",
        cm=cm,
        result=result2,
    )
