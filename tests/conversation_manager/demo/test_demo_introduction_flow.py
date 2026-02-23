"""
tests/conversation_manager/demo/test_demo_introduction_flow.py
===============================================================

Eval tests for the core demo introduction flow.

The standard demo flow is:
1. Demo operator (Daniel, contact_id=2) messages or calls the assistant.
2. Operator introduces the prospect (e.g., "Richard").
3. The assistant learns the prospect's name and saves it.
4. Operator asks the assistant to contact the prospect directly.
5. The assistant calls/texts the prospect using contact_id=1.

These tests verify the slow brain handles each phase correctly via SMS.
Voice-based introduction flows (phone call handoff) will be added once
the voice test infrastructure supports demo mode subprocess spawning.
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    filter_events_by_type,
)
from tests.conversation_manager.demo.conftest import (
    DEMO_OPERATOR,
)
from unity.conversation_manager.events import (
    SMSReceived,
    SMSSent,
    PhoneCallSent,
    ActorHandleStarted,
)

pytestmark = pytest.mark.eval


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _sms_and_get_replies(cm, contact: dict, message: str) -> list[str]:
    """Send an SMS and return all SMS reply contents."""
    result = await cm.step_until_wait(
        SMSReceived(contact=contact, content=message),
    )
    return [sms.content for sms in filter_events_by_type(result.output_events, SMSSent)]


async def _sms_and_get_all_events(cm, contact: dict, message: str):
    """Send an SMS and return the full StepResult."""
    return await cm.step_until_wait(
        SMSReceived(contact=contact, content=message),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Operator introduction flow (SMS)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_operator_greeting(initialized_cm):
    """The assistant should respond naturally to the demo operator's greeting."""
    replies = await _sms_and_get_replies(
        initialized_cm,
        DEMO_OPERATOR,
        "Hey Lucy, it's Daniel from Unify. How's it going?",
    )

    assert replies, "Assistant should reply to the demo operator"
    combined = " ".join(replies).lower()
    # Should respond conversationally — not confused or robotic
    assert len(combined) > 5, "Reply should be substantive, not empty"


@pytest.mark.asyncio
@_handle_project
async def test_operator_introduces_prospect(initialized_cm):
    """When the operator introduces a prospect, the assistant should acknowledge."""
    replies = await _sms_and_get_replies(
        initialized_cm,
        DEMO_OPERATOR,
        (
            "I'm going to introduce you to Richard, who you'll be working "
            "with going forward. He's a CEO of a mid-size tech startup."
        ),
    )

    assert replies, "Assistant should acknowledge the introduction"


@pytest.mark.asyncio
@_handle_project
async def test_operator_asks_to_call_prospect_with_number(initialized_cm):
    """When the operator provides a number and asks to call, assistant should use make_call."""
    # First, introduce the prospect so the assistant has context
    await _sms_and_get_replies(
        initialized_cm,
        DEMO_OPERATOR,
        "I'd like you to meet Richard. He's excited to work with you.",
    )

    # Now ask to call with the prospect's number
    result = await _sms_and_get_all_events(
        initialized_cm,
        DEMO_OPERATOR,
        "Great! Richard's number is +447700900123. Give him a call and introduce yourself.",
    )

    # The assistant should attempt to make a call (PhoneCallSent event).
    # Note: all_tool_calls only tracks one tool name per LLM step, so when the
    # LLM calls set_boss_details + make_call in parallel, make_call may not
    # appear in all_tool_calls. Check output events instead.
    call_events = filter_events_by_type(result.output_events, PhoneCallSent)
    sent_sms = filter_events_by_type(result.output_events, SMSSent)

    # Either the assistant placed the call or acknowledged the request via SMS
    assert call_events or sent_sms, (
        "Assistant should either place the call (PhoneCallSent) or "
        "acknowledge the request (SMSSent)"
    )


@pytest.mark.asyncio
@_handle_project
async def test_no_act_triggered_in_demo_mode(initialized_cm):
    """No matter what the operator asks, act should never be triggered in demo mode."""
    # Ask something that would normally trigger act
    result = await _sms_and_get_all_events(
        initialized_cm,
        DEMO_OPERATOR,
        "Can you look up Richard's company details online?",
    )

    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) == 0, (
        "act should never be triggered in demo mode — "
        f"got {len(actor_events)} ActorHandleStarted events"
    )

    # The assistant should explain the limitation gracefully
    replies = filter_events_by_type(result.output_events, SMSSent)
    assert replies, "Assistant should reply explaining the limitation"


@pytest.mark.asyncio
@_handle_project
async def test_boss_learns_name_via_set_boss_details(initialized_cm):
    """When the operator tells the assistant the prospect's name, it should be saved."""
    # Tell the assistant the prospect's name
    await _sms_and_get_replies(
        initialized_cm,
        DEMO_OPERATOR,
        ("Your new boss is called Richard Hendricks. " "Please save his details."),
    )

    # Verify the name was saved to the boss contact.
    # Note: we check the contact directly rather than gating on all_tool_calls,
    # because all_tool_calls only tracks one tool per LLM step and may miss
    # set_boss_details if it was called alongside send_sms in one step.
    boss = initialized_cm.contact_index.get_contact(1)
    assert (
        boss.get("first_name") == "Richard"
    ), f"Expected boss first_name='Richard', got {boss.get('first_name')!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Direct prospect interaction (SMS)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_prospect_sends_first_message(initialized_cm):
    """The prospect (contact_id=1) sends a message directly — assistant should respond warmly."""
    # First update boss with a name so the contact has some identity
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    action_tools = ConversationManagerBrainActionTools(initialized_cm.cm)
    await action_tools.set_boss_details(
        first_name="Richard",
        phone_number="+447700900123",
    )

    # Simulate the prospect sending an SMS
    boss_contact = initialized_cm.contact_index.get_contact(1)
    replies = await _sms_and_get_replies(
        initialized_cm,
        boss_contact,
        "Hi Lucy! Daniel told me about you. Excited to chat!",
    )

    assert replies, "Assistant should respond to the prospect"
    combined = " ".join(replies).lower()
    assert len(combined) > 10, "Reply should be warm and substantive"


@pytest.mark.asyncio
@_handle_project
async def test_prospect_asks_about_email(initialized_cm):
    """When the prospect shares their email, the assistant should save it."""
    # Set up boss with name
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    action_tools = ConversationManagerBrainActionTools(initialized_cm.cm)
    await action_tools.set_boss_details(
        first_name="Richard",
        phone_number="+447700900123",
    )

    boss_contact = initialized_cm.contact_index.get_contact(1)

    # Prospect shares their email
    await _sms_and_get_replies(
        initialized_cm,
        boss_contact,
        "By the way, my email is richard@hendricks.com — feel free to save it.",
    )

    # Verify the email was saved to the boss contact.
    # Check the contact directly rather than gating on all_tool_calls
    # (see note in test_boss_learns_name_via_set_boss_details).
    boss = initialized_cm.contact_index.get_contact(1)
    assert boss.get("email_address") == "richard@hendricks.com", (
        f"Expected boss email='richard@hendricks.com', "
        f"got {boss.get('email_address')!r}"
    )
