"""
tests/test_conversation_manager/test_multi_contact_outbound.py
=============================================================

Tests for outbound messages to contacts not yet in active_conversations.

These tests verify that when the boss asks to message someone who hasn't
messaged first (i.e., not in active_conversations), the ConversationManager
correctly routes through start_task to get contact details from the Actor,
then sends the message once details are returned.

Also tests that when the boss provides contact details inline (e.g., "call
David on +1234567890"), the LLM correctly uses the contact_details field
rather than delegating to start_task.

Uses SimulatedActor which returns plausible made-up contact details.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import (
    SMSReceived,
    SMSSent,
    EmailSent,
    PhoneCallSent,
    ActorHandleStarted,
)

pytestmark = pytest.mark.eval

# Convenience references to test contacts
BOSS = TEST_CONTACTS[1]  # contact_id 1 - the main user


def _only(events, typ):
    return [e for e in events if isinstance(e, typ)]


# ---------------------------------------------------------------------------
#  Outbound to unknown contact triggers start_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_email_unknown_contact_triggers_start_task(initialized_cm):
    """
    Boss asks to email someone not in contacts -> should call start_task.

    When the boss says "email David about X", and David is not in
    active_conversations, the assistant should use start_task to
    get David's contact details from the Actor.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Could you please email David and tell him the meeting is confirmed",
        ),
    )

    # Check that start_task was called (ActorHandleStarted event)
    actor_events = _only(result.output_events, ActorHandleStarted)

    assert len(actor_events) >= 1, (
        f"Expected start_task to be called (ActorHandleStarted event), "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )

    # The query should reference David and the message
    task_event = actor_events[0]
    assert (
        "david" in task_event.query.lower() or "email" in task_event.query.lower()
    ), f"start_task query should mention David or email, got: {task_event.query}"


# ---------------------------------------------------------------------------
#  Outbound with inline contact details - should use contact_details field
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_call_with_inline_phone_number(initialized_cm):
    """
    Boss provides phone number inline -> should call make_call with contact_details.

    When the boss says "call David, his number is +15551234567", the LLM should
    directly call make_call with contact_details containing the phone number,
    NOT delegate to start_task.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Please call David, his number is +15551234567",
        ),
    )

    # Check that make_call was triggered (PhoneCallSent event)
    call_events = _only(result.output_events, PhoneCallSent)

    # Should NOT have called start_task since we provided the number
    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) == 0, (
        f"Should not call start_task when phone number is provided inline, "
        f"got ActorHandleStarted events: {actor_events}"
    )

    assert len(call_events) >= 1, (
        f"Expected make_call to be triggered (PhoneCallSent event), "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_sms_with_inline_phone_number(initialized_cm):
    """
    Boss provides phone number inline -> should call send_sms with contact_details.

    When the boss says "text Joanna on +15559876543 saying hi", the LLM should
    directly call send_sms with contact_details containing the phone number,
    NOT delegate to start_task.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Please text Joanna on +15559876543, tell her the package arrived",
        ),
    )

    # Check that send_sms was triggered (SMSSent event)
    sms_events = _only(result.output_events, SMSSent)

    # Should NOT have called start_task since we provided the number
    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) == 0, (
        f"Should not call start_task when phone number is provided inline, "
        f"got ActorHandleStarted events: {actor_events}"
    )

    assert len(sms_events) >= 1, (
        f"Expected send_sms to be triggered (SMSSent event), "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_email_with_inline_email_address(initialized_cm):
    """
    Boss provides email address inline -> should call send_email with contact_details.

    When the boss says "email Johnny at johnny@example.com", the LLM should
    directly call send_email with contact_details containing the email address,
    NOT delegate to start_task.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Could you email Johnny at johnny@example.com and tell him the report is ready",
        ),
    )

    # Check that send_email was triggered (EmailSent event)
    email_events = _only(result.output_events, EmailSent)

    # Should NOT have called start_task since we provided the email
    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) == 0, (
        f"Should not call start_task when email address is provided inline, "
        f"got ActorHandleStarted events: {actor_events}"
    )

    assert len(email_events) >= 1, (
        f"Expected send_email to be triggered (EmailSent event), "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
