"""
tests/conversation_manager/test_multi_contact_outbound.py
=============================================================

Tests for outbound messages to contacts not yet in active_conversations.

All communication tools require a ``contact_id``.  When the boss asks to
contact someone not in active_conversations, the CM brain must delegate to
``act`` to find or create the contact — regardless of whether the boss
provided contact details (phone number, email) inline or not.

Uses SimulatedActor which returns plausible made-up contact details.
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import filter_events_by_type
from tests.conversation_manager.conftest import BOSS
from unity.conversation_manager.events import (
    SMSReceived,
    ActorHandleStarted,
)

pytestmark = pytest.mark.eval

# Note: BOSS (contact_id=1) is imported from conftest.py
# TEST_CONTACTS are regular contacts starting from contact_id 2.


# ---------------------------------------------------------------------------
#  Outbound to unknown contact (no details provided) triggers act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_email_unknown_contact_triggers_act(initialized_cm):
    """
    Boss asks to email someone not in contacts -> should call act.

    When the boss says "email David about X", and David is not in
    active_conversations, the assistant should use act to search for
    David's contact details from the Actor.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Could you please email David and tell him the meeting is confirmed",
        ),
    )

    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)

    assert len(actor_events) >= 1, (
        f"Expected act to be called (ActorHandleStarted event), "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )

    task_event = actor_events[0]
    assert (
        "david" in task_event.query.lower() or "email" in task_event.query.lower()
    ), f"act query should mention David or email, got: {task_event.query}"


@pytest.mark.asyncio
@_handle_project
async def test_sms_unknown_contact_triggers_act(initialized_cm):
    """
    Boss asks to text someone not in contacts (no number given) -> should call act.

    "Send David a text about the meeting" — David is not in
    active_conversations and no phone number is provided, so the LLM
    must use act to find David's contact details.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Send David a text letting him know the meeting is confirmed",
        ),
    )

    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called (ActorHandleStarted event), "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )

    task_event = actor_events[0]
    assert (
        "david" in task_event.query.lower()
        or "sms" in task_event.query.lower()
        or "text" in task_event.query.lower()
    ), f"act query should mention David or texting, got: {task_event.query}"


@pytest.mark.asyncio
@_handle_project
async def test_call_unknown_contact_triggers_act(initialized_cm):
    """
    Boss asks to call someone not in contacts (no number given) -> should call act.

    "Give David a call about the meeting" — David is not in
    active_conversations and no phone number is provided, so the LLM
    must use act to find David's contact details.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Give David a call and tell him the meeting is confirmed",
        ),
    )

    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called (ActorHandleStarted event), "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )

    task_event = actor_events[0]
    assert (
        "david" in task_event.query.lower() or "call" in task_event.query.lower()
    ), f"act query should mention David or calling, got: {task_event.query}"


# ---------------------------------------------------------------------------
#  Outbound with inline contact details still triggers act
#
#  Even when the boss provides a phone number or email address inline,
#  the CM brain must still delegate to act because comms tools only
#  accept contact_id.  act will find or create the contact and return
#  the contact_id for subsequent use.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_call_with_inline_phone_number(initialized_cm):
    """
    Boss provides phone number inline for unknown contact -> should still call act.

    "call David, his number is +15551234567" — David is not in
    active_conversations, so the LLM must use act to create/find the
    contact before calling.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Please call David, his number is +15551234567",
        ),
    )

    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called to resolve contact, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_sms_with_inline_phone_number(initialized_cm):
    """
    Boss provides phone number inline for unknown contact -> should still call act.

    "text Joanna on +15559876543" — Joanna is not in active_conversations,
    so the LLM must use act to create/find the contact before sending.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Please text Joanna on +15559876543, tell her the package arrived",
        ),
    )

    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called to resolve contact, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_email_with_inline_email_address(initialized_cm):
    """
    Boss provides email address inline for unknown contact -> should still call act.

    "email Johnny at johnny@example.com" — Johnny is not in
    active_conversations, so the LLM must use act to create/find the
    contact before sending.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Could you email Johnny at johnny@example.com and tell him the report is ready",
        ),
    )

    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called to resolve contact, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
