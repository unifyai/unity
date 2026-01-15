"""
tests/test_conversation_manager/test_multi_contact_outbound.py
=============================================================

Tests for outbound messages to contacts not yet in active_conversations.

These tests verify that when the boss asks to message someone who hasn't
messaged first (i.e., not in active_conversations), the ConversationManager
correctly routes through start_task to get contact details from the Actor,
then sends the message once details are returned.

Uses SimulatedActor which returns plausible made-up contact details.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import (
    SMSReceived,
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
