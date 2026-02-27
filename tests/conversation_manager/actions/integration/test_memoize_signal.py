"""
ConversationManager → CodeActActor integration test for memoization signals.

Verifies the slow brain correctly interprets "remember this" / "save this
workflow" as stop actions rather than interjections during in-flight sessions
with a real CodeActActor.
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    assert_steering_called,
)
from tests.conversation_manager.conftest import BOSS
from tests.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    wait_for_actor_completion,
)
from unity.conversation_manager.events import SMSReceived

pytestmark = [pytest.mark.integration, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_memoize_signal_triggers_stop(initialized_cm_codeact):
    """User says "remember this for next time" during a guided session.

    Flow:
    1. User describes a workflow and gives the first instruction (triggers act)
    2. User gives another guided step (interject)
    3. User says "remember this for next time"
    4. Verify stop_* is called (not interject_*)
    """
    cm = initialized_cm_codeact

    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "I'm going to show you how to process a refund in our CRM. "
                "First, look up the customer John Smith in my contacts."
            ),
        ),
    )
    actor_event = get_actor_started_event(result1)
    handle_id = actor_event.handle_id

    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Now search the transcripts for any previous refund discussions with John.",
        ),
    )

    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Perfect, remember this workflow for next time.",
        ),
    )

    assert_steering_called(
        cm,
        "stop_",
        "Memoization signal should call stop_* to end the guided session",
        result=result3,
    )

    actor_result = await wait_for_actor_completion(cm, handle_id, timeout=300)
    assert_no_errors(result1)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_save_workflow_signal_triggers_stop(initialized_cm_codeact):
    """User says "save this, I want you to do this on your own" — alternate phrasing.

    Flow:
    1. User gives a guided instruction (triggers act)
    2. User says they want the assistant to do this autonomously
    3. Verify stop_* is called
    """
    cm = initialized_cm_codeact

    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Let me walk you through how we onboard a new vendor. "
                "Start by looking up all contacts who work at Acme Corp."
            ),
        ),
    )
    actor_event = get_actor_started_event(result1)
    handle_id = actor_event.handle_id

    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Great, save this. I want you to be able to do this on your own next time."
            ),
        ),
    )

    assert_steering_called(
        cm,
        "stop_",
        "'Save this, do it on your own' should call stop_* to end the session",
        result=result2,
    )

    actor_result = await wait_for_actor_completion(cm, handle_id, timeout=300)
    assert_no_errors(result1)
