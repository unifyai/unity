import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.conftest import BOSS
from tests.test_conversation_manager.test_actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    wait_for_actor_completion,
    wait_for_condition,
)
from unity.conversation_manager.events import SMSReceived
from unity.conversation_manager.domains.brain_action_tools import (
    get_handle_paused_state,
)

pytestmark = [pytest.mark.integration, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_pause_resume_smoke(initialized_cm_codeact):
    """Smoke: pause → resume steering works on a CodeActActor handle."""
    cm = initialized_cm_codeact

    # Use a prompt that tends to trigger multiple tool calls (transcripts + tasks).
    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Search my transcripts for anything about the budget, then summarize key points "
                "and create a follow-up task."
            ),
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    handle = cm.cm.in_flight_actions[handle_id]["handle"]

    # Pause immediately (avoid races with very fast completions).
    await handle.pause()

    await wait_for_condition(
        lambda: get_handle_paused_state(handle) is True,
        timeout=30,
        poll=0.05,
        timeout_message="Timed out waiting for handle to enter paused state.",
    )

    await handle.resume()

    _final = await wait_for_actor_completion(cm, handle_id, timeout=90)
    assert_no_errors(result)
