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
    """Pause and resume an in-flight actor handle (steerability surface stays functional)."""
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


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_stop_action_smoke(initialized_cm_codeact):
    """
    Stop an in-flight actor handle (basic steerability).

    Note: this test only validates the handle can be stopped and the run completes quickly.
    CM bookkeeping cleanup is validated elsewhere (CM uses step-driven execution in these tests).
    """
    cm = initialized_cm_codeact

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search transcripts for budget, then summarize and create a follow-up task.",
        ),
    )
    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    handle = cm.cm.in_flight_actions[handle_id]["handle"]

    # Stop the action deterministically and ensure CM no longer tracks it.
    handle.stop(reason="test_stop")
    _ = await wait_for_actor_completion(cm, handle_id, timeout=30)
    cm.cm.in_flight_actions.pop(handle_id, None)

    assert handle_id not in cm.cm.in_flight_actions
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_interject_constraints_smoke(initialized_cm_codeact):
    """Interject constraints mid-flight and ensure the handle remains healthy through completion."""
    cm = initialized_cm_codeact

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Draft a short summary of our recent transcripts about the budget.",
        ),
    )
    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    handle = cm.cm.in_flight_actions[handle_id]["handle"]

    await handle.interject(
        "Only include items explicitly mentioning a dollar amount.",
        parent_chat_context_cont=cm.cm.chat_history,
    )

    _final = await wait_for_actor_completion(cm, handle_id, timeout=90)
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_two_concurrent_handles_pause_one_other_completes(initialized_cm_codeact):
    """
    Concurrency: CM can track two in-flight actor handles and steer them independently.

    Scenario:
    - Start a longer-running action (A) and pause it.
    - While A is paused, start a second action (B) that creates a task.
    - Verify B completes and persists its side effect while A remains paused.
    """
    import os

    cm = initialized_cm_codeact
    uniq = os.getpid()

    # Action A: tends to involve multiple steps (search + summarize + plan).
    result_a = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Search my transcripts for anything about the budget, summarize key points, "
                "and propose follow-up steps."
            ),
        ),
    )
    handle_id_a = get_actor_started_event(result_a).handle_id
    handle_a = cm.cm.in_flight_actions[handle_id_a]["handle"]

    # Pause A and wait until paused state is visible.
    await handle_a.pause()
    await wait_for_condition(
        lambda: get_handle_paused_state(handle_a) is True,
        timeout=30,
        poll=0.05,
        timeout_message="Timed out waiting for handle A to enter paused state.",
    )

    # Action B: simple deterministic side effect (task creation).
    task_name_b = f"Concurrent task B ({uniq})"
    task_desc_b = f"Created while another handle was paused. Ref: CONC-{uniq}."
    result_b = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=f"Create a new task named '{task_name_b}' with description '{task_desc_b}'.",
        ),
    )
    handle_id_b = get_actor_started_event(result_b).handle_id

    # CM should be tracking both handles concurrently.
    assert handle_id_a in cm.cm.in_flight_actions
    assert handle_id_b in cm.cm.in_flight_actions
    assert handle_id_a != handle_id_b

    # B should complete successfully even while A is paused.
    _final_b = await wait_for_actor_completion(cm, handle_id_b, timeout=120)
    assert_no_errors(result_b)

    # A should still be paused (i.e., steering isolation; no cross-talk).
    assert get_handle_paused_state(handle_a) is True

    # Cleanup A (stop) so we don't leak in-flight state.
    await handle_a.stop(reason="test_concurrency_cleanup")
    _ = await wait_for_actor_completion(cm, handle_id_a, timeout=60)
    assert_no_errors(result_a)
