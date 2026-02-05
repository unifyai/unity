"""
Clarification ConversationManager → CodeActActor integration tests.

These validate that when an actor run needs clarification, CM can:
- receive the clarification request
- present it / route it back into the handle
- continue execution deterministically
"""

import pytest

from tests.helpers import _handle_project, get_or_create_contact
from tests.conversation_manager.conftest import BOSS
from tests.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    inject_actor_clarification_request,
    run_cm_until_wait,
    wait_for_condition,
    wait_for_actor_completion,
)
from unity.conversation_manager.events import SMSReceived

pytestmark = [pytest.mark.integration, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_clarification_handle_contract(initialized_cm_codeact):
    """
    Smoke: CM → CodeActActor clarification contract works end-to-end.

    We do NOT rely on any manager prompt behavior. Instead, we verify the public
    actor handle clarification surface that CM relies on:
    - `handle.next_clarification()` yields a question
    - `handle.answer_clarification(call_id, answer)` delivers the answer
    """
    cm = initialized_cm_codeact

    # Ensure there are two Johns so the actor is forced to clarify.
    _ = get_or_create_contact(
        cm.cm.contact_manager,
        first_name="John",
        surname="Smith",
        email_address="john.smith@test.com",
        phone_number="+15555550111",
    )
    _ = get_or_create_contact(
        cm.cm.contact_manager,
        first_name="John",
        surname="Doe",
        email_address="john.doe@test.com",
        phone_number="+15555550112",
    )

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Find contacts named John in our records and return their phone numbers and any "
                "distinguishing info (company, last name, notes)."
            ),
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    handle = cm.cm.in_flight_actions[handle_id]["handle"]

    # Force a clarification through the handle surface (contract-level test).
    up_q = getattr(handle, "clarification_up_q", None)
    down_q = getattr(handle, "clarification_down_q", None)
    assert (
        up_q is not None and down_q is not None
    ), "Handle missing clarification queues"

    try:
        # Inject a clarification question directly into the handle's queue.
        up_q.put_nowait("Which John did you mean: John Smith or John Doe?")

        import asyncio

        clar = await asyncio.wait_for(handle.next_clarification(), timeout=30)
        assert isinstance(clar, dict)
        q = str(clar.get("question") or "")
        assert "which john" in q.lower()

        # Answer clarification (call_id may be omitted/ignored).
        call_id = str(clar.get("call_id") or "")
        await handle.answer_clarification(call_id, "John Smith")

        # Verify the down-channel received the answer.
        got = down_q.get_nowait()
        assert "john smith" in str(got).lower()

        # End the handle to complete the flow deterministically.
        await handle.stop(reason="clarification_answered")
        _final = await wait_for_actor_completion(cm, handle_id, timeout=30)
    finally:
        pass

    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_clarification_cm_event_broker_path(initialized_cm_codeact):
    """
    CM event-based clarification wiring works end-to-end (golden path).

    Validates the full production path:
      handle.next_clarification() → actor_watch_clarifications publishes event →
      EventHandler appends pending clarification → CM brain selects answer_clarification_* →
      steering tool routes to handle.answer_clarification(...)
    """
    cm = initialized_cm_codeact

    # Ensure two Johns so the actor prompt is naturally ambiguous.
    _ = get_or_create_contact(
        cm.cm.contact_manager,
        first_name="John",
        surname="Smith",
        email_address="john.smith@golden.test",
        phone_number="+15555550991",
    )
    _ = get_or_create_contact(
        cm.cm.contact_manager,
        first_name="John",
        surname="Doe",
        email_address="john.doe@golden.test",
        phone_number="+15555550992",
    )

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Find John in my contacts and tell me his phone number.",
        ),
    )
    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    handle = cm.cm.in_flight_actions[handle_id]["handle"]

    # Force a clarification to be emitted through the handle surface so the watcher publishes
    # ActorClarificationRequest. In step-mode tests, we then *apply* the event to CM state
    # deterministically (since the background broker consumer is not driving EventHandler).
    up_q = getattr(handle, "clarification_up_q", None)
    assert up_q is not None, "Handle missing clarification_up_q"
    question = "Which John did you mean: John Smith or John Doe?"
    up_q.put_nowait(question)
    # In step-driven tests, the background event-broker consumer is not running,
    # so we deterministically apply the clarification event to CM state.
    await inject_actor_clarification_request(
        cm,
        handle_id=handle_id,
        query=question,
        call_id="0",
    )

    # Wait until CM has recorded a pending clarification request.
    await wait_for_condition(
        lambda: any(
            a.get("action_name") == "clarification_request"
            for a in cm.cm.in_flight_actions[handle_id].get("handle_actions", [])
        ),
        timeout=30,
        timeout_message="Timed out waiting for CM to record clarification_request in handle_actions.",
    )

    # Send a user message that should cause CM brain to answer via answer_clarification_* tool.
    _ = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="I meant John Smith.",
        ),
    )

    # The act of answering should record an action in handle_actions.
    await wait_for_condition(
        lambda: any(
            str(a.get("action_name", "")).startswith("answer_clarification_")
            for a in cm.cm.in_flight_actions.get(handle_id, {}).get(
                "handle_actions",
                [],
            )
        ),
        timeout=30,
        timeout_message="Timed out waiting for CM brain to call answer_clarification_* steering tool.",
    )

    # Continue until wait; the action should complete or progress without errors.
    _ = await run_cm_until_wait(cm, max_steps=5)
    _final = await wait_for_actor_completion(cm, handle_id, timeout=90)

    assert_no_errors(result)
