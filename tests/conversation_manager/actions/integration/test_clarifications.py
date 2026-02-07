"""
Clarification ConversationManager → CodeActActor integration tests.

These validate that when an actor run needs clarification, CM can:
- receive the clarification request
- present it / route it back into the handle
- continue execution deterministically
"""

import asyncio

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
@pytest.mark.timeout(300)
@_handle_project
async def test_clarification_handle_contract(initialized_cm_codeact):
    """
    Smoke: CM → CodeActActor clarification contract works end-to-end.

    We do NOT rely on any manager prompt behavior. Instead, we verify the public
    actor handle clarification surface that CM relies on:
    - `handle.next_clarification()` yields a question
    - `handle.answer_clarification(call_id, answer)` delivers the answer

    The handle is a standard AsyncToolLoopHandle (same as ContactManager, etc.).
    Clarification questions arrive in the handle's internal _clar_q (populated by
    the inner loop's _handle_clarification when nested tools request clarification).
    Answers are routed back through the inner loop's mirror mechanism.
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

    # Inject a clarification question into the handle's internal clarification
    # queue — the same queue that the inner loop's _handle_clarification() populates
    # when a nested tool (e.g. ContactManager) calls request_clarification().
    question = "Which John did you mean: John Smith or John Doe?"
    call_id = "test-clar-0"
    handle._clar_q.put_nowait(
        {
            "type": "clarification",
            "call_id": call_id,
            "tool_name": "request_clarification",
            "question": question,
        },
    )

    # Verify next_clarification() surfaces the injected question.
    clar = await asyncio.wait_for(handle.next_clarification(), timeout=300)
    assert isinstance(clar, dict)
    q = str(clar.get("question") or "")
    assert "which john" in q.lower(), f"Expected clarification about John, got: {q!r}"

    # Answer clarification via the public API.
    # The answer is routed through the inner loop's mirror mechanism.
    resp_call_id = str(clar.get("call_id") or "")
    await handle.answer_clarification(resp_call_id, "John Smith")

    # End the handle to complete the flow deterministically.
    await handle.stop(reason="clarification_answered")
    _final = await wait_for_actor_completion(cm, handle_id, timeout=300)

    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
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

    # Inject a clarification question into the handle's internal clarification queue
    # (same queue that the inner loop populates when nested tools request clarification).
    # Then deterministically apply the corresponding CM-level event, since the background
    # broker consumer is not running in step-driven tests.
    question = "Which John did you mean: John Smith or John Doe?"
    call_id = "0"
    handle._clar_q.put_nowait(
        {
            "type": "clarification",
            "call_id": call_id,
            "tool_name": "request_clarification",
            "question": question,
        },
    )
    await inject_actor_clarification_request(
        cm,
        handle_id=handle_id,
        query=question,
        call_id=call_id,
    )

    # Wait until CM has recorded a pending clarification request.
    await wait_for_condition(
        lambda: any(
            a.get("action_name") == "clarification_request"
            for a in cm.cm.in_flight_actions[handle_id].get("handle_actions", [])
        ),
        timeout=300,
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
        timeout=300,
        timeout_message="Timed out waiting for CM brain to call answer_clarification_* steering tool.",
    )

    # Continue until wait; the action should complete or progress without errors.
    _ = await run_cm_until_wait(cm, max_steps=5)
    _final = await wait_for_actor_completion(cm, handle_id, timeout=300)

    assert_no_errors(result)
