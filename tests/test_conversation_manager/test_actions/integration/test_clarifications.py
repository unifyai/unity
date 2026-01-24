import pytest

from tests.helpers import _handle_project, get_or_create_contact
from tests.test_conversation_manager.conftest import BOSS
from tests.test_conversation_manager.test_actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    wait_for_actor_completion,
)
from unity.conversation_manager.events import SMSReceived

pytestmark = [pytest.mark.integration, pytest.mark.codeact, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_clarification_direct_handle_smoke(initialized_cm_codeact):
    """
    Smoke: CM → CodeActActor clarification contract works end-to-end.

    We do NOT rely on any manager prompt behavior. Instead, we verify the public
    ActorHandle clarification surface that CM relies on:
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
