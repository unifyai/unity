import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.conftest import BOSS
from tests.test_conversation_manager.test_actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    inject_actor_result,
    run_cm_until_wait,
    wait_for_actor_completion,
)
from unity.conversation_manager.events import SMSReceived, SMSSent

pytestmark = [pytest.mark.integration, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_find_contact_then_send_sms_smoke(initialized_cm_codeact):
    """
    Smoke: Multi-step flow where CM uses CodeActActor to look up contact info,
    then CM sends an SMS based on the actor result.
    """
    cm = initialized_cm_codeact

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "I don't have Alice's number handy. Please find it and then send her an SMS "
                "saying: Meeting at 3pm."
            ),
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id

    final = await wait_for_actor_completion(cm, handle_id, timeout=90)
    # Ensure the CM brain can observe completion deterministically.
    await inject_actor_result(cm, handle_id=handle_id, result=final, success=True)

    # Deterministically run the CM brain until it decides to wait again.
    followup_events = await run_cm_until_wait(cm, max_steps=5)

    sms_events = [e for e in followup_events if isinstance(e, SMSSent)]
    assert (
        sms_events
    ), "Expected an SMSSent event after actor completed and CM continued."
    assert "meeting at 3pm" in (sms_events[0].content or "").lower()
    assert sms_events[0].contact.get("first_name") == "Alice"
    assert_no_errors(result)
