import pytest

from tests.helpers import _handle_project, get_or_create_contact
from tests.test_conversation_manager.conftest import BOSS
from tests.test_conversation_manager.test_actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    inject_actor_result,
    run_cm_until_wait,
    wait_for_actor_completion,
    verify_contact_in_db,
    verify_task_in_db,
)
from unity.conversation_manager.events import SMSReceived, SMSSent, EmailSent


def _find_task_id_by_name(cm_driver, *, name: str) -> int:
    from unity.manager_registry import ManagerRegistry

    scheduler = ManagerRegistry.get_task_scheduler()
    store = getattr(scheduler, "_store", None)
    assert store is not None, "TaskScheduler missing _store"
    # Be tolerant of minor formatting differences (quotes, punctuation) by scanning
    # a small recent window instead of relying on exact equality.
    rows = store.get_rows(
        limit=25,
        include_fields=["task_id", "name"],
    )
    needle = name.lower()
    for r in rows or []:
        nm = str((getattr(r, "entries", None) or {}).get("name") or "")
        if needle in nm.lower():
            return int((r.entries or {}).get("task_id"))
    raise AssertionError(
        f"Expected to find a task whose name contains {name!r}, got 0 matches.",
    )


pytestmark = [pytest.mark.integration, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_find_contact_then_send_sms_smoke(initialized_cm_codeact):
    """
    Find a contact, then send an SMS.

    Contract: CM can take an actor result (contact lookup) and continue the workflow
    by emitting the correct outbound channel event.
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


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_find_contact_then_send_email_smoke(initialized_cm_codeact):
    """
    Find a contact, then send an email.

    Contract: CM can convert the actor result into an EmailSent action with the right
    recipient and body.
    """
    cm = initialized_cm_codeact

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Please email Alice to tell her: Meeting at 3pm.",
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id

    final = await wait_for_actor_completion(cm, handle_id, timeout=90)
    await inject_actor_result(cm, handle_id=handle_id, result=final, success=True)
    followup_events = await run_cm_until_wait(cm, max_steps=6)

    email_events = [e for e in followup_events if isinstance(e, EmailSent)]
    assert email_events, "Expected an EmailSent event after actor completed."
    assert "meeting" in (email_events[0].body or "").lower()
    assert email_events[0].contact.get("first_name") == "Alice"
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_summarize_file_then_create_task_smoke(
    initialized_cm_codeact,
    test_files,
):
    """
    Summarize a file, then create a follow-up task.
    """
    cm = initialized_cm_codeact
    # Use CSV rather than PDF here for determinism: CSV parsing is reliable without
    # external dependencies, while PDF text extraction can vary by environment.
    csv_path = test_files["test_data.csv"]
    task_name = "Follow up on uploaded data (integration)"

    # Step 1 (realistic): user asks for a summary of an attachment/path.
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                f"I just received a CSV at {csv_path}. Please read it and tell me how many rows it has and who is listed."
            ),
        ),
    )
    actor_event1 = get_actor_started_event(result1)
    handle_id1 = actor_event1.handle_id
    _summary = await wait_for_actor_completion(cm, handle_id1, timeout=90)
    assert_no_errors(result1)

    # Step 2 (realistic): user follows up asking to create a task based on the summary.
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                f"Great—please create a follow-up task named '{task_name}' with description "
                "'Review the uploaded data summary'."
            ),
        ),
    )

    actor_event2 = get_actor_started_event(result2)
    handle_id2 = actor_event2.handle_id
    _final = await wait_for_actor_completion(cm, handle_id2, timeout=90)

    task_id = _find_task_id_by_name(cm, name=task_name)
    verify_task_in_db(
        cm,
        task_id,
        expected_fields={"name": task_name},
    )
    assert_no_errors(result2)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
@pytest.mark.skip(
    reason=(
        "FIXME(Robust CM↔Actor E2E): CM brain can delegate only step 1 of a compound user "
        "instruction to the actor (lossy multi-step). This test must run via CM→Actor, "
        "but is currently unreliable until delegation is fixed."
    ),
)
async def test_single_message_file_then_create_task(
    initialized_cm_codeact,
    test_files,
    tmp_path,
):
    """
    Single-message multi-step: read a file, extract a token, then create a follow-up task.

    Contract: CodeActActor can execute a multi-step plan from ONE natural-language request
    and persist the requested side-effect (task creation), while actually reading the file
    (validated via a token that exists only inside the file).
    """
    # FIXME(Robust CM↔Actor E2E): Keep this test routed through CM (production path).
    # The actor-direct variant that *does* pass is intentionally left commented below
    # as guidance for what should succeed once CM delegation is fixed.
    import uuid
    from pathlib import Path

    cm = initialized_cm_codeact
    # Write a per-test plain-text file with an unguessable token.
    token = f"SINGLEMSG-TOKEN-{uuid.uuid4().hex}"
    note_path = Path(tmp_path) / f"single_message_note_{token[-8:]}.txt"
    note_path.write_text(f"Reference token: {token}\n", encoding="utf-8")

    task_name = f"Follow up on note ({token})"

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                f"I just received a small text note at {str(note_path)}. "
                "Please do all of the following steps:\n"
                "1) Read the note and extract the reference token.\n"
                "2) Create a task in my task list with the exact name I give you below.\n"
                "3) Set the task description to include the extracted token.\n"
                f"Task name: '{task_name}'.\n"
                "Reply 'DONE' only after the task is created."
            ),
        ),
    )
    handle_id = get_actor_started_event(result).handle_id
    _final = await wait_for_actor_completion(cm, handle_id, timeout=90)

    task_id = _find_task_id_by_name(cm, name=task_name)
    row = verify_task_in_db(cm, task_id, expected_fields={"name": task_name})
    desc = str(row.get("description") or "")
    assert (
        token in desc
    ), f"Expected task description to contain token {token!r}, got: {desc!r}"
    assert_no_errors(result)

    # Actor-direct variant (commented guidance):
    # import asyncio
    # actor = cm.cm.actor
    # handle = await actor.act(
    #     (
    #         f"I just received a small text note at {str(note_path)}. "
    #         "Read the note, extract the reference token, then create a task in my task list "
    #         f"with the exact name '{task_name}' and a description that includes the extracted token."
    #     ),
    #     _parent_chat_context=cm.cm.chat_history,
    # )
    # _final = await asyncio.wait_for(handle.result(), timeout=90)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
@pytest.mark.skip(
    reason=(
        "FIXME(Robust CM↔Actor E2E): CM brain can delegate only step 1 of a compound user "
        "instruction to the actor (lossy multi-step). This test must run via CM→Actor, "
        "but is currently unreliable until delegation is fixed."
    ),
)
async def test_single_message_update_contact_then_create_task(initialized_cm_codeact):
    """
    Single-message multi-step: update a contact, then create a task referencing the updated value.

    Contract: CodeActActor can chain ContactManager.update → TaskScheduler.update in a single request,
    and the final persisted state matches what was requested.
    """
    # FIXME(Robust CM↔Actor E2E): Keep this test routed through CM (production path).
    # The actor-direct variant that *does* pass is intentionally left commented below
    # as guidance for what should succeed once CM delegation is fixed.
    import os
    import uuid

    cm = initialized_cm_codeact
    uniq = os.getpid()
    email = f"sam.taylor.{uniq}@example.com"
    _ = get_or_create_contact(
        cm.cm.contact_manager,
        first_name="Sam",
        surname="Taylor",
        email_address=email,
        phone_number="+15555550001",
    )

    token = f"CONTACTTASK-{uuid.uuid4().hex}"
    new_phone = "+15555550002"
    task_name = f"Call Sam ({token})"
    task_desc = f"Call Sam at {new_phone}. Ref: {token}."

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Please do ALL of the following in order:\n"
                f"1) Update the contact with email {email} to have phone number {new_phone}.\n"
                f"2) Create a task named '{task_name}' with description '{task_desc}'.\n"
                "Reply 'DONE' only after both steps are complete."
            ),
        ),
    )
    handle_id = get_actor_started_event(result).handle_id
    _final = await wait_for_actor_completion(cm, handle_id, timeout=90)

    # Verify the contact update persisted.
    payload = cm.cm.contact_manager.filter_contacts(
        filter=f"email_address == '{email}'",
        limit=5,
    )
    contacts = payload.get("contacts") or []
    assert contacts, f"Expected to find contact for {email!r}, got: {payload}"
    c0 = contacts[0]
    contact_id = (
        int(c0.get("contact_id")) if isinstance(c0, dict) else int(c0.contact_id)
    )
    verify_contact_in_db(
        cm,
        contact_id,
        expected_fields={"email_address": email, "phone_number": new_phone},
    )

    # Verify the task creation persisted.
    task_id = _find_task_id_by_name(cm, name=task_name)
    verify_task_in_db(
        cm,
        task_id,
        expected_fields={"name": task_name, "description": task_desc},
    )
    assert_no_errors(result)

    # Actor-direct variant (commented guidance):
    # import asyncio
    # actor = cm.cm.actor
    # handle = await actor.act(
    #     (
    #         "Please do both steps in order: "
    #         f"update the contact with email {email} to phone number {new_phone}; "
    #         f"then create a task named '{task_name}' with description '{task_desc}'."
    #     ),
    #     _parent_chat_context=cm.cm.chat_history,
    # )
    # _final = await asyncio.wait_for(handle.result(), timeout=90)
