from __future__ import annotations

import asyncio
import pytest
import re
import json
from typing import List, Dict, Any
import os

import unify
from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.types.contact import Contact
from tests.assertion_helpers import assertion_failed


def _programmatic_contact_check(
    contact_manager: ContactManager,
    identifier_key: str,
    identifier_value: Any,
    expected_state: Dict[str, Any],
) -> Contact:
    """Programmatically retrieves and checks a contact's state."""
    if identifier_key == "contact_id":
        filter_str = f"contact_id == {identifier_value}"
    else:
        filter_str = f"{identifier_key} == '{identifier_value}'"

    retrieved_contacts = contact_manager._search_contacts(filter=filter_str)

    assert (
        len(retrieved_contacts) >= 1
    ), f"Expected at least 1 contact for {filter_str}, found {len(retrieved_contacts)}"
    # If multiple found, check the first one, or adapt logic if needing more specific selection
    actual_contact = retrieved_contacts[0]
    actual_contact_dict = actual_contact.model_dump(exclude_none=True)

    for key, expected_val in expected_state.items():
        assert (
            key in actual_contact_dict
        ), f"Expected key '{key}' not in contact {actual_contact_dict}"
        assert (
            actual_contact_dict[key] == expected_val
        ), f"For key '{key}', expected '{expected_val}', got '{actual_contact_dict[key]}'"
    return actual_contact


def _llm_judge_update_confirmation(
    command_description: str,
    assistant_response: str,
    reasoning_steps: List[Dict[str, Any]],
    expected_confirmation_fragment: str,
) -> None:
    """
    Uses an LLM to judge if the assistant's textual response appropriately
    confirms the outcome of the update command.
    """
    judge = unify.Unify(
        "o4-mini@openai",
        cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
        traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
    )
    system_prompt = (
        "You are a unit-test judge for a contact management assistant. "
        "The user attempted to perform an action described as: '{command_description}'. "
        "The assistant responded: '{assistant_response}'. "
        "Does the assistant's response clearly and accurately confirm that the intended action "
        "(e.g., 'created', 'updated', 'added details for') was attempted or completed, "
        "and does it mention the key information like '{expected_confirmation_fragment}'? "
        "Focus on the appropriateness of the confirmation message. "
        'Respond ONLY with valid JSON of the form {{"correct": true}} or {{"correct": false}}.'
    ).format(
        command_description=command_description,
        assistant_response=json.dumps(assistant_response),
        expected_confirmation_fragment=expected_confirmation_fragment,
    )
    judge.set_system_message(system_prompt)

    result_json = judge.generate(
        f"User command: {command_description}\nAssistant response: {assistant_response}",
    )

    try:
        # Attempt to find JSON within the response if it's not pure JSON
        match = re.search(r"\{.*\}", result_json, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            json_str = result_json

        verdict = json.loads(json_str)
        is_correct = verdict.get("correct")
    except json.JSONDecodeError:
        print(f"LLM Judge returned non-JSON: {result_json}")
        is_correct = False

    assert is_correct is True, assertion_failed(
        f"Confirmation containing '{expected_confirmation_fragment}' and acknowledging '{command_description}'",
        assistant_response,
        reasoning_steps,
        f"LLM Judge validation for update confirmation. Judge raw response: {result_json}",
    )
    print(f"LLM Judge: OK for update confirmation '{command_description}'.")


@pytest.mark.eval
@pytest.mark.asyncio
async def test_update_create_new_contact(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test creating a new contact using the update method via natural language."""
    cm, _ = contact_manager_scenario
    command = (
        "Add a new contact: Eve Adams, email eve@paradise.com, phone 777-000-1111."
    )

    handle = cm.update(command, _return_reasoning_steps=True)
    await handle.result()

    _programmatic_contact_check(
        cm,
        "email_address",
        "eve@paradise.com",
        {"first_name": "Eve", "surname": "Adams", "phone_number": "7770001111"},
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_update_existing_contact_details(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test updating an existing contact's details via natural language."""
    cm, id_map = contact_manager_scenario

    # Robustly get Alice Smith's ID
    alice_email_key = "alice_alice.smith@example.com"
    alice_smith_id = id_map.get(alice_email_key)
    if alice_smith_id is None:
        results = cm._search_contacts(
            filter="email_address == 'alice.smith@example.com'",
        )
        assert results, "Alice Smith not found for test setup"
        alice_smith_id = results[0].contact_id

    command = f"Update contact ID {alice_smith_id}: change her phone to 123-123-1234 and add WhatsApp +11231231234."
    desc = f"Update Alice Smith (ID {alice_smith_id})"
    expected_fragment = f"ID {alice_smith_id}"

    handle = cm.update(command, _return_reasoning_steps=True)
    assistant_response, reasoning_steps = await handle.result()

    _llm_judge_update_confirmation(
        desc,
        assistant_response,
        reasoning_steps,
        expected_fragment,
    )
    _programmatic_contact_check(
        cm,
        "contact_id",
        alice_smith_id,
        {
            "phone_number": "123-123-1234",
            "whatsapp_number": "+11231231234",
            "email_address": "alice.smith@example.com",
        },
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_update_with_parent_context_identification(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test update with parent context to identify the contact."""
    cm, id_map = contact_manager_scenario
    charlie_email_key = (
        "charlie_goodgrief@example.org"  # Key used in conftest for Charlie Brown
    )
    charlie_id = id_map.get(charlie_email_key)
    if charlie_id is None:
        results = cm._search_contacts(filter="email_address == 'goodgrief@example.org'")
        assert results, "Charlie Brown not found for test setup"
        charlie_id = results[0].contact_id

    parent_ctx = [
        {"role": "user", "content": "We were just talking about Charlie Brown."},
        {
            "role": "assistant",
            "content": "Yes, the one with email goodgrief@example.org. What about him?",
        },
    ]
    command = "Add his phone number: 555-PEANUTS."
    desc = "Add phone for Charlie Brown (identified by context)"
    expected_fragment = "Charlie Brown"

    handle = cm.update(
        command,
        parent_chat_context=parent_ctx,
        _return_reasoning_steps=True,
    )
    assistant_response, reasoning_steps = await handle.result()

    _llm_judge_update_confirmation(
        desc,
        assistant_response,
        reasoning_steps,
        expected_fragment,
    )
    _programmatic_contact_check(
        cm,
        "contact_id",
        charlie_id,
        {"first_name": "Charlie", "surname": "Brown", "phone_number": "555-PEANUTS"},
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_update_with_clarification_needed(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test update requiring clarification when multiple contacts match."""
    cm, _ = contact_manager_scenario
    # Two "Alice" contacts exist from the fixture data.

    clar_up_q = asyncio.Queue()
    clar_down_q = asyncio.Queue()

    command = "Add surname 'Wonderland' for Alice."

    handle = cm.update(
        command,
        clarification_up_q=clar_up_q,
        clarification_down_q=clar_down_q,
        _return_reasoning_steps=True,
    )

    await asyncio.wait_for(
        clar_up_q.get(),
        timeout=60,
    )

    await clar_down_q.put(
        "The one with email alice.wonder@example.com.",
    )  # Clarify Alice Wonder

    await handle.result()

    _programmatic_contact_check(
        cm,
        "email_address",
        "alice.wonder@example.com",
        {"first_name": "Alice", "surname": "Wonderland"},
    )
    # Optionally check that Alice Smith's surname wasn't changed
    alice_smith_contacts = cm._search_contacts(
        filter="email_address == 'alice.smith@example.com'",
    )
    assert alice_smith_contacts, "Alice Smith not found post-test"
    assert alice_smith_contacts[0].surname == "Smith"


@pytest.mark.eval
@pytest.mark.asyncio
async def test_update_interjection_modification(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test interjecting to modify details during an update operation."""
    cm, _ = contact_manager_scenario
    command = "Create a contact for Frank P. Castle, email frank@punisher.net."
    desc = "Create Frank Castle, then interject phone"
    expected_fragment = "Frank P. Castle"

    handle = cm.update(command, _return_reasoning_steps=True)
    await asyncio.sleep(0.2)
    await handle.interject("Actually, also add his phone as 555-SKULL.")
    assistant_response, reasoning_steps = await handle.result()

    _llm_judge_update_confirmation(
        desc,
        assistant_response,
        reasoning_steps,
        expected_fragment,
    )
    _programmatic_contact_check(
        cm,
        "email_address",
        "frank@punisher.net",
        {"first_name": "Frank P.", "surname": "Castle", "phone_number": "555-SKULL"},
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_update_stop_operation(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test stopping an update operation."""
    cm, _ = contact_manager_scenario
    handle = cm.update(
        "Create a very detailed contact for Professor Charles Xavier, email prox@xmen.com, phone 123-PROF-X, with notes about his telepathic abilities and founder of the X-Men.",
    )
    await asyncio.sleep(0.1)
    handle.stop()

    with pytest.raises(asyncio.CancelledError):
        await handle.result()
    assert handle.done()

    await asyncio.sleep(0.2)
    prof_x_search = cm._search_contacts(filter="email_address == 'prox@xmen.com'")
    assert (
        len(prof_x_search) == 0
    ), "Contact should ideally not be created if stopped early."
