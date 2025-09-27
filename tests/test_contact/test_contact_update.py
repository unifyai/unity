from __future__ import annotations

import asyncio
import pytest
from typing import Dict, Any

from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.types.contact import Contact
from tests.helpers import _handle_project


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

    retrieved_contacts = contact_manager._filter_contacts(filter=filter_str)

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


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
async def test_update_create_new_contact(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test creating a new contact using the update method via natural language."""
    cm, _ = contact_manager_scenario
    command = (
        "Add a new contact: Eve Adams, email eve@paradise.com, "
        "phone 777-000-1111, bio 'Digital nomad and writer'."
    )

    handle = await cm.update(command)
    await handle.result()

    _programmatic_contact_check(
        cm,
        "email_address",
        "eve@paradise.com",
        {
            "first_name": "Eve",
            "surname": "Adams",
            "phone_number": "7770001111",
            "bio": "Digital nomad and writer",
        },
    )


@_handle_project
@pytest.mark.slow
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
        results = cm._filter_contacts(
            filter="email_address == 'alice.smith@example.com'",
        )
        assert results, "Alice Smith not found for test setup"
        alice_smith_id = results[0].contact_id

    command = f"Update contact ID {alice_smith_id}: change her phone to 1231231234 and add WhatsApp +11231231234."

    handle = await cm.update(command)
    await handle.result()

    _programmatic_contact_check(
        cm,
        "contact_id",
        alice_smith_id,
        {
            "phone_number": "1231231234",
            "whatsapp_number": "+11231231234",
            "email_address": "alice.smith@example.com",
        },
    )


@_handle_project
@pytest.mark.slow
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
        results = cm._filter_contacts(filter="email_address == 'goodgrief@example.org'")
        assert results, "Charlie Brown not found for test setup"
        charlie_id = results[0].contact_id

    parent_ctx = [
        {"role": "user", "content": "We were just talking about Charlie Brown."},
        {
            "role": "assistant",
            "content": "Yes, the one with email goodgrief@example.org. What about him?",
        },
    ]
    command = "Add his phone number: 555-123456."

    handle = await cm.update(
        command,
        parent_chat_context=parent_ctx,
    )
    await handle.result()

    _programmatic_contact_check(
        cm,
        "contact_id",
        charlie_id,
        {"first_name": "Charlie", "surname": "Brown", "phone_number": "555123456"},
    )


@_handle_project
@pytest.mark.slow
@pytest.mark.eval
@pytest.mark.asyncio
async def test_update_with_clarification_needed(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test update requiring clarification when multiple contacts match.

    This version does not assume a single clarification. It spins up a small
    clarification agent that answers any number of clarification requests with a
    consistent intent: we mean Alice Wonder (email alice.wonder@example.com).
    """
    cm, _ = contact_manager_scenario
    # Two "Alice" contacts exist from the fixture data.

    clar_up_q = asyncio.Queue()
    clar_down_q = asyncio.Queue()

    command = "Add surname 'Wonderland' for Alice. Call the tool `request_clarification` if there is more than one Alice."

    handle = await cm.update(
        command,
        clarification_up_q=clar_up_q,
        clarification_down_q=clar_down_q,
    )

    target_name = "Alice Wonder"
    target_email = "alice.wonder@example.com"

    async def toy_clarification_llm(clar_message: Any) -> str:
        """A minimal agent that answers clarification prompts consistently.

        It always clarifies that we mean Alice Wonder with the specified email,
        phrasing the response to be robust to different prompt styles.
        """
        # Best-effort extraction of text content
        if isinstance(clar_message, dict):
            content = clar_message.get("content") or str(clar_message)
        else:
            content = str(clar_message)
        content_lower = content.lower()

        # Prefer answering with the unique identifier requested, if hinted
        if "email" in content_lower or "e-mail" in content_lower:
            return f"The one with email {target_email}."
        if "name" in content_lower or "which alice" in content_lower:
            return f"{target_name} (email {target_email})."
        if "id" in content_lower or "identifier" in content_lower:
            # We don't assume knowledge of an internal numeric ID here; email is unique
            return f"Use the contact with email {target_email} (name {target_name})."
        # Generic fallback that should work for most clarification wordings
        return f"{target_name}, with email {target_email}."

    async def clarification_agent():
        # Respond to any number of clarification prompts until the handle completes
        while not handle.done():
            try:
                clar_msg = await asyncio.wait_for(clar_up_q.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            reply = await toy_clarification_llm(clar_msg)
            await clar_down_q.put(reply)

    agent_task = asyncio.create_task(clarification_agent())

    await handle.result()

    # Ensure agent exits
    try:
        await asyncio.wait_for(agent_task, timeout=1)
    except asyncio.TimeoutError:
        agent_task.cancel()

    _programmatic_contact_check(
        cm,
        "email_address",
        "alice.wonder@example.com",
        {"first_name": "Alice", "surname": "Wonderland"},
    )
    # Check that Alice Smith's surname wasn't changed
    alice_smith_contacts = cm._filter_contacts(
        filter="email_address == 'alice.smith@example.com'",
    )
    assert alice_smith_contacts, "Alice Smith not found post-test"
    assert alice_smith_contacts[0].surname == "Smith"


@_handle_project
@pytest.mark.slow
@pytest.mark.eval
@pytest.mark.asyncio
async def test_update_interjection_modification(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test interjecting to modify details during an update operation."""
    cm, _ = contact_manager_scenario
    command = "Create a contact for Frank Castle, email frank@punisher.net."

    handle = await cm.update(command)
    await asyncio.sleep(0.2)
    await handle.interject("Actually, also add his phone as 555-54321.")
    await handle.result()

    _programmatic_contact_check(
        cm,
        "email_address",
        "frank@punisher.net",
        {"first_name": "Frank", "surname": "Castle", "phone_number": "55554321"},
    )


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
async def test_update_stop_operation(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test stopping an update operation."""
    cm, _ = contact_manager_scenario
    handle = await cm.update(
        "Create a very detailed contact for Professor Charles Xavier, email prox@xmen.com, phone 123-PROF-X, with notes about his telepathic abilities and founder of the X-Men.",
    )
    await asyncio.sleep(0.1)
    handle.stop()
    await handle.result()
    assert handle.done()

    await asyncio.sleep(0.2)
    prof_x_search = cm._filter_contacts(filter="email_address == 'prox@xmen.com'")
    assert (
        len(prof_x_search) == 0
    ), "Contact should ideally not be created if stopped early."


@_handle_project
@pytest.mark.slow
@pytest.mark.eval
@pytest.mark.asyncio
async def test_update_add_bio(
    contact_manager_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Add or change the *bio* field on an existing contact."""
    cm, _ = contact_manager_scenario

    # Pick Bob Johnson
    bob = cm._filter_contacts(filter="first_name == 'Bob' and surname == 'Johnson'")
    assert bob, "Bob Johnson must exist for this test"
    bob_id = bob[0].contact_id

    handle = await cm.update(
        f"Add a short bio 'Long-time customer' to contact ID {bob_id}.",
    )
    await handle.result()

    _programmatic_contact_check(
        cm,
        "contact_id",
        bob_id,
        {"bio": "Long-time customer"},
    )
