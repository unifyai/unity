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

    command = "Add surname 'Wonderland' for Alice. Call the tool `request_clarification` if there is more than one Alice."

    handle = await cm.update(
        command,
        clarification_up_q=clar_up_q,
        clarification_down_q=clar_down_q,
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
    # Check that Alice Smith's surname wasn't changed
    alice_smith_contacts = cm._filter_contacts(
        filter="email_address == 'alice.smith@example.com'",
    )
    assert alice_smith_contacts, "Alice Smith not found post-test"
    assert alice_smith_contacts[0].surname == "Smith"


@_handle_project
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

    with pytest.raises(asyncio.CancelledError):
        await handle.result()
    assert handle.done()

    await asyncio.sleep(0.2)
    prof_x_search = cm._filter_contacts(filter="email_address == 'prox@xmen.com'")
    assert (
        len(prof_x_search) == 0
    ), "Contact should ideally not be created if stopped early."


@_handle_project
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
