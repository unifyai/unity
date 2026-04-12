from __future__ import annotations

import asyncio
import pytest
from typing import Dict, Any

from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.types.contact import Contact
from unity.blacklist_manager.blacklist_manager import BlackListManager
from unity.conversation_manager.cm_types import Medium
from tests.helpers import _handle_project
from tests.async_helpers import _wait_for_next_assistant_response_event

# All tests in this file exercise end-to-end LLM reasoning for contact mutations
pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


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

    retrieved_contacts = contact_manager.filter_contacts(filter=filter_str)["contacts"]

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
@pytest.mark.asyncio
@pytest.mark.parametrize("identify_by", ["name", "email", "id"])
async def test_selects_move_to_blacklist(identify_by: str):
    """
    Verify that ContactManager.update routes to the _move_to_blacklist tool when asked in English,
    identifying the target contact by name, by contact detail, or by id.
    """
    cm = ContactManager()
    blm = BlackListManager()
    blm.clear()

    # Create a single unambiguous contact with multiple details
    cm._create_contact(
        first_name="Zed",
        surname="Quill",
        bio="Temporary test contact",
        email_address="zed.quill@example.org",
        phone_number="+15550100200",
    )
    # Get the created contact (exclude system contacts 0 and 1)
    created = [
        c for c in cm.filter_contacts()["contacts"] if c.contact_id not in (0, 1)
    ][0]

    reason = "policy violation"
    if identify_by == "name":
        directive = f"Please move {created.first_name} {created.surname} to the blacklist due to {reason}."
    elif identify_by == "email":
        directive = f"Blacklist the contact with email {created.email_address} because of {reason}."
    else:  # "id"
        directive = f"Blacklist contact ID {created.contact_id} due to {reason}."

    handle = await cm.update(directive)
    await handle.result()

    # Validate that blacklist entries exist for all present details and correct mediums
    entries = blm.filter_blacklist()["entries"]
    # Expect: EMAIL + (SMS_MESSAGE, PHONE_CALL) = 3 entries
    assert len(entries) == 3
    mediums = {e.medium for e in entries}
    details = {e.contact_detail for e in entries}
    assert Medium.EMAIL in mediums
    assert Medium.SMS_MESSAGE in mediums
    assert Medium.PHONE_CALL in mediums
    assert "zed.quill@example.org" in details
    assert "+15550100200" in details
    # Contact should no longer exist in Contacts
    remaining = cm.filter_contacts(filter=f"contact_id == {created.contact_id}")[
        "contacts"
    ]
    assert len(remaining) == 0


@_handle_project
@pytest.mark.asyncio
async def test_create_new(
    contact_manager_mutation_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test creating a new contact using the update method via natural language."""
    cm, _ = contact_manager_mutation_scenario
    command = (
        "Add a new contact: Eve Adams, email eve@paradise.com, "
        "phone 7770001111, bio 'Digital nomad and writer'."
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
@pytest.mark.asyncio
async def test_existing_details(
    contact_manager_mutation_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test updating an existing contact's details via natural language."""
    cm, id_map = contact_manager_mutation_scenario

    # Robustly get Alice Smith's ID
    alice_email_key = "alice_alice.smith@example.com"
    alice_smith_id = id_map.get(alice_email_key)
    if alice_smith_id is None:
        results = cm.filter_contacts(
            filter="email_address == 'alice.smith@example.com'",
        )["contacts"]
        assert results, "Alice Smith not found for test setup"
        alice_smith_id = results[0].contact_id

    command = f"Update contact ID {alice_smith_id}: change her phone to 1231231234."

    handle = await cm.update(command)
    await handle.result()

    _programmatic_contact_check(
        cm,
        "contact_id",
        alice_smith_id,
        {
            "phone_number": "1231231234",
            "email_address": "alice.smith@example.com",
        },
    )


@_handle_project
@pytest.mark.slow
@pytest.mark.asyncio
async def test_with_parent_context(
    contact_manager_mutation_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test update with parent context to identify the contact."""
    cm, id_map = contact_manager_mutation_scenario
    charlie_email_key = (
        "charlie_goodgrief@example.org"  # Key used in conftest for Charlie Brown
    )
    charlie_id = id_map.get(charlie_email_key)
    if charlie_id is None:
        results = cm.filter_contacts(
            filter="email_address == 'goodgrief@example.org'",
        )["contacts"]
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
        _parent_chat_context=parent_ctx,
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
@pytest.mark.asyncio
async def test_with_clarification(
    contact_manager_mutation_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test update requiring clarification when multiple contacts match.

    This version does not assume a single clarification. It spins up a small
    clarification agent that answers any number of clarification requests with a
    consistent intent: we mean Alice Wonder (email alice.wonder@example.com).
    """
    cm, _ = contact_manager_mutation_scenario
    # Two "Alice" contacts exist from the fixture data.

    clar_up_q = asyncio.Queue()
    clar_down_q = asyncio.Queue()

    command = "Add surname 'Wonderland' for Alice. Call the tool `request_clarification` if there is more than one Alice."

    handle = await cm.update(
        command,
        _clarification_up_q=clar_up_q,
        _clarification_down_q=clar_down_q,
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
    alice_smith_contacts = cm.filter_contacts(
        filter="email_address == 'alice.smith@example.com'",
    )["contacts"]
    assert alice_smith_contacts, "Alice Smith not found post-test"
    assert alice_smith_contacts[0].surname == "Smith"


@_handle_project
@pytest.mark.slow
@pytest.mark.asyncio
async def test_interjection(
    contact_manager_mutation_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test interjecting to modify details during an update operation."""
    cm, _ = contact_manager_mutation_scenario
    command = "Create a contact for Frank Castle, email frank@punisher.net."

    handle = await cm.update(command)
    await _wait_for_next_assistant_response_event(handle._client)
    await handle.interject("Actually, also add his phone as 555-54321.")
    await handle.result()

    _programmatic_contact_check(
        cm,
        "email_address",
        "frank@punisher.net",
        {"first_name": "Frank", "surname": "Castle", "phone_number": "55554321"},
    )


@_handle_project
@pytest.mark.asyncio
async def test_stop(
    contact_manager_mutation_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Test stopping an update operation."""
    cm, _ = contact_manager_mutation_scenario
    handle = await cm.update(
        "Create a very detailed contact for Professor Charles Xavier, email prox@xmen.com, phone 123-PROF-X, with notes about his telepathic abilities and founder of the X-Men.",
    )
    await asyncio.sleep(0.1)
    await handle.stop()
    await handle.result()
    assert handle.done()

    await asyncio.sleep(0.2)
    prof_x_search = cm.filter_contacts(filter="email_address == 'prox@xmen.com'")[
        "contacts"
    ]
    assert (
        len(prof_x_search) == 0
    ), "Contact should ideally not be created if stopped early."


@_handle_project
@pytest.mark.slow
@pytest.mark.asyncio
async def test_add_bio(
    contact_manager_mutation_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Add or change the *bio* field on an existing contact."""
    cm, _ = contact_manager_mutation_scenario

    # Pick Bob Johnson
    bob = cm.filter_contacts(filter="first_name == 'Bob' and surname == 'Johnson'")[
        "contacts"
    ]
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


@_handle_project
@pytest.mark.slow
@pytest.mark.asyncio
async def test_set_timezone_hint(
    contact_manager_mutation_scenario: tuple[ContactManager, Dict[str, int]],
):
    """Ensure the assistant can set the timezone field based on a location hint.

    We use a stable, non-DST example (Mumbai, India → Asia/Kolkata) to avoid ambiguity.
    """
    cm, _ = contact_manager_mutation_scenario

    # Find Diana Prince
    rows = cm.filter_contacts(filter="email_address == 'diana@themyscira.com'")[
        "contacts"
    ]
    assert rows, "Diana Prince must exist for this test"
    diana_id = rows[0].contact_id

    # Ask the assistant to ensure the timezone is logged correctly
    request_text = (
        "Diana Prince lives in Mumbai, India. Can you make sure we've logged "
        "her timezone correctly?"
    )
    handle = await cm.update(request_text)
    await handle.result()

    # Verify timezone has been set to Asia/Kolkata (Mumbai's IANA timezone)
    updated = cm.filter_contacts(filter=f"contact_id == {diana_id}")["contacts"][0]
    assert updated.timezone == "Asia/Kolkata"


@_handle_project
@pytest.mark.asyncio
async def test_nameless_service_contact_preserves_no_name(
    contact_manager_mutation_scenario: tuple[ContactManager, Dict[str, int]],
):
    """A service/org contact should keep first_name and surname as None even
    when a named representative appears in a transcript.

    Scenario: a support-line contact exists with only a phone number and a bio
    describing it as "Acme Corp billing support line".  A transcript excerpt
    shows someone named "Sarah" answering the call.  The ContactManager should
    update details about the call but must NOT populate the contact's name
    fields with "Sarah" — she is a transient representative, not the contact's
    identity.
    """
    cm, _ = contact_manager_mutation_scenario

    # 1. Seed a nameless service contact
    outcome = cm._create_contact(
        phone_number="8005550199",
        bio="Acme Corp billing support line",
    )
    service_contact_id = outcome["details"]["contact_id"]

    # 2. Provide a transcript-like parent context where a rep introduces herself
    parent_ctx = [
        {
            "role": "user",
            "content": (
                "I just called the Acme Corp billing support line (8005550199). "
                "Someone named Sarah from the billing department answered and "
                "confirmed our next invoice is due March 15. She said to email "
                "billing@acme.com for follow-ups."
            ),
        },
    ]
    command = (
        "Update the Acme billing support contact with any useful new details "
        "from this transcript."
    )

    handle = await cm.update(command, _parent_chat_context=parent_ctx)
    await handle.result()

    # 3. Verify: name fields must still be None (Sarah is a transient rep)
    contacts = cm.filter_contacts(
        filter=f"contact_id == {service_contact_id}",
    )["contacts"]
    assert len(contacts) == 1
    contact = contacts[0]
    assert (
        contact.first_name is None
    ), f"first_name should be None for a service contact, got '{contact.first_name}'"
    assert (
        contact.surname is None
    ), f"surname should be None for a service contact, got '{contact.surname}'"
