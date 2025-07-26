from __future__ import annotations

import pytest

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Low-level helper: _merge_contacts                                        #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
@_handle_project
def test_merge_contacts_private():
    """Programmatically merge two contacts and verify the result."""

    cm = ContactManager()

    # Create two test contacts
    cid1 = cm._create_contact(
        first_name="John",
        surname="Doe",
        email_address="john.doe@example.com",
        phone_number="1111111111",
    )["details"]["contact_id"]

    cid2 = cm._create_contact(
        first_name="Johnny",
        surname="Roe",
        email_address="johnny.roe@example.com",
        phone_number="2222222222",
    )["details"]["contact_id"]

    # Merge contacts: keep id of *cid1* but take names from *cid2*
    outcome = cm._merge_contacts(
        contact_id_1=cid1,
        contact_id_2=cid2,
        overrides={
            "first_name": 2,
            "surname": 2,
            "email_address": 1,
            "phone_number": 1,
            "contact_id": 1,  # keep cid1 as surviving id
        },
    )

    kept_id = outcome["details"]["kept_contact_id"]
    deleted_id = outcome["details"]["deleted_contact_id"]

    # Basic sanity checks on outcome payload
    assert kept_id == cid1
    assert deleted_id == cid2

    # Verify database state
    remaining = cm._search_contacts(filter=f"contact_id == {kept_id}")
    assert len(remaining) == 1, "Merged contact should exist under kept_id"

    merged = remaining[0]
    assert merged.first_name == "Johnny"
    assert merged.surname == "Roe"
    assert merged.email_address == "john.doe@example.com"
    assert merged.phone_number == "1111111111"

    # Deleted contact must be gone
    assert (
        len(cm._search_contacts(filter=f"contact_id == {deleted_id}")) == 0
    ), "Deleted contact should be removed after merge"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Natural-language merge via update()                                      #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.eval
@pytest.mark.asyncio
@_handle_project
async def test_merge_contacts_via_update():
    """Ask the LLM (update method) to merge two contacts."""

    cm = ContactManager()

    # Two initial contacts
    cid1 = cm._create_contact(
        first_name="Louise",
        surname="Lane",
        email_address="louise.lane@example.com",
        phone_number="1231231234",
    )["details"]["contact_id"]

    cid2 = cm._create_contact(
        first_name="Lois",
        surname="Lane",
        email_address="lois.lane@example.com",
        phone_number="5555555555",
    )["details"]["contact_id"]

    # NL instruction to merge
    command = (
        f"Merge contacts with IDs {cid1} and {cid2}. "
        f"Keep the phone number and email from contact {cid1}, "
        f"but use the first name and surname from contact {cid2}. "
        f"The resulting contact should keep the ID {cid1}."
    )

    handle = await cm.update(command)
    await handle.result()

    # Surviving contact must be *cid1* with combined fields
    remaining_1 = cm._search_contacts(filter=f"contact_id == {cid1}")
    remaining_2 = cm._search_contacts(filter=f"contact_id == {cid2}")

    assert len(remaining_1) == 1, "Merged contact should remain under cid1"
    assert len(remaining_2) == 0, "cid2 should be deleted after merge"

    merged = remaining_1[0]
    assert merged.first_name == "Lois"
    assert merged.surname == "Lane"
    assert merged.email_address == "louise.lane@example.com"
    assert merged.phone_number == "1231231234"
