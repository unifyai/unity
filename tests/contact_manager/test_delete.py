from __future__ import annotations

import pytest

from unity.contact_manager.contact_manager import ContactManager

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# 1.  Low-level helper: _delete_contact                                      #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_delete_private():
    """Creating a contact programmatically and deleting it via the private helper."""
    cm = ContactManager()

    cid = cm._create_contact(first_name="Eve")["details"]["contact_id"]
    # Ensure contact exists before deletion
    assert cm.filter_contacts(filter=f"contact_id == {cid}")[
        "contacts"
    ], "Contact creation failed"

    cm._delete_contact(contact_id=cid)

    # Verify contact is gone
    assert (
        len(cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"]) == 0
    ), "Contact should be deleted via _delete_contact"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Guard against deleting system contacts (assistant / default user)     #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_delete_system_raises():
    cm = ContactManager()

    # Assistant (0) and default user (1) must not be deletable
    for sys_id in (0, 1):
        with pytest.raises(RuntimeError):
            cm._delete_contact(contact_id=sys_id)


# ────────────────────────────────────────────────────────────────────────────
# 3.  Guard against deleting is_system=True contacts (org members, etc.)     #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_delete_is_system_contact_raises():
    """Contacts with is_system=True cannot be deleted."""
    cm = ContactManager()

    # Create a contact with is_system=True (simulating org member)
    cid = cm._create_contact(
        first_name="OrgMember",
        email_address="orgmember-delete-test@test.com",
        is_system=True,
    )["details"]["contact_id"]

    # Verify it exists with is_system=True
    contacts = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"]
    assert len(contacts) == 1
    assert contacts[0].is_system is True

    # Attempt to delete should raise
    with pytest.raises(RuntimeError, match="Cannot delete system contact"):
        cm._delete_contact(contact_id=cid)


# ────────────────────────────────────────────────────────────────────────────
# 4.  Regular contacts (is_system=False) can be deleted                       #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_delete_regular_contact_succeeds():
    """Contacts without is_system=True can be deleted normally."""
    cm = ContactManager()

    # Create a regular contact (is_system defaults to False)
    cid = cm._create_contact(
        first_name="Regular",
        email_address="regular-delete-test@test.com",
    )["details"]["contact_id"]

    # Verify is_system is False
    contacts = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"]
    assert len(contacts) == 1
    assert contacts[0].is_system is False

    # Delete should succeed
    cm._delete_contact(contact_id=cid)

    # Verify deleted
    contacts = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"]
    assert len(contacts) == 0


# ────────────────────────────────────────────────────────────────────────────
# 5.  Natural-language deletion via update()                                 #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_update_delete_via_nl():
    """Ensure the LLM can route a deletion request through _delete_contact."""
    cm = ContactManager()

    # Create a disposable contact that we will delete via NL interface
    cid = cm._create_contact(first_name="Victor", surname="Van Doom")["details"][
        "contact_id"
    ]

    # Kick off NL deletion. We expect the LLM tool loop to call _delete_contact.
    handle = await cm.update(f"Please delete contact ID {cid}.")
    await handle.result()

    remaining = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"]
    assert len(remaining) == 0, "Contact should be deleted after NL update request"
