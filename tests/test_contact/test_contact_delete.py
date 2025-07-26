from __future__ import annotations

import pytest

from unity.contact_manager.contact_manager import ContactManager

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# 1.  Low-level helper: _delete_contact                                      #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
@_handle_project
def test_delete_contact_private():
    """Creating a contact programmatically and deleting it via the private helper."""
    cm = ContactManager()

    cid = cm._create_contact(first_name="Eve")["details"]["contact_id"]
    # Ensure contact exists before deletion
    assert cm._search_contacts(filter=f"contact_id == {cid}"), "Contact creation failed"

    cm._delete_contact(contact_id=cid)

    # Verify contact is gone
    assert (
        len(cm._search_contacts(filter=f"contact_id == {cid}")) == 0
    ), "Contact should be deleted via _delete_contact"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Guard against deleting system contacts (assistant / default user)     #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
@_handle_project
def test_delete_system_contacts_raises():
    cm = ContactManager()

    # Assistant (0) and default user (1) must not be deletable
    for sys_id in (0, 1):
        with pytest.raises(RuntimeError):
            cm._delete_contact(contact_id=sys_id)


# ────────────────────────────────────────────────────────────────────────────
# 3.  Natural-language deletion via update()                                 #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.eval
@pytest.mark.asyncio
@_handle_project
async def test_update_delete_contact_via_nl():
    """Ensure the LLM can route a deletion request through _delete_contact."""
    cm = ContactManager()

    # Create a disposable contact that we will delete via NL interface
    cid = cm._create_contact(first_name="Victor", surname="Van Doom")["details"][
        "contact_id"
    ]

    # Kick off NL deletion. We expect the LLM tool loop to call _delete_contact.
    handle = await cm.update(f"Please delete contact ID {cid}.")
    await handle.result()

    remaining = cm._search_contacts(filter=f"contact_id == {cid}")
    assert len(remaining) == 0, "Contact should be deleted after NL update request"
