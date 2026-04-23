"""``Contact.assistant_id`` and ``Contact.authoring_assistant_id`` semantics.

``assistant_id`` flags a shared row as representing a Hive-member body;
``authoring_assistant_id`` is a write-once stamp recorded at create
time. The update surface must never rewrite the authoring stamp.
"""

from __future__ import annotations

from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.types.contact import Contact, ContactMembership
from tests.helpers import _handle_project


def test_contact_model_carries_both_new_fields():
    """The shared ``Contact`` model declares ``assistant_id`` and ``authoring_assistant_id``."""
    fields = set(Contact.model_fields.keys())
    assert "assistant_id" in fields
    assert "authoring_assistant_id" in fields


def test_should_respond_and_response_policy_live_on_overlay():
    """Response-related fields live on ``ContactMembership``, not ``Contact``.

    The shared row keeps pure identity and audit columns; body-local
    response policy lives on the overlay. This test pins the shape so
    the split does not silently regress.
    """
    contact_fields = set(Contact.model_fields.keys())
    assert "should_respond" not in contact_fields
    assert "response_policy" not in contact_fields

    overlay_fields = set(ContactMembership.model_fields.keys())
    assert "should_respond" in overlay_fields
    assert "response_policy" in overlay_fields
    assert "relationship" in overlay_fields
    assert "can_edit" in overlay_fields


def test_shorthand_map_registers_new_fields():
    """Both new fields are exposed via the shorthand alias registry."""
    assert Contact.SHORTHAND_MAP.get("assistant_id") == "aid"
    assert Contact.SHORTHAND_MAP.get("authoring_assistant_id") == "auth_aid"


@_handle_project
def test_assistant_id_roundtrips_through_create():
    """Creating a contact with ``assistant_id`` set round-trips the value."""
    cm = ContactManager()

    result = cm._create_contact(
        first_name="Buddy",
        email_address="buddy@hive.example.com",
        assistant_id=9001,
    )
    cid = result["details"]["contact_id"]

    rows = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"]
    assert len(rows) == 1
    assert rows[0].assistant_id == 9001


@_handle_project
def test_assistant_id_defaults_to_none_when_not_supplied():
    """A contact without ``assistant_id`` leaves the column NULL.

    Non-body contacts (humans, external systems) have a ``NULL``
    ``assistant_id`` — the only signal readers use to tell a Hive-mate
    body from every other kind of contact.
    """
    cm = ContactManager()

    result = cm._create_contact(
        first_name="Human",
        email_address="human@example.com",
    )
    cid = result["details"]["contact_id"]

    rows = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"]
    assert len(rows) == 1
    assert rows[0].assistant_id is None


@_handle_project
def test_authoring_assistant_id_is_stamped_at_create():
    """The authoring stamp is set to the session body at create time."""
    from unity.session_details import SESSION_DETAILS

    cm = ContactManager()

    SESSION_DETAILS.populate(agent_id=4242, user_id="author-user")

    result = cm._create_contact(
        first_name="Authored",
        email_address="authored@example.com",
    )
    cid = result["details"]["contact_id"]

    rows = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"]
    assert len(rows) == 1
    assert rows[0].authoring_assistant_id == 4242


@_handle_project
def test_authoring_assistant_id_is_not_rewritten_by_update():
    """An update never touches ``authoring_assistant_id`` — write-once audit.

    ``update_contact`` strips ``authoring_assistant_id`` from the update
    allowlist, so a call that only supplies that field is a no-op on the
    shared row. A parallel update that does supply allowed fields rewrites
    them but leaves the authoring stamp intact.
    """
    import pytest
    from unity.session_details import SESSION_DETAILS

    cm = ContactManager()
    SESSION_DETAILS.populate(agent_id=4242, user_id="author-user")

    result = cm._create_contact(
        first_name="Authored",
        email_address="authored@example.com",
    )
    cid = result["details"]["contact_id"]

    SESSION_DETAILS.populate(agent_id=9999, user_id="other-user")

    cm.update_contact(contact_id=cid, first_name="Renamed", bio="I was renamed.")

    with pytest.raises(ValueError):
        cm.update_contact(contact_id=cid, authoring_assistant_id=9999)

    rows = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"]
    assert rows[0].first_name == "Renamed"
    assert rows[0].authoring_assistant_id == 4242
