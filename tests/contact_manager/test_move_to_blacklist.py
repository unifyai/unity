import pytest

from unity.contact_manager.contact_manager import ContactManager
from unity.blacklist_manager.blacklist_manager import BlackListManager
from unity.conversation_manager.cm_types import Medium
from tests.helpers import _handle_project


def _get_user_contact(cm: ContactManager):
    # Exclude system contacts (0 = assistant, 1 = default user)
    users = [c for c in cm.filter_contacts()["contacts"] if c.contact_id not in (0, 1)]
    assert len(users) == 1, f"Expected exactly one user contact, got {len(users)}"
    return users[0]


@_handle_project
def test_email_only():
    cm = ContactManager()
    blm = BlackListManager()
    blm.clear()

    cm._create_contact(
        first_name="Alice",
        surname="Anderson",
        bio="Trusted partner at Example Co.",
        email_address="alice@example.com",
    )
    contact = _get_user_contact(cm)

    reason = "security policy"
    out = cm._move_to_blacklist(contact_id=contact.contact_id, reason=reason)
    assert out["outcome"] == "contact details moved to blacklist"
    assert len(out["details"]["blacklist_ids"]) == 1

    res = blm.filter_blacklist()
    entries = res["entries"]
    assert len(entries) == 1
    e = entries[0]
    assert e.medium == Medium.EMAIL
    assert e.contact_detail == "alice@example.com"
    expected_reason = "Alice, Anderson, Trusted partner at Example Co., moved to blacklist due to security policy"
    assert e.reason == expected_reason
    # Contact should no longer exist (only system contacts remain)
    remaining_users = [
        c for c in cm.filter_contacts()["contacts"] if c.contact_id not in (0, 1)
    ]
    assert len(remaining_users) == 0


@_handle_project
def test_phone_only():
    cm = ContactManager()
    blm = BlackListManager()
    blm.clear()

    cm._create_contact(
        phone_number="+15551234567",
    )
    contact = _get_user_contact(cm)

    reason = "spam calls"
    out = cm._move_to_blacklist(contact_id=contact.contact_id, reason=reason)
    assert out["outcome"] == "contact details moved to blacklist"
    # Two entries expected for phone: SMS and PHONE_CALL
    assert len(out["details"]["blacklist_ids"]) == 2

    entries = blm.filter_blacklist()["entries"]
    assert len(entries) == 2
    mediums = {e.medium for e in entries}
    details = {e.contact_detail for e in entries}
    assert mediums == {Medium.SMS_MESSAGE, Medium.PHONE_CALL}
    assert details == {"+15551234567"}
    # Reason should not have stray commas when no name/bio
    for e in entries:
        assert e.reason == "moved to blacklist due to spam calls"
    # Contact should no longer exist
    remaining_users = [
        c for c in cm.filter_contacts()["contacts"] if c.contact_id not in (0, 1)
    ]
    assert len(remaining_users) == 0


@_handle_project
def test_phone_and_email():
    cm = ContactManager()
    blm = BlackListManager()
    blm.clear()

    cm._create_contact(
        first_name="Bob",
        email_address="bob@example.com",
        phone_number="+441234567890",
    )
    contact = _get_user_contact(cm)

    reason = "abuse report"
    out = cm._move_to_blacklist(contact_id=contact.contact_id, reason=reason)
    assert out["outcome"] == "contact details moved to blacklist"
    # EMAIL + SMS_MESSAGE + PHONE_CALL
    assert len(out["details"]["blacklist_ids"]) == 3

    entries = blm.filter_blacklist()["entries"]
    mediums = sorted([e.medium for e in entries], key=lambda m: m.value)
    details = {e.contact_detail for e in entries}
    assert details == {"bob@example.com", "+441234567890"}
    assert set(mediums) == {Medium.EMAIL, Medium.SMS_MESSAGE, Medium.PHONE_CALL}
    expected_reason = "Bob, moved to blacklist due to abuse report"
    for e in entries:
        assert e.reason == expected_reason
    # Contact should no longer exist
    remaining_users = [
        c for c in cm.filter_contacts()["contacts"] if c.contact_id not in (0, 1)
    ]
    assert len(remaining_users) == 0


@_handle_project
def test_no_details():
    cm = ContactManager()
    blm = BlackListManager()
    blm.clear()

    cm._create_contact(first_name="Charlie")
    contact = _get_user_contact(cm)

    out = cm._move_to_blacklist(contact_id=contact.contact_id, reason="policy")
    assert out["outcome"] == "no contact details to blacklist"
    assert out["details"]["blacklist_ids"] == []
    assert blm.filter_blacklist()["entries"] == []
    # Contact should no longer exist
    remaining_users = [
        c for c in cm.filter_contacts()["contacts"] if c.contact_id not in (0, 1)
    ]
    assert len(remaining_users) == 0


@_handle_project
def test_idempotent():
    cm = ContactManager()
    blm = BlackListManager()
    blm.clear()

    cm._create_contact(
        email_address="dupe@example.com",
        phone_number="+19998887777",
    )
    contact = _get_user_contact(cm)

    cm._move_to_blacklist(contact_id=contact.contact_id, reason="dupe test")
    first = blm.filter_blacklist()["entries"]
    # Expect 3 entries: EMAIL + (SMS, PHONE_CALL) for phone
    assert len(first) == 3

    # Contact should be deleted; running again should raise since contact doesn't exist
    with pytest.raises(Exception):
        cm._move_to_blacklist(contact_id=contact.contact_id, reason="dupe test")
