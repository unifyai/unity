import pytest

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project


@pytest.mark.unit
@_handle_project
def test_create_contact():
    contact_manager = ContactManager()
    contact_manager._create_contact(
        first_name="Dan",
        bio="A bit of a loser",
    )

    # Exclude both the built-in assistant (id 0) *and* the default user (id 1)
    user_contacts = [
        c for c in contact_manager._search_contacts() if c.contact_id not in {0, 1}
    ]

    assert len(user_contacts) == 1, "Exactly one user contact should have been created"
    contact = user_contacts[0]

    # ID should be **greater** than 1 because 0 = assistant, 1 = default user
    assert contact.contact_id > 1
    assert contact.first_name == "Dan"
    assert contact.bio == "A bit of a loser"
    # Remaining built-in fields should default to None
    assert contact.surname is None
    assert contact.email_address is None
    assert contact.phone_number is None
    assert contact.whatsapp_number is None
    assert contact.rolling_summary is None


@pytest.mark.unit
@_handle_project
def test_update_contact():
    contact_manager = ContactManager()

    # create
    contact_manager._create_contact(
        first_name="Dan",
    )

    # check (exclude assistant)
    user_contacts = [
        c for c in contact_manager._search_contacts() if c.contact_id not in (0, 1)
    ]
    assert len(user_contacts) == 1
    contact = user_contacts[0]
    assert contact.first_name == "Dan"

    # update
    contact_manager._update_contact(
        contact_id=contact.contact_id,
        first_name="Daniel",
        bio="He's alright",
    )

    user_contacts = [
        c for c in contact_manager._search_contacts() if c.contact_id not in (0, 1)
    ]
    assert len(user_contacts) == 1
    contact = user_contacts[0]
    assert contact.first_name == "Daniel"
    assert contact.bio == "He's alright"


@pytest.mark.unit
@_handle_project
def test_create_contacts():
    contact_manager = ContactManager()

    # first
    contact_manager._create_contact(
        first_name="Dan",
    )
    user_contacts = [
        c for c in contact_manager._search_contacts() if c.contact_id not in (0, 1)
    ]
    assert len(user_contacts) == 1
    contact = user_contacts[0]
    assert contact.first_name == "Dan"

    # second
    contact_manager._create_contact(
        first_name="Tom",
    )
    user_contacts = [
        c for c in contact_manager._search_contacts() if c.contact_id not in (0, 1)
    ]
    assert len(user_contacts) == 2
    tom_contact = next(c for c in user_contacts if c.first_name == "Tom")
    dan_contact = next(c for c in user_contacts if c.first_name == "Dan")

    # ensure IDs are unique and not 0
    assert tom_contact.contact_id not in {0, 1} and dan_contact.contact_id not in {0, 1}
    assert tom_contact.surname is None
    assert tom_contact.bio is None
    assert tom_contact.email_address is None
    assert tom_contact.phone_number is None
    assert tom_contact.whatsapp_number is None
    assert tom_contact.bio is None
    assert tom_contact.rolling_summary is None


@pytest.mark.unit
@_handle_project
def test_search_contacts():
    contact_manager = ContactManager()
    contact_manager._create_contact(
        first_name="Dan",
    )
