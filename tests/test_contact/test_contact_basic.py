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
    assert contact.respond_to is False
    from unity.contact_manager.contact_manager import ContactManager

    assert contact.response_policy == ContactManager.DEFAULT_RESPONSE_POLICY


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
    assert contact.respond_to is False
    from unity.contact_manager.contact_manager import ContactManager

    assert contact.response_policy == ContactManager.DEFAULT_RESPONSE_POLICY


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
    custom_policy = "Treat them courteously but share no secrets."
    contact_manager._create_contact(
        first_name="Tom",
        response_policy=custom_policy,
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
    assert tom_contact.respond_to is False
    from unity.contact_manager.contact_manager import ContactManager

    assert tom_contact.response_policy == custom_policy
    assert dan_contact.response_policy == ContactManager.DEFAULT_RESPONSE_POLICY


@pytest.mark.unit
@_handle_project
def test_search_contacts():
    contact_manager = ContactManager()
    contact_manager._create_contact(
        first_name="Dan",
    )


# ────────────────────────────────────────────────────────────────────────────
#  System contacts respond_to defaults                              #
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@_handle_project
def test_system_contacts_respond_to_true():
    """Assistant (id 0) and default user (id 1) should have respond_to == True."""
    cm = ContactManager()

    assistant = cm._search_contacts(filter="contact_id == 0")
    assert assistant, "Assistant contact (id 0) must exist"
    assert (
        assistant[0].respond_to is True
    ), "Assistant should default to respond_to=True"
    assert assistant[0].response_policy == ""

    user = cm._search_contacts(filter="contact_id == 1")
    assert user, "Default user contact (id 1) must exist"
    assert user[0].respond_to is True, "User should default to respond_to=True"
    from unity.contact_manager.contact_manager import ContactManager

    assert user[0].response_policy == ContactManager.USER_MANAGER_RESPONSE_POLICY
