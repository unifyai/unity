import pytest

from unity.contact_manager.contact_manager import ContactManager
from unity.events.event_bus import EventBus
from tests.helpers import _handle_project


@pytest.mark.unit
@_handle_project
def test_create_contact():
    eb = EventBus()
    eb.register_event_types(["Messages", "MessageExchangeSummaries"])
    contact_manager = ContactManager(eb)
    contact_manager._create_contact(
        first_name="Dan",
        description="A bit of a loser",
    )
    contacts = contact_manager._search_contacts()
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 0,
        "first_name": "Dan",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
        "description": "A bit of a loser",
    }


@pytest.mark.unit
@_handle_project
def test_update_contact():
    eb = EventBus()
    eb.register_event_types(["Messages", "MessageExchangeSummaries"])
    contact_manager = ContactManager(eb)

    # create
    contact_manager._create_contact(
        first_name="Dan",
    )

    # check
    contacts = contact_manager._search_contacts()
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 0,
        "first_name": "Dan",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
        "description": None,
    }

    # update
    contact_manager._update_contact(
        contact_id=0,
        first_name="Daniel",
        description="He's alright",
    )

    # check
    contacts = contact_manager._search_contacts()
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 0,
        "first_name": "Daniel",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
        "description": "He's alright",
    }


@pytest.mark.unit
@_handle_project
def test_create_contacts():
    eb = EventBus()
    eb.register_event_types(["Messages", "MessageExchangeSummaries"])
    contact_manager = ContactManager(eb)

    # first
    contact_manager._create_contact(
        first_name="Dan",
    )
    contacts = contact_manager._search_contacts()
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 0,
        "first_name": "Dan",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
        "description": None,
    }

    # second
    contact_manager._create_contact(
        first_name="Tom",
    )
    contacts = contact_manager._search_contacts()
    assert len(contacts) == 2
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 1,
        "first_name": "Tom",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
        "description": None,
    }


@pytest.mark.unit
@_handle_project
def test_search_contacts():
    eb = EventBus()
    eb.register_event_types(["Messages", "MessageExchangeSummaries"])
    contact_manager = ContactManager(eb)
    contact_manager._create_contact(
        first_name="Dan",
    )
