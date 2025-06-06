from __future__ import annotations

import asyncio
import pytest
from typing import List, Dict, Tuple
import os
import functools

import unify  # Assuming global conftest.py handles unify activation/stubbing
from unity.contact_manager.contact_manager import ContactManager
from unity.events.event_bus import EventBus

# Initial contact data for seeding
_CONTACTS_DATA: List[Dict[str, str | None]] = [
    {
        "first_name": "Alice",
        "surname": "Smith",
        "email_address": "alice.smith@example.com",
        "phone_number": "1112223333",
        "whatsapp_number": None,
    },
    {
        "first_name": "Bob",
        "surname": "Johnson",
        "email_address": "bobbyj@example.net",
        "phone_number": "444-555-6666",
        "whatsapp_number": "+14445556666",
    },
    {
        "first_name": "Charlie",
        "surname": "Brown",
        "email_address": "goodgrief@example.org",
        "phone_number": None,
        "whatsapp_number": None,
    },
    {
        "first_name": "Diana",
        "surname": "Prince",
        "email_address": "diana@themyscira.com",
        "phone_number": "777-888-9999",
        "whatsapp_number": "+17778889999",
    },
    {
        "first_name": "Alice",  # Another Alice for disambiguation tests
        "surname": "Wonder",
        "email_address": "alice.wonder@example.com",
        "phone_number": "1110001111",
    },
]

_ID_BY_NAME_CONTACTS: Dict[str, int] = {}


class ScenarioBuilderContacts:
    """Populates Unify with initial contacts for ContactManager testing."""

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self.cm = ContactManager(event_bus=self._event_bus, traced=False)

    @classmethod
    async def create(cls, event_bus: EventBus) -> "ScenarioBuilderContacts":
        self = cls(event_bus)
        await self._seed_contacts()
        return self

    async def _seed_contacts(self) -> None:
        global _ID_BY_NAME_CONTACTS
        _ID_BY_NAME_CONTACTS.clear()  # Clear for idempotency if fixture is somehow rerun in a session

        for contact_data in _CONTACTS_DATA:
            # Create a copy to avoid modifying the original list dicts
            data_to_create = {k: v for k, v in contact_data.items() if v is not None}
            try:
                loop = asyncio.get_event_loop()
                contact_id = await loop.run_in_executor(
                    None,
                    functools.partial(self.cm._create_contact, **data_to_create),
                )

                if contact_data.get("first_name"):
                    # Create a unique key if names are not unique, e.g., by adding email
                    name_key = contact_data["first_name"].lower()
                    if contact_data.get("email_address"):
                        name_key = f"{name_key}_{contact_data['email_address']}"
                    elif contact_data.get("surname"):
                        name_key = f"{name_key}_{contact_data['surname'].lower()}"

                    if name_key not in _ID_BY_NAME_CONTACTS:
                        _ID_BY_NAME_CONTACTS[name_key] = contact_id
                    else:
                        # If name_key already exists, append contact_id to ensure uniqueness
                        _ID_BY_NAME_CONTACTS[f"{name_key}_{contact_id}"] = contact_id

            except ValueError as e:
                print(
                    f"Warning: Could not create contact {contact_data.get('first_name')} due to: {e}",
                )


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_session_context():
    """
    Create (and later clean up) a backend context so that *all* tests share the
    same seeded data.
    """
    file_path = __file__
    ctx = "/".join(file_path.split("/tests/")[1].split("/")[:-1])
    if unify.get_contexts(prefix=ctx):
        unify.delete_context(ctx)

    with unify.Context(ctx):
        unify.set_trace_context("Traces")
        yield

    if os.environ.get("UNIFY_DELETE_CONTEXT_ON_EXIT", "false").lower() == "true":
        unify.delete_context(ctx)


@pytest.fixture(scope="session")
def contact_manager_scenario(
    setup_session_context,
    event_loop: asyncio.AbstractEventLoop,
) -> Tuple[ContactManager, Dict[str, int]]:
    """
    Seeds the backend with contacts exactly once per test session and
    provides the ContactManager instance and a name-to-ID mapping.
    """
    session_event_bus = EventBus()

    builder = event_loop.run_until_complete(
        ScenarioBuilderContacts.create(session_event_bus),
    )
    return builder.cm, _ID_BY_NAME_CONTACTS
