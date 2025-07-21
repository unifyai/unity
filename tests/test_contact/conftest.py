from __future__ import annotations

import asyncio
import pytest
import pytest_asyncio
from typing import List, Dict, Tuple, Any
import os
import functools

import unify
from unity.contact_manager.contact_manager import ContactManager

SCENARIO_COMMIT_HASHES: Dict[str, Any] = {}

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
        "phone_number": "4445556666",
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
        "phone_number": "7778889999",
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

    def __init__(self):
        self.cm = ContactManager()
        self._populate_id_mapping()

    def _populate_id_mapping(self):
        """Populate _ID_BY_NAME_CONTACTS by searching for existing contacts."""
        global _ID_BY_NAME_CONTACTS
        _ID_BY_NAME_CONTACTS.clear()

        def search_and_map_contact(contact_data):
            """Helper function to search for a contact and return mapping tuple."""
            if not contact_data.get("email_address"):
                return None

            existing_contacts = self.cm._search_contacts(
                filter=f"email_address == '{contact_data['email_address']}'",
            )
            if existing_contacts:
                contact_id = existing_contacts[0].contact_id
                name_key = contact_data["first_name"].lower()
                name_key = f"{name_key}_{contact_data['email_address']}"
                return (name_key, contact_id)
            return None

        # Wrap each contact_data dict in a tuple to avoid unify.map treating dict keys as kwargs
        contact_data_tuples = [(contact_data,) for contact_data in _CONTACTS_DATA]

        results = unify.map(
            search_and_map_contact,
            contact_data_tuples,
            mode="asyncio",
        )

        for result in results:
            if result is not None:
                name_key, contact_id = result
                _ID_BY_NAME_CONTACTS[name_key] = contact_id

    @classmethod
    async def create(cls) -> "ScenarioBuilderContacts":
        self = cls()
        await self._seed_contacts()
        return self

    async def _seed_contacts(self) -> None:
        """Create contacts if they don't already exist."""
        for contact_data in _CONTACTS_DATA:
            # Check if contact already exists
            if contact_data.get("email_address"):
                existing_contacts = self.cm._search_contacts(
                    filter=f"email_address == '{contact_data['email_address']}'",
                )
                if existing_contacts:
                    continue  # Contact already exists, skip

            # Create a copy to avoid modifying the original list dicts
            data_to_create = {k: v for k, v in contact_data.items() if v is not None}
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    functools.partial(self.cm._create_contact, **data_to_create),
                )
                contact_id = response["details"]["contact_id"]

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


@pytest_asyncio.fixture(scope="session")
async def contact_scenario(
    request: pytest.FixtureRequest,
) -> Tuple[ContactManager, Dict[str, int]]:
    """
    Create (and later clean up) a versioned context so that *all* tests share the
    same seeded data. Build scenario once and reuse across tests.
    """
    os.environ["TQDM_DISABLE"] = "1"

    ctx = "test_contact/Scenario"
    unify.set_context(ctx)
    existing_contexts = unify.get_contexts(prefix=ctx)
    no_reuse_scenario = request.config.getoption("--no-reuse-scenario")

    # If --no-reuse-scenario is explicitly set, override reuse_scenario
    if no_reuse_scenario:
        reuse_scenario = False
    else:
        reuse_scenario = True

    if not reuse_scenario:
        # delete all contexts to freshly create the new scenario
        def recreate_contexts(ctx):
            try:
                unify.delete_context(ctx)
                unify.create_context(ctx)
            except Exception as e:
                pass

        unify.map(
            recreate_contexts,
            list(existing_contexts.keys()),
            mode="asyncio",
        )

    if reuse_scenario and not SCENARIO_COMMIT_HASHES:

        def get_context_commits_and_rollback(ctx):
            history = unify.get_context_commits(ctx)
            if history:
                unify.rollback_context(
                    name=ctx,
                    commit_hash=history[0]["commit_hash"],
                )
                SCENARIO_COMMIT_HASHES[ctx] = history[0]["commit_hash"]

        unify.map(
            get_context_commits_and_rollback,
            list(existing_contexts.keys()),
            mode="asyncio",
        )

    # --- One-time setup (per session) ---
    builder = ScenarioBuilderContacts()
    existing_contexts = unify.get_contexts(
        prefix=ctx,
    )  # fetch newly created contexts by builder

    if not SCENARIO_COMMIT_HASHES:
        print("Seeding contact manager scenario...")
        await builder.create()

        def commit_context_and_store(ctx):
            commit_info = unify.commit_context(
                name=ctx,
                commit_message="Initial seed data for contact manager tests",
            )
            SCENARIO_COMMIT_HASHES[ctx] = commit_info["commit_hash"]

        unify.map(
            commit_context_and_store,
            list(existing_contexts.keys()),
            mode="asyncio",
        )

    unify.unset_context()
    return builder.cm, _ID_BY_NAME_CONTACTS


@pytest.fixture(scope="function")
def contact_manager_scenario(contact_scenario):
    """
    Per-test fixture that provides fresh scenario data by rolling back to
    the committed state before each test and after each test completes.
    """
    cm, id_map = contact_scenario

    def rollback_context(ctx):
        unify.rollback_context(
            name=ctx,
            commit_hash=SCENARIO_COMMIT_HASHES[ctx],
        )

    # Rollback to clean state before test
    unify.map(rollback_context, list(SCENARIO_COMMIT_HASHES.keys()), mode="asyncio")

    yield cm, id_map
