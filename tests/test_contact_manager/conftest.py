from __future__ import annotations

import pytest
import pytest_asyncio
from typing import List, Dict, Tuple, Any
import os

import unify
from unity.contact_manager.contact_manager import ContactManager
from unity.manager_registry import ManagerRegistry
from unity.common.context_registry import ContextRegistry
from tests.helpers import (
    get_or_create_contact,
    rebuild_id_mapping,
    is_scenario_seeded,
    scenario_file_lock,
)

SCENARIO_COMMIT_HASHES: Dict[str, Any] = {}

# Initial contact data for seeding
_CONTACTS_DATA: List[Dict[str, str | None]] = [
    {
        "first_name": "Alice",
        "surname": "Smith",
        "email_address": "alice.smith@example.com",
        "phone_number": "1112223333",
    },
    {
        "first_name": "Bob",
        "surname": "Johnson",
        "email_address": "bobbyj@example.net",
        "phone_number": "4445556666",
    },
    {
        "first_name": "Charlie",
        "surname": "Brown",
        "email_address": "goodgrief@example.org",
        "phone_number": None,
    },
    {
        "first_name": "Diana",
        "surname": "Prince",
        "email_address": "diana@themyscira.com",
        "phone_number": "7778889999",
    },
    {
        "first_name": "Alice",  # Another Alice for disambiguation tests
        "surname": "Wonder",
        "email_address": "alice.wonder@example.com",
        "phone_number": "1110001111",
    },
]

_ID_BY_NAME_CONTACTS: Dict[str, int] = {}


def _make_name_key(contact_data: Dict[str, Any]) -> str:
    """Generate a unique key for contact lookup."""
    name_key = contact_data["first_name"].lower()
    if contact_data.get("email_address"):
        name_key = f"{name_key}_{contact_data['email_address']}"
    elif contact_data.get("surname"):
        name_key = f"{name_key}_{contact_data['surname'].lower()}"
    return name_key


def _seed_contacts(cm: ContactManager) -> Dict[str, int]:
    """Create contacts using race-safe idempotent helper. Returns id mapping."""
    id_mapping: Dict[str, int] = {}
    for contact_data in _CONTACTS_DATA:
        email = contact_data.get("email_address")
        if email:
            # Use race-safe helper that handles parallel creation
            data_to_create = {k: v for k, v in contact_data.items() if v is not None}
            contact_id = get_or_create_contact(cm, **data_to_create)
            name_key = _make_name_key(contact_data)
            id_mapping[name_key] = contact_id
    return id_mapping


def _rebuild_commit_hashes(ctx_prefix: str) -> None:
    """Rebuild SCENARIO_COMMIT_HASHES from existing context commits."""
    existing_contexts = unify.get_contexts(prefix=ctx_prefix)
    for ctx_name in existing_contexts.keys():
        history = unify.get_context_commits(ctx_name)
        if history:
            SCENARIO_COMMIT_HASHES[ctx_name] = history[0]["commit_hash"]


def _commit_contexts_for_rollback(ctx_prefix: str) -> None:
    """Commit all contexts under prefix and store hashes for rollback."""
    existing_contexts = unify.get_contexts(prefix=ctx_prefix)
    for ctx_name in existing_contexts.keys():
        commit_info = unify.commit_context(
            name=ctx_name,
            commit_message="Initial seed data for contact manager tests",
        )
        SCENARIO_COMMIT_HASHES[ctx_name] = commit_info["commit_hash"]


@pytest_asyncio.fixture(scope="session")
async def contact_scenario(
    request: pytest.FixtureRequest,
) -> Tuple[ContactManager, Dict[str, int]]:
    """
    Create (and later clean up) a versioned context so that *all* tests share the
    same seeded data. Build scenario once and reuse across tests.

    Uses file lock to coordinate parallel test processes - only one process
    seeds while others wait, then all rebuild local state from shared data.
    """
    ManagerRegistry.clear()
    ContextRegistry.clear()
    os.environ["TQDM_DISABLE"] = "1"

    ctx = "tests/test_contact/Scenario"
    no_reuse_scenario = request.config.getoption("--no-reuse-scenario")

    # If --no-reuse-scenario is explicitly set, delete existing contexts
    if no_reuse_scenario:
        existing_contexts = unify.get_contexts(prefix=ctx)
        for ctx_name in existing_contexts.keys():
            unify.delete_context(ctx_name)

    # Set context before any operations
    unify.create_context(ctx)  # exist_ok=True by default
    unify.set_context(ctx, relative=False)

    # Create manager
    cm = ContactManager()

    # Use file lock to coordinate seeding across parallel processes
    with scenario_file_lock("cm_scenario"):
        if is_scenario_seeded(cm, _CONTACTS_DATA):
            # Scenario exists - just rebuild local state
            print("Scenario already seeded, rebuilding local state...")
            ids = rebuild_id_mapping(cm, _CONTACTS_DATA)
            _ID_BY_NAME_CONTACTS.update(ids)
            _rebuild_commit_hashes(ctx)
        else:
            # Scenario not seeded - seed it
            print("Seeding contact manager scenario...")
            ids = _seed_contacts(cm)
            _ID_BY_NAME_CONTACTS.update(ids)
            _commit_contexts_for_rollback(ctx)

    unify.unset_context()
    return cm, dict(_ID_BY_NAME_CONTACTS)


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
    ctx_names = list(SCENARIO_COMMIT_HASHES.keys())
    if ctx_names:
        unify.map(rollback_context, ctx_names, mode="asyncio")

    yield cm, id_map
