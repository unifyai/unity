from __future__ import annotations

import pytest
import pytest_asyncio
from typing import List, Dict, Tuple, Any
import os

import unify
from unity.contact_manager.contact_manager import ContactManager
from unity.manager_registry import ManagerRegistry
from unity.common.context_registry import ContextRegistry
from unity.common.embed_utils import ensure_vector_column
from tests.helpers import (
    get_or_create_contact,
    rebuild_id_mapping,
    is_scenario_seeded,
    scenario_file_lock,
    mutation_test_lock,
)

# Pure columns that should have embeddings pre-computed during seeding.
# These are the text columns commonly used in semantic search queries.
# Pre-computing avoids recomputing embeddings on every test run.
_COLUMNS_TO_EMBED = [
    "first_name",
    "surname",
    "bio",
    "rolling_summary",
]

# Separate commit hash storage for read vs mutation contexts
# This ensures rollbacks in one context don't affect the other
_READ_SCENARIO_COMMIT_HASHES: Dict[str, Any] = {}
_MUTATION_SCENARIO_COMMIT_HASHES: Dict[str, Any] = {}

# Initial contact data for seeding (shared by both scenarios)
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
        "timezone": "Asia/Tokyo",
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


def _precompute_embeddings(context: str) -> None:
    """
    Pre-compute embeddings for pure columns used in semantic search.

    This avoids recomputing embeddings on every test run. The embeddings
    are computed once during initial seeding and then committed with the
    scenario data.
    """
    print(f"Pre-computing embeddings for {context}...")
    for column in _COLUMNS_TO_EMBED:
        embed_column = f"_{column}_emb"
        try:
            ensure_vector_column(
                context=context,
                embed_column=embed_column,
                source_column=column,
                derived_expr=None,
            )
            print(f"  - Created {embed_column}")
        except Exception as e:
            # Column might not have data or might already exist
            print(f"  - Skipped {embed_column}: {e}")


def _ensure_embeddings_exist(context: str) -> bool:
    """
    Check if embeddings exist for the pure columns, create them if missing.

    Returns True if any embeddings were created (scenario needs recommit).
    """
    try:
        fields = unify.get_fields(context=context)
    except Exception:
        return False

    embeddings_created = False
    for column in _COLUMNS_TO_EMBED:
        embed_column = f"_{column}_emb"
        if embed_column not in fields:
            print(f"Missing embedding {embed_column} in {context}, creating...")
            try:
                ensure_vector_column(
                    context=context,
                    embed_column=embed_column,
                    source_column=column,
                    derived_expr=None,
                )
                embeddings_created = True
                print(f"  - Created {embed_column}")
            except Exception as e:
                print(f"  - Failed to create {embed_column}: {e}")

    return embeddings_created


def _rebuild_commit_hashes(
    ctx_prefix: str,
    commit_hashes: Dict[str, Any],
) -> None:
    """Rebuild commit hashes from existing context commits."""
    existing_contexts = unify.get_contexts(prefix=ctx_prefix)
    for ctx_name in existing_contexts.keys():
        history = unify.get_context_commits(ctx_name)
        if history:
            commit_hashes[ctx_name] = history[0]["commit_hash"]


def _commit_contexts_for_rollback(
    ctx_prefix: str,
    commit_hashes: Dict[str, Any],
) -> None:
    """Commit all contexts under prefix and store hashes for rollback."""
    existing_contexts = unify.get_contexts(prefix=ctx_prefix)
    for ctx_name in existing_contexts.keys():
        commit_info = unify.commit_context(
            name=ctx_name,
            commit_message="Initial seed data for contact manager tests",
        )
        commit_hashes[ctx_name] = commit_info["commit_hash"]


def _setup_scenario(
    request: pytest.FixtureRequest,
    ctx: str,
    lock_name: str,
    commit_hashes: Dict[str, Any],
) -> Tuple[ContactManager, Dict[str, int]]:
    """
    Common setup logic for seeding a contact manager scenario.

    Creates/reuses a versioned context, seeds contacts if needed,
    and returns the manager + id mapping.

    Note: ContactManager instantiation is inside the file lock because
    ContactManager.__init__ calls _sync_required_contacts(), which creates
    system contacts (id=0, id=1). Without the lock, parallel pytest sessions
    can race and create duplicate contacts due to a TOCTOU vulnerability
    in the application-level uniqueness check.
    """
    ManagerRegistry.clear()
    ContextRegistry.clear()
    os.environ["TQDM_DISABLE"] = "1"

    overwrite_scenarios = request.config.getoption("--overwrite-scenarios")

    # If --overwrite-scenarios is set, delete existing contexts first
    if overwrite_scenarios:
        existing_contexts = unify.get_contexts(prefix=ctx)
        for ctx_name in existing_contexts.keys():
            unify.delete_context(ctx_name)

    # Set context before any operations
    unify.create_context(ctx)  # exist_ok=True by default
    unify.set_context(ctx, relative=False)

    # Use file lock to coordinate ContactManager creation and seeding.
    # ContactManager.__init__ creates system contacts (assistant id=0, user id=1)
    # via _sync_required_contacts(). This must be serialized to prevent duplicate
    # contact creation when multiple pytest sessions start in parallel.
    with scenario_file_lock(lock_name):
        cm = ContactManager()
        id_mapping: Dict[str, int] = {}

        if is_scenario_seeded(cm, _CONTACTS_DATA):
            # Scenario exists - just rebuild local state
            print(f"Scenario already seeded ({ctx}), rebuilding local state...")
            id_mapping = rebuild_id_mapping(cm, _CONTACTS_DATA)
            _rebuild_commit_hashes(ctx, commit_hashes)
            # Check if embeddings exist, create if missing (for older scenarios)
            if _ensure_embeddings_exist(cm._ctx):
                # Embeddings were created, need to recommit
                print(f"Recommitting {ctx} with new embeddings...")
                _commit_contexts_for_rollback(ctx, commit_hashes)
        else:
            # Scenario not seeded - seed it
            print(f"Seeding contact manager scenario ({ctx})...")
            id_mapping = _seed_contacts(cm)
            # Pre-compute embeddings for pure columns before committing
            # This avoids recomputing on every test run
            _precompute_embeddings(cm._ctx)
            _commit_contexts_for_rollback(ctx, commit_hashes)

    unify.unset_context()
    return cm, id_mapping


# ---------------------------------------------------------------------------
# READ-ONLY SCENARIO (for test_ask.py, test_semantic.py, etc.)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session")
async def contact_read_scenario(
    request: pytest.FixtureRequest,
) -> Tuple[ContactManager, Dict[str, int]]:
    """
    Session-scoped scenario for READ-ONLY tests.

    Uses context: tests/test_contact/ReadScenario

    Read-only tests can run fully in parallel since they only read data
    and their rollbacks don't affect mutation tests (separate context).
    """
    return _setup_scenario(
        request,
        ctx="tests/test_contact/ReadScenario",
        lock_name="cm_read_scenario",
        commit_hashes=_READ_SCENARIO_COMMIT_HASHES,
    )


@pytest.fixture(scope="function")
def contact_manager_scenario(contact_read_scenario):
    """
    Per-test fixture for tests using the read scenario (e.g., test_ask.py).

    Uses a file lock to serialize tests, ensuring the full sequence
    (rollback → run test → verify) is atomic. This prevents race conditions
    where parallel tests' rollbacks orphan each other's derived column data.

    Note: Despite being called "read scenario", these tests create derived
    columns (embeddings, composite fields) during semantic search, so they
    are not truly read-only and require serialization.
    """
    cm, id_map = contact_read_scenario

    def rollback_context(ctx):
        unify.rollback_context(
            name=ctx,
            commit_hash=_READ_SCENARIO_COMMIT_HASHES[ctx],
        )

    with mutation_test_lock("cm_read"):
        # Rollback INSIDE the lock to prevent other tests
        # from rolling back while this test is running
        ctx_names = list(_READ_SCENARIO_COMMIT_HASHES.keys())
        if ctx_names:
            unify.map(rollback_context, ctx_names, mode="asyncio")

        yield cm, id_map


# ---------------------------------------------------------------------------
# MUTATION SCENARIO (for test_update.py, etc.)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session")
async def contact_mutation_scenario(
    request: pytest.FixtureRequest,
) -> Tuple[ContactManager, Dict[str, int]]:
    """
    Session-scoped scenario for MUTATION tests.

    Uses context: tests/test_contact/MutationScenario

    Mutation tests use a separate context from read-only tests, ensuring
    that read tests' rollbacks cannot interfere with mutation operations.
    """
    return _setup_scenario(
        request,
        ctx="tests/test_contact/MutationScenario",
        lock_name="cm_mutation_scenario",
        commit_hashes=_MUTATION_SCENARIO_COMMIT_HASHES,
    )


@pytest.fixture(scope="function")
def contact_manager_mutation_scenario(contact_mutation_scenario):
    """
    Per-test fixture for tests that MUTATE contact data (create, update, delete).

    Uses a SEPARATE context from read-only tests, plus a file lock to serialize
    mutation tests among themselves. This ensures:

    1. Read tests' rollbacks cannot affect mutation tests (different context)
    2. Mutation tests don't race with each other (serialized via lock)
    3. The full sequence (rollback → mutate → verify) is atomic

    This allows running `parallel_run.sh test_contact_manager` safely:
    - Read tests run fully in parallel (their own context)
    - Mutation tests run serially (shared context + lock)
    """
    cm, id_map = contact_mutation_scenario

    def rollback_context(ctx):
        unify.rollback_context(
            name=ctx,
            commit_hash=_MUTATION_SCENARIO_COMMIT_HASHES[ctx],
        )

    with mutation_test_lock("cm_mutation"):
        # Rollback INSIDE the lock to prevent other mutation tests
        # from rolling back while this test is running
        ctx_names = list(_MUTATION_SCENARIO_COMMIT_HASHES.keys())
        if ctx_names:
            unify.map(rollback_context, ctx_names, mode="asyncio")

        yield cm, id_map


# ---------------------------------------------------------------------------
# BACKWARDS COMPATIBILITY
# ---------------------------------------------------------------------------
# Keep the old fixture name as an alias for tests that haven't been updated


@pytest_asyncio.fixture(scope="session")
async def contact_scenario(
    request: pytest.FixtureRequest,
) -> Tuple[ContactManager, Dict[str, int]]:
    """
    DEPRECATED: Use contact_read_scenario or contact_mutation_scenario instead.

    This alias exists for backwards compatibility with tests that directly
    depend on contact_scenario. It maps to the read scenario.
    """
    return _setup_scenario(
        request,
        ctx="tests/test_contact/ReadScenario",
        lock_name="cm_read_scenario",
        commit_hashes=_READ_SCENARIO_COMMIT_HASHES,
    )
