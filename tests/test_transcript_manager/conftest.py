"""
Global fixtures & shared data usable from any test module.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from datetime import timedelta
from typing import List, Dict, Any, Tuple
import pytest
import pytest_asyncio
import os
import unify

from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from unity.manager_registry import ManagerRegistry
from unity.common.context_registry import ContextRegistry
from tests.helpers import (
    get_or_create_contact,
    rebuild_id_mapping,
    is_scenario_seeded,
    scenario_file_lock,
    mutation_test_lock,
)
from unity.common.embed_utils import ensure_vector_column

SCENARIO_COMMIT_HASHES: Dict[str, Any] = {}

# Pure columns that should have embeddings pre-computed during seeding.
# These are the text columns commonly used in semantic search queries.
# Pre-computing avoids recomputing embeddings on every test run.
_COLUMNS_TO_EMBED = [
    "content",  # Main message text column used in semantic search
]


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


# --------------------------------------------------------------------------- #
#  CONTACTS (same as before)                                                  #
# --------------------------------------------------------------------------- #

_CONTACTS: List[dict] = [
    dict(  # id = 0
        first_name="Carlos",
        surname="Diaz",
        email_address="carlos.diaz@example.com",
        phone_number="+14155550000",
    ),
    dict(  # id = 1
        first_name="Dan",
        surname="Turner",
        email_address="dan.turner@example.com",
        phone_number="+447700900001",
    ),
    dict(  # id = 2
        first_name="Julia",
        surname="Nguyen",
        email_address="julia.nguyen@example.com",
        phone_number="+447700900002",
    ),
    dict(  # id = 3
        first_name="Jimmy",
        surname="O'Brien",
        email_address="jimmy.obrien@example.com",
        phone_number="+61240011000",
    ),
    dict(  # id = 4
        first_name="Anne",
        surname="Fischer",
        email_address="anne.fischer@example.com",
        phone_number="+49891234567",
    ),
]

_ID_BY_NAME: dict[str, int] = {}  # filled during seeding


# --------------------------------------------------------------------------- #
#  SCENARIO BUILDER                                                           #
# --------------------------------------------------------------------------- #


class ScenarioBuilder:
    """Populate Unify with contacts, 6 'meaningful' exchanges + filler."""

    def __init__(self, cm: ContactManager, tm: TranscriptManager) -> None:
        self.cm = cm
        self.tm = tm
        self._message_counter = 0  # For explicit message_id assignment

    def _seed_contacts(self) -> None:
        """Create contacts using race-safe idempotent helper."""
        for c in _CONTACTS:
            email = c.get("email_address")
            if email:
                # Use race-safe helper that handles parallel creation
                contact_id = get_or_create_contact(self.cm, **c)
                _ID_BY_NAME[c["first_name"].lower()] = contact_id

    # --------------------------------------------------------------------- #
    def _seed_key_exchanges(self) -> None:
        now = datetime(2025, 4, 20, 15, 0, tzinfo=timezone.utc)

        # E0: first Dan–Julia phone call
        dan_id = _ID_BY_NAME["dan"]
        julia_id = _ID_BY_NAME["julia"]

        self._log(
            0,
            "phone_call",
            [
                (
                    dan_id,
                    julia_id,
                    now,
                    "Hi Julia, it's Dan. Quick check-in about Q2 metrics.",
                ),
                (
                    julia_id,
                    dan_id,
                    now + timedelta(seconds=30),
                    "Sure Dan, ready when you are.",
                ),
            ],
        )

        # E1: *last* Dan–Julia phone call (later date)
        later = datetime(2025, 4, 26, 9, 30, tzinfo=timezone.utc)
        self._log(
            1,
            "phone_call",
            [
                (
                    dan_id,
                    julia_id,
                    later,
                    "Morning Julia – finalising the London event agenda today.",
                ),
                (
                    julia_id,
                    dan_id,
                    later + timedelta(seconds=45),
                    "Great. Let's confirm the speaker list and coffee budget.",
                ),
            ],
        )

        # E2: Carlos interest e-mail
        carlos_id = _ID_BY_NAME["carlos"]
        t_email = datetime(2025, 4, 21, 12, 0, tzinfo=timezone.utc)
        self._log(
            2,
            "email",
            [
                (
                    carlos_id,
                    dan_id,
                    t_email,
                    "Subject: Stapler bulk order\n\n"
                    "Hi Dan,\nI'm **interested in buying 200 units** of "
                    "your new stapler. Can you quote?\n\nThanks,\nCarlos",
                ),
                (
                    dan_id,
                    carlos_id,
                    t_email + timedelta(hours=2),
                    "Hi Carlos — sure, $4.50 per unit. See attached PDF.",
                ),
            ],
        )

        # E3: Jimmy holiday SMS
        jimmy_id = _ID_BY_NAME["jimmy"]
        t_holiday = datetime(2025, 4, 22, 18, 10, tzinfo=timezone.utc)
        self._log(
            3,
            "sms_message",
            [
                (
                    jimmy_id,
                    dan_id,
                    t_holiday,
                    "Heads-up Dan, I'll be **on holiday from 2025-05-15** "
                    "to 2025-05-30. Ping me before that if urgent.",
                ),
            ],
        )

        # E4: Anne passport excuse (SMS)
        anne_id = _ID_BY_NAME["anne"]
        t_excuse = datetime(2025, 4, 23, 9, 0, tzinfo=timezone.utc)
        self._log(
            4,
            "sms_message",
            [
                (
                    anne_id,
                    dan_id,
                    t_excuse,
                    "Sorry Dan, I *can't join the Berlin trip because my "
                    "passport expired* last week.",
                ),
            ],
        )

        # E5: Dan–Julia "basketball" phone call (latest Dan-Julia phone call)
        t_basketball = datetime(2025, 5, 20, 18, 0, tzinfo=timezone.utc)
        self._log(
            5,
            "phone_call",
            [
                (
                    dan_id,
                    julia_id,
                    t_basketball,
                    "Hey Julia, did you catch the basketball game last night? "
                    "The Lakers vs Celtics final was incredible!",
                ),
                (
                    julia_id,
                    dan_id,
                    t_basketball + timedelta(seconds=30),
                    "Absolutely – that last-minute three-pointer was unbelievable! "
                    "We should watch the next game together.",
                ),
            ],
        )

        # E6: Dan–Julia "holiday planning" email (for clarification test)
        t_holiday_email = datetime(2025, 5, 25, 20, 0, tzinfo=timezone.utc)
        self._log(
            6,
            "email",
            [
                (
                    dan_id,
                    julia_id,
                    t_holiday_email,
                    "Subject: Summer holiday plans\n\n"
                    "Hi Julia,\nWhen are you next going on holiday? "
                    "I'm thinking of booking something for August.",
                ),
                (
                    julia_id,
                    dan_id,
                    t_holiday_email + timedelta(hours=2),
                    "Hi Dan! I'm hoping to go in August too, "
                    "but let's see what my boss says about the timing.",
                ),
            ],
        )

    # --------------------------------------------------------------------- #
    def _seed_filler(self, exchanges: int = 20, msgs_per: int = 15) -> None:
        """Adds irrelevant chatter so filtering matters."""
        random.seed(12345)
        media = ["email", "phone_call", "sms_message"]
        start = datetime(
            2024,
            random.randint(1, 12),
            random.randint(1, 28),
            tzinfo=timezone.utc,
        )

        for ex_off in range(exchanges):
            ex_id = 10 + ex_off
            mtype = random.choice(media)
            a, b = random.sample(list(_ID_BY_NAME.values()), 2)
            batch: List[tuple[int, int, datetime, str]] = []
            for i in range(msgs_per):
                batch.append(
                    (
                        a if i % 2 else b,
                        b if i % 2 else a,
                        start + timedelta(minutes=ex_off * 3 + i),
                        random.choice(
                            [
                                "I didn't hear you, could you repeat?",
                                "The weather is lovely today.",
                                "Just to let you know, I'll need to leave soon.",
                                "The football game last night was sooo good, can't believe you missed it.",
                                "Are you even listening to what I'm saying?",
                                "We're really talking through eachother here aren't we?",
                            ],
                        ),
                    ),
                )
            self._log(ex_id, mtype, batch)

    # --------------------------------------------------------------------- #
    def _log(
        self,
        ex_id: int,
        medium: str,
        msgs: List[tuple[int, int, datetime, str]],
    ) -> None:
        # Build messages with explicit message_id for ordering
        messages = []
        for s, r, ts, txt in msgs:
            messages.append(
                Message(
                    medium=medium,
                    sender_id=s,
                    receiver_ids=[r],
                    timestamp=ts,
                    content=txt,
                    exchange_id=ex_id,
                    message_id=self._message_counter,
                ),
            )
            self._message_counter += 1

        # Async logging - fire and forget (ordering guaranteed by explicit message_id)
        # Skip EventBus to avoid cross-loop issues during fixture setup
        self.tm.log_messages(messages, synchronous=False, _skip_event_bus=True)

    def finalize(self) -> None:
        """Wait for all async log operations to complete."""
        self.tm.join_published()


# --------------------------------------------------------------------------- #
#  VERSIONED SCENARIO FIXTURE
# --------------------------------------------------------------------------- #


def _commit_contexts_for_rollback(ctx_prefix: str) -> None:
    """Commit all contexts under prefix for rollback support."""
    created_contexts = unify.get_contexts(prefix=ctx_prefix)
    created_context_names = list(created_contexts.keys())

    def commit_context_and_store(ctx_name):
        try:
            commit_info = unify.commit_context(
                name=ctx_name,
                commit_message="Initial seed data for tests",
            )
            SCENARIO_COMMIT_HASHES[ctx_name] = commit_info["commit_hash"]
        except Exception:
            pass  # May already be committed

    if created_context_names:
        unify.map(commit_context_and_store, created_context_names, mode="asyncio")


def _rebuild_commit_hashes(ctx_prefix: str) -> None:
    """Rebuild commit hashes from existing contexts for rollback support."""
    existing_contexts = unify.get_contexts(prefix=ctx_prefix)
    for ctx_name in existing_contexts.keys():
        try:
            history = unify.get_context_commits(ctx_name)
            if history:
                SCENARIO_COMMIT_HASHES[ctx_name] = history[0]["commit_hash"]
        except Exception:
            pass


def _setup_tm_scenario(
    request: pytest.FixtureRequest,
) -> Tuple[TranscriptManager, Dict[str, int]]:
    """
    Synchronous setup logic for the transcript manager scenario.

    Creates/reuses a versioned context, seeds data if needed,
    and returns the manager + id mapping.

    Note: This is intentionally synchronous to avoid blocking the event loop
    when using file locks. The async fixture wrapper just calls this.
    """
    ManagerRegistry.clear()
    ContextRegistry.clear()
    os.environ["TQDM_DISABLE"] = "1"

    ctx = "tests/test_transcript_manager/Scenario"
    overwrite_scenarios = request.config.getoption("--overwrite-scenarios")

    # If --overwrite-scenarios is set, delete existing contexts first
    if overwrite_scenarios:
        existing_contexts = unify.get_contexts(prefix=ctx)
        existing_context_names = list(existing_contexts.keys())
        if existing_context_names:
            unify.map(
                lambda c: unify.delete_context(c),
                existing_context_names,
                mode="asyncio",
            )

    # Set context before any operations (create first like ContactManager does)
    unify.create_context(ctx)  # exist_ok=True by default
    unify.set_context(ctx, relative=False)

    # Create managers
    cm = ContactManager()
    transcript_ctx = f"{ctx}/Transcripts"
    tm = TranscriptManager(contact_manager=cm)

    # Use file lock to coordinate seeding across parallel processes
    with scenario_file_lock("tm_scenario"):
        seeded = is_scenario_seeded(cm, _CONTACTS, transcript_context=transcript_ctx)
        if seeded:
            # Scenario exists - just rebuild local state
            print("Scenario already seeded, rebuilding local state...")
            ids = rebuild_id_mapping(cm, _CONTACTS)
            _ID_BY_NAME.update(ids)
            _rebuild_commit_hashes(ctx)
            # Check if embeddings exist, create if missing (for older scenarios)
            if _ensure_embeddings_exist(transcript_ctx):
                # Embeddings were created, need to recommit
                print(f"Recommitting {ctx} with new embeddings...")
                _commit_contexts_for_rollback(ctx)
        else:
            # Scenario not seeded - seed it
            print("Seeding transcript manager scenario...")
            sb = ScenarioBuilder(cm, tm)
            sb._seed_contacts()
            sb._seed_key_exchanges()
            sb._seed_filler()
            sb.finalize()  # Wait for all async log operations to complete
            # Pre-compute embeddings for pure columns before committing
            # This avoids recomputing on every test run
            _precompute_embeddings(transcript_ctx)
            _commit_contexts_for_rollback(ctx)

    # Unset context after setup, like ContactManager does
    unify.unset_context()

    return tm, dict(_ID_BY_NAME)


@pytest_asyncio.fixture(scope="session")
async def tm_scenario(
    request: pytest.FixtureRequest,
) -> Tuple[TranscriptManager, Dict[str, int]]:
    """
    Create (and later clean up) a versioned context so that *all* tests share the
    same seeded data.

    Uses a file lock to coordinate parallel test processes - only one process
    seeds the scenario, others wait and then rebuild local state from existing data.
    """
    return _setup_tm_scenario(request)


@pytest.fixture(scope="function")
def tm_manager_scenario(tm_scenario):
    """
    Per-test fixture for tests using the transcript scenario (e.g., test_ask.py).

    Uses a file lock to serialize tests, ensuring the full sequence
    (rollback → run test → verify) is atomic. This prevents race conditions
    where parallel tests' rollbacks orphan each other's derived column data.

    Note: Despite being called "read scenario", these tests create derived
    columns (embeddings, composite fields) during semantic search, so they
    are not truly read-only and require serialization.
    """
    tm, _ID_BY_NAME = tm_scenario

    def rollback_context(ctx):
        unify.rollback_context(
            name=ctx,
            commit_hash=SCENARIO_COMMIT_HASHES[ctx],
        )

    # Use mutation_test_lock to prevent parallel rollbacks from orphaning
    # derived column data (embeddings) created by concurrent search operations
    with mutation_test_lock("tm_read"):
        # Rollback INSIDE the lock to prevent other tests
        # from rolling back while this test is running
        scenario_names = list(SCENARIO_COMMIT_HASHES.keys())
        if scenario_names:
            unify.map(rollback_context, scenario_names, mode="asyncio")

        # Re-set the scenario context to ensure nested operations work
        unify.set_context("tests/test_transcript_manager/Scenario", relative=False)

        yield tm, _ID_BY_NAME
