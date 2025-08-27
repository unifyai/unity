"""
Global fixtures & shared data usable from any test module.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from datetime import timedelta
from typing import List, Dict, Any
import pytest
import os
import unify
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message

SCENARIO_COMMIT_HASHES: Dict[str, Any] = {}

# --------------------------------------------------------------------------- #
#  CONTACTS (same as before)                                                  #
# --------------------------------------------------------------------------- #

_CONTACTS: List[dict] = [
    dict(  # id = 0
        first_name="Carlos",
        surname="Diaz",
        email_address="carlos.diaz@example.com",
        phone_number="+14155550000",
        whatsapp_number="+14155550000",
    ),
    dict(  # id = 1
        first_name="Dan",
        surname="Turner",
        email_address="dan.turner@example.com",
        phone_number="+447700900001",
        whatsapp_number="+447700900001",
    ),
    dict(  # id = 2
        first_name="Julia",
        surname="Nguyen",
        email_address="julia.nguyen@example.com",
        phone_number="+447700900002",
        whatsapp_number="+447700900002",
    ),
    dict(  # id = 3
        first_name="Jimmy",
        surname="O'Brien",
        email_address="jimmy.obrien@example.com",
        phone_number="+61240011000",
        whatsapp_number="+61240011000",
    ),
    dict(  # id = 4
        first_name="Anne",
        surname="Fischer",
        email_address="anne.fischer@example.com",
        phone_number="+49891234567",
        whatsapp_number="+49891234567",
    ),
]

_ID_BY_NAME: dict[str, int] = {}  # filled during seeding


# --------------------------------------------------------------------------- #
#  SCENARIO BUILDER                                                           #
# --------------------------------------------------------------------------- #


class ScenarioBuilder:
    """Populate Unify with contacts, 6 'meaningful' exchanges + filler."""

    def __init__(self) -> None:
        self.cm = ContactManager()
        self.tm = TranscriptManager()
        # Mapping will be filled during contact creation (_seed_contacts)

    @classmethod
    def create(cls) -> "ScenarioBuilder":
        """Build an instance and run all async seeding steps."""
        self = cls()

        self._seed_contacts()
        self._seed_key_exchanges()
        self._seed_filler()

        return self

    # --------------------------------------------------------------------- #
    def _seed_contacts(self) -> None:
        for c in _CONTACTS:
            outcome = self.cm._create_contact(**c)
            assigned_id = outcome["details"]["contact_id"]
            _ID_BY_NAME[c["first_name"].lower()] = assigned_id

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

        # E3: Jimmy holiday WhatsApp
        jimmy_id = _ID_BY_NAME["jimmy"]
        t_holiday = datetime(2025, 4, 22, 18, 10, tzinfo=timezone.utc)
        self._log(
            3,
            "whatsapp_message",
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

        # E4: Anne passport excuse (WhatsApp)
        anne_id = _ID_BY_NAME["anne"]
        t_excuse = datetime(2025, 4, 23, 9, 0, tzinfo=timezone.utc)
        self._log(
            4,
            "whatsapp_message",
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

    # --------------------------------------------------------------------- #
    def _seed_filler(self, exchanges: int = 20, msgs_per: int = 15) -> None:
        """Adds irrelevant chatter so filtering matters."""
        random.seed(12345)
        media = ["email", "phone_call", "sms_message", "whatsapp_message"]
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
        [
            self.tm.log_messages(
                Message(
                    medium=medium,
                    sender_id=s,
                    receiver_ids=[r],
                    timestamp=ts,
                    content=txt,
                    exchange_id=ex_id,
                ),
            )
            for s, r, ts, txt in msgs
        ]
        self.tm.join_published()


# --------------------------------------------------------------------------- #
#  VERSIONED SCENARIO FIXTURE
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def tm_scenario(request: pytest.FixtureRequest):
    """
    Create (and later clean up) a versioned context so that *all* tests share the
    same seeded data.
    """
    os.environ["TQDM_DISABLE"] = "1"

    ctx = "tests/test_transcript_manager/Scenario"
    unify.set_context(ctx, relative=False)
    sb = ScenarioBuilder()
    existing_contexts = unify.get_contexts(prefix=ctx)
    existing_context_names = list(existing_contexts.keys())
    no_reuse_scenario = request.config.getoption("--no-reuse-scenario")

    # If --no-reuse-scenario is explicitly set, override reuse_scenario
    if no_reuse_scenario:
        reuse_scenario = False
    else:
        reuse_scenario = True

    if not reuse_scenario:
        # delete all contexts to freshly create the new scenario
        def delete_all_contexts(ctx):
            unify.delete_context(ctx)

        if existing_context_names:
            unify.map(
                delete_all_contexts,
                existing_context_names,
                mode="asyncio",
            )

    if reuse_scenario and not SCENARIO_COMMIT_HASHES:

        def get_and_rollback_context(ctx):
            history = unify.get_context_commits(ctx)
            if history:
                unify.rollback_context(
                    name=ctx,
                    commit_hash=history[0]["commit_hash"],
                )
                SCENARIO_COMMIT_HASHES[ctx] = history[0]["commit_hash"]

        if existing_context_names:
            unify.map(
                get_and_rollback_context,
                existing_context_names,
                mode="asyncio",
            )

    # --- One-time setup (per session) ---
    if not SCENARIO_COMMIT_HASHES:
        print("Seeding transcript manager scenario...")
        sb.create()

        def commit_context_and_store(ctx):
            commit_info = unify.commit_context(
                name=ctx,
                commit_message="Initial seed data for tests",
            )
            SCENARIO_COMMIT_HASHES[ctx] = commit_info["commit_hash"]

        # After seeding, re-fetch contexts created under the test prefix
        created_contexts = unify.get_contexts(prefix=ctx)
        created_context_names = list(created_contexts.keys())

        if created_context_names:
            unify.map(
                commit_context_and_store,
                created_context_names,
                mode="asyncio",
            )
        else:
            # Fallback: try committing known child contexts if present
            all_ctxs = unify.get_contexts()
            for _ctx in [
                f"{ctx}/Contacts",
                f"{ctx}/Transcripts",
            ]:
                if _ctx in all_ctxs:
                    commit_context_and_store(_ctx)

    unify.unset_context()
    yield sb.tm, _ID_BY_NAME


@pytest.fixture(scope="function")
def tm_manager_scenario(tm_scenario):
    tm, _ID_BY_NAME = tm_scenario

    def rollback_context(ctx):
        unify.rollback_context(
            name=ctx,
            commit_hash=SCENARIO_COMMIT_HASHES[ctx],
        )

    # Rollback to clean state before test
    scenario_names = list(SCENARIO_COMMIT_HASHES.keys())
    if scenario_names:
        unify.map(rollback_context, scenario_names, mode="asyncio")

    yield tm, _ID_BY_NAME
