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
import asyncio
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
        for idx, c in enumerate(_CONTACTS):
            _ID_BY_NAME[c["first_name"].lower()] = idx

    @classmethod
    async def create(cls) -> "ScenarioBuilder":
        """Build an instance and run all async seeding steps."""
        self = cls()

        await self._seed_contacts()
        await self._seed_key_exchanges()
        await self._seed_filler()

        # Store an initial summary so that summaries exist
        await self.tm.summarize(exchange_ids=[0, 1])

        return self

    # --------------------------------------------------------------------- #
    async def _seed_contacts(self) -> None:
        for idx, c in enumerate(_CONTACTS):
            self.cm._create_contact(**c)

    # --------------------------------------------------------------------- #
    async def _seed_key_exchanges(self) -> None:
        now = datetime(2025, 4, 20, 15, 0, tzinfo=timezone.utc)

        # E0: first Dan–Julia phone call
        await self._log(
            0,
            "phone_call",
            [
                (1, 2, now, "Hi Julia, it's Dan. Quick check-in about Q2 metrics."),
                (2, 1, now + timedelta(seconds=30), "Sure Dan, ready when you are."),
            ],
        )

        # E1: *last* Dan–Julia phone call (later date)
        later = datetime(2025, 4, 26, 9, 30, tzinfo=timezone.utc)
        await self._log(
            1,
            "phone_call",
            [
                (
                    1,
                    2,
                    later,
                    "Morning Julia – finalising the London event agenda today.",
                ),
                (
                    2,
                    1,
                    later + timedelta(seconds=45),
                    "Great. Let's confirm the speaker list and coffee budget.",
                ),
            ],
        )

        # E2: Carlos interest e-mail
        t_email = datetime(2025, 4, 21, 12, 0, tzinfo=timezone.utc)
        await self._log(
            2,
            "email",
            [
                (
                    0,
                    1,
                    t_email,
                    "Subject: Stapler bulk order\n\n"
                    "Hi Dan,\nI'm **interested in buying 200 units** of "
                    "your new stapler. Can you quote?\n\nThanks,\nCarlos",
                ),
                (
                    1,
                    0,
                    t_email + timedelta(hours=2),
                    "Hi Carlos — sure, $4.50 per unit. See attached PDF.",
                ),
            ],
        )

        # E3: Jimmy holiday WhatsApp
        t_holiday = datetime(2025, 4, 22, 18, 10, tzinfo=timezone.utc)
        await self._log(
            3,
            "whatsapp_message",
            [
                (
                    3,
                    1,
                    t_holiday,
                    "Heads-up Dan, I'll be **on holiday from 2025-05-15** "
                    "to 2025-05-30. Ping me before that if urgent.",
                ),
            ],
        )

        # E4: Anne passport excuse (WhatsApp)
        t_excuse = datetime(2025, 4, 23, 9, 0, tzinfo=timezone.utc)
        await self._log(
            4,
            "whatsapp_message",
            [
                (
                    4,
                    1,
                    t_excuse,
                    "Sorry Dan, I *can't join the Berlin trip because my "
                    "passport expired* last week.",
                ),
            ],
        )

    # --------------------------------------------------------------------- #
    async def _seed_filler(self, exchanges: int = 20, msgs_per: int = 15) -> None:
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
            await self._log(ex_id, mtype, batch)

    # --------------------------------------------------------------------- #
    async def _log(
        self,
        ex_id: int,
        medium: str,
        msgs: List[tuple[int, int, datetime, str]],
    ) -> None:
        self.tm._log_messages(
            [
                Message(
                    medium=medium,
                    sender_id=s,
                    receiver_id=r,
                    timestamp=ts,
                    content=txt,
                    exchange_id=ex_id,
                ).to_post_json()
                for s, r, ts, txt in msgs
            ],
        )


# --------------------------------------------------------------------------- #
#  AsyncIO event loop (session-scoped)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def event_loop():
    """A dedicated loop for the whole session – avoids 'event loop is closed'."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# --------------------------------------------------------------------------- #
#  VERSIONED SCENARIO FIXTURE
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def tm_scenario(event_loop: asyncio.AbstractEventLoop, request: pytest.FixtureRequest):
    """
    Create (and later clean up) a versioned context so that *all* tests share the
    same seeded data.
    """
    os.environ["TQDM_DISABLE"] = "1"

    unify.set_context("test_transcript_manager")
    sb = ScenarioBuilder()
    existing_contexts = unify.get_contexts(prefix="test_transcript_manager")
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

        unify.map(
            delete_all_contexts,
            list(existing_contexts.keys()),
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

        unify.map(
            get_and_rollback_context,
            list(existing_contexts.keys()),
            mode="asyncio",
        )

    # --- One-time setup (per session) ---
    if not SCENARIO_COMMIT_HASHES:
        print("Seeding transcript manager scenario...")
        event_loop.run_until_complete(sb.create())

        def commit_context_and_store(ctx):
            commit_info = unify.commit_context(
                name=ctx,
                commit_message="Initial seed data for tests",
            )
            SCENARIO_COMMIT_HASHES[ctx] = commit_info["commit_hash"]

        unify.map(
            commit_context_and_store,
            list(existing_contexts.keys()),
            mode="asyncio",
        )

    yield sb.tm, _ID_BY_NAME
