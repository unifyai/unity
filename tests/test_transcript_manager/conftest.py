"""
Global fixtures & shared data usable from any test module.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone, UTC
from unity.events.event_bus import EventBus, Event
from datetime import timedelta
from typing import List
import pytest
import os

import asyncio
import unify
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message

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
        self._event_bus = EventBus()
        self.cm = ContactManager(self._event_bus)
        self.tm = TranscriptManager(self._event_bus)

    @classmethod
    async def create(cls) -> "ScenarioBuilder":
        """Build an instance and run all async seeding steps."""
        self = cls()

        await self._seed_contacts()
        await self._seed_key_exchanges()
        await self._seed_filler()
        self._event_bus.join_published()

        # Store an initial summary so that summaries exist
        await self.tm.summarize(exchange_ids=[0, 1])
        self._event_bus.join_published()

        return self

    # --------------------------------------------------------------------- #
    async def _seed_contacts(self) -> None:
        for idx, c in enumerate(_CONTACTS):
            self.cm.create_contact(**c)
            _ID_BY_NAME[c["first_name"].lower()] = idx

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
        start = datetime(2025, 4, 24, tzinfo=timezone.utc)

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
                        f"Filler {ex_id}-{i} {mtype} random text.",
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
        [
            await self._event_bus.publish(
                Event(
                    type="Messages",
                    timestamp=datetime.now(UTC),
                    payload=Message(
                        medium=medium,
                        sender_id=s,
                        receiver_id=r,
                        timestamp=ts.isoformat(),
                        content=txt,
                        exchange_id=ex_id,
                    ),
                ),
            )
            for s, r, ts, txt in msgs
        ]


# --------------------------------------------------------------------------- #
# 1.  AsyncIO event loop (session-scoped)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def event_loop():
    """A dedicated loop for the whole session – avoids 'event loop is closed'."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# --------------------------------------------------------------------------- #
# 2.  One Unify context for the whole run
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 3.  Fully-populated TranscriptManager
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def tm_scenario(
    setup_session_context,
    event_loop: asyncio.AbstractEventLoop,
):
    """Seed the backend exactly once and share the TM instance."""
    builder = event_loop.run_until_complete(ScenarioBuilder.create())
    return builder.tm, _ID_BY_NAME
