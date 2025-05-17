"""
tests/test_ask.py
=================

Integration-style tests for ``TranscriptManager.ask`` that rely on a live
LLM to (a) choose tools and (b) judge whether the final answer is
correct.

Running the suite therefore requires:

* network access
* a valid OpenAI-compatible key (used by `unify.Unify`)
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import List

import pytest

import unify
from communication.transcript_manager.transcript_manager import TranscriptManager
from communication.types.message import Message
from common.llm_helpers import _dumps

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
        if "test_ask" in unify.list_projects():
            unify.delete_project("test_ask")
        unify.activate("test_ask")
        self.tm = TranscriptManager()
        self.tm.start()
        self._seed_contacts()
        self._seed_key_exchanges()
        self._seed_filler()
        # One stored summary just so summaries exist
        self.tm.summarize(exchange_ids=[0, 1])

    # --------------------------------------------------------------------- #
    def _seed_contacts(self) -> None:
        for idx, c in enumerate(_CONTACTS):
            self.tm.create_contact(**c)
            _ID_BY_NAME[c["first_name"].lower()] = idx

    # --------------------------------------------------------------------- #
    def _seed_key_exchanges(self) -> None:
        now = datetime(2025, 4, 20, 15, 0, tzinfo=timezone.utc)

        # E0: first Dan–Julia phone call
        self._log(
            0,
            "phone_call",
            [
                (1, 2, now, "Hi Julia, it's Dan. Quick check-in about Q2 metrics."),
                (2, 1, now + timedelta(seconds=30), "Sure Dan, ready when you are."),
            ],
        )

        # E1: *last* Dan–Julia phone call (later date)
        later = datetime(2025, 4, 26, 9, 30, tzinfo=timezone.utc)
        self._log(
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
        self._log(
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
        self._log(
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
        self._log(
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
    def _seed_filler(self, exchanges: int = 20, msgs_per: int = 15) -> None:
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
            self._log(ex_id, mtype, batch)

    # --------------------------------------------------------------------- #
    def _log(
        self,
        ex_id: int,
        medium: str,
        msgs: List[tuple[int, int, datetime, str]],
    ) -> None:
        self.tm.log_messages(
            [
                Message(
                    medium=medium,
                    sender_id=s,
                    receiver_id=r,
                    timestamp=ts.isoformat(),
                    content=txt,
                    exchange_id=ex_id,
                )
                for s, r, ts, txt in msgs
            ],
        )


# --------------------------------------------------------------------------- #
#  DETERMINISTIC GROUND-TRUTH GENERATOR                                       #
# --------------------------------------------------------------------------- #


def _answer_semantic(tm: TranscriptManager, question: str) -> str:
    """Compute the *correct* answer directly from stored data."""
    q = question.lower()
    contacts = tm._search_contacts()
    messages = tm._search_messages()

    def cid(name: str) -> int:
        return _ID_BY_NAME[name]

    if "carlos" in q and "buy" in q:
        msg = next(
            m
            for m in messages
            if m.sender_id == cid("carlos") and "buy" in m.content.lower()
        )
        quote = msg.content.splitlines()[0]
        return f"Yes – {quote}"

    if "when did dan last speak with julia" in q:
        last = max(
            m
            for m in messages
            if m.medium == "phone_call"
            and {m.sender_id, m.receiver_id} == {cid("dan"), cid("julia")}
        )
        return last.timestamp.split("T")[0]

    if "jimmy" in q and "holiday" in q:
        pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        msg = next(
            m
            for m in messages
            if m.sender_id == cid("jimmy") and "holiday" in m.content.lower()
        )
        return pattern.search(msg.content).group(0)

    if "anne" in q and "why" in q:
        msg = next(m for m in messages if m.sender_id == cid("anne"))
        return "passport expired"

    if "quantity" in q and "carlos" in q:
        return "200"

    if "medium does julia use most" in q:
        counts = Counter(m.medium for m in messages if m.sender_id == cid("julia"))
        return counts.most_common(1)[0][0]

    if "how many different media has dan used" in q:
        media = {m.medium for m in messages if m.sender_id == cid("dan")}
        return str(len(media))

    if "one-sentence summary" in q or "one sentence summary" in q:
        last_call = [
            m
            for m in messages
            if m.medium == "phone_call"
            and {m.sender_id, m.receiver_id} == {cid("dan"), cid("julia")}
        ]
        last_ts = max(m.timestamp for m in last_call)
        combined = " ".join(m.content for m in last_call if m.timestamp == last_ts)
        return " ".join(combined.split()[:12]) + "..."

    return "N/A"


# --------------------------------------------------------------------------- #
#  QUESTIONS                                                                  #
# --------------------------------------------------------------------------- #

QUESTIONS = [
    "Did Carlos seem interested in buying the product? Can you find a relevant quote to back up your answer?",
    "When did Dan last speak with Julia on the phone?",
    "Did Jimmy ever tell us when he's on holiday? If so, what date?",
    "Why didn't Anne want to come with us on the trip? I forgot her excuse.",
    "What quantity did Carlos say he wanted to buy?",
    "Which medium does Julia use most often to communicate?",
    "How many different media has Dan used so far?",
    "Give me a one-sentence summary of the last Dan-Julia phone call.",
]


# --------------------------------------------------------------------------- #
#  PYTEST FIXTURES                                                            #
# --------------------------------------------------------------------------- #


@pytest.fixture
def tm_scenario() -> TranscriptManager:
    """Fresh, fully-seeded manager for every test run."""
    return ScenarioBuilder().tm


# --------------------------------------------------------------------------- #
#  EVALUATION LLM                                                             #
# --------------------------------------------------------------------------- #


def _llm_assert_correct(
    question: str,
    expected: str,
    candidate: str,
    steps: list,  # noqa: D401 – clarity outweighs strict type accuracy
) -> None:
    """Assert *candidate* satisfies *expected* for *question* via an LLM judge.

    Any assertion failure is augmented with the full **reasoning steps** so
    that debugging always has a complete tool-use trace available.
    """

    judge = unify.Unify("o4-mini@openai", cache=True)
    judge.set_system_message(
        "You are a strict unit-test judge. "
        "You will be given a question, a ground-truth answer derived "
        "directly from the data, and a candidate answer produced by the "
        "system under test. "
        'Respond ONLY with valid JSON of the form {"correct": true} or {"correct": false}. '
        "Mark correct⇢true if a reasonable human would accept the candidate "
        "as answering the question fully and accurately; otherwise false.",
    )

    payload = _dumps(
        {"question": question, "ground_truth": expected, "candidate": candidate},
        indent=4,
    )
    result = judge.generate(payload)

    # Strip anything outside first {...}
    match = re.search(r"\{.*\}", result, re.S)
    assert match, (
        "LLM judge returned unexpected format: "
        f"{result!r}\nReasoning steps:\n{json.dumps(steps, indent=4)}"
    )
    verdict = json.loads(match.group(0))
    assert verdict.get("correct") is True, (
        "LLM judge marked answer incorrect:\n"
        f"Q: {question}\nExpected: {expected}\nGot: {candidate}\n"
        f"Reasoning steps:\n{json.dumps(steps, indent=4)}"
    )


# --------------------------------------------------------------------------- #
#  PARAMETRISED TEST                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.parametrize("question", QUESTIONS)
def test_ask_semantic_with_llm_judgement(
    question: str,
    tm_scenario: TranscriptManager,
) -> None:
    """
    Calls the real `.ask()` (which itself may call the LLM multiple
    times), then asks a _separate_ LLM whether the answer is acceptable.
    """
    try:
        candidate, steps = tm_scenario.ask(question, return_reasoning_steps=True)
        expected = _answer_semantic(tm_scenario, question)
        _llm_assert_correct(question, expected, candidate, steps)
    except Exception as e:
        if "test_ask" in unify.list_projects():
            unify.delete_project("test_ask")
        raise e
