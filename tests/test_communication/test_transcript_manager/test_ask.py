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
from datetime import datetime, timezone, timedelta, UTC
from unity.events.event_bus import EventBus, Event
from typing import List

import pytest

import pytest
import asyncio
import unify
from unity.communication.transcript_manager.transcript_manager import TranscriptManager
from unity.communication.types.message import Message
from unity.common.llm_helpers import _dumps
from tests.assertion_helpers import assertion_failed
from tests.helpers import _handle_project

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
            self.tm.create_contact(**c)
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
#  DETERMINISTIC GROUND-TRUTH GENERATOR                                       #
# --------------------------------------------------------------------------- #


def _answer_semantic(tm: TranscriptManager, question: str) -> str:
    """Compute the *correct* answer directly from stored data."""
    q = question.lower()
    messages = tm._search_messages(limit=None)

    def cid(name: str) -> int:
        return _ID_BY_NAME[name]

    if _is_summary_q(question):
        # return the *two utterances* that form the last Dan–Julia phone call.
        last_call = sorted(
            (
                m
                for m in messages
                if m.medium == "phone_call"
                and {m.sender_id, m.receiver_id} == {cid("dan"), cid("julia")}
            ),
            key=lambda m: m.timestamp,
        )[-2:]
        return "\n".join(m.content for m in last_call)

    if "quantity" in q and "carlos" in q:
        return "200"

    if "carlos" in q and "buy" in q:
        msg: Message = next(
            m
            for m in messages
            if m.sender_id == cid("carlos") and "buy" in m.content.lower()
        )
        quote = msg.content.splitlines()[0]
        return f"Yes – {quote}"

    if "when did dan last speak with julia" in q:
        last: str = max(
            m.timestamp
            for m in messages
            if m.medium == "phone_call"
            and {m.sender_id, m.receiver_id} == {cid("dan"), cid("julia")}
        )
        return last.split("T")[0]

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
#  LLM-AS-A-JUDGE SUMMARY COMPARISONS                                         #
# --------------------------------------------------------------------------- #


def _is_summary_q(q: str) -> bool:
    if isinstance(q, list):
        return all(_is_summary_q(q) for q in q)
    return "one-sentence summary" in q.lower() or "one sentence summary" in q.lower()


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
#  EVALUATION LLM                                                             #
# --------------------------------------------------------------------------- #


def _llm_assert_correct(
    question: str | List[str],
    expected: str | List[str],
    candidate: str,
    steps: list,
    multiple_answers: bool = False,
) -> None:
    """LLM-based validation with stricter or fuzzier rubric per question."""
    judge = unify.Unify("o4-mini@openai", cache=True, traced=True)

    if _is_summary_q(question):
        system_msg = (
            "You are a meticulous but fair summary evaluator. "
            "You will be given the *source dialogue* of a short phone call and a candidate **one-sentence** summary. "
            "Your task is to decide whether the summary accurately conveys the main intent and key factual points. "
            "Minor stylistic or tense differences, re-ordering, shortened wording, or inclusion of obviously correct contextual details are acceptable. "
            "Mark correct⇢true unless the summary is missing a crucial fact, introduces a contradiction, or otherwise misrepresents the dialogue. "
            'Respond ONLY with valid JSON of the form {"correct": true} or {"correct": false}. '
        )
        payload = _dumps(
            {"dialogue": expected, "summary": candidate},
            indent=4,
        )
    else:
        if multiple_answers:
            scenario_str = (
                "You will be given multiple questions, and ground-truth answers derived "
                "directly from the data, and candidate answers corresponding to each question respectively."
            )
        else:
            scenario_str = (
                "You will be given a question, a ground-truth answer derived "
                "directly from the data, and a candidate answer."
            )

        system_msg = (
            "You are a meticulous but fair unit-test judge. "
            + scenario_str
            + "Your role is to decide whether the candidate answer conveys the same factual information as the ground-truth answer. "
            "Formatting or wording differences should be considered equivalent as long as the facts match. "
            "Additional correct details that do not contradict the ground truth are acceptable. "
            "Mark correct⇢true if the candidate clearly contains the ground-truth fact(s) and introduces no contradiction; otherwise false. "
            'Respond ONLY with valid JSON of the form {"correct": true} or {"correct": false}. '
        )
        payload = _dumps(
            {"question": question, "ground_truth": expected, "candidate": candidate},
            indent=4,
        )

    judge.set_system_message(system_msg)
    result = judge.generate(payload)

    match = re.search(r"\{.*\}", result, re.S)
    assert match, assertion_failed(
        "Expected JSON format from LLM judge",
        result,
        steps,
        "LLM judge returned unexpected format",
    )
    verdict = json.loads(match.group(0))
    assert verdict.get("correct") is True, assertion_failed(
        expected,
        candidate,
        steps,
        f"Question: {question}",
    )


# --------------------------------------------------------------------------- #
#  PARAMETRISED TEST                                                          #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def event_loop():
    """
    Create a session-scoped asyncio event loop.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_session_context():
    """Set up a session-wide context for all tests in this module."""
    file_path = __file__
    ctx = "/".join(file_path.split("/tests/")[1].split("/"))[:-3]
    if unify.get_contexts(prefix=ctx):
        unify.delete_context(ctx)
    with unify.Context(ctx):
        unify.set_trace_context("Traces")
        yield

    unify.delete_context(ctx)


@pytest.fixture(scope="session")
def tm_scenario(
    setup_session_context,
    event_loop: asyncio.AbstractEventLoop,
) -> TranscriptManager:
    """
    Fixture to create a TranscriptManager with a populated scenario.
    """
    builder = event_loop.run_until_complete(ScenarioBuilder.create())
    return builder.tm


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question", QUESTIONS)
async def test_ask_semantic_with_llm_judgement(
    question: str,
    tm_scenario: TranscriptManager,
) -> None:
    """
    Calls the real `.ask()` (which itself may call the LLM multiple
    times), then asks a _separate_ LLM whether the answer is acceptable.
    """
    tm = tm_scenario
    handle = tm.ask(question, return_reasoning_steps=True)
    candidate, steps = await handle.result()
    expected = _answer_semantic(tm, question)
    _llm_assert_correct(question, expected, candidate, steps)


@pytest.mark.asyncio
@pytest.mark.eval
async def test_ask_allows_interjection(tm_scenario: TranscriptManager):
    """Ask one semantic question, then interject with a second, and verify both answers appear."""
    tm = tm_scenario
    # 1) Initial semantic query – last Dan ⇢ Julia phone call date
    q_initial = QUESTIONS[1]  # "When did Dan last speak with Julia on the phone?"
    handle = tm.ask(q_initial, return_reasoning_steps=True)

    # 2) Interject with a *different* question (Jimmy holiday date)
    q_follow_up = QUESTIONS[2]  # "Did Jimmy ever tell us when he's on holiday...?"
    await handle.interject(q_follow_up)

    # 3) Await combined answer
    answer, steps = await handle.result()
    expected_date_call = _answer_semantic(tm, q_initial)
    expected_date_holiday = _answer_semantic(tm, q_follow_up)

    # 4) Assertions
    _llm_assert_correct(
        [q_initial, q_follow_up],
        [expected_date_call, expected_date_holiday],
        answer,
        steps,
        multiple_answers=True,
    )


@_handle_project
@pytest.mark.asyncio
@pytest.mark.eval
async def test_ask_honors_stop():
    event_bus = EventBus()
    tm = TranscriptManager(event_bus)
    handle = tm.ask(
        "List every message received from Carlos, then provide a detailed summary of each one in chronological order.",
    )
    handle.stop()
    with pytest.raises(asyncio.CancelledError):
        await handle.result()
    assert handle.done()
