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
import re
from datetime import datetime, timezone
from typing import List
import pytest

import pytest
import asyncio
import unify
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from unity.common.llm_helpers import _dumps
from tests.assertion_helpers import assertion_failed
from tests.helpers import _handle_project, SETTINGS


# --------------------------------------------------------------------------- #
#  DETERMINISTIC GROUND-TRUTH GENERATOR                                       #
# --------------------------------------------------------------------------- #


def _answer_semantic(
    tm: TranscriptManager,
    question: str,
    _ID_BY_NAME: dict[str, int],
) -> str:
    """Compute the *correct* answer directly from stored data."""
    q = question.lower()
    messages = tm._filter_messages(limit=None)

    def cid(name: str) -> int:
        return _ID_BY_NAME[name]

    if _is_summary_q(question):
        # return the *two utterances* that form the last Dan–Julia phone call.
        NAME_BY_ID = {v: k.capitalize() for k, v in _ID_BY_NAME.items()}
        last_call_messages = sorted(
            (
                m
                for m in messages
                if m.medium == "phone_call"
                and {m.sender_id} | set(m.receiver_ids) == {cid("dan"), cid("julia")}
            ),
            key=lambda m: m.timestamp,
        )[-2:]
        dialogue_date_str = ""
        if last_call_messages:
            # Let's use the date of the first message in the snippet
            try:
                dialogue_timestamp_iso = last_call_messages[0].timestamp.isoformat()
                dialogue_date = datetime.fromisoformat(
                    dialogue_timestamp_iso.replace("Z", "+00:00"),
                ).strftime(
                    "%B %d, %Y",
                )  # Format as "April 26, 2025"
                dialogue_date_str = f"Dialogue from {dialogue_date}:\n"
            except ValueError:
                dialogue_date_str = ""  # Fallback if parsing fails
        # Construct dialogue with speaker names
        dialogue_with_speakers = []
        for m in last_call_messages:
            sender_name = NAME_BY_ID.get(m.sender_id, f"Unknown({m.sender_id})")
            dialogue_with_speakers.append(f"{sender_name}: {m.content}")
        return dialogue_date_str + "\n".join(dialogue_with_speakers)

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
            and {m.sender_id} | set(m.receiver_ids) == {cid("dan"), cid("julia")}
        )
        return last.isoformat().split("T")[0]

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

    if "how many different media has dan used" in q:
        media = {m.medium for m in messages if m.sender_id == cid("dan")}
        return str(len(media))

    if "one-sentence summary" in q or "one sentence summary" in q:
        last_call = [
            m
            for m in messages
            if m.medium == "phone_call"
            and {m.sender_id} | set(m.receiver_ids) == {cid("dan"), cid("julia")}
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
    "How many different media has Dan used so far?",
    "Give me a one-sentence summary of the last Dan-Julia phone call. Do not omit any crucial facts or introduce false information. ",
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
    judge = unify.Unify(
        "o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )

    if _is_summary_q(question):
        system_msg = (
            "You are a meticulous but fair summary evaluator. "
            "You will be given the *source dialogue* of a short phone call and a candidate **one-sentence** summary. "
            "Your task is to decide whether the summary accurately conveys the main intent and key factual points. "
            "A good one-sentence summary will often synthesize information from multiple utterances into a coherent statement, potentially reflecting the implied outcome or joint understanding if reasonably inferred from the dialogue. "
            "For example, if the dialogue discusses 'planning to do X' or 'working on X', the summary can state that the conversation was about 'planning X' or 'addressing X'. "
            "Minor stylistic or tense differences, re-ordering, shortened wording, or inclusion of obviously correct contextual details are also acceptable. "
            "The crucial factors are that the summary does not omit key topics discussed and does not introduce information that contradicts or is unsupported by the dialogue's intent. "
            'Respond ONLY with valid JSON of the form {"correct": true} or {"correct": false}. If false, explain why. '
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


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question", QUESTIONS)
async def test_ask_semantic_with_llm_judgement(
    question: str,
    tm_manager_scenario: tuple[TranscriptManager, dict[str, int]],
) -> None:
    """
    Calls the real `.ask()` (which itself may call the LLM multiple
    times), then asks a _separate_ LLM whether the answer is acceptable.
    """
    tm, _ID_BY_NAME = tm_manager_scenario
    handle = await tm.ask(question, _return_reasoning_steps=True)
    candidate, steps = await handle.result()
    expected = _answer_semantic(tm, question, _ID_BY_NAME)
    _llm_assert_correct(question, expected, candidate, steps)


@pytest.mark.asyncio
@pytest.mark.eval
async def test_ask_allows_interjection(
    tm_manager_scenario: tuple[TranscriptManager, dict[str, int]],
):
    """Ask one semantic question, then interject with a second, and verify both answers appear."""
    tm, _ID_BY_NAME = tm_manager_scenario
    # 1) Initial semantic query – last Dan ⇢ Julia phone call date
    q_initial = QUESTIONS[1]  # "When did Dan last speak with Julia on the phone?"
    handle = await tm.ask(q_initial, _return_reasoning_steps=True)

    # 2) Interject with a *different* question (Jimmy holiday date)
    q_follow_up = QUESTIONS[2]  # "Did Jimmy ever tell us when he's on holiday...?"
    await handle.interject(q_follow_up)

    # 3) Await combined answer
    answer, steps = await handle.result()
    expected_date_call = _answer_semantic(tm, q_initial, _ID_BY_NAME)
    expected_date_holiday = _answer_semantic(tm, q_follow_up, _ID_BY_NAME)

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
    tm = TranscriptManager()
    handle = await tm.ask(
        "List every message received from Carlos, then provide a detailed summary of each one in chronological order.",
    )
    handle.stop()
    with pytest.raises(asyncio.CancelledError):
        await handle.result()
    assert handle.done()


@pytest.mark.asyncio
@pytest.mark.eval
async def test_ask_respects_parent_context(
    tm_manager_scenario: tuple[TranscriptManager, dict[str, int]],
):
    # ── 1.  Seed a “basketball” exchange dated 2025-05-20 ───────────────
    tm, _ID_BY_NAME = tm_manager_scenario
    cid = _ID_BY_NAME
    t = datetime(2025, 5, 20, 15, 0, tzinfo=timezone.utc)

    for s, r, txt in [
        (cid["dan"], cid["julia"], "Did you catch the **basketball** game?"),
        (cid["julia"], cid["dan"], "Absolutely – great conversation!"),
    ]:
        tm.log_messages(
            {
                "medium": "phone_call",
                "sender_id": s,
                "receiver_ids": [r],
                "timestamp": t.isoformat(),
                "content": txt,
                "exchange_id": 99,
            },
        )
    tm.join_published()

    # ── 2.  Outer chat context in which `ask` will be called ────────────
    parent_ctx = [
        {
            "role": "user",
            "content": "I really enjoyed our conversation about basketball last week.",
        },
        {"role": "assistant", "content": "Me too."},
    ]

    # ── 3.  Call `.ask()` with that context ────────────────────────────
    handle = await tm.ask(
        "What date was the conversation referenced in the parent context?",
        _return_reasoning_steps=True,
        parent_chat_context=parent_ctx,
    )
    answer, steps = await handle.result()

    # ── 4.  Assertions ─────────────────────────────────────────────────
    # a) Broader-context header is present
    assert any(m.get("_ctx_header") for m in steps), "System context header missing."
    # b) LLM judged answer correct
    expected = "2025-05-20"
    _llm_assert_correct("What date was the conversation?", expected, answer, steps)


@pytest.mark.asyncio
async def test_ask_requests_clarification_when_context_missing(
    tm_manager_scenario: tuple[TranscriptManager, dict[str, int]],
) -> None:
    """
    Without a *parent_chat_context* the assistant should realise it does not
    know which conversation the user means and therefore invoke its internal
    `request_clarification` helper.  The question must bubble up via
    `clarification_up_q`; the supplied answer then flows back down and the
    tool use continues to completion.
    """

    tm, _ID_BY_NAME = tm_manager_scenario

    # ── 1.  Seed a short "basketball" conversation on 2025-05-20 ───────────
    t_conv_basketball = datetime(2025, 5, 20, 18, 0, tzinfo=timezone.utc)
    t_conv_holiday = datetime(2025, 5, 25, 20, 0, tzinfo=timezone.utc)
    dan, julia = _ID_BY_NAME["dan"], _ID_BY_NAME["julia"]

    for s, r, txt in [
        (dan, julia, "Did you catch the basketball game last night?"),
        (julia, dan, "Absolutely – it was great!"),
    ]:
        tm.log_messages(
            {
                "medium": "phone_call",
                "sender_id": s,
                "receiver_ids": [r],
                "timestamp": t_conv_basketball.isoformat(),
                "content": txt,
                "exchange_id": 123,
            },
        )

    for s, r, txt in [
        (dan, julia, "When are you next going on holiday?"),
        (julia, dan, "I'm hoping to go in August, but lets see what my boss says."),
    ]:
        tm.log_messages(
            {
                "medium": "email",
                "sender_id": s,
                "receiver_ids": [r],
                "timestamp": t_conv_holiday.isoformat(),
                "content": txt,
                "exchange_id": 321,
            },
        )
    tm.join_published()

    # ── 2.  Prepare clarification channels ────────────────────────────────
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    # ── 3.  Start a dynamic clarification responder (may handle multiple turns) ─
    saw_first_clarification = asyncio.Event()

    original_user_query = (
        "What day was the conversation? Request a clarification if you're unsure."
    )
    known_conversations = [
        {
            "topic": "basketball",
            "medium": "phone_call",
            "date": "2025-05-20",
            "exchange_id": 123,
        },
        {
            "topic": "holiday",
            "medium": "email",
            "date": "2025-05-25",
            "exchange_id": 321,
        },
    ]

    async def _clarification_worker() -> None:
        clarifier = unify.Unify(
            "o4-mini@openai",
            cache=SETTINGS.UNIFY_CACHE,
            traced=SETTINGS.UNIFY_TRACED,
        )
        clarifier.set_system_message(
            "You are a helpful assistant that answers clarification questions succinctly. "
            "You know about two possible conversations: a basketball phone call on 2025-05-20 (exchange 123), "
            "and a holiday email on 2025-05-25 (exchange 321). "
            "When asked to disambiguate which conversation the user means, prefer the basketball conversation. "
            "Keep responses short and directly disambiguate (e.g., 'The basketball phone call last week.').",
        )

        while True:
            try:
                q = await asyncio.wait_for(up_q.get(), timeout=60)
            except asyncio.TimeoutError:
                break

            if not saw_first_clarification.is_set():
                saw_first_clarification.set()

            payload = _dumps(
                {
                    "clarification_question": q,
                    "original_user_query": original_user_query,
                    "known_conversations": known_conversations,
                },
                indent=2,
            )
            try:
                answer = clarifier.generate(payload)
            except Exception:
                # Fallback deterministic answer if LLM call fails
                answer = "The basketball phone call last week."

            await down_q.put(answer.strip())

    worker_task = asyncio.create_task(_clarification_worker())

    # ── 4.  Call `.ask()` WITHOUT parent context ───────────────────────────
    handle = await tm.ask(
        original_user_query,
        _return_reasoning_steps=True,
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    # Ensure at least one clarification was requested
    await asyncio.wait_for(saw_first_clarification.wait(), timeout=60)

    # ── 5.  Await final answer and reasoning steps ─────────────────────────
    answer, steps = await handle.result()

    # Stop the worker
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    # ── 7.  Initial step roles ─────────────────────────
    assert steps[0]["role"] == "system"
    assert steps[1]["role"] == "user"

    # ── 8.  Assistant responds ─────────────────────────
    assert steps[-1]["role"] == "assistant"
    assert steps[-1]["tool_calls"] is None

    # ── 9.  Evaluate – should return the correct date 2025-05-20 ───────────
    expected = "2025-05-20"

    judge = unify.Unify(
        "o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    judge.set_system_message(
        'Answer strictly with JSON: {"correct": true|false} – '
        "true iff the candidate contains the exact date 2025-05-20, "
        "but this can be expressed in any date format (does **not** need to be yyyy-mm-dd)",
    )
    verdict = judge.generate(
        _dumps(
            {"candidate": answer},
            indent=2,
        ),
    )
    is_ok = json.loads(verdict[verdict.find("{") : verdict.rfind("}") + 1])["correct"]
    assert is_ok is True, assertion_failed(
        expected,
        answer,
        steps,
        "Clarification flow failed",
    )
