"""
Integration tests for TaskScheduler.ask
================================================

Identical content moved from test_ask.py to avoid module-name collision with
TranscriptManager tests.
"""

# pylint: disable=duplicate-code

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
import os

import pytest
import unify

from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.priority import Priority
from unity.task_scheduler.types.schedule import Schedule
from unity.common.llm_helpers import _dumps
from tests.assertion_helpers import assertion_failed
from tests.helpers import SETTINGS


class ScenarioBuilder:
    """Populate Unify with a small, meaningful task list."""

    def __init__(self) -> None:
        self.ts = TaskScheduler()
        self._seed_tasks()

    def _seed_tasks(self) -> None:
        """Create five tasks with various states for robust querying."""

        self.ts._create_task(  # Active
            name="Write quarterly report",
            description="Compile and draft the Q2 report for management.",
            status="primed",
        )

        self.ts._create_task(  # Queued
            name="Prepare slide deck",
            description="Create slides for the upcoming board meeting.",
            status="queued",
        )

        sched = Schedule(  # Scheduled
            prev_task=None,
            next_task=None,
            start_at=datetime(2050, 6, 1, 9, 0, tzinfo=timezone.utc).isoformat(),
        )
        self.ts._create_task(
            name="Client meeting",
            description="Meet with ABC Corp for contract renewal.",
            status="scheduled",
            schedule=sched,
        )

        self.ts._create_task(  # Paused
            name="Deploy new release",
            description="Roll out version 2.0 to production servers.",
            status="paused",
        )

        self.ts._create_task(  # High-priority queued
            name="Hotfix security vulnerability",
            description="Apply CVE-2025-1234 patch to all services.",
            status="queued",
            priority=Priority.high,
        )


# ---------------- Ground-truth helpers ---------------- #


def _answer_semantic(ts: TaskScheduler, question: str) -> str:
    q = question.lower()
    tasks = ts._filter_tasks()

    if "currently primed" in q:
        return next(t for t in tasks if t["status"] == "primed")["name"]

    if "tasks are queued" in q:
        return str(sum(1 for t in tasks if t["status"] == "queued"))

    if "client meeting" in q and "scheduled" in q:
        mtg = next(t for t in tasks if "client meeting" in t["name"].lower())
        return mtg["schedule"]["start_at"].split("T")[0]

    if "priority" in q and "hotfix" in q:
        hotfix = next(t for t in tasks if "hotfix" in t["name"].lower())
        return str(hotfix["priority"])

    return "N/A"


QUESTIONS = [
    "Which task is currently primed?",
    "How many tasks are queued at the moment?",
    "When is the client meeting scheduled for?",
    "What is the priority level of the hotfix task?",
]


def _llm_assert_correct(
    question: str,
    expected: str,
    candidate: str,
    steps: list,  # noqa: D401 – clarity outweighs strict type accuracy
) -> None:
    """Assert *candidate* satisfies *expected* for *question* via an LLM judge.

    On failure, the full reasoning *steps* are appended to the assertion
    message to aid debugging.
    """

    judge = unify.Unify(
        "o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    judge.set_system_message(
        "You are a strict unit-test judge. "
        "You will be given a question, a ground-truth answer derived directly "
        "from the data, and a candidate answer produced by the system under test. "
        'Respond ONLY with valid JSON of the form {"correct": true} or {"correct": false}. '
        "Mark correct⇢true if a reasonable human would accept the candidate as answering the question fully and accurately; otherwise false.",
    )

    payload = _dumps(
        {"question": question, "ground_truth": expected, "candidate": candidate},
        indent=4,
    )
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

    if os.environ.get("UNIFY_DELETE_CONTEXT_ON_EXIT", "false").lower() == "true":
        unify.delete_context(ctx)


@pytest.fixture(scope="session")
def ts_scenario(
    setup_session_context,
) -> TaskScheduler:  # noqa: D401 – fixture, not function
    return ScenarioBuilder().ts


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.timeout(300)
async def test_ask_semantic_with_llm_judgement(
    question: str,
    ts_scenario: TaskScheduler,
) -> None:
    try:
        handle = await ts_scenario.ask(
            text=question,
            _return_reasoning_steps=True,
        )
        candidate, steps = await handle.result()
        expected = _answer_semantic(ts_scenario, question)
        _llm_assert_correct(question, expected, candidate, steps)
    except Exception as exc:
        raise exc


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_ask_with_interjection(ts_scenario: TaskScheduler) -> None:
    """Ask a question, interject with a follow-up, and ensure the final answer covers both."""
    try:
        # 1) Initial question ⇢ primed task name
        handle = await ts_scenario.ask(
            text="Which task is currently primed?",
            _return_reasoning_steps=True,
        )

        # 2) Mid-conversation interjection ⇢ queued-task count
        await handle.interject("Also, how many tasks are queued?")

        # 3) Await combined answer
        answer, steps = await handle.result()
        primed_task = _answer_semantic(
            ts_scenario,
            QUESTIONS[0],
        )  # "Write quarterly report"
        queued_cnt = _answer_semantic(ts_scenario, QUESTIONS[1])  # e.g. "2"

        # 4) Assert presence of both pieces of information
        assert primed_task.lower() in answer.lower(), assertion_failed(
            f"Answer containing primed task '{primed_task}'",
            answer,
            steps,
            "Active task not mentioned in combined answer",
        )
        assert queued_cnt in answer, assertion_failed(
            f"Answer containing queued count '{queued_cnt}'",
            answer,
            steps,
            "Queued count not mentioned in combined answer",
        )
    except Exception as exc:
        raise exc


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_ask_stop(ts_scenario: TaskScheduler) -> None:
    """Test that we can stop the conversation mid-way."""
    try:
        # Start with a request that would take some time to complete
        handle = await ts_scenario.ask(
            text="List all tasks, then summarize each one in detail.",
        )

        # Give the LLM a moment to start processing, then stop it
        await asyncio.sleep(0.05)
        handle.stop(cancel=True)

        with pytest.raises(asyncio.CancelledError):
            await handle.result()
        assert handle.done()
    except Exception as exc:
        raise exc
