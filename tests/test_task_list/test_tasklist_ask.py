"""
Integration tests for TaskListManager.ask
================================================

Identical content moved from test_ask.py to avoid module-name collision with
TranscriptManager tests.
"""

# pylint: disable=duplicate-code

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest
import unify

from task_list_manager.task_list_manager import TaskListManager
from task_list_manager.types.priority import Priority
from task_list_manager.types.schedule import Schedule
from common.llm_helpers import _dumps


class ScenarioBuilder:
    """Populate Unify with a small, meaningful task list."""

    def __init__(self) -> None:
        if "test_task_ask" in unify.list_projects():
            unify.delete_project("test_task_ask")
        unify.activate("test_task_ask")
        self.tlm = TaskListManager()
        self.tlm.start()
        self._seed_tasks()

    def _seed_tasks(self) -> None:
        """Create five tasks with various states for robust querying."""

        self.tlm._create_task(  # Active
            name="Write quarterly report",
            description="Compile and draft the Q2 report for management.",
            status="active",
        )

        self.tlm._create_task(  # Queued
            name="Prepare slide deck",
            description="Create slides for the upcoming board meeting.",
            status="queued",
        )

        sched = Schedule(  # Scheduled
            prev_task=None,
            next_task=None,
            start_time=datetime(2025, 6, 1, 9, 0, tzinfo=timezone.utc).isoformat(),
        )
        self.tlm._create_task(
            name="Client meeting",
            description="Meet with ABC Corp for contract renewal.",
            status="scheduled",
            schedule=sched,
        )

        self.tlm._create_task(  # Paused
            name="Deploy new release",
            description="Roll out version 2.0 to production servers.",
            status="paused",
        )

        self.tlm._create_task(  # High-priority queued
            name="Hotfix security vulnerability",
            description="Apply CVE-2025-1234 patch to all services.",
            status="queued",
            priority=Priority.high,
        )


# ---------------- Ground-truth helpers ---------------- #


def _answer_semantic(tlm: TaskListManager, question: str) -> str:
    q = question.lower()
    tasks = tlm._search()

    if "currently active" in q:
        return next(t for t in tasks if t["status"] == "active")["name"]

    if "tasks are queued" in q:
        return str(sum(1 for t in tasks if t["status"] == "queued"))

    if "client meeting" in q and "scheduled" in q:
        mtg = next(t for t in tasks if "client meeting" in t["name"].lower())
        return mtg["schedule"]["start_time"].split("T")[0]

    if "priority" in q and "hotfix" in q:
        hotfix = next(t for t in tasks if "hotfix" in t["name"].lower())
        return str(hotfix["priority"])

    return "N/A"


QUESTIONS = [
    "Which task is currently active?",
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

    judge = unify.Unify("o4-mini@openai", cache=True)
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


@pytest.fixture
def tlm_scenario() -> TaskListManager:  # noqa: D401 – fixture, not function
    return ScenarioBuilder().tlm


@pytest.mark.eval
@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.timeout(180)
def test_ask_semantic_with_llm_judgement(
    question: str,
    tlm_scenario: TaskListManager,
) -> None:
    try:
        candidate, steps = tlm_scenario.ask(text=question, return_reasoning_steps=True)
        expected = _answer_semantic(tlm_scenario, question)
        _llm_assert_correct(question, expected, candidate, steps)
    except Exception as exc:
        if "test_task_ask" in unify.list_projects():
            unify.delete_project("test_task_ask")
        raise exc
