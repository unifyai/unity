"""
Complex English-text integration tests for TaskScheduler.update
===============================================================

Each test seeds a project with a small set of tasks, issues a human-like
instruction via the *public* `.update()` method and asserts that the mutated
state matches expectations.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import List
import os

import pytest
import unify
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.priority import Priority
from unity.task_scheduler.types.schedule import Schedule


# --------------------------------------------------------------------------- #
#  Helper to seed a deterministic task set                                   #
# --------------------------------------------------------------------------- #


def _seed_basic_tasks(ts: TaskScheduler) -> List[int]:
    """Return list of task-ids in creation order."""

    ids = []
    ids.append(
        ts._create_task(
            name="Write quarterly report",
            description="Draft the Q2 report (send email to finance).",
            status="primed",
        ),
    )
    ids.append(
        ts._create_task(
            name="Prepare slide deck",
            description="Create slides for the board meeting. Email once done.",
            status="queued",
        ),
    )
    ids.append(
        ts._create_task(
            name="Client follow-up email",
            description="Send email to prospective client about proposal.",
            status="queued",
        ),
    )
    return ids


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
def basic_task_scenario(setup_session_context):
    """
    Snapshot task state before each test that uses basic_task_scenario and restore after.
    """
    ts = TaskScheduler()
    ids = _seed_basic_tasks(ts)
    snapshot = ts._get_tasks()
    yield ts, ids
    new_snapshot = ts._get_tasks()
    for t_original, t_new in zip(snapshot, new_snapshot):
        if t_original["name"] != t_new["name"]:
            ts._update_task_name(
                task_id=t_original["task_id"],
                new_name=t_original["name"],
            )
        if t_original["description"] != t_new["description"]:
            ts._update_task_description(
                task_id=t_original["task_id"],
                new_description=t_original["description"],
            )
        if t_original["status"] != t_new["status"]:
            ts._update_task_status(
                task_ids=[t_original["task_id"]],
                new_status=t_original["status"],
            )
        if t_original["priority"] != t_new["priority"]:
            ts._update_task_priority(
                task_id=t_original["task_id"],
                new_priority=t_original["priority"],
            )
        if t_original["deadline"] != t_new["deadline"]:
            ts._update_task_deadline(
                task_id=t_original["task_id"],
                new_deadline=t_original["deadline"],
            )
        if t_original["repeat"] != t_new["repeat"]:
            ts._update_task_repetition(
                task_id=t_original["task_id"],
                new_repeat=t_original["repeat"],
            )


# --------------------------------------------------------------------------- #
#  1.  Re-ordering in the runnable queue                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_update_reorder_queue(basic_task_scenario):
    ts, ids = basic_task_scenario

    assert [t.task_id for t in ts._get_task_queue()] == ids  # initial order

    handle = ts.update(
        text="Could you update the queue order so that you write the client follow-up email *after* you write the quarterly report? Both tasks are already assigned, you just need to update their scheduling order.",
    )
    await handle.result()

    queue = [t.task_id for t in ts._get_task_queue()]
    # expected order: 0 (report) -> 2 (follow-up) -> 1 (slides)
    assert queue == [ids[0], ids[2], ids[1]]


# --------------------------------------------------------------------------- #
# 2. Cancel all tasks whose description mentions sending emails              #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_update_cancel_email_tasks(basic_task_scenario):  # FIXME
    ts, ids = basic_task_scenario

    handle = ts.update(text="Please cancel all tasks related to sending emails.")
    await handle.result()

    tasks = ts._search()
    for t in tasks:
        if "email" in t["description"].lower():
            assert t["status"] == "cancelled"
        else:
            assert t["status"] != "cancelled"


# --------------------------------------------------------------------------- #
# 3. Lower priority for tasks scheduled next Monday                          #
# --------------------------------------------------------------------------- #


def _next_weekday(dt: datetime, weekday: int) -> datetime:
    """Return dt on next weekday (0=Mon)."""

    days_ahead = (weekday - dt.weekday() + 7) % 7 or 7
    return dt + timedelta(days=days_ahead)


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_update_lower_priority_next_monday(basic_task_scenario):
    ts, ids = basic_task_scenario

    # create one scheduled next Monday with high priority
    base = datetime.now(timezone.utc)
    next_mon = _next_weekday(base, 0).replace(hour=9, minute=0, second=0, microsecond=0)

    sched = Schedule(start_time=next_mon.isoformat(), prev_task=None, next_task=None)
    ts._create_task(
        name="Send KPI report",
        description="Automated email of KPIs to leadership.",
        schedule=sched,
        priority=Priority.high,
    )

    handle = ts.update(
        text="Please lower the priority of all tasks which are scheduled for next Monday.",
    )
    await handle.result()

    task = ts._search(filter="'KPI report' in name")[0]
    assert task["priority"] == Priority.normal


# --------------------------------------------------------------------------- #
# 4. Bulk description edit (regex-like replace)                              #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_update_bulk_description_replace(basic_task_scenario):
    ts, ids = basic_task_scenario

    ts._create_task(
        name="Arrange viewing",
        description="Contact the estate agent to arrange the viewing.",
    )
    ts._create_task(
        name="Send brochure",
        description="Email the estate agent the sales brochure.",
    )

    handle = ts.update(
        text="Please update all task descriptions to refer to Mr. Smith instead of 'the estate agent'.",
    )
    await handle.result()

    for t in ts._search(filter="'Mr. Smith' in description"):
        assert re.search(r"Mr\.\s?Smith", t["description"]) is not None
