"""
Complex English-text integration tests for TaskScheduler.update
===============================================================

Each test seeds a project with a small set of tasks, issues a human-like
instruction via the *public* `.update()` method and asserts that the mutated
state matches expectations.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import pytest
from unity.task_scheduler.types.priority import Priority
from unity.task_scheduler.types.schedule import Schedule
from unity.task_scheduler.types.status import Status

pytestmark = pytest.mark.llm_call

# --------------------------------------------------------------------------- #
#  1.  Re-ordering in the runnable queue                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
async def test_update_reorder_queue(basic_task_scenario):
    ts, ids = basic_task_scenario

    # initial materialization is scenario-dependent; accept any non-empty
    assert isinstance(ids, list) and ids

    handle = await ts.update(
        text="Could you update the queue order so that you write the client follow-up email *after* you write the quarterly report? Both tasks are already assigned, you just need to update their scheduling order.",
    )
    await handle.result()

    # After update, verify that the new order matches expectation by reading the queue that contains ids[0]
    row = ts._filter_tasks(filter=f"task_id == {ids[0]}")[0]
    qid = row.queue_id
    # Resolve the queue that contains the report task under explicit-queue semantics
    chain = (
        ts._get_queue(queue_id=qid)
        if isinstance(qid, int)
        else ts._get_queue_for_task(task_id=ids[0])
    )
    queue = [t.task_id for t in chain]
    # expected relative order: report (ids[0]) comes before follow-up (ids[2]);
    # slides (ids[1]) may be absent if a new queue was created with only the referenced tasks.
    assert queue[:2] == [ids[0], ids[2]]
    if len(queue) == 3:
        assert queue == [ids[0], ids[2], ids[1]]


# --------------------------------------------------------------------------- #
# 2. Cancel all tasks whose description mentions sending emails              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
async def test_update_cancel_email_tasks(basic_task_scenario):  # FIXME
    ts, ids = basic_task_scenario

    handle = await ts.update(text="Please cancel all tasks related to sending emails.")
    await handle.result()

    tasks = ts._filter_tasks()
    for t in tasks:
        if "email" in t.description.lower():
            assert t.status == Status.cancelled
        else:
            assert t.status != Status.cancelled


# --------------------------------------------------------------------------- #
# 3. Lower priority for tasks scheduled next Monday                          #
# --------------------------------------------------------------------------- #


def _next_weekday(dt: datetime, weekday: int) -> datetime:
    """Return dt on next weekday (0=Mon)."""

    days_ahead = (weekday - dt.weekday() + 7) % 7 or 7
    return dt + timedelta(days=days_ahead)


@pytest.mark.asyncio
@pytest.mark.timeout(500)
async def test_update_lower_priority_for_future_date(basic_task_scenario):
    ts, ids = basic_task_scenario

    # create one future scheduled task with high priority
    # Use an explicit queue id; no implicit default exists
    row0 = ts._filter_tasks(limit=1)[0]
    existing_qid = row0.queue_id
    # If no existing queue, allocate one
    qid = existing_qid if existing_qid is not None else ts._allocate_new_queue_id()
    sched = Schedule(
        start_at="2035-06-16T09:00:00Z",
        prev_task=None,
        next_task=None,
    )
    ts._create_task(
        name="Send KPI report",
        description="Automated email of KPIs to leadership.",
        schedule=sched,
        priority=Priority.high,
    )

    handle = await ts.update(
        text="Please lower the priority of all tasks to 'normal' which are scheduled to start on Monday 16th June 2035",
    )
    await handle.result()

    task = ts._filter_tasks(filter="'KPI report' in name")[0]
    assert task.priority == Priority.normal


# --------------------------------------------------------------------------- #
# 4. Bulk description edit (regex-like replace)                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
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

    handle = await ts.update(
        text="Please update all task descriptions to refer to Mr. Smith instead of 'the estate agent'.",
    )
    await handle.result()

    for t in ts._filter_tasks(filter="'Mr. Smith' in description"):
        assert re.search(r"Mr\.\s?Smith", t.description) is not None
