"""
Complex English-text integration tests for TaskListManager.update
===============================================================

Each test seeds a project with a small set of tasks, issues a human-like
instruction via the *public* `.update()` method and asserts that the mutated
state matches expectations.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from tests.helpers import _handle_project
from unity.task_list_manager.task_list_manager import TaskListManager
from unity.task_list_manager.types.priority import Priority
from unity.task_list_manager.types.schedule import Schedule


# --------------------------------------------------------------------------- #
#  Helper to seed a deterministic task set                                   #
# --------------------------------------------------------------------------- #


def _seed_basic_tasks(tlm: TaskListManager) -> List[int]:
    """Return list of task-ids in creation order."""

    ids = []
    ids.append(
        tlm._create_task(
            name="Write quarterly report",
            description="Draft the Q2 report (send email to finance).",
            status="active",
        ),
    )
    ids.append(
        tlm._create_task(
            name="Prepare slide deck",
            description="Create slides for the board meeting. Email once done.",
            status="queued",
        ),
    )
    ids.append(
        tlm._create_task(
            name="Client follow-up email",
            description="Send email to prospective client about proposal.",
            status="queued",
        ),
    )
    return ids


# --------------------------------------------------------------------------- #
#  1.  Re-ordering in the runnable queue                                     #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.eval
@pytest.mark.timeout(240)
def test_update_reorder_queue():
    tlm = TaskListManager()
    tlm.start()

    ids = _seed_basic_tasks(tlm)
    assert [t.task_id for t in tlm._get_task_queue()] == ids  # initial order

    tlm.update(text="Could you do Client follow-up email after Write quarterly report?")

    queue = [t.task_id for t in tlm._get_task_queue()]
    # expected order: 0 (report) -> 2 (follow-up) -> 1 (slides)
    assert queue == [ids[0], ids[2], ids[1]]


# --------------------------------------------------------------------------- #
# 2. Cancel all tasks whose description mentions sending emails              #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.eval
@pytest.mark.timeout(240)
def test_update_cancel_email_tasks():
    tlm = TaskListManager()
    tlm.start()

    _seed_basic_tasks(tlm)

    tlm.update(text="Please cancel all tasks related to sending emails.")

    tasks = tlm._search()
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


@_handle_project
@pytest.mark.eval
@pytest.mark.timeout(240)
def test_update_lower_priority_next_monday():
    tlm = TaskListManager()
    tlm.start()

    # create one scheduled next Monday with high priority
    base = datetime.now(timezone.utc)
    next_mon = _next_weekday(base, 0).replace(hour=9, minute=0, second=0, microsecond=0)

    sched = Schedule(start_time=next_mon.isoformat(), prev_task=None, next_task=None)
    tlm._create_task(
        name="Send KPI report",
        description="Automated email of KPIs to leadership.",
        schedule=sched,
        priority=Priority.high,
    )

    tlm.update(
        text="Please lower the priority of all tasks which are scheduled for next Monday.",
    )

    task = tlm._search()[0]
    assert task["priority"] == Priority.normal


# --------------------------------------------------------------------------- #
# 4. Bulk description edit (regex-like replace)                              #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.eval
@pytest.mark.timeout(240)
def test_update_bulk_description_replace():
    tlm = TaskListManager()
    tlm.start()

    tlm._create_task(
        name="Arrange viewing",
        description="Contact the estate agent to arrange the viewing.",
    )
    tlm._create_task(
        name="Send brochure",
        description="Email the estate agent the sales brochure.",
    )

    tlm.update(
        text="Please update all task descriptions to refer to Mr. Smith instead of 'the estate agent'.",
    )

    for t in tlm._search():
        assert re.search(r"Mr\.\s?Smith", t["description"]) is not None
