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
from typing import List, Dict, Any, Callable

import pytest

from tests.helpers import _handle_project
from tests.assertion_helpers import assertion_failed
from unity.task_list_manager.task_list_manager import TaskListManager
from unity.task_list_manager.types.priority import Priority
from unity.task_list_manager.types.schedule import Schedule


# Monkey patch the TaskListManager's update method to capture reasoning steps
original_update = TaskListManager.update


def patched_update(self, text: str, return_reasoning_steps: bool = False):
    try:
        result, steps = original_update(self, text, return_reasoning_steps=True)
        self._last_reasoning_steps = steps
        return result if not return_reasoning_steps else (result, steps)
    except Exception as e:
        self._last_reasoning_steps = []
        raise e


TaskListManager.update = patched_update


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

    ids = _seed_basic_tasks(tlm)
    assert [t.task_id for t in tlm._get_task_queue()] == ids  # initial order

    tlm.update(text="Could you do Client follow-up email after Write quarterly report?")

    queue = [t.task_id for t in tlm._get_task_queue()]
    expected_order = [
        ids[0],
        ids[2],
        ids[1],
    ]  # 0 (report) -> 2 (follow-up) -> 1 (slides)
    assert queue == expected_order, assertion_failed(
        expected_order,
        queue,
        getattr(tlm, "_last_reasoning_steps", []),
        "Task queue order doesn't match expected order after update",
        {"Task Data": tlm._search()},
    )


# --------------------------------------------------------------------------- #
# 2. Cancel all tasks whose description mentions sending emails              #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.eval
@pytest.mark.timeout(240)
def test_update_cancel_email_tasks():
    tlm = TaskListManager()

    _seed_basic_tasks(tlm)

    tlm.update(text="Please cancel all tasks related to sending emails.")

    tasks = tlm._search()
    for t in tasks:
        if "email" in t["description"].lower():
            assert t["status"] == "cancelled", assertion_failed(
                "cancelled",
                t["status"],
                getattr(tlm, "_last_reasoning_steps", []),
                f"Task '{t['name']}' with email in description should be cancelled",
                {"Task Data": tasks},
            )
        else:
            assert t["status"] != "cancelled", assertion_failed(
                "not cancelled",
                t["status"],
                getattr(tlm, "_last_reasoning_steps", []),
                f"Task '{t['name']}' without email in description should not be cancelled",
                {"Task Data": tasks},
            )


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
    assert task["priority"] == Priority.normal, assertion_failed(
        Priority.normal,
        task["priority"],
        getattr(tlm, "_last_reasoning_steps", []),
        f"Task '{task['name']}' scheduled for next Monday should have normal priority",
        {"Task Data": tlm._search()},
    )


# --------------------------------------------------------------------------- #
# 4. Bulk description edit (regex-like replace)                              #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.eval
@pytest.mark.timeout(240)
def test_update_bulk_description_replace():
    tlm = TaskListManager()

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
        has_mr_smith = re.search(r"Mr\.\s?Smith", t["description"]) is not None
        assert has_mr_smith, assertion_failed(
            "Description containing 'Mr. Smith'",
            t["description"],
            getattr(tlm, "_last_reasoning_steps", []),
            f"Task '{t['name']}' description should contain 'Mr. Smith'",
            {"Task Data": tlm._search()},
        )
