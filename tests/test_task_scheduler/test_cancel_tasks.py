from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
import pytest


@_handle_project
@pytest.mark.unit
def test_cancel_single_task():
    """Cancelling a single active task should set its status to 'cancelled'."""
    ts = TaskScheduler()

    # Create an active task (id will be 0)
    ts._create_task(
        name="Follow-up with client",
        description="Send a thank-you email and next-steps proposal.",
    )

    # Cancel the task
    ts._cancel_tasks([0])

    # Verify the task was cancelled
    tasks = ts._filter_tasks()
    assert tasks[0]["status"] == "cancelled"


@_handle_project
@pytest.mark.unit
def test_cancel_multiple_tasks():
    """Cancelling multiple tasks at once should update all of their statuses."""
    ts = TaskScheduler()

    # Create two tasks (ids 0 and 1)
    ts._create_task(
        name="Prepare quarterly report",
        description="Compile Q1 financials into slide deck.",
    )
    ts._create_task(
        name="Schedule team off-site",
        description="Book venue and send calendar invites.",
    )

    # Cancel both tasks
    ts._cancel_tasks([0, 1])

    # Verify both tasks were cancelled
    tasks = ts._filter_tasks()
    status_by_id = {t["task_id"]: t["status"] for t in tasks}
    assert status_by_id[0] == "cancelled"
    assert status_by_id[1] == "cancelled"


@_handle_project
@pytest.mark.unit
def test_cancel_completed_task_raises():
    """Attempting to cancel a task that is already completed should raise an AssertionError."""
    ts = TaskScheduler()

    # Create a task that is already completed
    ts._create_task(
        name="Ship version 1.0",
        description="Publish release notes and push tags.",
        status="completed",
    )

    # Expect an AssertionError when trying to cancel it
    with pytest.raises(AssertionError):
        ts._cancel_tasks([0])
