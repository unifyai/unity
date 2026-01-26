import pytest
from datetime import datetime, timezone, timedelta
from tests.helpers import _handle_project
from unity.task_scheduler.types.status import Status
from unity.task_scheduler.types.priority import Priority
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.repetition import RepeatPattern, Frequency, Weekday
from unity.task_scheduler.types.schedule import Schedule


@_handle_project
def test_update_task_name():
    task_scheduler = TaskScheduler()

    # create
    task_scheduler._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_scheduler._filter_tasks()
    assert task_list[0].name == "Promote Jeff Smith"

    # rename
    task_scheduler._update_task(
        task_id=0,
        name="Give Jeff Smith a promotion",
    )
    task_list = task_scheduler._filter_tasks()
    assert task_list[0].name == "Give Jeff Smith a promotion"


@_handle_project
def test_update_task_description():
    task_scheduler = TaskScheduler()

    # create
    task_scheduler._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_scheduler._filter_tasks()
    assert (
        task_list[0].description
        == "Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager."
    )

    # rename
    task_scheduler._update_task(
        task_id=0,
        description="Call Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_scheduler._filter_tasks()
    assert (
        task_list[0].description
        == "Call Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager."
    )


@_handle_project
def test_update_task_status():
    task_scheduler = TaskScheduler()

    # create
    task_scheduler._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_scheduler._filter_tasks()
    assert (
        task_list[0].description
        == "Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager."
    )

    # update status
    task_scheduler._update_task(
        task_id=0,
        status=Status.cancelled,
    )
    task_list = task_scheduler._filter_tasks()
    assert task_list[0].status == Status.cancelled


@_handle_project
def test_head_of_queue_scheduled_cannot_be_queued():
    """A task at the queue head with an explicit start_at must stay 'scheduled'."""

    ts = TaskScheduler()

    # Create a task that sits at the head of the queue with a fixed start time
    future_start = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

    sched = Schedule(start_at=future_start, prev_task=None, next_task=None)

    tid = ts._create_task(
        name="Launch campaign",
        description="Kick-off marketing campaign on launch date.",
        schedule=sched,
    )["details"]["task_id"]

    # Sanity: the task should have been stored as 'scheduled'
    task_row = ts._filter_tasks(filter=f"task_id == {tid}", limit=1)[0]
    assert task_row.status == Status.scheduled

    # Attempting to mark it as 'queued' must fail
    with pytest.raises(ValueError):
        ts._update_task(task_id=tid, status="queued")


@_handle_project
def test_update_task_start_at():
    ts = TaskScheduler()

    ts._create_task(
        name="Send customer survey",
        description="Email Q2 customer-satisfaction survey.",
    )

    start = datetime.now(timezone.utc) + timedelta(days=1)
    ts._update_task(task_id=0, start_at=start)

    task_list = ts._filter_tasks()
    assert task_list[0].schedule is not None
    assert task_list[0].schedule.start_at == start


@_handle_project
def test_update_task_deadline():
    ts = TaskScheduler()

    ts._create_task(
        name="File quarterly taxes",
        description="Prepare documents for accounting.",
    )

    deadline = datetime.now(timezone.utc) + timedelta(days=30)
    ts._update_task(task_id=0, deadline=deadline)

    task_list = ts._filter_tasks()
    assert task_list[0].deadline == deadline


@_handle_project
def test_update_task_repetition():
    ts = TaskScheduler()

    ts._create_task(
        name="Daily stand-up",
        description="10-minute team sync",
    )

    rule = RepeatPattern(frequency=Frequency.WEEKLY, interval=1, weekdays=[Weekday.MO])
    ts._update_task(task_id=0, repeat=[rule])

    task_list = ts._filter_tasks()
    assert task_list[0].repeat[0] == rule


@_handle_project
def test_update_task_priority():
    ts = TaskScheduler()

    ts._create_task(
        name="Patch security vulnerability",
        description="Apply CVE-2025-1234 hot-fix to production.",
    )

    ts._update_task(task_id=0, priority=Priority.high)

    task_list = ts._filter_tasks()
    assert task_list[0].priority == Priority.high
