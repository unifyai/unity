import pytest
from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.priority import Priority
from unity.task_scheduler.types.status import Status


@_handle_project
@pytest.mark.unit
def test_create_task():
    task_scheduler = TaskScheduler()
    task_scheduler._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_scheduler._filter_tasks()
    assert len(task_list) == 1
    row = task_list[0]
    # After refactor, _filter_tasks returns raw JSON-serialisable values (enums as strings)
    assert row["name"] == "Promote Jeff Smith"
    assert (
        row["description"]
        == "Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager."
    )
    assert Status(row["status"]) == Status.primed
    assert row["trigger"] is None
    assert row["schedule"] is None
    assert row["deadline"] is None
    assert row["repeat"] is None
    assert Priority(row["priority"]) == Priority.normal
    assert row["task_id"] == 0
    assert row["instance_id"] == 0
    assert row["response_policy"] is None
    # New field surfaced by the Task model; should be present but unset on creation
    assert "activated_by" in row and row["activated_by"] is None


@_handle_project
@pytest.mark.unit
def test_delete_task():
    task_scheduler = TaskScheduler()

    # create
    task_scheduler._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_scheduler._filter_tasks()
    assert len(task_list) == 1

    # delete
    task_scheduler._delete_task(task_id=0)
    task_list = task_scheduler._filter_tasks()
    assert task_list == []


@_handle_project
@pytest.mark.unit
def test_create_task_with_response_policy():
    ts = TaskScheduler()

    policy = (
        "During this task, only the project manager may issue instructions. "
        "The client may view progress updates but cannot steer decisions."
    )

    ts._create_task(
        name="PM-only control",
        description="Carry out project setup steps.",
        response_policy=policy,
    )

    rows = ts._filter_tasks()
    assert len(rows) == 1
    assert rows[0]["response_policy"] == policy
