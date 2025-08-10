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
    task_list = task_scheduler._search_tasks()
    assert task_list == [
        {
            "name": "Promote Jeff Smith",
            "description": "Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
            "status": Status.primed,
            "trigger": None,
            "schedule": None,
            "deadline": None,
            "repeat": None,
            "priority": Priority.normal,
            "task_id": 0,
            "instance_id": 0,
            "response_policy": None,
        },
    ]


@_handle_project
@pytest.mark.unit
def test_delete_task():
    task_scheduler = TaskScheduler()

    # create
    task_scheduler._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_scheduler._search_tasks()
    assert len(task_list) == 1

    # delete
    task_scheduler._delete_task(task_id=0)
    task_list = task_scheduler._search_tasks()
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

    rows = ts._search_tasks()
    assert len(rows) == 1
    assert rows[0]["response_policy"] == policy
