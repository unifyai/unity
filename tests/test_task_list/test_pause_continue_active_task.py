import pytest
from tests.helpers import _handle_project
from unity.task_list_manager.task_list_manager import TaskListManager


@_handle_project
@pytest.mark.unit
def test_get_paused_task():
    task_list_manager = TaskListManager()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
        status="paused",
    )
    task_list = task_list_manager._search()
    assert task_list[0]["name"] == "Promote Jeff Smith"

    # verify it's the same task
    paused_task = task_list_manager._get_paused_task()
    assert paused_task["task_id"] == 0


@_handle_project
@pytest.mark.unit
def test_get_active_task():
    task_list_manager = TaskListManager()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
        status="active",
    )
    task_list = task_list_manager._search()
    assert task_list[0]["name"] == "Promote Jeff Smith"

    # verify it's the same task
    paused_task = task_list_manager._get_active_task()
    assert paused_task["task_id"] == 0


@_handle_project
@pytest.mark.unit
def test_pause():
    task_list_manager = TaskListManager()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
        status="active",
    )
    task_list = task_list_manager._search()
    assert task_list[0]["name"] == "Promote Jeff Smith"

    # verify the task is active
    paused_task = task_list_manager._get_active_task()
    assert paused_task["task_id"] == 0

    # pause the task
    task_list_manager._pause()

    # verify there is no active task
    assert task_list_manager._get_active_task() is None

    # verify there is a paused task
    assert task_list_manager._get_paused_task()


@_handle_project
@pytest.mark.unit
def test_continue():
    task_list_manager = TaskListManager()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
        status="paused",
    )
    task_list = task_list_manager._search()
    assert task_list[0]["name"] == "Promote Jeff Smith"

    # verify the task is paused
    paused_task = task_list_manager._get_paused_task()
    assert paused_task["task_id"] == 0

    # pause the task
    task_list_manager._continue()

    # verify there is no paused task
    assert task_list_manager._get_paused_task() is None

    # verify there is an active task
    assert task_list_manager._get_active_task()
