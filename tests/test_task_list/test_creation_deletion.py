from tests.helpers import _handle_project
from task_list_manager.task_list_manager import TaskListManager
from task_list_manager.types.priority import Priority
from task_list_manager.types.status import Status


@_handle_project
def test_create_task():
    task_list_manager = TaskListManager()
    task_list_manager.start()
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_list_manager._search()
    assert task_list == [
        {
            "name": "Promote Jeff Smith",
            "description": "Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
            "status": Status.active,
            "schedule": {"prev_task": None, "next_task": None},
            "deadline": None,
            "repeat": None,
            "priority": Priority.normal,
            "task_id": 0,
        },
    ]


@_handle_project
def test_delete_task():
    task_list_manager = TaskListManager()
    task_list_manager.start()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_list_manager._search()
    assert len(task_list) == 1

    # delete
    task_list_manager._delete_task(task_id=0)
    task_list = task_list_manager._search()
    assert task_list == []
