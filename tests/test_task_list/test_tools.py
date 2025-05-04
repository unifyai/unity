from tests.helpers import _handle_project
from task_list_manager.task_list_manager import TaskListManager


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
            "start_at": None,
            "deadline": None,
            "repeat": None,
            "priority": None,
            "task_id": 0,
        },
    ]
