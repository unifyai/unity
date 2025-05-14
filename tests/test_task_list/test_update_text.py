from tests.helpers import _handle_project
from task_list_manager.task_list_manager import TaskListManager
from task_list_manager.types.status import Status
from task_list_manager.types.priority import Priority


@_handle_project
def test_update_create_task_via_text():
    tlm = TaskListManager()
    tlm.start()

    cmd = (
        "Please add a new task called 'Promote Jeff Smith' with the "
        "description 'Send an email to Jeff Smith, kindly congratulating him and "
        "explaining that he has been promoted from sales rep to sales manager.'"
    )
    tlm.update(text=cmd)

    tasks = tlm._search()
    assert len(tasks) == 1
    task = tasks[0]
    assert task["name"] == "Promote Jeff Smith"
    assert task["description"].startswith("Send an email to Jeff Smith")
    assert task["status"] in (Status.active, Status.queued, "active", "queued")
    assert task["priority"] == Priority.normal


@_handle_project
def test_update_delete_task_via_text():
    tlm = TaskListManager()
    tlm.start()

    # create a task directly (bypassing LLM) so we know the ID is 0
    tlm._create_task(
        name="Write quarterly report",
        description="Compile and draft the Q2 report for management.",
    )
    assert len(tlm._search()) == 1

    # delete via plain-English update
    tlm.update(text="Delete the task with id 0.")

    assert tlm._search() == []
