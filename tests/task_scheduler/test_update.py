import pytest

pytestmark = pytest.mark.eval

from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.status import Status
from unity.task_scheduler.types.priority import Priority


@_handle_project
@pytest.mark.asyncio
async def test_update_create_task_via_text():
    ts = TaskScheduler()

    cmd = (
        "Please add a new task called 'Promote Jeff Smith' with the "
        "description 'Send an email to Jeff Smith, kindly congratulating him and "
        "explaining that he has been promoted from sales rep to sales manager.'"
    )
    handle = await ts.update(text=cmd)
    await handle.result()

    tasks = ts._filter_tasks()
    assert len(tasks) == 1
    task = tasks[0]
    assert task.name == "Promote Jeff Smith"
    assert task.description.startswith("Send an email to Jeff Smith")
    assert task.status in (
        Status.primed,
        Status.queued,
        Status.triggerable,
        Status.active,
    )
    assert task.priority == Priority.normal


@_handle_project
@pytest.mark.asyncio
async def test_update_delete_task_via_text():
    ts = TaskScheduler()

    # create a task directly (bypassing LLM) so we know the ID is 0
    ts._create_task(
        name="Write quarterly report",
        description="Compile and draft the Q2 report for management.",
    )
    assert len(ts._filter_tasks()) == 1

    # delete via plain-English update
    handle = await ts.update(text="Delete the task with id 0.")
    await handle.result()

    assert ts._filter_tasks() == []
