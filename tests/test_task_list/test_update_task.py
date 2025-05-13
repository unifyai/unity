from datetime import datetime, timezone, timedelta
from tests.helpers import _handle_project
from task_list_manager.types.status import Status
from task_list_manager.types.priority import Priority
from task_list_manager.task_list_manager import TaskListManager
from task_list_manager.types.repetition import RepeatPattern, Frequency, Weekday


@_handle_project
def test_update_task_name():
    task_list_manager = TaskListManager()
    task_list_manager.start()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_list_manager._search()
    assert task_list[0]["name"] == "Promote Jeff Smith"

    # rename
    task_list_manager._update_task_name(
        task_id=0,
        new_name="Give Jeff Smith a promotion",
    )
    task_list = task_list_manager._search()
    assert task_list[0]["name"] == "Give Jeff Smith a promotion"


@_handle_project
def test_update_task_description():
    task_list_manager = TaskListManager()
    task_list_manager.start()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_list_manager._search()
    assert (
        task_list[0]["description"]
        == "Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager."
    )

    # rename
    task_list_manager._update_task_description(
        task_id=0,
        new_description="Call Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_list_manager._search()
    assert (
        task_list[0]["description"]
        == "Call Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager."
    )


@_handle_project
def test_update_task_status():
    task_list_manager = TaskListManager()
    task_list_manager.start()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_list_manager._search()
    assert (
        task_list[0]["description"]
        == "Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager."
    )

    # update status
    task_list_manager._update_task_status(
        task_id=0,
        new_status=Status.cancelled,
    )
    task_list = task_list_manager._search()
    assert task_list[0]["status"] == "cancelled"


@_handle_project
def test_update_task_start_at():
    tlm = TaskListManager()
    tlm.start()

    tlm._create_task(
        name="Send customer survey",
        description="Email Q2 customer-satisfaction survey.",
    )

    start = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    tlm._update_task_start_at(task_id=0, new_start_at=start)

    task_list = tlm._search()
    assert task_list[0]["start_at"] == start


@_handle_project
def test_update_task_deadline():
    tlm = TaskListManager()
    tlm.start()

    tlm._create_task(
        name="File quarterly taxes",
        description="Prepare documents for accounting.",
    )

    deadline = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    tlm._update_task_deadline(task_id=0, new_deadline=deadline)

    task_list = tlm._search()
    assert task_list[0]["deadline"] == deadline


@_handle_project
def test_update_task_repetition():
    tlm = TaskListManager()
    tlm.start()

    tlm._create_task(
        name="Daily stand-up",
        description="10-minute team sync",
    )

    rule = RepeatPattern(frequency=Frequency.WEEKLY, interval=1, weekdays=[Weekday.MO])
    tlm._update_task_repetition(task_id=0, new_repeat=[rule])

    task_list = tlm._search()
    # The manager stores *.model_dump()* (a plain dict) so compare like-for-like
    assert task_list[0]["repeat"] == [rule.model_dump()]


@_handle_project
def test_update_task_priority():
    tlm = TaskListManager()
    tlm.start()

    tlm._create_task(
        name="Patch security vulnerability",
        description="Apply CVE-2025-1234 hot-fix to production.",
    )

    tlm._update_task_priority(task_id=0, new_priority=Priority.high)

    task_list = tlm._search()
    assert task_list[0]["priority"] == Priority.high
