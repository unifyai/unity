from tests.helpers import _handle_project
import pytest
import unify

from unity.common.context_registry import ContextRegistry
from unity.common.tool_outcome import ToolErrorException
from unity.session_details import SESSION_DETAILS
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.priority import Priority
from unity.task_scheduler.types.status import Status


@_handle_project
def test_create_task():
    task_scheduler = TaskScheduler()
    task_scheduler._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
        entrypoint=101,
    )
    task_list = task_scheduler._filter_tasks()
    assert len(task_list) == 1
    row = task_list[0]
    # After refactor, _filter_tasks returns raw JSON-serialisable values (enums as strings)
    assert row.name == "Promote Jeff Smith"
    assert (
        row.description
        == "Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager."
    )
    assert row.status == Status.primed
    assert row.trigger is None
    assert row.schedule is None
    assert row.deadline is None
    assert row.repeat is None
    assert row.priority == Priority.normal
    assert row.task_id == 0
    assert row.instance_id == 0
    assert row.response_policy is None
    assert row.entrypoint == 101
    assert row.activated_by is None


@_handle_project
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
    assert rows[0].response_policy == policy


@_handle_project
def test_create_space_task_routes_to_shared_root():
    ts = TaskScheduler()
    SESSION_DETAILS.space_ids = [987654]

    try:
        out = ts._create_task(
            name="Shared follow-up",
            description="Coordinate with the shared project room.",
            destination="space:987654",
        )
        task_id = out["details"]["task_id"]

        rows = ts._filter_tasks(filter=f"task_id == {task_id}")
        assert len(rows) == 1
        assert rows[0].destination == "space:987654"
        assert rows[0].assistant_id == SESSION_DETAILS.assistant_context
    finally:
        try:
            unify.delete_context("Spaces/987654/Tasks")
        except Exception:
            pass
        SESSION_DETAILS.space_ids = []
        ContextRegistry.forget_departed_space_roots([])


@_handle_project
def test_invalid_space_destination_raises_tool_error():
    ts = TaskScheduler()
    SESSION_DETAILS.space_ids = [987655]

    try:
        with pytest.raises(ToolErrorException):
            ts._create_task(
                name="Bad shared task",
                description="Should not write outside membership.",
                destination="space:987656",
            )
    finally:
        SESSION_DETAILS.space_ids = []


@_handle_project
def test_duplicate_task_id_update_requires_destination():
    ts = TaskScheduler()
    SESSION_DETAILS.space_ids = [987657]

    try:
        ts._create_task(name="Personal duplicate", description="Personal root.")
        ts._create_task(
            name="Shared duplicate",
            description="Shared root.",
            destination="space:987657",
        )

        with pytest.raises(ValueError, match="provide destination"):
            ts._update_task(task_id=0, name="Ambiguous update")

        ts._update_task(
            task_id=0,
            name="Shared duplicate updated",
            destination="space:987657",
        )
        rows = ts._filter_tasks(filter="task_id == 0")
        by_destination = {row.destination or "personal": row.name for row in rows}
        assert by_destination["personal"] == "Personal duplicate"
        assert by_destination["space:987657"] == "Shared duplicate updated"
    finally:
        try:
            unify.delete_context("Spaces/987657/Tasks")
        except Exception:
            pass
        SESSION_DETAILS.space_ids = []
        ContextRegistry.forget_departed_space_roots([])


@_handle_project
def test_create_tasks_single_queue_and_ids():
    ts = TaskScheduler()

    # Create three tasks and materialize a single queue [0,1,2]
    out = ts._create_tasks(
        tasks=[
            {"name": "A", "description": "a"},
            {"name": "B", "description": "b"},
            {"name": "C", "description": "c"},
        ],
        queue_ordering=[{"order": [0, 1, 2], "queue_head": {"primed": True}}],
    )

    # Ascending task ids should match creation order
    assert out["details"]["task_ids"] == [0, 1, 2]

    # Exactly one queue created with order [0,1,2]
    queues = ts._list_queues()
    assert isinstance(queues, list) and len(queues) == 1
    qid = queues[0]["queue_id"]
    q = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in q] == [0, 1, 2]
    assert queues[0]["head_id"] == 0
    # Primed head with no start_at
    assert not (q[0].schedule and q[0].schedule.start_at)
    head_row = ts._filter_tasks(filter="task_id == 0", limit=1)[0]
    assert head_row.status == Status.primed


@_handle_project
def test_create_tasks_multi_queues_with_start_times():
    ts = TaskScheduler()

    out = ts._create_tasks(
        tasks=[
            {"name": "W", "description": "w"},
            {"name": "X", "description": "x"},
            {"name": "Y", "description": "y"},
            {"name": "Z", "description": "z"},
        ],
        queue_ordering=[
            {"order": [0, 2], "queue_head": {"start_at": "2036-01-01T10:00:00+00:00"}},
            {"order": [1, 3], "queue_head": {"start_at": "2036-01-02T10:00:00+00:00"}},
        ],
    )

    # task_ids should be assigned in ascending order
    assert out["details"]["task_ids"] == [0, 1, 2, 3]

    # Map relative queues to real queue_ids
    qinfo_by_rel = {q["relative_queue_index"]: q for q in out["details"]["queues"]}
    qid0 = qinfo_by_rel[0]["queue_id"]
    qid1 = qinfo_by_rel[1]["queue_id"]
    assert isinstance(qid0, int) and isinstance(qid1, int) and qid0 != qid1

    # First queue should be [0,2] with start_at on head and scheduled status
    q0 = ts._get_queue(queue_id=qid0)
    assert [t.task_id for t in q0] == [0, 2]
    assert q0[0].schedule and q0[0].schedule.start_at
    assert q0[0].schedule.start_at.isoformat() == "2036-01-01T10:00:00+00:00"
    row0 = ts._filter_tasks(filter="task_id == 0", limit=1)[0]
    assert row0.status == Status.scheduled

    # Second queue should be [1,3] with scheduled head (explicit start_at provided)
    q1 = ts._get_queue(queue_id=qid1)
    assert [t.task_id for t in q1] == [1, 3]
    assert q1[0].schedule and q1[0].schedule.start_at
    assert q1[0].schedule.start_at.isoformat() == "2036-01-02T10:00:00+00:00"
    row1 = ts._filter_tasks(filter="task_id == 1", limit=1)[0]
    assert row1.status == Status.scheduled


@_handle_project
def test_task_scheduler_clear():
    ts = TaskScheduler()

    # Seed a couple of tasks
    out1 = ts._create_task(name="Alpha", description="alpha desc")
    out2 = ts._create_task(name="Beta", description="beta desc")
    id1 = out1["details"]["task_id"]
    id2 = out2["details"]["task_id"]
    # Fresh contexts start from 0 and increment
    assert id1 == 0 and id2 == 1

    # Sanity: tasks present before clear
    before = ts._filter_tasks()
    assert before and len(before) == 2

    # Execute clear
    ts.clear()

    # After clear: no tasks
    after = ts._filter_tasks()
    assert after == []

    # Creating again should work and ids should restart from 0
    out3 = ts._create_task(name="Gamma", description="gamma desc")
    assert out3["details"]["task_id"] == 0
