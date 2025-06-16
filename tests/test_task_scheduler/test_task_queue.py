import pytest
from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.schedule import Schedule


# Convenience to make schedules quickly
def _sch(prev_, next_):
    from datetime import datetime, timezone

    return Schedule(
        prev_task=prev_,
        next_task=next_,
        start_at=datetime.now(timezone.utc).isoformat(),
    )


@_handle_project
@pytest.mark.unit
def test_get_queue_and_reorder():
    ts = TaskScheduler()

    # -----  create three queued tasks with an explicit chain  -----
    t0 = ts._create_task(
        name="T0",
        description="first",
        schedule=_sch(None, 1),
    )
    t1 = ts._create_task(
        name="T1",
        description="second",
        schedule=_sch(0, 2),
    )
    t2 = ts._create_task(
        name="T2",
        description="third",
        schedule=_sch(1, None),
    )

    queue = ts._get_task_queue()
    assert [t.task_id for t in queue] == [0, 1, 2]

    # -----  swap the order (0,2,1)  -----
    ts._update_task_queue(original=[0, 1, 2], new=[0, 2, 1])

    new_q = ts._get_task_queue()
    assert [t.task_id for t in new_q] == [0, 2, 1]


@_handle_project
@pytest.mark.unit
def test_insert_into_queue():
    ts = TaskScheduler()

    # base queue with one task
    ts._create_task(name="base", description="x", schedule=_sch(-1, -1))

    # create a brand-new task that will be inserted
    new_id = ts._create_task(name="insert-me", description="y")

    ts._update_task_queue(original=[0], new=[0, new_id])

    q = ts._get_task_queue()
    assert [t.task_id for t in q] == [0, new_id]
    # also check the linkage of node 0 -> new_id
    assert q[0].schedule.next_task == new_id
