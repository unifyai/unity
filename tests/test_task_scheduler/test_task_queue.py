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

    # base task
    ts._create_task(name="base", description="x")

    # create a brand-new task that will be inserted
    new_id = ts._create_task(name="insert-me", description="y")["details"]["task_id"]

    ts._update_task_queue(original=[], new=[0, new_id])

    q = ts._get_task_queue()
    assert [t.task_id for t in q] == [0, new_id]
    # also check the linkage of node 0 -> new_id
    assert q[0].schedule.next_task == new_id


# ---------------------------------------------------------------------------#
#  Additional invariants: `start_at` sticks to the queue head                #
# ---------------------------------------------------------------------------#


# Helper – reusable assertion
def _assert_head_owns_timestamp(queue):
    """
    Check that **exactly one** task in *queue* has a non-null
    ``schedule.start_at`` and that that task is the head (index 0).
    """
    non_null = [i for i, t in enumerate(queue) if t.schedule and t.schedule.start_at]
    assert non_null == [0], f"Expected timestamp only on head, got indices {non_null}"


@_handle_project
@pytest.mark.unit
def test_start_time_moves_with_front_swap():
    """
    Queue = [A(start_at), B, C] → reorder to [C(start_at), A, B].
    The `start_at` value must migrate to C and be removed from all others.
    """
    ts = TaskScheduler()

    ts._create_task(
        name="A",
        description="first",
        schedule=Schedule(
            prev_task=None,
            next_task=1,
            start_at="2025-06-23T09:00:00+00:00",
        ),
    )
    ts._create_task(
        name="B",
        description="second",
        schedule=Schedule(prev_task=0, next_task=2, start_at=None),
    )
    ts._create_task(
        name="C",
        description="third",
        schedule=Schedule(prev_task=1, next_task=None, start_at=None),
    )

    ts._update_task_queue(original=[0, 1, 2], new=[2, 0, 1])

    q = ts._get_task_queue()
    assert [t.task_id for t in q] == [2, 0, 1]
    _assert_head_owns_timestamp(q)
    assert q[0].schedule.start_at == "2025-06-23T09:00:00+00:00"


@_handle_project
@pytest.mark.unit
def test_start_time_inherited_on_new_front_insert():
    """
    Insert a *brand-new* task at the front – it must inherit (and be the only
    owner of) the queue-level `start_at`.
    """
    ts = TaskScheduler()

    ts._create_task(
        name="Head",
        description="initial head",
        schedule=Schedule(
            prev_task=None,
            next_task=1,
            start_at="2025-06-23T09:00:00+00:00",
        ),
    )
    ts._create_task(
        name="Tail",
        description="initial tail",
        schedule=Schedule(prev_task=0, next_task=None, start_at=None),
    )

    new_front_id = ts._create_task(name="NewFront", description="inserted")["details"][
        "task_id"
    ]

    ts._update_task_queue(original=[0, 1], new=[new_front_id, 0, 1])

    q = ts._get_task_queue()
    assert [t.task_id for t in q] == [new_front_id, 0, 1]
    _assert_head_owns_timestamp(q)
    assert q[0].schedule.start_at == "2025-06-23T09:00:00+00:00"


@_handle_project
@pytest.mark.unit
def test_start_time_after_multiple_reorders():
    """
    After *any series* of re-orders there must be exactly one `start_at`
    and it must belong to the queue head.
    """
    ts = TaskScheduler()

    ts._create_task(
        name="A",
        description="first",
        schedule=Schedule(
            prev_task=None,
            next_task=1,
            start_at="2025-06-23T09:00:00+00:00",
        ),
    )
    ts._create_task(
        name="B",
        description="second",
        schedule=Schedule(prev_task=0, next_task=2, start_at=None),
    )
    ts._create_task(
        name="C",
        description="third",
        schedule=Schedule(prev_task=1, next_task=None, start_at=None),
    )

    # 1st reorder: B → C → A
    ts._update_task_queue(original=[0, 1, 2], new=[1, 2, 0])
    q1 = ts._get_task_queue()
    assert [t.task_id for t in q1] == [1, 2, 0]
    _assert_head_owns_timestamp(q1)

    # 2nd reorder: C → A → B
    ts._update_task_queue(original=[1, 2, 0], new=[2, 0, 1])
    q2 = ts._get_task_queue()
    assert [t.task_id for t in q2] == [2, 0, 1]
    _assert_head_owns_timestamp(q2)
