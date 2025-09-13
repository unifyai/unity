import pytest
from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.schedule import Schedule
from datetime import datetime, timezone


@_handle_project
@pytest.mark.unit
def test_get_queue_and_reorder():
    ts = TaskScheduler()

    # -----  create three queued tasks with an explicit chain  -----
    qid = ts._allocate_new_queue_id()
    t0 = ts._create_task(
        name="T0",
        description="first",
        schedule=Schedule(queue_id=qid, start_at=datetime.now(timezone.utc)),
    )
    t1 = ts._create_task(
        name="T1",
        description="second",
        schedule=Schedule(queue_id=qid, prev_task=0),
    )
    t2 = ts._create_task(
        name="T2",
        description="third",
        schedule=Schedule(queue_id=qid, prev_task=1),
    )

    queue = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in queue] == [0, 1, 2]

    # -----  swap the order (0,2,1)  -----
    ts._reorder_queue(queue_id=qid, new_order=[0, 2, 1])

    new_q = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in new_q] == [0, 2, 1]


@_handle_project
@pytest.mark.unit
def test_insert_into_queue():
    ts = TaskScheduler()

    # base task
    qid = ts._allocate_new_queue_id()
    ts._create_task(name="base", description="x", schedule=Schedule(queue_id=qid))

    # create a brand-new task that will be inserted
    new_id = ts._create_task(name="insert-me", description="y")["details"]["task_id"]

    ts._set_queue(queue_id=qid, order=[0, new_id])

    q = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in q] == [0, new_id]
    # also check the linkage of node 0 -> new_id
    assert q[0].schedule.next_task == new_id


# ---------------------------------------------------------------------------#
#  Primed → Queued invariant                                                  #
# ---------------------------------------------------------------------------#


@_handle_project
@pytest.mark.unit
def test_insert_primed_task_downgrades_to_queued():
    """A primed task inserted *behind* the head must be downgraded to queued."""

    ts = TaskScheduler()

    # 1) Queue head – starts in the default 'primed' state
    qid = ts._allocate_new_queue_id()
    ts._create_task(
        name="Research",
        description="Initial research phase",
        schedule=Schedule(queue_id=qid),
    )

    # 2) Second task explicitly queued behind the head
    ts._create_task(
        name="Write report",
        description="Summarise findings",
        schedule=Schedule(queue_id=qid, prev_task=0),
        status="queued",
    )

    # 3) Brand-new task – defaults to *primed* because no other primed exists yet
    t_email = ts._create_task(
        name="Email boss",
        description="Send the report via email",
    )["details"]["task_id"]

    # 4) Insert the email task *after* the report in the runnable queue
    ts._set_queue(queue_id=qid, order=[0, 1, t_email])

    # 5) The scheduler must automatically downgrade the status to 'queued'
    row = ts._filter_tasks(filter=f"task_id == {t_email}", limit=1)[0]
    assert (
        row["status"] == "queued"
    ), "Non-head tasks may not remain 'primed'; status should be 'queued'."


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

    qid = ts._allocate_new_queue_id()
    ts._create_task(
        name="A",
        description="first",
        schedule=Schedule(
            queue_id=qid,
            start_at="2025-06-23T09:00:00+00:00",
        ),
    )
    ts._create_task(
        name="B",
        description="second",
        schedule=Schedule(queue_id=qid, prev_task=0),
    )
    ts._create_task(
        name="C",
        description="third",
        schedule=Schedule(queue_id=qid, prev_task=1),
    )

    ts._reorder_queue(queue_id=qid, new_order=[2, 0, 1])

    q = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in q] == [2, 0, 1]
    _assert_head_owns_timestamp(q)
    assert q[0].schedule.start_at.isoformat() == "2025-06-23T09:00:00+00:00"


@_handle_project
@pytest.mark.unit
def test_start_time_inherited_on_new_front_insert():
    """
    Insert a *brand-new* task at the front – it must inherit (and be the only
    owner of) the queue-level `start_at`.
    """
    ts = TaskScheduler()

    qid = ts._allocate_new_queue_id()
    ts._create_task(
        name="Head",
        description="initial head",
        schedule=Schedule(
            queue_id=qid,
            start_at="2025-06-23T09:00:00+00:00",
        ),
    )
    ts._create_task(
        name="Tail",
        description="initial tail",
        schedule=Schedule(queue_id=qid, prev_task=0),
    )

    new_front_id = ts._create_task(name="NewFront", description="inserted")["details"][
        "task_id"
    ]

    ts._set_queue(queue_id=qid, order=[new_front_id, 0, 1])

    q = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in q] == [new_front_id, 0, 1]
    _assert_head_owns_timestamp(q)
    assert q[0].schedule.start_at.isoformat() == "2025-06-23T09:00:00+00:00"


@_handle_project
@pytest.mark.unit
def test_start_time_after_multiple_reorders():
    """
    After *any series* of re-orders there must be exactly one `start_at`
    and it must belong to the queue head.
    """
    ts = TaskScheduler()

    qid = ts._allocate_new_queue_id()
    ts._create_task(
        name="A",
        description="first",
        schedule=Schedule(
            queue_id=qid,
            start_at="2030-06-23T09:00:00+00:00",
        ),
    )
    ts._create_task(
        name="B",
        description="second",
        schedule=Schedule(queue_id=qid, prev_task=0),
    )
    ts._create_task(
        name="C",
        description="third",
        schedule=Schedule(queue_id=qid, prev_task=1),
    )

    # original order
    q0 = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in q0] == [0, 1, 2]

    # 1st reorder: B → C → A
    ts._reorder_queue(queue_id=qid, new_order=[1, 2, 0])
    q1 = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in q1] == [1, 2, 0]
    _assert_head_owns_timestamp(q1)

    # 2nd reorder: C → A → B
    ts._reorder_queue(queue_id=qid, new_order=[2, 0, 1])
    q2 = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in q2] == [2, 0, 1]
    _assert_head_owns_timestamp(q2)


@_handle_project
@pytest.mark.unit
def test_list_and_get_queues_basic():
    ts = TaskScheduler()

    # default queue: 0(start),1,2
    qid = ts._allocate_new_queue_id()
    ts._create_task(
        name="Q0",
        description="head",
        schedule=Schedule(queue_id=qid, start_at="2031-01-01T09:00:00+00:00"),
    )
    ts._create_task(
        name="Q1",
        description="mid",
        schedule=Schedule(queue_id=qid, prev_task=0),
    )
    ts._create_task(
        name="Q2",
        description="tail",
        schedule=Schedule(queue_id=qid, prev_task=1),
    )

    lst = ts._list_queues()
    assert isinstance(lst, list) and len(lst) == 1
    qinfo = lst[0]
    assert qinfo["head_id"] == 0
    assert qinfo["size"] == 3
    # Normalise Z vs +00:00 representation
    _start_norm = (qinfo.get("start_at") or "").replace("Z", "+00:00")
    assert _start_norm == "2031-01-01T09:00:00+00:00"

    # get_queue(queue_id) mirrors legacy _get_task_queue
    q_default = ts._get_queue(queue_id=qinfo["queue_id"])
    q_legacy = ts._get_task_queue()
    assert [t.task_id for t in q_default] == [t.task_id for t in q_legacy]


@_handle_project
@pytest.mark.unit
def test_reorder_queue_preserves_head_start_at():
    ts = TaskScheduler()

    qid = ts._allocate_new_queue_id()
    ts._create_task(
        name="A",
        description="first",
        schedule=Schedule(queue_id=qid, start_at="2032-06-23T09:00:00+00:00"),
    )
    ts._create_task(
        name="B",
        description="second",
        schedule=Schedule(queue_id=qid, prev_task=0),
    )
    ts._create_task(
        name="C",
        description="third",
        schedule=Schedule(queue_id=qid, prev_task=1),
    )

    ts._reorder_queue(queue_id=qid, new_order=[2, 0, 1])

    q = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in q] == [2, 0, 1]
    _assert_head_owns_timestamp(q)
    assert q[0].schedule.start_at.isoformat() == "2032-06-23T09:00:00+00:00"


@_handle_project
@pytest.mark.unit
def test_move_tasks_to_new_queue():
    ts = TaskScheduler()

    # default queue: 0(start),1,2,3
    qid0 = ts._allocate_new_queue_id()
    ts._create_task(
        name="T0",
        description="head",
        schedule=Schedule(queue_id=qid0, start_at="2033-02-01T08:00:00+00:00"),
    )
    ts._create_task(
        name="T1",
        description="desc1",
        schedule=Schedule(queue_id=qid0, prev_task=0),
    )
    ts._create_task(
        name="T2",
        description="desc2",
        schedule=Schedule(queue_id=qid0, prev_task=1),
    )
    ts._create_task(
        name="T3",
        description="desc3",
        schedule=Schedule(queue_id=qid0, prev_task=2),
    )

    # move [1,3] to a new queue at the front
    res = ts._move_tasks_to_queue(task_ids=[1, 3], queue_id=None, position="front")
    qid = res["details"]["queue_id"]
    assert isinstance(qid, int) and qid >= 1

    # default queue should now be [0,2]
    q_def = ts._get_queue(queue_id=qid0)
    assert [t.task_id for t in q_def] == [0, 2]
    _assert_head_owns_timestamp(q_def)
    assert q_def[0].schedule.start_at.isoformat() == "2033-02-01T08:00:00+00:00"

    # new queue should be [1,3] with no start_at by default
    q_new = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in q_new] == [1, 3]
    assert not (q_new[0].schedule and q_new[0].schedule.start_at)

    # list_queues should show two queues
    lst = ts._list_queues()
    assert len(lst) == 2


@_handle_project
@pytest.mark.unit
def test_partition_queue_split_with_dates():
    ts = TaskScheduler()

    # default queue: 0(start),1,2,3
    qid0 = ts._allocate_new_queue_id()
    ts._create_task(
        name="P0",
        description="head",
        schedule=Schedule(queue_id=qid0, start_at="2034-01-01T09:00:00+00:00"),
    )
    ts._create_task(
        name="P1",
        description="d1",
        schedule=Schedule(queue_id=qid0, prev_task=0),
    )
    ts._create_task(
        name="P2",
        description="d2",
        schedule=Schedule(queue_id=qid0, prev_task=1),
    )
    ts._create_task(
        name="P3",
        description="d3",
        schedule=Schedule(queue_id=qid0, prev_task=2),
    )

    res = ts._partition_queue(
        parts=[
            {"task_ids": [0, 2], "queue_start_at": "2034-02-01T09:00:00+00:00"},
            {"task_ids": [1, 3], "queue_start_at": "2034-02-02T09:00:00+00:00"},
        ],
        strategy="preserve_order",
    )

    # default queue now [0,2] with updated start_at
    q_def = ts._get_queue(queue_id=qid0)
    assert [t.task_id for t in q_def] == [0, 2]
    _assert_head_owns_timestamp(q_def)
    assert q_def[0].schedule.start_at.isoformat() == "2034-02-01T09:00:00+00:00"

    # new queue(s) returned in details
    newqs = res["details"]["new_queues"]
    assert len(newqs) == 1 and set(newqs[0]["task_ids"]) == {1, 3}
    qid = newqs[0]["queue_id"]
    q_new = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in q_new] == [1, 3]
    _assert_head_owns_timestamp(q_new)
    assert q_new[0].schedule.start_at.isoformat() == "2034-02-02T09:00:00+00:00"


@_handle_project
@pytest.mark.unit
def test_move_tasks_to_existing_queue_front_and_back():
    ts = TaskScheduler()

    # default queue: 0(start),1,2
    qid0 = ts._allocate_new_queue_id()
    ts._create_task(
        name="M0",
        description="head",
        schedule=Schedule(queue_id=qid0, start_at="2035-03-03T09:00:00+00:00"),
    )
    ts._create_task(
        name="M1",
        description="dx1",
        schedule=Schedule(queue_id=qid0, prev_task=0),
    )
    ts._create_task(
        name="M2",
        description="dx2",
        schedule=Schedule(queue_id=qid0, prev_task=1),
    )

    # create second queue by moving [2]
    res1 = ts._move_tasks_to_queue(task_ids=[2], queue_id=None, position="back")
    qid = res1["details"]["queue_id"]
    assert isinstance(qid, int)

    # now move [1] to the front of that existing queue
    ts._move_tasks_to_queue(task_ids=[1], queue_id=qid, position="front")

    # default queue left with [0]
    q_def = ts._get_queue(queue_id=qid0)
    assert [t.task_id for t in q_def] == [0]
    _assert_head_owns_timestamp(q_def)
    assert q_def[0].schedule.start_at.isoformat() == "2035-03-03T09:00:00+00:00"

    # target queue should be [1,2]
    q_tgt = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in q_tgt] == [1, 2]
    # no start_at on non-default queue unless explicitly set
    assert not (q_tgt[0].schedule and q_tgt[0].schedule.start_at)
