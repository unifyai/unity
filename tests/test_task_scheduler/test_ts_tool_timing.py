from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

import pytest

from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.repetition import RepeatPattern, Frequency
from unity.transcript_manager.types.message import Medium
from tests.helpers import _handle_project


def _uniq() -> str:
    return str(time.time_ns())


# Atomic timing tests: exactly one tool per test


@pytest.mark.unit
@_handle_project
def test_tool_list_columns_timing():
    ts = TaskScheduler()
    t0 = time.perf_counter()
    cols = ts._list_columns()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(cols, dict) and cols
    assert elapsed_ms < 0.1
    print(f"elapsed: {elapsed_ms} < 0.1")


@pytest.mark.unit
@_handle_project
def test_tool_num_tasks_timing():
    ts = TaskScheduler()
    ts._create_task(name="TT NumTasks " + _uniq(), description="nt")
    t0 = time.perf_counter()
    n = ts._num_tasks()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(n, int) and n >= 1
    assert elapsed_ms < 1000
    print(f"elapsed: {elapsed_ms} < 1000")


@pytest.mark.unit
@_handle_project
def test_tool_create_task_timing():
    ts = TaskScheduler()
    t0 = time.perf_counter()
    out = ts._create_task(name="TT Create " + _uniq(), description="timing create")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out["outcome"]
    assert elapsed_ms < 2700
    print(f"elapsed: {elapsed_ms} < 2700")


@pytest.mark.unit
@_handle_project
def test_tool_create_tasks_timing():
    ts = TaskScheduler()
    t0 = time.perf_counter()
    resp = ts._create_tasks(
        tasks=[
            {"name": "TT CT A " + _uniq(), "description": "A"},
            {"name": "TT CT B " + _uniq(), "description": "B"},
        ],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert resp["outcome"]
    assert len(resp["details"]["task_ids"]) == 2
    assert elapsed_ms < 7300
    print(f"elapsed: {elapsed_ms} < 7300")


@pytest.mark.unit
@_handle_project
def test_tool_filter_tasks_timing():
    ts = TaskScheduler()
    tid = ts._create_task(name="TT Filter " + _uniq(), description="flt")["details"][
        "task_id"
    ]
    t0 = time.perf_counter()
    rows = ts._filter_tasks(filter=f"task_id == {tid}")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert rows and rows[0]["task_id"] == tid
    assert elapsed_ms < 1200
    print(f"elapsed: {elapsed_ms} < 1200")


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_tool_search_tasks_timing():
    ts = TaskScheduler()
    ts._create_task(name="TT Search " + _uniq(), description="banking and budgeting")
    t0 = time.perf_counter()
    results = ts._search_tasks(references={"description": "banking"}, k=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(results, list)
    assert elapsed_ms < 5100
    print(f"elapsed: {elapsed_ms} < 5100")


@pytest.mark.unit
@_handle_project
def test_tool_delete_task_timing():
    ts = TaskScheduler()
    tid = ts._create_task(name="TT Delete " + _uniq(), description="del")["details"][
        "task_id"
    ]
    t0 = time.perf_counter()
    out = ts._delete_task(task_id=tid)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out["details"]["task_id"] == tid
    assert elapsed_ms < 1300
    print(f"elapsed: {elapsed_ms} < 1300")


@pytest.mark.unit
@_handle_project
def test_tool_cancel_tasks_timing():
    ts = TaskScheduler()
    tid = ts._create_task(name="TT Cancel " + _uniq(), description="can")["details"][
        "task_id"
    ]
    t0 = time.perf_counter()
    out = ts._cancel_tasks([tid])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert tid in out["details"]["task_ids"]
    assert elapsed_ms < 2500
    print(f"elapsed: {elapsed_ms} < 2500")


@pytest.mark.unit
@_handle_project
def test_tool_list_queues_timing():
    ts = TaskScheduler()
    a = ts._create_task(name="TT LQ A " + _uniq(), description="a")["details"][
        "task_id"
    ]
    b = ts._create_task(name="TT LQ B " + _uniq(), description="b")["details"][
        "task_id"
    ]
    set_out = ts._set_queue(queue_id=None, order=[a, b])
    qid = set_out["details"]["queue_id"]
    assert isinstance(qid, int)
    t0 = time.perf_counter()
    queues = ts._list_queues()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert any(q.get("queue_id") == qid for q in queues)
    assert elapsed_ms < 0.1
    print(f"elapsed: {elapsed_ms} < 0.1")


@pytest.mark.unit
@_handle_project
def test_tool_get_queue_timing():
    ts = TaskScheduler()
    a = ts._create_task(name="TT GQ A " + _uniq(), description="a")["details"][
        "task_id"
    ]
    b = ts._create_task(name="TT GQ B " + _uniq(), description="b")["details"][
        "task_id"
    ]
    set_out = ts._set_queue(queue_id=None, order=[a, b])
    qid = set_out["details"]["queue_id"]
    t0 = time.perf_counter()
    chain = ts._get_queue(queue_id=qid)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert [t.task_id for t in chain] == [a, b]
    assert elapsed_ms < 1200
    print(f"elapsed: {elapsed_ms} < 1200")


@pytest.mark.unit
@_handle_project
def test_tool_get_queue_for_task_timing():
    ts = TaskScheduler()
    a = ts._create_task(name="TT GTQ A " + _uniq(), description="a")["details"][
        "task_id"
    ]
    b = ts._create_task(name="TT GTQ B " + _uniq(), description="b")["details"][
        "task_id"
    ]
    set_out = ts._set_queue(queue_id=None, order=[a, b])
    t0 = time.perf_counter()
    q = ts._get_queue_for_task(task_id=a)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert [t.task_id for t in q][:2] == [a, b]
    assert elapsed_ms < 1100
    print(f"elapsed: {elapsed_ms} < 1100")


@pytest.mark.unit
@_handle_project
def test_tool_set_queue_timing():
    ts = TaskScheduler()
    a = ts._create_task(name="TT SQ A " + _uniq(), description="a")["details"][
        "task_id"
    ]
    b = ts._create_task(name="TT SQ B " + _uniq(), description="b")["details"][
        "task_id"
    ]
    t0 = time.perf_counter()
    out = ts._set_queue(queue_id=None, order=[a, b])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out["details"].get("queue_id") is not None
    assert elapsed_ms < 4700
    print(f"elapsed: {elapsed_ms} < 4700")


@pytest.mark.unit
@_handle_project
def test_tool_reorder_queue_timing():
    ts = TaskScheduler()
    a = ts._create_task(name="TT RQ A " + _uniq(), description="a")["details"][
        "task_id"
    ]
    b = ts._create_task(name="TT RQ B " + _uniq(), description="b")["details"][
        "task_id"
    ]
    set_out = ts._set_queue(queue_id=None, order=[a, b])
    qid = set_out["details"]["queue_id"]
    new_order = [b, a]
    t0 = time.perf_counter()
    out = ts._reorder_queue(queue_id=qid, new_order=new_order)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out["details"]["new_order"] == new_order
    assert elapsed_ms < 4100
    print(f"elapsed: {elapsed_ms} < 4100")


@pytest.mark.unit
@_handle_project
def test_tool_move_tasks_to_queue_timing():
    ts = TaskScheduler()
    a = ts._create_task(name="TT MV A " + _uniq(), description="a")["details"][
        "task_id"
    ]
    b = ts._create_task(name="TT MV B " + _uniq(), description="b")["details"][
        "task_id"
    ]
    ts._set_queue(queue_id=None, order=[a, b])
    t0 = time.perf_counter()
    out = ts._move_tasks_to_queue(task_ids=[a], queue_id=None, position="front")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out["details"].get("queue_id") is not None
    assert a in out["details"]["task_ids"]
    assert elapsed_ms < 12600
    print(f"elapsed: {elapsed_ms} < 12600")


@pytest.mark.unit
@_handle_project
def test_tool_set_schedules_atomic_timing():
    ts = TaskScheduler()
    a = ts._create_task(name="TT SSA A " + _uniq(), description="a")["details"][
        "task_id"
    ]
    b = ts._create_task(name="TT SSA B " + _uniq(), description="b")["details"][
        "task_id"
    ]
    # Use a consistent queue_id for both items to satisfy cross-queue guard
    rows = ts._filter_tasks(filter=f"task_id in [{a}, {b}]")
    qids = {r.get("task_id"): r.get("queue_id") for r in rows}
    chosen_qid = qids.get(a) if qids.get(a) is not None else qids.get(b)
    t0 = time.perf_counter()
    out = ts._set_schedules_atomic(
        schedules=[
            {
                "task_id": a,
                "queue_id": chosen_qid,
                "schedule": {"prev_task": None, "next_task": b},
            },
            {"task_id": b, "queue_id": chosen_qid, "schedule": {"prev_task": None}},
        ],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out["details"]["count"] == 2
    assert elapsed_ms < 6700
    print(f"elapsed: {elapsed_ms} < 6700")


@pytest.mark.unit
@_handle_project
def test_tool_partition_queue_timing():
    ts = TaskScheduler()
    a = ts._create_task(name="TT PQ A " + _uniq(), description="a")["details"][
        "task_id"
    ]
    b = ts._create_task(name="TT PQ B " + _uniq(), description="b")["details"][
        "task_id"
    ]
    c = ts._create_task(name="TT PQ C " + _uniq(), description="c")["details"][
        "task_id"
    ]
    ts._set_queue(queue_id=None, order=[a, b, c])
    t0 = time.perf_counter()
    out = ts._partition_queue(parts=[{"task_ids": [a]}, {"task_ids": [b, c]}])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out["outcome"] == "queue partitioned"
    assert elapsed_ms < 11200
    print(f"elapsed: {elapsed_ms} < 11200")


# Atomic field update timing: combined single-call update


@pytest.mark.unit
@_handle_project
def test_tool_update_task_timing():
    ts = TaskScheduler()
    # Start with a triggerable task so we can clear trigger and set start_at in one call
    tid = ts._create_task(
        name="TT Combined " + _uniq(),
        description="initial",
        trigger={"medium": Medium.EMAIL},
    )["details"]["task_id"]

    when = datetime.now(timezone.utc) + timedelta(minutes=5)
    dl = datetime.now(timezone.utc) + timedelta(days=1)

    t0 = time.perf_counter()
    ts._update_task(
        task_id=tid,
        name="TT Combined Renamed " + _uniq(),
        description="combined updated",
        start_at=when,
        deadline=dl,
        repeat=[RepeatPattern(frequency=Frequency.DAILY)],
        priority="high",
        status="scheduled",
        trigger=None,  # clear trigger while adding start_at
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Validate all fields via a single read
    row = ts._filter_tasks(filter=f"task_id == {tid}", limit=1)[0]
    assert "Renamed" in row["name"]
    assert row["description"] == "combined updated"
    assert (row.get("schedule") or {}).get("start_at") is not None
    assert row.get("deadline") is not None
    assert row.get("repeat") is not None
    assert row.get("priority") == "high"
    assert row.get("status") == "scheduled"
    assert row.get("trigger") is None
    assert elapsed_ms < 3000
    print(f"elapsed: {elapsed_ms} < 3000")


# Queue linkage helpers


@pytest.mark.unit
@_handle_project
def test_tool_detach_from_queue_for_activation_timing():
    ts = TaskScheduler()
    a = ts._create_task(name="TT Det A " + _uniq(), description="a")["details"][
        "task_id"
    ]
    b = ts._create_task(name="TT Det B " + _uniq(), description="b")["details"][
        "task_id"
    ]
    ts._set_queue(queue_id=None, order=[a, b])
    t0 = time.perf_counter()
    ts._detach_from_queue_for_activation(task_id=a, detach=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # Head should be detached now; just sanity check no exception and measure time
    assert elapsed_ms < 5000
    print(f"elapsed: {elapsed_ms} < 5000")


@pytest.mark.unit
@_handle_project
def test_tool_attach_with_links_timing():
    ts = TaskScheduler()
    a = ts._create_task(name="TT Att A " + _uniq(), description="a")["details"][
        "task_id"
    ]
    b = ts._create_task(name="TT Att B " + _uniq(), description="b")["details"][
        "task_id"
    ]
    c = ts._create_task(name="TT Att C " + _uniq(), description="c")["details"][
        "task_id"
    ]
    ts._set_queue(queue_id=None, order=[a, c])
    # Call the queue utility directly now that the scheduler wrapper was removed
    from unity.task_scheduler.queue_utils import attach_with_links as _attach_with_links

    t0 = time.perf_counter()
    _attach_with_links(
        ts,
        task_id=b,
        prev_task=a,
        next_task=c,
        head_start_at=None,
        err_prefix="attach",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    chain = ts._get_queue_for_task(task_id=a)
    chain = ts._get_queue_for_task(task_id=a)
    ids = [t.task_id for t in chain]
    assert ids[:3] == [a, b, c]
    assert elapsed_ms < 5300
    print(f"elapsed: {elapsed_ms} < 5300")


@pytest.mark.unit
@_handle_project
def test_tool_validated_write_timing():
    ts = TaskScheduler()
    tid = ts._create_task(name="TT VW " + _uniq(), description="d")["details"][
        "task_id"
    ]
    t0 = time.perf_counter()
    out = ts._validated_write(
        task_id=tid,
        entries={"description": "d2"},
        err_prefix="vw",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out
    assert elapsed_ms < 1200
    print(f"elapsed: {elapsed_ms} < 1200")
