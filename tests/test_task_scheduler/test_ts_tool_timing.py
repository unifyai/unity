from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta

import pytest

from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.repetition import RepeatPattern, Frequency
from unity.transcript_manager.types.message import Medium
from tests.helpers import _handle_project


def _enable_timing():
    os.environ["TASK_SCHEDULER_TOOL_TIMING"] = "1"
    # Keep prints off by default to keep CI logs clean
    # os.environ["TASK_SCHEDULER_TOOL_TIMING_PRINT"] = "1"


def _uniq() -> str:
    return str(time.time_ns())


# Atomic timing tests: exactly one tool per test


@pytest.mark.unit
@_handle_project
def test_tool_list_columns_timing():
    _enable_timing()
    ts = TaskScheduler()
    t0 = time.perf_counter()
    cols = ts._list_columns()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(cols, dict) and cols
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_num_tasks_timing():
    _enable_timing()
    ts = TaskScheduler()
    ts._create_task(name="TT NumTasks " + _uniq(), description="nt")
    t0 = time.perf_counter()
    n = ts._num_tasks()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(n, int) and n >= 1
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_create_task_timing():
    _enable_timing()
    ts = TaskScheduler()
    t0 = time.perf_counter()
    out = ts._create_task(name="TT Create " + _uniq(), description="timing create")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out["outcome"]
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_create_tasks_timing():
    _enable_timing()
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
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_filter_tasks_timing():
    _enable_timing()
    ts = TaskScheduler()
    tid = ts._create_task(name="TT Filter " + _uniq(), description="flt")["details"][
        "task_id"
    ]
    t0 = time.perf_counter()
    rows = ts._filter_tasks(filter=f"task_id == {tid}")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert rows and rows[0]["task_id"] == tid
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_tool_search_tasks_timing():
    _enable_timing()
    ts = TaskScheduler()
    ts._create_task(name="TT Search " + _uniq(), description="banking and budgeting")
    t0 = time.perf_counter()
    results = ts._search_tasks(references={"description": "banking"}, k=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(results, list)
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_delete_task_timing():
    _enable_timing()
    ts = TaskScheduler()
    tid = ts._create_task(name="TT Delete " + _uniq(), description="del")["details"][
        "task_id"
    ]
    t0 = time.perf_counter()
    out = ts._delete_task(task_id=tid)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out["details"]["task_id"] == tid
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_cancel_tasks_timing():
    _enable_timing()
    ts = TaskScheduler()
    tid = ts._create_task(name="TT Cancel " + _uniq(), description="can")["details"][
        "task_id"
    ]
    t0 = time.perf_counter()
    out = ts._cancel_tasks([tid])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert tid in out["details"]["task_ids"]
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_list_queues_timing():
    _enable_timing()
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
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_get_queue_timing():
    _enable_timing()
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
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_get_queue_for_task_timing():
    _enable_timing()
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
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_set_queue_timing():
    _enable_timing()
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
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_reorder_queue_timing():
    _enable_timing()
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
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_move_tasks_to_queue_timing():
    _enable_timing()
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
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_set_schedules_atomic_timing():
    _enable_timing()
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
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_partition_queue_timing():
    _enable_timing()
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
    print(f"elapsed: {elapsed_ms} < X")


# Atomic field update timing: combined single-call update


@pytest.mark.unit
@_handle_project
def test_tool_update_task_timing():
    _enable_timing()
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
    print(f"elapsed: {elapsed_ms} < X")


# Queue linkage helpers


@pytest.mark.unit
@_handle_project
def test_tool_detach_from_queue_for_activation_timing():
    _enable_timing()
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
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_attach_with_links_timing():
    _enable_timing()
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
    t0 = time.perf_counter()
    ts._attach_with_links(
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
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_validated_write_timing():
    _enable_timing()
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
    print(f"elapsed: {elapsed_ms} < X")
