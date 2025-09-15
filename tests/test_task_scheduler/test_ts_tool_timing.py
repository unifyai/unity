from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta

import pytest

from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.repetition import RepeatPattern, Frequency
from tests.helpers import _handle_project


def _enable_timing():
    os.environ["TASK_SCHEDULER_TOOL_TIMING"] = "1"
    # Keep prints off by default to keep CI logs clean
    # os.environ["TASK_SCHEDULER_TOOL_TIMING_PRINT"] = "1"


@pytest.mark.unit
@_handle_project
def test_tool_list_columns_timing():
    _enable_timing()
    ts = TaskScheduler()
    t0 = time.perf_counter()
    cols = ts._list_columns()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(cols, dict) and cols
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_create_and_filter_timing():
    _enable_timing()
    ts = TaskScheduler()
    name = "TS Perf Create"
    t0 = time.perf_counter()
    out = ts._create_task(name=name, description="timing create")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    tid = int(out["details"]["task_id"])
    rows = ts._filter_tasks(filter=f"task_id == {tid}")
    assert rows and rows[0]["task_id"] == tid
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_create_tasks_and_list_queues_timing():
    _enable_timing()
    ts = TaskScheduler()
    # Create a small queue via the batched tool
    t0 = time.perf_counter()
    now_iso = datetime.now(timezone.utc).isoformat()
    resp = ts._create_tasks(
        tasks=[
            {"name": "TS Q A", "description": "A"},
            {"name": "TS Q B", "description": "B"},
        ],
        queue_ordering=[
            {"order": [0, 1], "queue_head": {"start_at": now_iso}},
        ],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert resp["outcome"]
    queues = ts._list_queues()
    assert isinstance(queues, list) and queues
    qid = queues[0]["queue_id"]
    chain = ts._get_queue(queue_id=qid)
    assert len(chain) >= 2
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_update_fields_timing():
    _enable_timing()
    ts = TaskScheduler()
    tid = ts._create_task(name="TS Update Fields", description="fields")["details"][
        "task_id"
    ]
    # Name / description
    t0 = time.perf_counter()
    ts._update_task_name(task_id=tid, new_name="TS Update Fields (renamed)")
    ts._update_task_description(task_id=tid, new_description="fields desc")
    # Start_at (set to a near-future time)
    ts._update_task_start_at(
        task_id=tid,
        new_start_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    # Deadline
    ts._update_task_deadline(
        task_id=tid,
        new_deadline=datetime.now(timezone.utc) + timedelta(days=1),
    )
    # Priority
    ts._update_task_priority(task_id=tid, new_priority="high")
    # Repetition
    ts._update_task_repetition(
        task_id=tid,
        new_repeat=[RepeatPattern(frequency=Frequency.DAILY)],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms >= 0.0  # sanity: decorator executed
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_set_and_reorder_queue_timing():
    _enable_timing()
    ts = TaskScheduler()
    t1 = ts._create_task(name="TS Q1", description="q1")["details"]["task_id"]
    t2 = ts._create_task(name="TS Q2", description="q2")["details"]["task_id"]
    t3 = ts._create_task(name="TS Q3", description="q3")["details"]["task_id"]
    order = [t1, t2, t3]
    t0 = time.perf_counter()
    set_out = ts._set_queue(queue_id=None, order=order)
    qid = set_out["details"]["queue_id"]
    chain = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in chain][:3] == order
    # Reorder
    rev = list(reversed(order))
    ro0 = time.perf_counter()
    ts._reorder_queue(queue_id=qid, new_order=rev)
    ro_ms = (time.perf_counter() - ro0) * 1000.0
    chain2 = ts._get_queue(queue_id=qid)
    assert [t.task_id for t in chain2][:3] == rev
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms >= 0.0 and ro_ms >= 0.0
    # assert elapsed_ms < X
    # assert ro_ms < X
    # print(f"elapsed: {elapsed_ms} < X; reorder: {ro_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_delete_and_cancel_timing():
    _enable_timing()
    ts = TaskScheduler()
    tid_del = ts._create_task(name="TS Del", description="del")["details"]["task_id"]
    tid_can = ts._create_task(name="TS Can", description="can")["details"]["task_id"]
    t0 = time.perf_counter()
    del_out = ts._delete_task(task_id=tid_del)
    can_out = ts._cancel_tasks([tid_can])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert del_out["details"]["task_id"] == tid_del
    assert tid_can in can_out["details"]["task_ids"]
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_tool_search_tasks_timing():
    _enable_timing()
    ts = TaskScheduler()
    ts._create_task(name="TS Search", description="banking and budgeting")
    t0 = time.perf_counter()
    results = ts._search_tasks(references={"description": "banking"}, k=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(results, list)
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")
