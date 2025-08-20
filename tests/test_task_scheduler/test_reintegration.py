from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.schedule import Schedule
from unity.task_scheduler.types.trigger import Trigger, Medium


async def _make_ordered_queue(ts: TaskScheduler, names: list[str]) -> list[int]:
    """Create tasks and order them head→tail, returning their ids.

    Also assigns a queue-level start_at on the head to exercise timestamp moves.
    """
    ids: list[int] = []
    for name in names:
        ids.append(ts._create_task(name=name, description=name)["details"]["task_id"])  # type: ignore[index]

    original = [t.task_id for t in ts._get_task_queue()]
    ts._update_task_queue(original=original, new=ids)
    ts._update_task_start_at(task_id=ids[0], new_start_at=datetime.now(timezone.utc))
    return ids


@pytest.mark.asyncio
@_handle_project
async def test_starting_head_promotes_next_to_scheduled_with_start_at():
    ts = TaskScheduler()
    a, b, c = await _make_ordered_queue(ts, ["A", "B", "C"])  # type: ignore[misc]

    # Sanity: A is scheduled (head with start_at), B and C are queued
    row_a = ts._filter_tasks(filter=f"task_id == {a}")[0]
    row_b = ts._filter_tasks(filter=f"task_id == {b}")[0]
    assert row_a["status"] == "scheduled"
    assert row_b["status"] == "queued"
    original_start = (row_a.get("schedule") or {}).get("start_at")
    assert original_start

    # Start head explicitly (fast-path by id)
    handle = await ts.execute_task(text=str(a))

    # After detachment, B becomes the new head and should inherit start_at and be scheduled
    row_b2 = ts._filter_tasks(filter=f"task_id == {b}")[0]
    sched_b2 = row_b2.get("schedule") or {}
    assert sched_b2.get("prev_task") is None
    assert sched_b2.get("start_at") == original_start
    assert row_b2["status"] == "scheduled"

    # Clean up the active handle to avoid leaking across tests
    handle.stop()
    await handle.result()


@pytest.mark.asyncio
@_handle_project
async def test_starting_middle_detaches_and_links_neighbors():
    ts = TaskScheduler()
    a, b, c = await _make_ordered_queue(ts, ["A", "B", "C"])  # type: ignore[misc]

    # Sanity: A is scheduled (head with start_at), B and C are queued
    row_a = ts._filter_tasks(filter=f"task_id == {a}")[0]
    row_b = ts._filter_tasks(filter=f"task_id == {b}")[0]
    row_c = ts._filter_tasks(filter=f"task_id == {c}")[0]
    assert row_a["status"] == "scheduled"
    assert row_b["status"] == "queued"
    assert row_c["status"] == "queued"
    original_start = (row_a.get("schedule") or {}).get("start_at")
    assert original_start

    # Start the middle task explicitly (fast-path by id)
    handle = await ts.execute_task(text=str(b))

    # After detachment of B, A and C should be directly linked; B should have no schedule
    row_a2 = ts._filter_tasks(filter=f"task_id == {a}")[0]
    row_b2 = ts._filter_tasks(filter=f"task_id == {b}")[0]
    row_c2 = ts._filter_tasks(filter=f"task_id == {c}")[0]

    sa2 = row_a2.get("schedule") or {}
    sb2 = row_b2.get("schedule")
    sc2 = row_c2.get("schedule") or {}

    # Head A remains head with same start_at and points to C
    assert sa2.get("prev_task") is None
    assert sa2.get("next_task") == c
    assert sa2.get("start_at") == original_start
    assert row_a2["status"] == "scheduled"

    # Middle B is detached from the queue
    assert sb2 is None

    # Tail C now points back to A and must not carry start_at
    assert sc2.get("prev_task") == a
    assert "start_at" not in sc2 or not sc2.get("start_at")

    # Stop the active task with guidance to resume as originally scheduled
    handle.stop(
        "Actually this is taking longer than I expected, let's complete this task as per our original schedule instead",
    )
    await handle.result()

    # After reinstatement, restore A→B→C with head A carrying start_at and scheduled
    row_a3 = ts._filter_tasks(filter=f"task_id == {a}")[0]
    row_b3 = ts._filter_tasks(filter=f"task_id == {b}")[0]
    row_c3 = ts._filter_tasks(filter=f"task_id == {c}")[0]

    sa3 = row_a3.get("schedule") or {}
    sb3 = row_b3.get("schedule") or {}
    sc3 = row_c3.get("schedule") or {}

    # Head A remains head with original start_at and points to B
    assert sa3.get("prev_task") is None
    assert sa3.get("next_task") == b
    assert sa3.get("start_at") == original_start
    assert row_a3["status"] == "scheduled"

    # Middle B is back between A and C, with no start_at and queued status
    assert sb3.get("prev_task") == a and sb3.get("next_task") == c
    assert "start_at" not in sb3 or not sb3.get("start_at")
    assert row_b3["status"] == "queued"

    # Tail C points back to B and carries no start_at
    assert sc3.get("prev_task") == b
    assert "start_at" not in sc3 or not sc3.get("start_at")


@pytest.mark.asyncio
@_handle_project
async def test_reinstate_head_restores_head_and_start_at():
    ts = TaskScheduler()
    head_id, next_id = await _make_ordered_queue(ts, ["H", "N"])  # type: ignore[misc]

    # Activate head in isolate (default) and then cancel
    handle = await ts.execute_task(text=str(head_id))
    handle.stop()
    await handle.result()

    # Reinstate the task back to its original head position
    out = ts._reinstate_task_to_previous_queue(task_id=head_id)
    assert out["outcome"].startswith("task reinstated"), out

    rows_h = ts._filter_tasks(filter=f"task_id == {head_id}")[0]
    rows_n = ts._filter_tasks(filter=f"task_id == {next_id}")[0]
    sched_h = rows_h.get("schedule") or {}
    sched_n = rows_n.get("schedule") or {}

    # Head restored with start_at; next prev points back to head and carries no start_at
    assert sched_h.get("prev_task") is None
    assert sched_h.get("next_task") == next_id
    assert "start_at" in sched_h and sched_h.get("start_at")
    assert sched_n.get("prev_task") == head_id
    assert "start_at" not in sched_n or not sched_n.get("start_at")


@pytest.mark.asyncio
@_handle_project
async def test_reinstate_middle_restores_links():
    ts = TaskScheduler()
    a, b, c = await _make_ordered_queue(ts, ["A", "B", "C"])  # type: ignore[misc]

    # Activate middle task (B) in isolate and cancel it
    handle = await ts.execute_task(text=str(b))
    handle.stop()
    await handle.result()

    # Reinstate B → expect A→B→C restored
    _ = ts._reinstate_task_to_previous_queue(task_id=b)

    row_a = ts._filter_tasks(filter=f"task_id == {a}")[0]
    row_b = ts._filter_tasks(filter=f"task_id == {b}")[0]
    row_c = ts._filter_tasks(filter=f"task_id == {c}")[0]

    sa = row_a.get("schedule") or {}
    sb = row_b.get("schedule") or {}
    sc = row_c.get("schedule") or {}

    assert sa.get("next_task") == b
    assert sb.get("prev_task") == a and sb.get("next_task") == c
    assert sc.get("prev_task") == b


@pytest.mark.asyncio
@_handle_project
async def test_reinstate_with_deleted_next_fallback():
    ts = TaskScheduler()
    head_id, next_id, tail_id = await _make_ordered_queue(ts, ["X", "Y", "Z"])  # type: ignore[misc]

    # Activate head and cancel
    handle = await ts.execute_task(text=str(head_id))
    handle.stop()
    await handle.result()

    # Delete original next before reinstatement (drift)
    ts._delete_task(task_id=next_id)

    # Reinstate – should still place X as new head and restore start_at; next may be new head (Z) or None
    _ = ts._reinstate_task_to_previous_queue(task_id=head_id)

    row_x = ts._filter_tasks(filter=f"task_id == {head_id}")[0]
    sched_x = row_x.get("schedule") or {}
    assert sched_x.get("prev_task") is None
    assert "start_at" in sched_x and sched_x.get("start_at")


@pytest.mark.asyncio
@_handle_project
async def test_reinstate_refuses_when_trigger_present():
    ts = TaskScheduler()
    head_id, _ = await _make_ordered_queue(ts, ["TH", "TN"])  # type: ignore[misc]

    handle = await ts.execute_task(text=str(head_id))
    handle.stop()
    await handle.result()

    # Add a trigger to the task → schedule restoration should be refused
    ts._update_task_trigger(task_id=head_id, new_trigger=Trigger(medium=Medium.EMAIL))

    with pytest.raises(ValueError):
        ts._reinstate_task_to_previous_queue(task_id=head_id)


@pytest.mark.asyncio
@_handle_project
async def test_reinstate_refuses_while_active():
    ts = TaskScheduler()
    head_id, _ = await _make_ordered_queue(ts, ["AH", "AN"])  # type: ignore[misc]

    handle = await ts.execute_task(text=str(head_id))

    # Attempt reinstatement before cancelling → must raise
    with pytest.raises(RuntimeError):
        ts._reinstate_task_to_previous_queue(task_id=head_id)

    # Clean up
    handle.stop()
    await handle.result()


@pytest.mark.asyncio
@_handle_project
async def test_reinstate_primed_conflict_downgrades_to_queued():
    ts = TaskScheduler()

    # Create a simple two-task queue without a start_at so head would typically be 'primed'
    # First, make head without explicit start_at
    h_id = ts._create_task(name="HeadPrimed", description="hp")["details"]["task_id"]
    n_id = ts._create_task(
        name="NextQueued",
        description="nq",
        schedule=Schedule(prev_task=h_id),
    )["details"]["task_id"]

    # Sanity: head is 'primed'
    head_row = ts._filter_tasks(filter=f"task_id == {h_id}")[0]
    assert head_row["status"] in ("primed",)

    # Activate head and cancel
    handle = await ts.execute_task(text=str(h_id))
    handle.stop()
    await handle.result()

    # Create a new task now – with no active and no primed, this becomes the new 'primed'
    new_tid = ts._create_task(name="NewPrimed", description="np")["details"]["task_id"]
    new_row = ts._filter_tasks(filter=f"task_id == {new_tid}")[0]
    assert new_row["status"] == "primed"

    # Reinstate original head – original status was primed but conflict exists → should downgrade to queued
    _ = ts._reinstate_task_to_previous_queue(task_id=h_id)
    reinstated = ts._filter_tasks(filter=f"task_id == {h_id}")[0]
    assert reinstated["status"] == "queued"


@pytest.mark.asyncio
@_handle_project
async def test_reintegration_plan_clears_on_completion():
    ts = TaskScheduler()
    head_id, next_id = await _make_ordered_queue(ts, ["HC", "NC"])  # type: ignore[misc]

    # Start head in isolate and allow it to complete naturally
    handle = await ts.execute_task(text=str(head_id))
    # Awaiting result will mark the instance as completed internally
    await handle.result()

    # Attempting reinstatement should now fail because the plan was cleared
    with pytest.raises(ValueError):
        ts._reinstate_task_to_previous_queue(task_id=head_id)
