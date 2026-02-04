from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.schedule import Schedule
from unity.task_scheduler.types.status import Status
from unity.task_scheduler.types.trigger import Trigger, Medium

# Speed up only this module's SimulatedActor by monkeypatching the class symbols
# used by TaskScheduler to a shorter-duration variant. This does not affect
# other test modules.
import pytest


@pytest.fixture(autouse=True, scope="module")
def _short_simulated_actor():
    from pytest import MonkeyPatch

    mp = MonkeyPatch()
    from unity.actor.simulated import SimulatedActor as _BaseSA

    class _ShortSimActor(_BaseSA):  # type: ignore[misc]
        def __init__(
            self,
            *,
            steps: int | None = None,
            duration: float | None = None,
            _requests_clarification: bool = False,
            log_mode: "str | None" = "log",
            simulation_guidance: "str | None" = None,
        ) -> None:  # noqa: E501
            # Force a very short duration for tests in this module only
            super().__init__(
                steps=steps,
                duration=0.1 if duration is None or duration > 0.1 else duration,
                _requests_clarification=_requests_clarification,
                log_mode=log_mode,
                simulation_guidance=simulation_guidance,
            )

    # Replace both the canonical class and the symbol imported inside TaskScheduler
    mp.setattr("unity.actor.simulated.SimulatedActor", _ShortSimActor, raising=True)
    mp.setattr(
        "unity.task_scheduler.task_scheduler.SimulatedActor",
        _ShortSimActor,
        raising=True,
    )

    try:
        yield
    finally:
        mp.undo()


async def _make_ordered_queue(ts: TaskScheduler, names: list[str]) -> list[int]:
    """Create tasks and order them head→tail, returning their ids.

    Also assigns a queue-level start_at on the head to exercise timestamp moves.
    """
    ids: list[int] = []
    qid = ts._allocate_new_queue_id()
    for name in names:
        ids.append(
            ts._create_task(
                name=name,
                description=name,
                schedule=Schedule(),
            )[
                "details"
            ]["task_id"],
        )  # type: ignore[index]

    ts._set_queue(queue_id=qid, order=ids)
    ts._update_task(task_id=ids[0], start_at=datetime.now(timezone.utc))
    return ids


@pytest.mark.asyncio
@_handle_project
async def test_starting_head_promotes_next_to_scheduled_with_start_at():
    ts = TaskScheduler()
    a, b, c = await _make_ordered_queue(ts, ["A", "B", "C"])  # type: ignore[misc]

    # Sanity: A is scheduled (head with start_at), B and C are queued
    row_a = ts._filter_tasks(filter=f"task_id == {a}")[0]
    row_b = ts._filter_tasks(filter=f"task_id == {b}")[0]
    row_c = ts._filter_tasks(filter=f"task_id == {c}")[0]
    assert row_a.status == Status.scheduled
    assert row_b.status == Status.queued
    assert row_c.status == Status.queued
    assert row_a.schedule is not None
    original_start = row_a.schedule.start_at
    assert original_start is not None

    # Start head explicitly (fast-path by id) in isolation
    handle = await ts.execute(task_id=a, isolated=True)

    # After detachment, B becomes the new head and should inherit start_at and be scheduled
    row_b2 = ts._filter_tasks(filter=f"task_id == {b}")[0]
    sched_b2 = row_b2.schedule
    assert sched_b2 is not None
    assert sched_b2.prev_task is None
    assert sched_b2.start_at == original_start
    assert row_b2.status == Status.scheduled

    # Stop the active task with guidance to resume as originally scheduled
    await handle.stop(
        cancel=False,
        reason=(
            "Actually this is taking longer than I expected, let's complete this task as per our original schedule instead"
        ),
    )
    await handle.result()

    # After reinstatement, restore A→B→C with head A carrying start_at and scheduled
    row_a3 = ts._filter_tasks(filter=f"task_id == {a}")[0]
    row_b3 = ts._filter_tasks(filter=f"task_id == {b}")[0]
    row_c3 = ts._filter_tasks(filter=f"task_id == {c}")[0]

    sa3 = row_a3.schedule
    sb3 = row_b3.schedule
    sc3 = row_c3.schedule

    assert sa3 is not None
    assert sb3 is not None
    assert sc3 is not None

    # Head A restored with original start_at and points to B
    assert sa3.prev_task is None
    assert sa3.next_task == b
    assert sa3.start_at == original_start
    assert row_a3.status == Status.scheduled

    # Middle B back between A and C, no start_at, queued
    assert sb3.prev_task == a and sb3.next_task == c
    assert sb3.start_at is None
    assert row_b3.status == Status.queued

    # Tail C points back to B, no start_at, queued
    assert sc3.prev_task == b
    assert sc3.start_at is None
    assert row_c3.status == Status.queued


@pytest.mark.asyncio
@_handle_project
async def test_starting_middle_detaches_and_links_neighbors():
    ts = TaskScheduler()
    a, b, c = await _make_ordered_queue(ts, ["A", "B", "C"])  # type: ignore[misc]

    # Sanity: A is scheduled (head with start_at), B and C are queued
    row_a = ts._filter_tasks(filter=f"task_id == {a}")[0]
    row_b = ts._filter_tasks(filter=f"task_id == {b}")[0]
    row_c = ts._filter_tasks(filter=f"task_id == {c}")[0]
    assert row_a.status == Status.scheduled
    assert row_b.status == Status.queued
    assert row_c.status == Status.queued
    assert row_a.schedule is not None
    original_start = row_a.schedule.start_at
    assert original_start is not None

    # Start the middle task explicitly (fast-path by id) in isolation
    handle = await ts.execute(task_id=b, isolated=True)

    # After detachment of B, A and C should be directly linked; B should have no schedule
    row_a2 = ts._filter_tasks(filter=f"task_id == {a}")[0]
    row_b2 = ts._filter_tasks(filter=f"task_id == {b}")[0]
    row_c2 = ts._filter_tasks(filter=f"task_id == {c}")[0]

    sa2 = row_a2.schedule
    sb2 = row_b2.schedule
    sc2 = row_c2.schedule

    # Head A remains head with same start_at and points to C
    assert sa2.prev_task is None
    assert sa2.next_task == c
    assert sa2.start_at == original_start
    assert row_a2.status == Status.scheduled

    # Middle B is detached from the queue
    assert sb2 is None

    # Tail C now points back to A and must not carry start_at
    assert sc2.prev_task == a
    assert sc2.start_at is None

    # Stop the active task with guidance to resume as originally scheduled
    await handle.stop(
        cancel=False,
        reason=(
            "Actually this is taking longer than I expected, let's complete this task as per our original schedule instead"
        ),
    )
    await handle.result()

    # After reinstatement, restore A→B→C with head A carrying start_at and scheduled
    row_a3 = ts._filter_tasks(filter=f"task_id == {a}")[0]
    row_b3 = ts._filter_tasks(filter=f"task_id == {b}")[0]
    row_c3 = ts._filter_tasks(filter=f"task_id == {c}")[0]

    sa3 = row_a3.schedule
    sb3 = row_b3.schedule
    sc3 = row_c3.schedule
    assert sa3 is not None
    assert sb3 is not None
    assert sc3 is not None

    # Head A remains head with original start_at and points to B
    assert sa3.prev_task is None
    assert sa3.next_task == b
    assert sa3.start_at == original_start
    assert row_a3.status == Status.scheduled

    # Middle B is back between A and C, with no start_at and queued status
    assert sb3.prev_task == a and sb3.next_task == c
    assert sb3.start_at is None
    assert row_b3.status == Status.queued

    # Tail C points back to B and carries no start_at
    assert sc3.prev_task == b
    assert sc3.start_at is None


@pytest.mark.asyncio
@_handle_project
async def test_reinstate_head_restores_head_and_start_at():
    ts = TaskScheduler()
    head_id, next_id = await _make_ordered_queue(ts, ["H", "N"])  # type: ignore[misc]

    # Activate head in isolation and then cancel
    handle = await ts.execute(task_id=head_id, isolated=True)
    await handle.stop(cancel=True)
    await handle.result()

    # Reinstate the task back to its original head position
    out = ts._reinstate_task_to_previous_queue(task_id=head_id)
    assert out["outcome"].startswith("task reinstated"), out

    rows_h = ts._filter_tasks(filter=f"task_id == {head_id}")[0]
    rows_n = ts._filter_tasks(filter=f"task_id == {next_id}")[0]
    sched_h = rows_h.schedule
    sched_n = rows_n.schedule
    assert sched_h is not None
    assert sched_n is not None

    # Head restored with start_at; next prev points back to head and carries no start_at
    assert sched_h.prev_task is None
    assert sched_h.next_task == next_id
    assert sched_h.start_at is not None
    assert sched_n.prev_task == head_id
    assert sched_n.start_at is None


@pytest.mark.asyncio
@_handle_project
async def test_reinstate_middle_restores_links():
    ts = TaskScheduler()
    a, b, c = await _make_ordered_queue(ts, ["A", "B", "C"])  # type: ignore[misc]

    # Activate middle task (B) in isolation and cancel it
    handle = await ts.execute(task_id=b, isolated=True)
    await handle.stop(cancel=True)
    await handle.result()

    # Reinstate B → expect A→B→C restored
    _ = ts._reinstate_task_to_previous_queue(task_id=b)

    row_a = ts._filter_tasks(filter=f"task_id == {a}")[0]
    row_b = ts._filter_tasks(filter=f"task_id == {b}")[0]
    row_c = ts._filter_tasks(filter=f"task_id == {c}")[0]

    sa = row_a.schedule
    sb = row_b.schedule
    sc = row_c.schedule
    assert sa is not None
    assert sb is not None
    assert sc is not None

    assert sa.next_task == b
    assert sb.prev_task == a and sb.next_task == c
    assert sc.prev_task == b


@pytest.mark.asyncio
@_handle_project
async def test_reinstate_with_deleted_next_fallback():
    ts = TaskScheduler()
    head_id, next_id, tail_id = await _make_ordered_queue(ts, ["X", "Y", "Z"])  # type: ignore[misc]

    # Activate head in isolation and cancel
    handle = await ts.execute(task_id=head_id, isolated=True)
    await handle.stop(cancel=True)
    await handle.result()

    # Delete original next before reinstatement (drift)
    ts._delete_task(task_id=next_id)

    # Reinstate – should still place X as new head and restore start_at; next may be new head (Z) or None
    _ = ts._reinstate_task_to_previous_queue(task_id=head_id)

    row_x = ts._filter_tasks(filter=f"task_id == {head_id}")[0]
    assert row_x.schedule_prev is None
    assert row_x.schedule_start_at is not None


@pytest.mark.asyncio
@_handle_project
async def test_reinstate_refuses_when_trigger_present():
    ts = TaskScheduler()
    head_id, _ = await _make_ordered_queue(ts, ["TH", "TN"])  # type: ignore[misc]

    handle = await ts.execute(task_id=head_id, isolated=True)
    await handle.stop()
    await handle.result()

    # Auto‑reinstatement occurs on defer (cancel=False), so the head has a schedule again.
    # Adding a trigger while a schedule exists must now be refused immediately.
    with pytest.raises(ValueError):
        ts._update_task(task_id=head_id, trigger=Trigger(medium=Medium.EMAIL))


@pytest.mark.asyncio
@_handle_project
async def test_reinstate_head_with_all_neighbors_deleted_fallback():
    ts = TaskScheduler()
    head_id, next_id, tail_id = await _make_ordered_queue(ts, ["H2", "N2", "T2"])  # type: ignore[misc]

    # Activate head and cancel to record reintegration plan (captures original start_at)
    handle = await ts.execute(task_id=head_id, isolated=True)
    await handle.stop(cancel=True)
    await handle.result()

    # Delete both original neighbors before reinstatement (drift)
    ts._delete_task(task_id=next_id)
    ts._delete_task(task_id=tail_id)

    # Reinstate – should restore as a standalone head with original start_at and scheduled status
    _ = ts._reinstate_task_to_previous_queue(task_id=head_id)

    row_h = ts._filter_tasks(filter=f"task_id == {head_id}")[0]
    sched_h = row_h.schedule
    assert sched_h is not None
    assert sched_h.prev_task is None
    assert sched_h.next_task is None
    assert sched_h.start_at is not None
    assert row_h.status == Status.scheduled


@pytest.mark.asyncio
@_handle_project
async def test_reinstate_refuses_while_active():
    ts = TaskScheduler()
    head_id, _ = await _make_ordered_queue(ts, ["AH", "AN"])  # type: ignore[misc]

    handle = await ts.execute(task_id=head_id)

    # Attempt reinstatement before cancelling → must raise
    with pytest.raises(RuntimeError):
        ts._reinstate_task_to_previous_queue(task_id=head_id)

    # Clean up
    await handle.stop()
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
    assert head_row.status == Status.primed

    # Activate head and cancel
    handle = await ts.execute(task_id=h_id)
    await handle.stop(cancel=True)
    await handle.result()

    # Create a new task now – with no active and no primed, this becomes the new 'primed'
    new_tid = ts._create_task(name="NewPrimed", description="np")["details"]["task_id"]
    new_row = ts._filter_tasks(filter=f"task_id == {new_tid}")[0]
    assert new_row.status == Status.primed

    # Reinstate original head – original status was primed but conflict exists → should downgrade to queued
    _ = ts._reinstate_task_to_previous_queue(task_id=h_id)
    reinstated = ts._filter_tasks(filter=f"task_id == {h_id}")[0]
    assert reinstated.status == Status.queued


@pytest.mark.asyncio
@_handle_project
async def test_reintegration_plan_clears_on_completion():
    ts = TaskScheduler()
    head_id, next_id = await _make_ordered_queue(ts, ["HC", "NC"])  # type: ignore[misc]

    # Start head in isolation and allow it to complete naturally
    handle = await ts.execute(task_id=head_id, isolated=True)
    # Awaiting result will mark the instance as completed internally
    await handle.result()

    # Attempting reinstatement should now fail because the plan was cleared
    with pytest.raises(ValueError):
        ts._reinstate_task_to_previous_queue(task_id=head_id)


@pytest.mark.asyncio
@_handle_project
async def test_chain_then_defer_restores_next_head_start_at(monkeypatch):
    from datetime import datetime, timezone, timedelta

    ts = TaskScheduler()

    # Chain execution is the default; no environment variable required.

    # Create a chain of three tasks scheduled for next week
    future = datetime.now(timezone.utc) + timedelta(days=7)
    qid = ts._allocate_new_queue_id()
    head_id = ts._create_task(
        name="ChainHead",
        description="ch head",
        schedule=Schedule(start_at=future),
    )["details"]["task_id"]
    mid_id = ts._create_task(
        name="ChainMid",
        description="ch mid",
        schedule=Schedule(prev_task=head_id),
    )["details"]["task_id"]
    tail_id = ts._create_task(
        name="ChainTail",
        description="ch tail",
        schedule=Schedule(prev_task=mid_id),
    )["details"]["task_id"]

    # Capture the original head's start_at to compare later
    row_h = ts._filter_tasks(filter=f"task_id == {head_id}")[0]
    assert row_h.schedule is not None
    original_start = row_h.schedule.start_at
    assert original_start is not None

    # Start the head in chain mode but only allow the head to complete
    handle = await ts.execute(task_id=head_id)
    # Wait for just the head to finish
    await handle._active_task_done()

    # Start the second task (mid) and then request deferral "as originally scheduled"
    await handle.stop(
        cancel=False,
        reason="let's do the remaining tasks as per our original schedule",
    )
    await handle.result()

    # The middle task should now be reinstated as the head with the original start_at and scheduled status
    row_mid = ts._filter_tasks(filter=f"task_id == {mid_id}")[0]
    sched_mid = row_mid.schedule
    assert sched_mid is not None
    assert sched_mid.prev_task is None
    assert sched_mid.start_at == original_start
    assert row_mid.status == Status.scheduled

    # The tail should be queued behind the reinstated head without a start_at
    row_tail = ts._filter_tasks(filter=f"task_id == {tail_id}")[0]
    sched_tail = row_tail.schedule
    assert sched_tail is not None
    assert sched_tail.prev_task == mid_id
    assert sched_tail.start_at is None
