import pytest
from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.actor.simulated import SimulatedActor
from unity.task_scheduler.types.status import Status


@pytest.mark.asyncio
@_handle_project
async def test_defer_reinstate_failure_fallback_downgrades_status(monkeypatch):
    """If reinstate fails during a defer stop, status should downgrade to prior or 'queued'."""

    # Use an indefinite actor; stop explicitly
    actor = SimulatedActor(steps=None, duration=None)
    ts = TaskScheduler(actor=actor)

    # Create a queued task and start it
    tid = ts._create_task(name="T", description="T")["details"]["task_id"]
    handle = await ts.execute(task_id=tid)

    # Ensure a reintegration plan exists to carry original_status
    # Now sabotage reinstate to force fallback
    def boom(*args, **kwargs):
        raise RuntimeError("reinstate failed")

    monkeypatch.setattr(ts, "_reinstate_to_previous_queue", boom, raising=True)

    # Defer via interject with defer-like wording
    await handle.interject("Let's stop and resume later as originally scheduled.")
    result = await handle.result()

    assert "stopped" in result.lower()

    # After fallback, the task should no longer be active; expect queued (prior was queued)
    rows = ts._filter_tasks(filter=f"task_id == {tid}")
    assert any(r.status != Status.active for r in rows)
    assert any(
        r.status in (Status.queued, Status.scheduled, Status.primed) for r in rows
    )


@pytest.mark.asyncio
@_handle_project
async def test_orphan_active_guard_prevents_new_execution(monkeypatch):
    """If a row is marked 'active' without an active pointer, execute should refuse to start a new task."""

    actor = SimulatedActor(steps=None, duration=None)
    ts = TaskScheduler(actor=actor)

    # Create task and manually simulate an orphan 'active' row by updating instance status via internal instance method
    tid = ts._create_task(name="A", description="A")["details"]["task_id"]

    # Promote to active through normal execute
    h = await ts.execute(task_id=tid)
    # Immediately clear the pointer to simulate crash-after-activation
    ts._active_task = None  # type: ignore[attr-defined]

    # Now, attempt to start another task should be rejected
    with pytest.raises(RuntimeError):
        await ts.execute(task_id=tid)
    # Cleanup: stop the original handle to avoid leaking background threads
    try:
        await h.stop(cancel=False)
        await h.result()
    except Exception:
        pass


@pytest.mark.asyncio
@_handle_project
async def test_disallow_internal_status_edits_on_active_task(monkeypatch):
    """Direct updates to the active task's status should be refused and must go through the live handle."""

    actor = SimulatedActor(steps=None, duration=None)
    ts = TaskScheduler(actor=actor)

    tid = ts._create_task(name="B", description="B")["details"]["task_id"]
    h = await ts.execute(task_id=tid)

    # Attempt to change status away from active via update API should raise
    with pytest.raises(RuntimeError):
        ts._update_task(task_id=tid, status=Status.paused)

    # Clean stop to avoid leaking background threads
    await h.stop(cancel=False)
    await h.result()


@pytest.mark.asyncio
@_handle_project
async def test_validated_write_rejects_active_status_direct_write(monkeypatch):
    """Writing status='active' through the validated funnel should be rejected."""

    ts = TaskScheduler()
    tid = ts._create_task(name="C", description="C")["details"]["task_id"]

    with pytest.raises(ValueError):
        ts._validated_write(
            task_id=tid,
            entries={"status": "active"},
            err_prefix="Test:",
        )
