import pytest
from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.actor.simulated import SimulatedActor


@pytest.mark.asyncio
@_handle_project
async def test_defer_reinstate_failure_fallback_downgrades_status(monkeypatch):
    """If reinstate fails during a defer stop, status should downgrade to prior or 'queued'."""

    # Use a fast actor
    actor = SimulatedActor(steps=5)
    ts = TaskScheduler(actor=actor)

    # Create a queued task and start it
    tid = ts._create_task(name="T", description="T")["details"]["task_id"]
    handle = await ts.execute_task(text=str(tid))

    # Ensure a reintegration plan exists to carry original_status
    # Now sabotage reinstate to force fallback
    def boom(*args, **kwargs):
        raise RuntimeError("reinstate failed")

    monkeypatch.setattr(ts, "reinstate_to_previous_queue", boom, raising=True)

    # Defer via interject with defer-like wording
    await handle.interject("Let's stop and resume later as originally scheduled.")
    result = await handle.result()

    assert "stopped" in result.lower()

    # After fallback, the task should no longer be active; expect queued (prior was queued)
    rows = ts._filter_tasks(filter=f"task_id == {tid}")
    assert any(r.get("status") != "active" for r in rows)
    assert any(r.get("status") in ("queued", "scheduled", "primed") for r in rows)
