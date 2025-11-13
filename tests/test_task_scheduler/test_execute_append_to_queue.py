import pytest

from unity.actor.simulated import SimulatedActor
from unity.task_scheduler.task_scheduler import TaskScheduler
from tests.test_async_tool_loop.async_helpers import _wait_for_condition
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_execute_early_append_to_queue_is_buffered_and_replayed():
    """
    Calling ExecuteLoopHandle.append_to_queue() before an ActiveQueue is adopted
    should buffer the request and replay it as soon as the ActiveQueue is
    in-flight in passthrough mode, appending the task to the tail.
    """

    # Short-running actor so both tasks complete quickly and deterministically
    class _Short(SimulatedActor):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            kw["steps"] = None
            kw["duration"] = 0.2  # small but ample window for adoption/flush
            super().__init__(*a, **kw)

    actor = _Short()
    ts = TaskScheduler(actor=actor)

    # Seed two detached tasks A (to execute) and B (to append later)
    a_id = ts._create_task(name="EARLY_A", description="A")["details"]["task_id"]  # type: ignore[index]
    b_id = ts._create_task(name="EARLY_B", description="B")["details"]["task_id"]  # type: ignore[index]

    # Start execution by numeric id – returns an ExecuteLoopHandle (outer loop)
    outer = await ts.execute(text=str(a_id))

    # Immediately request an append while the ActiveQueue child is not yet adopted.
    # This should be buffered and later replayed once the queue is adopted in passthrough.
    outer.append_to_queue(task_id=int(b_id))  # type: ignore[attr-defined]

    # Wait until the live queue for A shows B appended at the tail
    async def _queue_has_b_at_tail():
        q = ts._get_queue_for_task(task_id=int(a_id))
        ids = [getattr(r, "task_id", None) for r in (q or [])]
        return bool(ids) and (int(b_id) in ids) and (ids[-1] == int(b_id))

    await _wait_for_condition(_queue_has_b_at_tail, poll=0.01, timeout=10.0)

    # Allow the queue to run to completion and capture the final summary
    final = await outer.result()

    # Final asserts: both tasks completed and appear in the queue summary
    assert isinstance(final, str)
    assert "Completed the following tasks:" in final
    assert f"Task {int(a_id)}: EARLY_A" in final
    assert f"Task {int(b_id)}: EARLY_B" in final

    # Note: after completion, the runnable view may be empty; we already asserted
    # the tail condition before completion and verified both tasks completed.
