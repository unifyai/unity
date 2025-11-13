from __future__ import annotations

import asyncio
import pytest

from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.actor.simulated import SimulatedActor


async def _wait_for_condition(predicate, *, poll: float = 0.01, timeout: float = 5.0):
    import time as _time

    start = _time.perf_counter()
    while _time.perf_counter() - start < timeout:
        try:
            if await predicate():
                return
        except Exception:
            pass
        await asyncio.sleep(poll)
    raise TimeoutError("timeout waiting for condition")


def _first_wrapped(handle):
    """Best-effort unwrap helper: returns the first wrapped handle (if any)."""
    try:
        if hasattr(handle, "get_wrapped_handles"):
            res = handle.get_wrapped_handles()
            # dict → first value; list/tuple → first item; single handle → itself
            if isinstance(res, dict):
                return list(res.values())[0]
            if isinstance(res, (list, tuple)):
                return res[0]
            return res
    except Exception:
        pass
    return None


def _current_actor_handle(outer_handle):
    """
    Unwrap ActiveQueue → ActiveTask → SimulatedActorHandle (first wrapped at each level).
    """
    try:
        active_task = _first_wrapped(outer_handle)  # ActiveTask
        return _first_wrapped(active_task)  # SimulatedActorHandle
    except Exception:
        return None


@pytest.mark.asyncio
@_handle_project
async def test_execute_outer_append_to_queue_passthrough_completes_both():
    """
    Start an execute loop on a singleton queue head (fast-path by id). While the inner
    ActiveQueue/ActiveTask is in-flight, call handle.append_to_queue(...)
    and verify that this appends the follower to the ActiveQueue.
    Then explicitly interject on each inner SimulatedActorHandle to complete both tasks.
    """
    # Arrange: use steps=2 so ActiveTask.result() simulates one step, and exactly one
    # explicit interjection completes each task deterministically (A then B).
    actor = SimulatedActor(steps=2, duration=None)
    ts = TaskScheduler(actor=actor)

    # Create a singleton queue head A
    qid = ts._allocate_new_queue_id()
    a_id = ts._create_task(
        name="E2E_A_APPEND",
        description="E2E_A_APPEND",
        queue_id=qid,
    )["details"][
        "task_id"
    ]  # type: ignore[index]
    ts._set_queue(queue_id=qid, order=[a_id])

    # Create a standalone follower candidate B (not yet in the queue)
    b_id = ts._create_task(name="E2E_B_APPEND", description="E2E_B_APPEND")["details"][
        "task_id"
    ]  # type: ignore[index]

    # Start the execute loop via fast-path (numeric string id)
    handle = await ts.execute(text=str(a_id))

    # Act: Call append_to_queue on the returned handle (ActiveQueue in fast-path)
    handle.append_to_queue(task_id=b_id)

    # Assert propagation: the live queue should now contain A followed by B
    async def _has_appended():
        live = ts._get_queue_for_task(task_id=a_id)
        ids = [getattr(r, "task_id", None) for r in (live or [])]
        return bool(
            ids
            and ids[0] == a_id
            and (b_id in ids)
            and (ids.index(b_id) == len(ids) - 1),
        )

    await _wait_for_condition(_has_appended, poll=0.01, timeout=5.0)

    # Drive Task A to completion by explicitly interjecting the inner SimulatedActorHandle
    first_actor = _current_actor_handle(handle)
    assert first_actor is not None, "could not locate inner actor handle for first task"
    await first_actor.interject("STEP_A")

    # Wait until ActiveQueue announces the next task started (robust vs identity checks)
    async def _wait_started_b(timeout: float = 30.0):
        import time as _time

        deadline = _time.perf_counter() + timeout
        while _time.perf_counter() < deadline:
            evt = await handle.next_notification()  # type: ignore[attr-defined]
            try:
                if (
                    isinstance(evt, dict)
                    and evt.get("type") == "queue.task.started"
                    and int(evt.get("task_id", -1)) == int(b_id)
                ):
                    return
            except Exception:
                pass
        raise TimeoutError("timeout waiting for queue.task.started for B")

    await _wait_started_b()

    # Drive Task B to completion via its inner SimulatedActorHandle
    second_actor = _current_actor_handle(handle)
    assert (
        second_actor is not None
    ), "could not locate inner actor handle for second task"
    await second_actor.interject("STEP_B")

    # Await final result and ensure both tasks are reported as completed
    final = await handle.result()
    assert isinstance(final, str) and final, "outer result should be a non-empty string"
    assert "E2E_A_APPEND" in final and "E2E_B_APPEND" in final, (
        "expected both task names in completion summary; got: " + final
    )
