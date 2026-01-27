"""
Tests for `TaskScheduler.execute` which returns an `ActiveQueue` handle.

These largely mirror *test_active_task.py* but go through the full
`TaskScheduler` surface so that we cover the integration layer that
retrieves the task from storage, wraps it in `ActiveTask` internally, and wires the
actor‐instance into the scheduler via an `ActiveQueue` public handle.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Dict, List
from datetime import datetime, timezone

import pytest

from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.actor.simulated import SimulatedActor
from unity.actor.simulated import SimulatedActorHandle
from unity.task_scheduler.types.schedule import Schedule
from unity.task_scheduler.types.activated_by import ActivatedBy
from unity.task_scheduler.types.status import Status

#  The helper used in the existing test‑suite – applies project‑level monkey‐
#  patches (e.g. env vars, tracers) so we keep behaviour consistent.
from tests.helpers import _handle_project

# --------------------------------------------------------------------------- #
#  Test helpers                                                               #
# --------------------------------------------------------------------------- #


async def _make_scheduler_with_task(description: str, *, steps: int = 1):
    """Return *(scheduler, handle)* where *handle* is the active task."""
    # Always keep the simulated actor alive indefinitely; tests will stop explicitly
    actor = SimulatedActor(steps=None, duration=None)
    scheduler = TaskScheduler(actor=actor)

    task_id = scheduler._create_task(name=description, description=description)[
        "details"
    ]["task_id"]
    handle = await scheduler.execute(task_id=task_id)
    return scheduler, handle


async def _make_ordered_queue(ts: TaskScheduler, names: List[str]) -> List[int]:
    """Create tasks and order them head→tail, returning the task_ids.

    Also assigns a queue-level start_at on the head.
    """
    ids: List[int] = []
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

    # Establish explicit order using the current queue snapshot as original
    ts._set_queue(queue_id=qid, order=ids)

    # Put a start_at timestamp on the head only
    ts._update_task(task_id=ids[0], start_at=datetime.now(timezone.utc))
    return ids


# --------------------------------------------------------------------------- #
#  0. Ask                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_ask(monkeypatch):
    """`ActiveTask.ask` should forward to the wrapped plan exactly once."""

    calls: Dict[str, int] = {"ask": 0}

    original_ask = SimulatedActorHandle.ask

    @functools.wraps(original_ask)
    async def spy_ask(self, question: str) -> str:  # type: ignore[override]
        calls["ask"] += 1
        return await original_ask(self, question)

    monkeypatch.setattr(SimulatedActorHandle, "ask", spy_ask, raising=True)

    _scheduler, task = await _make_scheduler_with_task(
        "Analyse new product launch performance.",
    )

    # Perform a read-only ask on the returned handle – should delegate once
    ask_h = await task.ask("Do we have any early metrics?")
    await ask_h.result()

    # Explicitly stop to avoid relying on step-based completion
    task.stop(cancel=False)
    await task.result()

    assert calls["ask"] == 1, "ask must be called exactly once"


# --------------------------------------------------------------------------- #
#  1. Interjection                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_interject(monkeypatch):
    """`ActiveTask.interject` should forward to the wrapped plan exactly once."""

    calls: Dict[str, int] = {"interject": 0}

    original_interject = SimulatedActorHandle.interject

    @functools.wraps(original_interject)
    async def spy_interject(self, instruction: str, *, images=None) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return await original_interject(self, instruction, images=images)

    monkeypatch.setattr(SimulatedActorHandle, "interject", spy_interject, raising=True)

    _scheduler, task = await _make_scheduler_with_task(
        "Investigate competitor pricing.",
    )

    await task.interject("First gather public filings.")
    # Give the background thread one beat to process the step counter.
    await asyncio.sleep(0.2)
    # Gracefully stop to avoid leaking the background thread.
    task.stop(cancel=False)
    await task.result()

    assert calls["interject"] == 1, "interject must be called exactly once"


# --------------------------------------------------------------------------- #
#  2. Pause / Resume                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_pause_resume(monkeypatch):
    """The wrapper should transparently forward `pause` and `resume`."""

    counts: Dict[str, int] = {"pause": 0, "resume": 0}

    orig_pause = SimulatedActorHandle.pause
    orig_resume = SimulatedActorHandle.resume

    @functools.wraps(orig_pause)
    def spy_pause(self) -> str:  # type: ignore[override]
        counts["pause"] += 1
        return orig_pause(self)

    @functools.wraps(orig_resume)
    def spy_resume(self) -> str:  # type: ignore[override]
        counts["resume"] += 1
        return orig_resume(self)

    monkeypatch.setattr(SimulatedActorHandle, "pause", spy_pause, raising=True)
    monkeypatch.setattr(SimulatedActorHandle, "resume", spy_resume, raising=True)

    _scheduler, task = await _make_scheduler_with_task(
        "Run SEO audit for the website.",
    )

    # Pause, wait a moment to ensure the thread blocks, then resume.
    await task.pause()
    await asyncio.sleep(0.1)
    await task.resume()
    # Stop the task to finish quickly and collect counts.
    task.stop(cancel=False)
    await task.result()

    assert counts == {"pause": 1, "resume": 1}, "pause/resume each called once"


# --------------------------------------------------------------------------- #
#  3. Stop                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_stop(monkeypatch):
    """Calling `ActiveTask.stop` should proxy to the plan and mark it done."""

    called = {"stop": 0}

    orig_stop = SimulatedActorHandle.stop

    @functools.wraps(orig_stop)
    def spy_stop(self, reason: str | None = None) -> str:  # type: ignore[override]
        called["stop"] += 1
        return orig_stop(self, reason=reason)

    monkeypatch.setattr(SimulatedActorHandle, "stop", spy_stop, raising=True)

    _scheduler, task = await _make_scheduler_with_task(
        "Extract sentiment from reviews.",
    )

    task.stop(cancel=False)
    result = await task.result()

    assert called["stop"] == 1, "stop must be invoked exactly once"
    assert "stopped" in result.lower()
    assert task.done(), "`done()` should report True after stopping"


# --------------------------------------------------------------------------- #
#  4. Result & Done Lifecycle                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_result_and_done():
    """A normal workflow should complete once enough steps have been taken."""

    _scheduler, task = await _make_scheduler_with_task(
        "Compile coverage metrics.",
    )

    # Perform an interjection for activity, then stop explicitly
    await task.interject("Provide initial outline first.")
    task.stop(cancel=False)
    result = await task.result()

    assert "stopped" in result.lower()
    assert task.done(), "`done()` must return True after explicit stop"


# --------------------------------------------------------------------------- #
#  6.1. Logged wrapper exposes append_to_queue with correct metadata           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_handle_introspection():
    """
    The handle returned by TaskScheduler.execute should expose `append_to_queue`
    with the correct signature and a meaningful docstring via standard inspection.
    """

    import inspect as _inspect  # local import for test isolation

    # Immediate completion per task to avoid timing races; we don't need to
    # exercise the loop, only to obtain the handle for introspection.
    actor = SimulatedActor(steps=None, duration=None)
    ts = TaskScheduler(actor=actor)

    # Create a single runnable task and start it by id
    desc = "Introspection target"
    task_id = ts._create_task(name=desc, description=desc)["details"]["task_id"]  # type: ignore[index]
    handle = await ts.execute(task_id=task_id)

    # The execute surface returns an ActiveQueue handle
    from unity.task_scheduler.active_queue import ActiveQueue  # local import

    assert handle.__class__ is ActiveQueue
    assert handle.__class__.__name__ == "ActiveQueue"

    # The custom queue method must be directly accessible on the proxy
    assert hasattr(handle, "append_to_queue"), "append_to_queue not exposed on handle"
    proxied_method = getattr(handle, "append_to_queue")
    assert callable(proxied_method)

    # The signature should accept exactly one required parameter: task_id
    sig = _inspect.signature(proxied_method)
    params = list(sig.parameters.values())
    assert len(params) == 1 and params[0].name == "task_id"

    # Docstring should be present and describe appending to the live task queue
    doc = _inspect.getdoc(proxied_method) or ""
    assert "append" in doc.lower() and "task" in doc.lower()

    # Cleanup: ensure any background work is finalised quickly
    handle.stop(cancel=False)
    await handle.result()


# --------------------------------------------------------------------------- #
#  6.2. End‑to‑end: async tool loop can call dynamic append_to_queue           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_async_tool_loop_calls_append_helper():
    """
    End-to-end: An outer async tool loop (proxying a higher-level orchestrator) should be able to:
      1) call TaskScheduler.execute to start a task, and then
      2) call the dynamically exposed helper whose name starts with `append_to_queue_`
         to append another task while the first is running.
    """
    # Localized imports to mirror other async tool loop tests
    from unity.common.async_tool_loop import start_async_tool_loop
    from unity.common.llm_client import new_llm_client
    from tests.settings import SETTINGS  # reuse cache/tracing settings
    from tests.async_helpers import (  # wait helpers
        _wait_for_tool_request,
        _wait_for_assistant_call_prefix,
        _wait_for_tool_message_prefix,
        _wait_for_condition,
    )

    # Keep the actor alive (no auto-complete by steps/time)
    actor = SimulatedActor(steps=None, duration=None)
    ts = TaskScheduler(actor=actor)

    # Create a singleton queue with one task A, and a standalone task B to append
    (a_id,) = tuple(await _make_ordered_queue(ts, ["ATask"]))  # type: ignore[misc]
    b_id = ts._create_task(name="BTask", description="BTask")["details"]["task_id"]  # type: ignore[index]

    # Tool that starts execution and returns the steerable ActiveQueue handle
    async def scheduler_execute(*, task_id: int) -> "object":
        return await ts.execute(task_id=task_id)

    # LLM client configured like other async tool loop tests
    client = new_llm_client()

    # Clear, step-by-step instructions to the model:
    #  1) call `scheduler_execute(task_id=a_id)`
    #  2) once running, call helper starting with `append_to_queue_` with task_id=b_id
    client.set_system_message(
        "You are orchestrating a running task.\n"
        "First, start the task by calling the tool `scheduler_execute` with the provided task_id.\n"
        "After it is in-flight, call the helper whose name starts with `append_to_queue_` and pass\n"
        f"task_id={int(b_id)} to append that task to the current queue.\n"
        "Do not invent tool names; use exactly the provided names. Finish with a short OK.",
    )

    outer = start_async_tool_loop(
        client,
        message=f"Start the task {int(a_id)} now, then append {int(b_id)} to its queue.",
        tools={"scheduler_execute": scheduler_execute},
        max_steps=30,
        timeout=300,
    )

    # 1) Wait deterministically until `scheduler_execute` has been requested
    await _wait_for_tool_request(client, "scheduler_execute")

    # Ensure the tool-result placeholder for scheduler_execute is appended so the
    # loop is fully between turns (prevents double logging on immediate interjection).
    await _wait_for_tool_message_prefix(client, "scheduler_execute")

    # 2) The loop won't produce another assistant turn until a tool finishes or we interject.
    #    Prompt the model explicitly to call the dynamic helper whose name starts with `append_to_queue_`.
    await outer.interject(
        f"Now append task_id={int(b_id)} to the current queue using the helper whose name starts with 'append_to_queue_'.",
    )

    # 3) Next, wait until the assistant calls a dynamic helper whose name starts with append_to_queue_
    await _wait_for_assistant_call_prefix(client, "append_to_queue_")
    # And wait until the tool result message for the append helper is recorded
    await _wait_for_tool_message_prefix(client, "append_to_queue_")

    # Verify that B was appended behind A in the live queue (wait deterministically)
    async def _has_appended() -> bool:
        live_local = ts._get_queue_for_task(task_id=a_id)
        ids_local = [getattr(r, "task_id", None) for r in (live_local or [])]
        try:
            return bool(
                ids_local
                and ids_local[0] == a_id
                and (b_id in ids_local)
                and ids_local.index(b_id) == len(ids_local) - 1,
            )
        except Exception:
            return False

    await _wait_for_condition(_has_appended, poll=0.01, timeout=60.0)

    # Proactively stop the running task inside the scheduler to avoid hanging on
    # a never-ending simulated actor (steps=None, duration=None).
    try:
        active = getattr(ts, "_active_task", None)
        if active is not None:
            active.stop(cancel=False)
    except Exception:
        pass

    # Also stop the outer async tool loop; the end-to-end goal (append helper) is verified.
    try:
        outer.stop("test cleanup")
    except Exception:
        pass

    # Allow the outer loop to finish cleanly
    try:
        final = await asyncio.wait_for(outer.result(), timeout=120)
        assert isinstance(final, str)
    except Exception:
        # Best-effort cleanup if the model doesn't finish on its own
        outer.stop("cleanup")
        await asyncio.wait_for(asyncio.shield(outer.result()), timeout=120)


# --------------------------------------------------------------------------- #
#  A. Activation metadata                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_sets_activated_by_explicit():
    """Starting a task explicitly via execute should set activated_by='explicit'."""

    actor = SimulatedActor(steps=None, duration=None)
    ts = TaskScheduler(actor=actor)

    # Seed a simple queued task
    name = "Simple queued task"
    task_id = ts._create_task(name=name, description=name)["details"]["task_id"]

    # Start by id (fast-path)
    handle = await ts.execute(task_id=task_id)
    handle.stop(cancel=False)
    await handle.result()

    # Verify activated_by on the activated instance (may already be completed)
    rows = ts._filter_tasks(filter=f"task_id == {task_id}")
    assert any(r.activated_by == ActivatedBy.explicit for r in rows)


@pytest.mark.asyncio
@_handle_project
async def test_update_status_cannot_force_active():
    """Direct status updates cannot set 'active' and should not set 'activated_by'."""

    actor = SimulatedActor(steps=None, duration=None)
    ts = TaskScheduler(actor=actor)

    # Create a normal queued task
    label = "Cannot force active"
    task_id = ts._create_task(name=label, description=label)["details"]["task_id"]

    # Attempt to force 'active' via status update should fail
    with pytest.raises(ValueError):
        ts._update_task(task_id=task_id, status="active")

    # Ensure no activation metadata exists prior to activation
    rows = ts._filter_tasks(filter=f"task_id == {task_id}")
    assert len(rows) == 1
    assert rows[0].activated_by is None

    # Change a non-active status and ensure activated_by remains unset
    ts._update_task(task_id=task_id, status="paused")
    rows2 = ts._filter_tasks(filter=f"task_id == {task_id}")
    assert rows2[0].status == Status.paused
    assert rows2[0].activated_by is None


@pytest.mark.asyncio
@_handle_project
async def test_tasks_table_has_activated_by_column():
    """The Tasks context should include the activated_by column based on the Task model."""

    actor = SimulatedActor(steps=None, duration=None)
    ts = TaskScheduler(actor=actor)

    # Create any task to ensure context exists
    title = "Column presence check"
    _ = ts._create_task(name=title, description=title)

    cols = ts._list_columns()
    if isinstance(cols, dict):
        assert "activated_by" in cols
    else:
        assert "activated_by" in cols


# --------------------------------------------------------------------------- #
#  B. Explicit activation scope: isolate vs chain                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_isolated_execute_detaches_entirely(monkeypatch):
    """Explicit isolation prompt: Detach the activated task entirely from the queue.

    Scenario: three tasks A->B->C, activate B with a prompt that is *explicitly*
    and *unambiguously* requesting isolation (detach B from the queue and do not
    keep followers attached). Expect B detached, A->C linked, head's start_at
    preserved/propagated, and B's schedule cleared.
    """

    actor = SimulatedActor(steps=0)
    ts = TaskScheduler(actor=actor)

    # Build queue A->B->C with start_at on A
    a, b, c = await _make_ordered_queue(ts, ["A", "B", "C"])  # type: ignore[misc]

    # Execute B with an explicit isolation request (avoid pure-numeric fast path)
    handle = await ts.execute(
        task_id=b,
        isolated=True,
    )
    await handle.result()

    rows_a = ts._filter_tasks(filter=f"task_id == {a}")
    rows_b = ts._filter_tasks(filter=f"task_id == {b}")
    rows_c = ts._filter_tasks(filter=f"task_id == {c}")

    # B should be isolated as a single-task head (no prev/next followers)
    sched_b = rows_b[0].schedule
    assert sched_b is None

    # A should now link directly to C; C.prev_task should be A
    sched_a = rows_a[0].schedule
    sched_c = rows_c[0].schedule
    assert sched_c is not None
    assert sched_a is not None
    assert sched_a.next_task == c
    assert sched_c.prev_task == a

    # Only the head owns start_at → ensure C (non-head) does not inherit it unless it became head
    # Here A remains head, so C must not have start_at
    assert sched_c.start_at is None


@pytest.mark.asyncio
@_handle_project
async def test_isolated_execute_start_at_moves(monkeypatch):
    """Branch A (head case): Explicit isolation – if activated task was head, next becomes head and inherits start_at."""

    actor = SimulatedActor(steps=0)
    ts = TaskScheduler(actor=actor)

    x, y = await _make_ordered_queue(ts, ["X", "Y"])  # type: ignore[misc]

    # Execute X (head) with an explicit isolation request (avoid pure-numeric fast path)
    handle = await ts.execute(task_id=x, isolated=True)
    await handle.result()

    rows_x = ts._filter_tasks(filter=f"task_id == {x}")
    rows_y = ts._filter_tasks(filter=f"task_id == {y}")

    # X detached
    assert rows_x[0].schedule is None

    # Y becomes new head: prev_task=None and has start_at
    sched_y = rows_y[0].schedule
    assert sched_y is not None
    assert sched_y.prev_task is None
    assert sched_y.start_at is not None


@pytest.mark.asyncio
@_handle_project
async def test_execute_default_keeps_followers():
    """Default behaviour: Keep followers attached when activating a middle task.

    Scenario: A2->B2->C2. Activate B2 without any explicit isolation request.
    Expect B2 to become sub-head (prev=None), keep next pointer to C2, and
    C2.prev_task == B2, with only the head owning start_at.
    """

    actor = SimulatedActor(steps=0)
    ts = TaskScheduler(actor=actor)

    a, b, c = await _make_ordered_queue(ts, ["A2", "B2", "C2"])  # type: ignore[misc]

    handle = await ts.execute(task_id=b)

    # Yield once to allow activation-side linkage writes to settle without advancing the queue
    await asyncio.sleep(0)

    # Inspect linkage immediately after activation
    rows_b = ts._filter_tasks(filter=f"task_id == {b}")
    rows_c = ts._filter_tasks(filter=f"task_id == {c}")

    sched_b = rows_b[0].schedule
    sched_c = rows_c[0].schedule
    assert sched_b is not None
    assert sched_c is not None

    # B becomes sub-head of its chain
    assert sched_b.prev_task is None
    assert sched_b.next_task == c
    # C follows B
    assert sched_c.prev_task == b
    # C must not carry start_at (non-head)
    assert sched_c.start_at is None

    # Stop to avoid leaking the background task and wait for shutdown
    try:
        handle.stop(cancel=True)
    except Exception:
        pass
    await handle.result()
