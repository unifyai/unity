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

import os
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
    actor = SimulatedActor(steps=steps)
    scheduler = TaskScheduler(actor=actor)

    task_id = scheduler._create_task(name=description, description=description)[
        "details"
    ]["task_id"]
    handle = await scheduler.execute(text=str(task_id))
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
        steps=1,
    )

    await task.ask("Do we have any early metrics?")
    # Give the background worker a beat and await completion.
    await asyncio.sleep(0.2)
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
    async def spy_interject(self, instruction: str) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return await original_interject(self, instruction)

    monkeypatch.setattr(SimulatedActorHandle, "interject", spy_interject, raising=True)

    _scheduler, task = await _make_scheduler_with_task(
        "Investigate competitor pricing.",
        steps=2,
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
        steps=2,
    )

    # Pause, wait a moment to ensure the thread blocks, then resume.
    task.pause()
    await asyncio.sleep(0.1)
    task.resume()
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
        steps=5,
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
        steps=1,
    )

    # One interjection increments the internal step counter to fulfil `_steps`.
    await task.interject("Provide initial outline first.")
    result = await task.result()

    assert "completed" in result.lower()
    assert task.done(), "`done()` must return True after natural completion"


# --------------------------------------------------------------------------- #
#  5. Free-form execute triggers internal ask                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_invokes_ask_when_id_missing(monkeypatch):
    """Executing via *description only* should call TaskScheduler.ask exactly once."""

    description = "prepare the monthly analytics dashboard."

    # Keep the activated task in-flight so the queue does not advance before assertions.
    actor = SimulatedActor(steps=1)
    ts = TaskScheduler(actor=actor)

    # Seed one queued task (the one we'll start)
    _ = ts._create_task(name=description, description=description)

    calls = {"ask": 0}

    original_ask = TaskScheduler.ask

    @functools.wraps(original_ask)
    async def spy_ask(self, text: str, **kw):  # type: ignore[override]
        calls["ask"] += 1
        return await original_ask(self, text, **kw)

    monkeypatch.setattr(TaskScheduler, "ask", spy_ask, raising=True)

    # Execute via free-form prompt WITHOUT numeric id
    handle = await ts.execute(text=description)

    # Wait for completion
    await handle.interject("please be quick")
    await handle.result()

    assert calls["ask"] == 1, "TaskScheduler.ask should be invoked exactly once"


# --------------------------------------------------------------------------- #
#  6.1. Logged wrapper exposes append_to_queue with correct metadata           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_returns_logged_handle_with_append_to_queue_introspection():
    """
    The handle returned by TaskScheduler.execute is wrapped by a logging proxy
    but must present itself as `ActiveQueue` and expose `append_to_queue` with
    the correct signature and docstring via standard inspection.
    """

    import inspect as _inspect  # local import for test isolation

    # Immediate completion per task to avoid timing races; we don't need to
    # exercise the loop, only to obtain the handle for introspection.
    actor = SimulatedActor(steps=0)
    ts = TaskScheduler(actor=actor)

    # Create a single runnable task and start it by id
    desc = "Introspection target"
    task_id = ts._create_task(name=desc, description=desc)["details"]["task_id"]  # type: ignore[index]
    handle = await ts.execute(text=str(task_id))

    # From the caller's perspective, the proxy should present the same class
    # as the inner handle: ActiveQueue. Avoid relying on implementation
    # details like `_inner`.
    from unity.task_scheduler.active_queue import ActiveQueue  # local import

    assert handle.__class__ is ActiveQueue
    assert handle.__class__.__name__ == "ActiveQueue"

    # The custom queue method must be directly accessible on the proxy
    assert hasattr(handle, "append_to_queue"), "append_to_queue not exposed on handle"
    proxied_method = getattr(handle, "append_to_queue")
    assert callable(proxied_method)

    # Compare signature and docstring against the inner BOUND method to avoid
    # bound vs unbound discrepancies (omit self in bound signatures).
    inner_bound = getattr(getattr(handle, "__wrapped__", handle), "append_to_queue")
    assert str(_inspect.signature(proxied_method)) == str(
        _inspect.signature(inner_bound),
    )

    # Docstrings should also match exactly (functools.wraps should preserve).
    assert _inspect.getdoc(proxied_method) == _inspect.getdoc(inner_bound)

    # Cleanup: ensure any background work is finalised quickly
    try:
        await handle.result()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  6. New task creation & execution                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_creates_new_task_and_executes(monkeypatch):
    """When the task clearly does not exist the scheduler should create it via
    `update` and then start it – `TaskScheduler.update` must therefore be
    invoked exactly once (or more, in very unlikely multi-step flows)."""

    description = "Organise annual security audit report."

    # Use a short duration so activation doesn't immediately advance the queue
    # before we can assert on linkage semantics.
    actor = SimulatedActor(duration=0.5)
    ts = TaskScheduler(actor=actor)

    # ---- spy on _create_task -----------------------------------------------
    calls: Dict[str, int] = {"_create_task": 0}

    original_create = TaskScheduler._create_task

    @functools.wraps(original_create)
    def spy_create(self, *, name: str, description: str, **kw):  # type: ignore[override]
        calls["_create_task"] += 1
        return original_create(self, name=name, description=description, **kw)

    monkeypatch.setattr(TaskScheduler, "_create_task", spy_create, raising=True)

    # ---- execute (no prior task with that description exists) -------------
    handle = await ts.execute(text=description)

    # Get the final result.
    await handle.result()

    # ---- assertions --------------------------------------------------------
    assert (
        calls["_create_task"] >= 1
    ), "Expected at least one call to TaskScheduler._create_task"

    # Verify that a task with the expected description now exists
    # Description may be normalised (e.g. trailing period removed).  Accept any
    # task whose *name* or *description* contains our original phrase without
    # the trailing period.
    created_tasks = ts._filter_tasks()
    phrase = description.rstrip(".").casefold()
    assert any(
        phrase in t.name.casefold() or phrase in t.description.casefold()
        for t in created_tasks
    ), "A new task with the provided description should have been created"


# --------------------------------------------------------------------------- #
#  6.2. Dynamic helper append_to_queue – end-to-end via async tool loop       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_dynamic_helper_append_to_queue_end_to_end():
    """
    End-to-end: start a tool loop that returns an ActiveQueue handle created
    by TaskScheduler.execute, then instruct the LLM to call the dynamic
    `append_to_queue` helper with an explicit task_id. Verify the helper is
    called and the queue membership reflects the append.
    """
    import unify
    from unity.common.async_tool_loop import start_async_tool_loop
    from tests.test_async_tool_loop.async_helpers import (
        _wait_for_tool_request,
        _wait_for_assistant_call_prefix,
        _wait_for_tool_message_prefix,
    )
    from tests.helpers import SETTINGS

    # Build a scheduler with a singleton queue (A) and a standalone task (B).
    # Use a short, step-based simulated actor so the inner handle stays alive.
    actor = SimulatedActor(steps=3, duration=None)
    ts = TaskScheduler(actor=actor)

    # Create a singleton queue head A
    qid = ts._allocate_new_queue_id()
    a_id = ts._create_task(name="E2E_A", description="E2E_A", queue_id=qid)["details"][
        "task_id"
    ]  # type: ignore[index]
    ts._set_queue(queue_id=qid, order=[a_id])

    # Create a standalone follower candidate B (not queued yet)
    b_id = ts._create_task(name="E2E_B", description="E2E_B")["details"]["task_id"]  # type: ignore[index]

    # Tool that returns an ActiveQueue handle by executing A via numeric fast-path.
    @unify.traced
    async def start_queue_handle():
        return await ts.execute(text=str(a_id))

    # Start the async tool loop with a simple instruction to call our start tool.
    client = unify.AsyncUnify(
        os.getenv("UNIFY_MODEL", "gpt-5@openai"),
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
        reasoning_effort="high",
        service_tier="priority",
    )
    client.set_system_message(
        "Call `start_queue_handle` to start a task, then wait for further instructions.",
    )

    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"start_queue_handle": start_queue_handle},
        timeout=120,
        max_steps=30,
    )

    # Ensure the start tool has been requested so the dynamic helpers are exposed.
    await _wait_for_tool_request(client, "start_queue_handle")

    # Now ask the model to call the dynamic append_to_queue helper explicitly.
    await outer.interject(f"Now call append_to_queue(task_id={int(b_id)}).")

    # Wait until the assistant requests a helper whose name starts with 'append_to_queue_'.
    await _wait_for_assistant_call_prefix(client, "append_to_queue_")
    # And wait for the corresponding tool message to appear.
    await _wait_for_tool_message_prefix(client, "append_to_queue_")

    # Immediately verify the live queue now contains A followed by the newly appended B.
    # We assert right after the tool completed to avoid later lifecycle operations (e.g., defer)
    # altering the queue snapshot.
    live = ts._get_queue_for_task(task_id=a_id)
    ids = [getattr(r, "task_id", None) for r in (live or [])]
    assert ids and ids[0] == a_id and b_id in ids and ids.index(b_id) == len(ids) - 1

    # Instruct the model to stop the running queue so no background tasks linger,
    # then reply only with "done" for determinism.
    await outer.interject("Now call stop(cancel=false). Then reply only with: done")
    await _wait_for_assistant_call_prefix(client, "stop_")

    # Final assistant answer should be 'done' (case-insensitive) after stop; queue
    # membership was already asserted immediately after append.
    final = await outer.result()
    assert isinstance(final, str) and "done" in final.strip().lower()


# --------------------------------------------------------------------------- #
#  7. Clarification request for unknown id                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_requests_clarification_for_unknown_id(monkeypatch):
    """Supplying a numeric task_id that does *not* exist should trigger the
    internal `request_clarification` helper (i.e. push a question onto the
    clarification_up_q)."""

    actor = SimulatedActor(steps=1)
    ts = TaskScheduler(actor=actor)

    # Provide queues so the tool can ask for clarification.
    clarification_up_q: asyncio.Queue[str] = asyncio.Queue()
    clarification_down_q: asyncio.Queue[str] = asyncio.Queue()

    nonexistent_id = 424242  # arbitrary id that will not exist in a fresh context

    handle = await ts.execute(
        text=str(nonexistent_id),
        _clarification_up_q=clarification_up_q,
        _clarification_down_q=clarification_down_q,
    )

    # Wait for the assistant to push a clarification question.
    question = await clarification_up_q.get()

    assert question, "A clarification question should have been requested"

    # Respond so the loop can terminate quickly.
    await clarification_down_q.put(
        "Oh sorry, my mistake. Let's not execute any task in that case then.",
    )

    # Gracefully stop the loop – we're only interested in the clarification behaviour.
    await handle.result()


# --------------------------------------------------------------------------- #
#  A. Activation metadata                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_sets_activated_by_explicit():
    """Starting a task explicitly via execute should set activated_by='explicit'."""

    actor = SimulatedActor(steps=0)
    ts = TaskScheduler(actor=actor)

    # Seed a simple queued task
    name = "Simple queued task"
    task_id = ts._create_task(name=name, description=name)["details"]["task_id"]

    # Start by id (fast-path)
    handle = await ts.execute(text=str(task_id))
    await handle.result()

    # Verify activated_by on the activated instance (may already be completed)
    rows = ts._filter_tasks(filter=f"task_id == {task_id}")
    assert any(r.activated_by == ActivatedBy.explicit for r in rows)


@pytest.mark.asyncio
@_handle_project
async def test_update_status_cannot_force_active_and_does_not_set_activation_metadata():
    """Direct status updates cannot set 'active' and should not set 'activated_by'."""

    actor = SimulatedActor(steps=0)
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

    actor = SimulatedActor(steps=0)
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
        text=(
            f"Please run task {b} in isolation. Detach it entirely from the queue, "
            "do not keep any followers attached, and execute only this task now."
        ),
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
async def test_isolated_execute_start_at_to_second_when_head_moves(monkeypatch):
    """Branch A (head case): Explicit isolation – if activated task was head, next becomes head and inherits start_at."""

    actor = SimulatedActor(steps=0)
    ts = TaskScheduler(actor=actor)

    x, y = await _make_ordered_queue(ts, ["X", "Y"])  # type: ignore[misc]

    # Execute X (head) with an explicit isolation request (avoid pure-numeric fast path)
    handle = await ts.execute(
        text=(
            f"Please run task {x} in isolation. Detach it entirely from the queue, "
            "do not keep any followers attached, and execute only this task now."
        ),
    )
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

    handle = await ts.execute(text=str(b))

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
