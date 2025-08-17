"""
Tests for `TaskScheduler.execute_task` which returns an `ActiveTask` handle.

These largely mirror *test_active_task.py* but go through the full
`TaskScheduler` surface so that we cover the integration layer that
retrieves the task from storage, wraps it in `ActiveTask`, and wires the
planner‐instance into the scheduler.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Dict, List
from datetime import datetime, timezone

import pytest

from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.planner.simulated import SimulatedPlanner, SimulatedActiveTask

#  The helper used in the existing test‑suite – applies project‑level monkey‐
#  patches (e.g. env vars, tracers) so we keep behaviour consistent.
from tests.helpers import _handle_project


# --------------------------------------------------------------------------- #
#  Test helpers                                                               #
# --------------------------------------------------------------------------- #


async def _make_scheduler_with_task(description: str, *, steps: int = 1):
    """Return *(scheduler, handle)* where *handle* is the active task."""
    planner = SimulatedPlanner(steps=steps)
    scheduler = TaskScheduler(planner=planner)

    task_id = scheduler._create_task(name=description, description=description)[
        "details"
    ]["task_id"]
    handle = await scheduler.execute_task(text=str(task_id))
    return scheduler, handle


async def _make_ordered_queue(ts: TaskScheduler, names: List[str]) -> List[int]:
    """Create tasks and order them head→tail, returning the task_ids.

    Also assigns a queue-level start_at on the head.
    """
    ids: List[int] = []
    for name in names:
        ids.append(ts._create_task(name=name, description=name)["details"]["task_id"])  # type: ignore[index]

    # Establish explicit order using the current queue snapshot as original
    original = [t.task_id for t in ts._get_task_queue()]
    ts._update_task_queue(original=original, new=ids)

    # Put a start_at timestamp on the head only
    ts._update_task_start_at(task_id=ids[0], new_start_at=datetime.now(timezone.utc))
    return ids


# --------------------------------------------------------------------------- #
#  0. Ask                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_task_ask(monkeypatch):
    """`ActiveTask.ask` should forward to the wrapped plan exactly once."""

    calls: Dict[str, int] = {"ask": 0}

    original_ask = SimulatedActiveTask.ask

    @functools.wraps(original_ask)
    async def spy_ask(self, question: str) -> str:  # type: ignore[override]
        calls["ask"] += 1
        return await original_ask(self, question)

    monkeypatch.setattr(SimulatedActiveTask, "ask", spy_ask, raising=True)

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
async def test_execute_task_interject(monkeypatch):
    """`ActiveTask.interject` should forward to the wrapped plan exactly once."""

    calls: Dict[str, int] = {"interject": 0}

    original_interject = SimulatedActiveTask.interject

    @functools.wraps(original_interject)
    async def spy_interject(self, instruction: str) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return await original_interject(self, instruction)

    monkeypatch.setattr(SimulatedActiveTask, "interject", spy_interject, raising=True)

    _scheduler, task = await _make_scheduler_with_task(
        "Investigate competitor pricing.",
        steps=2,
    )

    await task.interject("First gather public filings.")
    # Give the background thread one beat to process the step counter.
    await asyncio.sleep(0.2)
    # Gracefully stop to avoid leaking the background thread.
    task.stop()
    await task.result()

    assert calls["interject"] == 1, "interject must be called exactly once"


# --------------------------------------------------------------------------- #
#  2. Pause / Resume                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_task_pause_resume(monkeypatch):
    """The wrapper should transparently forward `pause` and `resume`."""

    counts: Dict[str, int] = {"pause": 0, "resume": 0}

    orig_pause = SimulatedActiveTask.pause
    orig_resume = SimulatedActiveTask.resume

    @functools.wraps(orig_pause)
    def spy_pause(self) -> str:  # type: ignore[override]
        counts["pause"] += 1
        return orig_pause(self)

    @functools.wraps(orig_resume)
    def spy_resume(self) -> str:  # type: ignore[override]
        counts["resume"] += 1
        return orig_resume(self)

    monkeypatch.setattr(SimulatedActiveTask, "pause", spy_pause, raising=True)
    monkeypatch.setattr(SimulatedActiveTask, "resume", spy_resume, raising=True)

    _scheduler, task = await _make_scheduler_with_task(
        "Run SEO audit for the website.",
        steps=2,
    )

    # Pause, wait a moment to ensure the thread blocks, then resume.
    task.pause()
    await asyncio.sleep(0.1)
    task.resume()
    # Stop the task to finish quickly and collect counts.
    task.stop()
    await task.result()

    assert counts == {"pause": 1, "resume": 1}, "pause/resume each called once"


# --------------------------------------------------------------------------- #
#  3. Stop                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_task_stop(monkeypatch):
    """Calling `ActiveTask.stop` should proxy to the plan and mark it done."""

    called = {"stop": 0}

    orig_stop = SimulatedActiveTask.stop

    @functools.wraps(orig_stop)
    def spy_stop(self) -> str:  # type: ignore[override]
        called["stop"] += 1
        return orig_stop(self)

    monkeypatch.setattr(SimulatedActiveTask, "stop", spy_stop, raising=True)

    _scheduler, task = await _make_scheduler_with_task(
        "Extract sentiment from reviews.",
        steps=5,
    )

    task.stop()
    result = await task.result()

    assert called["stop"] == 1, "stop must be invoked exactly once"
    assert "stopped task" in result.lower()
    assert task.done(), "`done()` should report True after stopping"


# --------------------------------------------------------------------------- #
#  4. Result & Done Lifecycle                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_task_result_and_done():
    """A normal workflow should complete once enough steps have been taken."""

    _scheduler, task = await _make_scheduler_with_task(
        "Compile coverage metrics.",
        steps=1,
    )

    # One interjection increments the internal step counter to fulfil `_steps`.
    await task.interject("Provide initial outline first.")
    result = await task.result()

    assert "completed task" in result.lower()
    assert task.done(), "`done()` must return True after natural completion"


# --------------------------------------------------------------------------- #
#  5. Free-form execute_task triggers internal ask                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_task_invokes_ask_when_id_missing(monkeypatch):
    """Executing via *description only* should call TaskScheduler.ask exactly once."""

    description = "prepare the monthly analytics dashboard."

    planner = SimulatedPlanner(steps=0)
    ts = TaskScheduler(planner=planner)

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
    handle = await ts.execute_task(text=description)

    # Wait for completion
    await handle.interject("please be quick")
    await handle.result()

    assert calls["ask"] == 1, "TaskScheduler.ask should be invoked exactly once"


# --------------------------------------------------------------------------- #
#  6. New task creation & execution                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_task_creates_new_task_and_executes(monkeypatch):
    """When the task clearly does not exist the scheduler should create it via
    `update` and then start it – `TaskScheduler.update` must therefore be
    invoked exactly once (or more, in very unlikely multi-step flows)."""

    description = "Organise annual security audit report."

    planner = SimulatedPlanner(steps=0)
    ts = TaskScheduler(planner=planner)

    # ---- spy on update -----------------------------------------------------
    calls: Dict[str, int] = {"update": 0}

    original_update = TaskScheduler.update

    @functools.wraps(original_update)
    async def spy_update(self, text: str, **kw):  # type: ignore[override]
        calls["update"] += 1
        return await original_update(self, text, **kw)

    monkeypatch.setattr(TaskScheduler, "update", spy_update, raising=True)

    # ---- execute (no prior task with that description exists) -------------
    handle = await ts.execute_task(text=description)

    # Get the final result.
    await handle.result()

    # ---- assertions --------------------------------------------------------
    assert calls["update"] >= 1, "Expected at least one call to TaskScheduler.update"

    # Verify that a task with the expected description now exists
    # Description may be normalised (e.g. trailing period removed).  Accept any
    # task whose *name* or *description* contains our original phrase without
    # the trailing period.
    created_tasks = ts._filter_tasks()
    phrase = description.rstrip(".")
    assert any(
        phrase in t.get("name", "") or phrase in t.get("description", "")
        for t in created_tasks
    ), "A new task with the provided description should have been created"


# --------------------------------------------------------------------------- #
#  7. Clarification request for unknown id                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_execute_task_requests_clarification_for_unknown_id(monkeypatch):
    """Supplying a numeric task_id that does *not* exist should trigger the
    internal `request_clarification` helper (i.e. push a question onto the
    clarification_up_q)."""

    planner = SimulatedPlanner(steps=0)
    ts = TaskScheduler(planner=planner)

    # Provide queues so the tool can ask for clarification.
    clarification_up_q: asyncio.Queue[str] = asyncio.Queue()
    clarification_down_q: asyncio.Queue[str] = asyncio.Queue()

    nonexistent_id = 424242  # arbitrary id that will not exist in a fresh context

    handle = await ts.execute_task(
        text=str(nonexistent_id),
        clarification_up_q=clarification_up_q,
        clarification_down_q=clarification_down_q,
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
async def test_execute_task_sets_activated_by_explicit():
    """Starting a task explicitly via execute_task should set activated_by='explicit'."""

    planner = SimulatedPlanner(steps=0)
    ts = TaskScheduler(planner=planner)

    # Seed a simple queued task
    name = "Simple queued task"
    task_id = ts._create_task(name=name, description=name)["details"]["task_id"]

    # Start by id (fast-path)
    handle = await ts.execute_task(text=str(task_id))
    await handle.result()

    # Verify activated_by on the activated instance (may already be completed)
    rows = ts._filter_tasks(filter=f"task_id == {task_id}")
    assert any(r.get("activated_by") == "explicit" for r in rows)


@pytest.mark.asyncio
@_handle_project
async def test_update_status_cannot_force_active_and_does_not_set_activation_metadata():
    """Direct status updates cannot set 'active' and should not set 'activated_by'."""

    planner = SimulatedPlanner(steps=0)
    ts = TaskScheduler(planner=planner)

    # Create a normal queued task
    label = "Cannot force active"
    task_id = ts._create_task(name=label, description=label)["details"]["task_id"]

    # Attempt to force 'active' via status update should fail
    with pytest.raises(ValueError):
        ts._update_task_status(task_ids=task_id, new_status="active")

    # Ensure no activation metadata exists prior to activation
    rows = ts._filter_tasks(filter=f"task_id == {task_id}")
    assert len(rows) == 1
    assert rows[0].get("activated_by") in (None, "")

    # Change a non-active status and ensure activated_by remains unset
    ts._update_task_status(task_ids=task_id, new_status="paused")
    rows2 = ts._filter_tasks(filter=f"task_id == {task_id}")
    assert rows2[0].get("status") == "paused"
    assert rows2[0].get("activated_by") in (None, "")


@pytest.mark.asyncio
@_handle_project
async def test_tasks_table_has_activated_by_column():
    """The Tasks context should include the activated_by column based on the Task model."""

    planner = SimulatedPlanner(steps=0)
    ts = TaskScheduler(planner=planner)

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
async def test_execute_task_isolate_detaches_entirely(monkeypatch):
    """Branch A: Detach the activated task entirely from the queue.

    Scenario: three tasks A->B->C, activate B explicitly with an 'isolate'-leaning
    prompt (ambiguous by default). Expect B detached, A->C linked, head's start_at
    preserved/propagated, and B's schedule cleared.
    """

    planner = SimulatedPlanner(steps=0)
    ts = TaskScheduler(planner=planner)

    # Build queue A->B->C with start_at on A
    a, b, c = await _make_ordered_queue(ts, ["A", "B", "C"])  # type: ignore[misc]

    # Execute B with an ambiguous request → defaults to isolate (A)
    handle = await ts.execute_task(text=str(b))
    await handle.result()

    rows_a = ts._filter_tasks(filter=f"task_id == {a}")
    rows_b = ts._filter_tasks(filter=f"task_id == {b}")
    rows_c = ts._filter_tasks(filter=f"task_id == {c}")

    # B must be detached entirely (no schedule)
    assert rows_b[0].get("schedule") in (None, {})

    # A should now link directly to C; C.prev_task should be A
    sched_a = rows_a[0].get("schedule") or {}
    sched_c = rows_c[0].get("schedule") or {}
    assert sched_a.get("next_task") == c
    assert sched_c.get("prev_task") == a

    # Only the head owns start_at → ensure C (non-head) does not inherit it unless it became head
    # Here A remains head, so C must not have start_at
    assert "start_at" not in sched_c or sched_c.get("start_at") in (None, "")


@pytest.mark.asyncio
@_handle_project
async def test_execute_task_isolate_when_head_moves_start_at_to_second(monkeypatch):
    """Branch A (head case): If activated task was head, next becomes head and inherits start_at."""

    planner = SimulatedPlanner(steps=0)
    ts = TaskScheduler(planner=planner)

    x, y = await _make_ordered_queue(ts, ["X", "Y"])  # type: ignore[misc]

    # Execute X (head) explicitly; ambiguous → isolate → detach X entirely
    handle = await ts.execute_task(text=str(x))
    await handle.result()

    rows_x = ts._filter_tasks(filter=f"task_id == {x}")
    rows_y = ts._filter_tasks(filter=f"task_id == {y}")

    # X detached
    assert rows_x[0].get("schedule") in (None, {})

    # Y becomes new head: prev_task=None and has start_at
    sched_y = rows_y[0].get("schedule") or {}
    assert sched_y.get("prev_task") is None
    assert "start_at" in sched_y and sched_y.get("start_at")


@pytest.mark.asyncio
@_handle_project
async def test_execute_task_chain_keeps_followers(monkeypatch):
    """Branch B: Keep tasks behind still queued to follow the activated task.

    We simulate an unambiguous 'chain' request by monkeypatching the internal
    classifier to return 'chain'. Expect B to become sub-head (prev=None), keep
    next pointer to C, and C.prev_task=B, with only the head owning start_at.
    """

    planner = SimulatedPlanner(steps=0)
    ts = TaskScheduler(planner=planner)

    a, b, c = await _make_ordered_queue(ts, ["A2", "B2", "C2"])  # type: ignore[misc]

    async def force_chain(*, request_text: str, parent_chat_context=None):  # type: ignore[override]
        return "chain"

    monkeypatch.setattr(ts, "_decide_execution_scope", force_chain, raising=True)

    handle = await ts.execute_task(text=str(b))
    await handle.result()

    rows_b = ts._filter_tasks(filter=f"task_id == {b}")
    rows_c = ts._filter_tasks(filter=f"task_id == {c}")

    sched_b = rows_b[0].get("schedule") or {}
    sched_c = rows_c[0].get("schedule") or {}

    # B becomes sub-head of its chain
    assert sched_b.get("prev_task") is None
    assert sched_b.get("next_task") == c
    # C follows B
    assert sched_c.get("prev_task") == b
    # C must not carry start_at (non-head)
    assert "start_at" not in sched_c or sched_c.get("start_at") in (None, "")
