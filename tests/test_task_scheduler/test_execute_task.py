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
from typing import Dict

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

    description = "Prepare monthly analytics dashboard."

    planner = SimulatedPlanner(steps=1)
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

    planner = SimulatedPlanner(steps=1)
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

    # Speed the simulated planner along so the outer loop can finish quickly.
    await handle.interject(
        "please proceed swiftly, but only create a new task if it is necessary",
    )
    await handle.result()

    # ---- assertions --------------------------------------------------------
    assert calls["update"] >= 1, "Expected at least one call to TaskScheduler.update"

    # Verify that a task with the expected description now exists
    # Description may be normalised (e.g. trailing period removed).  Accept any
    # task whose *name* or *description* contains our original phrase without
    # the trailing period.
    created_tasks = ts._search_tasks()
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

    planner = SimulatedPlanner(steps=1)
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
    handle.stop()
