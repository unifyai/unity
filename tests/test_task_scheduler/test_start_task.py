"""
Tests for `TaskScheduler.start_task` which returns an `ActiveTask` handle.

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
from unity.planner.simulated import SimulatedPlanner, SimulatedPlan

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

    task_id = scheduler._create_task(name=description, description=description)
    handle = await scheduler.start_task(task_id=task_id)
    return scheduler, handle


# --------------------------------------------------------------------------- #
#  0. Ask                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_start_task_ask(monkeypatch):
    """`ActiveTask.ask` should forward to the wrapped plan exactly once."""

    calls: Dict[str, int] = {"ask": 0}

    original_ask = SimulatedPlan.ask

    @functools.wraps(original_ask)
    async def spy_ask(self, question: str) -> str:  # type: ignore[override]
        calls["ask"] += 1
        return await original_ask(self, question)

    monkeypatch.setattr(SimulatedPlan, "ask", spy_ask, raising=True)

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
async def test_start_task_interject(monkeypatch):
    """`ActiveTask.interject` should forward to the wrapped plan exactly once."""

    calls: Dict[str, int] = {"interject": 0}

    original_interject = SimulatedPlan.interject

    @functools.wraps(original_interject)
    def spy_interject(self, instruction: str) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return original_interject(self, instruction)

    monkeypatch.setattr(SimulatedPlan, "interject", spy_interject, raising=True)

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
async def test_start_task_pause_resume(monkeypatch):
    """The wrapper should transparently forward `pause` and `resume`."""

    counts: Dict[str, int] = {"pause": 0, "resume": 0}

    orig_pause = SimulatedPlan.pause
    orig_resume = SimulatedPlan.resume

    @functools.wraps(orig_pause)
    def spy_pause(self) -> str:  # type: ignore[override]
        counts["pause"] += 1
        return orig_pause(self)

    @functools.wraps(orig_resume)
    def spy_resume(self) -> str:  # type: ignore[override]
        counts["resume"] += 1
        return orig_resume(self)

    monkeypatch.setattr(SimulatedPlan, "pause", spy_pause, raising=True)
    monkeypatch.setattr(SimulatedPlan, "resume", spy_resume, raising=True)

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
async def test_start_task_stop(monkeypatch):
    """Calling `ActiveTask.stop` should proxy to the plan and mark it done."""

    called = {"stop": 0}

    orig_stop = SimulatedPlan.stop

    @functools.wraps(orig_stop)
    def spy_stop(self) -> str:  # type: ignore[override]
        called["stop"] += 1
        return orig_stop(self)

    monkeypatch.setattr(SimulatedPlan, "stop", spy_stop, raising=True)

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
async def test_start_task_result_and_done():
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
