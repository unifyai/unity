"""
Tests for the thin `ActiveTask` → `BasePlan` wrapper.

The structure mirrors *test_simulated_actor.py* but talks directly to the
`ActiveTask` handle instead of orchestrating the outer tool–use loop.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Dict

import pytest

from unity.task_scheduler.active_task import ActiveTask
from unity.actor.simulated import SimulatedActor
from unity.actor.simulated import SimulatedActorHandle

#  The helper used in the existing test-suite – applies project-level monkey-
#  patches (e.g. env vars, tracers) so we keep behaviour consistent.
from tests.helpers import _handle_project


# --------------------------------------------------------------------------- #
#  0. Ask                                                                   //
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_active_task_ask(monkeypatch):
    """
    `ActiveTask.ask` should forward to the wrapped plan exactly once.
    """
    actor = SimulatedActor(steps=1)
    calls: Dict[str, int] = {"ask": 0}

    original_ask = SimulatedActorHandle.ask

    @functools.wraps(original_ask)
    async def spy_ask(self, question: str) -> str:  # type: ignore[override]
        calls["ask"] += 1
        return await original_ask(self, question)

    monkeypatch.setattr(SimulatedActorHandle, "ask", spy_ask, raising=True)

    task = await ActiveTask.create(
        actor,
        task_description="Analyse new product launch performance.",
    )

    # Trigger a single ask call that should propagate to the active task.
    await task.ask("Do we have any early metrics?")
    # Give the background worker a beat and await completion.
    await asyncio.sleep(0.2)
    await task.result()

    assert calls["ask"] == 1, "ask must be called exactly once"


# --------------------------------------------------------------------------- #
#  1. Interjection                                                          //
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_active_task_interject(monkeypatch):
    """
    `ActiveTask.interject` should forward to the wrapped plan exactly once.
    """
    actor = SimulatedActor(steps=2)
    calls: Dict[str, int] = {"interject": 0}

    original_interject = SimulatedActorHandle.interject

    @functools.wraps(original_interject)
    async def spy_interject(self, instruction: str) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return await original_interject(self, instruction)

    monkeypatch.setattr(SimulatedActorHandle, "interject", spy_interject, raising=True)

    task = await ActiveTask.create(
        actor,
        task_description="Investigate competitor pricing.",
    )

    await task.interject("First gather public filings.")
    # Give the background thread one beat to process the step counter.
    await asyncio.sleep(0.2)
    # Gracefully stop to avoid leaking the background thread.
    task.stop()
    await task.result()

    assert calls["interject"] == 1, "interject must be called exactly once"


# --------------------------------------------------------------------------- #
#  2. Pause / Resume                                                        //
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_active_task_pause_resume(monkeypatch):
    """
    The wrapper should transparently forward `pause` and `resume`.
    """
    actor = SimulatedActor(steps=2)
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

    task = await ActiveTask.create(
        actor,
        task_description="Run SEO audit for the website.",
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
#  3. Stop                                                                  //
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_active_task_stop(monkeypatch):
    """
    Calling `ActiveTask.stop` should proxy to the plan and mark it done.
    """
    actor = SimulatedActor(steps=5)  # value doesn't matter, we stop early
    called = {"stop": 0}

    orig_stop = SimulatedActorHandle.stop

    @functools.wraps(orig_stop)
    def spy_stop(self, reason: str | None = None) -> str:  # type: ignore[override]
        called["stop"] += 1
        return orig_stop(self, reason=reason)

    monkeypatch.setattr(SimulatedActorHandle, "stop", spy_stop, raising=True)

    task = await ActiveTask.create(
        actor,
        task_description="Extract sentiment from reviews.",
    )
    task.stop()
    result = await task.result()

    assert called["stop"] == 1, "stop must be invoked exactly once"
    assert "stopped" in result.lower()
    assert task.done(), "`done()` should report True after stopping"


# --------------------------------------------------------------------------- #
#  4. Result & Done Lifecycle                                               //
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_active_task_result_and_done():
    """
    A normal workflow should complete once enough steps have been taken.
    """
    actor = SimulatedActor(steps=1)  # finishes after a single steering op
    task = await ActiveTask.create(
        actor,
        task_description="Compile coverage metrics.",
    )

    # One interjection increments the internal step counter to fulfil `_steps`.
    await task.interject("Provide initial outline first.")
    result = await task.result()

    assert "completed" in result.lower()
    assert task.done(), "`done()` must return True after natural completion"
