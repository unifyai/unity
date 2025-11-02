from __future__ import annotations

import asyncio
import functools
import pytest

from unity.conductor.simulated import SimulatedConductor
from unity.actor.simulated import SimulatedActor, SimulatedActorHandle
from unity.task_scheduler.task_scheduler import TaskScheduler

from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_pause_actor_propagates_immediately_actor_path(monkeypatch):
    """
    Start an Actor-first Conductor.request, then call pause_actor and assert
    the underlying SimulatedActor receives pause without LLM delays.
    """

    # Signal when Actor.act is scheduled and when pause() is invoked on the actor handle
    scheduled_evt = asyncio.Event()
    paused_evt = asyncio.Event()

    # Wrap SimulatedActor.act to notify when the actor handle is created/scheduled
    _orig_act = SimulatedActor.act

    @functools.wraps(_orig_act)
    async def _wrapped_act(self: SimulatedActor, *a, **kw):
        handle = await _orig_act(self, *a, **kw)
        scheduled_evt.set()
        return handle

    monkeypatch.setattr(SimulatedActor, "act", _wrapped_act, raising=True)

    # Wrap SimulatedActorHandle.pause to flip an event immediately when called
    _orig_pause = SimulatedActorHandle.pause

    def _wrapped_pause(self: SimulatedActorHandle, *a, **kw):
        try:
            paused_evt.set()
        finally:
            return _orig_pause(self, *a, **kw)

    monkeypatch.setattr(SimulatedActorHandle, "pause", _wrapped_pause, raising=True)

    # Use a running simulated actor (no step auto-complete) so the session remains in-flight
    actor = SimulatedActor(steps=None, duration=20)
    cond = SimulatedConductor(actor=actor)

    handle = await cond.request(
        "Open a browser window so we can walk through the setup together.",
    )

    # Wait for the actor to be scheduled
    await asyncio.wait_for(scheduled_evt.wait(), timeout=30)
    # Give the outer loop a brief moment to adopt the nested handle
    await asyncio.sleep(0.2)

    # Invoke the new nested steer helper; assert the actor's pause is hit immediately
    result = await handle.pause_actor("test")

    # The pause must be observed without LLM turns (tight timeout)
    await asyncio.wait_for(paused_evt.wait(), timeout=1.0)

    # Sanity: the nested steer summary should record a pause application
    assert any(rec.get("method") == "pause" for rec in (result.get("applied") or []))

    handle.stop("done")
    await handle.result()


@pytest.mark.asyncio
@_handle_project
async def test_pause_actor_propagates_immediately_task_scheduler_path(monkeypatch):
    """
    Start a TaskScheduler.execute via Conductor.request, then call pause_actor and
    assert the underlying SimulatedActor receives pause immediately.
    """

    scheduled_evt = asyncio.Event()
    paused_evt = asyncio.Event()

    # Wrap SimulatedActor.act to signal when the actor session starts under the scheduler
    _orig_act = SimulatedActor.act

    @functools.wraps(_orig_act)
    async def _wrapped_act(self: SimulatedActor, *a, **kw):
        handle = await _orig_act(self, *a, **kw)
        scheduled_evt.set()
        return handle

    monkeypatch.setattr(SimulatedActor, "act", _wrapped_act, raising=True)

    # Wrap SimulatedActorHandle.pause to detect immediate propagation
    _orig_pause = SimulatedActorHandle.pause

    def _wrapped_pause(self: SimulatedActorHandle, *a, **kw):
        try:
            paused_evt.set()
        finally:
            return _orig_pause(self, *a, **kw)

    monkeypatch.setattr(SimulatedActorHandle, "pause", _wrapped_pause, raising=True)

    # Create a scheduler with the simulated actor and seed a runnable task
    actor = SimulatedActor(steps=None, duration=20)
    ts = TaskScheduler(actor=actor)
    name = "Prepare the monthly analytics dashboard"
    ts._create_task(name=name, description=name)

    cond = SimulatedConductor(task_scheduler=ts, actor=actor)

    handle = await cond.request(
        f"Run the task named '{name}' now.",
    )

    # Wait until the actor has started under the scheduler
    await asyncio.wait_for(scheduled_evt.wait(), timeout=120)
    # Briefly allow adoption of the nested handle in the outer loop
    await asyncio.sleep(0.4)

    result = await handle.pause_actor("maintenance")

    # Confirm the pause reached the actor quickly (no extra LLM steps)
    await asyncio.wait_for(paused_evt.wait(), timeout=1.0)

    # Sanity: the nested steer summary should include a pause application
    assert any(rec.get("method") == "pause" for rec in (result.get("applied") or []))

    handle.stop("done")
    await handle.result()


@pytest.mark.asyncio
@_handle_project
async def test_resume_actor_after_explicit_pause_actor_path(monkeypatch):
    """
    Actor-first path: explicitly pause via nested_steer (not pause_actor), then
    call resume_actor and assert SimulatedActor resumes immediately.
    """

    scheduled_evt = asyncio.Event()
    paused_evt = asyncio.Event()
    resumed_evt = asyncio.Event()

    _orig_act = SimulatedActor.act

    @functools.wraps(_orig_act)
    async def _wrapped_act(self: SimulatedActor, *a, **kw):
        handle = await _orig_act(self, *a, **kw)
        scheduled_evt.set()
        return handle

    monkeypatch.setattr(SimulatedActor, "act", _wrapped_act, raising=True)

    _orig_pause = SimulatedActorHandle.pause
    _orig_resume = SimulatedActorHandle.resume

    def _wrapped_pause(self: SimulatedActorHandle, *a, **kw):
        try:
            paused_evt.set()
        finally:
            return _orig_pause(self, *a, **kw)

    def _wrapped_resume(self: SimulatedActorHandle, *a, **kw):
        try:
            resumed_evt.set()
        finally:
            return _orig_resume(self, *a, **kw)

    monkeypatch.setattr(SimulatedActorHandle, "pause", _wrapped_pause, raising=True)
    monkeypatch.setattr(SimulatedActorHandle, "resume", _wrapped_resume, raising=True)

    actor = SimulatedActor(steps=None, duration=20)
    cond = SimulatedConductor(actor=actor)
    handle = await cond.request(
        "Open a browser window so we can walk through the setup together.",
    )

    await asyncio.wait_for(scheduled_evt.wait(), timeout=30)
    await asyncio.sleep(0.2)

    # Explicitly pause using nested_steer (independent of pause_actor)
    pause_spec = {
        "method": "interject",
        "args": "<Pausing actor for resume tests>",
        "children": {
            "TaskScheduler.execute": {"method": "pause"},
            "Actor.act": {"method": "pause"},
        },
    }
    await handle.nested_steer(pause_spec)

    await asyncio.wait_for(paused_evt.wait(), timeout=1.0)

    # Now resume via the high-level helper; should be immediate
    await handle.resume_actor("test-resume")
    await asyncio.wait_for(resumed_evt.wait(), timeout=1.0)

    handle.stop("done")
    await handle.result()


@pytest.mark.asyncio
@_handle_project
async def test_resume_actor_after_explicit_pause_task_scheduler_path(monkeypatch):
    """
    TaskScheduler path: explicitly pause via nested_steer (not pause_actor), then
    call resume_actor and assert SimulatedActor resumes immediately.
    """

    scheduled_evt = asyncio.Event()
    paused_evt = asyncio.Event()
    resumed_evt = asyncio.Event()

    _orig_act = SimulatedActor.act

    @functools.wraps(_orig_act)
    async def _wrapped_act(self: SimulatedActor, *a, **kw):
        handle = await _orig_act(self, *a, **kw)
        scheduled_evt.set()
        return handle

    monkeypatch.setattr(SimulatedActor, "act", _wrapped_act, raising=True)

    _orig_pause = SimulatedActorHandle.pause
    _orig_resume = SimulatedActorHandle.resume

    def _wrapped_pause(self: SimulatedActorHandle, *a, **kw):
        try:
            paused_evt.set()
        finally:
            return _orig_pause(self, *a, **kw)

    def _wrapped_resume(self: SimulatedActorHandle, *a, **kw):
        try:
            resumed_evt.set()
        finally:
            return _orig_resume(self, *a, **kw)

    monkeypatch.setattr(SimulatedActorHandle, "pause", _wrapped_pause, raising=True)
    monkeypatch.setattr(SimulatedActorHandle, "resume", _wrapped_resume, raising=True)

    actor = SimulatedActor(steps=None, duration=20)
    ts = TaskScheduler(actor=actor)
    name = "Prepare the monthly analytics dashboard"
    ts._create_task(name=name, description=name)

    cond = SimulatedConductor(task_scheduler=ts, actor=actor)
    handle = await cond.request(f"Run the task named '{name}' now.")

    await asyncio.wait_for(scheduled_evt.wait(), timeout=120)
    await asyncio.sleep(0.4)

    # Explicitly pause using nested_steer (independent of pause_actor)
    pause_spec = {
        "method": "interject",
        "args": "<Pausing actor for resume tests>",
        "children": {
            "TaskScheduler.execute": {"method": "pause"},
            "Actor.act": {"method": "pause"},
        },
    }
    await handle.nested_steer(pause_spec)

    await asyncio.wait_for(paused_evt.wait(), timeout=1.0)

    # Now resume via the high-level helper; should be immediate
    await handle.resume_actor("maintenance-resume")
    await asyncio.wait_for(resumed_evt.wait(), timeout=1.0)

    handle.stop("done")
    await handle.result()
