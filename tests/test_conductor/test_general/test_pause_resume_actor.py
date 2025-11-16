from __future__ import annotations

import asyncio
import functools
import pytest

from unity.conductor.simulated import SimulatedConductor
from unity.actor.simulated import SimulatedActor, SimulatedActorHandle
from unity.task_scheduler.task_scheduler import TaskScheduler

from tests.helpers import _handle_project
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.types import RawImageRef, AnnotatedImageRef
from pathlib import Path
import base64


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
    interjected_evt = asyncio.Event()
    captured_msgs: list[str] = []

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

    # Wrap SimulatedActorHandle.interject to capture the child interjection message
    _orig_interject = SimulatedActorHandle.interject

    async def _wrapped_interject(self: SimulatedActorHandle, instruction: str):
        try:
            captured_msgs.append(instruction)
            interjected_evt.set()
        finally:
            return await _orig_interject(self, instruction)

    monkeypatch.setattr(
        SimulatedActorHandle,
        "interject",
        _wrapped_interject,
        raising=True,
    )

    # Keep the actor alive for the whole test; complete only on explicit stop
    actor = SimulatedActor(steps=None, duration=None)
    cond = SimulatedConductor(actor=actor)

    handle = await cond.request(
        "Open a browser window so we can walk through the setup together.",
    )

    # Wait for the actor to be scheduled
    await asyncio.wait_for(scheduled_evt.wait(), timeout=300)
    # Give the outer loop a brief moment to adopt the nested handle
    await asyncio.sleep(0.2)

    # Invoke the new nested steer helper; assert the actor's pause is hit immediately
    result = await cond.pause_actor("test")

    # The pause must be observed without LLM turns (tight timeout)
    await asyncio.wait_for(paused_evt.wait(), timeout=1.0)

    # The child interjection should arrive promptly after pausing
    await asyncio.wait_for(interjected_evt.wait(), timeout=5.0)

    # Verify child-level interjection content
    assert any(
        "execution was paused due to test" in m for m in captured_msgs
    ), "Expected child interjection after pause"

    # Sanity: the nested steer summary should record a pause application
    assert any(rec.get("method") == "pause" for rec in (result.get("applied") or []))
    # Interjection should be applied when pause actually applied to a child
    assert any(
        rec.get("method") == "interject" for rec in (result.get("applied") or [])
    )

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
    interjected_evt = asyncio.Event()
    captured_msgs: list[str] = []

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

    # Capture interjections forwarded via ActiveTask → Actor path
    _orig_interject = SimulatedActorHandle.interject

    async def _wrapped_interject(self: SimulatedActorHandle, instruction: str):
        try:
            captured_msgs.append(instruction)
            interjected_evt.set()
        finally:
            return await _orig_interject(self, instruction)

    monkeypatch.setattr(
        SimulatedActorHandle,
        "interject",
        _wrapped_interject,
        raising=True,
    )

    # Create a scheduler with the simulated actor and seed a runnable task
    actor = SimulatedActor(steps=None, duration=None)
    ts = TaskScheduler(actor=actor)
    name = "Prepare the monthly analytics dashboard"
    ts._create_task(name=name, description=name)

    cond = SimulatedConductor(task_scheduler=ts, actor=actor)

    handle = await cond.request(
        f"Run the task named '{name}' now.",
    )

    # Wait until the actor has started under the scheduler
    await asyncio.wait_for(scheduled_evt.wait(), timeout=300)
    # Briefly allow adoption of the nested handle in the outer loop
    await asyncio.sleep(0.4)

    result = await cond.pause_actor("maintenance")

    # Confirm the pause reached the actor quickly (no extra LLM steps)
    await asyncio.wait_for(paused_evt.wait(), timeout=1.0)

    # Confirm the child interjection arrived shortly after
    await asyncio.wait_for(interjected_evt.wait(), timeout=5.0)

    # Verify child-level interjection content
    assert any(
        "execution was paused due to maintenance" in m for m in captured_msgs
    ), "Expected child interjection after pause (scheduler path)"

    # Sanity: the nested steer summary should include a pause application
    assert any(rec.get("method") == "pause" for rec in (result.get("applied") or []))
    # Interjection should be applied when pause actually applied to a child
    assert any(
        rec.get("method") == "interject" for rec in (result.get("applied") or [])
    )

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
    interjected_evt = asyncio.Event()
    captured_msgs: list[str] = []
    timestamps: dict[str, float] = {}

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
            timestamps["resume"] = __import__("time").monotonic()
            resumed_evt.set()
        finally:
            return _orig_resume(self, *a, **kw)

    monkeypatch.setattr(SimulatedActorHandle, "pause", _wrapped_pause, raising=True)
    monkeypatch.setattr(SimulatedActorHandle, "resume", _wrapped_resume, raising=True)

    # Capture child interjection and timestamp it
    _orig_interject = SimulatedActorHandle.interject

    async def _wrapped_interject(self: SimulatedActorHandle, instruction: str):
        try:
            captured_msgs.append(instruction)
            timestamps["interject"] = __import__("time").monotonic()
            interjected_evt.set()
        finally:
            return await _orig_interject(self, instruction)

    monkeypatch.setattr(
        SimulatedActorHandle,
        "interject",
        _wrapped_interject,
        raising=True,
    )

    actor = SimulatedActor(steps=None, duration=None)
    cond = SimulatedConductor(actor=actor)
    handle = await cond.request(
        "Open a browser window so we can walk through the setup together.",
    )

    await asyncio.wait_for(scheduled_evt.wait(), timeout=300)
    await asyncio.sleep(0.2)

    # Explicitly pause using nested_steer (independent of pause_actor)
    pause_spec = {
        "steps": [{"method": "interject", "args": "<Pausing actor for resume tests>"}],
        "children": [
            {"handle": "ActiveQueue", "steps": [{"method": "pause"}]},
            {"handle": "ActiveTask", "steps": [{"method": "pause"}]},
            {"handle": "ActorHandle", "steps": [{"method": "pause"}]},
        ],
    }
    await handle.nested_steer(pause_spec)

    await asyncio.wait_for(paused_evt.wait(), timeout=1.0)

    # Now resume via the high-level helper; should be immediate
    result = await cond.resume_actor("test-resume")
    # Interjection to child should occur before resume
    await asyncio.wait_for(interjected_evt.wait(), timeout=5.0)
    await asyncio.wait_for(resumed_evt.wait(), timeout=1.0)

    # Interjection should be applied when resume actually applied to a child
    assert any(
        rec.get("method") == "interject" for rec in (result.get("applied") or [])
    )

    # Verify child-level interjection content and ordering (interject before resume)
    assert any(
        "execution was resumed due to test-resume" in m for m in captured_msgs
    ), "Expected child interjection before resume"
    if "interject" in timestamps and "resume" in timestamps:
        assert (
            timestamps["interject"] <= timestamps["resume"]
        ), "Child interjection should precede resume"

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
    interjected_evt = asyncio.Event()
    captured_msgs: list[str] = []
    timestamps: dict[str, float] = {}

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
            timestamps["resume"] = __import__("time").monotonic()
            resumed_evt.set()
        finally:
            return _orig_resume(self, *a, **kw)

    monkeypatch.setattr(SimulatedActorHandle, "pause", _wrapped_pause, raising=True)
    monkeypatch.setattr(SimulatedActorHandle, "resume", _wrapped_resume, raising=True)

    # Capture child interjection and timestamp it (scheduler path)
    _orig_interject = SimulatedActorHandle.interject

    async def _wrapped_interject(self: SimulatedActorHandle, instruction: str):
        try:
            captured_msgs.append(instruction)
            timestamps["interject"] = __import__("time").monotonic()
            interjected_evt.set()
        finally:
            return await _orig_interject(self, instruction)

    monkeypatch.setattr(
        SimulatedActorHandle,
        "interject",
        _wrapped_interject,
        raising=True,
    )

    actor = SimulatedActor(steps=None, duration=None)
    ts = TaskScheduler(actor=actor)
    name = "Prepare the monthly analytics dashboard"
    ts._create_task(name=name, description=name)

    cond = SimulatedConductor(task_scheduler=ts, actor=actor)
    handle = await cond.request(f"Run the task named '{name}' now.")

    await asyncio.wait_for(scheduled_evt.wait(), timeout=300)
    await asyncio.sleep(0.4)

    # Explicitly pause using nested_steer (independent of pause_actor)
    pause_spec = {
        "steps": [{"method": "interject", "args": "<Pausing actor for resume tests>"}],
        "children": [
            {"handle": "ActiveQueue", "steps": [{"method": "pause"}]},
            {"handle": "ActiveTask", "steps": [{"method": "pause"}]},
            {"handle": "ActorHandle", "steps": [{"method": "pause"}]},
        ],
    }
    await handle.nested_steer(pause_spec)

    await asyncio.wait_for(paused_evt.wait(), timeout=1.0)

    # Now resume via the high-level helper; should be immediate
    result = await cond.resume_actor("maintenance-resume")
    await asyncio.wait_for(interjected_evt.wait(), timeout=5.0)
    await asyncio.wait_for(resumed_evt.wait(), timeout=1.0)

    # Interjection should be applied when resume actually applied to a child
    assert any(
        rec.get("method") == "interject" for rec in (result.get("applied") or [])
    )

    # Verify child-level interjection content and ordering (interject before resume)
    assert any(
        "execution was resumed due to maintenance-resume" in m for m in captured_msgs
    ), "Expected child interjection before resume (scheduler path)"
    if "interject" in timestamps and "resume" in timestamps:
        assert (
            timestamps["interject"] <= timestamps["resume"]
        ), "Child interjection should precede resume (scheduler path)"

    handle.stop("done")
    await handle.result()


@pytest.mark.asyncio
@_handle_project
async def test_pause_actor_no_interjection_for_read_only_contact_query():
    """
    When the Conductor handles a read-only flow (e.g., ContactManager.ask), calling
    pause_actor should NOT emit the conductor interjection since no pause applies.
    """

    cond = SimulatedConductor()

    # Ask a clearly read-only, contacts question to route to ContactManager.ask
    handle = await cond.request(
        "How many contacts are stored in my address book? Keep it brief.",
    )

    # Invoke pause_actor – since no Actor/TaskScheduler execution is in flight,
    # the conditional interjection should not occur.
    result = await cond.pause_actor("read-only")

    # Ensure no interjection was applied at the root level
    assert not any(
        rec.get("method") == "interject" for rec in (result.get("applied") or [])
    ), "Interjection should not be emitted for read-only ContactManager.ask flows"

    handle.stop("done")
    await handle.result()


@pytest.mark.asyncio
@_handle_project
async def test_resume_actor_image_guides_simulation_to_spreadsheet_actor_path(
    monkeypatch,
):
    """
    Actor-first path: resume with an annotated image and then ask the inner
    SimulatedActor about the file – reply should mention a sheet/spreadsheet.
    """

    # Prepare image id from the existing rota screenshot
    img_path = (
        Path(__file__).parent.parent.parent
        / "test_task_scheduler"
        / "organize_weekly_rotar.png"
    )
    raw_bytes = img_path.read_bytes()
    img_b64 = base64.b64encode(raw_bytes).decode("utf-8")

    im = ImageManager()
    [img_id] = im.add_images(
        [
            {"caption": "weekly rota", "data": img_b64},
        ],
    )

    scheduled_evt = asyncio.Event()
    paused_evt = asyncio.Event()
    interjected_evt = asyncio.Event()

    # Signal when Actor.act is scheduled
    _orig_act = SimulatedActor.act

    @functools.wraps(_orig_act)
    async def _wrapped_act(self: SimulatedActor, *a, **kw):
        handle = await _orig_act(self, *a, **kw)
        scheduled_evt.set()
        return handle

    monkeypatch.setattr(SimulatedActor, "act", _wrapped_act, raising=True)

    # Detect pause and interjection on the inner actor handle
    _orig_pause = SimulatedActorHandle.pause
    _orig_interject = SimulatedActorHandle.interject

    def _wrapped_pause(self: SimulatedActorHandle, *a, **kw):
        try:
            paused_evt.set()
        finally:
            return _orig_pause(self, *a, **kw)

    async def _wrapped_interject(
        self: SimulatedActorHandle,
        instruction: str,
        *,
        images=None,
    ):
        try:
            interjected_evt.set()
        finally:
            return await _orig_interject(self, instruction, images=images)

    monkeypatch.setattr(SimulatedActorHandle, "pause", _wrapped_pause, raising=True)
    monkeypatch.setattr(
        SimulatedActorHandle,
        "interject",
        _wrapped_interject,
        raising=True,
    )

    # Keep the actor alive for the whole test; complete only on explicit stop
    actor = SimulatedActor(steps=None, duration=None)
    cond = SimulatedConductor(actor=actor)

    handle = await cond.request(
        "Open a browser window so we can walk through the setup together.",
    )

    # Wait for actor to start and adopt the nested handle
    await asyncio.wait_for(scheduled_evt.wait(), timeout=300)
    await asyncio.sleep(0.2)

    # Explicitly pause the actor via nested steer to match the resume scenario
    pause_spec = {
        "children": {
            "Actor.act": {"steps": [{"method": "pause"}]},
        },
    }
    await handle.nested_steer(pause_spec)  # type: ignore[attr-defined]
    await asyncio.wait_for(paused_evt.wait(), timeout=5.0)

    # Now resume with an annotated image pointing to the rota spreadsheet
    await cond.resume_actor(
        "visual update",
        images=[
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=int(img_id)),
                annotation="this is the file you need to edit",
            ),
        ],
    )

    # Ensure the child interjection (with image) was delivered
    await asyncio.wait_for(interjected_evt.wait(), timeout=5.0)

    # Locate the inner SimulatedActor handle and ask about the file type
    inner_handle = None
    try:
        ti = getattr(handle._task, "task_info", {})  # type: ignore[attr-defined]
        if isinstance(ti, dict):
            for _t, meta in ti.items():
                h = getattr(meta, "handle", None)
                if isinstance(h, SimulatedActorHandle):
                    inner_handle = h
                    break
    except Exception:
        inner_handle = None

    assert inner_handle is not None, "Expected inner SimulatedActorHandle to be adopted"

    resp = await inner_handle.ask(
        "What type of file is shown in the screenshot we just sent during resume? Answer briefly.",
    )
    assert isinstance(resp, str) and resp.strip()
    assert "sheet" in resp.lower(), f"Expected 'sheet' mention in: {resp!r}"

    handle.stop("done")
    await handle.result()
