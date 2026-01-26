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
from unity.function_manager.function_manager import FunctionManager
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.types import RawImageRef, AnnotatedImageRef
from pathlib import Path
import base64

#  The helper used in the existing test-suite – applies project-level monkey-
#  patches (e.g. env vars, tracers) so we keep behaviour consistent.
from tests.helpers import _handle_project

# --------------------------------------------------------------------------- #
#  0. Ask                                                                   //
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_ask(monkeypatch):
    """
    `ActiveTask.ask` should forward to the wrapped plan exactly once.
    """
    actor = SimulatedActor(steps=None, duration=None)
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
    # Give the background worker a beat and then stop explicitly.
    await asyncio.sleep(0.2)
    task.stop(cancel=False)
    await task.result()

    assert calls["ask"] == 1, "ask must be called exactly once"


# --------------------------------------------------------------------------- #
#  1. Interjection                                                          //
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_interject(monkeypatch):
    """
    `ActiveTask.interject` should forward to the wrapped plan exactly once.
    """
    actor = SimulatedActor(steps=None, duration=None)
    calls: Dict[str, int] = {"interject": 0}

    original_interject = SimulatedActorHandle.interject

    @functools.wraps(original_interject)
    async def spy_interject(self, instruction: str, *, images=None) -> None:  # type: ignore[override]
        calls["interject"] += 1
        return await original_interject(self, instruction, images=images)

    monkeypatch.setattr(SimulatedActorHandle, "interject", spy_interject, raising=True)

    task = await ActiveTask.create(
        actor,
        task_description="Investigate competitor pricing.",
    )

    await task.interject("First gather public filings.")
    # Give the background thread one beat to process the step counter.
    await asyncio.sleep(0.2)
    # Gracefully stop to avoid leaking the background thread.
    task.stop(cancel=False)
    await task.result()

    assert calls["interject"] == 1, "interject must be called exactly once"


# --------------------------------------------------------------------------- #
#  2. Pause / Resume                                                        //
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_pause_resume(monkeypatch):
    """
    The wrapper should transparently forward `pause` and `resume`.
    """
    actor = SimulatedActor(steps=None, duration=None)
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
    await task.pause()
    await asyncio.sleep(0.1)
    await task.resume()
    # Stop the task to finish quickly and collect counts.
    task.stop(cancel=False)
    await task.result()

    assert counts == {"pause": 1, "resume": 1}, "pause/resume each called once"


# --------------------------------------------------------------------------- #
#  3. Stop                                                                  //
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_stop(monkeypatch):
    """
    Calling `ActiveTask.stop` should proxy to the plan and mark it done.
    """
    actor = SimulatedActor(
        steps=None,
        duration=None,
    )  # value doesn't matter, we stop early
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
    task.stop(cancel=False)
    result = await task.result()

    assert called["stop"] == 1, "stop must be invoked exactly once"
    assert "stopped" in result.lower()
    assert task.done(), "`done()` should report True after stopping"


# --------------------------------------------------------------------------- #
#  4. Result & Done Lifecycle                                               //
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_result_and_done():
    """
    A normal workflow should complete once enough steps have been taken.
    """
    actor = SimulatedActor(steps=None, duration=None)
    task = await ActiveTask.create(
        actor,
        task_description="Compile coverage metrics.",
    )

    # Perform one interjection for activity, then stop explicitly
    await task.interject("Provide initial outline first.")
    task.stop(cancel=False)
    result = await task.result()

    assert "stopped" in result.lower()
    assert task.done(), "`done()` must return True after explicit stop"


# --------------------------------------------------------------------------- #
#  5. Interject implying stop/defer should not mark as completed            //
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_interject_implies_defer_and_reinstate(monkeypatch):
    """
    An interjection like "let's stop this and do it later" should trigger a
    graceful stop and reintegration, resulting in a "stopped" outcome rather
    than "completed". Pre-patch, this would incorrectly complete.
    """

    class _FakeScheduler:
        def __init__(self):
            self.reinstate_called = False
            self.fallback_called = False

        async def _classify_steering_intent(
            self,
            message: str,
            _parent_chat_context=None,
        ):
            # Classify this interjection as a defer request
            return ("defer", message)

        def _reinstate_task_to_previous_queue(self, *, task_id: int):  # type: ignore[no-redef]
            self.reinstate_called = True
            return {"outcome": "ok", "details": {"task_id": task_id}}

        def _maybe_reinstate_after_stop(self, *, task_id: int, reason: str | None):
            self.fallback_called = True

    actor = SimulatedActor(steps=5)
    fake_sched = _FakeScheduler()

    task = await ActiveTask.create(
        actor,
        task_description="Draft release notes",
        scheduler=fake_sched,  # type: ignore[arg-type]
        task_id=101,
        instance_id=1,
    )

    # Interjection that implies stop & do later (defer)
    await task.interject("Let's stop this task and do it later.")
    result = await task.result()

    # Should not be marked as completed; should read as stopped
    assert "stopped" in result.lower()
    assert "completed" not in result.lower()

    # Reintegration should run via the primary path (not fallback)
    assert fake_sched.reinstate_called is True
    assert fake_sched.fallback_called is False


# --------------------------------------------------------------------------- #
#  6. Entrypoint observes FunctionManager doc via ActiveTask.ask              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_entrypoint_demonstrates_function_knowledge_during_ask():
    """
    Mirror the SimulatedActor entrypoint test, but run through ActiveTask to
    ensure the scheduler wrapper forwards the entrypoint so the actor observes
    the function metadata during ask().
    """

    fm = FunctionManager()

    impl = '''
def simulate_linkedin_sales_leads() -> str:
    """Simulated browser flow:
    1) Trouble logging into LinkedIn (login blocked initially).
    2) Issue resolved; proceed to search sales leads on LinkedIn."""
    print("Trouble logging into LinkedIn: login blocked")
    print("Issue resolved: Login successful; searching sales leads on LinkedIn")
    return "ok"
'''.strip()

    res = fm.add_functions(implementations=impl)
    status = res.get("simulate_linkedin_sales_leads", "")
    assert any(s in str(status) for s in ("added", "updated", "skipped"))

    fid = (
        fm.list_functions().get("simulate_linkedin_sales_leads", {}).get("function_id")
    )
    assert isinstance(fid, int)

    actor = SimulatedActor(steps=None, duration=None)
    task = await ActiveTask.create(
        actor,
        task_description="Search sales leads.",
        entrypoint=fid,
    )

    ask_handle = await task.ask(
        "Did you or are you encountering any problems logging in? Reply briefly, explaining any relevant websites.",
    )
    reply = await ask_handle.result()
    assert isinstance(reply, str) and reply.strip(), "Expected a non-empty reply"
    assert "linkedin" in reply.lower(), f"Expected LinkedIn mention in: {reply!r}"

    # Ensure clean shutdown to avoid relying on natural step completion
    task.stop(cancel=False)
    await task.result()


# --------------------------------------------------------------------------- #
#  7. Interject with image → simulation recognises spreadsheet                //
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_interject_image_guides_simulation_to_spreadsheet(
    monkeypatch,
):
    """
    Start an ActiveTask (wrapping SimulatedActor), interject with a screenshot (Google Sheets),
    then ask about progress; the reply should reference a sheet/spreadsheet.
    Mirrors the SimulatedActor handle test but via the ActiveTask wrapper.
    """

    # Store the screenshot and obtain an image id
    img_path = (
        Path(__file__).parent.parent
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

    actor = SimulatedActor(steps=None, duration=None)
    task = await ActiveTask.create(
        actor,
        task_description=(
            "We'll start working on organizing the rota for the admin assistants."
        ),
    )

    # Interject with the image attached; annotation intentionally does not say "spreadsheet"
    await task.interject(
        "Please start working on this file.",
        images=[
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=int(img_id)),
                annotation="rota file",
            ),
        ],
    )

    # Ask about status and infer file type from the visual context
    ask_handle = await task.ask(
        "How is it going? What file are you working on? What file type is it?",
    )
    reply = await ask_handle.result()
    assert isinstance(reply, str) and reply.strip()
    assert "sheet" in reply.lower(), f"Expected 'sheet' mention in: {reply!r}"

    # Ensure clean shutdown to avoid relying on natural step completion
    task.stop(cancel=False)
    await task.result()
