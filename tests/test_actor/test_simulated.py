import asyncio
import functools
import pytest

from unity.actor.simulated import SimulatedActor, SimulatedActorHandle
from pydantic import BaseModel, Field
from typing import List
from tests.helpers import (
    _handle_project,
    _assert_blocks_while_paused,
    DEFAULT_TIMEOUT,
)
from unity.function_manager.function_manager import FunctionManager
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.types import RawImageRef, AnnotatedImageRef
from pathlib import Path
import base64


# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic start-and-act                                                     #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_act():
    actor = SimulatedActor(duration=0.1)
    handle = await actor.act("Perform a quick demo.")
    result = await handle.result()
    assert isinstance(result, str) and result.strip(), "Result should be non-empty"


class ActionResult(BaseModel):
    """Structured result from an actor action."""

    completed: bool = Field(..., description="Whether the action completed")
    steps_taken: List[str] = Field(
        default_factory=list,
        description="List of steps taken",
    )
    outcome: str = Field(..., description="Description of the outcome")


@pytest.mark.asyncio
@_handle_project
async def test_simulated_act_response_format():
    """Simulated Actor.act should return structured output when response_format is provided."""
    actor = SimulatedActor(steps=2, duration=1)

    handle = await actor.act(
        "Perform a quick demo task and report results",
        response_format=ActionResult,
    )
    result = await handle.result()

    # SimulatedActor returns a JSON string; validate it conforms to the schema.
    parsed = ActionResult.model_validate_json(result)

    assert isinstance(parsed.completed, bool)
    assert isinstance(parsed.steps_taken, list)
    assert parsed.outcome.strip(), "Outcome should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Stateful memory – serial asks (via handle.ask)                          #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stateful_memory_serial_asks():
    """
    Two consecutive activities should share the same stateful LLM context.
    We exercise this by asking questions via the handle's ask() method.
    """
    actor = SimulatedActor(steps=1)

    h1 = await actor.act("Start some new research.")
    ask_handle1 = await h1.ask(
        "Invent a unique codename. Reply with only the codename.",
    )
    code = (await ask_handle1.result()).strip()
    assert code, "Codename should not be empty"

    h2 = await actor.act("Continue the research.")
    ask_handle2 = await h2.ask("What codename did you just suggest? ")
    answer2 = (await ask_handle2.result()).lower()
    assert code.lower().split(" ")[-1] in answer2

    # Allow both handles to complete
    await h1.result()
    await h2.result()


# ────────────────────────────────────────────────────────────────────────────
# Steerable handle tests                                                      #
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 3.  Interject                                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_interject(monkeypatch):
    calls = {"interject": 0}
    original = SimulatedActorHandle.interject

    @functools.wraps(original)
    async def wrapped(self, instruction: str):  # type: ignore[override]
        calls["interject"] += 1
        return await original(self, instruction)

    monkeypatch.setattr(SimulatedActorHandle, "interject", wrapped, raising=True)

    actor = SimulatedActor(steps=1)
    handle = await actor.act("Show me all steps performed so far.")
    await asyncio.sleep(0.05)
    await handle.interject("Also consider revenue trends.")
    await handle.result()
    assert calls["interject"] == 1, ".interject should be invoked exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Stop                                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_stop(monkeypatch):
    actor = SimulatedActor(steps=1)
    handle = await actor.act("Generate a long report.")
    await asyncio.sleep(0.05)
    stop_msg = handle.stop("Not needed")
    assert "stopped" in stop_msg.lower()
    result = await handle.result()
    assert isinstance(result, str) and result.strip()
    assert handle.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 5.  Clarification handshake                                                 #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_requests_clarification():
    actor = SimulatedActor(steps=2, _requests_clarification=True)

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await actor.act(
        "Compile the quarterly report",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in question.lower()

    await down_q.put("Yes, please compile the Q1 report now.")
    result = await handle.result()
    assert isinstance(result, str) and result.strip()
    assert "q1 report" in result.lower()


# ────────────────────────────────────────────────────────────────────────────
# 6.  Pause → Resume round-trip                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_pause_and_resume(monkeypatch):
    counts = {"pause": 0, "resume": 0}

    orig_pause = SimulatedActorHandle.pause

    @functools.wraps(orig_pause)
    async def _patched_pause(self):  # type: ignore[override]
        counts["pause"] += 1
        return await orig_pause(self)

    monkeypatch.setattr(
        SimulatedActorHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    orig_resume = SimulatedActorHandle.resume

    @functools.wraps(orig_resume)
    async def _patched_resume(self):  # type: ignore[override]
        counts["resume"] += 1
        return await orig_resume(self)

    monkeypatch.setattr(
        SimulatedActorHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    actor = SimulatedActor(steps=3)
    handle = await actor.act("Summarise all open opportunities.")

    pause_reply = await handle.pause()
    assert "pause" in pause_reply.lower()

    res = await _assert_blocks_while_paused(handle.result())

    resume_reply = await handle.resume()
    assert "resume" in resume_reply.lower() or "running" in resume_reply.lower()

    answer = await asyncio.wait_for(res, timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip()
    assert counts == {"pause": 1, "resume": 1}


# ────────────────────────────────────────────────────────────────────────────
# 7.  Ask on handle                                                           #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask():
    actor = SimulatedActor(steps=1)
    handle = await actor.act("Summarize all unread messages this week.")

    # Ask a follow-up while running
    await asyncio.sleep(0.05)
    ask_handle = await handle.ask("What is the key point to emphasize?")
    reply = await ask_handle.result()
    assert isinstance(reply, str) and reply.strip()

    # The original handle should still be awaitable and produce a result
    result = await handle.result()
    assert isinstance(result, str) and result.strip()


# ────────────────────────────────────────────────────────────────────────────
# 8.  Pause should freeze duration timer                                       #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_freezes_duration():
    actor = SimulatedActor(duration=0.2)
    handle = await actor.act("Time-sensitive work.")

    # Give the worker thread a moment to start, then pause quickly
    await asyncio.sleep(0.05)
    await handle.pause()

    # While paused, wait longer than the total duration; it should NOT complete
    res = asyncio.create_task(handle.result())
    await asyncio.sleep(0.3)
    assert (
        not res.done()
    ), "result() must not complete while paused even if wall time exceeds duration"

    # Resume and ensure it doesn't complete immediately; some time should elapse
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    await handle.resume()
    answer = await asyncio.wait_for(res, timeout=2)
    elapsed_after_resume = loop.time() - t0
    assert isinstance(answer, str) and answer.strip()
    assert (
        elapsed_after_resume >= 0.05
    ), "Should wait after resume; clock was frozen while paused"


# ────────────────────────────────────────────────────────────────────────────
# 9.  Entrypoint observes FunctionManager docstring via ask (LinkedIn flow)   #
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_entrypoint_demonstrates_function_knowledge_during_ask():
    """
    Simulate a browser-style function that works on LinkedIn sales leads.
    The docstring should state:
      1) trouble logging into LinkedIn; 2) then resolved.

    When we ask the running SimulatedActor if it is encountering problems,
    the response should reference LinkedIn, proving function metadata was observed.
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

    actor = SimulatedActor(steps=2, duration=None)
    handle = await actor.act("Search sales leads.", entrypoint=fid)

    ask_handle = await handle.ask(
        "Did you or are you encountering any problems logging in? Reply briefly, explaining any relevant websites.",
    )
    reply = await ask_handle.result()
    assert isinstance(reply, str) and reply.strip(), "Expected a non-empty reply"
    assert "linkedin" in reply.lower(), f"Expected LinkedIn mention in: {reply!r}"

    await handle.result()


# ────────────────────────────────────────────────────────────────────────────
# 10.  Interject with image → simulation recognises spreadsheet               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject_image_guides_simulation_to_spreadsheet(monkeypatch):
    """
    Start a simulated task, interject with a screenshot (Google Sheets), then
    ask about progress; the reply should reference a sheet/spreadsheet.
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

    actor = SimulatedActor(steps=3, duration=None)
    handle = await actor.act(
        "We'll start working on organizing the rota for the admin assistants.",
    )

    # Interject with the image attached; annotation intentionally does not say "spreadsheet"
    await handle.interject(
        "Please start working on this file.",
        images=[
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=int(img_id)),
                annotation="rota file",
            ),
        ],
    )

    # Ask about status and infer file type from the visual context
    ask_handle = await handle.ask(
        "How is it going? What file are you working on? What file type is it?",
    )
    reply = await ask_handle.result()
    assert isinstance(reply, str) and reply.strip()
    assert "sheet" in reply.lower(), f"Expected 'sheet' mention in: {reply!r}"

    await handle.result()


# ────────────────────────────────────────────────────────────────────────────
# 11. next_notification emits progress and consumes a step                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_next_notification_emits_and_consumes_step():
    actor = SimulatedActor(steps=2, duration=None)
    handle = await actor.act("Quick simulated task.")

    # Capture remaining steps before
    before = handle.get_remaining_steps()
    assert isinstance(before, int) and before == 2

    # next_notification should return a real event and consume one step
    evt = await asyncio.wait_for(handle.next_notification(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(evt, dict)
    assert evt.get("type") == "notification"
    assert evt.get("tool_name") == "simulated_actor"
    assert isinstance(evt.get("message"), str) and evt.get("message").strip()

    after = handle.get_remaining_steps()
    assert isinstance(after, int) and after == before - 1

    # Complete the simulation and ensure result is available
    handle.simulate_step()
    res = await asyncio.wait_for(handle.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(res, str) and res.strip()


# ────────────────────────────────────────────────────────────────────────────
# 12.  Stop while paused should finish immediately                           #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_paused_finishes_immediately():
    actor = SimulatedActor(steps=5, duration=None)
    handle = await actor.act("A long-running simulated activity.")

    # Pause quickly to freeze the worker thread
    await asyncio.sleep(0.05)
    await handle.pause()

    # Give the worker enough time to enter the paused wait state before stopping
    await asyncio.sleep(0.2)

    # Stopping should unpause and complete immediately
    handle.stop("cancelled by user")
    result = await asyncio.wait_for(handle.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(result, str) and "stopped" in result.lower()
    assert handle.done()
    # Verify the worker thread terminates promptly (would hang pre-fix)
    for _ in range(100):
        th = getattr(handle, "_action_thread", None)
        if th is None or not getattr(th, "is_alive", lambda: False)():
            break
        await asyncio.sleep(0.01)
    th = getattr(handle, "_action_thread", None)
    assert (
        th is None or not th.is_alive()
    ), "Action thread should terminate after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 13.  Stop while waiting for clarification should finish immediately         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_waiting_for_clarification_finishes_immediately():
    # Configure the actor to request a clarification at the start of work
    actor = SimulatedActor(steps=None, duration=None, _requests_clarification=True)
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await actor.act(
        "Compile the annual report",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
    )

    # Wait until the clarification question is asked
    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(question, str) and "clarify" in question.lower()

    # Without answering, issue stop and ensure result returns promptly
    handle.stop("no longer needed")
    result = await asyncio.wait_for(handle.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(result, str) and "stopped" in result.lower()
    assert handle.done()
    # Verify the worker thread terminates promptly (would spin-wait pre-fix)
    for _ in range(100):
        th = getattr(handle, "_action_thread", None)
        if th is None or not getattr(th, "is_alive", lambda: False)():
            break
        await asyncio.sleep(0.01)
    th = getattr(handle, "_action_thread", None)
    assert (
        th is None or not th.is_alive()
    ), "Action thread should terminate after stop()"
