import asyncio
import functools
import pytest

from unity.actor.simulated import SimulatedActor, SimulatedActorHandle
from tests.helpers import _handle_project
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
async def test_start_and_act_simulated_actor():
    actor = SimulatedActor(duration=0.1)
    handle = await actor.act("Perform a quick demo.")
    result = await handle.result()
    assert isinstance(result, str) and result.strip(), "Result should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Stateful memory – serial asks (via handle.ask)                          #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_actor_stateful_memory_serial_asks():
    """
    Two consecutive activities should share the same stateful LLM context.
    We exercise this by asking questions via the handle's ask() method.
    """
    actor = SimulatedActor(steps=1)

    h1 = await actor.act("Start some new research.")
    code = await h1.ask("Invent a unique codename. Reply with only the codename.")
    code = code.strip()
    assert code, "Codename should not be empty"

    h2 = await actor.act("Continue the research.")
    answer2 = (await h2.ask("What codename did you just suggest? ")).lower()
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

    question = await asyncio.wait_for(up_q.get(), timeout=60)
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
    def _patched_pause(self):  # type: ignore[override]
        counts["pause"] += 1
        return orig_pause(self)

    monkeypatch.setattr(
        SimulatedActorHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    orig_resume = SimulatedActorHandle.resume

    @functools.wraps(orig_resume)
    def _patched_resume(self):  # type: ignore[override]
        counts["resume"] += 1
        return orig_resume(self)

    monkeypatch.setattr(
        SimulatedActorHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    actor = SimulatedActor(steps=3)
    handle = await actor.act("Summarise all open opportunities.")

    pause_reply = handle.pause()
    assert "pause" in pause_reply.lower()

    res = asyncio.create_task(handle.result())
    await asyncio.sleep(0.1)
    assert not res.done(), "result() must wait while paused"

    resume_reply = handle.resume()
    assert "resume" in resume_reply.lower() or "running" in resume_reply.lower()

    answer = await asyncio.wait_for(res, timeout=60)
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
    reply = await handle.ask("What is the key point to emphasize?")
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
    handle.pause()

    # While paused, wait longer than the total duration; it should NOT complete
    res = asyncio.create_task(handle.result())
    await asyncio.sleep(0.3)
    assert (
        not res.done()
    ), "result() must not complete while paused even if wall time exceeds duration"

    # Resume and ensure it doesn't complete immediately; some time should elapse
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    handle.resume()
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

    reply = await handle.ask(
        "Did you or are you encountering any problems logging in? Reply briefly, explaining any relevant websites.",
    )
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
    reply = await handle.ask(
        "How is it going? What file are you working on? What file type is it?",
    )
    assert isinstance(reply, str) and reply.strip()
    assert "sheet" in reply.lower(), f"Expected 'sheet' mention in: {reply!r}"

    await handle.result()
