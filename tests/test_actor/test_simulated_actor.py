import asyncio
import functools
import pytest

from unity.actor.simulated import SimulatedActor, SimulatedActorHandle
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic start-and-act                                                     #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_act_simulated_actor():
    actor = SimulatedActor(timeout=0.1)
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
    actor = SimulatedActor(steps=1, _requests_clarification=True)

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await actor.act(
        "Compile the quarterly report",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=60)
    assert "clarify" in question.lower()

    await down_q.put("Yes, please compile the Q1 report now.")
    result = await handle.result()
    assert isinstance(result, str) and result.strip()
    assert "q1 report" in result.lower()


# ────────────────────────────────────────────────────────────────────────────
# 6.  Pause → Resume round-trip + valid_tools                                  #
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

    actor = SimulatedActor(steps=2)
    handle = await actor.act("Summarise all open opportunities.")

    tools_initial = handle.valid_tools
    assert "pause" in tools_initial and "resume" not in tools_initial

    pause_reply = handle.pause()
    assert "pause" in pause_reply.lower()

    tools_paused = handle.valid_tools
    assert "resume" in tools_paused and "pause" not in tools_paused

    res = asyncio.create_task(handle.result())
    await asyncio.sleep(0.1)
    assert not res.done(), "result() must wait while paused"

    resume_reply = handle.resume()
    assert "resume" in resume_reply.lower() or "running" in resume_reply.lower()

    tools_running = handle.valid_tools
    assert "pause" in tools_running and "resume" not in tools_running

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
