from __future__ import annotations

import asyncio
import functools
import pytest

from unity.conductor.simulated import SimulatedConductor
from unity.common.llm_helpers import AsyncToolUseLoopHandle

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import _handle_project


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Basic start-and-ask                                                    #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_simulated_conductor():
    cond = SimulatedConductor("Demo unified assistant for unit-tests.")
    h = await cond.ask("What are my open tasks today?")
    answer = await h.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Interject                                                             #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject_simulated_conductor(monkeypatch):
    calls = {"interject": 0}
    orig = AsyncToolUseLoopHandle.interject

    @functools.wraps(orig)
    async def wrapped(self, message: str):  # type: ignore[override]
        calls["interject"] += 1
        return await orig(self, message)

    monkeypatch.setattr(AsyncToolUseLoopHandle, "interject", wrapped, raising=True)

    cond = SimulatedConductor()
    h = await cond.ask("Summarise inbox and current tasks.")
    await asyncio.sleep(0.05)
    await h.interject("Also include any items due today.")

    await h.result()
    assert calls["interject"] == 1, ".interject should be invoked exactly once"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Stop                                                                  #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_simulated_conductor():
    cond = SimulatedConductor()
    h = await cond.ask("Produce a long consolidated report across domains.")
    await asyncio.sleep(0.05)
    h.stop()

    with pytest.raises(asyncio.CancelledError):
        await h.result()
    assert h.done(), "Handle should report done after stop()"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Optional clarification channels (best-effort)                         #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_conductor_supports_optional_clarification_channels():
    """
    Conductor provides an optional `request_clarification` tool to the loop when
    caller supplies duplex queues. We do not enforce that the LLM must use it,
    only that the presence of queues does not break execution.
    """
    cond = SimulatedConductor()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    h = await cond.ask(
        "Please help me triage ambiguous requests across domains.",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    # We don't assert that a clarification is asked (LLM-dependent).
    ans = await h.result()
    assert isinstance(ans, str) and ans.strip(), "Answer should be non-empty"


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Pause → Resume round-trip                                             #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume_simulated_conductor(monkeypatch):
    counts = {"pause": 0, "resume": 0}

    original_pause = AsyncToolUseLoopHandle.pause

    @functools.wraps(original_pause)
    def _patched_pause(self):  # type: ignore[override]
        counts["pause"] += 1
        return original_pause(self)

    monkeypatch.setattr(
        AsyncToolUseLoopHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    original_resume = AsyncToolUseLoopHandle.resume

    @functools.wraps(original_resume)
    def _patched_resume(self):  # type: ignore[override]
        counts["resume"] += 1
        return original_resume(self)

    monkeypatch.setattr(
        AsyncToolUseLoopHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    cond = SimulatedConductor()
    handle = await cond.ask("Generate a short multi-domain summary.")

    # Pause before awaiting the result
    handle.pause()

    # Start result() – it should block while paused
    res_task = asyncio.create_task(handle.result())
    await asyncio.sleep(0.1)
    assert not res_task.done(), "result() should block while the handle is paused"

    # Resume and ensure the task now completes
    handle.resume()

    answer = await asyncio.wait_for(res_task, timeout=60)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Exactly one pause and one resume
    assert counts == {
        "pause": 1,
        "resume": 1,
    }, "pause/resume should each be invoked exactly once"


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Reasoning steps toggle (ask + request)                                 #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_reasoning_steps_toggle_ask_and_request():
    cond = SimulatedConductor()

    # ask() – request hidden messages tuple
    h1 = await cond.ask(
        "List top priorities across tasks and contacts.",
        _return_reasoning_steps=True,
    )
    ans1, msgs1 = await h1.result()
    assert isinstance(ans1, str) and ans1.strip()
    assert isinstance(msgs1, list) and len(msgs1) >= 1

    # request() – also return (answer, messages)
    h2 = await cond.request(
        "Create a high-priority task to call Bob tomorrow morning.",
        _return_reasoning_steps=True,
    )
    ans2, msgs2 = await h2.result()
    assert isinstance(ans2, str) and ans2.strip()
    assert isinstance(msgs2, list) and len(msgs2) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Write (request) then Read (ask) – state carries via sub-managers        #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_conductor_request_then_ask_stateful():
    """
    A `request()` that (likely) touches the TaskScheduler should influence a
    subsequent `ask()` routed to the same sub-manager (stateful LLM behind it).
    """
    cond = SimulatedConductor()
    task_name = "Draft Budget FY26"

    # 1) Request creation of a high-priority task
    h_upd = await cond.request(
        f"Please create a new task called '{task_name}' with high priority.",
    )
    await h_upd.result()

    # 2) Ask about high-priority tasks – the answer should reference our task
    h_q = await cond.ask("Which tasks are high priority right now?")
    answer = (await h_q.result()).lower()

    assert "budget" in answer, "Answer should reference the task added via request()"


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Nested ask() on a running loop                                         #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_nested_ask_on_running_conductor_loop():
    cond = SimulatedConductor()
    handle = await cond.request("Prepare a plan to triage today's tasks.")

    # Ask a question about the running loop
    nested = await handle.ask("What key steps have been identified so far?")
    nested_answer = await nested.result()

    assert isinstance(nested_answer, str) and nested_answer.strip()
