# tests/test_simulated_task_scheduler.py
from __future__ import annotations

import re
import asyncio
import pytest
import functools

from unity.task_scheduler.simulated import (
    SimulatedTaskScheduler,
    _SimulatedTaskScheduleHandle,
)

# Helper identical to the one used elsewhere in the test-suite
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_simulated_ts_docstrings_match_base():
    """
    Public methods in SimulatedTaskScheduler should copy the real
    BaseTaskScheduler doc-strings one-for-one (via functools.wraps).
    """
    from unity.task_scheduler.base import BaseTaskScheduler
    from unity.task_scheduler.simulated import SimulatedTaskScheduler

    assert (
        BaseTaskScheduler.ask.__doc__.strip()
        in SimulatedTaskScheduler.ask.__doc__.strip()
    ), ".store doc-string was not copied correctly"

    assert (
        BaseTaskScheduler.update.__doc__.strip()
        in SimulatedTaskScheduler.update.__doc__.strip()
    ), ".retrieve doc-string was not copied correctly"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_simulated_ts():
    ts = SimulatedTaskScheduler("Demo list for unit-tests.")
    handle = await ts.ask("What are my open tasks today?")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial asks                                          #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_ts_stateful_memory_serial_asks():
    """
    Two consecutive .ask() calls should share the same conversation context.
    """
    ts = SimulatedTaskScheduler()

    # 1) Ask for a unique codename – any non-empty string
    h1 = await ts.ask(
        "Please invent a codename for our secret task-force. "
        "Respond with only the **single-word** codename and **nothing else**.",
    )
    codename = await h1.result()
    codename = re.sub(r"\W+", "", codename.strip().lower().replace("codename", ""))
    assert codename, "Codename should not be empty"

    # 2) Ask what codename was suggested
    h2 = await ts.ask("Great. What codename did you propose earlier?")
    answer2 = (await h2.result()).lower()
    answer2 = re.sub(r"\W+", "", answer2.strip().lower().replace("codename", ""))

    assert (codename in answer2) or (
        answer2 in codename
    ), "LLM should recall the previous codename"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Update then ask – state propagated                                     #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_ts_stateful_update_then_ask():
    """
    An .update() call should influence subsequent .ask() calls.
    """
    ts = SimulatedTaskScheduler()
    task_name = "Draft Budget FY26"

    # 1) Tell the manager to add a new high-priority task
    h_upd = await ts.update(
        f"Please create a new task called '{task_name}' with high priority.",
    )
    _ = await h_upd.result()  # we don't assert its exact wording

    # 2) Ask about high-priority tasks – should mention the one we just added
    h_q = await ts.ask("Which tasks are high priority right now?")
    answer = (await h_q.result()).lower()

    assert "budget" in answer, "Answer should reference the task added via update"


# ────────────────────────────────────────────────────────────────────────────
# Steerable handle tests                                                     #
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 5.  Interject                                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject_simulated_ts(monkeypatch):
    counts = {"interject": 0}
    original_interject = _SimulatedTaskScheduleHandle.interject

    @functools.wraps(original_interject)
    def wrapped(self, message: str) -> str:  # type: ignore[override]
        counts["interject"] += 1
        return original_interject(self, message)

    monkeypatch.setattr(
        _SimulatedTaskScheduleHandle,
        "interject",
        wrapped,
        raising=True,
    )

    ts = SimulatedTaskScheduler("Demo list")
    handle = await ts.ask("Give me a summary of all tasks.")
    # Send a follow-up while it is “running”
    await asyncio.sleep(0.05)
    reply = handle.interject("Also include any deadlines, please.")
    assert "noted" in reply.lower()

    await handle.result()
    assert counts["interject"] == 1, ".interject should be called exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 6.  Stop                                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_simulated_ts():
    ts = SimulatedTaskScheduler()
    handle = await ts.ask("Produce a very long report about my tasks.")
    await asyncio.sleep(0.05)
    handle.stop(cancel=True)

    with pytest.raises(asyncio.CancelledError):
        await handle.result()

    assert handle.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Clarification handshake                                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_ts_requests_clarification():
    ts = SimulatedTaskScheduler()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await ts.ask(
        "Please prioritise everything appropriately.",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
        _requests_clarification=True,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=60)
    assert "clarify" in question.lower()

    await down_q.put("Yes – prioritise according to deadlines.")
    answer = await handle.result()

    assert isinstance(answer, str) and answer.strip(), "Answer should not be empty"
    assert "priorit" in answer.lower(), "Answer should reference prioritisation"


# ────────────────────────────────────────────────────────────────────────────
# 8.  Pause → Resume round-trip + valid_tools                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume_simulated_ts(monkeypatch):
    """
    Verify that a `_SimulatedTaskScheduleHandle` may be paused and resumed and
    that `valid_tools` updates correspondingly.
    """
    counts = {"pause": 0, "resume": 0}

    # --- patch pause -------------------------------------------------------
    orig_pause = _SimulatedTaskScheduleHandle.pause

    @functools.wraps(orig_pause)
    def _patched_pause(self):  # type: ignore[override]
        counts["pause"] += 1
        return orig_pause(self)

    monkeypatch.setattr(
        _SimulatedTaskScheduleHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    # --- patch resume ------------------------------------------------------
    orig_resume = _SimulatedTaskScheduleHandle.resume

    @functools.wraps(orig_resume)
    def _patched_resume(self):  # type: ignore[override]
        counts["resume"] += 1
        return orig_resume(self)

    monkeypatch.setattr(
        _SimulatedTaskScheduleHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    ts = SimulatedTaskScheduler()
    handle = await ts.ask("List tomorrow's tasks.")

    tools_initial = handle.valid_tools
    assert "pause" in tools_initial and "resume" not in tools_initial

    pause_reply = handle.pause()
    assert "pause" in pause_reply.lower()

    tools_paused = handle.valid_tools
    assert "resume" in tools_paused and "pause" not in tools_paused

    res_task = asyncio.create_task(handle.result())
    await asyncio.sleep(0.1)
    assert not res_task.done(), "result() should block while the handle is paused"

    resume_reply = handle.resume()
    assert "resume" in resume_reply.lower() or "running" in resume_reply.lower()

    tools_running = handle.valid_tools
    assert "pause" in tools_running and "resume" not in tools_running

    answer = await asyncio.wait_for(res_task, timeout=60)
    assert isinstance(answer, str) and answer.strip()

    assert counts == {
        "pause": 1,
        "resume": 1,
    }, "pause/resume should each be called once"


# ────────────────────────────────────────────────────────────────────────────
# 9.  Nested ask on handle                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask():
    """
    The internal handle returned by SimulatedTaskScheduler.ask exposes a
    dynamic ask() method that should produce a nested handle whose result can
    be awaited independently of the parent.
    """
    ts = SimulatedTaskScheduler()

    # Start an initial ask to obtain the live handle
    handle = await ts.ask("Summarize all tasks due this week.")

    # Add extra context to ensure nested prompt includes it
    handle.interject("Focus on European enterprise accounts.")

    # Invoke the dynamic ask on the running handle
    nested = await handle.ask("What is the key task to prioritize?")

    nested_answer = await nested.result()
    assert isinstance(nested_answer, str) and nested_answer.strip(), (
        "Nested ask() should yield a non-empty string answer",
    )
    assert "europe" in nested_answer.lower()

    # The original handle should still be awaitable and produce an answer
    handle_answer = await handle.result()
    assert isinstance(handle_answer, str) and handle_answer.strip(), (
        "Handle should still yield a non-empty answer after nested ask",
    )
