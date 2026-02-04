# tests/task_scheduler/test_simulated.py
from __future__ import annotations

import asyncio
import pytest
import functools

from unity.task_scheduler.simulated import (
    SimulatedTaskScheduler,
    _SimulatedTaskScheduleHandle,
)

# Helper identical to the one used elsewhere in the test-suite
from tests.helpers import (
    _handle_project,
    _ack_ok,
    _assert_blocks_while_paused,
    DEFAULT_TIMEOUT,
    _normalize_alnum_lower,
)


# ────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_docstrings_match_base():
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
async def test_start_and_ask():
    ts = SimulatedTaskScheduler("Demo list for unit-tests.")
    handle = await ts.ask("What are my open tasks today?")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial asks                                          #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stateful_memory_serial_asks():
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
    codename = _normalize_alnum_lower(codename.replace("codename", ""))
    assert codename, "Codename should not be empty"

    # 2) Ask what codename was suggested
    h2 = await ts.ask("Great. What codename did you propose earlier?")
    answer2 = (await h2.result()).lower()
    answer2 = _normalize_alnum_lower(answer2.replace("codename", ""))

    assert (codename in answer2) or (
        answer2 in codename
    ), "LLM should recall the previous codename"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Update then ask – state propagated                                     #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stateful_update_then_ask():
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
async def test_interject(monkeypatch):
    counts = {"interject": 0}
    original_interject = _SimulatedTaskScheduleHandle.interject

    @functools.wraps(original_interject)
    async def wrapped(self, message: str, **kwargs) -> str:  # type: ignore[override]
        counts["interject"] += 1
        return await original_interject(self, message, **kwargs)

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
    reply = await handle.interject("Also include any deadlines, please.")
    assert _ack_ok(reply)

    await handle.result()
    assert counts["interject"] == 1, ".interject should be called exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 6.  Stop                                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop():
    ts = SimulatedTaskScheduler()
    handle = await ts.ask("Produce a very long report about my tasks.")
    await asyncio.sleep(0.05)
    await handle.stop(cancel=True)
    await handle.result()
    assert handle.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Clarification handshake                                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_requests_clarification():
    ts = SimulatedTaskScheduler()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await ts.ask(
        "Please prioritise everything appropriately.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in question.lower()

    await down_q.put("Yes – prioritise according to deadlines.")
    answer = await handle.result()

    assert isinstance(answer, str) and answer.strip(), "Answer should not be empty"
    assert "priorit" in answer.lower(), "Answer should reference prioritisation"


# ────────────────────────────────────────────────────────────────────────────
# 8.  Pause → Resume round-trip                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume(monkeypatch):
    """
    Verify that a `_SimulatedTaskScheduleHandle` may be paused and resumed.
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

    pause_reply = await handle.pause()
    assert "pause" in (pause_reply or "").lower()

    res_task = await _assert_blocks_while_paused(handle.result())

    resume_reply = await handle.resume()
    assert (
        "resume" in (resume_reply or "").lower()
        or "running" in (resume_reply or "").lower()
    )

    answer = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
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
    await handle.interject("Focus on emails that need to be sent.")

    # Invoke the dynamic ask on the running handle
    nested = await handle.ask("What is the key task to prioritize within this summary?")

    nested_answer = await nested.result()
    assert isinstance(nested_answer, str) and nested_answer.strip(), (
        "Nested ask() should yield a non-empty string answer",
    )
    assert "email" in nested_answer.lower()

    # The original handle should still be awaitable and produce an answer
    handle_answer = await handle.result()
    assert isinstance(handle_answer, str) and handle_answer.strip(), (
        "Handle should still yield a non-empty answer after nested ask",
    )


# ────────────────────────────────────────────────────────────────────────────
# 10.  Execute – basic completion                                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_execute_basic_completion():
    """
    SimulatedTaskScheduler.execute should return a live handle that completes.
    Keep the inner actor alive indefinitely; stop explicitly to finish.
    """
    ts = SimulatedTaskScheduler(actor_steps=None, actor_duration=None)
    handle = await ts.execute("Prepare slides for kickoff")
    # Explicitly stop to avoid relying on step-based completion
    await handle.stop(cancel=False)
    answer = await asyncio.wait_for(handle.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip()
    # The simulated actor should report it was stopped
    assert "stopped" in answer.lower()


# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
# 11.  Execute – clarification handshake                                      #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_execute_requests_clarification():
    """
    When _requests_clarification=True, execute should request clarification via queues.
    """
    ts = SimulatedTaskScheduler(actor_steps=None, actor_duration=None)
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await ts.execute(
        "Run the data export",
        _requests_clarification=True,
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in question.lower()
    await down_q.put("Export only records updated in the last 24 hours.")

    answer = await asyncio.wait_for(handle.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip()
    # The simulated actor completes immediately after clarification
    assert "clarification" in answer.lower()


# ────────────────────────────────────────────────────────────────────────────
# 12.  Clear – reset and remain usable                                        #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_clear():
    """
    SimulatedTaskScheduler.clear should reset the manager (hard-coded completion)
    and remain usable afterwards.
    """
    ts = SimulatedTaskScheduler()
    # Do an update to create some prior state in the stateful LLM
    h_upd = await ts.update(
        "Create a temporary task called 'Temp Task' with low priority.",
    )
    await asyncio.wait_for(h_upd.result(), timeout=DEFAULT_TIMEOUT)

    # Clear should not raise and should be quick (no LLM roundtrip)
    ts.clear()

    # Post-clear, an ask should still work
    h_q = await ts.ask("List any tasks scheduled for today.")
    answer = await asyncio.wait_for(h_q.result(), timeout=DEFAULT_TIMEOUT)
    assert (
        isinstance(answer, str) and answer.strip()
    ), "Answer should be non-empty after clear()"


# ────────────────────────────────────────────────────────────────────────────
# 13.  Stop while paused should finish immediately (ask handle)              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_paused_finishes_immediately():
    ts = SimulatedTaskScheduler()
    h = await ts.ask("Generate an exhaustive task summary.")
    await h.pause()
    res_task = asyncio.create_task(h.result())
    await asyncio.sleep(0.1)
    assert not res_task.done()
    await h.stop(cancel=True, reason="cancelled by user")
    out = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str)
    assert h.done()


# ────────────────────────────────────────────────────────────────────────────
# 14.  Stop while waiting for clarification should finish immediately        #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_waiting_for_clarification_finishes_immediately():
    ts = SimulatedTaskScheduler()
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()
    h = await ts.ask(
        "Please prioritise everything appropriately.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )
    q = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in q.lower()
    await h.stop(cancel=True, reason="no longer needed")
    out = await asyncio.wait_for(h.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str)
    assert h.done()


@_handle_project
def test_simulated_task_scheduler_reduce_shapes():
    ts = SimulatedTaskScheduler()

    scalar = ts.reduce(metric="sum", keys="task_id")
    assert isinstance(scalar, (int, float))

    multi = ts.reduce(metric="max", keys=["task_id"])
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"task_id"}

    grouped = ts.reduce(metric="sum", keys="task_id", group_by="status")
    assert isinstance(grouped, dict)
