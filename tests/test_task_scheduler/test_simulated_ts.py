# tests/test_simulated_task_scheduler.py
from __future__ import annotations

import asyncio
import pytest

from unity.task_scheduler.simulated import (
    SimulatedTaskScheduler,
    _SimulatedTaskScheduleHandle,
)

# helper used by the simulated-planner tests – keeps each test in its own
# temporary Unify project / trace context
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic start-and-ask                                                     #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_simulated_ts():
    ts = SimulatedTaskScheduler("Demo list for unit-tests.")
    handle = ts.ask("What are my open tasks today?")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Interject                                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject_simulated_ts(monkeypatch):
    # Count how many times the handle's .interject method is invoked
    counts = {"interject": 0}
    original_interject = _SimulatedTaskScheduleHandle.interject

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
    handle = ts.ask("Give me a summary of all tasks.")
    # Send a follow-up while it is “running”
    await asyncio.sleep(0.05)
    reply = handle.interject("Also include any deadlines, please.")
    assert "noted" in reply.lower()

    await handle.result()
    assert counts["interject"] == 1, ".interject should be called exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stop                                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_simulated_ts():
    ts = SimulatedTaskScheduler()
    handle = ts.ask("Produce a very long report about my tasks.")
    await asyncio.sleep(0.05)  # let the background thread spin up
    handle.stop()

    with pytest.raises(asyncio.CancelledError):
        await handle.result()

    assert handle.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Clarification handshake                                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_ts_requests_clarification():
    ts = SimulatedTaskScheduler()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = ts.ask(
        "Please prioritise everything appropriately.",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    # The handle must first raise a clarification question
    question = await asyncio.wait_for(up_q.get(), timeout=30)
    assert "clarify" in question.lower()

    # Provide an answer and ensure it flows through to the final result
    await down_q.put("Yes – prioritise according to deadlines.")
    answer = await handle.result()

    # Looser, semantically-grounded assertion
    assert isinstance(answer, str) and answer.strip(), "Answer should not be empty"
    # It should reflect the task theme (prioritisation) somehow
    assert "priorit" in answer.lower(), "Answer should reference prioritisation"


# ────────────────────────────────────────────────────────────────────────────
# 5.  Stateful memory across serial asks                                     #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_ts_stateful_memory_serial_asks():
    """
    Two consecutive .ask() calls should share the same conversation context.
    """
    ts = SimulatedTaskScheduler()

    # 1) Ask for a unique codename – any non-empty string
    h1 = ts.ask(
        "Please invent a codename for our secret task-force. "
        "Respond with only the codename.",
    )
    codename = (await h1.result()).strip()
    assert codename, "Codename should not be empty"

    # 2) Ask what codename was suggested
    h2 = ts.ask("Great. What codename did you propose earlier?")
    answer2 = (await h2.result()).lower()

    assert codename.lower() in answer2, "LLM should recall the previous codename"


# ────────────────────────────────────────────────────────────────────────────
# 6.  Update then ask – state propagated                                     #
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
    h_upd = ts.update(
        f"Please create a new task called '{task_name}' with high priority.",
    )
    _ = await h_upd.result()  # we don't assert its exact wording

    # 2) Ask about high-priority tasks – should mention the one we just added
    h_q = ts.ask("Which tasks are high priority right now?")
    answer = (await h_q.result()).lower()

    assert "budget" in answer, "Answer should reference the task added via update"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_simulated_cm_docstrings_match_base():
    """
    Public methods in SimulatedContactManager should copy the real
    BaseContactManager doc-strings one-for-one (via functools.wraps).
    """
    from unity.task_scheduler.base import BaseTaskScheduler
    from unity.task_scheduler.simulated import SimulatedTaskScheduler

    assert (
        SimulatedTaskScheduler.ask.__doc__.strip()
        == BaseTaskScheduler.ask.__doc__.strip()
    ), ".store doc-string was not copied correctly"

    assert (
        SimulatedTaskScheduler.update.__doc__.strip()
        == BaseTaskScheduler.update.__doc__.strip()
    ), ".retrieve doc-string was not copied correctly"
