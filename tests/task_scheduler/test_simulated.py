# tests/task_scheduler/test_simulated.py
from __future__ import annotations

import asyncio
import pytest

from unity.task_scheduler.simulated import (
    SimulatedTaskScheduler,
)

# Helper identical to the one used elsewhere in the test-suite
from tests.helpers import (
    _handle_project,
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
@pytest.mark.llm_call
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
@pytest.mark.llm_call
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
@pytest.mark.llm_call
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
# 10.  Execute – basic completion                                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
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
# 12.  Clear – reset and remain usable                                        #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
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
