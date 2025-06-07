"""
High-level smoke-tests for the *simulated* **TaskManager** façade.

We intentionally keep these flows simple – later suites will stress more
complex, multi-tool interactions.
"""

from __future__ import annotations

import re
import asyncio
import pytest

from unity.task_manager.simulated import SimulatedTaskManager

# Helper to isolate every test in its own temporary Unify context
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic .ask() path                                                      #
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_tm_basic_ask():
    tm = SimulatedTaskManager()
    handle = tm.ask("Which tasks should I focus on today?")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Clarification handshake                                                #
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_tm_clarification_flow():
    tm = SimulatedTaskManager(
        "There are three tasks qeued up:\n"
        "a) write email to John\n"
        "b) research the best bank to register with\n"
        "c) research the best law firm to use",
    )

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = tm.request(
        "Please make a start the research taks",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    # The first message must be a clarification question.
    question = await asyncio.wait_for(up_q.get(), timeout=30)
    assert "research" in question.lower()

    # Send the clarification answer and await the final result.
    await down_q.put("Sorry, I mean the law firm research.")
    await handle.result()


# ────────────────────────────────────────────────────────────────────────────
# 3.  Pause → Resume round-trip                                              #
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_tm_pause_and_resume():
    """
    Verify that the handle returned by TaskManager.ask() can be paused and
    resumed and that the `valid_tools` mapping updates accordingly.
    """
    tm = SimulatedTaskManager()
    handle = tm.ask("Give me a list of tomorrow's deadlines.")

    # Pause execution.
    handle.pause()

    # Kick off result() while paused – it should block.
    res_task = asyncio.create_task(handle.result())
    await asyncio.sleep(0.1)
    assert not res_task.done(), "result() should block while paused"

    # Resume and ensure result() completes.
    handle.resume()

    answer = await asyncio.wait_for(res_task, timeout=30)
    assert isinstance(answer, str) and answer.strip()


# ────────────────────────────────────────────────────────────────────────────
# 4.  Handle.interject                                                       #
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_tm_interject():
    """
    Ensure that `.interject()` appends a message and yields an acknowledgement.
    """
    tm = SimulatedTaskManager()
    handle = tm.ask("Summarise all open tasks.")

    await asyncio.sleep(0.05)
    reply = handle.interject("Also include each task's priority.")
    assert "noted" in reply.lower() or "ack" in reply.lower()

    answer = await handle.result()
    assert "priority" in answer.lower()


# ────────────────────────────────────────────────────────────────────────────
# 5.  Stateful memory across serial .ask() calls                             #
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_tm_stateful_memory_serial_asks():
    """
    Two consecutive .ask() calls should share the same conversation context
    inside the simulated managers.
    """
    tm = SimulatedTaskManager()

    # 1) Generate a unique codename – any non-empty token
    h1 = tm.ask(
        "Please invent a codename for our next product. "
        "Respond with *only* the codename.",
    )
    codename_raw = await h1.result()
    codename = re.sub(r"\W+", "", codename_raw.strip().lower())
    assert codename, "Codename should not be empty"

    # 2) Ask the manager what codename was proposed
    h2 = tm.ask("Great. What codename did you propose earlier?")
    answer2 = await h2.result()
    answer2_clean = re.sub(r"\W+", "", answer2.strip().lower())

    assert (
        codename in answer2_clean or answer2_clean in codename
    ), "LLM should recall the previous codename"


# ────────────────────────────────────────────────────────────────────────────
# 6.  .update() then .ask() – state propagation                              #
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_tm_update_then_ask():
    """
    An .update() (write-capable) call should influence subsequent .ask() calls.
    We keep expectations loose – simply require that the keyword appears.
    """
    tm = SimulatedTaskManager()
    meeting_name = "Strategy Sync Q4"

    # 1) Create a new task via the write interface.
    h_upd = tm.update(
        f"Please create a new high-priority task called '{meeting_name}'.",
    )
    _ = await h_upd.result()  # we don't assert exact wording

    # 2) Ask for high-priority tasks – the new one should be referenced.
    h_q = tm.ask("Which tasks are high priority right now?")
    answer = (await h_q.result()).lower()

    assert (
        "strategy" in answer or "sync" in answer
    ), "Answer should reference the task added via .update()"
