from __future__ import annotations

import asyncio
import pytest

from unity.conductor.simulated import SimulatedConductor
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.actor.simulated import SimulatedActor
from unity.task_scheduler.types.status import Status

from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


async def _create_task(ts: TaskScheduler, name: str) -> int:
    out = ts._create_task(name=name, description=name)
    return int(out["details"]["task_id"])


@pytest.mark.asyncio
@_handle_project
async def test_start_task_immediate_execute_no_outer_llm_turn():
    # Steps-based actor: single step completes immediately on result()
    actor = SimulatedActor(steps=1)
    ts = TaskScheduler(actor=actor)

    # Seed a queued task and capture its id
    task_name = "Email Contoso about invoices"
    tid = await _create_task(ts, task_name)

    cond = SimulatedConductor(task_scheduler=ts, actor=actor)

    # Start execution from task id with no initial LLM turn in Conductor.request
    handle = await cond.start_task(
        tid,
        trigger_reason="the scheduled start time has arrived",
    )

    # Await final outcome
    answer = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str)

    # Verify the task completed
    rows = ts._filter_tasks(filter=f"task_id == {tid}")
    assert any(r.status == Status.completed for r in rows)

    # Inspect outer loop transcript: seeded assistant tool_call occurred; outer may add a final assistant summary
    messages = handle.get_history()
    asst_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert asst_msgs, "Expected at least one assistant message (seeded tool_call)"
    # Exactly one assistant message should contain tool_calls – the seeded execute call
    asst_with_tc = [m for m in asst_msgs if (m.get("tool_calls") or [])]
    assert (
        len(asst_with_tc) == 1
    ), f"Expected exactly one assistant tool_call message; saw {len(asst_with_tc)}"
    tcs = asst_with_tc[0].get("tool_calls") or []
    assert any(
        (tc.get("function", {}) or {})
        .get("name", "")
        .startswith("TaskScheduler_execute")
        for tc in tcs
    ), f"Seeded assistant tool_call should be TaskScheduler_execute; saw: {tcs}"

    # Cross-check using helper utilities
    executed_list = tool_names_from_messages(messages, "TaskScheduler")
    requested_list = assistant_requested_tool_names(messages, "TaskScheduler")
    assert executed_list, "Expected the execute tool to run"
    assert executed_list.count("TaskScheduler_execute") == 1
    assert set(executed_list) <= {"TaskScheduler_execute"}
    assert set(requested_list) <= {"TaskScheduler_execute"}


@pytest.mark.asyncio
@_handle_project
async def test_start_task_interject_pause_resume_then_complete():
    # Actor requires two steps; we'll steer with pause/resume and an interjection
    actor = SimulatedActor(steps=2)
    ts = TaskScheduler(actor=actor)

    task_name = "Prepare the weekly analytics dashboard"
    tid = await _create_task(ts, task_name)

    cond = SimulatedConductor(task_scheduler=ts, actor=actor)

    handle = await cond.start_task(
        tid,
        trigger_reason="an incoming phone call matched the trigger criteria",
    )

    # Exercise steering APIs on the returned handle
    handle.pause()
    handle.resume()
    await handle.interject("Please proceed with the next step promptly.")

    answer = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str)

    # Verify completion
    rows = ts._filter_tasks(filter=f"task_id == {tid}")
    assert any(r.status == Status.completed for r in rows)

    # Outer loop should still only have the single seeded assistant tool_call (final assistant summary may be present)
    messages = handle.get_history()
    asst_msgs = [m for m in messages if m.get("role") == "assistant"]
    asst_with_tc = [m for m in asst_msgs if (m.get("tool_calls") or [])]
    # Allow dynamic status checks; require exactly one TaskScheduler_execute tool_call overall
    all_tc_names = [
        (tc.get("function", {}) or {}).get("name", "")
        for m in asst_with_tc
        for tc in (m.get("tool_calls") or [])
    ]
    assert (
        sum(1 for n in all_tc_names if n.startswith("TaskScheduler_execute")) == 1
    ), f"Expected exactly one TaskScheduler_execute among assistant tool_calls; saw: {all_tc_names}"
    # Any other assistant tool_calls (if present) must be dynamic check_status helpers
    for n in all_tc_names:
        assert n.startswith("TaskScheduler_execute") or n.startswith(
            "check_status_",
        ), f"Unexpected assistant tool_call: {n}"

    executed_list = tool_names_from_messages(messages, "TaskScheduler")
    requested_list = assistant_requested_tool_names(messages, "TaskScheduler")
    assert executed_list.count("TaskScheduler_execute") == 1
    assert set(requested_list) <= {"TaskScheduler_execute"}
