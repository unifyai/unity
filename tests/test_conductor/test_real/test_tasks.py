from __future__ import annotations

import asyncio
import pytest

pytestmark = pytest.mark.eval

from unity.conductor.simulated import SimulatedConductor
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.actor.simulated import SimulatedActor

from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


@pytest.mark.asyncio
@_handle_project
async def test_ask_calls_scheduler():
    actor = SimulatedActor(steps=0)
    ts = TaskScheduler(actor=actor)

    # Seed a primed task to be discoverable via ask
    label = "Quarterly report"
    ts._create_task(name=label, description=label, status="primed")

    cond = SimulatedConductor(task_scheduler=ts, actor=actor)

    handle = await cond.request(
        "Which task is currently primed?",
        _return_reasoning_steps=True,
    )
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and label.lower() in answer.lower()

    executed_list = tool_names_from_messages(messages, "TaskScheduler")
    requested_list = assistant_requested_tool_names(messages, "TaskScheduler")
    assert executed_list, "Expected at least one tool call"
    assert set(executed_list) == {
        "TaskScheduler_ask",
    }, f"Only TaskScheduler_ask should run; saw: {sorted(set(executed_list))}"
    assert (
        executed_list.count("TaskScheduler_ask") == 1
    ), f"Expected exactly one TaskScheduler_ask call, saw order: {executed_list}"
    assert set(requested_list) <= {
        "TaskScheduler_ask",
    }, f"Assistant should request only TaskScheduler_ask, saw: {sorted(set(requested_list))}"

    # Global exclusivity: verify no other manager tools ran
    all_tool_names = [
        str(m.get("name"))
        for m in messages
        if m.get("role") == "tool"
        and not str(m.get("name") or "").startswith("check_status_")
    ]
    assert all_tool_names, "Expected at least one tool call overall"
    assert all(
        n.startswith("TaskScheduler_ask") or n.startswith("continue_TaskScheduler_ask")
        for n in all_tool_names
    ), f"Unexpected tools executed: {sorted(set(all_tool_names))}"
    assert (
        len(all_tool_names) == 1
    ), f"Only one tool call expected; saw: {all_tool_names}"


@pytest.mark.asyncio
@_handle_project
async def test_update_calls_scheduler():
    actor = SimulatedActor(steps=0)
    ts = TaskScheduler(actor=actor)
    cond = SimulatedConductor(task_scheduler=ts, actor=actor)

    request_text = (
        "Create a new task called 'Promote Jeff Smith' with the description "
        "'Send an email to Jeff Smith to congratulate him on the promotion.'"
    )
    handle = await cond.request(request_text, _return_reasoning_steps=True)
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    executed_list = tool_names_from_messages(messages, "TaskScheduler")
    requested_list = assistant_requested_tool_names(messages, "TaskScheduler")
    assert executed_list, "Expected at least one tool call"
    assert (
        executed_list[0] == "TaskScheduler_update"
    ), f"First call must be TaskScheduler_update, saw order: {executed_list}"
    assert set(executed_list) <= {
        "TaskScheduler_update",
    }, f"Only TaskScheduler_update should run, saw: {sorted(set(executed_list))}"
    assert "TaskScheduler_ask" not in set(
        executed_list,
    ), f"TaskScheduler_ask must not run, saw: {sorted(set(executed_list))}"
    assert set(requested_list) <= {
        "TaskScheduler_update",
    }, f"Assistant should request only TaskScheduler_update, saw: {sorted(set(requested_list))}"

    # Verify the mutation took effect in the real TaskScheduler instance
    tasks = ts._filter_tasks()
    assert tasks and any(
        t.name.lower() == "promote jeff smith"
        or "promote jeff smith" in t.description.lower()
        for t in tasks
    )


@pytest.mark.asyncio
@_handle_project
async def test_execute_calls_scheduler():
    # Use a short, step-bound actor so execution completes quickly
    actor = SimulatedActor(steps=1)
    ts = TaskScheduler(actor=actor)

    # Seed a queued task we will execute by description
    name = "Prepare the monthly analytics dashboard"
    ts._create_task(name=name, description=name)

    cond = SimulatedConductor(task_scheduler=ts, actor=actor)

    handle = await cond.request(
        f"Run the task named '{name}' now.",
        _return_reasoning_steps=True,
    )
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    executed_list = tool_names_from_messages(messages, "TaskScheduler")
    requested_list = assistant_requested_tool_names(messages, "TaskScheduler")
    assert executed_list, "Expected at least one tool call"
    # Execute must happen exactly once; ask may occur to resolve id, but update must not
    assert (
        executed_list.count("TaskScheduler_execute") == 1
    ), f"Expected exactly one TaskScheduler_execute; saw order: {executed_list}"
    assert "TaskScheduler_update" not in set(
        executed_list,
    ), f"TaskScheduler_update must not run during execute; saw: {sorted(set(executed_list))}"
    assert "TaskScheduler_execute" in set(
        requested_list,
    ), f"Assistant should request TaskScheduler_execute; saw: {sorted(set(requested_list))}"
