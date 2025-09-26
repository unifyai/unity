from __future__ import annotations

import asyncio
import functools

import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


TASK_LIKE_REQUESTS: list[str] = [
    "Start the task to call Alice about the Q3 budget now.",
    "Execute the task named 'Draft Budget FY26' immediately.",
    "Run the task: Email Contoso about invoices today at 4pm.",
    (
        "Begin execution of the 'Prepare slides for kickoff' task and confirm once started."
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text", TASK_LIKE_REQUESTS)
@_handle_project
async def test_task_like_requests_use_taskscheduler_execute_not_actor(
    request_text: str,
    monkeypatch,
):
    # Ensure the underlying SimulatedActor completes immediately during execute
    import unity.actor.simulated as _actor_sim

    _orig_sim_actor = _actor_sim.SimulatedActor

    @functools.wraps(_orig_sim_actor)
    def _patched_sim_actor(*args, **kwargs):  # type: ignore
        kw = dict(kwargs)
        kw.setdefault("steps", 0)
        kw.setdefault("duration", None)
        return _orig_sim_actor(*args, **kw)

    monkeypatch.setattr(_actor_sim, "SimulatedActor", _patched_sim_actor, raising=True)
    cond = SimulatedConductor(
        description=(
            "Assistant that executes clearly-defined tasks when asked to start or run a task."
        ),
    )

    handle = await cond.request(
        request_text,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # TaskScheduler.execute should be invoked at least once
    executed_ts_list = tool_names_from_messages(messages, "SimulatedTaskScheduler")
    executed_ts = set(executed_ts_list)
    assert executed_ts, "Expected at least one tool call"
    assert (
        executed_ts_list.count("SimulatedTaskScheduler_execute") >= 1
    ), f"Expected SimulatedTaskScheduler_execute to run at least once, saw order: {executed_ts_list}"

    # Actor.act must NOT be called for task-execution requests
    executed_actor_list = tool_names_from_messages(messages, "SimulatedActor")
    executed_actor = set(executed_actor_list)
    assert (
        "SimulatedActor_act" not in executed_actor
    ), f"Actor.act must not run for execute scenarios, saw: {sorted(executed_actor)}"

    # If the assistant explicitly requested tools, they should reference execute here
    requested_ts = set(
        assistant_requested_tool_names(messages, "SimulatedTaskScheduler"),
    )
    if requested_ts:
        assert requested_ts <= {
            "SimulatedTaskScheduler_execute",
        }, f"Assistant should only request SimulatedTaskScheduler_execute here, saw: {sorted(requested_ts)}"
