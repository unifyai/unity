from __future__ import annotations

import asyncio

import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


SANDBOX_REQUESTS: list[str] = [
    "Open a browser window so we can walk through the setup together.",
    "Can you open the Settings app? I want to show you something.",
    (
        "Let's start a quick sandbox session: open the browser and navigate to the "
        "dashboard; I'll guide you live."
    ),
    "Open Notes so we can jot down ideas as we talk.",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text", SANDBOX_REQUESTS)
@_handle_project
async def test_actor_sandbox_requests_use_actor_not_task_execute(request_text: str):
    cond = SimulatedConductor(
        description=(
            "Assistant available to act directly in a sandbox; tasks are not required for these interactions."
        ),
    )

    handle = await cond.request(
        request_text,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Actor should be invoked at least once
    executed_actor_list = tool_names_from_messages(messages, "SimulatedActor")
    executed_actor = set(executed_actor_list)
    assert executed_actor, "Expected at least one tool call"
    assert (
        executed_actor_list.count("SimulatedActor_act") >= 1
    ), f"Expected SimulatedActor_act to run at least once, saw order: {executed_actor_list}"

    # TaskScheduler.execute must NOT be called for sandbox-style requests
    executed_ts_list = tool_names_from_messages(messages, "SimulatedTaskScheduler")
    executed_ts = set(executed_ts_list)
    assert (
        "SimulatedTaskScheduler_execute" not in executed_ts
    ), f"TaskScheduler.execute must not run for sandbox scenarios, saw: {sorted(executed_ts)}"

    # If the assistant explicitly requested tools, they should reference Actor.act for this scenario
    requested_actor = set(assistant_requested_tool_names(messages, "SimulatedActor"))
    if requested_actor:
        assert requested_actor <= {
            "SimulatedActor_act",
        }, f"Assistant should only request SimulatedActor_act here, saw: {sorted(requested_actor)}"
