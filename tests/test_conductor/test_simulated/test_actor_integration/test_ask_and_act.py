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


# Each query intentionally contains an unrelated read (ask) and an ad-hoc action (act)
COMBINED_REQUESTS: list[str] = [
    ("What tasks are due today? Also, open a browser so we can review them together."),
    (
        "List all high-priority tasks. After that, start a sandbox session to "
        "troubleshoot the new dashboard."
    ),
]


@pytest.fixture(autouse=True)
def _patch_simulated_actor_to_complete_instantly(monkeypatch):
    """
    Ensure the underlying SimulatedActor completes immediately (steps=0)
    so that handle.result() doesn't hang. This is critical for any
    test that involves calling Actor_act or TaskScheduler_execute.
    """
    import unity.actor.simulated as _actor_sim

    _orig_sim_actor = _actor_sim.SimulatedActor

    @functools.wraps(_orig_sim_actor)
    def _patched_sim_actor(*args, **kwargs):  # type: ignore
        kw = dict(kwargs)
        kw.setdefault("steps", 0)
        kw.setdefault("duration", None)
        return _orig_sim_actor(*args, **kw)

    monkeypatch.setattr(_actor_sim, "SimulatedActor", _patched_sim_actor, raising=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text", COMBINED_REQUESTS)
@_handle_project
async def test_actor_combined_ask_and_act_only_expected_tools(
    request_text: str,
):
    cond = SimulatedConductor(
        description=(
            "Assistant that can both read from the task list and start "
            "ad-hoc actor sessions from a single request."
        ),
    )

    handle = await cond.request(
        request_text,
        _return_reasoning_steps=True,
    )

    # We can await the result directly because the
    # _patch_simulated_actor_to_complete_instantly fixture ensures
    # the Actor_act tool returns immediately.
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Get all executed tools, not just from one manager
    executed_list = tool_names_from_messages(messages)
    executed = set(executed_list)

    # Get all requested tools
    requested_list = assistant_requested_tool_names(messages)
    requested = set(requested_list)

    # 1. Verify that both expected tools were EXECUTED
    assert executed, "Expected at least some tool activity"
    assert {
        "TaskScheduler_ask",
        "Actor_act",
    }.issubset(
        executed,
    ), f"Expected both TaskScheduler_ask and Actor_act to run; saw: {sorted(executed)}"

    # 2. Verify that NO other conflicting or unexpected tools ran
    assert "TaskScheduler_execute" not in executed, "TaskScheduler_execute must not run"
    assert "ContactManager_ask" not in executed, "ContactManager must not run"

    # 3. Verify that the assistant's PLAN (requested tools) was correct
    assert requested, "Assistant should have requested tools"
    assert {
        "TaskScheduler_ask",
        "Actor_act",
    }.issubset(
        requested,
    ), f"Expected both TaskScheduler_ask and Actor_act to be requested; saw: {sorted(requested)}"
