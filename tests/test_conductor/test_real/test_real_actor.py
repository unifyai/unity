from __future__ import annotations

import asyncio
import functools
import textwrap
import pytest
from unittest.mock import AsyncMock, MagicMock

from unity.conductor.simulated import SimulatedConductor
from unity.actor.hierarchical_actor import (
    HierarchicalActor,
    HierarchicalPlan,
    _HierarchicalPlanState,
    ImplementationDecision,
    InterjectionDecision,
    FunctionPatch,
    VerificationAssessment,
)

# Test helpers
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)

SANDBOX_REQUEST: str = (
    "Open a browser window so we can walk through the setup together."
)


@pytest.mark.asyncio
@_handle_project
async def test_real_conductor_actor_request_routes_to_actor_not_task(monkeypatch):
    """
    Validate Conductor.request routes sandbox-like requests to Actor.act (real actor),
    and does not execute TaskScheduler.execute. The test stops the actor early after
    scheduling to avoid external side-effects, then asserts routing via message log.
    """

    # Wrap HierarchicalActor.act to signal once scheduled so we can stop early
    _orig_act = HierarchicalActor.act

    tool_started_evt = asyncio.Event()

    @functools.wraps(_orig_act)
    async def _wrapped_act(self, *a, **kw):
        handle = await _orig_act(self, *a, **kw)
        tool_started_evt.set()
        return handle

    monkeypatch.setattr(HierarchicalActor, "act", _wrapped_act, raising=True)

    # Use real HierarchicalActor but configure it to avoid eager external connections
    actor = HierarchicalActor(
        browser_mode="legacy",  # legacy avoids running magnitude service
        headless=True,
        connect_now=False,  # lazy-init browser only if used
        timeout=30,
    )

    cond = SimulatedConductor(actor=actor)

    handle = await cond.request(
        SANDBOX_REQUEST,
        _return_reasoning_steps=True,
    )

    # Wait until the Actor tool has been scheduled, then stop to finish quickly
    await asyncio.wait_for(tool_started_evt.wait(), timeout=120)
    handle.stop()

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str)

    # Actor should be invoked at least once
    executed_actor_list = tool_names_from_messages(messages, "Actor")
    assert executed_actor_list, "Expected at least one tool call"
    assert (
        executed_actor_list.count("Actor_act") >= 1
    ), f"Expected Actor_act to run at least once, saw order: {executed_actor_list}"

    # TaskScheduler.execute must NOT be called for sandbox-style requests
    executed_ts_list = tool_names_from_messages(messages, "TaskScheduler")
    assert "TaskScheduler_execute" not in set(
        executed_ts_list,
    ), f"TaskScheduler.execute must not run for sandbox scenarios, saw: {sorted(set(executed_ts_list))}"

    # If assistant explicitly requested tools, it should reference Actor_act for this scenario
    requested_actor = set(assistant_requested_tool_names(messages, "Actor"))
    if requested_actor:
        assert requested_actor <= {
            "Actor_act",
        }, f"Assistant should only request Actor_act here, saw: {sorted(requested_actor)}"



