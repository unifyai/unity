from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.eval

from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)
from unity.conductor.simulated import SimulatedConductor
from unity.guidance_manager.simulated import SimulatedGuidanceManager


MANAGER = "GuidanceManager"


@pytest.mark.asyncio
@_handle_project
async def test_ask_calls_manager():
    sim_gm = SimulatedGuidanceManager(
        description=(
            "Invent a consistent workplace scenario with existing guidance entries, "
            "focusing on onboarding, deployment runbooks, and incident response."
        ),
        rolling_summary_in_prompts=True,
        simulation_guidance=(
            "Maintain plausible guidance items such as 'Onboarding Overview', "
            "'Deployment Checklist', and 'Incident Response'."
        ),
    )

    cond = SimulatedConductor(
        description=(
            "Assistant that answers questions about internal guidance; other managers exist "
            "but are not relevant for these queries."
        ),
        guidance_manager=sim_gm,
    )

    handle = await cond.request(
        "What guidance do you have for incident response?",
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # The only executed tool must be GuidanceManager.ask and it should run exactly once
    executed_list = tool_names_from_messages(messages, MANAGER)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"
    assert executed == {
        "GuidanceManager_ask",
    }, f"Only GuidanceManager_ask should run, saw: {sorted(executed)}"
    assert (
        executed_list.count("GuidanceManager_ask") == 1
    ), f"Expected exactly one GuidanceManager_ask call, saw order: {executed_list}"

    # Additionally confirm that any assistant tool selection(s) referenced only that tool
    requested = set(assistant_requested_tool_names(messages, MANAGER))
    assert requested, "Assistant should have requested at least one tool"
    assert requested <= {
        "GuidanceManager_ask",
    }, f"Assistant should request only GuidanceManager_ask, saw: {sorted(requested)}"

    # Global exclusivity: verify no other manager tools ran
    all_tool_names = [
        str(m.get("name"))
        for m in messages
        if m.get("role") == "tool"
        and not str(m.get("name") or "").startswith("check_status_")
    ]
    assert all_tool_names, "Expected at least one tool call overall"
    assert all(
        n.startswith("GuidanceManager_ask")
        or n.startswith("continue_GuidanceManager_ask")
        for n in all_tool_names
    ), f"Unexpected tools executed: {sorted(set(all_tool_names))}"
    assert (
        len(all_tool_names) == 1
    ), f"Only one tool call expected; saw: {all_tool_names}"
