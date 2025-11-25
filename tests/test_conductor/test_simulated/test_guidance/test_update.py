from __future__ import annotations

import asyncio

import pytest

from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)
from unity.conductor.simulated import SimulatedConductor
from unity.guidance_manager.simulated import SimulatedGuidanceManager


MANAGER = "GuidanceManager"


@pytest.mark.eval
@pytest.mark.asyncio
@_handle_project
async def test_update_calls_manager():
    sim_gm = SimulatedGuidanceManager(
        description=(
            "Invent a consistent workplace scenario and accept mutation requests for guidance entries."
        ),
        rolling_summary_in_prompts=True,
        simulation_guidance=(
            "Be ready to add or update guidance titled 'Deployment Checklist' and 'Incident Response'."
        ),
    )

    cond = SimulatedConductor(
        description=(
            "Assistant that performs guidance updates via the GuidanceManager; other managers exist "
            "but are not needed for this request."
        ),
        guidance_manager=sim_gm,
    )

    request_text = (
        "Create a new guidance entry titled 'Runbook: DB Failover' with the content "
        "'Promote replica and update connection strings.'"
    )
    handle = await cond.request(request_text, _return_reasoning_steps=True)
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Ensure only GuidanceManager.update was invoked and not ask
    executed_list = tool_names_from_messages(messages, MANAGER)
    requested_list = assistant_requested_tool_names(messages, MANAGER)
    assert executed_list, "Expected at least one tool call"
    assert (
        executed_list[0] == "GuidanceManager_update"
    ), f"First call must be GuidanceManager_update, saw order: {executed_list}"
    assert set(executed_list) <= {
        "GuidanceManager_update",
    }, f"Only GuidanceManager_update should run, saw: {sorted(set(executed_list))}"
    assert "GuidanceManager_ask" not in set(
        executed_list,
    ), f"GuidanceManager_ask must not run, saw: {sorted(set(executed_list))}"
    assert set(requested_list) <= {
        "GuidanceManager_update",
    }, f"Assistant should request only GuidanceManager_update, saw: {sorted(set(requested_list))}"

    # Global exclusivity: verify no other manager tools ran
    all_tool_names = [
        str(m.get("name"))
        for m in messages
        if m.get("role") == "tool"
        and not str(m.get("name") or "").startswith("check_status_")
    ]
    assert all_tool_names, "Expected at least one tool call overall"
    assert all(
        n.startswith("GuidanceManager_update")
        or n.startswith("continue_GuidanceManager_update")
        for n in all_tool_names
    ), f"Unexpected tools executed: {sorted(set(all_tool_names))}"
    assert (
        len(all_tool_names) == 1
    ), f"Only one tool call expected; saw: {all_tool_names}"
