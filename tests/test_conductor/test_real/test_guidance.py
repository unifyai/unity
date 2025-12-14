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
from unity.guidance_manager.guidance_manager import GuidanceManager


@pytest.mark.asyncio
@_handle_project
async def test_ask_calls_manager():
    # Seed a guidance entry via the real GuidanceManager
    gm = GuidanceManager()
    gm._add_guidance(
        title="Onboarding Overview",
        content="We walk through onboarding steps for new users.",
    )

    # Wire a SimulatedConductor to the real GuidanceManager instance
    cond = SimulatedConductor(guidance_manager=gm)

    # Ask a question that clearly targets Guidance content
    handle = await cond.request(
        "What does the Guidance entry titled 'Onboarding Overview' say?",
        _return_reasoning_steps=True,
    )
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    # Basic content check – answer should mention onboarding
    assert isinstance(answer, str) and "onboarding" in answer.lower()

    # Ensure GuidanceManager.ask was invoked (and nothing else from any manager)
    executed_list = tool_names_from_messages(messages, "GuidanceManager")
    requested_list = assistant_requested_tool_names(messages, "GuidanceManager")
    assert executed_list, "Expected at least one tool call"
    assert set(executed_list) == {
        "GuidanceManager_ask",
    }, f"Only GuidanceManager_ask should run; saw: {sorted(set(executed_list))}"
    assert (
        executed_list.count("GuidanceManager_ask") == 1
    ), f"Expected exactly one GuidanceManager_ask call, saw order: {executed_list}"
    assert set(requested_list) <= {
        "GuidanceManager_ask",
    }, f"Assistant should request only GuidanceManager_ask, saw: {sorted(set(requested_list))}"

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


@pytest.mark.asyncio
@_handle_project
async def test_update_calls_manager():
    gm = GuidanceManager()
    cond = SimulatedConductor(guidance_manager=gm)

    request_text = (
        "Create a new guidance entry titled 'Incident Response' with the content "
        "'Escalate sev-1 to on-call within 5 minutes.'"
    )
    handle = await cond.request(request_text, _return_reasoning_steps=True)
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Ensure only GuidanceManager.update was invoked (not ask) and nothing else
    executed_list = tool_names_from_messages(messages, "GuidanceManager")
    requested_list = assistant_requested_tool_names(messages, "GuidanceManager")
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

    # Verify the mutation took effect in the real GuidanceManager instance
    rows = gm._filter(filter="title == 'Incident Response'")
    assert rows and any(
        "sev-1" in (r.content or "").lower() or "on-call" in (r.content or "").lower()
        for r in rows
    )

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
