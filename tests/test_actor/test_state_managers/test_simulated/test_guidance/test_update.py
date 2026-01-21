"""
Actor tests for GuidanceManager.update operations.

Tests that HierarchicalActor correctly generates plans calling `primitives.guidance.update`.

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_update_uses_only_guidance_update_tool(
    mock_verification,
):
    async with make_hierarchical_actor(impl="simulated") as actor:
        request_text = (
            "Create a new guidance entry titled 'Runbook: DB Failover' with the content "
            "'Promote replica and update connection strings.'"
        )

        handle = await actor.act(
            f"{request_text} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        assert isinstance(result, str) and result.strip()
        assert handle.plan_source_code

        from tests.test_actor.test_state_managers.utils import (
            assert_tool_called,
            get_state_manager_tools,
        )

        assert_tool_called(handle, "primitives.guidance.update")

        # Verify primitives.guidance.update was called. Read-before-write patterns are allowed
        # (e.g., guidance.ask to check existing entries before updating).
        state_manager_tools = get_state_manager_tools(handle)
        assert state_manager_tools, "Expected at least one state manager tool call"
        assert "primitives.guidance.update" in set(
            state_manager_tools,
        ), f"Expected primitives.guidance.update to be called, saw: {state_manager_tools}"
        # Allow guidance.ask for read-before-write, but restrict to guidance tools only
        allowed_tools = {"primitives.guidance.ask", "primitives.guidance.update"}
        assert (
            set(state_manager_tools) <= allowed_tools
        ), f"Expected only guidance tools, saw: {state_manager_tools}"
