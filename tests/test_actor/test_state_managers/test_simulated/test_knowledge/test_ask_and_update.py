"""
Actor tests for KnowledgeManager combined ask+update operations.

Tests that HierarchicalActor correctly generates plans calling both
`primitives.knowledge.ask` and `primitives.knowledge.update`.

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


COMBINED_QUERIES: list[str] = [
    "Check knowledge for Unify.AI's office hours. Also store that the support inbox is monitored 7 days a week.",
    "Check knowledge for the onboarding policy for engineers at Unify.AI. Also record that a security review occurs in week two.",
    "Check knowledge for the refund policy for Unify.AI by date. Also record that exchanges are allowed within 45 days.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("combined_text", COMBINED_QUERIES)
async def test_combined_queries_call_ask_and_update(
    combined_text: str,
    mock_verification,
):
    """Verify Actor generates plans calling both primitives.knowledge.ask and update."""
    async with make_hierarchical_actor(impl="simulated") as actor:
        handle = await actor.act(
            f"{combined_text} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        # Result type is not part of the routing contract: plans may return structured objects.
        # We only require a non-empty result (stringifiable) and correct tool routing.
        assert result is not None and str(result).strip()

        assert handle.plan_source_code
        assert "async def" in handle.plan_source_code
        assert "primitives." in handle.plan_source_code

        from tests.test_actor.test_state_managers.utils import (
            assert_tool_called,
            get_state_manager_tools,
        )

        assert_tool_called(handle, "primitives.knowledge.ask")
        assert_tool_called(handle, "primitives.knowledge.update")

        # Verify both ask and update were called. Other tools may be used for context
        # (e.g., guidance.ask for policies, transcripts.ask for historical info).
        state_manager_tools = set(get_state_manager_tools(handle))
        assert (
            "primitives.knowledge.ask" in state_manager_tools
        ), f"Expected primitives.knowledge.ask to be called, saw: {state_manager_tools}"
        assert (
            "primitives.knowledge.update" in state_manager_tools
        ), f"Expected primitives.knowledge.update to be called, saw: {state_manager_tools}"

        log_text = "\n".join(handle.action_log)
        assert "verification failed" not in log_text.lower()
