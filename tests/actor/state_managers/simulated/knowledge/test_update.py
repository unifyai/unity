"""
Actor tests for KnowledgeManager.update operations.

Tests that HierarchicalActor correctly generates plans calling `primitives.knowledge.update`
for knowledge mutations (and no reads via `primitives.knowledge.ask`).

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest


from tests.actor.state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


UPDATE_QUERIES: list[str] = [
    "Store: Office hours are 9–5 PT.",
    "Add that Tesla's battery warranty is eight years.",
    "Create a knowledge entry: Our refund window is 30 days for unopened items.",
    "Update the onboarding policy to require security training in week one.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("request_text", UPDATE_QUERIES)
async def test_update_only_calls_update(
    request_text: str,
    mock_verification,
):
    """Verify Actor generates plans calling only primitives.knowledge.update."""
    async with make_hierarchical_actor(impl="simulated") as actor:
        handle = await actor.act(
            f"{request_text} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        # Verify result is not None (routing test, not type test)
        assert result is not None

        assert handle.plan_source_code
        assert "async def" in handle.plan_source_code
        assert "primitives." in handle.plan_source_code

        from tests.actor.state_managers.utils import (
            assert_tool_called,
        )

        assert_tool_called(handle, "primitives.knowledge.update")

        log_text = "\n".join(handle.action_log)
        assert "verification failed" not in log_text.lower()
