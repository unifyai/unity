"""
Actor tests for WebSearcher.ask operations.

Tests that HierarchicalActor correctly generates plans calling `primitives.web_search.ask`.

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


WEB_LIVE_QUESTIONS: list[str] = [
    "What is the weather in Berlin today?",
    "What are the major world news headlines this week?",
    "Did the UN Security Council approve the resolution yesterday?",
    "What notable AI research announcements were made this week?",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", WEB_LIVE_QUESTIONS)
async def test_live_events_use_only_web_tool(
    question: str,
    mock_verification,
):
    async with make_hierarchical_actor(impl="simulated") as actor:
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        # Don't wait for the entire plan to finish (it may over-plan and JIT extra helpers).
        # We only need to verify that routing reaches the web tool.
        from tests.test_actor.test_state_managers.utils import wait_for_tool_call

        await wait_for_tool_call(handle, "primitives.web.ask", timeout=60)
        assert handle.plan_source_code

        from tests.test_actor.test_state_managers.utils import (
            assert_tool_called,
        )

        assert_tool_called(handle, "primitives.web.ask")
