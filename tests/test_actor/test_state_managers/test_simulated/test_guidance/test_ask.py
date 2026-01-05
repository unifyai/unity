"""
Actor tests for GuidanceManager.ask operations.

Ports tests from `tests/test_conductor/test_simulated/test_guidance/test_ask.py` to verify
that HierarchicalActor correctly generates plans calling `primitives.guidance.ask`.

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_actor

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_ask_uses_only_guidance_ask_tool(
    mock_verification,
):
    async with make_actor(impl="simulated") as actor:
        question = "What guidance do you have for incident response?"
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        assert isinstance(result, str) and result.strip()
        assert handle.plan_source_code

        from tests.test_actor.test_state_managers.utils import (
            assert_tool_called,
        )

        assert_tool_called(handle, "primitives.guidance.ask")
