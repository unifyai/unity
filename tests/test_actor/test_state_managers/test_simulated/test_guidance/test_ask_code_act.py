"""
CodeActActor routing tests for GuidanceManager.ask (simulated managers).

Mirrors `test_ask.py` but validates CodeActActor produces Python that calls
`primitives.guidance.ask(...)` (on-the-fly; no FunctionManager).
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import make_code_act_actor

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_code_act_ask_uses_only_guidance_ask_tool():
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        question = "What guidance do you have for incident response?"
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )
        result = await handle.result()
        assert isinstance(result, str) and result.strip()

        assert calls, "Expected at least one state manager call."
        assert "primitives.guidance.ask" in set(calls), f"Calls seen: {calls}"
