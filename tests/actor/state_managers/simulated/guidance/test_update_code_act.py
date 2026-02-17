"""
CodeActActor routing tests for GuidanceManager.update (simulated managers).

Mirrors `test_update.py` but validates CodeActActor produces Python that calls
`primitives.guidance.update(...)` (on-the-fly; no FunctionManager).

Note: read-before-write (guidance.ask) is allowed, but only guidance tools.
"""

from __future__ import annotations

import pytest

from tests.actor.state_managers.utils import make_code_act_actor

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_code_act_update_uses_only_guidance_tools():
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        request_text = (
            "Create a new guidance entry titled 'Runbook: DB Failover' with the content "
            "'Promote replica and update connection strings.'"
        )

        handle = await actor.act(
            f"{request_text} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )
        result = await handle.result()
        # Verify result is not None (routing test, not type test)
        assert result is not None

        assert calls, "Expected at least one state manager call."
        assert "primitives.guidance.update" in set(calls), f"Calls seen: {calls}"

        allowed_tools = {"primitives.guidance.ask", "primitives.guidance.update"}
        assert (
            set(calls) <= allowed_tools
        ), f"Expected only guidance tools, saw: {calls}"
