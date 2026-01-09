"""
CodeActActor routing tests for TaskScheduler.update (simulated managers).

Mirrors `test_update.py` but validates CodeActActor produces Python that calls
`primitives.tasks.update(...)` (on-the-fly; no FunctionManager).
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import make_code_act_actor

pytestmark = pytest.mark.eval


UPDATE_QUERIES: list[str] = [
    "Create a new task: Call Alice about the Q3 budget tomorrow at 09:00.",
    "Delete the task named 'Old Onboarding Checklist'.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("request_text", UPDATE_QUERIES)
async def test_code_act_update_only_calls_update(
    request_text: str,
):
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{request_text} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )
        result = await handle.result()
        assert isinstance(result, str) and result.strip()

        assert calls, "Expected at least one state manager call."
        assert "primitives.tasks.update" in set(calls), f"Calls seen: {calls}"
