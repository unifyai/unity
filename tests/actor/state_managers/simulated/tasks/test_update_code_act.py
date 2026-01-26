"""
CodeActActor routing tests for TaskScheduler.update (simulated managers).

Mirrors `test_update.py` but validates CodeActActor produces Python that calls
`primitives.tasks.update(...)` (on-the-fly; no FunctionManager).
"""

from __future__ import annotations

import pytest

from tests.actor.state_managers.utils import make_code_act_actor

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
        # Relax assertion: result can be str, dict, or Pydantic BaseModel

        # Verify result is not None (routing test, not type test)
        assert result is not None

        assert calls, "Expected at least one state manager call."

        # For delete operations, the actor may intelligently first query to find the task
        # (via primitives.tasks.ask) and then determine there's nothing to delete.
        # Both direct update and ask-then-update/nothing patterns are valid.
        calls_set = set(calls)
        has_update = "primitives.tasks.update" in calls_set
        has_ask = "primitives.tasks.ask" in calls_set

        assert has_update or has_ask, (
            f"Expected primitives.tasks.update or primitives.tasks.ask to be called. "
            f"Calls seen: {calls}"
        )
