"""
CodeActActor routing tests for TaskScheduler.update with FunctionManager discovery
tools available (simulated managers).

Validates that even with FunctionManager search/filter/list tools exposed,
the LLM routes simple task mutations via ``execute_function`` calling the
primitive directly.
"""

from __future__ import annotations

import pytest

from tests.actor.state_managers.utils import (
    assert_used_execute_function,
    make_code_act_actor,
)

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


UPDATE_QUERIES: list[str] = [
    "Create a new task: Call Alice about the Q3 budget tomorrow at 09:00.",
    "Delete the task named 'Old Onboarding Checklist'.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("request_text", UPDATE_QUERIES)
async def test_code_act_updates_use_execute_function_with_fm_tools(
    request_text: str,
):
    async with make_code_act_actor(
        impl="simulated",
        include_function_manager_tools=True,
    ) as (actor, _primitives, calls):
        handle = await actor.act(
            f"{request_text} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )
        result = await handle.result()
        assert result is not None

        assert_used_execute_function(handle)

        calls_set = set(calls)
        has_update = "primitives.tasks.update" in calls_set
        has_ask = "primitives.tasks.ask" in calls_set
        assert has_update or has_ask, (
            f"Expected primitives.tasks.update or primitives.tasks.ask to be called. "
            f"Calls seen: {calls}"
        )
