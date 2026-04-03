"""
CodeActActor routing tests for KnowledgeManager.update with FunctionManager discovery
tools available (simulated managers).

Validates that even with FunctionManager search/filter/list tools exposed,
the LLM routes simple knowledge mutations via ``execute_function`` calling the
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
    "Store: Office hours are 9–5 PT.",
    "Update the onboarding policy to require security training in week one.",
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
        assert "primitives.knowledge.update" in set(calls), f"Calls seen: {calls}"
