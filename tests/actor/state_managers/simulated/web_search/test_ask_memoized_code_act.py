"""
CodeActActor routing tests for WebSearcher.ask with FunctionManager discovery
tools available (simulated managers).

Validates that even with FunctionManager search/filter/list tools exposed,
the LLM routes simple web search queries via ``execute_function`` calling the
primitive directly.
"""

from __future__ import annotations

import pytest

from tests.actor.state_managers.utils import (
    assert_used_execute_function,
    make_code_act_actor,
)

pytestmark = pytest.mark.eval


WEB_LIVE_QUESTIONS: list[str] = [
    "What is the weather in Berlin today?",
    "What are the major world news headlines this week?",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", WEB_LIVE_QUESTIONS)
async def test_code_act_live_events_use_execute_function_with_fm_tools(
    question: str,
):
    async with make_code_act_actor(
        impl="simulated",
        include_function_manager_tools=True,
    ) as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )
        result = await handle.result()
        assert result is not None

        assert_used_execute_function(handle)
        assert "primitives.web.ask" in set(calls), f"Calls seen: {calls}"
