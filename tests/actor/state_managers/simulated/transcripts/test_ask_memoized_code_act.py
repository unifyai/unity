"""
CodeActActor routing tests for TranscriptManager.ask with FunctionManager discovery
tools available (simulated managers).

Validates that even with FunctionManager search/filter/list tools exposed,
the LLM routes simple transcript queries via ``execute_function`` calling the
primitive directly.
"""

from __future__ import annotations

import pytest

from tests.actor.state_managers.utils import (
    assert_used_execute_function,
    make_code_act_actor,
)

pytestmark = pytest.mark.eval


TRANSCRIPT_QUESTIONS: list[str] = [
    "Show me the most recent message that mentions the Q3 budget.",
    "Find our last SMS with Sarah.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", TRANSCRIPT_QUESTIONS)
async def test_code_act_questions_use_execute_function_with_fm_tools(
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
        assert "primitives.transcripts.ask" in set(calls), f"Calls seen: {calls}"
