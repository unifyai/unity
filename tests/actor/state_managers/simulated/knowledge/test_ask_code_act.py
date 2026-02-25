"""
CodeActActor routing tests for KnowledgeManager.ask (simulated managers).

Validates that CodeActActor uses ``execute_function`` (not ``execute_code``)
for simple single-primitive knowledge queries.
"""

from __future__ import annotations

import pytest

from tests.actor.state_managers.utils import (
    assert_used_execute_function,
    make_code_act_actor,
)

pytestmark = pytest.mark.eval


KNOWLEDGE_QUESTIONS: list[str] = [
    "Summarise the employee onboarding policy. (use knowledge only).",
    "What are our office hours? (use knowledge only).",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", KNOWLEDGE_QUESTIONS)
async def test_code_act_questions_use_execute_function(
    question: str,
):
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )
        result = await handle.result()
        assert result is not None

        assert_used_execute_function(handle)
        assert "primitives.knowledge.ask" in set(calls), f"Calls seen: {calls}"
