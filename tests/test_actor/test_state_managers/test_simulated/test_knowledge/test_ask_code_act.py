"""
CodeActActor routing tests for KnowledgeManager.ask (simulated managers).

Mirrors `test_ask.py` but validates CodeActActor produces Python that calls
`primitives.knowledge.ask(...)` (on-the-fly; no FunctionManager).
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import make_code_act_actor

pytestmark = pytest.mark.eval


KNOWLEDGE_QUESTIONS: list[str] = [
    "Summarise the employee onboarding policy. (use knowledge only).",
    "What are our office hours? (use knowledge only).",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", KNOWLEDGE_QUESTIONS)
async def test_code_act_questions_use_only_knowledge_ask(
    question: str,
):
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )
        result = await handle.result()
        # Verify result is not None (routing test, not type test)
        assert result is not None

        assert calls, "Expected at least one state manager call."
        assert "primitives.knowledge.ask" in set(calls), f"Calls seen: {calls}"
