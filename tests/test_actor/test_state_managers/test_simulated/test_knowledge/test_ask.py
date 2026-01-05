"""
Actor tests for KnowledgeManager.ask operations.

This module ports tests from `tests/test_conductor/test_simulated/test_knowledge/test_ask.py`
to verify that HierarchicalActor correctly generates plans calling `primitives.knowledge.ask`
for read-only knowledge queries.

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_actor

pytestmark = pytest.mark.eval


KNOWLEDGE_QUESTIONS: list[str] = [
    "Summarise the employee onboarding policy.",
    "What are our office hours?",
    "List return policies for ACME by effective date.",
    "What warranty information do we hold about Tesla vehicles?",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", KNOWLEDGE_QUESTIONS)
async def test_questions_use_only_knowledge_tool(
    question: str,
    mock_verification,
):
    """Verify Actor generates plans calling only primitives.knowledge.ask."""
    async with make_actor(impl="simulated") as actor:
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        assert isinstance(result, str) and result.strip()

        assert handle.plan_source_code
        assert "async def" in handle.plan_source_code
        assert "primitives." in handle.plan_source_code

        from tests.test_actor.test_state_managers.utils import (
            assert_tool_called,
        )

        assert_tool_called(handle, "primitives.knowledge.ask")

        log_text = "\n".join(handle.action_log)
        assert "verification failed" not in log_text.lower()
