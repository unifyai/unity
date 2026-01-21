"""
Actor tests for KnowledgeManager.ask via memoized functions.

This verifies that when `can_compose=True`, HierarchicalActor selects an existing
FunctionManager skill via semantic search, injects it as the entrypoint, and that
the underlying primitive tool call is `primitives.knowledge.ask`.

Pattern: Memoized function (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


KNOWLEDGE_QUESTIONS: list[str] = [
    "Summarise the employee onboarding policy.",
    "What are Unify.AI's office hours?",
    "List return policies for ACME by effective date.",
    "What warranty information do we hold about Tesla vehicles?",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", KNOWLEDGE_QUESTIONS)
async def test_questions_use_memoized_function(
    question: str,
    mock_verification,
):
    """Verify Actor selects memoized function via semantic search for asks."""

    implementation = '''
async def ask_knowledge(question: str, response_format=None) -> str:
    """Query internal structured knowledge via the knowledge manager (read-only).

    **Use when** the question should be answered from stored organizational knowledge:
    policies, facts, reference material, and previously recorded information.

    **How it works**: calls:
    - `await primitives.knowledge.ask(question, response_format=response_format)`

    **Do NOT use when**:
    - the user needs current external facts (use `primitives.web.ask`)
    - the user is asking about message history/transcripts (use `primitives.transcripts.ask`)
    - the user is asking about contact records (use `primitives.contacts.ask`)
    - the user is requesting a knowledge mutation (use `primitives.knowledge.update`)

    Args:
        question: The knowledge-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The answer from the knowledge manager as a string.
    """
    handle = await primitives.knowledge.ask(question, response_format=response_format)
    result = await handle.result()
    return result
'''
    async with make_hierarchical_actor(impl="simulated") as actor:
        from unity.function_manager.function_manager import FunctionManager

        fm = FunctionManager()

        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        assert isinstance(result, str) and result.strip()

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
        )

        assert_memoized_function_used(handle, "ask_knowledge")
        assert_tool_called(handle, "primitives.knowledge.ask")

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
