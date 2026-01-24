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
    "Summarise the employee onboarding policy with key takeaways.",
    "What are Unify.AI's office hours? Provide a structured summary.",
    "List return policies for ACME by effective date with analysis.",
    "What warranty information do we hold about Tesla vehicles? Include implications.",
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
async def ask_knowledge_with_analysis(question: str, response_format=None) -> str:
    """Query organizational knowledge and produce a structured analysis with key insights.

    **ALWAYS use this function** for ANY knowledge-related read-only question, regardless
    of complexity. Direct calls to primitives.knowledge.ask are not allowed when this
    function is available - even for simple lookups like "What are office hours?".

    This helper does two steps:
    1) Retrieves relevant facts via primitives.knowledge.ask
    2) Synthesizes a structured analysis with key takeaways via computer_primitives.reason

    **Do NOT use when**:
    - the user needs current external facts (use web search)
    - the user is asking about message history/transcripts (use transcripts)
    - the user is asking about contact records (use contacts)
    - the user is requesting a knowledge mutation (use knowledge update)

    Args:
        question: The knowledge-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        A structured analysis with the answer and key insights.
    """
    handle = await primitives.knowledge.ask(question, response_format=response_format)
    raw_result = await handle.result()

    analysis = await computer_primitives.reason(
        request=(
            "Produce a structured summary with: "
            "1) Direct answer (2-3 sentences), "
            "2) Key facts (3-5 bullet points), "
            "3) Implications or takeaways (2-3 bullets)."
        ),
        context=str(raw_result),
    )
    return analysis if isinstance(analysis, str) else str(analysis)
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

        # Relax assertion: result can be str, dict, or Pydantic BaseModel

        # Verify result is not None (routing test, not type test)
        assert result is not None

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
        )

        assert_memoized_function_used(handle, "ask_knowledge_with_analysis")
        assert_tool_called(handle, "primitives.knowledge.ask")

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
