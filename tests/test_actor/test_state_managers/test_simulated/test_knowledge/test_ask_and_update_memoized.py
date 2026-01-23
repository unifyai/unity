"""
Actor tests for KnowledgeManager ask+update via memoized functions.

This verifies that when `can_compose=True`, HierarchicalActor selects separate
memoized functions via semantic search for combined requests and that both
`primitives.knowledge.ask` and `primitives.knowledge.update` are invoked.

Pattern: Memoized functions (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


COMBINED_QUERIES: list[str] = [
    "Check knowledge for Unify.AI's office hours and provide a structured summary. Also store that the support inbox is monitored 7 days a week and confirm what was stored.",
    "Check knowledge for the onboarding policy for engineers at Unify.AI with key takeaways. Also record that a security review occurs in week two and confirm the change.",
    "Check knowledge for the refund policy for Unify.AI by date with analysis. Also record that exchanges are allowed within 45 days and provide confirmation.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("combined_text", COMBINED_QUERIES)
async def test_combined_queries_use_memoized_function(
    combined_text: str,
    mock_verification,
):
    """Verify Actor selects separate memoized functions via semantic search for combined ops."""

    implementations = [
        '''
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
''',
        '''
async def update_knowledge_with_confirmation(instruction: str, response_format=None) -> str:
    """Mutate organizational knowledge and produce a confirmation summary of changes.

    **ALWAYS use this function** for ANY knowledge mutation request, regardless of
    complexity. Direct calls to primitives.knowledge.update are not allowed when this
    function is available - even for simple updates like "Store: X is Y".

    This helper does two steps:
    1) Performs the knowledge mutation via primitives.knowledge.update
    2) Synthesizes a confirmation summary via computer_primitives.reason

    **Do NOT use when**:
    - the request is read-only (use knowledge ask)
    - the user is asking about transcripts, contacts, tasks, guidance, or web facts

    Args:
        instruction: The knowledge update instruction text.
        response_format: Optional Pydantic model for structured output.

    Returns:
        A confirmation summary of the changes made.
    """
    handle = await primitives.knowledge.update(instruction, response_format=response_format)
    raw_result = await handle.result()

    confirmation = await computer_primitives.reason(
        request=(
            "Summarize what was changed: "
            "1) Action taken (created/updated/deleted), "
            "2) Key details of the change, "
            "3) Confirmation that the operation completed."
        ),
        context=str(raw_result),
    )
    return confirmation if isinstance(confirmation, str) else str(confirmation)
''',
    ]
    async with make_hierarchical_actor(impl="simulated") as actor:
        from unity.function_manager.function_manager import FunctionManager

        fm = FunctionManager()

        fm.add_functions(implementations=implementations, overwrite=True)
        actor.function_manager = fm

        handle = await actor.act(
            f"{combined_text} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        # Relax assertion: result can be str, dict, or Pydantic BaseModel
        from pydantic import BaseModel

        assert result and (
            isinstance(result, (str, dict)) or isinstance(result, BaseModel)
        )

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
            get_state_manager_tools,
        )

        assert_memoized_function_used(handle, "ask_knowledge_with_analysis")
        assert_memoized_function_used(handle, "update_knowledge_with_confirmation")
        assert_tool_called(handle, "primitives.knowledge.ask")
        assert_tool_called(handle, "primitives.knowledge.update")

        # Allow additional tools (e.g., verification steps)
        state_manager_tools = set(get_state_manager_tools(handle))
        assert "primitives.knowledge.ask" in state_manager_tools
        assert "primitives.knowledge.update" in state_manager_tools

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
