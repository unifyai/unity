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
    "Check knowledge for Unify.AI's office hours. Also store that the support inbox is monitored 7 days a week.",
    "Check knowledge for the onboarding policy for engineers at Unify.AI. Also record that a security review occurs in week two.",
    "Check knowledge for the refund policy for Unify.AI by date. Also record that exchanges are allowed within 45 days.",
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
''',
        '''
async def update_or_create_or_delete_knowledge(instruction: str, response_format=None) -> str:
    """Mutate internal knowledge via the knowledge manager (create/update facts).

    **Use when** the user requests to store new knowledge, update an existing policy/fact,
    or otherwise change the knowledge base.

    **How it works**: calls:
    - `await primitives.knowledge.update(instruction, response_format=response_format)`

    **Do NOT use when**:
    - the request is read-only (use `primitives.knowledge.ask`)
    - the user is asking about transcripts, contacts, tasks, guidance, or web facts

    Args:
        instruction: The knowledge update instruction text.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The result from the knowledge manager update operation as a string.
    """
    handle = await primitives.knowledge.update(instruction, response_format=response_format)
    result = await handle.result()
    return result
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

        assert isinstance(result, str) and result.strip()

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
            get_state_manager_tools,
        )

        assert_memoized_function_used(handle, "ask_knowledge")
        assert_memoized_function_used(handle, "update_or_create_or_delete_knowledge")
        assert_tool_called(handle, "primitives.knowledge.ask")
        assert_tool_called(handle, "primitives.knowledge.update")

        # Allow additional tools (e.g., verification steps)
        state_manager_tools = set(get_state_manager_tools(handle))
        assert "primitives.knowledge.ask" in state_manager_tools
        assert "primitives.knowledge.update" in state_manager_tools

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
