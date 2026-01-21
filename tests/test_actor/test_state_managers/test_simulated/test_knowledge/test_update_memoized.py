"""
Actor tests for KnowledgeManager.update via memoized functions.

This verifies that when `can_compose=True`, HierarchicalActor selects an existing
FunctionManager skill via semantic search, injects it as the entrypoint, and that
the underlying primitive tool call is `primitives.knowledge.update`.

Pattern: Memoized function (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


UPDATE_QUERIES: list[str] = [
    "Store: Office hours are 9–5 PT.",
    "Add that Tesla's battery warranty is eight years.",
    "Create a knowledge entry: Our refund window is 30 days for unopened items.",
    "Update the onboarding policy to require security training in week one.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("request_text", UPDATE_QUERIES)
async def test_updates_use_memoized_function(
    request_text: str,
    mock_verification,
):
    """Verify Actor selects memoized function via semantic search for updates."""

    implementation = '''
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
'''
    async with make_hierarchical_actor(impl="simulated") as actor:
        from unity.function_manager.function_manager import FunctionManager

        fm = FunctionManager()

        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        handle = await actor.act(
            f"{request_text} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        assert isinstance(result, str) and result.strip()

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
            get_state_manager_tools,
        )

        assert_memoized_function_used(handle, "update_or_create_or_delete_knowledge")
        assert_tool_called(handle, "primitives.knowledge.update")

        # Allow additional tools (e.g., verification steps calling read-only tools)
        state_manager_tools = set(get_state_manager_tools(handle))
        assert "primitives.knowledge.update" in state_manager_tools

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
