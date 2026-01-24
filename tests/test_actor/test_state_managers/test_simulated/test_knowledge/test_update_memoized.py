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
    "Store: Office hours are 9–5 PT. Confirm what was stored.",
    "Add that Tesla's battery warranty is eight years. Provide confirmation summary.",
    "Create a knowledge entry: Our refund window is 30 days for unopened items. Confirm the change.",
    "Update the onboarding policy to require security training in week one. Summarize what changed.",
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

        # Relax assertion: result can be str, dict, or Pydantic BaseModel

        # Verify result is not None (routing test, not type test)
        assert result is not None

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
            get_state_manager_tools,
        )

        assert_memoized_function_used(handle, "update_knowledge_with_confirmation")
        assert_tool_called(handle, "primitives.knowledge.update")

        # Allow additional tools (e.g., verification steps calling read-only tools)
        state_manager_tools = set(get_state_manager_tools(handle))
        assert "primitives.knowledge.update" in state_manager_tools

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
