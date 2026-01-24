"""
Actor tests for GuidanceManager.update operations via memoized functions.

Pattern: Memoized function (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_update_uses_memoized_function(
    mock_verification,
):
    implementation = '''
async def update_guidance_with_confirmation(instruction: str, response_format=None) -> str:
    """Mutate guidance entries and produce a confirmation summary of changes.

    **ALWAYS use this function** for ANY guidance mutation request, regardless of
    complexity. Direct calls to primitives.guidance.update are not allowed when this
    function is available - even for simple updates like "Create guidance entry X".

    This helper does two steps:
    1) Performs the guidance mutation via primitives.guidance.update
    2) Synthesizes a confirmation summary via computer_primitives.reason

    **Do NOT use when**:
    - the user is asking a read-only question about existing guidance (use guidance ask)
    - the user is asking about transcripts, contacts, tasks, or current web facts

    Args:
        instruction: The guidance update instruction text.
        response_format: Optional Pydantic model for structured output.

    Returns:
        A confirmation summary of the changes made.
    """
    handle = await primitives.guidance.update(instruction, response_format=response_format)
    raw_result = await handle.result()

    confirmation = await computer_primitives.reason(
        request=(
            "Summarize what was changed: "
            "1) Action taken (created/updated/deleted), "
            "2) Guidance entry details (title, content summary), "
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

        request_text = (
            "Create a new guidance entry titled 'Runbook: DB Failover' with the content "
            "'Promote replica and update connection strings.' Confirm what was created."
        )

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
        )

        assert_memoized_function_used(handle, "update_guidance_with_confirmation")
        assert_tool_called(handle, "primitives.guidance.update")
