"""
Actor tests for GuidanceManager.update operations via memoized functions.

Pattern: Memoized function (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_actor

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_update_uses_memoized_function(
    mock_verification,
):
    implementation = '''
async def update_guidance(instruction: str, response_format=None) -> str:
    """Create/update/delete guidance entries via the guidance manager (mutation).

    **Use when** the user requests changes to internal guidance content: add a runbook,
    update an existing entry, or correct/replace guidance text.

    **How it works**: calls the guidance mutation tool:
    - `await primitives.guidance.update(instruction, response_format=response_format)`

    **Do NOT use when**:
    - the user is asking a read-only question about existing guidance (use `primitives.guidance.ask`)
    - the user is asking about transcripts, contacts, tasks, or current web facts

    Args:
        instruction: The guidance update instruction text.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The result from the guidance manager update operation as a string.
    """
    handle = await primitives.guidance.update(instruction, response_format=response_format)
    result = await handle.result()
    return result
'''
    async with make_actor(impl="simulated") as actor:
        from unity.function_manager.function_manager import FunctionManager

        fm = FunctionManager()

        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        request_text = (
            "Create a new guidance entry titled 'Runbook: DB Failover' with the content "
            "'Promote replica and update connection strings.'"
        )

        handle = await actor.act(
            f"{request_text} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        assert isinstance(result, str) and result.strip()

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
        )

        assert_memoized_function_used(handle, "update_guidance")
        assert_tool_called(handle, "primitives.guidance.update")
