"""
Actor tests for GuidanceManager.ask operations via memoized functions.

Pattern: Memoized function (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_ask_uses_memoized_function(
    mock_verification,
):
    implementation = '''
async def ask_guidance_question(question: str, response_format=None) -> str:
    """Query internal guidance/policies/runbooks via the guidance manager (read-only).

    **Use when** the question is about internal operating guidance, runbooks, incident
    response procedures, best practices, or other curated guidance content.

    **How it works**: calls the guidance read tool:
    - `await primitives.guidance.ask(question, response_format=response_format)`

    **Do NOT use when**:
    - the user wants to create/update guidance entries (use `primitives.guidance.update`)
    - the user is asking about their message history/transcripts (use `primitives.transcripts.ask`)
    - the user needs current external facts (use `primitives.web.ask`)
    - the user is asking about contacts or tasks (use the appropriate manager)

    Args:
        question: The guidance-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The answer from the guidance manager as a string.
    """
    handle = await primitives.guidance.ask(question, response_format=response_format)
    result = await handle.result()
    return result
'''
    async with make_hierarchical_actor(impl="simulated") as actor:
        from unity.function_manager.function_manager import FunctionManager

        fm = FunctionManager()

        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        question = "What guidance do you have for incident response?"
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

        assert_memoized_function_used(handle, "ask_guidance_question")
        assert_tool_called(handle, "primitives.guidance.ask")
