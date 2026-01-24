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
async def ask_guidance_with_analysis(question: str, response_format=None) -> str:
    """Query guidance and produce a structured analysis with actionable steps.

    **ALWAYS use this function** for ANY guidance-related read-only question, regardless
    of complexity. Direct calls to primitives.guidance.ask are not allowed when this
    function is available - even for simple lookups like "What's our policy on X?".

    This helper does two steps:
    1) Retrieves relevant guidance via primitives.guidance.ask
    2) Synthesizes a structured analysis with steps via computer_primitives.reason

    **Do NOT use when**:
    - the user wants to create/update guidance entries (use guidance update)
    - the user is asking about their message history/transcripts (use transcripts)
    - the user needs current external facts (use web)
    - the user is asking about contacts or tasks (use the appropriate manager)

    Args:
        question: The guidance-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        A structured analysis with actionable guidance.
    """
    handle = await primitives.guidance.ask(question, response_format=response_format)
    raw_result = await handle.result()

    analysis = await computer_primitives.reason(
        request=(
            "Produce a structured guidance summary with: "
            "1) Overview (what the guidance covers), "
            "2) Key steps or procedures (numbered list), "
            "3) Important considerations or warnings."
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

        question = (
            "What guidance do you have for incident response? Provide actionable steps."
        )
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

        assert_memoized_function_used(handle, "ask_guidance_with_analysis")
        assert_tool_called(handle, "primitives.guidance.ask")
