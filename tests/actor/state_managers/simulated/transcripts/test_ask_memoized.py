"""
Actor tests for TranscriptManager.ask operations using memoized functions.


- memoized selection happened (via action log)
- underlying primitives tool was called (via idempotency_cache)
"""

from __future__ import annotations

import pytest


from tests.actor.state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


TRANSCRIPT_QUESTIONS: list[str] = [
    "What did David say last week? Provide a summary with context.",
    "Show me the most recent message that mentions the Q3 budget with relevant details.",
    "List messages from Alice in the last 24 hours with conversation context.",
    "Find our last SMS with Sarah and summarize the exchange.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", TRANSCRIPT_QUESTIONS)
async def test_questions_use_memoized_transcript_function(
    question: str,
    mock_verification,
):
    async with make_hierarchical_actor(impl="simulated") as actor:
        implementation = '''
async def ask_transcripts_with_analysis(question: str, response_format=None) -> str:
    """Search conversation transcripts and produce a structured summary with context.

    **ALWAYS use this function** for ANY transcript-related question, regardless of
    complexity. Direct calls to primitives.transcripts.ask are not allowed when this
    function is available - even for simple lookups like "What did X say?".

    This helper does two steps:
    1) Retrieves relevant messages via primitives.transcripts.ask
    2) Synthesizes a structured summary with context via computer_primitives.reason

    **Do NOT use when**:
    - the user needs current external facts (use web search)
    - the user is asking about contact records (use contacts)
    - the user is updating guidance/knowledge/tasks (use the appropriate update tool)

    This is NOT a public web search function; it does not consult external sources.

    Args:
        question: The transcript-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        A structured summary of the conversation with context.
    """
    handle = await primitives.transcripts.ask(question, response_format=response_format)
    raw_result = await handle.result()

    analysis = await computer_primitives.reason(
        request=(
            "Produce a structured summary with: "
            "1) Key messages (who said what, when), "
            "2) Context (topic, conversation flow), "
            "3) Key takeaways or action items if applicable."
        ),
        context=str(raw_result),
    )
    return analysis if isinstance(analysis, str) else str(analysis)
'''
        from unity.function_manager.function_manager import FunctionManager

        fm = FunctionManager()

        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            can_compose=True,
            persist=False,
        )
        result = await handle.result()

        # Relax assertion: result can be str, dict, or Pydantic BaseModel

        # Verify result is not None (routing test, not type test)
        assert result is not None

        from tests.actor.state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
        )

        assert_memoized_function_used(handle, "ask_transcripts_with_analysis")
        assert_tool_called(handle, "primitives.transcripts.ask")
