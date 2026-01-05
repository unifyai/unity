"""
Actor tests for TranscriptManager.ask operations using memoized functions.


- memoized selection happened (via action log)
- underlying primitives tool was called (via idempotency_cache)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_actor

pytestmark = pytest.mark.eval


TRANSCRIPT_QUESTIONS: list[str] = [
    "What did David say last week?",
    "Show me the most recent message that mentions the Q3 budget.",
    "List messages from Alice in the last 24 hours.",
    "Find our last SMS with Sarah.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", TRANSCRIPT_QUESTIONS)
async def test_questions_use_memoized_transcript_function(
    question: str,
    mock_verification,
):
    async with make_actor(impl="simulated") as actor:
        implementation = '''
async def ask_transcripts_question(question: str, response_format=None) -> str:
    """Answer questions by searching YOUR conversation transcripts/messages (including SMS).

    **Use when** the user asks about *their own communication history*:
    - what someone said, when, and where (chat/SMS/email transcript content)
    - find the most recent message mentioning a topic
    - list messages from a person in a time window (e.g., "last 24 hours")
    - find the last SMS with a contact

    **How it works**: this function calls the transcripts state manager:
    - `await primitives.transcripts.ask(question, response_format=response_format)`

    **Do NOT use when**:
    - the user needs *current external facts* (use web search: `primitives.web.ask`)
    - the user is asking about contact records (use contacts: `primitives.contacts.ask`)
    - the user is updating guidance/knowledge/tasks (use the appropriate update tool)

    This is NOT a public web search function; it does not consult external sources.

    Args:
        question: The transcript-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The answer from the transcript manager as a string.
    """
    handle = await primitives.transcripts.ask(question, response_format=response_format)
    result = await handle.result()
    return result
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

        assert isinstance(result, str) and result.strip()

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
        )

        assert_memoized_function_used(handle, "ask_transcripts_question")
        assert_tool_called(handle, "primitives.transcripts.ask")
