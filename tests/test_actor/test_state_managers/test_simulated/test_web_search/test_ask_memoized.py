"""
Actor tests for WebSearcher.ask operations via memoized functions.

Pattern: Memoized function (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_actor

pytestmark = pytest.mark.eval


WEB_LIVE_QUESTIONS: list[str] = [
    "What is the weather in Berlin today?",
    "What are the major world news headlines this week?",
    "Did the UN Security Council approve the resolution yesterday?",
    "What notable AI research announcements were made this week?",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", WEB_LIVE_QUESTIONS)
async def test_live_events_use_memoized_function(
    question: str,
    mock_verification,
):
    implementation = '''
async def ask_web(question: str, response_format=None) -> str:
    """Use live web search for time-sensitive, external-information questions.

    **Use when** the user asks about current events, current weather, latest headlines,
    or anything that requires up-to-date public information from the internet.

    **How it works**: this function calls the web-search state manager:
    - `await primitives.web.ask(question, response_format=response_format)`

    **Do NOT use when**:
    - the answer should come from *your own stored messages/transcripts* (use transcripts)
    - the answer should come from *your contacts* (use contacts)
    - the request is to *update guidance* (use guidance.update)
    - the request is to *create/update tasks* (use tasks.update)

    Args:
        question: The web search question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The answer from the web searcher as a string.
    """
    handle = await primitives.web.ask(question, response_format=response_format)
    result = await handle.result()
    return result
'''
    async with make_actor(impl="simulated") as actor:
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

        assert_memoized_function_used(handle, "ask_web")
        assert_tool_called(handle, "primitives.web.ask")
