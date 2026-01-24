"""
Actor tests for WebSearcher.ask operations via memoized functions.

Pattern: Memoized function (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import textwrap

import pytest

from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

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
    implementation = textwrap.dedent(
        '''
        async def ask_web(question: str, response_format=None) -> str:
            """Use live web search to produce a structured briefing for external/public topics.

            **Use when** the user asks about current events, general concepts, definitions,
            or anything that requires up-to-date public information from the internet
            and a clear, structured summary with sources.

            **Do NOT use when**:
            - the answer should come from *your own stored messages/transcripts* (use transcripts)
            - the answer should come from *your contacts* (use contacts)
            - the request is to *update guidance* (use guidance.update)
            - the request is to *create/update tasks* (use tasks.update)

            This helper does two steps:
            1) Web research via primitives.web.ask
            2) Structured briefing synthesis via computer_primitives.reason

            Args:
                question: The web search question to ask.
                response_format: Optional Pydantic model for structured output.

            Returns:
                The answer from the web searcher as a string.
            """
            cleaned = " ".join(str(question).split()).strip()
            if not cleaned:
                cleaned = "Provide a concise briefing on the topic."
            if not cleaned.endswith("?"):
                cleaned = f"{cleaned}?"
            research_prompt = f"{cleaned} Provide reliable sources and include dates/numbers when available."
            research_handle = await primitives.web.ask(research_prompt)
            research_result = await research_handle.result()
            briefing_request = (
                "Create a structured briefing with: "
                "1) Short answer (2-3 sentences), "
                "2) Key facts (3-5 bullets, include dates/numbers if available), "
                "3) Context/implications (2-3 bullets), "
                "4) Sources (2-4 citations with titles or URLs)."
            )
            briefing = await computer_primitives.reason(
                request=briefing_request,
                context=research_result,
            )
            if isinstance(briefing, str):
                return briefing.strip()
            return briefing
    ''',
    ).strip()
    async with make_hierarchical_actor(impl="simulated") as actor:
        from unity.function_manager.function_manager import FunctionManager

        fm = FunctionManager()

        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        handle = await actor.act(
            f"{question} Please provide a structured briefing with a short answer, key facts with dates, "
            "context/implications, and sources. Do not ask clarifying questions. Do not create any stubs. "
            "Generate the full plan. Proceed with the best interpretation of the request.",
            can_compose=True,
            persist=False,
        )
        result = await handle.result()

        # Verify result is not None (routing test, not type test)
        assert result is not None

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
        )

        assert_memoized_function_used(handle, "ask_web")
        assert_tool_called(handle, "primitives.web.ask")
