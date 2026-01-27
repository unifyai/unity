"""Real WebSearcher tests for Actor.

Tests that Actor correctly calls real WebSearcher methods.
"""

import textwrap

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    assert_memoized_function_used,
    assert_tool_called,
    get_state_manager_tools,
    make_hierarchical_actor,
)
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_searcher(mock_verification):
    """Test that Actor calls WebSearcher.ask for web queries."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real WebSearcher (no seeding needed - external search)
        ws = ManagerRegistry.get_web_searcher()

        # Call actor with natural language query requiring web search
        handle = await actor.act(
            "What is the Eisenhower Matrix and when should it be used?",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Verify result is not None (routing test, not type test)
        assert result is not None

        # Assert correct tool was called
        assert_tool_called(handle, "primitives.web.ask")

        # Assert only web tools were used
        state_manager_tools = get_state_manager_tools(handle)
        assert all("web" in tool for tool in state_manager_tools)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_searcher_memoized(mock_verification):
    """Test that Actor uses memoized function for web queries."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real WebSearcher (no seeding needed - external search)
        ws = ManagerRegistry.get_web_searcher()

        # Create FunctionManager and seed memoized function
        fm = FunctionManager()

        implementation = textwrap.dedent(
            '''
            async def search_web(query: str, response_format=None) -> str:
                """Create a concise, sourced briefing for external/public topics.

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
                    query: The web search question to ask.
                    response_format: Optional Pydantic model for structured output.

                Returns:
                    The answer from the web searcher as a string.
                """
                cleaned = " ".join(str(query).split()).strip()
                if not cleaned:
                    cleaned = "Provide a concise briefing on the topic."
                if not cleaned.endswith("?"):
                    cleaned = f"{cleaned}?"
                research_prompt = f"{cleaned} Provide reliable sources and include dates/numbers when available."
                research_handle = await primitives.web.ask(research_prompt)
                research_result = await research_handle.result()
                briefing_request = (
                    "Create a concise briefing for a new team lead with these sections: "
                    "1) Definition (2-3 sentences), "
                    "2) Four quadrants explained briefly, "
                    "3) Two practical workplace examples, "
                    "4) Common pitfalls to avoid, "
                    "5) Quick checklist for when to use it, "
                    "6) Sources (titles or URLs). "
                    "Keep it structured and actionable."
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
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        # Call actor with natural language query requiring web search
        handle = await actor.act(
            "Please create a concise briefing on the Eisenhower Matrix for a new team lead. "
            "I need a structured write-up with: a short definition, the four quadrants, "
            "two practical workplace examples, common pitfalls, a quick checklist for when to use it, "
            "and sources. Do not ask clarifying questions. Do not create any stubs. "
            "Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Verify result is not None (routing test, not type test)
        assert result is not None

        # Assert memoized function was used
        assert_memoized_function_used(handle, "search_web")

        # Assert underlying primitive was called
        assert_tool_called(handle, "primitives.web.ask")
