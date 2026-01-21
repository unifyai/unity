"""Real WebSearcher tests for Actor.

Tests that Actor correctly calls real WebSearcher methods.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_actor.test_state_managers.utils import (
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

        # Assert result is non-empty string
        assert isinstance(result, str)
        assert len(result) > 0

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
        implementation = '''
async def search_web(query: str, response_format=None) -> str:
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
        query: The web search question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The answer from the web searcher as a string.
    """
    handle = await primitives.web.ask(query, response_format=response_format)
    result = await handle.result()
    return result
'''
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        # Call actor with natural language query requiring web search
        handle = await actor.act(
            "What is the Eisenhower Matrix and when should it be used? Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result is non-empty string
        assert isinstance(result, str)
        assert len(result) > 0

        # Assert memoized function was used
        assert_memoized_function_used(handle, "search_web")

        # Assert underlying primitive was called
        assert_tool_called(handle, "primitives.web.ask")
