"""Real WebSearcher routing tests for CodeActActor.

These mirror `test_web_search.py` but use CodeActActor (code-first tool loop).
"""

import pytest

from tests.helpers import _handle_project
from tests.test_actor.test_state_managers.utils import (
    assert_code_act_function_manager_used,
    extract_code_act_execute_code_snippets,
    make_code_act_actor,
)
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_searcher():
    """CodeAct routes web question → primitives.web.ask."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        _ws = ManagerRegistry.get_web_searcher()

        handle = await actor.act(
            "What is the Eisenhower Matrix and when should it be used?",
            clarification_enabled=False,
        )
        result = await handle.result()

        # Verify result is not None (routing test, not type test)
        assert result is not None
        assert "primitives.web.ask" in calls
        assert all(c.startswith("primitives.web.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_searcher_memoized():
    """CodeAct uses FunctionManager (when available) for web queries."""
    fm = FunctionManager()
    implementation = """
async def search_web(query: str, response_format=None) -> str:
    \"\"\"Use web search for time-sensitive, external information questions.\"\"\"
    handle = await primitives.web.ask(query, response_format=response_format)
    return await handle.result()
"""
    fm.add_functions(implementations=implementation, overwrite=True)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        _ws = ManagerRegistry.get_web_searcher()

        handle = await actor.act(
            "What is the Eisenhower Matrix and when should it be used?",
            clarification_enabled=False,
        )
        result = await handle.result()

        # Verify result is not None (routing test, not type test)
        assert result is not None

        assert_code_act_function_manager_used(handle)
        snippets = "\n\n".join(extract_code_act_execute_code_snippets(handle))
        assert "search_web" in snippets

        assert "primitives.web.ask" in calls
        assert all(c.startswith("primitives.web.") for c in calls)
