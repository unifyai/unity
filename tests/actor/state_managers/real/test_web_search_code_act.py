"""Real WebSearcher routing tests for CodeActActor.

Validates that CodeActActor uses ``execute_function`` for simple single-primitive
web search operations, both with and without FunctionManager discovery tools.
"""

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    assert_used_execute_function,
    make_code_act_actor,
)
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_searcher():
    """CodeAct routes web question via execute_function."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        _ws = ManagerRegistry.get_web_searcher()

        handle = await actor.act(
            "What is the Eisenhower Matrix and when should it be used?",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result is not None
        assert_used_execute_function(handle)
        assert "primitives.web.ask" in calls
        assert all(c.startswith("primitives.web.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_searcher_with_fm_tools():
    """CodeAct routes web query via execute_function even with FM discovery tools present."""
    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
    ) as (actor, _primitives, calls):
        _ws = ManagerRegistry.get_web_searcher()

        handle = await actor.act(
            "What is the Eisenhower Matrix and when should it be used?",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result is not None
        assert_used_execute_function(handle)
        assert "primitives.web.ask" in calls
        assert all(c.startswith("primitives.web.") for c in calls)
