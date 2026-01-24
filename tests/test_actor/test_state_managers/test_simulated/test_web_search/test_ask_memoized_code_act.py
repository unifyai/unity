"""
CodeActActor memoized-function routing tests for WebSearcher.ask (simulated managers).

Mirrors `test_ask_memoized.py` but validates CodeActActor:
1) Uses FunctionManager search tooling (single-call + auto-injection)
2) Executes the injected function
3) Ultimately invokes `primitives.web.ask(...)` via that memoized function.
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import (
    assert_code_act_function_manager_used,
    extract_code_act_execute_code_snippets,
    make_code_act_actor,
)

pytestmark = pytest.mark.eval


WEB_LIVE_QUESTIONS: list[str] = [
    "What is the weather in Berlin today?",
    "What are the major world news headlines this week?",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", WEB_LIVE_QUESTIONS)
async def test_code_act_live_events_use_memoized_function(
    question: str,
):
    from unity.function_manager.function_manager import FunctionManager

    implementation = '''
async def ask_web(question: str, response_format=None) -> str:
    """Use live web search for time-sensitive, external-information questions."""
    handle = await primitives.web.ask(question, response_format=response_format)
    result = await handle.result()
    return result
'''

    fm = FunctionManager()
    fm.add_functions(implementations=implementation, overwrite=True)

    async with make_code_act_actor(
        impl="simulated",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )
        result = await handle.result()
        # Verify result is not None (routing test, not type test)
        assert result is not None

        assert_code_act_function_manager_used(handle)

        code_snippets = "\n\n".join(
            extract_code_act_execute_code_snippets(handle),
        )
        assert "ask_web" in code_snippets, (
            "Expected CodeAct to execute the memoized function in Python code. "
            f"Snippets tail:\n{code_snippets[-800:]}"
        )

        assert "primitives.web.ask" in set(calls), f"Calls seen: {calls}"
