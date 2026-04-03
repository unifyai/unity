"""
CodeActActor routing tests for WebSearcher.ask (simulated managers).

Validates that CodeActActor uses ``execute_function`` (not ``execute_code``)
for simple single-primitive web search queries.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.actor.state_managers.utils import (
    assert_used_execute_function,
    make_code_act_actor,
    wait_for_recorded_primitives_call,
)

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


WEB_LIVE_QUESTIONS: list[str] = [
    "What is the weather in Berlin today?",
    "What are the major world news headlines this week?",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", WEB_LIVE_QUESTIONS)
async def test_code_act_live_events_use_execute_function(
    question: str,
):
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )

        await wait_for_recorded_primitives_call(
            calls,
            "primitives.web.ask",
            timeout=60.0,
        )
        try:
            await asyncio.wait_for(handle.stop("Routing verified"), timeout=30.0)
        except Exception:
            pass

        assert_used_execute_function(handle)
        assert "primitives.web.ask" in set(calls), f"Calls seen: {calls}"
