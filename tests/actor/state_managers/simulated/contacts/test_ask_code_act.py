"""
CodeActActor routing tests for ContactManager.ask (simulated managers).

Validates that CodeActActor uses ``execute_function`` (not ``execute_code``)
for simple single-primitive contact queries, ensuring the returned handle
is steerable from the outer loop.
"""

from __future__ import annotations

import pytest

from tests.actor.state_managers.utils import (
    assert_used_execute_function,
    make_code_act_actor,
)

pytestmark = pytest.mark.eval


CONTACT_QUESTIONS: list[str] = [
    "Which of our contacts prefers to be contacted by phone? (use your contacts only).",
    "List any contacts located in Berlin. (use your contacts only).",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", CONTACT_QUESTIONS)
async def test_code_act_questions_use_execute_function(
    question: str,
):
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )
        result = await handle.result()
        assert result is not None

        assert_used_execute_function(handle)
        assert "primitives.contacts.ask" in set(calls), f"Calls seen: {calls}"
