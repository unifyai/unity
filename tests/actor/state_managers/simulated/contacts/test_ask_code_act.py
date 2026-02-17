"""
CodeActActor routing tests for ContactManager.ask (simulated managers).

Mirrors `test_ask.py` but validates CodeActActor produces Python that calls
`primitives.contacts.ask(...)` (on-the-fly; no FunctionManager).
"""

from __future__ import annotations

import pytest

from tests.actor.state_managers.utils import make_code_act_actor

pytestmark = pytest.mark.eval


CONTACT_QUESTIONS: list[str] = [
    "Which of our contacts prefers to be contacted by phone? (use your contacts only).",
    "List any contacts located in Berlin. (use your contacts only).",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", CONTACT_QUESTIONS)
async def test_code_act_questions_use_only_contact_tool(
    question: str,
):
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )
        result = await handle.result()
        # Verify result is not None (routing test, not type test)
        assert result is not None

        # Routing: must hit contacts.ask for contact questions.
        assert calls, "Expected at least one state manager call."
        assert "primitives.contacts.ask" in set(calls), f"Calls seen: {calls}"
