"""
CodeActActor routing tests for TranscriptManager.ask (simulated managers).

Mirrors `test_ask.py` but validates CodeActActor produces Python that calls
`primitives.transcripts.ask(...)` (on-the-fly; no FunctionManager).
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import make_code_act_actor

pytestmark = pytest.mark.eval


TRANSCRIPT_QUESTIONS: list[str] = [
    "Show me the most recent message that mentions the Q3 budget. (use transcripts only).",
    "Find our last SMS with Sarah. (use transcripts only).",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", TRANSCRIPT_QUESTIONS)
async def test_code_act_questions_use_only_transcript_ask(
    question: str,
):
    async with make_code_act_actor(impl="simulated") as (actor, _primitives, calls):
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )
        result = await handle.result()
        assert isinstance(result, str) and result.strip()

        assert calls, "Expected at least one state manager call."
        assert "primitives.transcripts.ask" in set(calls), f"Calls seen: {calls}"
