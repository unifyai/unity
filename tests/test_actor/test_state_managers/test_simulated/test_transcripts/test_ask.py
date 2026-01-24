"""
Actor tests for TranscriptManager.ask operations.

Tests that HierarchicalActor correctly generates plans calling `primitives.transcripts.ask`
for read-only transcript queries.

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


TRANSCRIPT_QUESTIONS: list[str] = [
    "What did David say last week?",
    "Show me the most recent message that mentions the Q3 budget.",
    "List messages from Alice in the last 24 hours.",
    "Find our last SMS with Sarah.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", TRANSCRIPT_QUESTIONS)
async def test_questions_use_only_transcript_tool(
    question: str,
    mock_verification,
):
    """Verify Actor generates plans calling primitives.transcripts.ask."""
    async with make_hierarchical_actor(impl="simulated") as actor:
        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        # Verify result is non-empty
        # Verify result is not None (routing test, not type test)
        assert result is not None

        # Verify plan was generated
        assert handle.plan_source_code
        assert "async def" in handle.plan_source_code
        assert "primitives." in handle.plan_source_code

        # Verify only primitives.transcripts.ask was called
        from tests.test_actor.test_state_managers.utils import (
            assert_tool_called,
        )

        assert_tool_called(handle, "primitives.transcripts.ask")
        # Verify that verification was bypassed (no verification failures in log).
        log_text = "\n".join(handle.action_log)
        assert "verification failed" not in log_text.lower()
