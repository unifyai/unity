"""
Actor tests for TaskScheduler.ask operations.

This module ports tests from `tests/test_conductor/test_simulated/test_tasks/test_ask.py`
to verify that HierarchicalActor correctly generates plans calling `primitives.tasks.ask`
for read-only task queries.

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import make_actor

pytestmark = pytest.mark.eval


TASK_QUESTIONS: list[str] = [
    "Which tasks are due today?",
    "List all high-priority tasks.",
    "What tasks are scheduled for tomorrow?",
    "Show tasks assigned to Alice that are still open.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", TASK_QUESTIONS)
async def test_questions_use_only_scheduler_tool(
    question: str,
    mock_verification,
):
    """Verify Actor generates plans calling only primitives.tasks.ask."""
    async with make_actor(impl="simulated") as actor:

        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        assert isinstance(result, str) and result.strip()

        assert handle.plan_source_code
        assert "async def" in handle.plan_source_code
        assert "primitives." in handle.plan_source_code

        from tests.test_actor.test_state_managers.utils import (
            assert_tool_called,
        )

        assert_tool_called(handle, "primitives.tasks.ask")

        log_text = "\n".join(handle.action_log)
        assert "verification failed" not in log_text.lower()
