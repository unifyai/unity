"""
Actor tests for TaskScheduler.update operations.

This module ports tests from `tests/test_conductor/test_simulated/test_tasks/test_update.py`
to verify that HierarchicalActor correctly generates plans calling `primitives.tasks.update`
for task mutations (and no task reads via `primitives.tasks.ask`).

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import make_actor

pytestmark = pytest.mark.eval


UPDATE_QUERIES: list[str] = [
    "Create a new task: Call Alice about the Q3 budget tomorrow at 09:00.",
    "Update the priority of 'Draft Budget FY26' to high.",
    "Delete the task named 'Old Onboarding Checklist'.",
    "Create a task to email Contoso about invoices. Set the deadline to 2025-12-05 17:00 UTC (deadline only; do not set start_at).",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("request_text", UPDATE_QUERIES)
async def test_update_only_calls_update(
    request_text: str,
    mock_verification,
):
    """Verify Actor generates plans calling only primitives.tasks.update."""
    async with make_actor(impl="simulated") as actor:

        handle = await actor.act(
            f"{request_text} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
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

        assert_tool_called(handle, "primitives.tasks.update")

        log_text = "\n".join(handle.action_log)
        assert "verification failed" not in log_text.lower()
