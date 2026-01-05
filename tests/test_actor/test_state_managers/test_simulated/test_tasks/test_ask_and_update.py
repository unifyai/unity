"""
Actor tests for TaskScheduler combined ask+update operations.

This module ports tests from `tests/test_conductor/test_simulated/test_tasks/test_ask_and_update.py`
to verify that HierarchicalActor correctly generates plans calling both
`primitives.tasks.ask` and `primitives.tasks.update`, with an optional single
`primitives.contacts.ask` for resolving assignees.

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import make_actor

pytestmark = pytest.mark.eval


COMBINED_REQUESTS: list[str] = [
    (
        "Which tasks are due tomorrow? Also create a new task: Call Alice about the Q3 budget tomorrow at 09:00."
    ),
    (
        "List all high-priority tasks. Also update the priority of 'Draft Budget FY26' to high."
    ),
    (
        "What tasks are assigned to Bob Johnson? Also delete the task named 'Old Onboarding Checklist'."
    ),
    (
        "Summarise tasks scheduled for next week. Also set 'Prepare slides for kickoff' to start today at 10:00."
    ),
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("request_text", COMBINED_REQUESTS)
async def test_combined_queries_call_ask_and_update(
    request_text: str,
    mock_verification,
):
    """Verify Actor generates plans calling primitives.tasks.ask and primitives.tasks.update."""
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
            get_state_manager_tools,
        )

        assert_tool_called(handle, "primitives.tasks.ask")
        assert_tool_called(handle, "primitives.tasks.update")
        # Allow additional tools (e.g., verification steps)
        state_manager_tools = set(get_state_manager_tools(handle))
        assert "primitives.tasks.ask" in state_manager_tools
        assert "primitives.tasks.update" in state_manager_tools

        log_text = "\n".join(handle.action_log)
        assert "verification failed" not in log_text.lower()
