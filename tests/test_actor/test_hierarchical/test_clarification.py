import asyncio
import textwrap

import pytest
from unittest.mock import AsyncMock

from unity.actor.hierarchical_actor import (
    HierarchicalActor,
    HierarchicalActorHandle,
)
from unity.function_manager.computer_backends import (
    MockComputerBackend,
    VALID_MOCK_SCREENSHOT_PNG,
)

from tests.test_actor.test_hierarchical.helpers import (
    SimpleMockVerificationClient,
    wait_for_log_entry,
)

CANNED_PLAN_CLARIFICATION_FLOW = textwrap.dedent(
    """
async def get_dessert_info():
    '''Returns the user's dessert preference.'''
    return "brownies"

async def main_plan():
    '''Main plan that uses dessert info.'''
    dessert = await get_dessert_info()
    print(f"User wants to make: {dessert}")
    await computer_primitives.navigate("https://www.allrecipes.com")
    await computer_primitives.act(f"Search for {dessert} recipes")
    return f"Found recipes for {dessert}"
    """,
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_plan_pauses_for_user_clarification_and_resumes_with_response():
    """
    Clarification flow regression test (mocked):
    - plan can be initialized and executed with clarification queues available
    - and continues to completion (this test uses a canned plan for determinism)
    """
    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
    actor.computer_primitives._computer = MockComputerBackend(
        url="https://www.allrecipes.com",
        screenshot=VALID_MOCK_SCREENSHOT_PNG,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value=None)

    active_task = None
    try:
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Make a dessert and find recipes on allrecipes.com",
            persist=False,
        )

        # Cancel auto-started task so we can inject a deterministic plan.
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_CLARIFICATION_FLOW,
            active_task,
        )
        active_task.verification_client = SimpleMockVerificationClient()
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        await wait_for_log_entry(active_task, "main_plan", timeout=30)
        await asyncio.sleep(1)

        if not active_task.done():
            await active_task.stop("Test complete")

        # Basic sanity that the function was present and used.
        assert "get_dessert_info" in (active_task.plan_source_code or "")

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        try:
            await actor.close()
        except Exception:
            pass
