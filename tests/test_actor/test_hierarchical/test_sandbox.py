"""Sandbox isolation & scope preservation tests for HierarchicalActor."""

import asyncio
import textwrap
from unittest.mock import AsyncMock

import pytest

from unity.actor.hierarchical_actor import (
    HierarchicalActor,
    HierarchicalActorHandle,
    ImplementationDecision,
)
from unity.function_manager.computer_backends import (
    MockComputerBackend,
    VALID_MOCK_SCREENSHOT_PNG,
)
from unity.function_manager.function_manager import FunctionManager

from tests.test_actor.test_hierarchical.helpers import (
    SimpleMockVerificationClient,
    wait_for_log_entry,
)

CANNED_PLAN_FOR_SANDBOX_TEST_SANDBOX_ISOLATION_AND_MERGE = textwrap.dedent(
    """
    async def main_plan():
        '''Searches for a recipe appropriate for today's weather.'''
        print("--- Main Plan: Navigating to allrecipes.com ---")
        await computer_primitives.navigate("https://www.allrecipes.com/")
        print("--- Main Plan: Pausing for interjection to get weather...")
        await asyncio.sleep(2)  # Reduced sleep for mocked test
        print("--- Main Plan: Original logic searching for 'soup' ---")
        await computer_primitives.act("Search for 'soup'")
        return "Original plan finished."
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_exploration_runs_in_isolated_sandbox_and_merges_results():
    fm = FunctionManager()
    fm.clear()

    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        function_manager=fm,
        can_store=False,
    )

    actor.computer_primitives._computer = MockComputerBackend(
        url="https://mock-url.com",
        screenshot=VALID_MOCK_SCREENSHOT_PNG,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value=None)

    async def mock_dynamic_implement(*args, **kwargs):
        _ = (args, kwargs)
        return ImplementationDecision(
            action="implement_function",
            reason="Re-implementing after course correction.",
            code="async def main_plan(): return 'Plan completed.'",
        )

    actor._dynamic_implement = mock_dynamic_implement

    active_task = None
    try:
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Find a recipe on allrecipes.com suitable for today's weather in Karachi.",
            persist=False,
        )

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = SimpleMockVerificationClient()
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_FOR_SANDBOX_TEST_SANDBOX_ISOLATION_AND_MERGE,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        # These tests assert against action_log (not stdout prints).
        await wait_for_log_entry(
            active_task,
            "Executing computer_primitives.navigate",
            timeout=30,
        )

        _ = await asyncio.wait_for(active_task.result(), timeout=30)

    finally:
        if active_task and not active_task.done():
            await active_task.stop()
        await actor.close()


CANNED_PLAN_FOR_CONTEXT_TEST_SCOPED_CONTEXT = textwrap.dedent(
    """
    async def grandchild_function():
        '''A nested function that accesses parent scope variable.'''
        print("Calling grandchild...")
        return f"grandchild ok: {some_value}"

    async def child_function():
        '''Calls grandchild.'''
        print("Calling child...")
        return await grandchild_function()

    async def parent_function():
        '''Defines a variable and calls child.'''
        print("Calling parent...")
        nonlocal_var = "ok"
        # Alias to a name we reference in grandchild
        global some_value
        some_value = nonlocal_var
        result = await child_function()
        return f"parent ok: {result}"

    async def main_plan():
        return await parent_function()
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_nested_functions_maintain_correct_scope_in_prompts():
    fm = FunctionManager()
    fm.clear()
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        function_manager=fm,
        can_store=False,
    )
    actor.computer_primitives._computer = MockComputerBackend(
        url="https://mock-url.com",
        screenshot=VALID_MOCK_SCREENSHOT_PNG,
    )
    actor.computer_primitives.act = AsyncMock(return_value="Mock action completed.")
    actor.computer_primitives.navigate = AsyncMock(return_value=None)

    active_task = None
    try:
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test that plan execution uses scoped context.",
            persist=False,
        )

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = SimpleMockVerificationClient()
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_FOR_CONTEXT_TEST_SCOPED_CONTEXT,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        await wait_for_log_entry(active_task, "main_plan", timeout=30)
        _ = await asyncio.wait_for(active_task.result(), timeout=30)

        final_code = active_task.plan_source_code
        assert "async def parent_function" in final_code
        assert "async def grandchild_function" in final_code
        assert "async def main_plan" in final_code

    finally:
        if active_task and not active_task.done():
            await active_task.stop()
        await actor.close()
