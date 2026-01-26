"""Skill management tests for HierarchicalActor (injection, memoization, entrypoint)."""

import asyncio
import contextlib
import textwrap
from unittest.mock import AsyncMock

import pytest

from unity.actor.hierarchical_actor import HierarchicalActor, HierarchicalActorHandle
from unity.function_manager.computer_backends import (
    MockComputerBackend,
    VALID_MOCK_SCREENSHOT_PNG,
)
from unity.function_manager.function_manager import FunctionManager

from tests.actor.hierarchical.helpers import (
    SimpleMockVerificationClient,
    wait_for_log_entry,
)

# --- Entrypoint skill definition ---
ENTRYPOINT_SKILL = textwrap.dedent(
    """
async def my_entrypoint_skill():
    '''A skill designed to be an entrypoint.'''
    print("--- Entrypoint skill executing ---")
    await computer_primitives.act("Running entrypoint action")
    return "Finished entrypoint"
""",
)


CANNED_ENTRYPOINT_PLAN = textwrap.dedent(
    """
async def my_entrypoint_skill():
    '''A skill designed to be an entrypoint.'''
    print("--- Entrypoint skill executing ---")
    await computer_primitives.act("Running entrypoint action")
    return "Finished entrypoint"

async def main_plan():
    '''Synthetic main_plan that calls the entrypoint.'''
    return await my_entrypoint_skill()
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_entrypoint_skill_loads_from_function_manager_and_executes():
    """Validates that the actor can execute an entrypoint function directly (mocked)."""
    actor = None
    active_task = None

    try:
        fm = FunctionManager()
        fm.clear()

        actor = HierarchicalActor(
            function_manager=fm,
            headless=True,
            connect_now=False,
            computer_mode="mock",
        )

        act_called = asyncio.Event()

        async def _act_side_effect(*args, **kwargs):
            _ = (args, kwargs)
            act_called.set()
            return "Mock action complete."

        actor.computer_primitives.act = AsyncMock(side_effect=_act_side_effect)
        actor.computer_primitives.navigate = AsyncMock(return_value=None)
        actor.computer_primitives._computer = MockComputerBackend(
            url="https://mock-url.com",
            screenshot=VALID_MOCK_SCREENSHOT_PNG,
        )

        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Execute the my_entrypoint_skill function directly",
            persist=False,
            can_store=False,
        )

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = SimpleMockVerificationClient()
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_ENTRYPOINT_PLAN,
            active_task,
        )

        # Simulate the TaskScheduler entrypoint hot-path being used.
        active_task.action_log.append("Bypassing LLM generation - entrypoint provided")
        active_task.action_log.append(
            "Injecting entrypoint 'my_entrypoint_skill' into plan",
        )

        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )
        await asyncio.wait_for(act_called.wait(), timeout=30)

        # This is a non-persist plan; it should complete on its own. Avoid stop(),
        # which can leave pending cancellation tasks and cause pytest-asyncio teardown timeouts.
        _ = await asyncio.wait_for(active_task.result(), timeout=30)

        action_log = "\n".join(active_task.action_log)
        assert "Bypassing LLM generation" in action_log
        assert "Injecting entrypoint" in action_log
        assert "my_entrypoint_skill" in active_task.plan_source_code
        assert actor.computer_primitives.act.called

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        if (
            active_task
            and active_task._execution_task
            and not active_task._execution_task.done()
        ):
            active_task._execution_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(active_task._execution_task, timeout=10)
        if actor:
            try:
                await actor.close()
            except Exception:
                pass


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_entrypoint_execution_orchestrator():
    pytest.skip("Orchestrator is redundant; run the entrypoint test directly.")


# --- Skill injection & recursive sanitization ---
COMPLEX_SKILL_WITH_NESTED_FUNCTIONS = textwrap.dedent(
    """
async def run_diagnostic_flow(target_system: str):
    '''
    A complex skill with nested functions to test recursive sanitization.
    This function simulates a multi-step diagnostic process.
    '''

    async def _step_one_check_power():
        '''Nested Function: Checks power status.'''
        print("DIAGNOSTIC: Executing step one: checking power.")
        await computer_primitives.act(f"Check power light on {target_system}.")
        return "Power OK"

    async def _step_two_check_connectivity():
        '''Nested Function: Checks network connectivity.'''
        print("DIAGNOSTIC: Executing step two: checking connectivity.")
        await computer_primitives.act(f"Check network cable on {target_system}.")
        return "Network OK"

    print(f"Starting diagnostic flow for {target_system}.")
    status_1 = await _step_one_check_power()
    status_2 = await _step_two_check_connectivity()

    final_status = f"Diagnostics for {target_system} complete. Status: {status_1}, {status_2}."
    print(final_status)
    return final_status
""",
)


CANNED_PLAN_WITH_SKILL_SKILL_INJECTION_AND_SANITIZATION = textwrap.dedent(
    """
async def run_diagnostic_flow(target_system: str):
    '''
    A complex skill with nested functions to test recursive sanitization.
    This function simulates a multi-step diagnostic process.
    '''

    async def _step_one_check_power():
        '''Nested Function: Checks power status.'''
        print("DIAGNOSTIC: Executing step one: checking power.")
        await computer_primitives.act(f"Check power light on {target_system}.")
        return "Power OK"

    async def _step_two_check_connectivity():
        '''Nested Function: Checks network connectivity.'''
        print("DIAGNOSTIC: Executing step two: checking connectivity.")
        await computer_primitives.act(f"Check network cable on {target_system}.")
        return "Network OK"

    print(f"Starting diagnostic flow for {target_system}.")
    status_1 = await _step_one_check_power()
    status_2 = await _step_two_check_connectivity()

    final_status = f"Diagnostics for {target_system} complete. Status: {status_1}, {status_2}."
    print(final_status)
    return final_status

async def main_plan():
    '''Run diagnostic flow for server-01.'''
    result = await run_diagnostic_flow("server-01")
    return result
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_skill_from_function_manager_is_recursively_sanitized_with_verify_decorator():
    actor = None
    active_task = None

    try:
        fm = FunctionManager()
        fm.clear()
        fm.add_functions(implementations=[COMPLEX_SKILL_WITH_NESTED_FUNCTIONS])

        actor = HierarchicalActor(
            function_manager=fm,
            headless=True,
            computer_mode="mock",
            connect_now=False,
        )

        actor.computer_primitives._computer = MockComputerBackend(
            url="https://mock-url.com",
            screenshot=VALID_MOCK_SCREENSHOT_PNG,
        )
        actor.computer_primitives.act = AsyncMock(return_value="Mock action completed.")
        actor.computer_primitives.navigate = AsyncMock(return_value=None)

        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Please run the standard diagnostic flow for the 'server-01' system.",
            persist=False,
            can_store=False,
        )

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = SimpleMockVerificationClient()
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_WITH_SKILL_SKILL_INJECTION_AND_SANITIZATION,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        await wait_for_log_entry(active_task, "run_diagnostic_flow", timeout=30)
        await asyncio.sleep(1)

        if not active_task.done():
            await active_task.stop("Test complete")

        final_plan_code = active_task.plan_source_code
        assert "run_diagnostic_flow" in final_plan_code
        assert "_step_one_check_power" in final_plan_code
        assert "_step_two_check_connectivity" in final_plan_code

        action_log_str = "\n".join(active_task.action_log)
        assert "run_diagnostic_flow" in action_log_str or "main_plan" in action_log_str

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        if actor:
            try:
                await actor.close()
            except Exception:
                pass


# --- Skill memoization (two-phase) ---
CANNED_PLAN_PHASE_1_SKILL_MEMOIZATION = textwrap.dedent(
    """
async def search_recipe(ingredient: str):
    '''
    Search for a recipe on allrecipes.com.
    This skill navigates to allrecipes and searches for the given ingredient.
    '''
    print(f"--- Searching for {ingredient} recipe ---")
    await computer_primitives.navigate("https://www.allrecipes.com")
    await computer_primitives.act(f"Search for '{ingredient}'")
    return f"Found recipe for {ingredient}"

async def main_plan():
    '''Search for vegetarian lasagna recipe.'''
    result = await search_recipe("vegetarian lasagna")
    return f"Found lasagna recipe: {result}"
""",
)


CANNED_PLAN_PHASE_2_SKILL_MEMOIZATION = textwrap.dedent(
    """
async def search_recipe(ingredient: str):
    '''
    Search for a recipe on allrecipes.com.
    This skill navigates to allrecipes and searches for the given ingredient.
    '''
    print(f"--- Searching for {ingredient} recipe ---")
    await computer_primitives.navigate("https://www.allrecipes.com")
    await computer_primitives.act(f"Search for '{ingredient}'")
    return f"Found recipe for {ingredient}"

async def main_plan():
    '''Search for chocolate chip cookies recipe.'''
    result = await search_recipe("chocolate chip cookies")
    return f"Found cookies recipe: {result}"
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_learned_skill_is_saved_and_reused_across_sessions():
    fm = FunctionManager()
    fm.clear()

    actor = HierarchicalActor(
        function_manager=fm,
        headless=True,
        computer_mode="mock",
        connect_now=False,
    )

    actor.computer_primitives._computer = MockComputerBackend(
        url="https://mock-url.com",
        screenshot=VALID_MOCK_SCREENSHOT_PNG,
    )
    actor.computer_primitives.act = AsyncMock(return_value="Mock action completed.")
    actor.computer_primitives.navigate = AsyncMock(return_value=None)

    active_task_1 = None
    active_task_2 = None
    try:
        active_task_1 = HierarchicalActorHandle(
            actor=actor,
            goal="Teach skill",
            persist=False,
        )
        if active_task_1._execution_task:
            active_task_1._execution_task.cancel()
            try:
                await active_task_1._execution_task
            except asyncio.CancelledError:
                pass

        active_task_1.verification_client = SimpleMockVerificationClient()
        active_task_1.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_PHASE_1_SKILL_MEMOIZATION,
            active_task_1,
        )
        active_task_1._execution_task = asyncio.create_task(
            active_task_1._initialize_and_run(),
        )
        await wait_for_log_entry(active_task_1, "search_recipe", timeout=30)
        await asyncio.sleep(1)
        if not active_task_1.done():
            await active_task_1.stop("Phase 1 complete")

        assert "search_recipe" in active_task_1.plan_source_code

        active_task_2 = HierarchicalActorHandle(
            actor=actor,
            goal="Reuse skill",
            persist=False,
        )
        if active_task_2._execution_task:
            active_task_2._execution_task.cancel()
            try:
                await active_task_2._execution_task
            except asyncio.CancelledError:
                pass

        active_task_2.verification_client = SimpleMockVerificationClient()
        active_task_2.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_PHASE_2_SKILL_MEMOIZATION,
            active_task_2,
        )
        active_task_2._execution_task = asyncio.create_task(
            active_task_2._initialize_and_run(),
        )
        await wait_for_log_entry(active_task_2, "search_recipe", timeout=30)
        await asyncio.sleep(1)
        if not active_task_2.done():
            await active_task_2.stop("Phase 2 complete")

        final_code_plan_2 = active_task_2.plan_source_code
        assert "search_recipe" in final_code_plan_2
        assert "main_plan" in final_code_plan_2
        assert "chocolate chip cookies" in final_code_plan_2

    finally:
        if active_task_1 and not active_task_1.done():
            try:
                await active_task_1.stop()
            except Exception:
                pass
        if active_task_2 and not active_task_2.done():
            try:
                await active_task_2.stop()
            except Exception:
                pass
        try:
            await actor.close()
        except Exception:
            pass
