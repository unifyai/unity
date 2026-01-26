"""Retrospective refactoring / skill generalization tests for HierarchicalActor."""

import asyncio
import textwrap
from unittest.mock import AsyncMock

import pytest

from unity.actor.hierarchical_actor import (
    CacheInvalidateSpec,
    FunctionPatch,
    HierarchicalActor,
    HierarchicalActorHandle,
    InterjectionDecision,
    _HierarchicalHandleState,
)
from unity.function_manager.computer_backends import (
    MockComputerBackend,
    VALID_MOCK_SCREENSHOT_PNG,
)

from tests.actor.hierarchical.helpers import (
    SimpleMockVerificationClient,
    wait_for_state,
)

INITIAL_PLAN = textwrap.dedent(
    """
async def main_plan():
    '''Initial empty plan waiting for instructions.'''
    return "Waiting for first instruction"
""",
)

PLAN_AFTER_INTERJECTION_1 = textwrap.dedent(
    """
async def search_recipe(ingredient: str):
    '''Search for a recipe on allrecipes.com.'''
    print(f"--- Navigating to allrecipes.com ---")
    await computer_primitives.navigate("https://www.allrecipes.com")
    print(f"--- Searching for {ingredient} ---")
    await computer_primitives.act(f"Search for '{ingredient}'")
    return f"Searched for {ingredient}"

async def main_plan():
    '''Search for chicken soup recipe.'''
    result = await search_recipe("chicken soup")
    return result
""",
)

PLAN_AFTER_INTERJECTION_2 = textwrap.dedent(
    """
async def search_recipe(ingredient: str):
    '''Search for a recipe on allrecipes.com.'''
    print(f"--- Navigating to allrecipes.com ---")
    await computer_primitives.navigate("https://www.allrecipes.com")
    print(f"--- Searching for {ingredient} ---")
    await computer_primitives.act(f"Search for '{ingredient}'")
    return f"Searched for {ingredient}"

async def get_recipe_summary(ingredient: str):
    '''Get a summary of the first recipe result.'''
    print(f"--- Clicking first result for {ingredient} ---")
    await computer_primitives.act("Click on the first search result")
    print(f"--- Getting recipe summary ---")
    await computer_primitives.act("Read and summarize the recipe")
    return f"Recipe summary for {ingredient}"

async def main_plan():
    '''Search for chicken soup recipe and get summary.'''
    await search_recipe("chicken soup")
    summary = await get_recipe_summary("chicken soup")
    return summary
""",
)

PLAN_AFTER_GENERALIZATION = textwrap.dedent(
    """
async def search_recipe(ingredient: str):
    '''Search for a recipe on allrecipes.com.'''
    print(f"--- Navigating to allrecipes.com ---")
    await computer_primitives.navigate("https://www.allrecipes.com")
    print(f"--- Searching for {ingredient} ---")
    await computer_primitives.act(f"Search for '{ingredient}'")
    return f"Searched for {ingredient}"

async def get_recipe_summary(ingredient: str):
    '''Get a summary of the first recipe result.'''
    print(f"--- Clicking first result for {ingredient} ---")
    await computer_primitives.act("Click on the first search result")
    print(f"--- Getting recipe summary ---")
    await computer_primitives.act("Read and summarize the recipe")
    return f"Recipe summary for {ingredient}"

async def main_plan():
    '''Search for chocolate chip cookies recipe and get summary.'''
    await search_recipe("chocolate chip cookies")
    summary = await get_recipe_summary("chocolate chip cookies")
    return summary
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_demonstration_is_generalized_into_reusable_parameterized_skill():
    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
    actor.computer_primitives._computer = MockComputerBackend(
        url="https://mock-url.com",
        screenshot=VALID_MOCK_SCREENSHOT_PNG,
    )
    actor.computer_primitives.act = AsyncMock(return_value="Mock action completed.")
    actor.computer_primitives.navigate = AsyncMock(return_value=None)

    active_task = None
    interjection_count = 0

    try:
        active_task = HierarchicalActorHandle(actor=actor, goal=None)

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = SimpleMockVerificationClient()

        def create_mock_modification_response(count: int) -> InterjectionDecision:
            if count == 1:
                return InterjectionDecision(
                    action="modify_task",
                    reason="Teaching navigation to allrecipes.com",
                    patches=[
                        FunctionPatch(
                            function_name="main_plan",
                            new_code=PLAN_AFTER_INTERJECTION_1,
                        ),
                    ],
                    cache=CacheInvalidateSpec(invalidate_steps=[]),
                )
            if count == 2:
                return InterjectionDecision(
                    action="modify_task",
                    reason="Teaching recipe summary step",
                    patches=[
                        FunctionPatch(
                            function_name="main_plan",
                            new_code=PLAN_AFTER_INTERJECTION_2,
                        ),
                    ],
                    cache=CacheInvalidateSpec(invalidate_steps=[]),
                )
            if count == 3:
                return InterjectionDecision(
                    action="modify_task",
                    reason="Generalizing for chocolate chip cookies",
                    patches=[
                        FunctionPatch(
                            function_name="main_plan",
                            new_code=PLAN_AFTER_GENERALIZATION,
                        ),
                    ],
                    cache=CacheInvalidateSpec(invalidate_steps=[]),
                )
            return InterjectionDecision(
                action="complete_task",
                reason="User indicated task is complete",
                patches=[],
                cache=CacheInvalidateSpec(invalidate_steps=[]),
            )

        async def mock_modification_generate(*args, **kwargs):
            _ = (args, kwargs)
            nonlocal interjection_count
            interjection_count += 1
            response = create_mock_modification_response(interjection_count)
            return response.model_dump_json()

        active_task.modification_client.generate = mock_modification_generate
        active_task.plan_source_code = actor._sanitize_code(INITIAL_PLAN, active_task)

        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        await wait_for_state(
            active_task,
            _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
        )

        await active_task.interject(
            "Navigate to allrecipes.com and search for 'chicken soup'",
        )
        await wait_for_state(
            active_task,
            _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
        )

        await active_task.interject(
            "Click on the first search result and give me a brief summary.",
        )
        await wait_for_state(
            active_task,
            _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
        )

        await active_task.interject(
            "Perfect. Now, repeat the same process for 'chocolate chip cookies'.",
        )
        await wait_for_state(
            active_task,
            _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
        )

        await active_task.interject("Perfect. That's all. Thank you.")

        await asyncio.sleep(1)
        if not active_task.done():
            await active_task.stop("Test complete")

        final_log = "\n".join(active_task.action_log)
        final_code = active_task.plan_source_code

        assert "def main_plan" in final_code
        assert "async def search_recipe" in final_code
        assert "allrecipes.com" in final_log.lower()
        assert (
            "chicken soup" in final_log.lower()
            or "chocolate chip cookies" in final_log.lower()
        )

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
