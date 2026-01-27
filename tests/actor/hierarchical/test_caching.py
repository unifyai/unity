"""Caching & idempotency tests for HierarchicalActor."""

import asyncio
import json
import textwrap
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from unity.actor.hierarchical_actor import HierarchicalActor, HierarchicalActorHandle

from tests.actor.hierarchical.helpers import wait_for_log_entry


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_loop_iterations_get_unique_cache_keys():
    """
    Validates that loop context tracking prevents false cache hits across loop iterations,
    while allowing proper cache hits during replay. Tests both simple and nested loops.
    """
    print("\n\n--- Starting Test Harness for 'Loop Context Caching' ---")

    LOOP_CACHING_PLAN = textwrap.dedent(
        """
        async def main_plan():
            '''Test plan with loops to validate loop context tracking in cache keys.'''
            from pydantic import BaseModel, Field

            class StepResult(BaseModel):
                status: str = Field(description="Status of the step")
            StepResult.model_rebuild()

            print("--- Starting Loop Test ---")

            # Simple loop - each iteration should get cache misses on first run
            iteration_count = 0
            while iteration_count < 3:
                print(f"--- Loop iteration {iteration_count} ---")
                # These calls have same args but different loop context each iteration
                await computer_primitives.act(f"Perform action in iteration {iteration_count}")
                result = await computer_primitives.observe(
                    "Check the status",
                    response_format=StepResult
                )
                print(f"Iteration {iteration_count} status: {result.status}")
                iteration_count += 1

            print("--- Loop completed ---")
            await asyncio.sleep(1)
            return "Loop plan finished"
    """,
    )

    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
    active_task = None

    try:

        class StepResult(BaseModel):
            status: str = Field(description="Status of the step")

        StepResult.model_rebuild()

        actor.computer_primitives.act = AsyncMock(return_value="done")
        actor.computer_primitives.observe = AsyncMock(
            return_value=StepResult(status="ok"),
        )

        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test loop context tracking in cache keys",
            persist=True,
        )

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.plan_source_code = actor._sanitize_code(
            LOOP_CACHING_PLAN,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        await wait_for_log_entry(
            active_task,
            "STATE CHANGE: RUNNING -> PAUSED_FOR_INTERJECTION",
            timeout=30,
        )

        initial_log = "\n".join(active_task.action_log)
        initial_miss_count = initial_log.count("CACHE MISS")
        expected_initial_misses = 6
        assert initial_miss_count == expected_initial_misses

        assert initial_log.count("CACHE HIT") == 0

        interjection_message = "Great! Now add one more action at the end."
        modified_plan = LOOP_CACHING_PLAN.replace(
            'return "Loop plan finished"',
            'await computer_primitives.act("Final action after loop")\n    return "Modified loop plan finished"',
        )
        modified_plan_escaped = json.dumps(modified_plan)

        active_task.modification_client.generate = AsyncMock(
            return_value=textwrap.dedent(
                f"""
                {{
                    "action": "modify_task",
                    "reason": "Adding final action after loop.",
                    "patches": [
                        {{
                            "function_name": "main_plan",
                            "new_code": {modified_plan_escaped}
                        }}
                    ]
                }}
            """,
            ),
        )

        _ = await active_task.interject(interjection_message)

        restart_log_index = -1
        for i, entry in enumerate(active_task.action_log):
            if (
                "RUN TRANSITION" in entry
                or "RESTART: Restarting execution loop" in entry
            ):
                restart_log_index = i
                break
        assert restart_log_index != -1

        loop = asyncio.get_event_loop()
        deadline = loop.time() + 30
        while loop.time() < deadline:
            current_log_slice = active_task.action_log[restart_log_index + 1 :]
            if any(
                "STATE CHANGE: RUNNING -> PAUSED_FOR_INTERJECTION" in e
                for e in current_log_slice
            ):
                break
            await asyncio.sleep(0.1)

        await active_task.stop("Modified plan ran, stopping test.")
        _ = await active_task.result()

        replay_log = "\n".join(active_task.action_log[restart_log_index:])
        assert replay_log.count("CACHE HIT") == 6
        assert replay_log.count("CACHE MISS") == 1

    finally:
        if active_task and not active_task.done():
            await active_task.stop("Test cleanup")
        await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_nested_loop_combinations_get_unique_cache_keys():
    """
    Validates that nested loop context tracking works correctly, ensuring
    each combination of outer/inner loop iterations gets unique cache keys.
    """
    print("\n\n--- Starting Test Harness for 'Nested Loop Context Caching' ---")

    NESTED_LOOP_PLAN = textwrap.dedent(
        """
        async def main_plan():
            '''Test plan with nested loops to validate nested loop context tracking.'''
            from pydantic import BaseModel, Field

            class NestedResult(BaseModel):
                value: int = Field(description="Result value")
            NestedResult.model_rebuild()

            print("--- Starting Nested Loop Test ---")

            outer = 0
            while outer < 2:
                print(f"--- Outer loop iteration {outer} ---")

                inner = 0
                while inner < 2:
                    print(f"  --- Inner loop iteration {inner} ---")
                    # Each combination (outer, inner) should get unique cache key
                    result = await computer_primitives.observe(
                        f"Get value for outer={outer}, inner={inner}",
                        response_format=NestedResult
                    )
                    print(f"  Result for ({outer},{inner}): {result.value}")
                    inner += 1

                outer += 1

            print("--- Nested loops completed ---")
            await asyncio.sleep(1)
            return "Nested loop plan finished"
    """,
    )

    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
    active_task = None

    try:

        class NestedResult(BaseModel):
            value: int = Field(description="Result value")

        NestedResult.model_rebuild()

        call_count = 0

        async def mock_observe(*args, **kwargs):
            nonlocal call_count
            _ = (args, kwargs)
            call_count += 1
            return NestedResult(value=call_count)

        actor.computer_primitives.observe = AsyncMock(side_effect=mock_observe)

        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test nested loop context tracking",
            persist=True,
        )

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.plan_source_code = actor._sanitize_code(
            NESTED_LOOP_PLAN,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        await wait_for_log_entry(
            active_task,
            "STATE CHANGE: RUNNING -> PAUSED_FOR_INTERJECTION",
            timeout=30,
        )

        initial_log = "\n".join(active_task.action_log)
        assert initial_log.count("CACHE MISS") == 4
        assert initial_log.count("CACHE HIT") == 0

        await active_task.stop("Test completed")
        _ = await active_task.result()

    finally:
        if active_task and not active_task.done():
            await active_task.stop("Test cleanup")
        await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_action_caching_orchestrator():
    """
    Orchestrator test maintained for backwards compatibility with the original monolith.
    Prefer running the individual tests directly.
    """
    pytest.skip(
        "Orchestrator test is redundant; run individual caching tests instead.",
    )
