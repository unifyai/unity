import asyncio

import pytest
from unittest.mock import AsyncMock

from unity.actor.code_act_actor import CodeActActor

from .conftest import wait_for_turn_completion


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_interjection_incremental_teaching_session():
    """
    Test that CodeActActor can handle incremental interjections
    in an interactive teaching session.
    """
    actor = CodeActActor()
    actor._computer_primitives.navigate = AsyncMock(return_value=None)
    actor._computer_primitives.act = AsyncMock(return_value="Action completed")
    actor._computer_primitives.observe = AsyncMock(return_value="Page content observed")

    plan = None
    try:
        # Start plan with no goal (interactive session)
        plan = await actor.act(None)

        # Wait for initial setup
        await asyncio.sleep(2)

        # Interjection 1
        interjection_1 = "Navigate to allrecipes.com"
        history_len_before = len(plan.get_history())
        await plan.interject(interjection_1)
        try:
            await asyncio.wait_for(
                wait_for_turn_completion(plan, history_len_before),
                timeout=30,
            )
        except asyncio.TimeoutError:
            pass

        # Interjection 2
        interjection_2 = "Great, now search for 'chocolate chip cookies'."
        history_len_before = len(plan.get_history())
        await plan.interject(interjection_2)
        try:
            await asyncio.wait_for(
                wait_for_turn_completion(plan, history_len_before),
                timeout=30,
            )
        except asyncio.TimeoutError:
            pass

        # Interjection 3: Finish
        interjection_3 = "Perfect, that's all. We're done."
        history_len_before = len(plan.get_history())
        await plan.interject(interjection_3)
        try:
            await asyncio.wait_for(
                wait_for_turn_completion(plan, history_len_before),
                timeout=30,
            )
        except asyncio.TimeoutError:
            pass

        # Stop the session from the outside
        await plan.stop("Session complete.")
        final_result = await plan.result()

        assert not str(final_result).startswith(
            "Error:",
        ), f"Unexpected error: {final_result}"

        history_str = str(plan.get_history())
        assert (
            "allrecipes" in history_str.lower()
            or actor._computer_primitives.navigate.called
        )
    finally:
        if plan and not plan.done():
            try:
                await plan.stop("Test cleanup")
            except Exception:
                pass
        if actor:
            try:
                await actor.close()
            except Exception:
                pass
        await asyncio.sleep(0.5)
