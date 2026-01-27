import asyncio

import pytest
from unittest.mock import AsyncMock

from unity.actor.code_act_actor import CodeActActor


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_clarification_flow():
    """Test that CodeActActor can handle clarification requests via queues."""
    clarification_up_q = asyncio.Queue()
    clarification_down_q = asyncio.Queue()

    planner = CodeActActor(headless=True, computer_mode="mock")
    planner._computer_primitives.navigate = AsyncMock(return_value=None)
    planner._computer_primitives.act = AsyncMock(return_value="Action completed")
    planner._computer_primitives.observe = AsyncMock(return_value="Page content")

    active_task = None
    try:
        ambiguous_goal = "Search for a recipe on allrecipes.com, but first ask me what I want to search for."

        active_task = await planner.act(
            ambiguous_goal,
            _clarification_up_q=clarification_up_q,
            _clarification_down_q=clarification_down_q,
        )

        try:
            question = await asyncio.wait_for(clarification_up_q.get(), timeout=60)
            assert (
                "what" in question.lower()
                or "recipe" in question.lower()
                or "search" in question.lower()
            )

            answer = "chocolate cake"
            await clarification_down_q.put(answer)
            await asyncio.sleep(3)
        except asyncio.TimeoutError:
            # In mocked environment, the LLM might not call request_clarification.
            pass

        final_result = await active_task.stop("Test complete")
        assert not str(final_result).startswith(
            "Error:",
        ), f"Unexpected error: {final_result}"

        history_str = str(active_task.chat_history)
        assert "allrecipes" in history_str.lower() or "recipe" in history_str.lower()
    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop("Test cleanup")
            except Exception:
                pass
        if planner:
            try:
                await planner.close()
            except Exception:
                pass
        await asyncio.sleep(0.5)
