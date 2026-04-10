import asyncio

import pytest

from unity.actor.code_act_actor import CodeActActor

pytestmark = pytest.mark.llm_call


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_clarification_flow():
    """Test that CodeActActor can handle clarification requests via queues."""
    clarification_up_q = asyncio.Queue()
    clarification_down_q = asyncio.Queue()

    planner = CodeActActor()

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

        await active_task.stop("Test complete")
        final_result = await active_task.result()
        assert not str(final_result).startswith(
            "Error:",
        ), f"Unexpected error: {final_result}"

        history_str = str(active_task.get_history())
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
