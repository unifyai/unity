import pytest
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock

from unity.actor.hierarchical_actor import (
    HierarchicalActor,
)


class ActionResult(BaseModel):
    completed: bool = Field(..., description="Whether the action completed")
    outcome: str = Field(..., description="Short outcome string")


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_hierarchical_actor_response_format_returns_model():
    """
    If response_format is provided, HierarchicalActorHandle.result() should return
    a Pydantic model instance (not a raw JSON string).
    """
    ActionResult.model_rebuild()

    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
    # Avoid any real browser interactions
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value="ok")
    actor.computer_primitives.observe = AsyncMock(return_value="ok")

    try:
        handle = await actor.act(
            "Do a quick sanity check that you're able to execute a minimal plan. "
            "Then summarize whether you completed the request and what happened.",
            response_format=ActionResult,
            persist=False,
            clarification_enabled=False,
        )
        res = await handle.result()
        assert isinstance(res, ActionResult)
        assert isinstance(res.completed, bool)
        assert isinstance(res.outcome, str) and res.outcome.strip()

    finally:
        try:
            await actor.close()
        except Exception:
            pass
