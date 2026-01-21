import asyncio

import pytest
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock

from unity.actor.code_act_actor import CodeActActor


class SecretModel(BaseModel):
    secret: int = Field(description="The secret number from context.")


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_code_act_initial_parent_chat_context_is_used():
    """CodeActActor should append _parent_chat_context before the first LLM turn."""
    SecretModel.model_rebuild()

    actor = CodeActActor(headless=True, computer_mode="mock", timeout=60)
    actor._computer_primitives.navigate = AsyncMock(return_value=None)
    actor._computer_primitives.act = AsyncMock(return_value="Action completed")
    actor._computer_primitives.observe = AsyncMock(return_value="Page content observed")

    parent_ctx = [
        {"role": "user", "content": "The secret number is 456."},
        {"role": "assistant", "content": "Acknowledged."},
    ]

    handle = await actor.act(
        "What is the secret number? Return {secret: <int>} and do not guess.",
        clarification_enabled=False,
        response_format=SecretModel,
        _parent_chat_context=parent_ctx,
        persist=False,
    )
    try:
        res = await asyncio.wait_for(handle.result(), timeout=90)
        assert isinstance(res, SecretModel)
        assert res.secret == 456
    finally:
        try:
            await actor.close()
        except Exception:
            pass
