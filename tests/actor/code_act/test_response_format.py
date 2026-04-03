import asyncio

import pytest
from pydantic import BaseModel, Field

from unity.actor.code_act_actor import CodeActActor

pytestmark = pytest.mark.llm_call


class AnswerModel(BaseModel):
    answer: int = Field(description="The integer answer.")


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_code_act_response_format_roundtrip():
    """CodeActActor should return structured output when response_format is provided."""
    AnswerModel.model_rebuild()

    actor = CodeActActor(timeout=60)

    handle = await actor.act(
        "Return {answer: 123}. Do not include any extra keys.",
        clarification_enabled=False,
        response_format=AnswerModel,
        persist=False,
    )
    try:
        res = await asyncio.wait_for(handle.result(), timeout=90)
        assert isinstance(res, AnswerModel)
        assert res.answer == 123
    finally:
        try:
            await actor.close()
        except Exception:
            pass
