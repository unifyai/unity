import asyncio

import pytest
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock, MagicMock

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


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_execute_function_forwards_parent_chat_context():
    """Parent chat context should flow from the outer act() loop through the
    execute_function tool closure into FunctionManager.execute_function.

    Scenario: two contacts named Lucy exist. The parent conversation
    mentions "Baker" as the surname, but the act() description just says
    "Find Lucy's phone number." Without the parent context forwarded to
    the inner primitive, the contacts lookup would be ambiguous.

    We verify the plumbing by asserting that fm.execute_function was
    called with _parent_chat_context containing the disambiguation hint.
    """
    _primitives_list = [
        {
            "function_id": 1,
            "name": "primitives.contacts.ask",
            "docstring": "Ask a question about contacts. Use for lookups, searches, etc.",
            "is_primitive": True,
        },
    ]

    fm = MagicMock()
    fm.search_functions = MagicMock(return_value={"metadata": _primitives_list})
    fm.filter_functions = MagicMock(return_value={"metadata": _primitives_list})
    fm.list_functions = MagicMock(return_value={"metadata": _primitives_list})
    fm.execute_function = AsyncMock(return_value="Lucy Baker: 555-0199")

    actor = CodeActActor(
        function_manager=fm,
        headless=True,
        computer_mode="mock",
        timeout=60,
    )

    parent_ctx = [
        {
            "role": "user",
            "content": (
                "Can you find Lucy's number? I think her surname is Baker."
            ),
        },
        {"role": "assistant", "content": "Sure, let me look that up for you."},
    ]

    try:
        handle = await actor.act(
            "Find Lucy's phone number from contacts.",
            can_compose=False,
            persist=False,
            clarification_enabled=False,
            _parent_chat_context=parent_ctx,
        )
        await asyncio.wait_for(handle.result(), timeout=90)

        fm.execute_function.assert_called_once()
        call_kwargs = fm.execute_function.call_args.kwargs
        assert "_parent_chat_context" in call_kwargs, (
            "execute_function was not called with _parent_chat_context — "
            "the CodeActActor closure needs to declare _parent_chat_context "
            "in its signature and forward it to fm.execute_function()"
        )
        assert call_kwargs["_parent_chat_context"] is not None
    finally:
        try:
            await actor.close()
        except Exception:
            pass
