from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from tests.actor.state_managers.utils import (
    extract_code_act_execute_code_snippets,
    get_code_act_tool_calls,
    make_code_act_actor,
)
from unity.common import reasoning

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


def _coerce_reason_result(response_format: Any, result: dict[str, Any]) -> Any:
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        fields = response_format.model_fields
        return response_format(**{k: v for k, v in result.items() if k in fields})
    if isinstance(response_format, dict):
        return result
    return (
        f"category={result['category']}; "
        f"needs_reply={result['needs_reply']}; "
        f"confidence={result['confidence']}"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_code_act_sprinkles_reason_into_semantic_python_loop(
    monkeypatch: pytest.MonkeyPatch,
):
    reason_calls: list[str] = []

    async def fake_reason(
        prompt: str,
        *,
        response_format: Any = None,
        **_kwargs: Any,
    ) -> Any:
        reason_calls.append(prompt)
        text = prompt.lower()
        needs_reply = "m1" in text or "m3" in text
        category = "scheduling" if "move" in text else "follow_up"
        if "newsletter" in text:
            category = "newsletter"
        result = {
            "category": category,
            "needs_reply": needs_reply,
            "confidence": 0.96,
            "rationale": "Semantic intent requires or does not require a reply.",
        }
        return _coerce_reason_result(response_format, result)

    monkeypatch.setattr(reasoning, "reason", fake_reason)

    messages = [
        {
            "id": "m1",
            "subject": "Calendar question",
            "body": "Thursday fits my week better than Tuesday, if that still works on your side.",
        },
        {
            "id": "m2",
            "subject": "Product newsletter",
            "body": "Here are this month's feature highlights and release notes.",
        },
        {
            "id": "m3",
            "subject": "Follow-up needed",
            "body": "The renewal paperwork is waiting on your signature before the account can continue.",
        },
    ]

    request = (
        "Classify these messages and return the ids that need a human reply. "
        "Write Python that loops over the messages and keeps exact iteration, "
        "list construction, and output formatting deterministic. The hard part "
        "is interpreting message intent, so do not use substring rules as the "
        f"whole classifier. Messages: {messages}. "
        "Do not ask clarifying questions. Return only the final list of ids."
    )

    async with make_code_act_actor(impl="simulated") as (actor, _primitives, _calls):
        handle = await actor.act(
            request,
            clarification_enabled=False,
        )
        result = await handle.result()

    assert result is not None
    assert "execute_code" in set(get_code_act_tool_calls(handle))

    snippets = extract_code_act_execute_code_snippets(handle)
    assert any("reason(" in snippet for snippet in snippets), snippets
    assert reason_calls, "Expected generated Python to call reason(...)."

    result_text = str(result)
    assert "m1" in result_text
    assert "m3" in result_text
    assert "m2" not in result_text
