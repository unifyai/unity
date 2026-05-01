from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic import BaseModel

from unity.common import reasoning


class _FakeClient:
    def __init__(self, result: Any):
        self.result = result
        self.generate_calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Any:
        self.generate_calls.append(kwargs)
        return self.result


@pytest.mark.asyncio
async def test_reason_uses_standard_client_defaults(monkeypatch: pytest.MonkeyPatch):
    client = _FakeClient("yes")
    created: list[tuple[str | None, dict[str, Any]]] = []

    def fake_new_llm_client(model: str | None = None, **kwargs: Any) -> _FakeClient:
        created.append((model, kwargs))
        return client

    monkeypatch.setattr(reasoning, "new_llm_client", fake_new_llm_client)

    result = await reasoning.reason("Does this need a reply?")

    assert result == "yes"
    assert created == [
        (
            None,
            {
                "async_client": True,
                "stateful": False,
                "origin": "CodeActActor.reason",
            },
        ),
    ]
    assert client.generate_calls == [
        {
            "user_message": "Does this need a reply?",
            "system_message": reasoning.DEFAULT_REASONING_SYSTEM,
            "response_format": None,
            "temperature": 0.0,
        },
    ]


@pytest.mark.asyncio
async def test_reason_supports_pydantic_structured_output(
    monkeypatch: pytest.MonkeyPatch,
):
    class Decision(BaseModel):
        category: str
        needs_reply: bool
        confidence: float

    client = _FakeClient(
        '{"category": "billing", "needs_reply": true, "confidence": 0.91}',
    )
    monkeypatch.setattr(
        reasoning,
        "new_llm_client",
        lambda *args, **kwargs: client,
    )

    result = await reasoning.reason(
        "Classify this message.",
        system="Use the inbox triage rubric.",
        response_format=Decision,
    )

    assert isinstance(result, Decision)
    assert result.category == "billing"
    assert result.needs_reply is True
    assert result.confidence == 0.91
    assert client.generate_calls[0]["response_format"] is Decision
    assert client.generate_calls[0]["system_message"] == "Use the inbox triage rubric."


@pytest.mark.asyncio
async def test_reason_supports_dict_structured_output(
    monkeypatch: pytest.MonkeyPatch,
):
    response_format = {"type": "json_object"}
    client = _FakeClient('{"needs_followup": true, "rationale": "expires soon"}')
    monkeypatch.setattr(
        reasoning,
        "new_llm_client",
        lambda *args, **kwargs: client,
    )

    result = await reasoning.reason(
        "Does this certificate need follow-up?",
        response_format=response_format,
    )

    assert result == {"needs_followup": True, "rationale": "expires soon"}
    assert client.generate_calls[0]["response_format"] == response_format


def test_reason_docstring_contains_actor_usage_guidance():
    doc = reasoning.reason.__doc__ or ""

    assert "Good uses" in doc
    assert "Prefer direct symbolic code instead" in doc
    assert "Structured output for downstream control flow" in doc
    assert "Anti-patterns" in doc
    assert "billable UniLLM call" in doc
    assert "substring checks" in doc


def test_reasoning_prompt_context_uses_introspected_signature():
    context = reasoning.get_reasoning_prompt_context()

    assert f"async def reason{inspect.signature(reasoning.reason)}" in context
