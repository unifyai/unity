from __future__ import annotations

import asyncio
import json

import pytest
import unify

from tests.helpers import _handle_project, SETTINGS

from unity.common.async_tool_loop import start_async_tool_loop
from unity.common.read_only_ask_guard import ReadOnlyAskGuardHandle


@pytest.mark.asyncio
@_handle_project
async def test_guard_triggers_early_stop_and_returns_early_response(monkeypatch):
    """
    When the classifier flags mutation intent, the guard should stop the outer
    loop and `result()` should return the early response. The assistant message
    with the early response should also be appended to the transcript.
    """

    # Stub generate(): detect classifier via presence of explicit `messages` kw.
    async def _stub_generate(self, **kwargs):  # type: ignore[no-untyped-def]
        if "messages" in kwargs:
            # Classifier path → return strict JSON indicating mutation intent
            return json.dumps(
                {
                    "mutation_intent": True,
                    "early_response": "This looks like a change request – please use update.",
                },
            )

        # Main loop path → append an assistant with a long-running tool call
        self.append_messages(  # type: ignore[attr-defined]
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "long_job",
                                "arguments": json.dumps({"seconds": 5}),
                            },
                        },
                    ],
                },
            ],
        )
        return None

    monkeypatch.setattr(unify.AsyncUnify, "generate", _stub_generate, raising=True)

    # Define a long-running tool so the loop would otherwise wait
    @unify.traced
    async def long_job(seconds: int) -> str:  # noqa: D401
        await asyncio.sleep(float(seconds))
        return "done"

    client = unify.AsyncUnify(
        "gpt-5@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message("You may call tools.")

    handle = start_async_tool_loop(
        client=client,
        message="Please change John Doe's phone number to +123...",
        tools={"long_job": long_job},
        handle_cls=ReadOnlyAskGuardHandle,
    )

    result = await handle.result()

    assert (
        "change request" in result.lower()
    ), "Expected guard's early response to be returned as the final result."

    # Transcript should contain the assistant early response appended by the guard
    assert any(
        m.get("role") == "assistant"
        and "change request" in str(m.get("content", "")).lower()
        for m in client.messages
    )


@pytest.mark.asyncio
@_handle_project
async def test_guard_allows_normal_completion_when_no_mutation(monkeypatch):
    """
    When classifier returns mutation_intent=False, the guard should not stop the
    loop and the normal assistant answer should be returned.
    """

    async def _stub_generate(self, **kwargs):  # type: ignore[no-untyped-def]
        if "messages" in kwargs:
            # Classifier path → no mutation intent
            return json.dumps({"mutation_intent": False, "early_response": ""})

        # Main loop path → a plain answer with no tools
        self.append_messages(  # type: ignore[attr-defined]
            [
                {
                    "role": "assistant",
                    "content": "All good, read-only answer.",
                    "tool_calls": [],
                },
            ],
        )
        return None

    monkeypatch.setattr(unify.AsyncUnify, "generate", _stub_generate, raising=True)

    client = unify.AsyncUnify(
        "gpt-5@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message("Answer normally without tools.")

    handle = start_async_tool_loop(
        client=client,
        message="What is Jane's current title?",
        tools={},
        handle_cls=ReadOnlyAskGuardHandle,
    )

    result = await handle.result()
    assert result.strip().startswith("All good")
    # Guard should not have appended any extra answer; last message is the normal one
    assert client.messages and client.messages[-1]["role"] == "assistant"
    assert client.messages[-1]["content"].startswith("All good")
