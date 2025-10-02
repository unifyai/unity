"""
Chat context propagation tests for async tool loop.

Verifies that `parent_chat_context` is threaded into tools that accept it and
that the loop inserts the synthetic system context header.
"""

from __future__ import annotations

import os
from typing import List

import pytest
import unify
from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project, SETTINGS

MODEL_NAME = os.getenv("UNIFY_MODEL", "gpt-4o@openai")


def new_client() -> unify.AsyncUnify:
    return unify.AsyncUnify(
        MODEL_NAME,
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )


@pytest.mark.asyncio
@_handle_project
async def test_chat_context_propagation() -> None:
    client = new_client()

    root_ctx = [{"role": "user", "content": "root-level message"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(*, parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    handle = start_async_tool_loop(
        client=client,
        message="Please call the function `record_context()` once, then reply 'done'.",
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
    )

    final_ans = await handle.result()
    assert "done" in final_ans.lower()

    assert client.messages[0]["role"] == "system"
    assert client.messages[0].get("_ctx_header") is True

    assert len(captured_ctx) == 1
    combined = captured_ctx[0]

    assert combined[0]["content"] == "root-level message"
    assert "children" in combined[0]
    child_msgs = combined[0]["children"]
    assert child_msgs and child_msgs[0]["content"].startswith(
        "Please call the function",
    )


@pytest.mark.asyncio
@_handle_project
async def test_ask_uses_continued_parent_context() -> None:
    """Verify that ask() packages continued parent context and influences the answer.

    The inner inspection loop should choose "apple" only because that signal
    exists in the provided continued context, not in the current prompt.
    """

    client = new_client()

    # Start a trivial outer loop (no tools needed for this test).
    handle = start_async_tool_loop(
        client=client,
        message=("We will later follow-up with a question requiring broader context."),
        tools={},
        log_steps=False,
    )

    # Provide a contrived continued parent context that carries the deciding hint.
    continued_ctx = [
        {
            "role": "assistant",
            "content": (
                "Important persistent preference: If asked to choose a fruit, "
                "always choose APPLE and not BANANA."
            ),
        },
    ]

    # Ask a question whose correct answer requires the continued parent context.
    helper = await handle.ask(
        ("Which fruit should we choose? Please answer in one short phrase."),
        parent_chat_context_cont=continued_ctx,
    )
    ans = await helper.result()

    assert "apple" in ans.lower(), "Answer did not reflect continued parent context."


@pytest.mark.asyncio
@_handle_project
async def test_interject_with_continued_parent_context_influences_decision() -> None:
    """Verify that an interjection with continued parent context steers the LLM decision.

    The outer loop should incorporate the interjection (and its continued context)
    such that the next assistant reply reflects that broader context.
    """

    client = new_client()

    handle = start_async_tool_loop(
        client=client,
        message=(
            "We need to pick a fruit between APPLE and BANANA. "
            "Decide shortly after considering any additional context."
        ),
        tools={},
        log_steps=False,
    )

    continued_ctx = [
        {
            "role": "assistant",
            "content": (
                "If asked to decide between fruits, the correct choice is APPLE."
            ),
        },
    ]

    # Inject guidance that includes the continued parent context.
    await handle.interject(
        "FYI: see additional context that determines the correct fruit.",
        parent_chat_context_cont=continued_ctx,
    )

    final = await handle.result()
    assert (
        "apple" in final.lower()
    ), "Final decision did not reflect continued parent context."
