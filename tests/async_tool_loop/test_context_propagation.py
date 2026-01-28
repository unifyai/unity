"""
Chat context propagation tests for async tool loop.

Verifies that `parent_chat_context` is threaded into tools that accept it and
that the loop inserts the synthetic system context header.
"""

from __future__ import annotations

from typing import List

import pytest
from unity.common.async_tool_loop import ChatContextPropagation, start_async_tool_loop
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client


@pytest.mark.asyncio
@_handle_project
async def test_chat_context_propagation(model) -> None:
    client = new_llm_client(model=model)

    root_ctx = [{"role": "user", "content": "root-level message"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(*, _parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(_parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    handle = start_async_tool_loop(
        client=client,
        message="Please call the function `record_context()` once, then reply 'done'.",
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.ALWAYS,
    )

    final_ans = await handle.result()
    assert final_ans is not None, "Loop should complete with a response"

    # Find the runtime context header message (may not be at position 0 due to
    # other system messages like User Visibility Context being prepended)
    ctx_header_msg = next(
        (m for m in client.messages if m.get("_ctx_header") is True),
        None,
    )
    assert ctx_header_msg is not None, "Expected a system message with _ctx_header=True"
    assert ctx_header_msg["role"] == "system"

    assert len(captured_ctx) == 1
    combined = captured_ctx[0]

    assert combined[0]["content"] == "root-level message"
    assert "children" in combined[0]
    child_msgs = combined[0]["children"]
    # Find the user message (may not be at position 0 due to system messages
    # like User Visibility Context being prepended)
    user_msg = next(
        (m for m in child_msgs if m.get("role") == "user"),
        None,
    )
    assert user_msg is not None and user_msg["content"].startswith(
        "Please call the function",
    )


@pytest.mark.asyncio
@_handle_project
async def test_chat_context_propagation_never(model) -> None:
    """Verify that NEVER mode does NOT pass context to tools, even when they accept it."""
    client = new_llm_client(model=model)

    root_ctx = [{"role": "user", "content": "secret-context-message"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(*, _parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(_parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    handle = start_async_tool_loop(
        client=client,
        message="Please call the function `record_context()` once, then reply 'done'.",
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.NEVER,
    )

    final_ans = await handle.result()
    assert final_ans is not None, "Loop should complete with a response"

    # Tool should have been called
    assert len(captured_ctx) == 1, "Tool should have been called exactly once"

    # But context should be empty (NEVER mode)
    assert captured_ctx[0] == [], "Context should be empty in NEVER mode"


@pytest.mark.asyncio
@_handle_project
async def test_chat_context_propagation_llm_decides_include(model) -> None:
    """Verify that LLM_DECIDES mode passes context when LLM includes it (default)."""
    client = new_llm_client(model=model)

    root_ctx = [{"role": "user", "content": "root-level-context-marker"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(*, _parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(_parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    # Prompt the LLM to explicitly include context
    handle = start_async_tool_loop(
        client=client,
        message=(
            "Please call the function `record_context()` once with "
            "`include_parent_chat_context` set to `true`, then reply 'done'."
        ),
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.LLM_DECIDES,
    )

    final_ans = await handle.result()
    assert final_ans is not None, "Loop should complete with a response"

    # Tool should have been called
    assert len(captured_ctx) == 1, "Tool should have been called exactly once"

    # Context should be passed (LLM chose to include it)
    combined = captured_ctx[0]
    assert len(combined) > 0, "Context should be non-empty when LLM includes it"
    assert combined[0]["content"] == "root-level-context-marker"


@pytest.mark.asyncio
@_handle_project
async def test_chat_context_propagation_llm_decides_exclude(model) -> None:
    """Verify that LLM_DECIDES mode omits context when LLM explicitly excludes it."""
    client = new_llm_client(model=model)

    root_ctx = [{"role": "user", "content": "secret-context-should-not-appear"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(*, _parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(_parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    # Prompt the LLM to explicitly exclude context
    handle = start_async_tool_loop(
        client=client,
        message=(
            "Please call the function `record_context()` once with "
            "`include_parent_chat_context` set to `false`, then reply 'done'."
        ),
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.LLM_DECIDES,
    )

    final_ans = await handle.result()
    assert final_ans is not None, "Loop should complete with a response"

    # Tool should have been called
    assert len(captured_ctx) == 1, "Tool should have been called exactly once"

    # Context should be empty (LLM chose to exclude it)
    assert captured_ctx[0] == [], "Context should be empty when LLM excludes it"


@pytest.mark.asyncio
@_handle_project
async def test_ask_uses_continued_parent_context(model) -> None:
    """Verify that ask() packages continued parent context and influences the answer.

    The inner inspection loop should choose "apple" only because that signal
    exists in the provided continued context, not in the current prompt.
    """

    client = new_llm_client(model=model)

    # Start a trivial outer loop (no tools needed for this test).
    handle = start_async_tool_loop(
        client=client,
        message=("We will later follow-up with a question requiring broader context."),
        tools={},
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
async def test_interject_with_continued_parent_context_influences_decision(
    model,
) -> None:
    """Verify that an interjection with continued parent context steers the LLM decision.

    The outer loop should incorporate the interjection (and its continued context)
    such that the next assistant reply reflects that broader context.
    """

    client = new_llm_client(model=model)

    handle = start_async_tool_loop(
        client=client,
        message=(
            "We need to pick a fruit between APPLE and BANANA. "
            "Decide shortly after considering any additional context."
        ),
        tools={},
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
