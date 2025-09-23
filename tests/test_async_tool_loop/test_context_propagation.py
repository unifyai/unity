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
from unity.common.async_tool_loop import start_async_tool_use_loop
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

    handle = start_async_tool_use_loop(
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
