from __future__ import annotations

import pytest
import unify

from tests.helpers import _handle_project
from unity.common.async_tool_loop import start_async_tool_loop, AsyncToolLoopHandle
from unity.common.tool_spec import read_only


# Define a top-level function so it is importable by module + qualname
@read_only
def greet():
    return "Hello from Inline!"


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_inline_tools_resume():
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "Always call the greet tool exactly once and reply with the result only.",
    )

    handle = start_async_tool_loop(
        client,
        "please greet",
        tools={"greet": greet},
    )

    # Snapshot immediately; assistant may or may not have requested the tool yet
    snap = handle.serialize()

    # Resume from snapshot
    resumed = AsyncToolLoopHandle.deserialize(snap)
    answer = await resumed.result()
    assert isinstance(answer, str) and "hello" in answer.lower()
