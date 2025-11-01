from __future__ import annotations

import pytest
import unify

from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_message_prefix,
)
from unity.common.async_tool_loop import start_async_tool_loop
from unity.common.tool_spec import read_only


@pytest.mark.asyncio
@_handle_project
async def test_serialize_rejects_nested_handles():
    # Inner loop: trivial echo to complete immediately
    @read_only
    def echo():
        return "ok"

    # Outer tool that spawns an inner loop and returns its handle
    @read_only
    async def spawn_nested():
        inner_client = unify.AsyncUnify(
            "gpt-5@openai",
            reasoning_effort="high",
            service_tier="priority",
            cache=True,
        )
        inner_client.set_system_message(
            "Always call the echo tool exactly once and reply with the result only.",
        )
        handle = start_async_tool_loop(
            inner_client,
            "please run",
            tools={"echo": echo},
        )
        return handle

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "Always call the spawn_nested tool exactly once and reply with its result only.",
    )

    handle = start_async_tool_loop(
        client,
        "start",
        tools={"spawn_nested": spawn_nested},
    )

    # Wait until assistant requests the nested tool, then wait until the nested
    # handle has been recognised by the outer loop (placeholder tool message).
    await _wait_for_tool_request(client, "spawn_nested")
    await _wait_for_tool_message_prefix(client, "spawn_nested")

    with pytest.raises(
        ValueError,
        match="Nested tool loops are not supported by v1 snapshot",
    ):
        handle.serialize()
