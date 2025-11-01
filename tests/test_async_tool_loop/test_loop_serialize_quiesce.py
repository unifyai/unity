from __future__ import annotations

import pytest
import unify

from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    make_gated_sync_tool,
    _wait_for_tool_request,
)
from unity.common.async_tool_loop import start_async_tool_loop


@pytest.mark.asyncio
@_handle_project
async def test_serialize_cancels_inflight_and_keeps_requests():
    gate, hold_tool = make_gated_sync_tool(return_value="done")

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "Always call the hold tool exactly once and then respond with the result only.",
    )

    handle = start_async_tool_loop(
        client,
        "Please proceed",
        tools={"hold": hold_tool},
    )

    # Wait until assistant requests the tool so we know it is in-flight
    await _wait_for_tool_request(client, "hold")

    # Serialize should cancel the loop and capture the assistant tool-call
    snap = handle.serialize()

    assert isinstance(snap, dict)
    assert snap["version"] == 1
    # Assistant must have at least one tool_call for 'hold'
    assert any(
        any(
            tc.get("function", {}).get("name") == "hold"
            for tc in m.get("tool_calls", [])
        )
        for m in snap.get("assistant_steps", [])
        if m.get("role") == "assistant"
    )
    # No tool result should be present because we never opened the gate
    assert not any(
        m.get("role") == "tool" and m.get("name") == "hold"
        for m in snap.get("tool_results", [])
    )
