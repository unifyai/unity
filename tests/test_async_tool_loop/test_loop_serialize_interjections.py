from __future__ import annotations

import asyncio
import pytest
import unify

from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_message_prefix,
    _wait_for_system_interjection_event,
)
from unity.common.async_tool_loop import start_async_tool_loop, AsyncToolLoopHandle


# Gated slow tool (no sleeps) – gate is set by the test to release the tool
SLOW_GATE: asyncio.Event | None = None


# Define an importable slow_tool at module scope so inline-tools deserialization can resolve it
async def slow_tool(*, _pause_event=None):  # type: ignore[unused-argument]
    global SLOW_GATE
    gate = SLOW_GATE
    if gate is None:
        return "SLOW_DONE"
    await gate.wait()
    return "SLOW_DONE"


@pytest.mark.asyncio
@_handle_project
async def test_interjection_applied_only_serialize_deserialize_no_dup():
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "Always call the slow_tool exactly once and then respond with 'OK' (no extra words).",
    )
    # Gate the tool so it stays pending until we explicitly release it later
    globals()["SLOW_GATE"] = asyncio.Event()
    handle = start_async_tool_loop(
        client,
        "begin",
        tools={"slow_tool": slow_tool},
    )

    # Ensure the tool is requested and a placeholder exists so it remains pending
    await _wait_for_tool_request(client, "slow_tool")
    await _wait_for_tool_message_prefix(client, "slow_tool", timeout=120.0)

    # Interject – register event trigger *before* to guarantee ordering
    interjection_text = "Please consider blue theme"
    wait_task = asyncio.create_task(
        _wait_for_system_interjection_event(contains=interjection_text, timeout=120.0),
    )
    await handle.interject(interjection_text)
    await wait_task

    snap = handle.serialize()
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    # Release the gated tool and await completion
    globals()["SLOW_GATE"] = asyncio.Event()
    globals()["SLOW_GATE"].set()
    out = await resumed.result()
    assert isinstance(out, str) and len(out) > 0

    # Verify the interjection system message is present exactly once after resume
    msgs = resumed.get_history()
    seen = [
        m
        for m in msgs
        if m.get("role") == "system" and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1


@pytest.mark.asyncio
@_handle_project
async def test_interjection_immediate_attention_on_resume():
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "Always call the slow_tool exactly once and then respond with 'OK' (no extra words).",
    )
    globals()["SLOW_GATE"] = asyncio.Event()
    handle = start_async_tool_loop(
        client,
        "begin",
        tools={"slow_tool": slow_tool},
    )

    await _wait_for_tool_request(client, "slow_tool")
    await _wait_for_tool_message_prefix(client, "slow_tool", timeout=120.0)

    # Interject and ensure it is applied; register trigger before interjection
    interjection_text = "Switch to compact layout"
    wait_task = asyncio.create_task(
        _wait_for_system_interjection_event(contains=interjection_text, timeout=120.0),
    )
    await handle.interject(interjection_text)
    await wait_task

    # Serialize *before* any subsequent assistant turn to exercise the immediate-attention replay
    snap = handle.serialize()
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    globals()["SLOW_GATE"] = asyncio.Event()
    globals()["SLOW_GATE"].set()
    out = await resumed.result()
    assert isinstance(out, str) and len(out) > 0

    # Ensure no duplicate system interjection appeared after resume
    msgs = resumed.get_history()
    seen = [
        m
        for m in msgs
        if m.get("role") == "system" and interjection_text in str(m.get("content", ""))
    ]
    assert len(seen) == 1


@pytest.mark.asyncio
@_handle_project
async def test_interjection_between_tool_calls_preserves_ordering():
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "Always call the slow_tool exactly once and then respond with 'OK' (no extra words).",
    )
    globals()["SLOW_GATE"] = asyncio.Event()
    handle = start_async_tool_loop(
        client,
        "start",
        tools={"slow_tool": slow_tool},
    )

    # Wait for the tool to be requested and placeholder to be present
    await _wait_for_tool_request(client, "slow_tool")
    await _wait_for_tool_message_prefix(client, "slow_tool", timeout=120.0)

    # Interject between assistant tool call and tool result – trigger first
    interjection_text = "Please log progress frequently"
    wait_task = asyncio.create_task(
        _wait_for_system_interjection_event(contains=interjection_text, timeout=120.0),
    )
    await handle.interject(interjection_text)
    await wait_task

    # Snapshot and resume
    snap = handle.serialize()
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    globals()["SLOW_GATE"] = asyncio.Event()
    globals()["SLOW_GATE"].set()
    out = await resumed.result()
    assert isinstance(out, str) and len(out) > 0

    # In the final transcript, the tool result for slow_tool should appear directly after
    # its requesting assistant message (ordering constraint), irrespective of any interjections.
    msgs = resumed.get_history()
    asst_idx = None
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            tcs = m.get("tool_calls") or []
            if any((tc.get("function", {}).get("name") == "slow_tool") for tc in tcs):
                asst_idx = i
                break
    assert asst_idx is not None, "Assistant turn that called slow_tool not found"

    # The very next message should be the tool reply for slow_tool
    assert asst_idx + 1 < len(msgs)
    nxt = msgs[asst_idx + 1]
    assert nxt.get("role") == "tool" and nxt.get("name") == "slow_tool"


@pytest.mark.asyncio
@_handle_project
async def test_snapshot_includes_full_messages_dump():
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "Always call the slow_tool exactly once and then respond with 'OK' (no extra words).",
    )
    # Gate the tool so it remains pending while we snapshot
    globals()["SLOW_GATE"] = asyncio.Event()
    handle = start_async_tool_loop(
        client,
        "begin",
        tools={"slow_tool": slow_tool},
    )

    # Ensure an assistant tool-call and placeholder exist before snapshot
    await _wait_for_tool_request(client, "slow_tool")
    await _wait_for_tool_message_prefix(client, "slow_tool", timeout=120.0)

    snap = handle.serialize()
    assert isinstance(snap, dict)
    assert "full_messages" in snap, "Snapshot should include full_messages dump"
    assert isinstance(snap["full_messages"], list)

    # The dump should structurally equal the current client messages at snapshot time
    assert snap["full_messages"] == client.messages
