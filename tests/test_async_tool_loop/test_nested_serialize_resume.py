import asyncio

import pytest
import unify

from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_message_prefix,
)
from unity.common.async_tool_loop import start_async_tool_loop, AsyncToolLoopHandle


# Module‑level gate so the resumed inner loop sees the same event
INNER_GATE: asyncio.Event | None = None


async def inner_tool():
    global INNER_GATE
    gate = INNER_GATE
    if gate is None:
        return "INNER_DONE"
    await gate.wait()
    return "INNER_DONE"


async def outer_tool() -> AsyncToolLoopHandle:
    """Spawn an inner loop that calls `inner_tool` once and replies 'done'."""
    inner_client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    inner_client.set_system_message(
        "You are in a nested test.\n"
        "1) Call `inner_tool` (no args).\n"
        "2) After it finishes, reply exactly 'done'.",
    )
    h = start_async_tool_loop(
        inner_client,
        "start",
        tools={"inner_tool": inner_tool},
        timeout=120,
    )
    return h


def _outer_client():
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "You are the outer loop.\n"
        "1) Call `outer_tool`.\n"
        "2) Continue running it until finished.\n"
        "3) Respond exactly 'all done'.",
    )
    return client


@pytest.mark.asyncio
@_handle_project
async def test_nested_serialize_inline_resume():
    global INNER_GATE
    INNER_GATE = asyncio.Event()

    client = _outer_client()
    handle = start_async_tool_loop(
        client,
        "begin",
        tools={"outer_tool": outer_tool},
        timeout=240,
    )

    # Ensure the outer tool call is requested and placeholder exists
    await _wait_for_tool_request(client, "outer_tool")
    await _wait_for_tool_message_prefix(client, "outer_tool", timeout=120.0)

    # Snapshot recursively (embed child snapshot inline)
    snap = handle.serialize(recursive=True)
    assert isinstance(snap, dict) and isinstance(snap.get("meta", {}), dict)
    children = snap.get("meta", {}).get("children")
    assert isinstance(children, list) and len(children) >= 1

    # Resume from snapshot
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)

    # Release the inner gate so the child can complete
    INNER_GATE.set()
    out = await resumed.result()
    assert out.strip().lower() == "all done"

    # Verify only a single tool reply for outer_tool exists (no duplicates)
    msgs = resumed.get_history()
    tool_msgs = [
        m for m in msgs if m.get("role") == "tool" and m.get("name") == "outer_tool"
    ]
    assert len(tool_msgs) == 1
