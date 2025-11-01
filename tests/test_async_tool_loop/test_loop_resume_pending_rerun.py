from __future__ import annotations

import asyncio
import pytest
import unify

from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_message_prefix,
)
from unity.common.async_tool_loop import start_async_tool_loop, AsyncToolLoopHandle


GATE: asyncio.Event | None = None


async def blocking_tool() -> str:
    global GATE
    gate = GATE
    if gate is None:
        return "done"
    await gate.wait()
    return "done"


@pytest.mark.asyncio
@_handle_project
async def test_flat_resume_retriggers_pending_base_tool():
    """Pending base tool at snapshot time is re-scheduled after resume (flat loop)."""

    # Ensure a fresh gate for this run
    global GATE
    GATE = asyncio.Event()

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "You are testing resume.\n"
        "1) Call `blocking_tool` exactly once.\n"
        "2) After it finishes, reply exactly 'done'.",
    )

    handle = start_async_tool_loop(
        client,
        "begin",
        tools={"blocking_tool": blocking_tool},
        timeout=180,
    )

    # Ensure tool-call requested and placeholder present
    await _wait_for_tool_request(client, "blocking_tool")
    await _wait_for_tool_message_prefix(client, "blocking_tool", timeout=120.0)

    # Snapshot and resume
    snap = handle.serialize()
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)

    # Release the gate so resumed loop can complete
    GATE.set()
    out = await asyncio.wait_for(resumed.result(), timeout=180)
    assert out.strip().lower() == "done"

    # Verify a single final tool reply exists (no duplicate placeholders)
    msgs = resumed.get_history()
    tool_msgs = [
        m for m in msgs if m.get("role") == "tool" and m.get("name") == "blocking_tool"
    ]
    assert len(tool_msgs) == 1
