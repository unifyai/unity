from __future__ import annotations

import pytest
import unify

from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import _wait_for_tool_message_prefix
from unity.common.async_tool_loop import start_async_tool_loop, AsyncToolLoopHandle


# Define the tool at module scope so it is importable by module+qualname
async def needs_clar(*, _clarification_up_q=None, _clarification_down_q=None):
    assert _clarification_up_q is not None and _clarification_down_q is not None
    await _clarification_up_q.put("What colour should we use?")
    ans = await _clarification_down_q.get()
    return f"ACK: {ans}"


@pytest.mark.asyncio
@_handle_project
async def test_serialize_deserialize_with_clarification_inline():
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "Always call the needs_clar tool exactly once and then respond with the result only.",
    )

    handle = start_async_tool_loop(
        client,
        "please proceed",
        tools={"needs_clar": needs_clar},
    )

    # Wait for clarification request to appear in transcript
    await _wait_for_tool_message_prefix(client, "clarification_request_", timeout=120.0)

    snap = handle.serialize()

    assert isinstance(snap, dict)
    assert snap.get("version") == 1
    # Should persist at least one clarification entry
    clars = snap.get("clarifications") or []
    assert isinstance(clars, list) and len(clars) >= 1
    cid = clars[0].get("call_id")
    assert isinstance(cid, str) and len(cid) > 0

    # Resume and answer the clarification
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    await resumed.answer_clarification(cid, "Blue")
    out = await resumed.result()
    assert isinstance(out, str) and "blue" in out.lower()
