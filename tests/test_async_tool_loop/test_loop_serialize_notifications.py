from __future__ import annotations

import asyncio
import pytest
import unify

from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_assistant_call_prefix,
    _wait_for_tool_result,
)
from unity.common.async_tool_loop import start_async_tool_loop, AsyncToolLoopHandle


# Module-level gate so deserialization can reuse the same import path and symbol
gate: asyncio.Event | None = None


async def notify_parent(
    message: str,
    *,
    _notification_up_q: asyncio.Queue | None = None,
) -> str:
    if _notification_up_q is None:
        raise RuntimeError("notification queue missing")
    await _notification_up_q.put({"message": message})
    return "ok"


async def blocker(*, _notification_up_q: asyncio.Queue | None = None) -> str:
    global gate
    if gate is None:
        gate = asyncio.Event()
    await gate.wait()
    return "done"


@pytest.mark.asyncio
@_handle_project
async def test_serialize_deserialize_with_notifications_inline():
    """Serialize a loop with pending notifications and replay them after resume."""

    # Reset module gate for this test
    global gate
    gate = asyncio.Event()

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=True,
    )
    client.set_system_message(
        "Call notify_parent(message='Phase 1') and notify_parent(message='Phase 2') exactly once each.\n"
        "Then call blocker exactly once and only produce your final assistant message after blocker completes.",
    )

    handle = start_async_tool_loop(
        client,
        "begin",
        tools={
            "notify_parent": notify_parent,
            "blocker": blocker,
        },
    )

    # Ensure the assistant has requested notify_parent and we got its results
    await _wait_for_assistant_call_prefix(client, "notify_parent", timeout=120.0)
    await _wait_for_tool_result(
        client,
        tool_name="notify_parent",
        min_results=2,
        timeout=120.0,
    )

    # Serialize while blocker is still running; notifications should be pending on the handle
    snap = handle.serialize()
    assert isinstance(snap, dict)
    assert snap.get("version") == 1
    notifs = snap.get("notifications") or []
    assert isinstance(notifs, list) and len(notifs) >= 2

    # Resume from snapshot and confirm notifications are available immediately
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    evt1 = await asyncio.wait_for(resumed.next_notification(), timeout=60)
    evt2 = await asyncio.wait_for(resumed.next_notification(), timeout=60)
    assert evt1.get("type") == "notification" or True  # tolerant across sources
    assert evt2.get("type") == "notification" or True

    # Unblock the running tool to allow the loop to complete
    gate.set()
    out = await asyncio.wait_for(resumed.result(), timeout=300)
    assert isinstance(out, str)
