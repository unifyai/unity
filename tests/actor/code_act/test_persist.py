import asyncio

import pytest
from unittest.mock import AsyncMock

from unity.actor.code_act_actor import CodeActActor


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_code_act_persist_keeps_loop_alive_until_stopped():
    """
    When persist=True, the underlying async tool loop should not terminate after a
    single assistant message. Instead it should wait for interjections until stopped.
    """
    actor = CodeActActor(timeout=60)
    actor._computer_primitives.navigate = AsyncMock(return_value=None)
    actor._computer_primitives.act = AsyncMock(return_value="Action completed")
    actor._computer_primitives.observe = AsyncMock(return_value="Page content observed")

    handle = await actor.act(
        "Reply with exactly 'READY' and do not call any tools. Then wait for further instructions.",
        clarification_enabled=False,
        persist=True,
    )

    try:
        # Give the loop a moment to produce its initial content and enter persist wait mode.
        await asyncio.sleep(3)
        assert (
            not handle.done()
        ), "persist=True should keep the loop alive after first message"

        # Interject and ensure we're still alive afterwards (persist loop continues).
        _ = await handle.interject(
            "Now reply with exactly 'ACK' and keep waiting.",
        )
        await asyncio.sleep(3)
        assert (
            not handle.done()
        ), "persist=True should keep the loop alive after interjections"

        # Stop should terminate the loop.
        await asyncio.wait_for(handle.stop("test complete"), timeout=30)
        await handle.result()
        assert handle.done()
    finally:
        try:
            await actor.close()
        except Exception:
            pass
