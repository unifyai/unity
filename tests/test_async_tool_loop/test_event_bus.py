# tests/test_tool_loop_event_bus.py
#
# These tests assume the project already contains
# ─  async_tool_use_loop.py   (with _async_tool_use_loop_inner / start_async_tool_loop)
# ─  event_bus.py            (with EventBus / Event)
#
# No stubs for “unify” are provided – the real library is expected to be
# importable in the test environment.

from __future__ import annotations

import asyncio

import pytest

from unity.common.async_tool_loop import (
    start_async_tool_loop,
)
from unity.common._async_tool.loop import async_tool_loop_inner
from tests.helpers import _handle_project, capture_events
from unity.common.llm_client import new_llm_client

# All tests in this file require EventBus publishing to verify event behavior
pytestmark = pytest.mark.enable_eventbus


async def echo(text: str) -> str:  # noqa: D401 – simple echo tool
    # Avoid time-based sleeping; just return immediately
    return text.upper()


# --------------------------------------------------------------------------- #
#                         Integration-level expectations                       #
# --------------------------------------------------------------------------- #


def _filter_runtime_context(events: list) -> list:
    """Filter out internal runtime context events (user visibility guidance, etc.)."""
    return [
        evt
        for evt in events
        if not evt.payload.get("message", {}).get("_runtime_context")
    ]


@pytest.mark.asyncio
@_handle_project
async def test_basic_event_flow(model) -> None:
    """
    End-to-end check:

        user/msg → assistant/tool-call → tool/result → assistant/final-text
    """

    client = new_llm_client(model=model).set_system_message(
        "You are an automated test agent.\n"
        "You MUST call the tool named `echo` exactly once, passing the user's message as the `text` argument.\n"
        "Do NOT reply directly without first calling the `echo` tool (even if you think you know the answer).\n"
        "After the tool returns, reply with exactly the tool result text.",
    )

    pause_event = asyncio.Event()
    pause_event.set()  # start un-paused

    async with capture_events("ToolLoop") as captured_events:
        await async_tool_loop_inner(
            client=client,
            message="world",
            tools={"echo": echo},
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
            pause_event=pause_event,
            prune_tool_duplicates=True,
        )

    # Filter out internal runtime context events and check conversation flow.
    # Captured events are already in chronological order (oldest first).
    events = _filter_runtime_context(captured_events)
    assert len(events) == 4

    roles = [evt.payload["message"]["role"] for evt in events]
    assert roles == ["user", "assistant", "tool", "assistant"]

    assert events[0].payload["message"]["content"] == "world"  # original user question
    assert (
        events[2].payload["message"]["content"].strip("'").strip('"') == "WORLD"
    )  # tool result
    assert (
        events[3].payload["message"]["content"].strip("'").strip('"').upper() == "WORLD"
    )  # final assistant reply (may either echo the user of the capitalized tool)


# --------------------------------------------------------------------------- #
#               Publishing still works while the loop is running              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_interjection_publishes_user_event(model) -> None:
    """
    Verify that interjections are published to the event bus as user messages.

    This test is purely about event bus mechanics, not model behavior.
    Model response quality and instruction-following are tested separately
    in test_interjections.py.
    """
    client = new_llm_client(model=model)
    # Minimal prompt - we don't care about model's response quality here
    client.set_system_message("Acknowledge any messages you receive.")

    async with capture_events("ToolLoop") as captured_events:
        handle = start_async_tool_loop(
            client=client,
            message="initial message",
            tools={},
            max_consecutive_failures=1,
        )

        await handle.interject("interjected message")

        # We don't need to verify model output - just let it complete
        await handle.result()

    # Filter out internal runtime context events
    events = _filter_runtime_context(captured_events)
    roles = [evt.payload["message"]["role"] for evt in events]

    # EVENT BUS ASSERTIONS ONLY - no model behavior checks
    assert (
        roles.count("user") == 2
    ), "Event bus should record both initial and interjected user messages"

    user_contents = [
        evt.payload["message"].get("content", "")
        for evt in events
        if evt.payload["message"]["role"] == "user"
    ]
    assert any(
        "initial" in c for c in user_contents
    ), "Initial message should be recorded"
    assert any(
        "interjected" in c for c in user_contents
    ), "Interjection should be recorded"
