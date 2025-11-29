# tests/test_tool_loop_event_bus.py
#
# These tests assume the project already contains
# ─  async_tool_use_loop.py   (with _async_tool_use_loop_inner / start_async_tool_loop)
# ─  event_bus.py            (with EventBus / Event)
#
# No stubs for “unify” are provided – the real library is expected to be
# importable in the test environment.

from __future__ import annotations

import unify
import asyncio

import pytest

from unity.common.async_tool_loop import (
    start_async_tool_loop,
)
from unity.common._async_tool.loop import async_tool_loop_inner
from tests.helpers import _handle_project, capture_events
from unity.common.llm_client import new_llm_client


@unify.traced
async def echo(text: str) -> str:  # noqa: D401 – simple echo tool
    # Avoid time-based sleeping; just return immediately
    return text.upper()


# --------------------------------------------------------------------------- #
#                         Integration-level expectations                       #
# --------------------------------------------------------------------------- #
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

    # Exactly four events should have been published for the run.
    # Captured events are already in chronological order (oldest first).
    events = captured_events
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
    Run the *wrapper* helper so that we can inject an extra user turn while the
    loop is still thinking, then confirm that the event bus recorded it.
    """

    client = new_llm_client(model=model)
    client.set_system_message(
        "CRITICAL INSTRUCTION - YOU MUST FOLLOW THIS EXACTLY:\n"
        "Your ONLY task is to echo back the user's most recent message.\n"
        "Response format: 'You said: X' where X is their latest message.\n"
        "Example: If user says 'apple', respond EXACTLY: 'You said: apple'\n"
        "Do NOT add greetings, emojis, questions, or any other text.\n"
        "Do NOT be creative or helpful. Just echo the message.",
    )

    async with capture_events("ToolLoop") as captured_events:
        handle = start_async_tool_loop(
            client=client,
            message="greetings",
            tools={},  # no tools needed
            max_consecutive_failures=1,
        )

        # Interject with a different message (avoid sequential words like first/second
        # which some models interpret as a counting pattern to continue).
        await handle.interject("pineapple")

        final = await handle.result()

    # The model should acknowledge the interjected message ("pineapple") in its response.
    # We check for the word appearing in the response rather than a specific format,
    # as different models have varying instruction-following fidelity.
    assert "pineapple" in final.lower()

    events = captured_events
    roles = [evt.payload["message"]["role"] for evt in events]
    assert "user" in roles  # initial user
    # Interjection is now published as a simple user message (not system message)
    # for Claude/Gemini compatibility. We expect 2 user messages: initial + interjection.
    assert roles.count("user") == 2
    assert any(
        evt.payload["message"]["role"] == "user"
        and "pineapple" in (evt.payload["message"].get("content") or "")
        for evt in events
    )
