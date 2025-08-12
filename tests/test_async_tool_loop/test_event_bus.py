# tests/test_tool_loop_event_bus.py
#
# These tests assume the project already contains
# ─  async_tool_use_loop.py   (with _async_tool_use_loop_inner / start_async_tool_use_loop)
# ─  event_bus.py            (with EventBus / Event)
#
# No stubs for “unify” are provided – the real library is expected to be
# importable in the test environment.

from __future__ import annotations

import unify
import asyncio

import pytest

from unity.common.llm_helpers import (
    _async_tool_use_loop_inner,
    start_async_tool_use_loop,
)
from unity.events.event_bus import EVENT_BUS
from tests.helpers import _handle_project, _get_unity_test_env_var


@unify.traced
async def echo(text: str) -> str:  # noqa: D401 – simple echo tool
    await asyncio.sleep(0.01)  # prove we can yield control
    return text.upper()


# --------------------------------------------------------------------------- #
#                         Integration-level expectations                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_basic_event_flow() -> None:
    """
    End-to-end check:

        user/msg → assistant/tool-call → tool/result → assistant/final-text
    """

    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    ).set_system_message(
        "please echo whatever the user says",
    )

    pause_event = asyncio.Event()
    pause_event.set()  # start un-paused

    await _async_tool_use_loop_inner(
        client=client,
        message="world",
        tools={"echo": echo},
        interject_queue=asyncio.Queue(),
        cancel_event=asyncio.Event(),
        pause_event=pause_event,
        prune_tool_duplicates=True,
        log_steps=False,
    )

    # Exactly four events should have been published for the run
    #    (newest-first order → reverse for readability).
    events = list(
        reversed(await EVENT_BUS.search(filter="type == 'ToolLoop'", limit=10)),
    )
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
async def test_interjection_publishes_user_event() -> None:
    """
    Run the *wrapper* helper so that we can inject an extra user turn while the
    loop is still thinking, then confirm that the event bus recorded it.
    """

    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=_get_unity_test_env_var("UNIFY_CACHE"),
        traced=_get_unity_test_env_var("UNIFY_TRACED"),
    )
    client.set_system_message(
        "Please always respond with 'You said: {my_latest_message}', with the placeholder containing whatever I said most recently, and do not include the quoation marks in your response.",
    )

    handle = start_async_tool_use_loop(
        client=client,
        message="first",
        tools={},  # no tools needed
        max_consecutive_failures=1,
    )

    # Interject with second.
    await handle.interject("second")

    final = await handle.result()
    assert final == "You said: second"

    events = await EVENT_BUS.search(filter="type == 'ToolLoop'", limit=10)
    roles = [evt.payload["message"]["role"] for evt in events]
    assert "user" in roles  # initial user
    # Interjection is now published as a system message that includes the
    # user-visible text in bold. Ensure we saw exactly one initial user and a system interjection.
    assert roles.count("user") == 1
    assert any(
        evt.payload["message"]["role"] == "system"
        and "user: **second**" in (evt.payload["message"].get("content") or "")
        for evt in events
    )
