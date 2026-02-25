# tests/async_tool_loop/test_event_bus.py
#
# These tests assume the project already contains
# ─  async_tool_use_loop.py   (with _async_tool_use_loop_inner / start_async_tool_loop)
# ─  event_bus.py            (with EventBus / Event)
#
# No stubs for “unify” are provided – the real library is expected to be
# importable in the test environment.

from __future__ import annotations

import asyncio
import re

import pytest

from unity.common.async_tool_loop import (
    start_async_tool_loop,
)
from unity.common._async_tool.loop import async_tool_loop_inner
from tests.async_helpers import _wait_for_next_assistant_response_event
from tests.helpers import _handle_project, capture_events
from unity.common.llm_client import new_llm_client
from unity.events.event_bus import EVENT_BUS

_SUFFIX_RE = re.compile(r"\(([0-9a-f]{4})\)$")

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
async def test_basic_event_flow(llm_config) -> None:
    """
    End-to-end check:

        user/msg → assistant/tool-call → tool/result → assistant/final-text
    """

    client = new_llm_client(**llm_config).set_system_message(
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
            time_awareness=False,
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
async def test_interjection_publishes_user_event(llm_config) -> None:
    """
    Verify that interjections are published to the event bus as user messages.

    This test is purely about event bus mechanics, not model behavior.
    Model response quality and instruction-following are tested separately
    in test_interjections.py.
    """
    client = new_llm_client(**llm_config)
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


# --------------------------------------------------------------------------- #
#          Tool results contain actual content, not placeholders              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_tool_result_content_is_not_placeholder(llm_config) -> None:
    """
    Verify that tool messages published to EventBus contain actual tool results,
    not placeholder text like "Streaming..." or "In progress...".

    This is a regression test for an issue where ToolData objects were passed
    by reference to the EventBus, and the content was still a placeholder when
    the event was published (before the actual result was available).
    """
    # Use a deterministic tool that returns known content
    expected_result = "EXPECTED_TOOL_OUTPUT_12345"

    async def deterministic_tool(input_text: str) -> str:
        """A tool that returns a predictable result for testing."""
        return expected_result

    client = new_llm_client(**llm_config).set_system_message(
        "You are an automated test agent.\n"
        "You MUST call the tool named `deterministic_tool` exactly once, "
        "passing any text as the `input_text` argument.\n"
        "After the tool returns, reply with the tool result.",
    )

    pause_event = asyncio.Event()
    pause_event.set()

    async with capture_events("ToolLoop") as captured_events:
        await async_tool_loop_inner(
            client=client,
            message="please call the tool",
            tools={"deterministic_tool": deterministic_tool},
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
            pause_event=pause_event,
            prune_tool_duplicates=True,
        )

    # Filter out internal runtime context events
    events = _filter_runtime_context(captured_events)

    # Find tool result events
    tool_events = [evt for evt in events if evt.payload["message"]["role"] == "tool"]

    assert len(tool_events) >= 1, "Should have at least one tool result event"

    # Verify tool result content is the actual result, not a placeholder
    tool_content = tool_events[0].payload["message"]["content"]

    # Check it's NOT a placeholder
    placeholder_patterns = [
        "Streaming...",
        "In progress...",
        "Loading...",
        "Pending...",
        "...",  # Common placeholder suffix
    ]
    for pattern in placeholder_patterns:
        assert (
            pattern not in tool_content or expected_result in tool_content
        ), f"Tool content appears to be a placeholder: {tool_content!r}"

    # Check it IS the expected result
    assert (
        expected_result in tool_content
    ), f"Tool content should contain '{expected_result}', got: {tool_content!r}"


# --------------------------------------------------------------------------- #
#                     ask() boundary event tests                               #
# --------------------------------------------------------------------------- #


def _extract_suffix(hierarchy_label: str) -> str | None:
    """Extract the trailing 4-hex-char suffix from a hierarchy_label."""
    m = _SUFFIX_RE.search(hierarchy_label)
    return m.group(1) if m else None


@pytest.mark.asyncio
@_handle_project
async def test_ask_publishes_boundary_events(llm_config) -> None:
    """ask() on a running handle should publish incoming + outgoing
    ManagerMethod events with method='ask' and a unique calling_id."""
    client = new_llm_client(**llm_config).set_system_message(
        "You are a test agent. Acknowledge messages briefly.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="do something",
        tools={},
        persist=True,
        max_consecutive_failures=1,
    )

    # Wait for the loop to produce its first response
    await asyncio.sleep(1)

    async with capture_events("ManagerMethod") as mm_events:
        ask_handle = await handle.ask("What are you doing?")
        await ask_handle.result()

    EVENT_BUS.join_published()

    # Stop the parent loop
    await handle.stop()

    ask_events = [
        e
        for e in mm_events
        if e.payload.get("method") == "ask"
        and "Question(" in e.payload.get("hierarchy_label", "")
    ]

    incoming = [e for e in ask_events if e.payload.get("phase") == "incoming"]
    outgoing = [e for e in ask_events if e.payload.get("phase") == "outgoing"]

    assert (
        len(incoming) >= 1
    ), f"Expected incoming ask boundary event, got {len(incoming)}"
    assert (
        len(outgoing) >= 1
    ), f"Expected outgoing ask boundary event, got {len(outgoing)}"

    # Both should share the same calling_id
    ask_call_id = incoming[0].calling_id
    assert ask_call_id, "ask boundary event should have a calling_id"
    assert any(
        e.calling_id == ask_call_id for e in outgoing
    ), "Outgoing ask event should share the incoming calling_id"

    # display_label should be present
    assert incoming[0].payload.get("display_label") == "Answering Question"


@pytest.mark.asyncio
@_handle_project
async def test_ask_sibling_hierarchy(llm_config) -> None:
    """The ask boundary hierarchy should be a sibling of the parent loop,
    not nested under it. It shares the parent's parent lineage."""
    client = new_llm_client(**llm_config).set_system_message(
        "You are a test agent. Acknowledge messages briefly.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="do something",
        tools={},
        loop_id="TestManager.act",
        parent_lineage=[],
        persist=True,
        max_consecutive_failures=1,
    )

    await asyncio.sleep(1)

    async with capture_events("ManagerMethod") as mm_events:
        ask_handle = await handle.ask("What status?")
        await ask_handle.result()

    EVENT_BUS.join_published()
    await handle.stop()

    incoming = [
        e
        for e in mm_events
        if e.payload.get("method") == "ask"
        and e.payload.get("phase") == "incoming"
        and "Question(" in e.payload.get("hierarchy_label", "")
    ]
    assert incoming, "No incoming ask boundary event found"

    hierarchy = incoming[0].payload.get("hierarchy", [])
    # Parent loop_id = "TestManager.act" with parent_lineage = []
    # So parent's hierarchy = ["TestManager.act"]
    # Sibling lineage = parent's parent = [] (nothing above)
    # Ask hierarchy = [*sibling_lineage, "Question(...)"] = ["Question(...)"]
    assert (
        len(hierarchy) == 1
    ), f"Expected sibling hierarchy of length 1, got {hierarchy}"
    assert hierarchy[0].startswith(
        "Question(",
    ), f"Expected Question(...), got {hierarchy[0]}"


@pytest.mark.asyncio
@_handle_project
async def test_ask_multiple_distinguishable(llm_config) -> None:
    """Two ask() calls on the same handle should produce distinct
    calling_ids and hierarchy_labels."""
    client = new_llm_client(**llm_config).set_system_message(
        "You are a test agent. Acknowledge messages briefly.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="do something",
        tools={},
        persist=True,
        max_consecutive_failures=1,
    )

    await _wait_for_next_assistant_response_event(client)

    async with capture_events("ManagerMethod") as mm_events:
        ask1 = await handle.ask("Question one?")
        await ask1.result()
        ask2 = await handle.ask("Question two?")
        await ask2.result()

    EVENT_BUS.join_published()
    await handle.stop()

    incoming = [
        e
        for e in mm_events
        if e.payload.get("method") == "ask"
        and e.payload.get("phase") == "incoming"
        and "Question(" in e.payload.get("hierarchy_label", "")
    ]
    assert len(incoming) >= 2, f"Expected 2 incoming ask events, got {len(incoming)}"

    call_ids = [e.calling_id for e in incoming]
    assert len(set(call_ids)) == len(
        call_ids,
    ), f"ask() calling_ids should be unique, got: {call_ids}"

    suffixes = [_extract_suffix(e.payload.get("hierarchy_label", "")) for e in incoming]
    assert len(set(suffixes)) == len(
        suffixes,
    ), f"ask() hierarchy_label suffixes should be unique, got: {suffixes}"


@pytest.mark.asyncio
@_handle_project
async def test_ask_boundary_suffix_matches_sub_loop(llm_config) -> None:
    """The ask boundary's hierarchy_label suffix should match the
    sub-loop's ToolLoop hierarchy_label suffix."""
    client = new_llm_client(**llm_config).set_system_message(
        "You are a test agent. Acknowledge messages briefly.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="do something",
        tools={},
        persist=True,
        max_consecutive_failures=1,
    )

    await asyncio.sleep(1)

    async with (
        capture_events("ManagerMethod") as mm_events,
        capture_events("ToolLoop") as tl_events,
    ):
        ask_handle = await handle.ask("What are you doing?")
        await ask_handle.result()

    EVENT_BUS.join_published()
    await handle.stop()

    # Find the ask boundary incoming event
    ask_incoming = [
        e
        for e in mm_events
        if e.payload.get("method") == "ask"
        and e.payload.get("phase") == "incoming"
        and "Question(" in e.payload.get("hierarchy_label", "")
    ]
    assert ask_incoming, "No incoming ask boundary event found"

    boundary_suffix = _extract_suffix(
        ask_incoming[0].payload.get("hierarchy_label", ""),
    )
    assert boundary_suffix, "Could not extract suffix from ask boundary event"

    # Find ToolLoop events from the ask sub-loop (contain "Question(" in hierarchy_label)
    ask_tl_events = [
        e for e in tl_events if "Question(" in e.payload.get("hierarchy_label", "")
    ]

    if ask_tl_events:
        tl_suffix = _extract_suffix(ask_tl_events[0].payload.get("hierarchy_label", ""))
        assert tl_suffix == boundary_suffix, (
            f"Suffix mismatch: boundary=({boundary_suffix}), "
            f"sub-loop ToolLoop=({tl_suffix})"
        )
