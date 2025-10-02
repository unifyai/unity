from __future__ import annotations

import time
import re
import pytest
import unify

from unity.common.async_tool_loop import (
    start_async_tool_loop,
    AsyncToolLoopHandle,
)
from unity.events.event_bus import EVENT_BUS
from tests.helpers import SETTINGS


@pytest.mark.asyncio
async def test_nested_logging_hierarchy_labels():
    """
    Verify that nested async tool loops emit ToolLoop events with hierarchical
    lineage in payload: `hierarchy` (list[str]) and `hierarchy_label` (str).

    We create an outer loop (loop_id="Outer") whose tool starts an inner loop
    (loop_id="Inner"). We assert that events exist for both levels:
    - hierarchy == ["Outer"]
    - hierarchy == ["Outer", "Inner"] with label "Outer -> Inner"
    """

    # ── inner tool: trivial sync function ──────────────────────────────────
    def inner_tool() -> str:  # noqa: D401
        time.sleep(0.1)
        return "inner-ok"

    # ── outer tool: launches a nested loop and returns its handle ──────────
    async def outer_tool() -> AsyncToolLoopHandle:
        inner_client = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=SETTINGS.UNIFY_CACHE,
            traced=SETTINGS.UNIFY_TRACED,
        )
        inner_client.set_system_message(
            "You are running inside an automated test.\n"
            "1️⃣  Call `inner_tool` (no arguments).\n"
            "2️⃣  Wait for its response.\n"
            "3️⃣  Reply with exactly 'done'.",
        )

        return start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={"inner_tool": inner_tool},
            loop_id="Inner",
            max_steps=10,
            timeout=120,
        )

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"

    # ── top-level loop: uses the outer tool ────────────────────────────────
    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "You are running inside an automated test. Perform the steps exactly:\n"
        "1️⃣  Call `outer_tool` with no arguments.\n"
        "2️⃣  Continue running this tool call, when given the option.\n"
        "3️⃣  Once it is completed, respond with exactly 'outer done'.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        loop_id="Outer",
        max_steps=10,
        timeout=240,
    )

    # Wait for completion
    final_reply = await handle.result()
    assert final_reply.strip().lower() == "outer done"

    # Gather recent ToolLoop events
    events = await EVENT_BUS.search(filter="type == 'ToolLoop'", limit=200)

    # Presence checks for hierarchy payloads
    has_outer_only = any(
        (evt.payload or {}).get("hierarchy") == ["Outer"] for evt in events
    )
    has_outer_inner = any(
        (evt.payload or {}).get("hierarchy") == ["Outer", "Inner"] for evt in events
    )
    has_outer_inner_label = any(
        isinstance((evt.payload or {}).get("hierarchy_label"), str)
        and re.fullmatch(
            r"Outer->Inner(?:\([0-9a-f]{4}\))?",
            (evt.payload or {}).get("hierarchy_label"),
        )
        for evt in events
    )

    assert has_outer_only, "No ToolLoop event recorded with hierarchy ['Outer']"
    assert (
        has_outer_inner
    ), "No ToolLoop event recorded with hierarchy ['Outer', 'Inner']"
    assert (
        has_outer_inner_label
    ), "No ToolLoop event recorded with hierarchy_label 'Outer -> Inner'"


@pytest.mark.asyncio
async def test_single_loop_logging_hierarchy_label():
    """
    Verify that a single (non-nested) async tool loop emits ToolLoop events
    with a flat hierarchy and label equal to its loop_id.

    We start a solo loop with loop_id="Solo" and a trivial tool.
    Assertions:
    - hierarchy == ["Solo"] exists
    - hierarchy_label == "Solo" exists
    - no event exists with hierarchy beginning ["Solo", ...] (i.e., nested)
    """

    @unify.traced
    def noop_tool() -> str:  # noqa: D401
        return "ok"

    client = unify.AsyncUnify(
        "gpt-4o@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "1️⃣  Call `noop_tool`. 2️⃣ Then reply exactly 'done'.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"noop_tool": noop_tool},
        loop_id="Solo",
        max_steps=10,
        timeout=120,
    )

    final_reply = await handle.result()
    assert "done" in final_reply.strip().lower()

    events = await EVENT_BUS.search(filter="type == 'ToolLoop'", limit=200)

    has_solo = any((evt.payload or {}).get("hierarchy") == ["Solo"] for evt in events)
    has_solo_label = any(
        isinstance((evt.payload or {}).get("hierarchy_label"), str)
        and re.fullmatch(
            r"Solo(?:\([0-9a-f]{4}\))?",
            (evt.payload or {}).get("hierarchy_label"),
        )
        for evt in events
    )
    has_nested_under_solo = any(
        isinstance((evt.payload or {}).get("hierarchy"), list)
        and (evt.payload or {}).get("hierarchy")[:1] == ["Solo"]
        and len((evt.payload or {}).get("hierarchy")) > 1
        for evt in events
    )

    assert has_solo, "No ToolLoop event recorded with hierarchy ['Solo']"
    assert has_solo_label, "No ToolLoop event recorded with hierarchy_label 'Solo'"
    assert not has_nested_under_solo, "Unexpected nested hierarchy found under 'Solo'"
