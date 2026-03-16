from __future__ import annotations
import re
import pytest

from tests.helpers import _handle_project, capture_events
from unity.events.event_bus import EVENT_BUS
from unity.events.manager_event_logging import (
    log_manager_result,
)
from unity.common._async_tool.loop_config import TOOL_LOOP_LINEAGE

_SUFFIX_RE = re.compile(r"\(([0-9a-f]{4})\)$")


# ============================================================================
#  log_manager_result decorator tests
# ============================================================================
# The ``log_manager_result`` decorator is the counterpart to
# ``log_manager_call`` for manager methods that return plain results
# (str, dict, etc.) rather than SteerableToolHandle objects.  It is used
# by MemoryManager.
#
# Tests verify:
# 1. Incoming/outgoing ManagerMethod event pairs are published
# 2. ``display_label`` propagates to both events
# 3. ``TOOL_LOOP_LINEAGE`` is set for the method body and reset afterwards
# 4. Error paths publish outgoing events with status="error"
# 5. The payload_key is correctly extracted from args/kwargs


class _StubManager:
    """Minimal manager stub for testing the log_manager_result decorator."""

    @log_manager_result(
        "StubManager",
        "process",
        payload_key="text",
        display_label="Processing data",
    )
    async def process(self, text: str) -> str:
        # Record the lineage visible inside the method body
        self._inner_lineage = list(TOOL_LOOP_LINEAGE.get([]))
        return f"processed: {text}"

    @log_manager_result(
        "StubManager",
        "fail",
        payload_key="text",
        display_label="Failing gracefully",
    )
    async def fail(self, text: str) -> str:
        raise ValueError("intentional error")

    @log_manager_result(
        "StubManager",
        "pollute",
        payload_key="text",
        display_label="Polluting lineage",
    )
    async def pollute(self, text: str) -> str:
        """Simulates what a real MemoryManager method does: the inner body
        modifies TOOL_LOOP_LINEAGE (as start_async_tool_loop would) and does
        NOT restore it.  The decorator must handle this gracefully."""
        current = list(TOOL_LOOP_LINEAGE.get([]))
        TOOL_LOOP_LINEAGE.set([*current, "inner_tool_loop"])
        return f"polluted: {text}"


@pytest.mark.asyncio
@_handle_project
async def test_result_publishes_incoming_and_outgoing():
    """A successful call should produce exactly one incoming and one outgoing event."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        result = await mgr.process("hello")

    EVENT_BUS.join_published()

    assert result == "processed: hello"

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("method") == "process"
        and e.payload.get("phase") == "incoming"
    ]
    assert len(incoming) == 1, f"Expected 1 incoming event, got {len(incoming)}"
    assert incoming[0].payload.get("text") == "hello"

    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert len(outgoing) == 1, f"Expected 1 outgoing event, got {len(outgoing)}"
    assert outgoing[0].payload.get("answer") == "processed: hello"
    assert outgoing[0].payload.get("status", "ok") == "ok"


@pytest.mark.asyncio
@_handle_project
async def test_result_display_label_propagates():
    """Both incoming and outgoing events should carry the display_label."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        await mgr.process("test")

    EVENT_BUS.join_published()

    stub_events = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("method") == "process"
    ]
    assert len(stub_events) >= 2

    for evt in stub_events:
        assert (
            evt.payload.get("display_label") == "Processing data"
        ), f"display_label missing or wrong on {evt.payload.get('phase')} event"


@pytest.mark.asyncio
@_handle_project
async def test_result_lineage_set_during_body():
    """TOOL_LOOP_LINEAGE should include the method's leaf during execution."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod"):
        await mgr.process("lineage-test")

    assert hasattr(mgr, "_inner_lineage")
    assert any("StubManager.process" in s for s in mgr._inner_lineage)


@pytest.mark.asyncio
@_handle_project
async def test_result_lineage_reset_after_return():
    """TOOL_LOOP_LINEAGE should be restored to its original value after the call."""
    mgr = _StubManager()
    lineage_before = list(TOOL_LOOP_LINEAGE.get([]))

    async with capture_events("ManagerMethod"):
        await mgr.process("reset-test")

    lineage_after = list(TOOL_LOOP_LINEAGE.get([]))
    assert lineage_after == lineage_before


@pytest.mark.asyncio
@_handle_project
async def test_result_lineage_reset_after_error():
    """TOOL_LOOP_LINEAGE should be restored even when the method raises."""
    mgr = _StubManager()
    lineage_before = list(TOOL_LOOP_LINEAGE.get([]))

    async with capture_events("ManagerMethod"):
        with pytest.raises(ValueError, match="intentional error"):
            await mgr.fail("boom")

    lineage_after = list(TOOL_LOOP_LINEAGE.get([]))
    assert lineage_after == lineage_before


@pytest.mark.asyncio
@_handle_project
async def test_result_error_publishes_outgoing_with_error_status():
    """When the method raises, the outgoing event should carry error details."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        with pytest.raises(ValueError):
            await mgr.fail("error-test")

    EVENT_BUS.join_published()

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("method") == "fail"
        and e.payload.get("phase") == "incoming"
    ]
    assert incoming, "No incoming event for failing method"
    call_id = incoming[0].calling_id

    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing, "No outgoing event for failing method"
    assert outgoing[0].payload.get("status") == "error"
    assert "intentional error" in outgoing[0].payload.get("error", "")
    assert outgoing[0].payload.get("error_type") == "ValueError"
    assert outgoing[0].payload.get("display_label") == "Failing gracefully"


@pytest.mark.asyncio
@_handle_project
async def test_result_inherits_parent_lineage():
    """When called under an existing lineage, the hierarchy should include the parent."""
    mgr = _StubManager()

    token = TOOL_LOOP_LINEAGE.set(["OuterManager.act"])
    try:
        async with capture_events("ManagerMethod") as events:
            await mgr.process("nested-test")

        EVENT_BUS.join_published()

        incoming = [
            e
            for e in events
            if e.payload.get("manager") == "StubManager"
            and e.payload.get("method") == "process"
            and e.payload.get("phase") == "incoming"
        ]
        assert incoming
        hierarchy = incoming[0].payload.get("hierarchy", [])
        assert len(hierarchy) == 2
        assert hierarchy[0] == "OuterManager.act"
        assert re.match(r"StubManager\.process\([0-9a-f]{4}\)$", hierarchy[1])
    finally:
        TOOL_LOOP_LINEAGE.reset(token)


@pytest.mark.asyncio
@_handle_project
async def test_result_payload_key_from_positional_arg():
    """The payload_key should be extracted from the first positional arg."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        await mgr.process("positional-value")

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("phase") == "incoming"
    ]
    assert incoming
    assert incoming[0].payload.get("text") == "positional-value"


@pytest.mark.asyncio
@_handle_project
async def test_result_outgoing_hierarchy_not_polluted_by_inner_lineage():
    """When the method body modifies TOOL_LOOP_LINEAGE (as start_async_tool_loop
    does), the outgoing event must still carry the correct hierarchy — not the
    polluted value left by the inner code."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        await mgr.pollute("test")

    EVENT_BUS.join_published()

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("method") == "pollute"
        and e.payload.get("phase") == "incoming"
    ]
    assert incoming, "No incoming event"
    incoming_hierarchy = incoming[0].payload.get("hierarchy", [])

    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing, "No outgoing event"
    outgoing_hierarchy = outgoing[0].payload.get("hierarchy", [])

    assert (
        incoming_hierarchy == outgoing_hierarchy
    ), f"Hierarchy mismatch: incoming={incoming_hierarchy}, outgoing={outgoing_hierarchy}"
    assert len(incoming_hierarchy) == 1
    assert re.match(r"StubManager\.pollute\([0-9a-f]{4}\)$", incoming_hierarchy[0])
    assert len(outgoing_hierarchy) == 1
    assert re.match(r"StubManager\.pollute\([0-9a-f]{4}\)$", outgoing_hierarchy[0])


@pytest.mark.asyncio
@_handle_project
async def test_result_incoming_hierarchy_not_doubled():
    """The incoming event hierarchy must not contain the leaf twice.

    Regression: if TOOL_LOOP_LINEAGE is set *before* publishing the incoming
    event, publish_manager_method_event reads the already-set lineage and
    appends the leaf a second time."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        await mgr.process("double-check")

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("method") == "process"
        and e.payload.get("phase") == "incoming"
    ]
    assert incoming
    hierarchy = incoming[0].payload.get("hierarchy", [])
    assert len(hierarchy) == 1 and re.match(
        r"StubManager\.process\([0-9a-f]{4}\)$",
        hierarchy[0],
    ), f"Hierarchy wrong: {hierarchy}"


def _extract_suffix(hierarchy_label: str) -> str | None:
    """Extract the trailing 4-hex-char suffix from a hierarchy_label like 'Foo.bar(c8f5)'."""
    m = _SUFFIX_RE.search(hierarchy_label)
    return m.group(1) if m else None


@pytest.mark.asyncio
@_handle_project
async def test_result_suffix_consistent_across_incoming_outgoing():
    """log_manager_result: the hierarchy_label suffix on the incoming event must
    match the suffix on the outgoing event."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        await mgr.process("suffix-result-check")

    EVENT_BUS.join_published()

    stub_events = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("method") == "process"
    ]
    assert len(stub_events) >= 2

    incoming = [e for e in stub_events if e.payload.get("phase") == "incoming"]
    outgoing = [e for e in stub_events if e.payload.get("phase") == "outgoing"]
    assert incoming and outgoing

    incoming_suffix = _extract_suffix(incoming[0].payload.get("hierarchy_label", ""))
    outgoing_suffix = _extract_suffix(outgoing[0].payload.get("hierarchy_label", ""))
    assert (
        incoming_suffix
    ), f"No suffix on incoming: {incoming[0].payload.get('hierarchy_label')}"
    assert (
        outgoing_suffix
    ), f"No suffix on outgoing: {outgoing[0].payload.get('hierarchy_label')}"
    assert (
        incoming_suffix == outgoing_suffix
    ), f"Suffix mismatch: incoming=({incoming_suffix}), outgoing=({outgoing_suffix})"
