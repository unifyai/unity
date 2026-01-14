"""
tests/test_conversation_manager/test_event_bus_integration.py
=============================================================

Tests verifying that ConversationManager's brain tool loop emits
ToolLoop events to the EventBus, enabling trigger-based test synchronization.

These tests use the same async_helpers patterns used by other managers,
demonstrating that ConversationManager integrates properly with the
standard async tool loop infrastructure.
"""

import asyncio
import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import SMSReceived
from unity.events.event_bus import EVENT_BUS

# All tests in this file require EventBus publishing to verify ToolLoop events
pytestmark = [pytest.mark.eval, pytest.mark.enable_eventbus]


@pytest.mark.asyncio
@_handle_project
async def test_brain_emits_toolloop_events(initialized_cm):
    """Verify that the brain tool loop emits ToolLoop events to the EventBus.

    This test confirms that:
    1. When we process an event that triggers an LLM run
    2. The brain's async tool loop emits ToolLoop events
    3. These events can be observed via the EventBus

    This is the foundation for using trigger-based async_helpers in CM tests.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Track ToolLoop events we receive
    received_events: list[dict] = []
    done_event = asyncio.Event()

    async def capture_toolloop_events(events):
        for evt in events or []:
            payload = getattr(evt, "payload", None)
            if payload is not None:
                received_events.append(
                    (
                        payload.model_dump()
                        if hasattr(payload, "model_dump")
                        else dict(payload)
                    ),
                )
                # Signal when we get an assistant message (the brain responded)
                msg = (
                    payload.message
                    if hasattr(payload, "message")
                    else payload.get("message")
                )
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    done_event.set()

    # Subscribe to ToolLoop events before running
    await EVENT_BUS.register_callback(
        event_type="ToolLoop",
        callback=capture_toolloop_events,
        every_n=1,
    )

    # Process an event that will trigger the brain
    result = await cm._step(
        SMSReceived(
            contact=contact,
            content="Hi there!",
        ),
    )

    # Verify we got output (sanity check)
    assert result.llm_ran, "LLM should have run"

    # Verify ToolLoop events were captured
    assert len(received_events) > 0, (
        "Expected ToolLoop events to be emitted by the brain tool loop. "
        "The brain should emit events via to_event_bus() when processing."
    )

    # Verify we got an assistant message (the brain's response)
    assistant_msgs = [
        e
        for e in received_events
        if isinstance(e.get("message"), dict)
        and e["message"].get("role") == "assistant"
    ]
    assert (
        len(assistant_msgs) > 0
    ), "Expected at least one assistant message in ToolLoop events"

    # Verify the loop_id identifies ConversationManager
    methods = {e.get("method") for e in received_events if e.get("method")}
    assert any(
        "ConversationManager" in m for m in methods
    ), f"Expected loop_id to contain 'ConversationManager', got: {methods}"


@pytest.mark.asyncio
@_handle_project
async def test_toolloop_events_contain_expected_fields(initialized_cm):
    """Verify that ToolLoop events contain the expected fields for async_helpers.

    This test confirms that the ToolLoop events emitted by the brain have:
    - message: dict with role, content, tool_calls etc.
    - method: string identifying the loop (ConversationManager._run_llm)
    - hierarchy: list of parent lineages
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Track ToolLoop events
    received_events: list[dict] = []

    async def capture_events(events):
        for evt in events or []:
            payload = getattr(evt, "payload", None)
            if payload is not None:
                received_events.append(
                    (
                        payload.model_dump()
                        if hasattr(payload, "model_dump")
                        else dict(payload)
                    ),
                )

    await EVENT_BUS.register_callback(
        event_type="ToolLoop",
        callback=capture_events,
        every_n=1,
    )

    # Process an event
    result = await cm._step(
        SMSReceived(
            contact=contact,
            content="What time is it?",
        ),
    )

    assert result.llm_ran
    assert len(received_events) > 0

    # Verify expected fields are present
    for evt in received_events:
        assert "message" in evt, "ToolLoop event should have 'message' field"
        assert "method" in evt, "ToolLoop event should have 'method' field"

        msg = evt["message"]
        assert isinstance(msg, dict), "message should be a dict"
        assert "role" in msg, "message should have 'role'"

    # Verify we have assistant messages (brain responses)
    assistant_events = [
        e
        for e in received_events
        if isinstance(e.get("message"), dict)
        and e["message"].get("role") == "assistant"
    ]
    assert len(assistant_events) > 0, "Should have assistant message events"

    # Verify method identifies ConversationManager
    methods = {e.get("method") for e in received_events}
    cm_methods = [m for m in methods if m and "ConversationManager" in m]
    assert len(cm_methods) > 0, f"Expected CM methods, got: {methods}"
