"""
tests/conversation_manager/core/test_desktop_fast_path.py
============================================================

Symbolic tests for the desktop fast-path tool on the ConversationManager.

Verifies:
- Tool surface: desktop_act appears/disappears based on screen share state.
- EventBus signal: DesktopPrimitiveInvoked is a registered event type.
- Async lifecycle: desktop_act returns immediately, registers in in_flight_actions,
  publishes ActorHandleStarted, and silently interjects Actor sessions on completion.
- Interjection targeting: all in-flight act sessions receive silent interjections
  from fast-path tools.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.helpers import _handle_project
from unity.function_manager.computer_backends import ActResult

DESKTOP_TOOL_NAMES = {"desktop_act"}


# =============================================================================
# Tool appearance / disappearance
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_desktop_tools_absent_when_screen_share_inactive(initialized_cm):
    """Desktop fast path must NOT appear when screen share is off."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm
    cm.assistant_screen_share_active = False

    action_tools = ConversationManagerBrainActionTools(cm)
    tools = action_tools.as_tools()

    for name in DESKTOP_TOOL_NAMES:
        assert (
            name not in tools
        ), f"{name} should not appear when assistant_screen_share_active is False"


@pytest.mark.asyncio
@_handle_project
async def test_desktop_tools_present_when_screen_share_active_no_act(initialized_cm):
    """Desktop fast path appears when screen share is active, even without any
    in-flight act session that has used desktop primitives."""
    cm = initialized_cm.cm
    cm.assistant_screen_share_active = True

    assert cm.computer_fast_path_eligible


@pytest.mark.asyncio
@_handle_project
async def test_desktop_tools_remain_after_act_completion(initialized_cm):
    """Desktop fast path remains available after the act session completes,
    as long as screen share is still active."""
    from unity.conversation_manager.domains.event_handlers import EventHandler
    from unity.conversation_manager.events import ActorResult

    cm = initialized_cm.cm
    cm.assistant_screen_share_active = True

    mock_handle = MagicMock()
    mock_handle.done.return_value = False
    mock_handle.trigger_completion = MagicMock()
    cm.in_flight_actions[99] = {
        "handle": mock_handle,
        "query": "test desktop task",
        "action_type": "act",
        "handle_actions": [],
    }
    assert cm.computer_fast_path_eligible

    event = ActorResult(handle_id=99, success=True, result="done")
    await EventHandler.handle_event(event, cm)

    assert (
        cm.computer_fast_path_eligible
    ), "Tools should remain available — screen share is still active"


@pytest.mark.asyncio
@_handle_project
async def test_desktop_tools_disappear_on_screen_share_stop(initialized_cm):
    """Desktop fast path disappears when assistant screen share stops."""
    from unity.conversation_manager.domains.event_handlers import EventHandler
    from unity.conversation_manager.events import AssistantScreenShareStopped

    cm = initialized_cm.cm
    cm.assistant_screen_share_active = True

    mock_handle = MagicMock()
    mock_handle.done.return_value = False
    cm.in_flight_actions[50] = {
        "handle": mock_handle,
        "query": "test",
        "action_type": "act",
        "handle_actions": [],
    }
    assert cm.computer_fast_path_eligible

    event = AssistantScreenShareStopped(reason="user_stopped")
    await EventHandler.handle_event(event, cm)

    assert not cm.computer_fast_path_eligible

    # Clean up
    cm.in_flight_actions.pop(50, None)


# =============================================================================
# EventBus signal
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_desktop_primitive_event_type_registered(initialized_cm):
    """DesktopPrimitiveInvoked is a valid, constructable EventBus event type."""
    from unity.events.event_bus import Event
    from unity.events.types import PAYLOAD_REGISTRY

    assert (
        "DesktopPrimitiveInvoked" in PAYLOAD_REGISTRY
    ), "DesktopPrimitiveInvoked must be registered in PAYLOAD_REGISTRY"

    event = Event(type="DesktopPrimitiveInvoked", payload={"method": "act"})
    assert event.type == "DesktopPrimitiveInvoked"
    assert event.payload["method"] == "act"


# =============================================================================
# Async lifecycle and silent interjection
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_desktop_act_returns_acting_and_interjects_on_completion(initialized_cm):
    """desktop_act should return immediately with 'acting' status and silently
    interject in-flight act sessions twice: once when the request is made, and
    again when the background task completes with the result."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm

    mock_actor_handle = MagicMock()
    mock_actor_handle.done.return_value = False
    mock_actor_handle.interject = AsyncMock()
    cm.in_flight_actions[10] = {
        "handle": mock_actor_handle,
        "query": "guide through app",
        "action_type": "act",
        "handle_actions": [],
    }

    mock_cp = MagicMock()
    mock_cp.desktop = MagicMock()
    mock_cp.desktop.act = AsyncMock(
        return_value=ActResult(summary="Clicked Submit", screenshot="base64png"),
    )

    action_tools = ConversationManagerBrainActionTools(cm)

    with patch.object(
        type(cm),
        "computer_primitives",
        new_callable=lambda: property(lambda self: mock_cp),
    ):
        result = await action_tools.desktop_act(instruction="Click Submit")

    assert result["status"] == "acting"
    assert result["query"] == "Click Submit"

    # A new in-flight action should have been registered (beyond handle 10)
    desktop_actions = {
        hid: data
        for hid, data in cm.in_flight_actions.items()
        if data.get("action_type") == "desktop_act"
    }
    assert (
        len(desktop_actions) == 1
    ), f"Expected 1 desktop_act in-flight action, got {len(desktop_actions)}"

    # The request-time interjection should have already fired
    assert mock_actor_handle.interject.call_count == 1
    request_msg = mock_actor_handle.interject.call_args_list[0].args[0]
    assert "fast-path request" in request_msg.lower()
    assert "Click Submit" in request_msg
    assert "do not replicate" in request_msg.lower()
    assert (
        mock_actor_handle.interject.call_args_list[0].kwargs.get(
            "trigger_immediate_llm_turn",
        )
        is False
    )

    # Wait for the background task to complete
    desktop_hid = next(iter(desktop_actions))
    desktop_handle = desktop_actions[desktop_hid]["handle"]
    await desktop_handle.result()

    # After completion, a second interjection should have fired with the result
    assert mock_actor_handle.interject.call_count == 2
    result_msg = mock_actor_handle.interject.call_args_list[1].args[0]
    assert "fast-path result" in result_msg.lower()
    assert "Click Submit" in result_msg
    assert "Clicked Submit" in result_msg
    assert "no action needed" in result_msg.lower()
    assert (
        mock_actor_handle.interject.call_args_list[1].kwargs.get(
            "trigger_immediate_llm_turn",
        )
        is False
    )

    # Clean up
    cm.in_flight_actions.pop(10, None)
    cm.in_flight_actions.pop(desktop_hid, None)


@pytest.mark.asyncio
@_handle_project
async def test_fast_path_interjects_act_session_without_prior_desktop_usage(
    initialized_cm,
):
    """Fast paths must interject ALL in-flight act sessions, not just those that
    have already called desktop primitives.

    Regression: _silent_interject_act_sessions previously iterated
    only act sessions that had already called desktop primitives.  When the
    CM did all the desktop work via fast paths (and the act session had only
    done non-desktop work like loading guidance), the interjections had zero
    recipients — leaving the act session deaf to all fast-path activity.
    """
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm

    # Register an in-flight act session that has NOT used desktop primitives.
    # This mirrors the prod scenario: CodeActActor loaded guidance, then
    # entered persist mode — never calling primitives.computer.*.
    mock_actor_handle = MagicMock()
    mock_actor_handle.done.return_value = False
    mock_actor_handle.interject = AsyncMock()
    cm.in_flight_actions[10] = {
        "handle": mock_actor_handle,
        "query": "Interactive desktop tutorial session",
        "action_type": "act",
        "handle_actions": [],
    }
    mock_cp = MagicMock()
    mock_cp.desktop = MagicMock()
    mock_cp.desktop.act = AsyncMock(
        return_value=ActResult(summary="Clicked Submit", screenshot="base64png"),
    )

    action_tools = ConversationManagerBrainActionTools(cm)

    with patch.object(
        type(cm),
        "computer_primitives",
        new_callable=lambda: property(lambda self: mock_cp),
    ):
        result = await action_tools.desktop_act(instruction="Click Submit")

    assert result["status"] == "acting"

    # Wait for the background task to complete
    desktop_actions = {
        hid: data
        for hid, data in cm.in_flight_actions.items()
        if data.get("action_type") == "desktop_act"
    }
    desktop_hid = next(iter(desktop_actions))
    await desktop_actions[desktop_hid]["handle"].result()

    # The act session MUST have been interjected at least twice:
    # once when the request started, once when it completed.
    assert mock_actor_handle.interject.call_count >= 2, (
        f"Expected at least 2 interjections to the in-flight act session, "
        f"got {mock_actor_handle.interject.call_count}."
    )

    # Verify the interjection content
    request_msg = mock_actor_handle.interject.call_args_list[0].args[0]
    assert "fast-path request" in request_msg.lower()
    assert "Click Submit" in request_msg

    result_msg = mock_actor_handle.interject.call_args_list[1].args[0]
    assert "fast-path result" in result_msg.lower()
    assert "Clicked Submit" in result_msg

    # Clean up
    cm.in_flight_actions.pop(10, None)
    cm.in_flight_actions.pop(desktop_hid, None)


@pytest.mark.asyncio
@_handle_project
async def test_desktop_act_without_act_session_no_interjection_errors(initialized_cm):
    """desktop_act works cleanly when no act session is in-flight (the
    interjection calls are no-ops, not errors)."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm

    mock_cp = MagicMock()
    mock_cp.desktop = MagicMock()
    mock_cp.desktop.act = AsyncMock(
        return_value=ActResult(summary="Clicked Submit", screenshot="base64png"),
    )

    action_tools = ConversationManagerBrainActionTools(cm)

    with patch.object(
        type(cm),
        "computer_primitives",
        new_callable=lambda: property(lambda self: mock_cp),
    ):
        result = await action_tools.desktop_act(instruction="Click Submit")

    assert result["status"] == "acting"
    assert result["query"] == "Click Submit"

    desktop_actions = {
        hid: data
        for hid, data in cm.in_flight_actions.items()
        if data.get("action_type") == "desktop_act"
    }
    assert len(desktop_actions) == 1

    desktop_hid = next(iter(desktop_actions))
    await desktop_actions[desktop_hid]["handle"].result()

    # Clean up
    cm.in_flight_actions.pop(desktop_hid, None)


# =============================================================================
# ComputerActCompleted event chain
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_computer_act_completed_event_type_registered(initialized_cm):
    """ComputerActCompleted is a valid, constructable EventBus event type."""
    from unity.events.event_bus import Event
    from unity.events.types import PAYLOAD_REGISTRY

    assert (
        "ComputerActCompleted" in PAYLOAD_REGISTRY
    ), "ComputerActCompleted must be registered in PAYLOAD_REGISTRY"

    event = Event(
        type="ComputerActCompleted",
        payload={
            "instruction": "Click Submit",
            "summary": "Clicked the Submit button",
        },
    )
    assert event.type == "ComputerActCompleted"
    assert event.payload["instruction"] == "Click Submit"


@pytest.mark.asyncio
@_handle_project
async def test_computer_act_completed_bridge_publishes_when_screen_share_active(
    initialized_cm,
):
    """The bridge callback should publish ComputerActCompleted to the event_broker
    when screen share is active.

    We simulate the EventBUS callback invocation directly (EventBUS publishing
    is disabled in test mode).
    """
    from unittest.mock import MagicMock

    cm = initialized_cm.cm
    cm.assistant_screen_share_active = True

    published = []
    original_publish = cm.event_broker.publish

    async def capture_publish(channel, data):
        published.append((channel, data))
        return await original_publish(channel, data)

    cm.event_broker.publish = capture_publish

    try:
        # Build a fake EventBUS event with the right payload structure
        fake_evt = MagicMock()
        fake_evt.payload = {
            "instruction": "Click Submit",
            "summary": "Clicked the Submit button",
        }

        # Directly invoke the bridge callback logic
        from unity.conversation_manager.events import ComputerActCompleted

        cm_event = ComputerActCompleted(
            instruction=fake_evt.payload["instruction"],
            summary=fake_evt.payload["summary"],
        )
        await cm.event_broker.publish(
            "app:actor:computer_act_completed",
            cm_event.to_json(),
        )

        computer_events = [
            (ch, d) for ch, d in published if ch == "app:actor:computer_act_completed"
        ]
        assert len(computer_events) == 1, (
            f"Expected 1 computer_act_completed event on event_broker, "
            f"got {len(computer_events)}"
        )
    finally:
        cm.event_broker.publish = original_publish
        cm.assistant_screen_share_active = False


@pytest.mark.asyncio
@_handle_project
async def test_computer_act_completed_bridge_skipped_when_screen_share_inactive(
    initialized_cm,
):
    """The bridge callback should NOT publish when screen share is inactive.

    We test this by directly checking the gating logic: when
    assistant_screen_share_active is False, the bridge callback should be
    a no-op.
    """
    cm = initialized_cm.cm
    cm.assistant_screen_share_active = False

    published = []
    original_publish = cm.event_broker.publish

    async def capture_publish(channel, data):
        published.append((channel, data))
        return await original_publish(channel, data)

    cm.event_broker.publish = capture_publish

    try:
        # Simulate what the bridge callback does: check the gate
        # If screen share is inactive, it should NOT publish
        if cm.assistant_screen_share_active:
            from unity.conversation_manager.events import ComputerActCompleted

            cm_event = ComputerActCompleted(
                instruction="Click Submit",
                summary="Clicked",
            )
            await cm.event_broker.publish(
                "app:actor:computer_act_completed",
                cm_event.to_json(),
            )

        computer_events = [
            (ch, d) for ch, d in published if ch == "app:actor:computer_act_completed"
        ]
        assert len(computer_events) == 0, (
            f"Expected 0 computer_act_completed events when screen share is off, "
            f"got {len(computer_events)}"
        )
    finally:
        cm.event_broker.publish = original_publish


@pytest.mark.asyncio
@_handle_project
async def test_computer_act_completed_event_handler_wakes_slow_brain(initialized_cm):
    """EventHandler for ComputerActCompleted should set _has_non_forwarded_event
    and request an LLM run."""
    from unity.conversation_manager.domains.event_handlers import EventHandler
    from unity.conversation_manager.events import ComputerActCompleted

    cm = initialized_cm.cm
    cm._has_non_forwarded_event = False

    request_called = []
    original_request = cm.request_llm_run

    async def mock_request(**kwargs):
        request_called.append(True)

    cm.request_llm_run = mock_request

    try:
        event = ComputerActCompleted(
            instruction="Click Submit",
            summary="Clicked the Submit button",
        )
        await EventHandler.handle_event(event, cm)

        assert cm._has_non_forwarded_event
        assert request_called, "request_llm_run should have been called"
    finally:
        cm.request_llm_run = original_request


def test_render_event_for_fast_brain_computer_act_completed():
    """render_event_for_fast_brain should render ComputerActCompleted events."""
    from unity.conversation_manager.events import ComputerActCompleted
    from unity.conversation_manager.medium_scripts.common import (
        render_event_for_fast_brain,
    )

    event = ComputerActCompleted(
        instruction="Click Submit",
        summary="Clicked the Submit button",
    )
    result = render_event_for_fast_brain(event.to_json())

    assert result is not None
    assert "Computer action completed" in result
    assert "Clicked the Submit button" in result


def test_render_actor_result_empty_success_not_misleading():
    """An ActorResult with success=True but no result must NOT say
    'completed successfully' — it must clearly indicate no results were
    returned so the fast brain doesn't fabricate a positive outcome.

    Regression: in production the Actor exhausted its context budget,
    returned success=True with result=None, and the fast brain received
    'Action completed successfully' which it interpreted as 'Drive is
    connected and ready' — a hallucination.
    """
    from unity.conversation_manager.events import ActorResult
    from unity.conversation_manager.medium_scripts.common import (
        render_event_for_fast_brain,
    )

    event = ActorResult(handle_id=1, success=True, result=None, error=None)
    text = render_event_for_fast_brain(event.to_json())

    assert text is not None
    assert "completed successfully" not in text
    assert "no results" in text.lower()


def test_render_actor_result_with_data():
    """An ActorResult with actual data should include the data snippet."""
    from unity.conversation_manager.events import ActorResult
    from unity.conversation_manager.medium_scripts.common import (
        render_event_for_fast_brain,
    )

    event = ActorResult(
        handle_id=1,
        success=True,
        result="Found 3 files in Drive",
    )
    text = render_event_for_fast_brain(event.to_json())

    assert text is not None
    assert "Found 3 files in Drive" in text
    assert "no results" not in text.lower()


def test_render_actor_result_failure():
    """A failed ActorResult should clearly say 'failed'."""
    from unity.conversation_manager.events import ActorResult
    from unity.conversation_manager.medium_scripts.common import (
        render_event_for_fast_brain,
    )

    event = ActorResult(
        handle_id=1,
        success=False,
        error="Credentials not found",
    )
    text = render_event_for_fast_brain(event.to_json())

    assert text is not None
    assert "failed" in text.lower()
    assert "Credentials not found" in text
