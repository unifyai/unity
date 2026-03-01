"""
tests/conversation_manager/core/test_desktop_fast_path.py
============================================================

Symbolic tests for the desktop fast-path tool on the ConversationManager.

Verifies:
- Tool surface: desktop_act appears/disappears based on screen share state.
- EventBus signal: DesktopPrimitiveInvoked is a registered event type.
- Async lifecycle: desktop_act returns immediately, registers in in_flight_actions,
  publishes ActorHandleStarted, and silently interjects Actor sessions on completion.
- Interjection targeting: _act_handles_with_desktop_usage tracks which act sessions
  to interject, but does not gate tool exposure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.helpers import _handle_project

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
    cm._act_handles_with_desktop_usage = set()

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
    cm._act_handles_with_desktop_usage = set()

    assert cm.desktop_fast_path_eligible


@pytest.mark.asyncio
@_handle_project
async def test_desktop_tools_present_when_screen_share_active_with_act(initialized_cm):
    """Desktop fast path appears when screen share is active and an in-flight
    act session has used desktop primitives (interjection targets exist)."""
    cm = initialized_cm.cm
    cm.assistant_screen_share_active = True

    mock_handle = MagicMock()
    mock_handle.done.return_value = False
    cm.in_flight_actions[42] = {
        "handle": mock_handle,
        "query": "test",
        "action_type": "act",
        "handle_actions": [],
    }
    cm._act_handles_with_desktop_usage = {42}

    assert cm.desktop_fast_path_eligible

    # Clean up
    cm.in_flight_actions.pop(42, None)
    cm._act_handles_with_desktop_usage.clear()


@pytest.mark.asyncio
@_handle_project
async def test_desktop_tools_remain_after_act_completion(initialized_cm):
    """Desktop fast path remains available after the act session completes,
    as long as screen share is still active.  The _act_handles_with_desktop_usage
    set is cleaned up (for interjection targeting) but tool exposure persists."""
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
    cm._act_handles_with_desktop_usage = {99}
    assert cm.desktop_fast_path_eligible

    event = ActorResult(handle_id=99, success=True, result="done")
    await EventHandler.handle_event(event, cm)

    assert 99 not in cm._act_handles_with_desktop_usage
    assert (
        cm.desktop_fast_path_eligible
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
    cm._act_handles_with_desktop_usage = {50}
    assert cm.desktop_fast_path_eligible

    event = AssistantScreenShareStopped(reason="user_stopped")
    await EventHandler.handle_event(event, cm)

    assert len(cm._act_handles_with_desktop_usage) == 0
    assert not cm.desktop_fast_path_eligible

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
    cm._act_handles_with_desktop_usage = {10}

    mock_cp = MagicMock()
    mock_cp.desktop = MagicMock()
    mock_cp.desktop.act = AsyncMock(return_value="Clicked Submit")

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
    assert "being handled" in request_msg.lower()
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
    assert "already done" in result_msg.lower()
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
    cm._act_handles_with_desktop_usage.clear()


@pytest.mark.asyncio
@_handle_project
async def test_desktop_act_without_act_session_no_interjection_errors(initialized_cm):
    """desktop_act works cleanly when no act session is in-flight (the
    interjection calls are no-ops, not errors)."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm
    cm._act_handles_with_desktop_usage = set()

    mock_cp = MagicMock()
    mock_cp.desktop = MagicMock()
    mock_cp.desktop.act = AsyncMock(return_value="Clicked Submit")

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
