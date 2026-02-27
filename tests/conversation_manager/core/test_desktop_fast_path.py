"""
tests/conversation_manager/core/test_desktop_fast_path.py
============================================================

Symbolic tests for the desktop fast-path tools on the ConversationManager.

Verifies:
- Tool surface: desktop_act, desktop_observe, desktop_get_screenshot appear/disappear
  based on gating conditions (assistant screen share + in-flight act with desktop usage).
- EventBus signal: DesktopPrimitiveInvoked is a registered event type.
- Async lifecycle: fast-path tools return immediately, register in in_flight_actions,
  publish ActorHandleStarted, and silently interject Actor sessions on completion.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.helpers import _handle_project

DESKTOP_TOOL_NAMES = {"desktop_act", "desktop_observe", "desktop_get_screenshot"}


# =============================================================================
# Tool appearance / disappearance
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_desktop_tools_absent_when_screen_share_inactive(initialized_cm):
    """Desktop fast paths must NOT appear when screen share is off."""
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
async def test_desktop_tools_absent_when_no_desktop_usage(initialized_cm):
    """Desktop fast paths must NOT appear when screen share is on but no act
    session has used desktop primitives."""
    cm = initialized_cm.cm
    cm.assistant_screen_share_active = True
    cm._act_handles_with_desktop_usage = set()

    assert not cm.desktop_fast_path_eligible


@pytest.mark.asyncio
@_handle_project
async def test_desktop_tools_present_when_eligible(initialized_cm):
    """Desktop fast paths appear when both conditions are met:
    screen share active AND an in-flight act has used desktop primitives."""
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
async def test_desktop_tools_disappear_on_act_completion(initialized_cm):
    """Desktop fast paths disappear when the triggering act completes."""
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
    assert not cm.desktop_fast_path_eligible


@pytest.mark.asyncio
@_handle_project
async def test_desktop_tools_disappear_on_screen_share_stop(initialized_cm):
    """Desktop fast paths disappear when assistant screen share stops."""
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
    interject in-flight act sessions once the background task completes."""
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

    # Wait for the background task to complete
    desktop_hid = next(iter(desktop_actions))
    desktop_handle = desktop_actions[desktop_hid]["handle"]
    await desktop_handle.result()

    # After completion, the silent interjection should have fired
    mock_actor_handle.interject.assert_called_once()
    call_kwargs = mock_actor_handle.interject.call_args
    assert call_kwargs.kwargs.get("trigger_immediate_llm_turn") is False
    assert "Click Submit" in call_kwargs.args[0]

    # Clean up
    cm.in_flight_actions.pop(10, None)
    cm.in_flight_actions.pop(desktop_hid, None)
    cm._act_handles_with_desktop_usage.clear()


@pytest.mark.asyncio
@_handle_project
async def test_desktop_observe_returns_acting_and_interjects_on_completion(
    initialized_cm,
):
    """desktop_observe should return immediately with 'acting' status and silently
    interject in-flight act sessions once the background task completes."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm

    mock_actor_handle = MagicMock()
    mock_actor_handle.done.return_value = False
    mock_actor_handle.interject = AsyncMock()
    cm.in_flight_actions[11] = {
        "handle": mock_actor_handle,
        "query": "guide through app",
        "action_type": "act",
        "handle_actions": [],
    }
    cm._act_handles_with_desktop_usage = {11}

    mock_cp = MagicMock()
    mock_cp.desktop = MagicMock()
    mock_cp.desktop.observe = AsyncMock(return_value="Login page with username field")

    action_tools = ConversationManagerBrainActionTools(cm)

    with patch.object(
        type(cm),
        "computer_primitives",
        new_callable=lambda: property(lambda self: mock_cp),
    ):
        result = await action_tools.desktop_observe(query="What is on screen?")

    assert result["status"] == "acting"

    # Wait for the background task
    desktop_actions = {
        hid: data
        for hid, data in cm.in_flight_actions.items()
        if data.get("action_type") == "desktop_observe"
    }
    assert len(desktop_actions) == 1
    desktop_hid = next(iter(desktop_actions))
    await desktop_actions[desktop_hid]["handle"].result()

    mock_actor_handle.interject.assert_called_once()
    call_kwargs = mock_actor_handle.interject.call_args
    assert call_kwargs.kwargs.get("trigger_immediate_llm_turn") is False

    # Clean up
    cm.in_flight_actions.pop(11, None)
    cm.in_flight_actions.pop(desktop_hid, None)
    cm._act_handles_with_desktop_usage.clear()


@pytest.mark.asyncio
@_handle_project
async def test_desktop_get_screenshot_returns_acting(initialized_cm):
    """desktop_get_screenshot should return immediately with 'acting' status
    and register as an in-flight action."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm

    mock_image = MagicMock()
    mock_cp = MagicMock()
    mock_cp.desktop = MagicMock()
    mock_cp.desktop.get_screenshot = AsyncMock(return_value=mock_image)

    action_tools = ConversationManagerBrainActionTools(cm)

    with patch.object(
        type(cm),
        "computer_primitives",
        new_callable=lambda: property(lambda self: mock_cp),
    ):
        result = await action_tools.desktop_get_screenshot()

    assert result["status"] == "acting"

    desktop_actions = {
        hid: data
        for hid, data in cm.in_flight_actions.items()
        if data.get("action_type") == "desktop_get_screenshot"
    }
    assert len(desktop_actions) == 1

    # Wait for completion and verify
    desktop_hid = next(iter(desktop_actions))
    await desktop_actions[desktop_hid]["handle"].result()

    # Clean up
    cm.in_flight_actions.pop(desktop_hid, None)
