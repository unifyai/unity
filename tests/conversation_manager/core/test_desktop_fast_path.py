"""
tests/conversation_manager/core/test_desktop_fast_path.py
============================================================

Symbolic tests for the desktop fast-path tools on the ConversationManager.

Verifies:
- Tool surface: desktop_act, desktop_observe, desktop_get_screenshot appear/disappear
  based on gating conditions (assistant screen share + in-flight act with desktop usage).
- EventBus signal: DesktopPrimitiveInvoked is published for desktop method calls.
- Silent interjection: fast-path tools silently interject in-flight act sessions.
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
# Silent interjection
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_desktop_act_sends_silent_interjection(initialized_cm):
    """desktop_act should silently interject in-flight act sessions."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm

    mock_handle = MagicMock()
    mock_handle.done.return_value = False
    mock_handle.interject = AsyncMock()
    cm.in_flight_actions[10] = {
        "handle": mock_handle,
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

    assert result["status"] == "completed"
    assert result["result"] == "Clicked Submit"

    mock_handle.interject.assert_called_once()
    call_kwargs = mock_handle.interject.call_args
    assert call_kwargs.kwargs.get("trigger_immediate_llm_turn") is False
    assert "Click Submit" in call_kwargs.args[0]
    assert "Clicked Submit" in call_kwargs.args[0]

    # Clean up
    cm.in_flight_actions.pop(10, None)
    cm._act_handles_with_desktop_usage.clear()


@pytest.mark.asyncio
@_handle_project
async def test_desktop_observe_sends_silent_interjection(initialized_cm):
    """desktop_observe should silently interject in-flight act sessions."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm

    mock_handle = MagicMock()
    mock_handle.done.return_value = False
    mock_handle.interject = AsyncMock()
    cm.in_flight_actions[11] = {
        "handle": mock_handle,
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

    assert result["status"] == "completed"
    assert result["result"] == "Login page with username field"

    mock_handle.interject.assert_called_once()
    call_kwargs = mock_handle.interject.call_args
    assert call_kwargs.kwargs.get("trigger_immediate_llm_turn") is False

    # Clean up
    cm.in_flight_actions.pop(11, None)
    cm._act_handles_with_desktop_usage.clear()


@pytest.mark.asyncio
@_handle_project
async def test_desktop_get_screenshot_no_interjection(initialized_cm):
    """desktop_get_screenshot should NOT interject (read-only, no side effects)."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm

    mock_handle = MagicMock()
    mock_handle.done.return_value = False
    mock_handle.interject = AsyncMock()
    cm.in_flight_actions[12] = {
        "handle": mock_handle,
        "query": "guide through app",
        "action_type": "act",
        "handle_actions": [],
    }
    cm._act_handles_with_desktop_usage = {12}

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

    assert result["status"] == "completed"
    assert result["image"] is mock_image

    mock_handle.interject.assert_not_called()

    # Clean up
    cm.in_flight_actions.pop(12, None)
    cm._act_handles_with_desktop_usage.clear()
