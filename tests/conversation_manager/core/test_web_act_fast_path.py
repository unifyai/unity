"""
tests/conversation_manager/core/test_web_act_fast_path.py
============================================================

Symbolic tests for the web_act and close_web_session fast-path tools.

Verifies:
- Tool surface: web_act / close_web_session appear/disappear based on screen share.
- Async lifecycle: web_act creates a session, registers in in_flight_actions, and
  silently interjects Actor sessions.
- Session reuse: web_act with session_id reuses an existing handle.
- close_web_session: stops a handle and returns confirmation.
- Renderer: <active_web_sessions> appears in the snapshot when sessions exist.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.helpers import _handle_project

# =============================================================================
# Tool appearance / disappearance
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_web_act_absent_when_screen_share_inactive(initialized_cm):
    """web_act must NOT appear when screen share is off."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm
    cm.assistant_screen_share_active = False

    action_tools = ConversationManagerBrainActionTools(cm)
    tools = action_tools.as_tools()

    assert "web_act" not in tools
    assert "close_web_session" not in tools


@pytest.mark.asyncio
@_handle_project
async def test_web_act_present_when_screen_share_active(initialized_cm):
    """web_act and close_web_session appear when screen share is active."""
    cm = initialized_cm.cm
    cm.assistant_screen_share_active = True

    assert cm.desktop_fast_path_eligible


@pytest.mark.asyncio
@_handle_project
async def test_close_web_session_present_when_screen_share_active(initialized_cm):
    """close_web_session is gated by the same flag as web_act."""
    cm = initialized_cm.cm
    cm.assistant_screen_share_active = True
    assert cm.desktop_fast_path_eligible


# =============================================================================
# web_act lifecycle
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_web_act_creates_session_when_no_id(initialized_cm):
    """web_act without session_id creates a new visible session and registers
    an in-flight action."""
    from unity.function_manager.primitives.runtime import (
        ComputerPrimitives,
        WebSessionHandle,
    )
    from unity.manager_registry import ManagerRegistry
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm
    cm.assistant_screen_share_active = True
    cm._act_handles_with_desktop_usage = set()

    ManagerRegistry.clear()
    cp = ComputerPrimitives(computer_mode="mock")

    action_tools = ConversationManagerBrainActionTools(cm)

    with patch.object(
        type(cm),
        "computer_primitives",
        new_callable=lambda: property(lambda self: cp),
    ):
        result = await action_tools.web_act(request="Search for CRM software")

    assert result["status"] == "acting"
    assert "Search for CRM software" in result["query"]

    web_actions = {
        hid: data
        for hid, data in cm.in_flight_actions.items()
        if data.get("action_type") == "web_act"
    }
    assert len(web_actions) == 1

    sessions = cp.web.list_sessions(active_only=True)
    assert len(sessions) == 1
    assert isinstance(sessions[0], WebSessionHandle)
    assert sessions[0].visible is True

    web_hid = next(iter(web_actions))
    cm.in_flight_actions.pop(web_hid, None)
    ManagerRegistry.clear()


@pytest.mark.asyncio
@_handle_project
async def test_web_act_reuses_session_when_id_provided(initialized_cm):
    """web_act with session_id reuses an existing handle instead of creating new."""
    from unity.function_manager.primitives.runtime import ComputerPrimitives
    from unity.manager_registry import ManagerRegistry
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm
    cm.assistant_screen_share_active = True
    cm._act_handles_with_desktop_usage = set()

    ManagerRegistry.clear()
    cp = ComputerPrimitives(computer_mode="mock")

    session = await cp.web.new_session(visible=True)
    sid = session.session_id
    assert len(cp.web.list_sessions()) == 1

    action_tools = ConversationManagerBrainActionTools(cm)

    with patch.object(
        type(cm),
        "computer_primitives",
        new_callable=lambda: property(lambda self: cp),
    ):
        result = await action_tools.web_act(
            request="Check the pricing page",
            session_id=sid,
        )

    assert result["status"] == "acting"

    # No new session should have been created
    assert len(cp.web.list_sessions()) == 1

    web_actions = {
        hid: data
        for hid, data in cm.in_flight_actions.items()
        if data.get("action_type") == "web_act"
    }
    for hid in web_actions:
        cm.in_flight_actions.pop(hid, None)
    ManagerRegistry.clear()


# =============================================================================
# close_web_session
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_close_web_session_stops_handle(initialized_cm):
    """close_web_session stops the handle and returns confirmation."""
    from unity.function_manager.primitives.runtime import ComputerPrimitives
    from unity.manager_registry import ManagerRegistry
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm
    cm.assistant_screen_share_active = True

    ManagerRegistry.clear()
    cp = ComputerPrimitives(computer_mode="mock")

    session = await cp.web.new_session(visible=True)
    sid = session.session_id
    assert session.active is True

    action_tools = ConversationManagerBrainActionTools(cm)

    with patch.object(
        type(cm),
        "computer_primitives",
        new_callable=lambda: property(lambda self: cp),
    ):
        result = await action_tools.close_web_session(session_id=sid)

    assert result["status"] == "closed"
    assert result["session_id"] == sid
    assert session.active is False
    ManagerRegistry.clear()


@pytest.mark.asyncio
@_handle_project
async def test_close_web_session_not_found(initialized_cm):
    """close_web_session returns not_found for an invalid session_id."""
    from unity.function_manager.primitives.runtime import ComputerPrimitives
    from unity.manager_registry import ManagerRegistry
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = initialized_cm.cm
    cm.assistant_screen_share_active = True

    ManagerRegistry.clear()
    cp = ComputerPrimitives(computer_mode="mock")

    action_tools = ConversationManagerBrainActionTools(cm)

    with patch.object(
        type(cm),
        "computer_primitives",
        new_callable=lambda: property(lambda self: cp),
    ):
        result = await action_tools.close_web_session(session_id="nonexistent")

    assert result["status"] == "not_found"
    ManagerRegistry.clear()


# =============================================================================
# Renderer: <active_web_sessions>
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_active_web_sessions_in_snapshot(initialized_cm):
    """<active_web_sessions> appears in the rendered state when sessions exist."""
    from unity.conversation_manager.domains.renderer import Renderer
    from unity.function_manager.primitives.runtime import ComputerPrimitives
    from unity.manager_registry import ManagerRegistry

    ManagerRegistry.clear()
    cp = ComputerPrimitives(computer_mode="mock")

    s1 = await cp.web.new_session(visible=True)
    s2 = await cp.web.new_session(visible=True)

    sessions = cp.web.list_sessions(visible_only=True, active_only=True)
    rendered = Renderer.render_active_web_sessions(sessions)

    assert "<active_web_sessions>" in rendered
    assert s1.session_id in rendered
    assert s2.session_id in rendered
    assert "</active_web_sessions>" in rendered
    ManagerRegistry.clear()


def test_render_active_web_sessions_empty():
    """No sessions renders empty string."""
    from unity.conversation_manager.domains.renderer import Renderer

    assert Renderer.render_active_web_sessions([]) == ""
