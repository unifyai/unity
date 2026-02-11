"""
tests/conversation_manager/actions/test_persist_action.py
==========================================================

Tests for persistent ``act`` sessions (``persist=True``), including:

- The ``persist`` parameter on the CM's ``act`` tool
- Persistent action metadata and rendering
- ``ActorResponse`` vs ``ActorNotification`` event routing
- Event handlers for both event types
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)
from unity.conversation_manager.domains.contact_index import ContactIndex
from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.domains.renderer import Renderer
from unity.conversation_manager.events import (
    ActorNotification,
    ActorSessionResponse,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_cm():
    """Minimal mock ConversationManager for unit-level tests."""
    cm = MagicMock()
    cm.mode = "text"
    cm.contact_index = ContactIndex()
    cm.in_flight_actions = {}
    cm.completed_actions = {}
    cm.notifications_bar = NotificationBar()
    cm.chat_history = []
    cm._current_state_snapshot = None
    cm._current_snapshot_state = None
    cm._pending_steering_tasks = set()
    cm._initialized = asyncio.Event()
    cm._initialized.set()
    cm._session_logger = MagicMock()
    cm.request_llm_run = AsyncMock()
    cm.event_broker = MagicMock()
    cm.event_broker.publish = AsyncMock()
    cm.call_manager = MagicMock()
    cm.call_manager.uses_realtime_api = False
    return cm


@pytest.fixture
def brain_action_tools(mock_cm):
    """ConversationManagerBrainActionTools wired to the mock CM."""
    with patch(
        "unity.conversation_manager.domains.brain_action_tools.get_event_broker",
    ) as mock_broker:
        mock_broker.return_value = MagicMock()
        mock_broker.return_value.publish = AsyncMock()
        yield ConversationManagerBrainActionTools(mock_cm)


def _make_fake_actor():
    """Create a mock Actor whose act() returns a controllable handle."""
    captured_kwargs: dict[str, Any] = {}

    async def fake_act(request, **kwargs):
        captured_kwargs.update(kwargs)
        handle = MagicMock()
        handle.result = AsyncMock(return_value="done")
        handle.next_notification = AsyncMock(side_effect=asyncio.CancelledError)
        handle.next_clarification = AsyncMock(side_effect=asyncio.CancelledError)
        return handle

    actor = MagicMock()
    actor.act = fake_act
    return actor, captured_kwargs


# ═════════════════════════════════════════════════════════════════════════════
# 1. act tool — persist parameter
# ═════════════════════════════════════════════════════════════════════════════


class TestActPersistParameter:
    """Tests for the persist parameter on the act tool."""

    def test_persist_in_signature(self, brain_action_tools):
        """act() signature includes persist as a bool with default False."""
        sig = inspect.signature(brain_action_tools.act)
        assert "persist" in sig.parameters
        param = sig.parameters["persist"]
        assert param.default is False

    @pytest.mark.asyncio
    async def test_persist_true_forwarded_to_actor(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """When persist=True, it is forwarded to Actor.act()."""
        actor, captured = _make_fake_actor()
        mock_cm.actor = actor

        await brain_action_tools.act(query="Guide onboarding", persist=True)

        assert captured.get("persist") is True

    @pytest.mark.asyncio
    async def test_persist_false_forwarded_to_actor(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """When persist=False (default), it is forwarded as False."""
        actor, captured = _make_fake_actor()
        mock_cm.actor = actor

        await brain_action_tools.act(query="Find contacts")

        assert captured.get("persist") is False

    @pytest.mark.asyncio
    async def test_persist_stored_in_action_metadata(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """The persist flag is stored in the in_flight_actions metadata."""
        actor, _ = _make_fake_actor()
        mock_cm.actor = actor

        await brain_action_tools.act(query="Long session", persist=True)

        assert len(mock_cm.in_flight_actions) == 1
        action_data = next(iter(mock_cm.in_flight_actions.values()))
        assert action_data["persist"] is True

    @pytest.mark.asyncio
    async def test_non_persistent_action_has_persist_false(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Non-persistent actions store persist=False in metadata."""
        actor, _ = _make_fake_actor()
        mock_cm.actor = actor

        await brain_action_tools.act(query="Quick task")

        action_data = next(iter(mock_cm.in_flight_actions.values()))
        assert action_data["persist"] is False


# ═════════════════════════════════════════════════════════════════════════════
# 2. Event handlers — ActorResponse and ActorNotification
# ═════════════════════════════════════════════════════════════════════════════


class TestActorSessionResponseHandler:
    """Tests for the ActorSessionResponse event handler."""

    @pytest.mark.asyncio
    async def test_response_does_not_create_notification(self, mock_cm):
        """ActorSessionResponse records in handle_actions, no notification pushed."""
        mock_cm.in_flight_actions = {
            1: {"query": "Onboarding session", "handle_actions": []},
        }
        event = ActorSessionResponse(handle_id=1, content="Screen is ready. What next?")

        await EventHandler.handle_event(event, mock_cm)

        assert len(mock_cm.notifications_bar.notifications) == 0

    @pytest.mark.asyncio
    async def test_response_recorded_in_handle_actions(self, mock_cm):
        """ActorSessionResponse is recorded in handle_actions with awaiting_input status."""
        mock_cm.in_flight_actions = {
            1: {"query": "Onboarding session", "handle_actions": []},
        }
        event = ActorSessionResponse(handle_id=1, content="Ready for step 2")

        await EventHandler.handle_event(event, mock_cm)

        actions = mock_cm.in_flight_actions[1]["handle_actions"]
        assert len(actions) == 1
        assert actions[0]["action_name"] == "response"
        assert actions[0]["status"] == "awaiting_input"
        assert "Ready for step 2" in actions[0]["query"]

    @pytest.mark.asyncio
    async def test_response_wakes_brain(self, mock_cm):
        """ActorSessionResponse triggers an LLM run so the brain can process it."""
        mock_cm.in_flight_actions = {
            1: {"query": "Session", "handle_actions": []},
        }
        event = ActorSessionResponse(handle_id=1, content="Done")

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called()


class TestActorNotificationHandler:
    """Tests for the ActorNotification event handler."""

    @pytest.mark.asyncio
    async def test_notification_records_progress_in_handle_actions(self, mock_cm):
        """ActorNotification records progress in handle_actions, no notification pushed."""
        mock_cm.in_flight_actions = {
            1: {"query": "Long task", "handle_actions": []},
        }
        event = ActorNotification(handle_id=1, response="Processing 50%...")

        await EventHandler.handle_event(event, mock_cm)

        assert len(mock_cm.notifications_bar.notifications) == 0
        actions = mock_cm.in_flight_actions[1]["handle_actions"]
        assert len(actions) == 1
        assert actions[0]["action_name"] == "progress"
        assert "Processing 50%" in actions[0]["query"]

    @pytest.mark.asyncio
    async def test_notification_wakes_brain(self, mock_cm):
        """ActorNotification triggers an LLM run."""
        event = ActorNotification(handle_id=1, response="Working...")

        await EventHandler.handle_event(event, mock_cm)

        mock_cm.request_llm_run.assert_called()


# ═════════════════════════════════════════════════════════════════════════════
# 3. Rendering — persistent actions
# ═════════════════════════════════════════════════════════════════════════════


class TestPersistentActionRendering:
    """Tests for how persistent actions are rendered in the state snapshot."""

    @pytest.fixture
    def renderer(self):
        return Renderer()

    def test_persistent_action_has_mode_attribute(self, renderer):
        """Persistent actions render with mode='persistent' attribute."""
        in_flight = {
            0: {
                "handle": MagicMock(),
                "query": "Guided onboarding",
                "persist": True,
                "handle_actions": [],
            },
        }

        result = renderer.render_in_flight_actions(in_flight)

        assert "mode='persistent'" in result

    def test_non_persistent_action_lacks_mode_attribute(self, renderer):
        """Non-persistent actions do not have a mode attribute."""
        in_flight = {
            0: {
                "handle": MagicMock(),
                "query": "Quick search",
                "persist": False,
                "handle_actions": [],
            },
        }

        result = renderer.render_in_flight_actions(in_flight)

        assert "mode='persistent'" not in result

    def test_persistent_action_includes_note(self, renderer):
        """Persistent actions include a note about not self-completing."""
        in_flight = {
            0: {
                "handle": MagicMock(),
                "query": "Long session",
                "persist": True,
                "handle_actions": [],
            },
        }

        result = renderer.render_in_flight_actions(in_flight)

        assert "will NOT self-complete" in result
        assert "stop_*" in result

    def test_response_event_rendered_with_awaiting_input(self, renderer):
        """Response events in history show awaiting_input status."""
        in_flight = {
            0: {
                "handle": MagicMock(),
                "query": "Session",
                "persist": True,
                "handle_actions": [
                    {
                        "action_name": "act_started",
                        "query": "Session",
                        "timestamp": "2025-01-01 12:00",
                    },
                    {
                        "action_name": "response",
                        "query": "Screen is ready",
                        "status": "awaiting_input",
                        "timestamp": "2025-01-01 12:01",
                    },
                ],
            },
        }

        result = renderer.render_in_flight_actions(in_flight)

        assert "type='response'" in result
        assert "status='awaiting_input'" in result
        assert "Screen is ready" in result


# ═════════════════════════════════════════════════════════════════════════════
# 4. Notification routing — ActorResponse vs ActorNotification
# ═════════════════════════════════════════════════════════════════════════════


class TestNotificationRouting:
    """Tests for actor_watch_notifications routing logic."""

    @pytest.mark.asyncio
    async def test_response_type_publishes_actor_session_response(self):
        """Notifications with type='response' are published as ActorSessionResponse."""
        from unity.conversation_manager.domains.managers_utils import (
            actor_watch_notifications,
        )
        from unity.conversation_manager.events import Event

        handle = MagicMock()
        call_count = 0

        async def fake_next_notification():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"type": "response", "content": "Turn complete"}
            raise asyncio.CancelledError

        handle.next_notification = fake_next_notification
        handle.done = MagicMock(side_effect=[False, True])

        with patch(
            "unity.conversation_manager.domains.managers_utils.event_broker",
        ) as mock_broker:
            mock_broker.publish = AsyncMock()

            await actor_watch_notifications(handle_id=1, handle=handle)

            # Should have published to app:actor:session_response channel
            assert mock_broker.publish.called
            channel = mock_broker.publish.call_args[0][0]
            assert channel == "app:actor:session_response"
            # Verify the event payload
            payload_json = mock_broker.publish.call_args[0][1]
            evt = Event.from_json(payload_json)
            assert isinstance(evt, ActorSessionResponse)
            assert evt.content == "Turn complete"

    @pytest.mark.asyncio
    async def test_notification_type_publishes_actor_notification(self):
        """Regular notifications are published as ActorNotification."""
        from unity.conversation_manager.domains.managers_utils import (
            actor_watch_notifications,
        )
        from unity.conversation_manager.events import Event

        handle = MagicMock()
        call_count = 0

        async def fake_next_notification():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"type": "notification", "message": "Working..."}
            raise asyncio.CancelledError

        handle.next_notification = fake_next_notification
        handle.done = MagicMock(side_effect=[False, True])

        with patch(
            "unity.conversation_manager.domains.managers_utils.event_broker",
        ) as mock_broker:
            mock_broker.publish = AsyncMock()

            await actor_watch_notifications(handle_id=1, handle=handle)

            assert mock_broker.publish.called
            channel = mock_broker.publish.call_args[0][0]
            assert channel == "app:actor:notification"
            payload_json = mock_broker.publish.call_args[0][1]
            evt = Event.from_json(payload_json)
            assert isinstance(evt, ActorNotification)
            assert evt.response == "Working..."
