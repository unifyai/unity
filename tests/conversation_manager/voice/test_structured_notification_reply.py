"""
tests/conversation_manager/voice/test_structured_notification_reply.py
======================================================================

Tests for the structured notification reply path in call.py.

When a notification arrives during a voice call, a sidecar structured-output
LLM call ({speak: bool, content: str}) decides whether to speak. This replaces
the old generate_reply() path which relied on the model outputting empty text
to stay silent (brittle — models have a strong generative bias).
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from unity.conversation_manager.medium_scripts import call as call_script

# ---------------------------------------------------------------------------
# Fakes (shared with test_fast_brain_debounce.py pattern)
# ---------------------------------------------------------------------------


class _ImmediateAwaitable:
    def __await__(self):
        async def _done():
            return None

        return _done().__await__()


class _FakeRoom:
    name = "fake-room"

    def on(self, *args, **kwargs):
        return lambda fn: fn


class _FakeJobContext:
    def __init__(self):
        self.room = _FakeRoom()

    async def connect(self):
        return None

    def add_shutdown_callback(self, cb):
        pass

    def shutdown(self, reason=""):
        pass


class _FakeEventBroker:
    def __init__(self):
        self.callbacks = {}
        self.pattern_callbacks = []

    def set_logger(self, fb_logger):
        pass

    def register_callback(self, channel, handler):
        if "*" in channel or "?" in channel or "[" in channel:
            self.pattern_callbacks.append((channel, handler))
        else:
            self.callbacks[channel] = handler

    async def publish(self, channel, message):
        return 1


class _FakeSession:
    def __init__(self, *args, **kwargs):
        from livekit.agents import llm

        self._chat_ctx = llm.ChatContext()
        self.current_agent = None
        self._events = {}
        self.generate_reply_calls = 0
        self.say_calls = []
        self.agent_state = "listening"
        self.current_speech = None

    def on(self, event_name):
        def _decorator(fn):
            self._events[event_name] = fn
            return fn

        return _decorator

    async def start(self, room, agent, room_input_options=None):
        self.current_agent = agent

    def generate_reply(self, **kwargs):
        self.generate_reply_calls += 1
        return _ImmediateAwaitable()

    def say(self, text, **kwargs):
        self.say_calls.append(text)
        return _ImmediateAwaitable()


class _FakeAssistant:
    def __init__(self, *args, **kwargs):
        from livekit.agents import llm

        self._chat_ctx = llm.ChatContext()
        self.call_received = True

    def set_call_received(self):
        self.call_received = True


CONTACT = {
    "contact_id": 1,
    "first_name": "Dan",
    "surname": "Lenton",
    "phone_number": "+15550100001",
    "email_address": "dan@example.com",
}
BOSS = CONTACT


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def fast_brain_env(monkeypatch):
    """Set up a fake call.py entrypoint and return (session, broker)."""
    session_holder = {}
    fake_broker = _FakeEventBroker()

    async def _noop_async(*args, **kwargs):
        return None

    async def _noop_end_call():
        return None

    fake_session_details = SimpleNamespace(
        populate_from_env=lambda: None,
        voice=SimpleNamespace(provider="cartesia", id=""),
        voice_call=SimpleNamespace(
            outbound=False,
            channel="meet",
            contact_json=json.dumps(CONTACT),
            boss_json=json.dumps(BOSS),
        ),
        assistant=SimpleNamespace(about="Assistant bio", name="Olivia"),
    )

    class _CapturingSession(_FakeSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            session_holder["session"] = self

    monkeypatch.setattr(call_script, "event_broker", fake_broker)
    monkeypatch.setattr(call_script, "SESSION_DETAILS", fake_session_details)
    monkeypatch.setattr(call_script, "AgentSession", _CapturingSession)
    monkeypatch.setattr(call_script, "Assistant", _FakeAssistant)
    monkeypatch.setattr(call_script, "UnifyLLM", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        call_script,
        "build_voice_agent_prompt",
        lambda **kwargs: SimpleNamespace(flatten=lambda: "system prompt"),
    )
    monkeypatch.setattr(call_script, "start_event_broker_receive", _noop_async)
    monkeypatch.setattr(call_script, "publish_call_started", _noop_async)
    monkeypatch.setattr(
        call_script,
        "create_end_call",
        lambda *args, **kwargs: _noop_end_call,
    )
    monkeypatch.setattr(
        call_script,
        "setup_inactivity_timeout",
        lambda end_call: (lambda: None),
    )
    monkeypatch.setattr(
        call_script,
        "setup_participant_disconnect_handler",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(call_script, "RoomInputOptions", lambda **kwargs: object())
    monkeypatch.setattr(call_script, "EnglishModel", lambda: object())
    monkeypatch.setattr(call_script.cartesia, "TTS", lambda **kwargs: object())
    monkeypatch.setattr(call_script.elevenlabs, "TTS", lambda **kwargs: object())
    if hasattr(call_script, "noise_cancellation"):
        monkeypatch.setattr(call_script.noise_cancellation, "BVC", lambda: object())
    monkeypatch.setattr(call_script, "STT", object())
    monkeypatch.setattr(call_script, "VAD", object())

    await call_script.entrypoint(_FakeJobContext())

    session = session_holder["session"]
    yield session, fake_broker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_user_said(session, text: str):
    session._chat_ctx.add_message(role="user", content=[text])


def _simulate_assistant_said(session, text: str):
    session._chat_ctx.add_message(role="assistant", content=[text])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStructuredNotificationReply:
    """Verify that notifications use the structured-output evaluator and
    respect speak/no-speak decisions."""

    async def test_does_not_call_generate_reply(self, fast_brain_env):
        """Notifications should NOT trigger session.generate_reply()."""
        session, broker = fast_brain_env
        baseline_calls = session.generate_reply_calls

        notification_cb = broker.callbacks["app:call:notification"]

        mock_decision = AsyncMock()
        from unity.conversation_manager.domains.notification_reply import (
            NotificationReply,
        )

        mock_decision.return_value = (
            NotificationReply(speak=False, content=""),
            "/fake/log/path",
        )

        with patch(
            "unity.conversation_manager.domains.notification_reply."
            "NotificationReplyEvaluator.evaluate",
            mock_decision,
        ):
            notification_cb(
                {
                    "payload": {
                        "content": "Action update: browser is open",
                        "source": "system",
                    },
                },
            )
            await asyncio.sleep(0.3)

        assert session.generate_reply_calls == baseline_calls
        assert len(session.say_calls) == 0
        mock_decision.assert_called_once()

    async def test_speaks_when_decision_is_true(self, fast_brain_env):
        """When the evaluator says speak=True, content is spoken via session.say()."""
        session, broker = fast_brain_env

        notification_cb = broker.callbacks["app:call:notification"]

        mock_decision = AsyncMock()
        from unity.conversation_manager.domains.notification_reply import (
            NotificationReply,
        )

        mock_decision.return_value = (
            NotificationReply(speak=True, content="Found three results nearby."),
            "/fake/log/path",
        )

        with patch(
            "unity.conversation_manager.domains.notification_reply."
            "NotificationReplyEvaluator.evaluate",
            mock_decision,
        ):
            notification_cb(
                {
                    "payload": {
                        "content": "Action completed: search returned 3 results",
                        "source": "system",
                    },
                },
            )
            await asyncio.sleep(0.3)

        assert "Found three results nearby." in session.say_calls

    async def test_coalesces_rapid_notifications(self, fast_brain_env):
        """Rapid-fire notifications should coalesce into a single evaluator call."""
        session, broker = fast_brain_env

        notification_cb = broker.callbacks["app:call:notification"]

        call_count = 0
        from unity.conversation_manager.domains.notification_reply import (
            NotificationReply,
        )

        async def _counting_evaluate(self, chat_history, system_prompt):
            nonlocal call_count
            call_count += 1
            return NotificationReply(speak=False, content=""), ""

        with patch(
            "unity.conversation_manager.domains.notification_reply."
            "NotificationReplyEvaluator.evaluate",
            _counting_evaluate,
        ):
            notification_cb(
                {
                    "payload": {
                        "content": "Action started: desktop_act — open browser",
                        "source": "system",
                    },
                },
            )
            notification_cb(
                {
                    "payload": {
                        "content": "Action started: act — load guidance",
                        "source": "system",
                    },
                },
            )
            await asyncio.sleep(0.3)

        assert call_count == 1, (
            f"Rapid notifications should coalesce into one evaluator call, "
            f"got {call_count}."
        )

    async def test_triple_costar_scenario(self, fast_brain_env):
        """Simulate the 'triple CoStar' scenario with spaced notifications.
        The evaluator is called for each, but suppresses redundant speech."""
        session, broker = fast_brain_env

        notification_cb = broker.callbacks["app:call:notification"]

        from unity.conversation_manager.domains.notification_reply import (
            NotificationReply,
        )

        eval_call_count = 0

        async def _smart_evaluate(self, chat_history, system_prompt):
            nonlocal eval_call_count
            eval_call_count += 1
            if eval_call_count == 1:
                return (
                    NotificationReply(
                        speak=True,
                        content="On it — navigating to costar.com now.",
                    ),
                    "",
                )
            return NotificationReply(speak=False, content=""), ""

        with patch(
            "unity.conversation_manager.domains.notification_reply."
            "NotificationReplyEvaluator.evaluate",
            _smart_evaluate,
        ):
            _simulate_user_said(session, "So just go to costar dot com.")

            notification_cb(
                {
                    "payload": {
                        "content": "Action update: browser is open, Google start page.",
                        "source": "system",
                    },
                },
            )
            await asyncio.sleep(0.3)

            _simulate_assistant_said(
                session,
                "On it — navigating to costar.com now.",
            )

            notification_cb(
                {
                    "payload": {
                        "content": (
                            "Action started: desktop_act — Click address bar, "
                            "type costar.com, press Enter."
                        ),
                        "source": "system",
                    },
                },
            )
            await asyncio.sleep(0.3)

            notification_cb(
                {
                    "payload": {
                        "content": "Desktop action completed: navigated to costar.com",
                        "source": "system",
                    },
                },
            )
            await asyncio.sleep(0.3)

        assert eval_call_count == 3
        assert len(session.say_calls) == 1
        assert session.say_calls[0] == "On it — navigating to costar.com now."
