"""
tests/conversation_manager/voice/test_structured_notification_reply.py
======================================================================

A/B tests for structured notification reply vs. baseline generate_reply.

The structured path uses a sidecar LLM call with response_format={speak, content}
to let the model explicitly decide whether to speak. The baseline path fires
session.generate_reply() for every notification, relying on the model to output
empty text (brittle).

Scenario under test: the "triple CoStar" pattern from the 2026-03-02 demo where
three notifications in quick succession each triggered speech saying essentially
the same thing ("navigating to costar.com").
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


def _chat_ctx_texts(ctx) -> list[str]:
    return [
        item.text_content or ""
        for item in ctx.items
        if getattr(item, "type", None) == "message"
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def structured_env(monkeypatch):
    """Set up call.py with FAST_BRAIN_STRUCTURED_NOTIFICATION_REPLY=True."""
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

    # Enable structured notification reply for this env
    monkeypatch.setattr(
        "unity.settings.SETTINGS.conversation.FAST_BRAIN_STRUCTURED_NOTIFICATION_REPLY",
        True,
    )

    await call_script.entrypoint(_FakeJobContext())

    session = session_holder["session"]
    yield session, fake_broker


@pytest_asyncio.fixture
async def baseline_env(monkeypatch):
    """Set up call.py with FAST_BRAIN_STRUCTURED_NOTIFICATION_REPLY=False (baseline)."""
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

    # Ensure structured notification reply is OFF for baseline
    monkeypatch.setattr(
        "unity.settings.SETTINGS.conversation.FAST_BRAIN_STRUCTURED_NOTIFICATION_REPLY",
        False,
    )

    await call_script.entrypoint(_FakeJobContext())

    session = session_holder["session"]
    yield session, fake_broker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_user_said(session, assistant_cls, text: str):
    """Inject a user message into both chat contexts."""
    session._chat_ctx.add_message(role="user", content=[text])
    # assistant._chat_ctx is separate in real code but we only need session for evaluation


def _simulate_assistant_said(session, text: str):
    """Inject an assistant message into chat context."""
    session._chat_ctx.add_message(role="assistant", content=[text])


# ---------------------------------------------------------------------------
# A: Baseline tests (generate_reply for every notification)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBaselineNotificationBehavior:
    """Verify that baseline mode fires generate_reply for every notification."""

    async def test_triple_notification_fires_three_generate_replies(
        self,
        baseline_env,
    ):
        """The 'triple CoStar' scenario: three spaced notifications each trigger
        generate_reply, producing three potential speech outputs."""
        session, broker = baseline_env
        baseline_calls = session.generate_reply_calls

        notification_cb = broker.callbacks["app:call:notification"]

        # User says "go to costar.com"
        _simulate_user_said(session, None, "So just go to costar dot com.")

        # Notification 1: action update from CodeActActor
        notification_cb(
            {
                "payload": {
                    "content": (
                        "Action update: The browser is now open. "
                        "I'm ready for the next instruction."
                    ),
                    "source": "system",
                },
            },
        )
        await asyncio.sleep(0.15)

        # Notification 2: desktop_act started
        notification_cb(
            {
                "payload": {
                    "content": (
                        "Action started: desktop_act — Click on the address bar, "
                        "type costar.com and press Enter."
                    ),
                    "source": "system",
                },
            },
        )
        await asyncio.sleep(0.15)

        # Notification 3: desktop_act completed
        notification_cb(
            {
                "payload": {
                    "content": "Desktop action completed: navigated to costar.com",
                    "source": "system",
                },
            },
        )
        await asyncio.sleep(0.15)

        new_calls = session.generate_reply_calls - baseline_calls
        assert new_calls == 3, (
            f"Baseline should fire generate_reply for each notification. "
            f"Got {new_calls} calls instead of 3."
        )
        # No say_calls expected — generate_reply doesn't go through session.say
        assert len(session.say_calls) == 0


# ---------------------------------------------------------------------------
# B: Structured notification reply tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStructuredNotificationReply:
    """Verify that structured mode uses the sidecar LLM evaluator instead of
    generate_reply, and respects speak/no-speak decisions."""

    async def test_structured_path_does_not_call_generate_reply(
        self,
        structured_env,
    ):
        """When structured notification reply is enabled, notifications should
        NOT trigger session.generate_reply()."""
        session, broker = structured_env
        baseline_calls = session.generate_reply_calls

        notification_cb = broker.callbacks["app:call:notification"]

        # Mock the evaluator to return speak=False
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
            # Wait for debounce + async evaluation
            await asyncio.sleep(0.3)

        # generate_reply should NOT have been called
        assert session.generate_reply_calls == baseline_calls, (
            f"Structured path should not call generate_reply. "
            f"Got {session.generate_reply_calls - baseline_calls} extra calls."
        )
        # No speech either
        assert len(session.say_calls) == 0
        # But the evaluator was called
        mock_decision.assert_called_once()

    async def test_structured_path_speaks_when_decision_is_true(
        self,
        structured_env,
    ):
        """When the evaluator says speak=True, the content should be spoken
        via session.say()."""
        session, broker = structured_env

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

    async def test_structured_path_coalesces_rapid_notifications(
        self,
        structured_env,
    ):
        """Rapid-fire notifications should coalesce into a single evaluator call."""
        session, broker = structured_env

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
            # Two rapid notifications
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

    async def test_triple_costar_scenario_structured(
        self,
        structured_env,
    ):
        """Simulate the 'triple CoStar' scenario with spaced notifications.
        The evaluator should be called for each, but mock it to suppress
        redundant speech (speak=False for 2nd and 3rd)."""
        session, broker = structured_env

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
            # Subsequent calls: model recognizes redundancy
            return NotificationReply(speak=False, content=""), ""

        with patch(
            "unity.conversation_manager.domains.notification_reply."
            "NotificationReplyEvaluator.evaluate",
            _smart_evaluate,
        ):
            # Simulate user request
            _simulate_user_said(session, None, "So just go to costar dot com.")

            # Notification 1: action update
            notification_cb(
                {
                    "payload": {
                        "content": "Action update: browser is open, DuckDuckGo start page.",
                        "source": "system",
                    },
                },
            )
            await asyncio.sleep(0.3)

            # Simulate that the first notification produced speech
            _simulate_assistant_said(
                session,
                "On it — navigating to costar.com now.",
            )

            # Notification 2: desktop_act started
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

            # Notification 3: desktop_act completed
            notification_cb(
                {
                    "payload": {
                        "content": "Desktop action completed: navigated to costar.com",
                        "source": "system",
                    },
                },
            )
            await asyncio.sleep(0.3)

        assert eval_call_count == 3, (
            f"Each spaced notification should trigger one evaluator call, "
            f"got {eval_call_count}."
        )
        assert len(session.say_calls) == 1, (
            f"Only the first notification should produce speech. "
            f"Got {len(session.say_calls)} say calls: {session.say_calls}"
        )
        assert session.say_calls[0] == "On it — navigating to costar.com now."
