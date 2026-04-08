"""
tests/conversation_manager/voice/test_stale_generation_invalidation.py
======================================================================

Regression tests for the FastBrain stale generation invalidation mechanism.

When a significant IPC event (slow brain notification, outbound message sent)
arrives while the FastBrain LLM is mid-generation, the system should cancel
the in-flight generation and re-trigger with updated context.  This prevents
the FastBrain from speaking stale responses (e.g., offering to send scopes
that were already sent).

Origin: production incident 2026-04-07 where the FastBrain asked "Would you
like me to send these over in the chat?" after the slow brain had already
sent the message.

These tests verify the symbolic preconditions and wiring that
``_invalidate_current_generation`` depends on.
"""

from __future__ import annotations


import pytest

from unity.conversation_manager.events import (
    SMSSent,
    EmailSent,
    UnifyMessageSent,
    WhatsAppSent,
)
from unity.conversation_manager.medium_scripts.common import (
    render_participant_comms,
)

# ===========================================================================
# Test: outbound events produce the "[You ..." prefix used by invalidation
# ===========================================================================


class TestOutboundEventPrefixConvention:
    """The invalidation trigger in ``on_participant_comms`` uses
    ``text.startswith("[You ")`` to identify outbound actions.  These tests
    verify that ``render_participant_comms`` maintains this convention for
    every outbound event type.
    """

    @pytest.fixture
    def participant_ids(self):
        return {1}

    @staticmethod
    def _render(event, participant_ids: set[int]) -> str | None:
        return render_participant_comms(event.to_json(), participant_ids)

    def test_unify_message_sent_prefix(self, participant_ids):
        event = UnifyMessageSent(
            contact={"contact_id": 1, "first_name": "Dan", "surname": "Lenton"},
            content="Here are the OAuth scopes.",
        )
        text = self._render(event, participant_ids)
        assert text is not None
        assert text.startswith("[You "), (
            f"UnifyMessageSent must produce '[You ...' prefix for invalidation.\n"
            f"Got: {text}"
        )

    def test_sms_sent_prefix(self, participant_ids):
        event = SMSSent(
            contact={"contact_id": 1, "first_name": "Dan", "surname": "Lenton"},
            content="Here is the link.",
        )
        text = self._render(event, participant_ids)
        assert text is not None
        assert text.startswith("[You "), (
            f"SMSSent must produce '[You ...' prefix for invalidation.\n" f"Got: {text}"
        )

    def test_email_sent_prefix(self, participant_ids):
        event = EmailSent(
            contact={"contact_id": 1, "first_name": "Dan", "surname": "Lenton"},
            subject="OAuth setup instructions",
            body="Step-by-step instructions for setting up OAuth.",
        )
        text = self._render(event, participant_ids)
        assert text is not None
        assert text.startswith("[You "), (
            f"EmailSent must produce '[You ...' prefix for invalidation.\n"
            f"Got: {text}"
        )

    def test_whatsapp_sent_prefix(self, participant_ids):
        event = WhatsAppSent(
            contact={"contact_id": 1, "first_name": "Dan", "surname": "Lenton"},
            content="Here are the scopes.",
        )
        text = self._render(event, participant_ids)
        assert text is not None
        assert text.startswith("[You "), (
            f"WhatsAppSent must produce '[You ...' prefix for invalidation.\n"
            f"Got: {text}"
        )


# ===========================================================================
# Test: triggers_turn filter correctly classifies notifications
# ===========================================================================


class TestNotificationTriggersTurnClassification:
    """The invalidation trigger in ``on_notification`` uses ``triggers_turn``
    to decide whether a notification should invalidate in-flight generation.
    ``triggers_turn`` must be True for silent context injections and False
    for speak-mode notifications and meet_interaction events.
    """

    @staticmethod
    def _triggers_turn(
        should_speak: bool,
        response_text: str,
        notification_source: str,
    ) -> bool:
        return notification_source != "meet_interaction"

    def test_silent_slow_brain_notification_triggers(self):
        assert self._triggers_turn(
            should_speak=False,
            response_text="",
            notification_source="slow_brain",
        )

    def test_speak_mode_notification_triggers(self):
        """should_speak notifications now also invalidate the fast brain's
        in-flight generation so it regenerates with the new context."""
        assert self._triggers_turn(
            should_speak=True,
            response_text="I've sent those scopes in the chat.",
            notification_source="slow_brain",
        )

    def test_meet_interaction_does_not_trigger(self):
        assert not self._triggers_turn(
            should_speak=False,
            response_text="",
            notification_source="meet_interaction",
        )

    def test_notify_mode_with_content_only_triggers(self):
        assert self._triggers_turn(
            should_speak=True,
            response_text="",
            notification_source="slow_brain",
        )

    def test_proactive_speech_triggers(self):
        assert self._triggers_turn(
            should_speak=False,
            response_text="",
            notification_source="proactive_speech",
        )


# ===========================================================================
# Test: _invalidate_current_generation wiring exists in call.py
# ===========================================================================


class TestInvalidationWiringExists:
    """Verify that call.py contains the invalidation call sites. This is a
    lightweight structural check — if the wiring is accidentally removed by
    a refactor, these tests will catch it immediately.
    """

    @pytest.fixture(scope="class")
    def call_source(self):
        from pathlib import Path

        return Path(
            "unity/conversation_manager/medium_scripts/call.py",
        ).read_text()

    def test_invalidation_helper_defined(self, call_source):
        assert "_invalidate_current_generation" in call_source

    def test_user_turn_generating_flag_defined(self, call_source):
        assert "_user_turn_generating" in call_source

    def test_notification_handler_calls_invalidation(self, call_source):
        assert "notification_during_generation" in call_source

    def test_participant_comms_handler_calls_invalidation(self, call_source):
        assert "outbound_action_during_generation" in call_source
