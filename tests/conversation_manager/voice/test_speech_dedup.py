"""
tests/conversation_manager/voice/test_speech_dedup.py
=====================================================

Tests for the speech deduplication gate.

Dedup runs in the fast brain subprocess at speak time (inside
``maybe_speak_queued`` → ``_dedup_and_speak``).  Before playing queued slow
brain speech, a lightweight LLM check compares the proposed text against recent
assistant utterances in the fast brain's chat context and suppresses it when the
information has already been communicated.

Test categories:

1. **Unit tests** — SpeechDeduplicationChecker in isolation.
2. **Symbolic integration tests** — verify the slow brain no longer runs dedup
   and passes ``should_speak`` through to the fast brain unmodified.
3. **Eval tests** — end-to-end with real LLM judgment on overlapping content.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.conversation_manager.domains.speech_dedup import (
    SpeechDedup,
    SpeechDeduplicationChecker,
)
from unity.conversation_manager.events import (
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    PhoneCallStarted,
    UnifyMeetReceived,
    UnifyMeetStarted,
)
from unity.conversation_manager.cm_types import Medium, Mode

from tests.conversation_manager.conftest import BOSS, TEST_CONTACTS
from tests.helpers import _handle_project

# =============================================================================
# Unit tests — SpeechDeduplicationChecker in isolation
# =============================================================================


@pytest.mark.asyncio
class TestSpeechDeduplicationCheckerUnit:

    async def test_empty_utterances_skips_check(self):
        """When there are no recent utterances, the checker returns
        already_covered=False without making an LLM call."""
        checker = SpeechDeduplicationChecker()

        result = await checker.evaluate(
            proposed_speech="The task is complete.",
            recent_utterances=[],
        )

        assert isinstance(result, SpeechDedup)
        assert result.already_covered is False

    async def test_evaluate_returns_structured_output(self):
        """With recent utterances and proposed speech, the evaluator makes
        an LLM call and returns a valid SpeechDedup result.

        Uses FAST_BRAIN_MODEL for a cheaper round-trip in default CI.
        Production ``SpeechDeduplicationChecker()`` uses ``UNIFY_MODEL``
        (Anthropic); see ``test_evaluate_sends_user_role_message`` for the
        message-shape contract Anthropic requires, and the eval test for a
        full default-model call.
        """
        from unity.settings import SETTINGS

        checker = SpeechDeduplicationChecker(
            model=SETTINGS.conversation.FAST_BRAIN_MODEL,
        )

        result = await checker.evaluate(
            proposed_speech="Found three Italian restaurants nearby.",
            recent_utterances=[
                "Yeah, I found three Italian places near you — the top rated is Chez Laurent.",
            ],
        )

        assert isinstance(result, SpeechDedup)
        assert isinstance(result.already_covered, bool)
        assert isinstance(result.reasoning, str)

    async def test_evaluate_error_fails_open(self):
        """On LLM error, the checker returns already_covered=False so speech
        is allowed rather than silently suppressed."""
        checker = SpeechDeduplicationChecker(model="invalid-model@nowhere")

        result = await checker.evaluate(
            proposed_speech="The task is done.",
            recent_utterances=["Done with the task."],
        )

        assert result.already_covered is False
        assert "failed" in result.reasoning.lower()

    async def test_evaluate_sends_user_role_message(self):
        """LiteLLM/Anthropic reject chat completions with only ``system`` messages.

        ``SpeechDeduplicationChecker()`` defaults to ``UNIFY_MODEL`` (Anthropic in
        prod). This test does not call the API: it asserts we always include at
        least one non-system message so provider transforms do not empty the payload.
        """
        captured: dict = {}

        mock_client = MagicMock()
        mock_client.set_response_format = MagicMock()

        async def capture_generate(*, messages=None, **_kwargs):
            captured["messages"] = messages
            return '{"already_covered": false, "reasoning": "ok"}'

        mock_client.generate = AsyncMock(side_effect=capture_generate)

        with patch(
            "unity.conversation_manager.domains.speech_dedup.new_llm_client",
            return_value=mock_client,
        ):
            checker = SpeechDeduplicationChecker(model="claude-3-5-haiku@anthropic")
            result = await checker.evaluate(
                proposed_speech="The report is ready.",
                recent_utterances=["I already told you the report is ready."],
            )

        assert result.already_covered is False
        assert captured.get("messages") is not None
        roles = [m["role"] for m in captured["messages"]]
        assert "system" in roles
        assert "user" in roles
        assert any(m["role"] != "system" for m in captured["messages"])


# =============================================================================
# Unit tests — contradiction detection and notification awareness
# =============================================================================


@pytest.mark.asyncio
class TestSpeechGateContradictionDetection:
    """Tests for the expanded speech gate that detects contradiction and
    staleness in addition to redundancy."""

    async def test_should_suppress_combines_both_fields(self):
        """SpeechDedup.should_suppress is True when either already_covered
        or contradicts_current_state is True."""
        assert SpeechDedup(already_covered=True).should_suppress is True
        assert (
            SpeechDedup(
                already_covered=False,
                contradicts_current_state=True,
            ).should_suppress
            is True
        )
        assert (
            SpeechDedup(
                already_covered=True,
                contradicts_current_state=True,
            ).should_suppress
            is True
        )
        assert (
            SpeechDedup(
                already_covered=False,
                contradicts_current_state=False,
            ).should_suppress
            is False
        )

    async def test_empty_context_skips_check(self):
        """No LLM call when both utterances and notifications are empty."""
        checker = SpeechDeduplicationChecker()

        result = await checker.evaluate(
            proposed_speech="Want me to set it up?",
            recent_utterances=[],
            recent_notifications=[],
        )

        assert result.should_suppress is False

    async def test_notifications_only_triggers_check(self):
        """When there are notifications but no utterances, the checker still
        runs (notifications alone can reveal contradiction)."""
        captured: dict = {}
        mock_client = MagicMock()
        mock_client.set_response_format = MagicMock()

        async def capture_generate(*, messages=None, **_kwargs):
            captured["messages"] = messages
            return json.dumps(
                {
                    "already_covered": False,
                    "contradicts_current_state": False,
                    "reasoning": "no overlap",
                },
            )

        mock_client.generate = AsyncMock(side_effect=capture_generate)

        with patch(
            "unity.conversation_manager.domains.speech_dedup.new_llm_client",
            return_value=mock_client,
        ):
            checker = SpeechDeduplicationChecker()
            await checker.evaluate(
                proposed_speech="Let me walk you through that.",
                recent_utterances=[],
                recent_notifications=["Setup completed successfully."],
            )

        assert captured.get("messages") is not None
        system_msg = captured["messages"][0]["content"]
        assert "Setup completed successfully" in system_msg

    async def test_evaluate_includes_notifications_in_prompt(self):
        """The expanded prompt includes recent notifications alongside
        recent utterances so the LLM can detect contradictions."""
        captured: dict = {}
        mock_client = MagicMock()
        mock_client.set_response_format = MagicMock()

        async def capture_generate(*, messages=None, **_kwargs):
            captured["messages"] = messages
            return json.dumps(
                {
                    "already_covered": False,
                    "contradicts_current_state": False,
                    "reasoning": "novel info",
                },
            )

        mock_client.generate = AsyncMock(side_effect=capture_generate)

        with patch(
            "unity.conversation_manager.domains.speech_dedup.new_llm_client",
            return_value=mock_client,
        ):
            checker = SpeechDeduplicationChecker()
            await checker.evaluate(
                proposed_speech="Want me to help set up Gmail?",
                recent_utterances=["Everything is done."],
                recent_notifications=[
                    "Gmail check completed: 201 unread emails.",
                ],
            )

        system_msg = captured["messages"][0]["content"]
        assert "Gmail check completed" in system_msg
        assert "Everything is done" in system_msg

    async def test_error_fails_open_with_both_fields(self):
        """On LLM error, both fields are False (fails open)."""
        checker = SpeechDeduplicationChecker(model="invalid-model@nowhere")

        result = await checker.evaluate(
            proposed_speech="Let me set up delegation.",
            recent_utterances=["All done."],
            recent_notifications=["Setup complete."],
        )

        assert result.should_suppress is False
        assert result.contradicts_current_state is False

    async def test_backward_compat_without_notifications(self):
        """Calling evaluate without recent_notifications still works
        (parameter is optional with default None)."""
        captured: dict = {}
        mock_client = MagicMock()
        mock_client.set_response_format = MagicMock()

        async def capture_generate(*, messages=None, **_kwargs):
            captured["messages"] = messages
            return json.dumps(
                {
                    "already_covered": True,
                    "reasoning": "same info",
                },
            )

        mock_client.generate = AsyncMock(side_effect=capture_generate)

        with patch(
            "unity.conversation_manager.domains.speech_dedup.new_llm_client",
            return_value=mock_client,
        ):
            checker = SpeechDeduplicationChecker()
            result = await checker.evaluate(
                proposed_speech="Found restaurants.",
                recent_utterances=["I found three Italian places."],
            )

        assert result.should_suppress is True
        system_msg = captured["messages"][0]["content"]
        assert "(none)" in system_msg  # notifications section shows "(none)"


@pytest.mark.eval
@pytest.mark.asyncio
class TestSpeechGateContradictionEval:
    """Eval tests verifying the LLM correctly identifies contradiction
    between proposed speech and current notification state."""

    async def test_detects_offering_setup_when_already_complete(self):
        """The gate should suppress speech that offers setup steps when
        a notification confirms the setup is already complete."""
        from unity.settings import SETTINGS

        checker = SpeechDeduplicationChecker(
            model=SETTINGS.conversation.FAST_BRAIN_MODEL,
        )

        result = await checker.evaluate(
            proposed_speech=(
                "Want me to walk you through enabling domain-wide delegation "
                "on that service account?"
            ),
            recent_utterances=[
                "Everything is set up and working. You have about 201 unread emails.",
            ],
            recent_notifications=[
                "Gmail check completed successfully. Dan has about 201 unread emails.",
                "Everything is fully set up. The Gmail API is already enabled.",
            ],
        )

        assert result.should_suppress is True, (
            f"Expected suppression for setup offer when notifications confirm "
            f"completion. Reasoning: {result.reasoning}"
        )

    async def test_allows_genuinely_new_information(self):
        """The gate should allow speech that contains genuinely new
        information not present in utterances or notifications."""
        from unity.settings import SETTINGS

        checker = SpeechDeduplicationChecker(
            model=SETTINGS.conversation.FAST_BRAIN_MODEL,
        )

        result = await checker.evaluate(
            proposed_speech="Your 3pm meeting with Sarah was moved to 4pm.",
            recent_utterances=["Let me check on that."],
            recent_notifications=[
                "Calendar event updated: meeting with Sarah rescheduled to 4pm.",
            ],
        )

        assert result.should_suppress is False, (
            f"Expected novel info to pass through. " f"Reasoning: {result.reasoning}"
        )


# =============================================================================
# Symbolic integration tests — slow brain passes should_speak through
# =============================================================================


@pytest.mark.asyncio
class TestSlowBrainPassesSpeakThrough:
    """Verify the slow brain no longer runs dedup and passes should_speak
    through to the fast brain unmodified.

    After the refactor, dedup is a fast-brain-only concern.  The slow brain
    publishes FastBrainNotification with the LLM's original should_speak value.
    """

    @pytest.fixture
    def boss_contact(self):
        return TEST_CONTACTS[1]

    async def test_slow_brain_does_not_have_dedup_checker(
        self,
        initialized_cm,
    ):
        """ConversationManager no longer carries a _speech_dedup_checker."""
        cm = initialized_cm.cm
        assert not hasattr(cm, "_speech_dedup_checker")

    async def test_should_speak_passed_through_with_recent_utterances(
        self,
        initialized_cm,
        boss_contact,
    ):
        """Even when recent assistant utterances exist in the voice thread,
        the slow brain publishes should_speak as the LLM produced it (no
        server-side suppression)."""
        cm = initialized_cm.cm

        await initialized_cm.step(PhoneCallStarted(contact=boss_contact))
        assert cm.mode == Mode.CALL

        cm.contact_index.push_message(
            contact_id=boss_contact["contact_id"],
            sender_name="You",
            thread_name=Medium.PHONE_CALL,
            message_content="That's done — found three Italian restaurants near you.",
            role="assistant",
        )

        cm.completed_actions[0] = {
            "query": "Search for nearby Italian restaurants",
            "handle_actions": [
                {
                    "action_name": "act_completed",
                    "query": "Found 3 Italian restaurants nearby.",
                    "status": "completed",
                },
            ],
        }

        published: list[dict] = []
        original_publish = cm.event_broker.publish

        async def capture_publish(channel: str, message: str) -> int:
            if channel == "app:call:notification":
                published.append(json.loads(message))
            return await original_publish(channel, message)

        cm.event_broker.publish = capture_publish

        try:
            initialized_cm.all_tool_calls.clear()

            await initialized_cm.step_until_wait(
                InboundPhoneUtterance(
                    contact=boss_contact,
                    content="Any restaurants nearby?",
                ),
                max_steps=5,
            )

            for event_data in published:
                payload = event_data.get("payload", event_data)
                if payload.get("source") == "slow_brain" and payload.get(
                    "response_text",
                ):
                    assert payload.get("should_speak") is True, (
                        "The slow brain should pass should_speak=True through "
                        "unmodified; dedup is now a fast-brain concern.\n"
                        f"Payload: {payload}"
                    )
        finally:
            cm.event_broker.publish = original_publish


# =============================================================================
# Eval test — LLM-based deduplication judgment
# =============================================================================


@pytest.mark.eval
@pytest.mark.asyncio
class TestSpeechDedupEval:
    """End-to-end eval test verifying the LLM correctly identifies when the
    fast brain has already covered the slow brain's proposed speech."""

    async def test_default_model_structured_completion_roundtrip(self):
        """Production path: no ``model=`` → ``UNIFY_MODEL`` (Anthropic).

        Catches provider-specific request-shape or schema issues that mocks miss.
        """
        checker = SpeechDeduplicationChecker()
        result = await checker.evaluate(
            proposed_speech="Found three Italian restaurants nearby.",
            recent_utterances=[
                "Yeah, I found three Italian places near you — "
                "the top rated is Chez Laurent.",
            ],
        )
        assert isinstance(result, SpeechDedup)
        assert isinstance(result.already_covered, bool)
        assert isinstance(result.reasoning, str)

    @_handle_project
    async def test_slow_brain_passes_speak_through_e2e(
        self,
        initialized_cm,
    ):
        """Verify the slow brain passes should_speak=True through to the
        fast brain without running dedup.

        Dedup now runs in the fast brain subprocess at speak time.  The slow
        brain publishes the LLM's original decision unmodified.

        Scenario:
        1. Start a Meet, complete an action with concrete results.
        2. Push an outbound assistant utterance covering the result
           (simulating the fast brain's reactive response).
        3. Step the CM with a user utterance asking about the result.
        4. Assert the slow brain's published event preserves should_speak
           as the LLM produced it (no server-side suppression).
        """
        cm = initialized_cm

        await cm.step(UnifyMeetReceived(contact=BOSS))
        await cm.step(UnifyMeetStarted(contact=BOSS))
        assert cm.cm.mode == Mode.MEET

        cm.cm.completed_actions[0] = {
            "query": "Count unread emails in Gmail inbox",
            "handle_actions": [
                {
                    "action_name": "act_completed",
                    "query": "You have 47 unread emails in your inbox.",
                    "status": "completed",
                },
            ],
        }

        cm.cm.contact_index.push_message(
            contact_id=BOSS["contact_id"],
            sender_name="You",
            thread_name=Medium.UNIFY_MEET,
            message_content=(
                "Yeah, the email check came back — you've got 47 unread "
                "emails in your inbox."
            ),
            role="assistant",
        )

        published: list[dict] = []
        original_publish = cm.cm.event_broker.publish

        async def capture_publish(channel: str, message: str) -> int:
            if channel == "app:call:notification":
                published.append(json.loads(message))
            return await original_publish(channel, message)

        cm.cm.event_broker.publish = capture_publish

        try:
            cm.all_tool_calls.clear()

            await cm.step_until_wait(
                InboundUnifyMeetUtterance(
                    contact=BOSS,
                    content="How did the email check go?",
                ),
                max_steps=5,
            )

            for event_data in published:
                payload = event_data.get("payload", event_data)
                if payload.get("source") == "slow_brain" and payload.get(
                    "response_text",
                ):
                    assert payload.get("should_speak") is True, (
                        "The slow brain should pass should_speak=True through "
                        "unmodified; dedup is now a fast-brain concern.\n"
                        f"Payload: {payload}\n"
                        f"Tool calls: {cm.all_tool_calls}"
                    )
        finally:
            cm.cm.event_broker.publish = original_publish
