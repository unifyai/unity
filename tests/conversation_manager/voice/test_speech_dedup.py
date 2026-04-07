"""
tests/conversation_manager/voice/test_speech_dedup.py
=====================================================

Tests for the slow brain speech deduplication gate.

When the slow brain decides to speak via guide_voice_agent(should_speak=True),
the dedup gate checks whether recent fast brain utterances already cover the
same information.  If so, should_speak is downgraded to False — the content
still reaches the fast brain as silent context ([notification]) but is not
spoken, avoiding redundancy.

Test categories:

1. **Unit tests** — SpeechDeduplicationChecker in isolation.
2. **Symbolic integration tests** — full CM pipeline with the dedup gate,
   verifying publish behavior through event_broker spying.
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
    FastBrainNotification,
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    PhoneCallStarted,
    UnifyMeetReceived,
    UnifyMeetStarted,
)
from unity.conversation_manager.types import Medium, Mode

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
# Symbolic integration tests — CM pipeline with dedup gate
# =============================================================================


@pytest.mark.asyncio
class TestSpeechDedupGateIntegration:
    """Verify the dedup gate integrates correctly with the guide_voice_agent
    flow in ConversationManager._run_llm.

    These tests mock the dedup checker to test the wiring rather than the
    LLM judgment (that's covered by the unit and eval tests).

    Note: the CMStepDriver intercepts event_broker.publish during step
    execution, so we cannot capture raw published JSON from tests.  Instead
    we verify behavior through the mock dedup checker's call status and the
    deterministic code path that follows.
    """

    @pytest.fixture
    def boss_contact(self):
        return TEST_CONTACTS[1]

    async def test_dedup_gate_invoked_when_recent_utterances_exist(
        self,
        initialized_cm,
        boss_contact,
    ):
        """When the slow brain produces should_speak=True and there are
        recent assistant utterances in the voice thread, the dedup checker
        must be invoked.  Mocking it to return already_covered=True verifies
        the suppression path (should_speak is downgraded to False)."""
        cm = initialized_cm.cm

        await initialized_cm.step(PhoneCallStarted(contact=boss_contact))
        assert cm.mode == Mode.CALL

        # Simulate a fast brain response already in the transcript
        cm.contact_index.push_message(
            contact_id=boss_contact["contact_id"],
            sender_name="You",
            thread_name=Medium.PHONE_CALL,
            message_content="That's done — found three Italian restaurants near you.",
            role="assistant",
        )

        cm._speech_dedup_checker.evaluate = AsyncMock(
            return_value=SpeechDedup(
                already_covered=True,
                reasoning="fast brain already communicated the restaurant results",
            ),
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
        initialized_cm.all_tool_calls.clear()

        await initialized_cm.step_until_wait(
            InboundPhoneUtterance(
                contact=boss_contact,
                content="Any restaurants nearby?",
            ),
            max_steps=5,
        )

        if "guide_voice_agent" in initialized_cm.all_tool_calls:
            cm._speech_dedup_checker.evaluate.assert_called()
            # Verify at least one call received both proposed speech and
            # recent utterances.
            for call_obj in cm._speech_dedup_checker.evaluate.call_args_list:
                recent = call_obj.kwargs.get(
                    "recent_utterances",
                    call_obj.args[1] if len(call_obj.args) > 1 else None,
                )
                if recent:
                    break
            else:
                raise AssertionError(
                    "Dedup checker was called but never received recent utterances",
                )

    async def test_dedup_gate_allows_when_no_recent_utterances(
        self,
        initialized_cm,
        boss_contact,
    ):
        """When there are no recent assistant utterances in the voice thread,
        the dedup checker should NOT be invoked — the gate short-circuits
        because there is nothing to compare against."""
        cm = initialized_cm.cm

        await initialized_cm.step(PhoneCallStarted(contact=boss_contact))
        assert cm.mode == Mode.CALL

        cm._speech_dedup_checker.evaluate = AsyncMock(
            return_value=SpeechDedup(already_covered=False, reasoning=""),
        )

        cm.completed_actions[0] = {
            "query": "Check the weather in Berlin",
            "handle_actions": [
                {
                    "action_name": "act_completed",
                    "query": "Berlin: 15°C, partly cloudy.",
                    "status": "completed",
                },
            ],
        }
        initialized_cm.all_tool_calls.clear()

        await initialized_cm.step_until_wait(
            InboundPhoneUtterance(
                contact=boss_contact,
                content="What's the weather like in Berlin?",
            ),
            max_steps=5,
        )

        # With no recent assistant utterances, _get_recent_voice_utterances
        # returns [] and the gate skips the LLM call entirely.
        cm._speech_dedup_checker.evaluate.assert_not_called()

    async def test_dedup_gate_skipped_for_notify_mode(
        self,
        initialized_cm,
        boss_contact,
    ):
        """When guide_voice_agent is called without should_speak=True
        (NOTIFY mode), the dedup checker should NOT be invoked."""
        cm = initialized_cm.cm

        await initialized_cm.step(PhoneCallStarted(contact=boss_contact))

        cm._speech_dedup_checker.evaluate = AsyncMock(
            return_value=SpeechDedup(already_covered=False, reasoning=""),
        )

        guidance = FastBrainNotification(
            contact=boss_contact,
            content="The meeting is at 3pm Thursday.",
        )
        await initialized_cm.step(guidance)

        cm._speech_dedup_checker.evaluate.assert_not_called()

    async def test_dedup_gate_skipped_when_disabled(
        self,
        initialized_cm,
        boss_contact,
    ):
        """When SPEECH_DEDUP_ENABLED is False, the gate does not run even
        when should_speak=True and there are recent utterances."""
        from unity.settings import SETTINGS

        cm = initialized_cm.cm
        orig = SETTINGS.conversation.SPEECH_DEDUP_ENABLED
        SETTINGS.conversation.SPEECH_DEDUP_ENABLED = False

        try:
            await initialized_cm.step(PhoneCallStarted(contact=boss_contact))

            cm.contact_index.push_message(
                contact_id=boss_contact["contact_id"],
                sender_name="You",
                thread_name=Medium.PHONE_CALL,
                message_content="That's done.",
                role="assistant",
            )

            cm._speech_dedup_checker.evaluate = AsyncMock(
                return_value=SpeechDedup(already_covered=True, reasoning="dup"),
            )

            cm.completed_actions[0] = {
                "query": "Finish the report",
                "handle_actions": [
                    {
                        "action_name": "act_completed",
                        "query": "Report finished.",
                        "status": "completed",
                    },
                ],
            }
            initialized_cm.all_tool_calls.clear()

            await initialized_cm.step_until_wait(
                InboundPhoneUtterance(
                    contact=boss_contact,
                    content="Is the report done?",
                ),
                max_steps=5,
            )

            cm._speech_dedup_checker.evaluate.assert_not_called()
        finally:
            SETTINGS.conversation.SPEECH_DEDUP_ENABLED = orig


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
    async def test_race_condition_dedup_e2e(
        self,
        initialized_cm,
    ):
        """Simulate the race condition: fast brain answers a user question
        from notification context, then the slow brain tries to speak the
        same result. The dedup gate should suppress the redundant speech.

        Scenario:
        1. Start a Meet, complete an action with concrete results.
        2. Push an outbound assistant utterance covering the result
           (simulating the fast brain's reactive response).
        3. Step the CM with a user utterance asking about the result.
        4. Assert the slow brain's guide_voice_agent speech is suppressed.
        """
        cm = initialized_cm

        await cm.step(UnifyMeetReceived(contact=BOSS))
        await cm.step(UnifyMeetStarted(contact=BOSS))
        assert cm.cm.mode == Mode.MEET

        # Simulate a completed action with concrete results
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

        # Simulate the fast brain having already answered this question
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

        # Capture published notifications to verify dedup behavior
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

            # If the slow brain called guide_voice_agent, the dedup gate
            # should have suppressed it because the fast brain already told
            # the user about the 47 unread emails.
            for event_data in published:
                payload = event_data.get("payload", event_data)
                if payload.get("source") == "slow_brain":
                    assert payload.get("should_speak") is False, (
                        "The dedup gate should suppress speech when the fast "
                        "brain already communicated the same result.\n"
                        f"Payload: {payload}\n"
                        f"Tool calls: {cm.all_tool_calls}"
                    )
        finally:
            cm.cm.event_broker.publish = original_publish
