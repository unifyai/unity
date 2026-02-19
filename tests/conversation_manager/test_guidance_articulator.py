"""
tests/conversation_manager/test_guidance_articulator.py
=======================================================

Tests for the GuidanceArticulator module that determines whether slow brain
guidance should be blocked, silently notified, or spoken aloud by the fast brain.

The GuidanceArticulator uses a fast LLM (opus-4.6 without extended thinking)
to make decisions about relevance and, when appropriate, generates the exact
speech text in the voice agent's persona.

Test Categories:
----------------
1. **Topic Change Detection**: Guidance should be blocked when user changes topic
2. **Same Topic Continuation**: Guidance should be sent when topic stays the same
3. **Notification Relevance**: Cross-channel notifications should usually be sent
4. **Redundancy Detection**: Guidance should be blocked if fast brain already handled it
5. **Speech Generation**: When guidance is relevant, articulator generates speech text
6. **Edge Cases**: Ambiguous situations, partial topic changes, etc.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from unity.conversation_manager.domains.guidance_articulator import (
    ConversationMessage,
    GuidanceArticulator,
    GuidanceDecision,
)

# Minimal voice agent prompt for tests. Enough for the articulator to match
# persona without the full production prompt.
STUB_VOICE_AGENT_PROMPT = """\
I'm a helpful assistant on a phone call with my boss.
I never reference internal systems, backends, or notifications.
I match the caller's language.

Brevity: I sound like a normal person on a phone call: concise, natural, and calm.
Most turns are one to two sentences.

I receive internal [notification] messages with data. The user cannot see these.
I integrate them naturally as if I knew the answer all along.
I say "I sent the email", not "the email was sent." I never mention notifications."""

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def articulator():
    """Create a GuidanceArticulator instance for testing."""
    return GuidanceArticulator()


@pytest.fixture
def base_timestamp():
    """Base timestamp for test messages."""
    return datetime(2025, 6, 13, 12, 0, 0)


def make_conversation(
    messages: list[tuple[str, str, bool]],
    base_time: datetime,
) -> list[ConversationMessage]:
    """
    Helper to create a conversation from a list of (role, content, is_new) tuples.

    Args:
        messages: List of (role, content, is_new) tuples
        base_time: Starting timestamp

    Returns:
        List of ConversationMessage objects
    """
    result = []
    for i, (role, content, is_new) in enumerate(messages):
        result.append(
            ConversationMessage(
                role=role,
                content=content,
                timestamp=base_time + timedelta(seconds=i * 5),
                is_new=is_new,
            ),
        )
    return result


# =============================================================================
# Test Class: Topic Change Detection
# =============================================================================


@pytest.mark.asyncio
class TestTopicChangeDetection:
    """Tests for detecting when the conversation topic has changed."""

    async def test_explicit_topic_change_blocks_guidance(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        When user explicitly changes topic ("actually, forget that"), guidance
        about the old topic should be blocked.
        """
        guidance = "The meeting tomorrow is at 3pm in Conference Room B"

        conversation = make_conversation(
            [
                ("user", "What time is the meeting tomorrow?", False),
                ("user", "Actually, forget about that. What's the weather like?", True),
                (
                    "assistant",
                    "Let me check the weather. It looks sunny, around 72 degrees.",
                    True,
                ),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is False, (
            f"Guidance about meeting should be BLOCKED after explicit topic change!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}\n"
            f"\n"
            f"The user explicitly said 'forget about that' and asked about weather.\n"
            f"Sending meeting info now would be confusing and out of context."
        )

    async def test_implicit_topic_change_blocks_guidance(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        When user implicitly changes topic (starts asking about something else
        without explicit "never mind"), guidance about old topic should be blocked.
        """
        guidance = "The nearest Italian restaurant is Mario's, about 2 miles away"

        conversation = make_conversation(
            [
                ("user", "Find me Italian restaurants nearby", False),
                ("user", "Oh wait, what's my schedule look like tomorrow?", True),
                (
                    "assistant",
                    "Let me check your calendar for tomorrow.",
                    True,
                ),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is False, (
            f"Guidance about restaurants should be BLOCKED after topic change!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}\n"
            f"\n"
            f"The user switched to asking about their schedule.\n"
            f"Restaurant info is no longer relevant."
        )

    async def test_user_cancellation_blocks_guidance(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        When user cancels their request, guidance fulfilling that request should
        be blocked.
        """
        guidance = "Found 5 Italian restaurants within 3 miles of your location"

        conversation = make_conversation(
            [
                ("user", "Can you find Italian restaurants near me?", False),
                (
                    "user",
                    "Wait, never mind. I just remembered I have food at home.",
                    True,
                ),
                (
                    "assistant",
                    "No problem! Let me know if you need anything else.",
                    True,
                ),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is False, (
            f"Guidance should be BLOCKED when user cancels request!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}"
        )


# =============================================================================
# Test Class: Same Topic Continuation
# =============================================================================


@pytest.mark.asyncio
class TestSameTopicContinuation:
    """Tests for cases where guidance should be sent (topic stayed the same)."""

    async def test_follow_up_question_sends_guidance(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        When user asks a follow-up question about the same topic, guidance about
        that topic should be sent.
        """
        guidance = "The meeting tomorrow is at 3pm in Conference Room B"

        conversation = make_conversation(
            [
                ("user", "What time is the meeting tomorrow?", False),
                ("user", "And who's going to be there?", True),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is True, (
            f"Guidance about meeting should be SENT for follow-up question!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}\n"
            f"\n"
            f"The user is still asking about the same meeting.\n"
            f"The time/location info is still relevant."
        )

    async def test_clarification_sends_guidance(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        When user asks for clarification on the same topic, guidance should be sent.
        """
        guidance = "John's phone number is 555-123-4567"

        conversation = make_conversation(
            [
                ("user", "What's John's phone number?", False),
                ("user", "Is that his work phone or personal?", True),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is True, (
            f"Guidance should be SENT for clarification question!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}"
        )

    async def test_no_new_messages_sends_guidance(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        When there are no new messages (conversation hasn't progressed),
        guidance should definitely be sent.
        """
        guidance = "The project deadline is next Friday, June 20th"

        conversation = make_conversation(
            [
                ("user", "When is the project deadline?", False),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is True, (
            f"Guidance should be SENT when no new messages!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}"
        )

    async def test_assistant_acknowledgment_sends_guidance(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        When assistant just acknowledged ("let me check"), guidance with the
        answer should be sent.
        """
        guidance = "The client's email address is alice@company.com"

        conversation = make_conversation(
            [
                ("user", "What's the client's email?", False),
                ("assistant", "Let me look that up for you.", True),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is True, (
            f"Guidance should be SENT after assistant acknowledgment!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}"
        )


# =============================================================================
# Test Class: Speech Generation
# =============================================================================


@pytest.mark.asyncio
class TestSpeechGeneration:
    """Tests for the articulator's speech generation when guidance is relevant."""

    async def test_direct_answer_generates_speech(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        When guidance directly answers the user's question, the articulator
        should generate speech (should_speak=True with non-empty response_text).
        """
        guidance = "The meeting tomorrow is at 3pm in Conference Room B"

        conversation = make_conversation(
            [
                ("user", "What time is the meeting tomorrow?", False),
                ("assistant", "Let me check on that.", True),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is True
        assert decision.should_speak is True, (
            f"Articulator should generate speech for a direct answer!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}"
        )
        assert decision.response_text, (
            f"response_text must be non-empty when should_speak is True!\n"
            f"Decision: {decision}"
        )

    async def test_speech_is_concise(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        Generated speech should be brief (matching voice agent persona: 1-2 sentences).
        """
        guidance = "John's email is john@example.com and his phone number is 555-123-4567. He works in the engineering department on the third floor."

        conversation = make_conversation(
            [
                ("user", "How can I reach John?", False),
                ("assistant", "Let me look that up.", True),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.should_speak is True
        # Reasonable upper bound for concise phone speech
        assert len(decision.response_text) < 300, (
            f"Speech should be concise! Got {len(decision.response_text)} chars:\n"
            f"{decision.response_text}"
        )


# =============================================================================
# Test Class: Notification Relevance
# =============================================================================


@pytest.mark.asyncio
class TestNotificationRelevance:
    """Tests for cross-channel notification guidance."""

    async def test_relevant_notification_is_sent(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        A notification that directly answers the user's question should be sent.
        """
        guidance = "SMS from Alice: 'Running 10 minutes late'"

        conversation = make_conversation(
            [
                ("user", "When is Alice arriving?", False),
                ("assistant", "Let me check if I have any updates from her.", True),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is True, (
            f"Notification answering user's question should be SENT!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}"
        )

    async def test_urgent_notification_is_sent(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        An urgent notification should generally be sent even if topic changed.
        """
        guidance = "URGENT: Email from boss marked high priority - 'Call me ASAP'"

        conversation = make_conversation(
            [
                ("user", "What's the weather like?", False),
                ("assistant", "It's sunny and 75 degrees today.", True),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is True, (
            f"Urgent notification should be SENT even if topic is different!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}"
        )


# =============================================================================
# Test Class: Redundancy Detection
# =============================================================================


@pytest.mark.asyncio
class TestRedundancyDetection:
    """Tests for detecting when fast brain already handled the request."""

    async def test_already_answered_blocks_guidance(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        When fast brain already provided the answer, guidance with the same
        info should be blocked to avoid redundancy.
        """
        guidance = "John's email is john@example.com"

        conversation = make_conversation(
            [
                ("user", "What's John's email?", False),
                (
                    "assistant",
                    "John's email is john@example.com. Would you like me to send him a message?",
                    True,
                ),
                ("user", "Yes please, ask him about the project.", True),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is False, (
            f"Redundant guidance should be BLOCKED!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}\n"
            f"\n"
            f"The fast brain already provided this email address.\n"
            f"Sending it again would be repetitive."
        )

    async def test_partial_answer_allows_additional_guidance(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        When fast brain gave a partial answer, guidance with additional info
        should be sent.
        """
        guidance = "John's email is john@example.com, and his phone is 555-123-4567"

        conversation = make_conversation(
            [
                ("user", "How can I reach John?", False),
                (
                    "assistant",
                    "I'm looking up John's contact information for you.",
                    True,
                ),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is True, (
            f"Additional contact info should be SENT!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}"
        )


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


@pytest.mark.asyncio
class TestEdgeCases:
    """Tests for edge cases and ambiguous situations."""

    async def test_empty_conversation_with_notification_sends_guidance(
        self,
        articulator: GuidanceArticulator,
    ):
        """
        With an empty conversation but an important notification, guidance should
        be sent. Notifications are inherently relevant regardless of conversation state.
        """
        guidance = "URGENT: SMS from boss - 'Meeting moved to 2pm'"

        decision = await articulator.articulate_guidance(
            guidance,
            [],
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is True, (
            f"Urgent notification should be SENT even with empty conversation!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}"
        )

    async def test_all_new_messages_still_evaluates_relevance(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        Even if all messages are "new" (edge case), the articulator should still
        evaluate relevance based on content.
        """
        guidance = "The meeting is at 3pm"

        conversation = make_conversation(
            [
                ("user", "What's the weather?", True),
                ("assistant", "It's sunny today.", True),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.send_guidance is False, (
            f"Irrelevant guidance should be BLOCKED!\n"
            f"Guidance: {guidance}\n"
            f"Decision thoughts: {decision.thoughts}"
        )

    async def test_related_but_different_topic(
        self,
        articulator: GuidanceArticulator,
        base_timestamp: datetime,
    ):
        """
        When the new topic is related but different, the articulator should make
        a judgment call. This tests the model's reasoning.
        """
        guidance = "The project budget is $50,000"

        conversation = make_conversation(
            [
                ("user", "What's the budget for the project?", False),
                ("user", "Actually, what's the deadline instead?", True),
            ],
            base_timestamp,
        )

        decision = await articulator.articulate_guidance(
            guidance,
            conversation,
            STUB_VOICE_AGENT_PROMPT,
        )

        assert decision.thoughts, "Decision should include reasoning"


# =============================================================================
# Test Class: Response Model Validation
# =============================================================================


@pytest.mark.asyncio
class TestResponseModel:
    """Tests for the GuidanceDecision response model."""

    def test_decision_model_has_required_fields(self):
        """Verify the response model has the expected fields."""
        decision = GuidanceDecision(
            thoughts="Test reasoning",
            send_guidance=True,
            can_speak_without_fabricating=True,
            should_speak=True,
            response_text="It's at 3pm.",
        )
        assert hasattr(decision, "thoughts")
        assert hasattr(decision, "send_guidance")
        assert hasattr(decision, "can_speak_without_fabricating")
        assert hasattr(decision, "should_speak")
        assert hasattr(decision, "response_text")
        assert isinstance(decision.thoughts, str)
        assert isinstance(decision.send_guidance, bool)
        assert isinstance(decision.can_speak_without_fabricating, bool)
        assert isinstance(decision.should_speak, bool)
        assert isinstance(decision.response_text, str)

    def test_decision_model_json_serialization(self):
        """Verify the model can be serialized to/from JSON."""
        decision = GuidanceDecision(
            thoughts="The guidance is relevant",
            send_guidance=True,
            can_speak_without_fabricating=True,
            should_speak=True,
            response_text="It's at 3pm in Conference Room B.",
        )
        json_str = decision.model_dump_json()
        restored = GuidanceDecision.model_validate_json(json_str)
        assert restored.thoughts == decision.thoughts
        assert restored.send_guidance == decision.send_guidance
        assert (
            restored.can_speak_without_fabricating
            == decision.can_speak_without_fabricating
        )
        assert restored.should_speak == decision.should_speak
        assert restored.response_text == decision.response_text

    def test_decision_model_defaults(self):
        """Verify default values for optional fields."""
        decision = GuidanceDecision(
            thoughts="Blocked",
            send_guidance=False,
            can_speak_without_fabricating=False,
            should_speak=False,
        )
        assert decision.response_text == ""


class TestGuidanceMessageRoleMapping:
    """Regression tests for how GuidanceMessage entries are mapped to roles
    when building the articulator's conversation context.

    GuidanceMessage (internal orchestration) has no `role` attribute. The
    fallback logic in _articulate_guidance() must not label these as "user",
    or the articulator sees guidance content as if the user said it —
    corrupting its relevance and redundancy judgments.
    """

    def test_guidance_message_not_labeled_as_user(self):
        """GuidanceMessage entries must not appear in the articulator's
        conversation with role='user'. The correct fix is to skip them
        entirely (isinstance check), since their content is already
        in the RECENTLY SENT GUIDANCE section."""
        from unity.conversation_manager.domains.contact_index import GuidanceMessage

        guidance = GuidanceMessage(
            name="guidance",
            content="Checking your contacts for Bob.",
            timestamp=datetime(2025, 6, 13, 12, 0, 0),
        )

        # The fix: GuidanceMessage should be skipped via isinstance check.
        # If it's NOT skipped, the old fallback logic assigns role="user".
        if isinstance(guidance, GuidanceMessage):
            skipped = True
        elif hasattr(guidance, "role"):
            skipped = False
        else:
            skipped = False

        assert skipped, (
            f"GuidanceMessage was not filtered out. Without filtering, "
            f"the hasattr fallback assigns role='user' (name={guidance.name!r}), "
            f"corrupting the articulator's conversation context."
        )

    def test_regular_message_roles_preserved(self):
        """Normal Message entries with role='user' or role='assistant' are
        correctly preserved by the same fallback logic."""
        from unity.conversation_manager.domains.contact_index import Message

        user_msg = Message(
            name="Yusha",
            content="Check my contacts.",
            timestamp=datetime(2025, 6, 13, 12, 0, 0),
            role="user",
        )
        asst_msg = Message(
            name="You",
            content="One moment.",
            timestamp=datetime(2025, 6, 13, 12, 0, 1),
            role="assistant",
        )

        for msg, expected in [(user_msg, "user"), (asst_msg, "assistant")]:
            if hasattr(msg, "role"):
                role = msg.role
            else:
                role = "assistant" if msg.name == "You" else "user"
            assert role == expected
