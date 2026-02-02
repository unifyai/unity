"""
tests/conversation_manager/voice/test_fast_slow_brain_coordination.py
======================================================================

Tests for coordination between the fast brain (Voice Agent) and slow brain
(Main CM Brain) during voice calls.

Architecture Overview:
----------------------
- **Slow Brain** (Main CM Brain / ConversationManager): Handles orchestration,
  task execution, and cross-channel communication. Runs in the main process.

- **Fast Brain** (Voice Agent): Lightweight LLM in a subprocess that handles
  real-time voice conversation. Runs in medium_scripts/call.py or sts_call.py.

What This File Tests:
---------------------
1. **Decision boundaries**: When should the slow brain provide guidance vs stay silent?
2. **Guidance content**: Is guidance appropriate (data/notifications) vs inappropriate
   (conversational steering)?
3. **Coordination timing**: Does the slow brain avoid duplicating what the fast brain
   already handles autonomously?
4. **Schema correctness**: Do the response models allow the intended behavior?

Key Principle (per system prompt in prompt_builders.py):
--------------------------------------------------------
The slow brain's `call_guidance` field should ONLY be used for:
- Providing data: "The meeting time the boss mentioned earlier was 3pm on Thursday"
- Requesting data: "Please ask for their preferred contact method"
- Notifications: "The boss just confirmed via SMS that the budget is approved"

It should NOT be used for:
- Steering the conversation
- Suggesting responses or dialogue
- Providing conversational guidance
- Micromanaging the Voice Agent's approach

The Voice Agent independently handles ALL conversational aspects including greetings.

Known Issue (duplicate speech bug):
-----------------------------------
When a call starts, the slow brain may provide conversational guidance like
"Greet Ved warmly" even though the fast brain has already greeted the user
autonomously. This causes duplicate speech. Tests in this file document and
verify the fix for this issue.
"""

from __future__ import annotations

import json
import re

import pytest

from unity.conversation_manager.events import (
    PhoneCallStarted,
    CallGuidance,
)
from unity.conversation_manager.types import Medium

from tests.conversation_manager.conftest import TEST_CONTACTS

# =============================================================================
# Test: call_guidance should not contain conversational guidance on call start
# =============================================================================


@pytest.mark.asyncio
class TestSlowBrainDecisionBoundaries:
    """
    Tests for when the slow brain should vs should NOT provide guidance.

    The fast brain (Voice Agent) autonomously handles all conversational aspects,
    including greetings. The slow brain should only provide data, requests, or
    notifications - not conversational guidance.
    """

    @pytest.fixture
    def boss_contact(self):
        """The boss contact (contact_id=1) who is calling."""
        return TEST_CONTACTS[1]

    async def test_call_start_should_not_trigger_greeting_guidance(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        FAILING TEST: When a call starts, the slow brain should NOT provide
        conversational guidance like 'Greet the user' or 'Say hello'.

        The fast brain handles greetings autonomously. If the slow brain also
        sends greeting guidance, the fast brain will speak twice (duplicate speech).

        This test documents the bug reported by Ved:
        - Call starts
        - Fast brain greets user immediately (autonomous behavior)
        - Slow brain runs, sees "Call started" notification
        - Slow brain outputs call_guidance: "Greet Ved warmly..."
        - Fast brain receives this guidance and greets AGAIN

        Expected: call_guidance should be empty or contain only data/notifications,
        not conversational directives.
        """
        # Track what call_guidance is published
        published_guidance: list[str] = []

        original_publish = initialized_cm.cm.event_broker.publish

        async def capture_guidance(channel: str, message: str) -> int:
            if channel == "app:call:call_guidance":
                try:
                    data = json.loads(message)
                    # Handle both Event format and plain dict format
                    if "payload" in data:
                        content = data["payload"].get("content", "")
                    else:
                        content = data.get("content", "")
                    if content:
                        published_guidance.append(content)
                except (json.JSONDecodeError, KeyError):
                    pass
            return await original_publish(channel, message)

        initialized_cm.cm.event_broker.publish = capture_guidance

        try:
            # Simulate a call starting with the boss
            # This triggers the slow brain to run via request_llm_run(delay=0)
            event = PhoneCallStarted(contact=boss_contact)
            result = await initialized_cm.step(event)

            # The slow brain ran
            assert result.llm_ran, "Expected slow brain to run on PhoneCallStarted"

            # Check that NO conversational guidance was published
            # Conversational guidance includes things like:
            # - "Greet..." / "Say hello..." / "Welcome..."
            # - "Ask how they are..."
            # - "Start by..." / "Begin with..."
            # - Any directive about what to SAY

            conversational_patterns = [
                r"\bgreet\b",
                r"\bsay\s+(hello|hi|hey)\b",
                r"\bwelcome\b",
                r"\bask\s+how\s+(they|he|she)\s+(are|is)\b",
                r"\bstart\s+(by|with)\b",
                r"\bbegin\s+(by|with)\b",
                r"\bsay\s+something\s+like\b",
                r"\btell\s+(them|him|her)\b",
                r"\bintroduce\b",
                r"\bopen\s+(with|by)\b",
            ]

            for guidance in published_guidance:
                guidance_lower = guidance.lower()
                for pattern in conversational_patterns:
                    match = re.search(pattern, guidance_lower)
                    assert match is None, (
                        f"Slow brain provided conversational guidance on call start!\n"
                        f"  Guidance: '{guidance}'\n"
                        f"  Matched pattern: '{pattern}'\n"
                        f"\n"
                        f"This causes duplicate speech because the fast brain has already\n"
                        f"greeted the user autonomously. The slow brain should NOT provide\n"
                        f"conversational guidance - only data, requests, or notifications.\n"
                        f"\n"
                        f"See: _build_voice_calls_guide() in prompt_builders.py for the\n"
                        f"intended behavior documented in the system prompt."
                    )

        finally:
            initialized_cm.cm.event_broker.publish = original_publish

    async def test_call_guidance_field_allows_empty_value(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        FAILING TEST: The call_guidance field should allow an empty value.

        Currently, VoiceResponse.call_guidance is defined with Field(...) which
        makes it REQUIRED. The LLM is forced to fill this field even when there's
        nothing to communicate, leading to unnecessary conversational guidance.

        The system prompt says:
        "Leave `call_guidance` empty unless you need to exchange specific
        information with the Voice Agent."

        But the Pydantic model doesn't allow this - the field is required.

        Expected: call_guidance should be optional (Field(default="")) so the
        LLM can leave it empty when there's nothing to communicate.
        """
        from pydantic import ValidationError
        from unity.conversation_manager.domains.brain import build_response_models
        from unity.conversation_manager.types import Mode

        models = build_response_models()
        VoiceResponse = models[Mode.CALL]

        # Try to create a VoiceResponse with empty call_guidance
        # This SHOULD succeed but currently FAILS because call_guidance is required
        try:
            response = VoiceResponse(thoughts="No action needed", call_guidance="")
            # If we get here, the field accepts empty string (good!)
            assert response.call_guidance == ""
        except ValidationError as e:
            pytest.fail(
                f"VoiceResponse.call_guidance should accept empty string!\n"
                f"Validation error: {e}\n"
                f"\n"
                f"The system prompt says to leave call_guidance empty when there's\n"
                f"nothing to communicate, but the Pydantic model requires a value.\n"
                f"\n"
                f"Fix: Change Field(...) to Field(default='') in brain.py",
            )

        # Also verify the field description doesn't encourage conversational guidance
        schema = VoiceResponse.model_json_schema()
        call_guidance_schema = schema.get("properties", {}).get("call_guidance", {})
        description = call_guidance_schema.get("description", "")

        # The description should make it clear this is for data/notifications ONLY
        assert "guidance" not in description.lower() or "data" in description.lower(), (
            f"call_guidance field description is misleading!\n"
            f"  Current: '{description}'\n"
            f"\n"
            f"The description says 'guidance' which makes the LLM think it should\n"
            f"provide conversational guidance. It should emphasize that this is\n"
            f"only for data provision, data requests, and notifications."
        )


@pytest.mark.asyncio
class TestSlowBrainAppropriateGuidance:
    """
    Tests that call_guidance IS used correctly for its intended purposes:
    data provision, data requests, and notifications.

    These are "positive" tests showing what the slow brain SHOULD do,
    complementing the "negative" tests in TestSlowBrainDecisionBoundaries.
    """

    @pytest.fixture
    def boss_contact(self):
        return TEST_CONTACTS[1]

    async def test_call_guidance_appropriate_for_data_provision(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        Verify that call_guidance IS appropriate for providing data that the
        fast brain doesn't have access to.

        Examples of appropriate call_guidance:
        - "The meeting time mentioned earlier was 3pm on Thursday"
        - "The client's email is john@example.com"
        - "The budget has been approved - $50,000"
        """
        # Start a call
        started_event = PhoneCallStarted(contact=boss_contact)
        await initialized_cm.step(started_event)

        # Manually publish a data-provision guidance (this is appropriate)
        guidance = CallGuidance(
            contact=boss_contact,
            content="The meeting time mentioned in the earlier SMS was 3pm on Thursday",
        )
        result = await initialized_cm.step(guidance)

        # This should be recorded in the voice thread
        contact_id = boss_contact["contact_id"]
        conv = initialized_cm.cm.contact_index.active_conversations.get(contact_id)
        voice_thread = conv.threads.get(Medium.PHONE_CALL, [])

        # Find guidance messages
        guidance_msgs = [msg for msg in voice_thread if msg.name == "guidance"]
        assert len(guidance_msgs) >= 1
        assert any("3pm on Thursday" in msg.content for msg in guidance_msgs)

    async def test_call_guidance_appropriate_for_notifications(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        Verify that call_guidance IS appropriate for cross-channel notifications.

        Examples of appropriate call_guidance:
        - "The boss just confirmed via SMS that the budget is approved"
        - "Email received from the client with updated requirements"
        """
        # Start a call
        started_event = PhoneCallStarted(contact=boss_contact)
        await initialized_cm.step(started_event)

        # Manually publish a notification guidance (this is appropriate)
        guidance = CallGuidance(
            contact=boss_contact,
            content="SMS just received from Alice: 'Running 10 minutes late'",
        )
        result = await initialized_cm.step(guidance)

        # Verify it was recorded
        contact_id = boss_contact["contact_id"]
        conv = initialized_cm.cm.contact_index.active_conversations.get(contact_id)
        voice_thread = conv.threads.get(Medium.PHONE_CALL, [])

        guidance_msgs = [msg for msg in voice_thread if msg.name == "guidance"]
        assert any("Running 10 minutes late" in msg.content for msg in guidance_msgs)
