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
5. **Rapid utterance handling**: Does the slow brain complete thinking even with
   rapid user turns?

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

Known Issue (rapid utterance cancellation):
-------------------------------------------
When user turns occur faster than the slow brain can think, every new utterance
cancels the in-flight LLM run (because interject_or_run uses cancel_running=True).
This means the slow brain NEVER completes thinking if the user keeps talking.
Tests in this file document and verify the fix for this issue.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from unity.conversation_manager.events import (
    PhoneCallStarted,
    InboundPhoneUtterance,
    CallGuidance,
)
from unity.conversation_manager.types import Medium, Mode

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
        When a call starts, the slow brain should NOT run at all and should NOT
        provide conversational guidance like 'Greet the user' or 'Say hello'.

        The fast brain handles greetings and all conversational aspects autonomously.
        The slow brain should only be triggered later when there's actual content
        that requires its attention (user utterance, action completion, etc.).

        This test documents the bug reported by Ved and verifies the fix:

        BUG (before fix):
        - Call starts
        - Slow brain runs immediately via request_llm_run(delay=0)
        - Slow brain sees "Call started" notification
        - Slow brain outputs call_guidance: "Greet Ved warmly..."
        - Fast brain (which already greeted) receives this guidance and greets AGAIN
        - Result: duplicate speech

        FIX:
        - Call starts
        - Slow brain does NOT run on call start
        - Fast brain handles greeting autonomously
        - Slow brain only runs later when there's actual content to process
        - Result: no duplicate speech
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
            event = PhoneCallStarted(contact=boss_contact)
            result = await initialized_cm.step(event)

            # The slow brain should NOT run on call start.
            # The fast brain handles the greeting autonomously.
            # Triggering the slow brain here would cause unnecessary call_guidance.
            assert not result.llm_ran, (
                "Slow brain should NOT run on PhoneCallStarted!\n"
                "\n"
                "The fast brain handles greetings and all conversational aspects\n"
                "autonomously. If the slow brain runs on call start, it may provide\n"
                "call_guidance like 'Greet the user', causing duplicate speech.\n"
                "\n"
                "The slow brain should only be triggered by:\n"
                "- InboundPhoneUtterance (user says something)\n"
                "- ActorResult (action completes)\n"
                "- NotificationInjectedEvent (cross-channel notification)\n"
                "- SMSReceived/EmailReceived while on call"
            )

            # Also verify no call_guidance was published
            assert len(published_guidance) == 0, (
                f"call_guidance was published on call start: {published_guidance}\n"
                "No guidance should be sent when a call starts - the fast brain\n"
                "handles the initial interaction autonomously."
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


# =============================================================================
# Test: Rapid utterance handling - slow brain should complete even with fast turns
# =============================================================================


@pytest.mark.asyncio
class TestRapidUtteranceHandling:
    """
    Tests for slow brain behavior during rapid user turns.

    When the user speaks faster than the slow brain can think, the system
    must handle this gracefully. The key principle is:

    - Running LLM tasks should complete (not be cancelled by new utterances)
    - Pending tasks can be replaced/debounced
    - This creates a "queue of 2": 1 running + 1 pending

    Without this, rapid speech causes the slow brain to NEVER complete
    thinking, which breaks any functionality that depends on slow brain output
    (action completion, cross-channel notifications, etc.).

    The bug: interject_or_run() uses cancel_running=True, which cancels
    even the in-flight LLM call. With rapid utterances, none ever complete.
    """

    @pytest.fixture
    def boss_contact(self):
        """The boss contact (contact_id=1) who is on the call."""
        return TEST_CONTACTS[1]

    async def test_rapid_utterances_should_allow_llm_completion(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        Test that rapid user utterances don't cancel running LLM tasks.

        This test replicates the REAL production scenario:
        - User is on a voice call
        - User speaks rapidly (multiple utterances while LLM is thinking)
        - LLM takes ~10-15 seconds per thinking step (realistic timing)
        - Each utterance triggers the slow brain via interject_or_run()

        THE BUG (without fix):
        - interject_or_run uses cancel_running=True for all modes
        - Each new utterance CANCELS the in-flight LLM run
        - With rapid speech, NO LLM runs ever complete
        - The slow brain becomes completely non-functional

        THE FIX:
        - interject_or_run uses cancel_running=False for voice mode
        - Debouncer uses asyncio.shield() to protect running tasks
        - Running LLM completes, only pending tasks are debounced
        - "Queue of 2" behavior: 1 running + 1 pending

        Expected results:
        - With fix: cancelled_count = 0, completed_count >= 1
        - Without fix: cancelled_count > 0 (running tasks get cancelled)
        """
        cm = initialized_cm.cm

        # Start a call first
        started_event = PhoneCallStarted(contact=boss_contact)
        await initialized_cm.step(started_event)

        # Verify we're in voice mode
        assert cm.mode == Mode.CALL, "Should be in CALL mode after PhoneCallStarted"

        # Track LLM run completions with REALISTIC simulated thinking time
        llm_completions = []

        # Realistic LLM thinking time - opus-4.5 typically takes 10-20 seconds
        # We use 2s as a compromise between realism and test speed, while still
        # being much longer than the utterance interval (0.1s)
        SIMULATED_LLM_THINKING_TIME = (
            2.0  # seconds (realistic ratio to utterance timing)
        )

        async def tracked_run_llm_with_simulated_delay():
            """
            Simulates realistic LLM thinking time and tracks completion/cancellation.

            We don't call the original _run_llm because:
            - We need deterministic timing for test assertions
            - The simulated delay accurately represents the timing relationship
              between LLM processing and user speech
            """
            try:
                await asyncio.sleep(SIMULATED_LLM_THINKING_TIME)
                llm_completions.append(("completed", None))
                return None
            except asyncio.CancelledError:
                llm_completions.append(("cancelled", None))
                raise

        cm._run_llm = tracked_run_llm_with_simulated_delay

        # Simulate rapid user utterances - this is the REAL scenario
        # User speaks naturally, with multiple utterances arriving while
        # the slow brain is still processing
        utterances = [
            "Hello",
            "Can you help me with something?",
            "I need to schedule a meeting",
            "Actually make that two meetings",
            "One on Monday and one on Friday",
        ]

        # Time between utterances - realistic rapid speech
        # Much shorter than LLM thinking time to trigger the bug
        UTTERANCE_INTERVAL = 0.1  # seconds

        try:
            from unity.conversation_manager.domains.event_handlers import EventHandler

            for i, text in enumerate(utterances):
                event = InboundPhoneUtterance(contact=boss_contact, content=text)

                # Handle the event (triggers interject_or_run -> request_llm_run)
                await EventHandler.handle_event(
                    event,
                    cm,
                    is_voice_call=cm.call_manager.uses_realtime_api,
                )

                # Flush triggers the debouncer (matches production behavior)
                await cm.flush_llm_requests()

                # Rapid speech - utterances arrive faster than LLM can complete
                if i < len(utterances) - 1:
                    await asyncio.sleep(UTTERANCE_INTERVAL)

            # Wait for LLM runs to complete
            # Timeline with fix (cancel_running=False + shield):
            #   t=0.0: U1 -> run 1 starts (2s duration)
            #   t=0.1: U2 -> pending waits for run 1
            #   t=0.2: U3 -> pending replaced
            #   t=0.3: U4 -> pending replaced
            #   t=0.4: U5 -> pending replaced (final pending)
            #   t=2.0: run 1 COMPLETES, run 5 starts
            #   t=4.0: run 5 COMPLETES
            #   Result: 0 cancelled, 2 completed
            #
            # Timeline with bug (cancel_running=True, no shield):
            #   t=0.0: U1 -> run 1 starts
            #   t=0.1: U2 -> run 1 CANCELLED, run 2 starts
            #   t=0.2: U3 -> run 2 CANCELLED, run 3 starts
            #   t=0.3: U4 -> run 3 CANCELLED, run 4 starts
            #   t=0.4: U5 -> run 4 CANCELLED, run 5 starts
            #   t=2.4: run 5 COMPLETES (if we wait)
            #   Result: 4 cancelled, 1 completed

            # Wait long enough for 2 LLM runs to complete with fix
            max_wait_time = 6.0  # 2x LLM time + buffer
            await asyncio.sleep(max_wait_time)

            # Count results
            completed_count = sum(
                1 for status, _ in llm_completions if status == "completed"
            )
            cancelled_count = sum(
                1 for status, _ in llm_completions if status == "cancelled"
            )

            # THE KEY ASSERTION: No running tasks should be cancelled
            #
            # With the fix, running tasks are protected and complete normally.
            # Without the fix, running tasks get cancelled by each new utterance.
            #
            # This assertion will:
            # - PASS with fix: cancelled_count = 0
            # - FAIL without fix: cancelled_count > 0 (typically 3-4)
            assert cancelled_count == 0, (
                f"Running LLM tasks were cancelled by rapid utterances!\n"
                f"  Completed: {completed_count}\n"
                f"  Cancelled: {cancelled_count}\n"
                f"  Utterances sent: {len(utterances)}\n"
                f"  Simulated LLM time: {SIMULATED_LLM_THINKING_TIME}s\n"
                f"  Utterance interval: {UTTERANCE_INTERVAL}s\n"
                f"\n"
                f"This indicates the bug where rapid utterances cancel running\n"
                f"LLM tasks instead of just debouncing pending tasks.\n"
                f"\n"
                f"Required fixes:\n"
                f"1. interject_or_run must use cancel_running=False for voice mode\n"
                f"2. Debouncer must use asyncio.shield() to protect running tasks"
            )

            # Secondary assertion: at least one task should complete
            assert completed_count >= 1, (
                f"No LLM runs completed!\n"
                f"  Completed: {completed_count}\n"
                f"  Cancelled: {cancelled_count}\n"
                f"  Wait time: {max_wait_time}s\n"
            )

        finally:
            cm._run_llm = tracked_run_llm_with_simulated_delay  # Keep mock for teardown
