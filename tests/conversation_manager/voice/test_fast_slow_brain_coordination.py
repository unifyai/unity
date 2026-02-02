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
        FAILING TEST: With rapid user utterances, at least some LLM runs
        should complete within a reasonable time window.

        This test simulates the scenario where:
        - User is on a voice call
        - User speaks multiple times rapidly (faster than LLM thinking time)
        - Each utterance triggers interject_or_run() -> request_llm_run()
        - flush_llm_requests() submits to the debouncer

        Current behavior (BUG):
        - interject_or_run uses cancel_running=True
        - Each new utterance cancels the in-flight LLM run
        - With 5 utterances, we see ~4 cancellations, 0-1 completions
        - If user keeps talking indefinitely, slow brain never completes

        Expected behavior (after fix):
        - interject_or_run uses cancel_running=False for voice mode
        - Running LLM completes, only pending tasks are replaced
        - With 5 utterances, we see 0 cancellations, 1-2 completions
        - First run completes, final debounced run completes

        NOTE: We simulate LLM thinking time (0.5s) because UNIFY_CACHE=True
        causes cached LLM calls to return immediately, which would not
        properly test the cancellation behavior.
        """
        cm = initialized_cm.cm

        # Start a call first
        started_event = PhoneCallStarted(contact=boss_contact)
        await initialized_cm.step(started_event)

        # Verify we're in voice mode
        assert cm.mode == Mode.CALL, "Should be in CALL mode after PhoneCallStarted"

        # Track LLM run completions with simulated thinking time
        llm_completions = []
        original_run_llm = cm._run_llm

        # Simulated LLM thinking time - must be longer than the interval
        # between utterances (0.1s) to demonstrate the cancellation issue
        SIMULATED_LLM_THINKING_TIME = 0.5  # seconds

        async def tracked_run_llm_with_simulated_delay():
            """
            Wrapper that:
            1. Simulates LLM thinking time (to test cancellation behavior)
            2. Tracks when _run_llm completes vs gets cancelled
            """
            try:
                # Simulate slow LLM thinking time BEFORE the actual call
                # This ensures the test works regardless of cache hits
                await asyncio.sleep(SIMULATED_LLM_THINKING_TIME)

                # Now run the actual LLM (may be cached/fast, but we already waited)
                result = await original_run_llm()
                llm_completions.append(("completed", result))
                return result
            except asyncio.CancelledError:
                llm_completions.append(("cancelled", None))
                raise

        cm._run_llm = tracked_run_llm_with_simulated_delay

        # Simulate rapid user utterances
        # We'll send 5 utterances with 0.1s between them
        # Simulated LLM thinking time is 0.5s, so utterances arrive faster
        # than the LLM can complete
        utterances = [
            "Hello",
            "Can you help me with something?",
            "I need to schedule a meeting",
            "Actually make that two meetings",
            "One on Monday and one on Friday",
        ]

        # Time between utterances - must be LESS than simulated LLM time
        # to demonstrate the rapid-fire cancellation issue
        UTTERANCE_INTERVAL = 0.1  # seconds

        try:
            # Send utterances rapidly using the real event handling path
            # We must call flush_llm_requests() after each event to trigger
            # the debouncer, just like the production event loop does.
            from unity.conversation_manager.domains.event_handlers import EventHandler

            for i, text in enumerate(utterances):
                event = InboundPhoneUtterance(contact=boss_contact, content=text)

                # Handle the event (this calls interject_or_run -> request_llm_run)
                await EventHandler.handle_event(
                    event,
                    cm,
                    is_voice_call=cm.call_manager.uses_realtime_api,
                )

                # Flush triggers the debouncer (matches production behavior)
                # This is where cancel_running=True causes the problem
                await cm.flush_llm_requests()

                # Small delay between utterances (simulating rapid speech)
                # This also gives async tasks a chance to execute
                if i < len(utterances) - 1:
                    await asyncio.sleep(UTTERANCE_INTERVAL)

            # Wait for any pending LLM runs to have a chance to complete
            # Timeline with bug (cancel_running=True):
            #   t=0.0: utterance 1 -> LLM run 1 starts
            #   t=0.1: utterance 2 -> LLM run 1 CANCELLED, run 2 starts
            #   t=0.2: utterance 3 -> LLM run 2 CANCELLED, run 3 starts
            #   t=0.3: utterance 4 -> LLM run 3 CANCELLED, run 4 starts
            #   t=0.4: utterance 5 -> LLM run 4 CANCELLED, run 5 starts
            #   t=0.9: LLM run 5 completes (if we wait)
            #   Result: 4 cancelled, 0-1 completed
            #
            # Timeline with fix (cancel_running=False):
            #   t=0.0: utterance 1 -> LLM run 1 starts
            #   t=0.1: utterance 2 -> pending for run 2 created (waits for run 1)
            #   t=0.2: utterance 3 -> pending replaced with run 3
            #   t=0.3: utterance 4 -> pending replaced with run 4
            #   t=0.4: utterance 5 -> pending replaced with run 5
            #   t=0.5: LLM run 1 COMPLETES, run 5 starts
            #   t=1.0: LLM run 5 COMPLETES
            #   Result: 0 cancelled, 2 completed

            max_wait_time = 3.0  # seconds (enough for 2 LLM runs @ 0.5s each + buffer)
            check_interval = 0.2  # seconds
            elapsed = 0.0

            while elapsed < max_wait_time:
                await asyncio.sleep(check_interval)
                elapsed += check_interval

                # Check if we have any completions
                completed_count = sum(
                    1 for status, _ in llm_completions if status == "completed"
                )
                if completed_count > 0:
                    break

                # Also check if debouncer has a running task that's done
                if cm.debouncer.running_task and cm.debouncer.running_task.done():
                    # Give a moment for completion tracking to update
                    await asyncio.sleep(0.1)
                    break

            # Count results
            completed_count = sum(
                1 for status, _ in llm_completions if status == "completed"
            )
            cancelled_count = sum(
                1 for status, _ in llm_completions if status == "cancelled"
            )

            # The key assertion: at least ONE LLM run should complete
            #
            # Before fix: cancelled_count ~= 4, completed_count ~= 0-1
            #   (only the last run might complete if we wait long enough)
            #
            # After fix: cancelled_count = 0, completed_count >= 1
            #   (first run completes, final debounced run also completes)
            assert completed_count >= 1, (
                f"No LLM runs completed with rapid utterances!\n"
                f"  Completed: {completed_count}\n"
                f"  Cancelled: {cancelled_count}\n"
                f"  Utterances sent: {len(utterances)}\n"
                f"  Simulated LLM time: {SIMULATED_LLM_THINKING_TIME}s\n"
                f"  Utterance interval: {UTTERANCE_INTERVAL}s\n"
                f"  Wait time: {elapsed:.1f}s\n"
                f"\n"
                f"This indicates the bug where interject_or_run() uses\n"
                f"cancel_running=True, causing every new utterance to cancel\n"
                f"the in-flight LLM run. With rapid speech, none ever complete.\n"
                f"\n"
                f"Fix: Use cancel_running=False for voice mode so running\n"
                f"LLM tasks complete while only pending tasks are debounced."
            )

        finally:
            cm._run_llm = original_run_llm
