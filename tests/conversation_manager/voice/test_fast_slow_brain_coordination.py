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
6. **Stale guidance filtering**: Does the system filter out guidance that's no longer
   relevant because the conversation moved on?

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

Known Issue (stale guidance after topic change):
------------------------------------------------
When the user changes topics while the slow brain is thinking, the slow brain's
guidance may be about the OLD topic. Without filtering, this stale guidance is
sent to the fast brain, causing confusing out-of-context speech.
Tests in this file document and verify the fix for this issue.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from unity.conversation_manager.events import (
    PhoneCallStarted,
    InboundPhoneUtterance,
    OutboundPhoneUtterance,
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

    async def test_inbound_call_start_should_not_trigger_greeting_guidance(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        When an INBOUND call starts, the slow brain should NOT run at all and
        should NOT provide conversational guidance like 'Greet the user'.

        The fast brain handles greetings and all conversational aspects autonomously
        for inbound calls where the user is initiating with their own agenda.

        This test documents the bug reported by Ved and verifies the fix:

        BUG (before fix):
        - Inbound call starts
        - Slow brain runs immediately via request_llm_run(delay=0)
        - Slow brain sees "Call started" notification
        - Slow brain outputs call_guidance: "Greet Ved warmly..."
        - Fast brain (which already greeted) receives this guidance and greets AGAIN
        - Result: duplicate speech

        FIX:
        - Inbound call starts
        - Slow brain does NOT run on call start (is_outbound=False)
        - Fast brain handles greeting autonomously
        - Slow brain only runs later when there's actual content to process
        - Result: no duplicate speech

        Note: OUTBOUND calls DO trigger the slow brain to provide initial guidance
        on what to say and why we're calling. See test_outbound_call_start_triggers_guidance.
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

            # The slow brain should NOT run on INBOUND call start.
            # The fast brain handles the greeting autonomously.
            # Triggering the slow brain here would cause unnecessary call_guidance.
            assert not result.llm_ran, (
                "Slow brain should NOT run on PhoneCallStarted for INBOUND calls!\n"
                "\n"
                "The fast brain handles greetings and all conversational aspects\n"
                "autonomously for inbound calls. If the slow brain runs, it may provide\n"
                "call_guidance like 'Greet the user', causing duplicate speech.\n"
                "\n"
                "The slow brain should only be triggered by:\n"
                "- InboundPhoneUtterance (user says something)\n"
                "- ActorResult (action completes)\n"
                "- NotificationInjectedEvent (cross-channel notification)\n"
                "- SMSReceived/EmailReceived while on call\n"
                "\n"
                "Note: Outbound calls DO trigger the slow brain for initial guidance."
            )

            # Also verify no call_guidance was published
            assert len(published_guidance) == 0, (
                f"call_guidance was published on inbound call start: {published_guidance}\n"
                "No guidance should be sent when an inbound call starts - the fast brain\n"
                "handles the initial interaction autonomously."
            )

        finally:
            initialized_cm.cm.event_broker.publish = original_publish

    async def test_outbound_call_start_triggers_guidance(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        When an OUTBOUND call starts, the slow brain SHOULD run to provide
        initial guidance on what to say and why we're calling.

        Unlike inbound calls (where the user initiates with their own agenda),
        outbound calls are initiated by the assistant with a specific purpose.
        The fast brain needs context from the slow brain about:
        - Why we're calling this contact
        - What information to convey or gather
        - Any relevant context from recent interactions

        Flow for outbound calls:
        1. PhoneCallSent triggers call_manager.start_call(outbound=True)
        2. call_manager.is_outbound = True
        3. Fast brain subprocess starts and waits for call_answered
        4. PhoneCallStarted arrives, slow brain sees is_outbound=True
        5. Slow brain runs immediately via request_llm_run(delay=0)
        6. Slow brain generates call_guidance with purpose/context
        7. Guidance is buffered by fast brain (in pending_guidance)
        8. When call is answered, fast brain speaks with the guidance
        """
        cm = initialized_cm.cm

        # Simulate outbound call setup - set is_outbound BEFORE PhoneCallStarted
        # In real flow, this happens in PhoneCallSent handler calling start_call(outbound=True)
        cm.call_manager.is_outbound = True

        try:
            # Simulate call starting (for outbound call)
            event = PhoneCallStarted(contact=boss_contact)
            result = await initialized_cm.step(event)

            # The slow brain SHOULD run for outbound calls to provide initial guidance
            assert result.llm_ran, (
                "Slow brain should run on PhoneCallStarted for OUTBOUND calls!\n"
                "\n"
                "Outbound calls are initiated by the assistant with a purpose.\n"
                "The slow brain must provide initial guidance so the fast brain\n"
                "knows what to say when the call is answered.\n"
                "\n"
                "Check that call_manager.is_outbound is True before PhoneCallStarted."
            )

        finally:
            # Reset outbound flag
            cm.call_manager.is_outbound = False

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

            # Poll for LLM runs to complete (deterministic wait instead of fixed sleep)
            # We expect at least 1 completion. With fix, we get 2 completions (first + final).
            # Use generous timeout for slow CI environments, but poll frequently.
            MAX_WAIT = 30.0  # generous timeout for uncached LLM or slow CI
            POLL_INTERVAL = 0.1
            import time as _time

            start = _time.perf_counter()
            while _time.perf_counter() - start < MAX_WAIT:
                completed_so_far = sum(
                    1 for status, _ in llm_completions if status == "completed"
                )
                # With the fix, we expect 2 completions (first run + final pending)
                # With the bug, we'd see cancellations instead
                if completed_so_far >= 1:
                    # Give a small buffer for any additional completions to register
                    await asyncio.sleep(0.5)
                    break
                await asyncio.sleep(POLL_INTERVAL)

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


# =============================================================================
# Test: Stale guidance filtering - guidance should not be sent if topic changed
# =============================================================================


@pytest.mark.asyncio
class TestStaleGuidanceFiltering:
    """
    Tests for filtering out stale guidance when the conversation has moved on.

    The slow brain takes 10-20 seconds to think. During this time, the conversation
    continues - the user may change topics, the fast brain may respond, etc.

    When the slow brain finally produces guidance, it may be about the OLD topic
    that was being discussed when it STARTED thinking, not the CURRENT topic.

    Without filtering, this stale guidance causes confusing out-of-context speech:
    - User: "What time is the meeting?"
    - (slow brain starts thinking about meeting time)
    - User: "Actually never mind, what's the weather like?"
    - Fast brain: "Let me check the weather for you..."
    - (slow brain finishes): "The meeting is at 3pm Thursday"  <-- STALE!
    - Result: Confusing, out-of-context mention of meeting time

    The fix is a relevance filter that checks if guidance is still relevant before
    sending it to the fast brain. The filter uses a fast model (no extended thinking)
    to quickly assess relevance based on:
    - The guidance content
    - Messages that arrived AFTER the slow brain started thinking
    - Whether the topic/context has changed
    """

    @pytest.fixture
    def boss_contact(self):
        """The boss contact (contact_id=1) who is on the call."""
        return TEST_CONTACTS[1]

    async def test_stale_guidance_should_be_filtered_after_topic_change(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        FAILING TEST: Stale guidance about old topic should NOT be sent to fast brain.

        This test simulates a real-world scenario:
        1. User asks about topic A (meeting time)
        2. Slow brain starts thinking (simulated 3 second delay)
        3. While slow brain thinks, user changes to topic B (weather)
        4. Fast brain responds about weather
        5. Slow brain finishes with guidance about topic A (meeting)
        6. This guidance is NOW STALE - the conversation moved on

        Expected behavior (with relevance filter):
        - The guidance about meeting time should NOT be published to fast brain
        - The filter should detect that the conversation has moved to weather

        Current behavior (without filter - THIS TEST SHOULD FAIL):
        - The guidance about meeting time IS published to fast brain
        - Fast brain speaks about meeting time, confusing the user

        The relevance filter will:
        - Intercept guidance before publishing to app:call:call_guidance
        - Check messages that arrived AFTER the slow brain's snapshot
        - Use a fast model to determine if guidance is still relevant
        - Only publish if relevant; drop if stale
        """
        cm = initialized_cm.cm

        # Start a call
        started_event = PhoneCallStarted(contact=boss_contact)
        await initialized_cm.step(started_event)

        # Verify we're in voice mode
        assert cm.mode == Mode.CALL, "Should be in CALL mode after PhoneCallStarted"

        # Track what guidance is actually published to the fast brain
        published_guidance: list[dict] = []
        original_publish = cm.event_broker.publish

        async def capture_guidance(channel: str, message: str) -> int:
            if channel == "app:call:call_guidance":
                try:
                    data = json.loads(message)
                    # Extract content from either Event format or plain dict
                    if "payload" in data:
                        content = data["payload"].get("content", "")
                    else:
                        content = data.get("content", "")
                    if content:
                        published_guidance.append(
                            {
                                "content": content,
                                "raw": data,
                            },
                        )
                except (json.JSONDecodeError, KeyError):
                    pass
            return await original_publish(channel, message)

        cm.event_broker.publish = capture_guidance

        # Simulate the scenario with realistic timing
        SLOW_BRAIN_THINKING_TIME = 3.0  # Simulated slow brain delay

        try:
            from unity.conversation_manager.domains.event_handlers import EventHandler

            # ─────────────────────────────────────────────────────────────────
            # Step 1: User asks about topic A (meeting time)
            # ─────────────────────────────────────────────────────────────────
            topic_a_utterance = InboundPhoneUtterance(
                contact=boss_contact,
                content="Hey, what time is the meeting tomorrow?",
            )
            await EventHandler.handle_event(
                topic_a_utterance,
                cm,
                is_voice_call=cm.call_manager.uses_realtime_api,
            )

            # Record the state snapshot that slow brain will use
            # (In reality, this happens inside _run_llm, but we need to track it)
            slow_brain_snapshot_time = asyncio.get_event_loop().time()

            # ─────────────────────────────────────────────────────────────────
            # Step 2: Mock slow brain to take time and return guidance about topic A
            # ─────────────────────────────────────────────────────────────────
            # We need to intercept the LLM call, delay it, and return specific guidance
            original_run_llm = cm._run_llm
            slow_brain_started = asyncio.Event()
            slow_brain_can_finish = asyncio.Event()

            # Track when guidance has been published
            guidance_published = asyncio.Event()

            async def slow_brain_with_delay_and_stale_guidance():
                """
                Simulates slow brain that:
                1. Waits for signal to finish (after topic change)
                2. Returns guidance about topic A (meeting time) - which will be stale
                3. Runs the guidance filter to check relevance before publishing
                """
                from unity.common.prompt_helpers import now as prompt_now

                # Get the timestamp of the last message in the voice thread
                # This represents "when the slow brain started" - messages AFTER this
                # timestamp are "new" (arrived while slow brain was thinking)
                contact_id = boss_contact["contact_id"]
                conv_state = cm.contact_index.get_conversation_state(contact_id)
                voice_medium = Medium.PHONE_CALL
                voice_thread = list(conv_state.threads.get(voice_medium, []))

                # Use the timestamp of the LAST message as reference
                # Any message with timestamp > this is "new" (arrived after slow brain started)
                # NOTE: With UNITY_INCREMENTING_TIMESTAMPS, each timestamp is unique and
                # monotonically increasing, so no offset is needed
                if voice_thread:
                    last_msg = voice_thread[-1]
                    slow_brain_start_time = last_msg.timestamp
                else:
                    # Use prompt_now which is monkeypatched in tests to return fixed time
                    slow_brain_start_time = prompt_now(as_string=False)

                slow_brain_started.set()

                # Wait for the signal that we can finish (after topic change)
                await slow_brain_can_finish.wait()

                # Simulate thinking time
                await asyncio.sleep(0.1)  # Small delay for realism

                # The slow brain's guidance is about the ORIGINAL topic (meeting)
                # because that's what was in its snapshot when it started thinking
                stale_guidance_content = (
                    "The meeting tomorrow is scheduled for 3pm in Conference Room B"
                )

                # This simulates what _run_llm does: check guidance relevance before publishing
                # The filter will see that NEW messages (topic change, weather response)
                # arrived after slow_brain_start_time, and should block stale guidance
                should_send = await cm._check_guidance_relevance(
                    stale_guidance_content,
                    slow_brain_start_time,
                )
                guidance_published.set()

                if should_send:
                    # Publish the guidance (only if filter says it's relevant)
                    guidance_event = CallGuidance(
                        contact=boss_contact,
                        content=stale_guidance_content,
                    )
                    await cm.event_broker.publish(
                        "app:call:call_guidance",
                        guidance_event.to_json(),
                    )

                return None

            cm._run_llm = slow_brain_with_delay_and_stale_guidance

            # Trigger the slow brain (it will start but wait for our signal)
            await cm.flush_llm_requests()

            # Wait for slow brain to start
            await asyncio.wait_for(slow_brain_started.wait(), timeout=2.0)

            # ─────────────────────────────────────────────────────────────────
            # Step 3: While slow brain is "thinking", user changes topic to B
            # ─────────────────────────────────────────────────────────────────
            topic_b_utterance = InboundPhoneUtterance(
                contact=boss_contact,
                content="Actually, forget about that. What's the weather like today?",
            )
            await EventHandler.handle_event(
                topic_b_utterance,
                cm,
                is_voice_call=cm.call_manager.uses_realtime_api,
            )

            # ─────────────────────────────────────────────────────────────────
            # Step 4: Fast brain responds about weather (simulated)
            # ─────────────────────────────────────────────────────────────────
            # Simulate fast brain's response about weather
            # Note: Fast brain only acknowledges - it doesn't hallucinate actual weather data
            # (the slow brain would provide real data via guidance)
            fast_brain_response = OutboundPhoneUtterance(
                contact=boss_contact,
                content="Let me check the weather for you.",
            )
            await EventHandler.handle_event(
                fast_brain_response,
                cm,
                is_voice_call=cm.call_manager.uses_realtime_api,
            )

            # ─────────────────────────────────────────────────────────────────
            # Step 5: Now let slow brain finish (it will publish stale guidance)
            # ─────────────────────────────────────────────────────────────────
            slow_brain_can_finish.set()

            # Give time for slow brain to complete - includes LLM call to GuidanceFilter
            # which can take several seconds (opus-4.5 without thinking)
            await asyncio.sleep(10.0)

            # ─────────────────────────────────────────────────────────────────
            # Step 6: Verify that stale guidance was NOT sent to fast brain
            # ─────────────────────────────────────────────────────────────────

            # Find any guidance about the meeting (topic A)
            meeting_guidance = [
                g
                for g in published_guidance
                if "meeting" in g["content"].lower()
                or "3pm" in g["content"].lower()
                or "conference room" in g["content"].lower()
            ]

            # THE KEY ASSERTION: Stale guidance should be filtered out
            #
            # With relevance filter: meeting_guidance should be empty (filtered)
            # Without filter (current): meeting_guidance will contain the stale guidance
            #
            # This test SHOULD FAIL until the relevance filter is implemented
            assert len(meeting_guidance) == 0, (
                f"Stale guidance was sent to fast brain!\n"
                f"\n"
                f"The conversation moved from 'meeting time' to 'weather', but the\n"
                f"slow brain's guidance about the meeting was still published.\n"
                f"\n"
                f"Published guidance about meeting:\n"
                f"  {[g['content'] for g in meeting_guidance]}\n"
                f"\n"
                f"Conversation flow:\n"
                f"  1. User: 'What time is the meeting tomorrow?'\n"
                f"  2. (slow brain starts thinking...)\n"
                f"  3. User: 'Actually, forget about that. What's the weather?'\n"
                f"  4. Fast brain: 'Let me check the weather for you.'\n"
                f"  5. Slow brain finishes: 'Meeting is at 3pm' <-- STALE!\n"
                f"\n"
                f"Required fix:\n"
                f"  Implement a relevance filter that checks if guidance is still\n"
                f"  relevant before sending it to the fast brain. The filter should:\n"
                f"  1. Capture the conversation state when slow brain STARTED thinking\n"
                f"  2. Compare to current state when slow brain FINISHES\n"
                f"  3. Use a fast model to determine if guidance is still relevant\n"
                f"  4. Drop guidance if the topic/context has changed"
            )

        finally:
            cm.event_broker.publish = original_publish
            cm._run_llm = original_run_llm

    async def test_relevant_guidance_should_still_be_sent(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        Control test: Guidance that IS still relevant should be sent normally.

        This is an UNAMBIGUOUS scenario where the guidance DIRECTLY ANSWERS the
        user's follow-up question. The relevance filter should clearly allow this.

        Scenario:
        1. User asks: "What time is the meeting tomorrow?"
        2. Slow brain starts thinking...
        3. User asks: "Sorry, which room is the meeting in?"
        4. Slow brain finishes with: "Meeting is at 3pm in Conference Room B"
        5. Guidance DIRECTLY ANSWERS the follow-up (Conference Room B) - MUST be sent

        This is unambiguous because:
        - Topic is clearly the same (meeting)
        - Guidance contains the exact info the user is asking about (room)
        - No topic change whatsoever
        """
        cm = initialized_cm.cm

        # Start a call
        started_event = PhoneCallStarted(contact=boss_contact)
        await initialized_cm.step(started_event)

        # Track published guidance
        published_guidance: list[dict] = []
        original_publish = cm.event_broker.publish

        async def capture_guidance(channel: str, message: str) -> int:
            if channel == "app:call:call_guidance":
                try:
                    data = json.loads(message)
                    if "payload" in data:
                        content = data["payload"].get("content", "")
                    else:
                        content = data.get("content", "")
                    if content:
                        published_guidance.append({"content": content})
                except (json.JSONDecodeError, KeyError):
                    pass
            return await original_publish(channel, message)

        cm.event_broker.publish = capture_guidance

        try:
            from unity.conversation_manager.domains.event_handlers import EventHandler

            # User asks about meeting time
            utterance1 = InboundPhoneUtterance(
                contact=boss_contact,
                content="What time is the meeting tomorrow?",
            )
            await EventHandler.handle_event(
                utterance1,
                cm,
                is_voice_call=cm.call_manager.uses_realtime_api,
            )

            # Mock slow brain
            original_run_llm = cm._run_llm
            slow_brain_started = asyncio.Event()
            slow_brain_can_finish = asyncio.Event()
            guidance_published = asyncio.Event()

            async def slow_brain_with_relevant_guidance():
                from unity.common.prompt_helpers import now as prompt_now

                # Get the timestamp of the last message in the voice thread
                # This represents "when the slow brain started" - messages AFTER this
                # timestamp are "new" (arrived while slow brain was thinking)
                contact_id = boss_contact["contact_id"]
                conv_state = cm.contact_index.get_conversation_state(contact_id)
                voice_medium = Medium.PHONE_CALL
                voice_thread = list(conv_state.threads.get(voice_medium, []))

                # Use the timestamp of the LAST message as reference
                # Any message with timestamp > this is "new" (arrived after slow brain started)
                # NOTE: With UNITY_INCREMENTING_TIMESTAMPS, each timestamp is unique and
                # monotonically increasing, so no offset is needed
                if voice_thread:
                    last_msg = voice_thread[-1]
                    slow_brain_start_time = last_msg.timestamp
                else:
                    # Use prompt_now which is monkeypatched in tests to return fixed time
                    slow_brain_start_time = prompt_now(as_string=False)

                slow_brain_started.set()
                await slow_brain_can_finish.wait()

                # Guidance contains BOTH time AND room - directly answers follow-up
                relevant_guidance = (
                    "The meeting tomorrow is at 3pm in Conference Room B"
                )

                # Check if guidance is still relevant (it should be - same topic)
                should_send = await cm._check_guidance_relevance(
                    relevant_guidance,
                    slow_brain_start_time,
                )

                if should_send:
                    guidance_event = CallGuidance(
                        contact=boss_contact,
                        content=relevant_guidance,
                    )
                    await cm.event_broker.publish(
                        "app:call:call_guidance",
                        guidance_event.to_json(),
                    )
                return None

            cm._run_llm = slow_brain_with_relevant_guidance
            await cm.flush_llm_requests()
            await asyncio.wait_for(slow_brain_started.wait(), timeout=2.0)

            # ─────────────────────────────────────────────────────────────────
            # KEY DIFFERENCE FROM STALE TEST: User asks for CLARIFICATION about
            # information that IS IN THE GUIDANCE. This is unambiguously relevant.
            # ─────────────────────────────────────────────────────────────────
            await asyncio.sleep(0.2)
            utterance2 = InboundPhoneUtterance(
                contact=boss_contact,
                content="Sorry, which room did you say the meeting is in?",
            )
            await EventHandler.handle_event(
                utterance2,
                cm,
                is_voice_call=cm.call_manager.uses_realtime_api,
            )

            # Let slow brain finish and wait for guidance to be published
            slow_brain_can_finish.set()
            # Wait for the mock to complete - includes LLM call to GuidanceFilter
            # which can take several seconds (opus-4.5 without thinking)
            await asyncio.sleep(10.0)

            # Guidance about meeting should be sent - it DIRECTLY ANSWERS the follow-up
            # question about the room (Conference Room B is in the guidance)
            meeting_guidance = [
                g
                for g in published_guidance
                if "conference room" in g["content"].lower()
            ]

            assert len(meeting_guidance) >= 1, (
                f"Relevant guidance was filtered incorrectly!\n"
                f"\n"
                f"The user asked about meeting time, then asked 'which room is it in?'\n"
                f"The guidance contains 'Conference Room B' - it DIRECTLY ANSWERS\n"
                f"the follow-up question. This should definitely be sent.\n"
                f"\n"
                f"Published guidance: {[g['content'] for g in published_guidance]}\n"
            )

        finally:
            cm.event_broker.publish = original_publish
            cm._run_llm = original_run_llm


# =============================================================================
# Test: User corrections and restatements - guidance about wrong entity
# =============================================================================


@pytest.mark.asyncio
class TestUserCorrectionsAndRestatements:
    """
    Tests for when the user corrects or clarifies their request while slow brain is thinking.

    This is different from a topic CHANGE - the user is still asking about the same
    type of thing (e.g., "a meeting"), but they're correcting WHICH specific instance
    they mean (e.g., "the Friday meeting, not Thursday").

    The guidance filter currently checks for topic changes, but it may not catch
    corrections where the general topic stays the same but the specific entity changes.

    Example scenario:
        User: "What time is the meeting?"
        (slow brain starts thinking about THE meeting - assumes Thursday)
        User: "I mean the Friday meeting, not Thursday"
        Fast brain: "Got it, checking the Friday meeting"
        Slow brain: "Meeting is at 3pm in Room A"  ← This is THURSDAY's meeting!

    The guidance is about "a meeting" (same topic), but it's the WRONG meeting.
    Sending this guidance would cause the fast brain to give incorrect information.
    """

    @pytest.fixture
    def boss_contact(self):
        """The boss contact (contact_id=1) who is on the call."""
        return TEST_CONTACTS[1]

    async def test_implicit_entity_correction_should_block_wrong_entity_guidance(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        SUBTLE TEST: When user implicitly switches to a different entity (without
        explicitly rejecting the original), guidance about the original entity
        should still be blocked.

        This is harder than explicit correction ("not the Thursday one") because:
        - User just mentions a DIFFERENT entity
        - No explicit rejection of the original
        - The filter must infer the correction from context

        Scenario:
            User: "What time is the status meeting?"
            (slow brain starts thinking about status meeting)
            User: "Oh wait, I meant the budget review"
            Fast brain: "Checking the budget review..."
            Slow brain: "The status meeting is at 2pm"  ← Should this be blocked?

        The guidance filter might see:
        - Both are about "meetings" (same general topic)
        - User didn't explicitly say "not the status meeting"
        - Example 2 in prompt says same-topic follow-ups should SEND

        But the CORRECT behavior is:
        - User switched to asking about a DIFFERENT meeting (budget review)
        - Guidance about status meeting is now stale/irrelevant
        - Should be BLOCKED

        This tests whether the filter understands IMPLICIT entity corrections.
        """
        cm = initialized_cm.cm

        # Start a call
        started_event = PhoneCallStarted(contact=boss_contact)
        await initialized_cm.step(started_event)

        assert cm.mode == Mode.CALL, "Should be in CALL mode"

        # Track published guidance
        published_guidance: list[dict] = []
        original_publish = cm.event_broker.publish

        async def capture_guidance(channel: str, message: str) -> int:
            if channel == "app:call:call_guidance":
                try:
                    data = json.loads(message)
                    if "payload" in data:
                        content = data["payload"].get("content", "")
                    else:
                        content = data.get("content", "")
                    if content:
                        published_guidance.append({"content": content})
                except (json.JSONDecodeError, KeyError):
                    pass
            return await original_publish(channel, message)

        cm.event_broker.publish = capture_guidance

        try:
            from unity.conversation_manager.domains.event_handlers import EventHandler

            # ─────────────────────────────────────────────────────────────────
            # Step 1: User asks about the "status meeting"
            # ─────────────────────────────────────────────────────────────────
            initial_question = InboundPhoneUtterance(
                contact=boss_contact,
                content="What time is the status meeting?",
            )
            await EventHandler.handle_event(
                initial_question,
                cm,
                is_voice_call=cm.call_manager.uses_realtime_api,
            )

            # Mock slow brain
            original_run_llm = cm._run_llm
            slow_brain_started = asyncio.Event()
            slow_brain_can_finish = asyncio.Event()

            async def slow_brain_with_status_meeting_guidance():
                """
                Simulates slow brain that:
                1. Started thinking about "status meeting"
                2. Produces guidance about the STATUS meeting
                3. But user has since switched to asking about BUDGET REVIEW
                """
                from unity.common.prompt_helpers import now as prompt_now

                contact_id = boss_contact["contact_id"]
                conv_state = cm.contact_index.get_conversation_state(contact_id)
                voice_medium = Medium.PHONE_CALL
                voice_thread = list(conv_state.threads.get(voice_medium, []))

                if voice_thread:
                    last_msg = voice_thread[-1]
                    slow_brain_start_time = last_msg.timestamp
                else:
                    slow_brain_start_time = prompt_now(as_string=False)

                slow_brain_started.set()
                await slow_brain_can_finish.wait()
                await asyncio.sleep(0.1)

                # Guidance is about STATUS meeting - but user switched to BUDGET REVIEW
                # No explicit rejection, just a different entity mentioned
                wrong_meeting_guidance = (
                    "The status meeting is scheduled for 2pm in the Main Conference Room. "
                    "The usual attendees are the engineering team leads."
                )

                should_send = await cm._check_guidance_relevance(
                    wrong_meeting_guidance,
                    slow_brain_start_time,
                )

                if should_send:
                    guidance_event = CallGuidance(
                        contact=boss_contact,
                        content=wrong_meeting_guidance,
                    )
                    await cm.event_broker.publish(
                        "app:call:call_guidance",
                        guidance_event.to_json(),
                    )

                return None

            cm._run_llm = slow_brain_with_status_meeting_guidance
            await cm.flush_llm_requests()
            await asyncio.wait_for(slow_brain_started.wait(), timeout=2.0)

            # ─────────────────────────────────────────────────────────────────
            # Step 2: User IMPLICITLY switches to a different meeting
            # Note: NO explicit "not the status meeting" - just mentions different one
            # ─────────────────────────────────────────────────────────────────
            await asyncio.sleep(0.2)
            user_implicit_switch = InboundPhoneUtterance(
                contact=boss_contact,
                content="Oh wait, I meant the budget review. When is that?",
            )
            await EventHandler.handle_event(
                user_implicit_switch,
                cm,
                is_voice_call=cm.call_manager.uses_realtime_api,
            )

            # ─────────────────────────────────────────────────────────────────
            # Step 3: Fast brain acknowledges and switches context
            # ─────────────────────────────────────────────────────────────────
            fast_brain_ack = OutboundPhoneUtterance(
                contact=boss_contact,
                content="Sure, let me look up the budget review meeting.",
            )
            await EventHandler.handle_event(
                fast_brain_ack,
                cm,
                is_voice_call=cm.call_manager.uses_realtime_api,
            )

            # ─────────────────────────────────────────────────────────────────
            # Step 4: Let slow brain finish with WRONG meeting guidance
            # ─────────────────────────────────────────────────────────────────
            slow_brain_can_finish.set()

            # Wait for guidance filter (LLM call)
            await asyncio.sleep(10.0)

            # ─────────────────────────────────────────────────────────────────
            # Step 5: Verify guidance about WRONG meeting was blocked
            # ─────────────────────────────────────────────────────────────────
            status_meeting_guidance = [
                g
                for g in published_guidance
                if "status meeting" in g["content"].lower()
                or "2pm" in g["content"].lower()
                or "engineering team" in g["content"].lower()
            ]

            # THE KEY ASSERTION: Guidance about the WRONG entity should be blocked
            #
            # This test may FAIL because:
            # - The filter sees "meeting" in both guidance and conversation
            # - User didn't explicitly say "not the status meeting"
            # - Filter might think this is same-topic (both about meetings)
            #
            # But the user clearly switched to a DIFFERENT meeting (budget review).
            # The guidance about status meeting is now irrelevant.
            assert len(status_meeting_guidance) == 0, (
                f"Guidance about WRONG meeting was sent to fast brain!\n"
                f"\n"
                f"The user implicitly switched: 'Oh wait, I meant the budget review'\n"
                f"But slow brain guidance was about the status meeting.\n"
                f"\n"
                f"Published status meeting guidance:\n"
                f"  {[g['content'] for g in status_meeting_guidance]}\n"
                f"\n"
                f"Conversation flow:\n"
                f"  1. User: 'What time is the status meeting?'\n"
                f"  2. (slow brain starts thinking about status meeting)\n"
                f"  3. User: 'Oh wait, I meant the budget review. When is that?'\n"
                f"  4. Fast brain: 'Sure, let me look up the budget review meeting.'\n"
                f"  5. Slow brain: 'The status meeting is at 2pm...'  ← WRONG!\n"
                f"\n"
                f"Unlike explicit correction ('not the status meeting'), this was\n"
                f"an IMPLICIT switch - user just mentioned a different meeting.\n"
                f"The filter needs to recognize that 'status meeting' ≠ 'budget review'\n"
                f"even though both are meetings (same general topic, different entity).\n"
            )

        finally:
            cm.event_broker.publish = original_publish
            cm._run_llm = original_run_llm
