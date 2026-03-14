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
  real-time voice conversation. Runs in medium_scripts/call.py.

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
    PhoneCallSent,
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    UnifyMeetReceived,
    UnifyMeetStarted,
    FastBrainNotification,
)
from unity.conversation_manager.types import Medium, Mode

from tests.conversation_manager.conftest import BOSS, TEST_CONTACTS
from tests.helpers import _handle_project

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
            if channel == "app:call:notification":
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
                "Note: Outbound calls receive initial guidance via the make_call\n"
                "tool's `context` param, not from a separate LLM run."
            )

            # Also verify no call_guidance was published
            assert len(published_guidance) == 0, (
                f"call_guidance was published on inbound call start: {published_guidance}\n"
                "No guidance should be sent when an inbound call starts - the fast brain\n"
                "handles the initial interaction autonomously."
            )

        finally:
            initialized_cm.cm.event_broker.publish = original_publish

    async def test_outbound_call_sent_does_not_trigger_llm(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        When an OUTBOUND call is initiated (PhoneCallSent), the slow brain should
        NOT run a separate LLM step.

        Initial guidance for outbound calls is captured by the make_call tool's
        `context` parameter and published to the fast brain by
        CallManager.start_call() after the subprocess spawns.  This eliminates
        the race condition where a slow LLM finishes after the fast brain has
        already spoken.

        Flow for outbound calls:
        1. Slow brain decides to call → calls make_call(context="...")
        2. make_call stores context on call_manager.initial_notification
        3. comms_utils.start_call() places the call
        4. PhoneCallSent arrives → event handler spawns subprocess
        5. CallManager.start_call() publishes stored guidance as FastBrainNotification
        6. Fast brain receives guidance via on_guidance / pending_guidance buffer
        7. PhoneCallStarted arrives → mode set to CALL
        8. Ongoing guidance flows via the guide_voice_agent tool (called in parallel)
        """
        cm = initialized_cm.cm

        # Mock start_call to avoid spawning actual subprocess
        original_start_call = cm.call_manager.start_call

        async def mock_start_call(contact, boss, outbound=False):
            cm.call_manager.is_outbound = outbound

        cm.call_manager.start_call = mock_start_call

        try:
            # Simulate outbound call being sent
            event = PhoneCallSent(contact=boss_contact)
            result = await initialized_cm.step(event)

            # is_outbound should be set by the event handler
            assert (
                cm.call_manager.is_outbound
            ), "is_outbound should be True after PhoneCallSent"

            # The slow brain should NOT run on PhoneCallSent.
            # Initial guidance is provided by the make_call tool's context param,
            # not by a separate LLM run triggered from this event.
            assert not result.llm_ran, (
                "Slow brain should NOT run on PhoneCallSent!\n"
                "\n"
                "Initial guidance for outbound calls is captured by the make_call\n"
                "tool's `context` parameter and published to the fast brain by\n"
                "CallManager.start_call() after the subprocess spawns.\n"
                "\n"
                "Triggering a separate LLM run here creates a race condition where\n"
                "the slow brain may not finish before the fast brain speaks."
            )

        finally:
            # Reset state
            cm.call_manager.start_call = original_start_call
            cm.call_manager.is_outbound = False
            cm.mode = Mode.TEXT

    async def test_call_guidance_delivered_via_standalone_tool(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        Guidance is delivered via the standalone guide_voice_agent tool, not as
        a parameter on wait/send_sms/act, and not via the response content field.

        Verify: guide_voice_agent exists as a standalone method with the expected
        signature, wait() does NOT have call_guidance params, and the response
        model for voice modes does NOT have a call_guidance field.
        """
        import inspect
        from unity.conversation_manager.domains.brain import build_response_models
        from unity.conversation_manager.domains.brain_action_tools import (
            ConversationManagerBrainActionTools,
        )
        from unity.conversation_manager.types import Mode

        # guide_voice_agent should exist as a standalone method
        guide_sig = inspect.signature(
            ConversationManagerBrainActionTools.guide_voice_agent,
        )
        assert (
            "content" in guide_sig.parameters
        ), "guide_voice_agent() must accept a content parameter"
        assert (
            "should_speak" in guide_sig.parameters
        ), "guide_voice_agent() must accept a should_speak parameter"
        assert (
            "response_text" in guide_sig.parameters
        ), "guide_voice_agent() must accept a response_text parameter"

        # wait() should NOT have call_guidance params (moved to standalone tool)
        wait_sig = inspect.signature(ConversationManagerBrainActionTools.wait)
        assert "call_guidance" not in wait_sig.parameters, (
            "call_guidance should NOT be a parameter on wait() — "
            "it is now a standalone guide_voice_agent tool"
        )

        # The response model for voice modes should NOT contain call_guidance
        models = build_response_models()
        voice_model = models[Mode.CALL]
        schema = voice_model.model_json_schema()
        props = schema.get("properties", {})
        assert "call_guidance" not in props, (
            "call_guidance should NOT be in the response model — "
            "it is delivered via the standalone guide_voice_agent tool"
        )


@pytest.mark.asyncio
class TestSlowBrainAppropriateGuidance:
    """
    Tests that guide_voice_agent IS used correctly for its intended purposes:
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
        guidance = FastBrainNotification(
            contact=boss_contact,
            content="The meeting time mentioned in the earlier SMS was 3pm on Thursday",
        )
        result = await initialized_cm.step(guidance)

        # This should be recorded in the voice thread
        contact_id = boss_contact["contact_id"]
        voice_thread = initialized_cm.cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )

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
        guidance = FastBrainNotification(
            contact=boss_contact,
            content="SMS just received from Alice: 'Running 10 minutes late'",
        )
        result = await initialized_cm.step(guidance)

        # Verify it was recorded
        contact_id = boss_contact["contact_id"]
        voice_thread = initialized_cm.cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )

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

        # Disable the speech urgency evaluator for this test. The urgency
        # evaluator is a separate sidecar that can legitimately preempt
        # stale slow-brain runs — tested independently in test_speech_urgency.
        # Here we isolate the base debouncing guarantee: cancel_running=False
        # plus asyncio.shield() protect running tasks from being replaced.
        from unity.settings import SETTINGS

        orig = SETTINGS.conversation.SPEECH_URGENCY_PREEMPT_ENABLED
        SETTINGS.conversation.SPEECH_URGENCY_PREEMPT_ENABLED = False

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

        async def tracked_run_llm_with_simulated_delay(**kwargs):
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

            # Wait for all LLM runs to resolve (completed or cancelled).
            # With fix: 2 events (run 1 completes at ~2s, run 5 completes at ~4s)
            # With bug: 5 events (4 instant cancellations + 1 completion at ~2.4s)
            # Minimum possible total is 2 (fix case), so wait for that.
            EXPECTED_MIN_TOTAL = 2
            MAX_WAIT = 30.0  # generous safety timeout
            POLL_INTERVAL = 0.1
            import time as _time

            start = _time.perf_counter()
            while _time.perf_counter() - start < MAX_WAIT:
                if len(llm_completions) >= EXPECTED_MIN_TOTAL:
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
                f"  Wait time: {MAX_WAIT}s\n"
            )

        finally:
            SETTINGS.conversation.SPEECH_URGENCY_PREEMPT_ENABLED = orig
            cm._run_llm = tracked_run_llm_with_simulated_delay  # Keep mock for teardown


# (TestGuidanceRelevanceGuardrails and TestStaleGuidanceArticulation removed —
# the GuidanceArticulator has been deleted.)


# (TestInFlightActionOrchestration removed — its premise that the slow brain
# must relay actor results via guide_voice_agent was invalidated when the fast
# brain gained direct event visibility via boss notification rendering.)


# =============================================================================
# Test: guide_voice_agent SPEAK mode — should_speak + response_text pipeline
# =============================================================================


@pytest.mark.asyncio
class TestFastBrainNotificationSpeakMode:
    """Verify that guide_voice_agent's should_speak and response_text
    parameters propagate correctly through the FastBrainNotification event to the
    fast brain."""

    async def test_should_speak_params_carried_through_to_event(
        self,
        initialized_cm,
    ):
        """When the slow brain calls guide_voice_agent with should_speak=True
        and response_text, the published FastBrainNotification event must carry both
        fields so the fast brain can speak via TTS."""
        import json

        cm = initialized_cm.cm
        boss = BOSS

        # Enter voice mode
        await initialized_cm.step(PhoneCallStarted(contact=boss))
        assert cm.mode == Mode.CALL

        # Capture published guidance events
        published: list[dict] = []
        original_publish = cm.event_broker.publish

        async def capture_publish(channel: str, message: str) -> int:
            if channel == "app:call:notification":
                published.append(json.loads(message))
            return await original_publish(channel, message)

        cm.event_broker.publish = capture_publish

        try:
            # Simulate a user question that should trigger the slow brain
            # to produce guidance with the result
            result = await initialized_cm.step_until_wait(
                InboundPhoneUtterance(
                    contact=boss,
                    content="What's the weather like in San Francisco today?",
                ),
                max_steps=5,
            )

            # Check that guide_voice_agent exists with the expected params
            import inspect
            from unity.conversation_manager.domains.brain_action_tools import (
                ConversationManagerBrainActionTools,
            )

            guide_sig = inspect.signature(
                ConversationManagerBrainActionTools.guide_voice_agent,
            )
            assert (
                "should_speak" in guide_sig.parameters
            ), "guide_voice_agent() must accept should_speak"
            assert (
                "response_text" in guide_sig.parameters
            ), "guide_voice_agent() must accept response_text"

            # Verify published guidance events carry the fields (even if
            # the LLM didn't use them this turn, the schema must support them)
            for event_data in published:
                payload = event_data.get("payload", event_data)
                assert (
                    "should_speak" in payload
                ), f"FastBrainNotification event missing should_speak field: {payload}"
                assert (
                    "response_text" in payload
                ), f"FastBrainNotification event missing response_text field: {payload}"

        finally:
            cm.event_broker.publish = original_publish


# =============================================================================
# Test: ask answers are forwarded to fast brain via event rendering
# =============================================================================


@pytest.mark.asyncio
class TestAskAnswerEventForwarding:
    """Verify that ActorHandleResponse events are rendered for the fast brain
    and that guide_voice_agent is stripped during boss calls.

    The fast brain receives ask answers directly via the notification pipeline
    (render_event_for_fast_brain renders ActorHandleResponse). The slow brain
    does not need guide_voice_agent on boss calls — all relevant state flows
    through event-level forwarding.
    """

    def test_render_event_for_fast_brain_renders_ask_answer(self):
        """render_event_for_fast_brain should render ActorHandleResponse
        events so _render_boss_notifications forwards them to the fast brain.
        """
        from unity.conversation_manager.events import ActorHandleResponse
        from unity.conversation_manager.medium_scripts.common import (
            render_event_for_fast_brain,
        )

        event = ActorHandleResponse(
            handle_id=0,
            action_name="ask",
            query="How did you break down the task?",
            response=(
                "I followed the Colliers standard workflow: "
                "1) searched for guidance, 2) verified all 4 PDFs, "
                "3) rendered pages 2-3 at a time, 4) extracted into "
                "FiscalYearData schema, 5) saved JSON, 6) generated "
                "Excel. Key finding: FYE 2024 had a net loss of "
                "£136K with interest costs doubling."
            ),
            call_id="",
        )
        rendered = render_event_for_fast_brain(event.to_json())
        assert rendered is not None, "ActorHandleResponse should be rendered"
        assert "Ask answered" in rendered
        assert "Colliers standard workflow" in rendered

    @_handle_project
    async def test_guide_voice_agent_stripped_during_boss_meet(
        self,
        initialized_cm,
    ):
        """guide_voice_agent should be stripped for boss-on-call, preventing
        stale slow brain guidance. The fast brain receives all relevant state
        via the notification pipeline instead.
        """
        cm = initialized_cm

        await cm.step(UnifyMeetReceived(contact=BOSS))
        await cm.step(UnifyMeetStarted(contact=BOSS))
        assert cm.cm.mode == Mode.MEET

        cm.cm.completed_actions[0] = {
            "query": "Process 4 ASH PDF accounts into standard format",
            "handle_actions": [
                {
                    "action_name": "act_completed",
                    "query": "All 4 PDFs processed, Excel output generated.",
                    "status": "completed",
                },
                {
                    "action_name": "ask_0",
                    "query": "How did you break down the task?",
                    "status": "completed",
                    "response": "I followed the Colliers standard workflow.",
                },
            ],
        }

        cm.all_tool_calls.clear()

        await cm.step_until_wait(
            InboundUnifyMeetUtterance(
                contact=BOSS,
                content=(
                    "Hey, you mentioned you were pulling together "
                    "the ASH process breakdown. What did you find?"
                ),
            ),
            max_steps=5,
        )

        assert "guide_voice_agent" not in cm.all_tool_calls, (
            f"guide_voice_agent should be stripped for boss-on-call. "
            f"The fast brain receives ask answers via event forwarding.\n"
            f"Tool calls: {cm.all_tool_calls}"
        )


# =============================================================================
# Test: slow brain should not send text messages during voice calls
# =============================================================================


@pytest.mark.eval
@pytest.mark.asyncio
class TestSlowBrainSuppressesTextDuringVoiceCall:
    """During a Unify Meet with the boss, the fast brain receives all system
    events directly and handles verbal communication autonomously.  The slow
    brain should NOT silently send text messages (Unify messages, SMS) to relay
    results that the fast brain is already handling — this manifests as the
    caller receiving an unexpected text notification during a live voice
    conversation with no verbal acknowledgement.
    """

    @_handle_project
    async def test_slow_brain_waits_after_action_completes_during_meet(
        self,
        initialized_cm,
    ):
        """When an action completes during a boss Unify Meet, the slow brain
        should call wait() — not send_unify_message — because the fast brain
        already receives the ActorResult via the notification pipeline and
        will relay results verbally.

        Regression: in production the slow brain consistently sent detailed
        Unify messages with action results during live voice calls, even
        though the fast brain was simultaneously verbalising the same
        information.  The caller would get a silent text notification with
        no verbal indication that anything was sent.

        The production scenario had an active Unify message thread alongside
        the meet (the user had sent text messages before joining the call),
        which biased the LLM toward replying in that same text channel.
        The brain was woken by the ActorResult itself — not by a user
        utterance — and decided to "deliver" the results via text.
        """
        from unity.conversation_manager.events import (
            ActorResult,
            UnifyMessageReceived,
            UnifyMessageSent,
        )

        cm = initialized_cm

        # Establish a pre-existing text thread where the user explicitly
        # asked for structured information via text.
        await cm.step(
            UnifyMessageReceived(
                contact=BOSS,
                content=(
                    "Hey David, I've set up the OneDrive API credentials "
                    "in the secrets page. Can you connect and send me a "
                    "summary of what's in there? A breakdown with folder "
                    "names and file counts would be great."
                ),
            ),
        )
        await cm.step(
            UnifyMessageSent(
                contact=BOSS,
                content="On it, connecting now.",
            ),
        )

        # User joins a Unify Meet to discuss interactively.
        await cm.step(UnifyMeetReceived(contact=BOSS))
        await cm.step(UnifyMeetStarted(contact=BOSS))
        assert cm.cm.mode == Mode.MEET

        # Simulate an in-flight action that was dispatched during the call.
        cm.cm.in_flight_actions[0] = {
            "query": (
                "Connect to Microsoft OneDrive using the stored API "
                "credentials and list the contents of Dan's drive."
            ),
            "handle_actions": [],
        }

        cm.all_tool_calls.clear()

        # Action completes — the brain is woken by the ActorResult.
        # The fast brain also receives this via the notification pipeline.
        await cm.step_until_wait(
            ActorResult(
                handle_id=0,
                success=True,
                result=(
                    "Connected successfully. Dan's OneDrive root "
                    "contains: Projects/ (3 subfolders), Finance/ "
                    "(invoices, receipts), HR/ (contracts, policies), "
                    "plus 50 miscellaneous files at root level "
                    "including duplicates and temp files."
                ),
            ),
            max_steps=5,
        )

        text_tools = {"send_unify_message", "send_sms", "send_email"}
        used_text_tools = text_tools & set(cm.all_tool_calls)
        assert not used_text_tools, (
            f"During a Unify Meet with the boss, the slow brain must not "
            f"send text messages to relay action results — the fast brain "
            f"handles verbal communication autonomously.  An active text "
            f"thread from before the call does not justify switching to "
            f"text mid-conversation.\n"
            f"Text tools called: {used_text_tools}\n"
            f"All tool calls: {cm.all_tool_calls}"
        )
