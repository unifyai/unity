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
    ActorHandleStarted,
    ActorResult,
    PhoneCallStarted,
    PhoneCallSent,
    InboundPhoneUtterance,
    CallGuidance,
)
from unity.conversation_manager.types import Medium, Mode

from tests.conversation_manager.conftest import BOSS, TEST_CONTACTS

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
        2. make_call stores context on call_manager.initial_call_guidance
        3. comms_utils.start_call() places the call
        4. PhoneCallSent arrives → event handler spawns subprocess
        5. CallManager.start_call() publishes stored guidance as CallGuidance
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
        guidance = CallGuidance(
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
        guidance = CallGuidance(
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
            cm._run_llm = tracked_run_llm_with_simulated_delay  # Keep mock for teardown


# (TestGuidanceRelevanceGuardrails and TestStaleGuidanceArticulation removed —
# the GuidanceArticulator has been deleted.)


# =============================================================================
# Test: In-flight action orchestration during voice calls
# =============================================================================


def _get_guidance_messages(cm, contact_id: int) -> list:
    """Extract guidance messages from the voice thread for a contact."""
    voice_thread = cm.contact_index.get_messages_for_contact(
        contact_id,
        Medium.PHONE_CALL,
    )
    return [msg for msg in voice_thread if getattr(msg, "name", None) == "guidance"]


@pytest.mark.asyncio
class TestInFlightActionOrchestration:
    """
    Tests that the slow brain correctly orchestrates long-running actions
    during voice calls, relaying results to the fast brain via guide_voice_agent.

    This validates the full deterministic event-driven coordination loop:

        User utterance
            → slow brain calls `act` (starts action)
            → action completes (ActorResult arrives)
            → slow brain calls `guide_voice_agent` with the result
            → fast brain receives guidance and shares info naturally

    All steps are trigger-based via CMStepDriver — no sleeps or timing.
    The real slow brain LLM runs at each step.
    """

    @pytest.mark.eval
    async def test_act_result_relayed_via_call_guidance(self, initialized_cm):
        """
        Full coordination flow: user requests task → act → result → guidance.

        Phase 1: Enter voice call mode
        Phase 2: User asks for research → slow brain calls `act`
        Phase 3: Action completes → slow brain calls `guide_voice_agent` with result
        Phase 4: Verify fast brain can use the guidance to answer directly

        NOTE: With boss-on-call, the slow brain may correctly decide NOT to send
        guidance (the fast brain has direct event visibility). This is eval-dependent.
        """
        cm = initialized_cm
        boss = BOSS

        # ─── Phase 1: Enter voice call mode ───────────────────────────────
        await cm.step(PhoneCallStarted(contact=boss))
        assert cm.cm.mode == Mode.CALL, "Should be in CALL mode"

        # ─── Phase 2: User asks for research → slow brain calls `act` ─────
        cm.all_tool_calls.clear()
        guidance_before_request = _get_guidance_messages(cm.cm, boss["contact_id"])

        result = await cm.step_until_wait(
            InboundPhoneUtterance(
                contact=boss,
                content="Can you search for information about the Henderson project?",
            ),
            max_steps=5,
        )

        # Slow brain should have called `act` to start the research
        assert "act" in cm.all_tool_calls, (
            f"Slow brain should call `act` to start the research!\n"
            f"Tool calls: {cm.all_tool_calls}\n"
            f"The user asked to search for info — this requires `act`."
        )

        # An ActorHandleStarted event should have been published
        actor_started = [
            e for e in result.output_events if isinstance(e, ActorHandleStarted)
        ]
        assert len(actor_started) >= 1, (
            f"Expected ActorHandleStarted event but got: "
            f"{[type(e).__name__ for e in result.output_events]}"
        )

        # Get the handle_id from the event (resilient to the action having
        # already completed and moved to completed_actions).
        handle_id = actor_started[0].handle_id

        # ─── Phase 3: Action completes → slow brain relays via guidance ───
        # Inject ActorResult deterministically (no background task dependency).
        # Capture guidance count BEFORE stepping the result, so we can detect
        # new guidance produced in response to the completion.
        cm.all_tool_calls.clear()
        guidance_before_result = _get_guidance_messages(cm.cm, boss["contact_id"])

        result = await cm.step_until_wait(
            ActorResult(
                handle_id=handle_id,
                success=True,
                result=(
                    "Found 3 relevant documents about the Henderson project: "
                    "a contract signed in January 2025, meeting notes from last "
                    "week discussing timeline changes, and a budget proposal "
                    "totalling $85,000."
                ),
            ),
            max_steps=5,
        )

        # Collect ALL guidance produced across the entire flow
        all_guidance = _get_guidance_messages(cm.cm, boss["contact_id"])
        guidance_after_result = all_guidance[len(guidance_before_result) :]

        # The slow brain should have produced call_guidance at some point
        # during this flow — either eagerly in Phase 2 (acknowledging the
        # request) or in Phase 3 (relaying the action result), or both.
        # What matters is that the result information reaches the fast brain.
        all_guidance_text = " ".join(
            getattr(g, "content", "") for g in all_guidance
        ).lower()

        assert any(
            term in all_guidance_text
            for term in ["henderson", "contract", "budget", "85,000", "document"]
        ), (
            f"guide_voice_agent should contain Henderson project findings!\n"
            f"All guidance texts: {[getattr(g, 'content', '') for g in all_guidance]}\n"
            f"Tool calls across phases: {cm.all_tool_calls}\n"
            f"Guidance before request: {len(guidance_before_request)}\n"
            f"Guidance before result: {len(guidance_before_result)}\n"
            f"Guidance after result: {len(guidance_after_result)}\n"
            f"\n"
            f"The slow brain should relay action results to the fast brain\n"
            f"via guide_voice_agent so the user gets the information."
        )

        # ─── Phase 4: Verify fast brain can use the guidance ──────────────
        # Build fast brain prompt and inject the guidance as a notification
        from unity.common.llm_client import new_llm_client
        from unity.conversation_manager.prompt_builders import (
            build_voice_agent_prompt,
        )
        from unity.settings import SETTINGS

        fast_prompt = build_voice_agent_prompt(
            bio="I am a virtual assistant.",
            assistant_name="Alex",
            boss_first_name=boss["first_name"],
            boss_surname=boss["surname"],
            boss_phone_number=boss.get("phone_number"),
            boss_email_address=boss.get("email_address"),
            is_boss_user=True,
        ).flatten()
        client = new_llm_client(
            model=SETTINGS.conversation.FAST_BRAIN_MODEL,
            reasoning_effort="minimal",
        )
        # Use the actual guidance content that was produced, or a fallback
        result_guidance = [
            getattr(g, "content", "")
            for g in all_guidance
            if any(
                t in getattr(g, "content", "").lower()
                for t in ["henderson", "contract", "budget", "document"]
            )
        ]
        notification_content = (
            result_guidance[-1]
            if result_guidance
            else "Found 3 documents about Henderson: contract, meeting notes, budget."
        )
        messages = [
            {"role": "system", "content": fast_prompt},
            {
                "role": "user",
                "content": (
                    "Can you search for information about the Henderson project?"
                ),
            },
            {"role": "assistant", "content": "Sure, let me look into that."},
            {
                "role": "system",
                "content": f"[notification] {notification_content}",
            },
            {"role": "user", "content": "So what did you find?"},
        ]
        fast_response = await client.generate(messages=messages)
        fast_response_lower = fast_response.lower()

        # Fast brain should answer directly using the notification data
        deferral_phrases = ["let me check", "i'll check", "looking into"]
        deferred = any(p in fast_response_lower for p in deferral_phrases)
        assert not deferred, (
            f"Fast brain deferred instead of using the notification data!\n"
            f"Response: {fast_response}\n"
            f"The notification contained the Henderson project results — "
            f"the fast brain should share them directly."
        )

        # Fast brain should reference something from the result
        assert any(
            term in fast_response_lower
            for term in [
                "henderson",
                "contract",
                "budget",
                "document",
                "meeting",
                "85,000",
                "three",
                "3",
            ]
        ), (
            f"Fast brain should reference the action results!\n"
            f"Response: {fast_response}\n"
            f"Expected mention of Henderson project findings from the notification."
        )


# =============================================================================
# Test: guide_voice_agent SPEAK mode — should_speak + response_text pipeline
# =============================================================================


@pytest.mark.asyncio
class TestCallGuidanceSpeakMode:
    """Verify that guide_voice_agent's should_speak and response_text
    parameters propagate correctly through the CallGuidance event to the
    fast brain."""

    async def test_should_speak_params_carried_through_to_event(
        self,
        initialized_cm,
    ):
        """When the slow brain calls guide_voice_agent with should_speak=True
        and response_text, the published CallGuidance event must carry both
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
            if channel == "app:call:call_guidance":
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
                ), f"CallGuidance event missing should_speak field: {payload}"
                assert (
                    "response_text" in payload
                ), f"CallGuidance event missing response_text field: {payload}"

        finally:
            cm.event_broker.publish = original_publish
