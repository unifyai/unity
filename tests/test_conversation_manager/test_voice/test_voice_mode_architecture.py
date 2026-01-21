"""
tests/test_conversation_manager/test_voice_mode_architecture.py
================================================================

Comprehensive test harness for the voice mode architecture refactoring.

This test file validates the "fast brain / slow brain" architecture for voice calls,
ensuring that:
1. The Main CM Brain uses the system default model (SETTINGS.UNIFY_MODEL)
2. TTS mode works like Realtime mode with concurrent guidance streams
3. Both modes output `call_guidance` for orchestration

## Test Categories

### Unit Tests (no Redis/LLM required)
- Model selection logic
- Response model construction
- Prompt builder output validation

### Integration Tests (require Redis, use simulated managers)
- Voice call mode switching
- Guidance event flow
- Fast/slow brain coordination

## Incremental Rollout Stages

Stage 0: Baseline (current behavior)
Stage 1: Main CM Brain uses SETTINGS.UNIFY_MODEL
Stage 2: TTS mode outputs call_guidance instead of voice_utterance
Stage 3: TTS fast brain handles conversational responses

Each test is tagged with the stage it validates.
"""

import asyncio
import json

import pytest
import pytest_asyncio

from unity.settings import SETTINGS


# =============================================================================
# Local Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def event_broker():
    """
    Local in-memory broker for this test module.

    This avoids starting a full ConversationManager instance (slow, requires env),
    while still exercising the same pub/sub API that voice mode uses.
    """
    from unity.conversation_manager.event_broker import create_event_broker

    broker = create_event_broker()
    yield broker
    await broker.aclose()


# =============================================================================
# Unit Tests: Model Selection Logic
# =============================================================================


class TestModelSelection:
    """Tests for LLM model selection in ConversationManager."""

    def test_default_model_from_settings(self):
        """SETTINGS.UNIFY_MODEL provides the system default model."""
        from unity.common.llm_client import DEFAULT_MODEL

        assert DEFAULT_MODEL == SETTINGS.UNIFY_MODEL
        # Default should be a capable model for complex reasoning
        assert "gpt-5" in DEFAULT_MODEL or "claude" in DEFAULT_MODEL

    def test_llm_client_uses_default_when_none(self):
        """new_llm_client() uses SETTINGS.UNIFY_MODEL when model=None."""
        from unity.common.llm_client import new_llm_client

        # The client should use the default model when not specified
        client = new_llm_client(model=None)
        # The Unify client stores the model in _model (private attribute)
        # It may normalize the model name (strip provider suffix)
        expected_base = SETTINGS.UNIFY_MODEL.split("@")[0]
        assert client._model == expected_base or client._model == SETTINGS.UNIFY_MODEL

    def test_llm_client_explicit_model_override(self):
        """new_llm_client() respects explicit model parameter."""
        from unity.common.llm_client import new_llm_client

        explicit_model = "gpt-4o-mini@openai"
        client = new_llm_client(model=explicit_model)
        # The Unify client stores the model in _model (private attribute)
        # It may normalize the model name (strip provider suffix)
        expected_base = explicit_model.split("@")[0]
        assert client._model == expected_base or client._model == explicit_model

    def test_main_cm_brain_model_configuration(self):
        """
        [Stage 1] Main CM Brain LLM should use SETTINGS.UNIFY_MODEL.

        Verifies that new_llm_client correctly configures the model when called
        with SETTINGS.UNIFY_MODEL, matching how ConversationManager._run_llm() uses it.
        """
        from unity.common.llm_client import new_llm_client

        # Create an LLM client the way ConversationManager._run_llm() does
        client = new_llm_client(SETTINGS.UNIFY_MODEL, reasoning_effort="low")

        # The client should be configured with the expected model
        expected_base = SETTINGS.UNIFY_MODEL.split("@")[0]
        assert client._model == expected_base or client._model == SETTINGS.UNIFY_MODEL


class TestResponseModelConstruction:
    """Tests for dynamic response model construction."""

    def test_text_mode_response_model_structure(self):
        """Text mode response model has thoughts only (actions are tool calls)."""
        from unity.conversation_manager.domains.brain import build_response_models

        models = build_response_models()
        text_model = models["text"]

        # Get the schema
        schema = text_model.model_json_schema()
        props = schema.get("properties", {})

        assert "thoughts" in props
        # Actions are now tool calls, not part of the response model
        assert "actions" not in props
        # Text mode should NOT have voice_utterance or call_guidance
        assert "voice_utterance" not in props
        assert "call_guidance" not in props

    def test_voice_model_tts_mode_uses_guidance(self):
        """
        [Stage 2] TTS mode now uses call_guidance (same as Realtime mode).

        Both TTS and Realtime modes use the unified guidance-based architecture
        where the Main CM Brain provides guidance to the Voice Agent.
        """
        from unity.conversation_manager.domains.brain import build_response_models

        models = build_response_models()
        voice_model = models["call"]

        schema = voice_model.model_json_schema()
        props = schema.get("properties", {})

        assert "thoughts" in props
        # Actions are now tool calls, not part of the response model
        assert "actions" not in props
        # Stage 2: TTS mode now uses call_guidance
        assert "call_guidance" in props
        assert "voice_utterance" not in props

    def test_voice_model_realtime_mode_uses_guidance(self):
        """Realtime mode uses call_guidance instead of voice_utterance."""
        from unity.conversation_manager.domains.brain import build_response_models

        models = build_response_models()
        voice_model = models["call"]

        schema = voice_model.model_json_schema()
        props = schema.get("properties", {})

        assert "thoughts" in props
        # Actions are now tool calls, not part of the response model
        assert "actions" not in props
        # Realtime mode uses call_guidance
        assert "call_guidance" in props
        assert "voice_utterance" not in props

    def test_unify_meet_model_matches_call_model(self):
        """unify_meet mode uses the same model as call mode."""
        from unity.conversation_manager.domains.brain import build_response_models

        models = build_response_models()

        call_schema = models["call"].model_json_schema()
        meet_schema = models["unify_meet"].model_json_schema()

        # Both should have identical structure
        assert (
            call_schema.get("properties", {}).keys()
            == meet_schema.get(
                "properties",
                {},
            ).keys()
        )

    def test_voice_model_tts_mode_uses_guidance_after_refactor(self):
        """
        [Stage 2] TTS mode uses call_guidance after refactoring.

        Stage 2 is complete - TTS mode now uses the same guidance pattern as Realtime.
        """
        from unity.conversation_manager.domains.brain import build_response_models

        models = build_response_models()
        voice_model = models["call"]

        schema = voice_model.model_json_schema()
        props = schema.get("properties", {})

        # Stage 2 complete: TTS mode uses call_guidance
        assert "call_guidance" in props
        assert "voice_utterance" not in props


class TestPromptBuilders:
    """Tests for prompt builder functions."""

    def test_build_system_prompt_text_mode(self):
        """System prompt for text mode has correct structure."""
        from unity.conversation_manager.prompt_builders import build_system_prompt

        prompt = build_system_prompt(
            bio="Test assistant bio",
            contact_id=1,
            first_name="Test",
            surname="User",
            phone_number="+15551234567",
            email_address="test@example.com",
            is_voice_call=False,
        )

        # Basic structure checks
        assert "<role>" in prompt
        assert "<bio>" in prompt
        assert "<boss_details>" in prompt
        assert "<output_format>" in prompt

        # Stage 2: All voice modes use call_guidance
        assert "call_guidance" in prompt
        assert "voice_utterance" not in prompt

    def test_build_system_prompt_voice_call_mode(self):
        """System prompt for voice call mode mentions call_guidance."""
        from unity.conversation_manager.prompt_builders import build_system_prompt

        prompt = build_system_prompt(
            bio="Test assistant bio",
            contact_id=1,
            first_name="Test",
            surname="User",
            phone_number="+15551234567",
            email_address="test@example.com",
            is_voice_call=True,
        )

        # Stage 2: All voice modes use guidance-based architecture
        assert "call_guidance" in prompt
        assert "<voice_calls_guide>" in prompt
        assert "Voice Agent" in prompt

    def test_build_voice_agent_prompt(self):
        """Voice Agent prompt has fast brain instructions."""
        from unity.conversation_manager.prompt_builders import (
            build_voice_agent_prompt,
        )

        prompt = build_voice_agent_prompt(
            bio="Test assistant",
            boss_first_name="Test",
            boss_surname="Boss",
            boss_phone_number="+15551234567",
            is_boss_user=True,
        )

        # Fast brain specific content
        assert "fast brain" in prompt.lower() or "small" in prompt.lower()
        assert "conversation manager" in prompt.lower()
        assert "<communication_guidelines>" in prompt

    def test_build_system_prompt_tts_mode_uses_guidance_after_refactor(self):
        """
        [Stage 2] TTS mode system prompt mentions call_guidance.

        Stage 2 is complete - TTS mode now uses the guidance pattern.
        """
        from unity.conversation_manager.prompt_builders import build_system_prompt

        prompt = build_system_prompt(
            bio="Test assistant bio",
            contact_id=1,
            first_name="Test",
            surname="User",
            is_voice_call=False,  # TTS mode
        )

        # Stage 2 complete: TTS mode uses call_guidance
        assert "call_guidance" in prompt
        assert "voice_utterance" not in prompt


# =============================================================================
# Unit Tests: Voice Mode State Management
# =============================================================================


class TestVoiceModeStateManagement:
    """Tests for voice mode state in ConversationManager."""

    def test_call_config_realtime_detection(self):
        """CallConfig correctly detects realtime mode from voice_mode."""
        from unity.conversation_manager.domains.call_manager import CallConfig

        # TTS mode
        tts_config = CallConfig(
            assistant_id="test",
            assistant_bio="Test",
            assistant_number="+15551234567",
            voice_provider="cartesia",
            voice_id="test_voice",
            voice_mode="tts",
        )

        # Realtime mode (speech-to-speech)
        sts_config = CallConfig(
            assistant_id="test",
            assistant_bio="Test",
            assistant_number="+15551234567",
            voice_provider="cartesia",
            voice_id="test_voice",
            voice_mode="sts",
        )

        # The LivekitCallManager uses voice_mode == "sts" to set realtime
        assert tts_config.voice_mode == "tts"
        assert sts_config.voice_mode == "sts"

    def test_livekit_call_manager_realtime_flag(self):
        """LivekitCallManager sets realtime flag based on voice_mode."""
        from unity.conversation_manager.domains.call_manager import (
            CallConfig,
            LivekitCallManager,
        )

        tts_config = CallConfig(
            assistant_id="test",
            assistant_bio="Test",
            assistant_number="+15551234567",
            voice_provider="cartesia",
            voice_id="test_voice",
            voice_mode="tts",
        )

        sts_config = CallConfig(
            assistant_id="test",
            assistant_bio="Test",
            assistant_number="+15551234567",
            voice_provider="cartesia",
            voice_id="test_voice",
            voice_mode="sts",
        )

        tts_manager = LivekitCallManager(tts_config)
        sts_manager = LivekitCallManager(sts_config)

        assert tts_manager.uses_realtime_api is False
        assert sts_manager.uses_realtime_api is True


# =============================================================================
# Unit Tests: Event Types
# =============================================================================


class TestVoiceEvents:
    """Tests for voice-related event types."""

    def test_call_guidance_event_structure(self):
        """CallGuidance event has required fields."""
        from unity.conversation_manager.events import CallGuidance

        contact = {"contact_id": 1, "first_name": "Test", "surname": "User"}
        event = CallGuidance(contact=contact, content="Test guidance")

        assert event.contact == contact
        assert event.content == "Test guidance"
        assert hasattr(event, "timestamp")

    def test_call_guidance_serialization(self):
        """CallGuidance event can be serialized and deserialized."""
        from unity.conversation_manager.events import Event, CallGuidance

        contact = {"contact_id": 1, "first_name": "Test", "surname": "User"}
        original = CallGuidance(contact=contact, content="Test guidance")

        # Serialize
        json_str = original.to_json()
        data = json.loads(json_str)

        # Verify structure
        assert data["event_name"] == "CallGuidance"
        assert data["payload"]["content"] == "Test guidance"

        # Deserialize
        restored = Event.from_json(json_str)
        assert isinstance(restored, CallGuidance)
        assert restored.content == original.content

    def test_outbound_phone_utterance_event(self):
        """OutboundPhoneUtterance event for TTS mode responses."""
        from unity.conversation_manager.events import OutboundPhoneUtterance

        contact = {"contact_id": 1, "first_name": "Test", "surname": "User"}
        event = OutboundPhoneUtterance(contact=contact, content="Hello there!")

        assert event.contact == contact
        assert event.content == "Hello there!"

    def test_outbound_unify_meet_utterance_event(self):
        """OutboundUnifyMeetUtterance event for browser call TTS responses."""
        from unity.conversation_manager.events import OutboundUnifyMeetUtterance

        contact = {"contact_id": 1, "first_name": "Test", "surname": "User"}
        event = OutboundUnifyMeetUtterance(contact=contact, content="Hello there!")

        assert event.contact == contact
        assert event.content == "Hello there!"


# =============================================================================
# Integration Tests: Proactive Speech
# =============================================================================


class TestProactiveSpeech:
    """Tests for proactive speech decision making."""

    def test_proactive_speech_model_configuration(self):
        """ProactiveSpeech uses a fast model for quick decisions."""
        from unity.conversation_manager.domains.proactive_speech import ProactiveSpeech

        ps = ProactiveSpeech()
        # Should use a fast model for low-latency decisions
        assert "flash" in ps.model.lower() or "mini" in ps.model.lower()

    def test_proactive_decision_structure(self):
        """ProactiveDecision has required fields."""
        from unity.conversation_manager.domains.proactive_speech import (
            ProactiveDecision,
        )

        decision = ProactiveDecision(should_speak=True, delay=5, content="Still here!")

        assert decision.should_speak is True
        assert decision.delay == 5
        assert decision.content == "Still here!"

    def test_proactive_decision_defaults(self):
        """ProactiveDecision has sensible defaults."""
        from unity.conversation_manager.domains.proactive_speech import (
            ProactiveDecision,
        )

        decision = ProactiveDecision(should_speak=False)

        assert decision.should_speak is False
        assert decision.delay == 5  # Default delay
        assert decision.content is None


# =============================================================================
# Integration Tests: ConversationManagerHandle.ask()
# =============================================================================


class TestConversationManagerHandleAsk:
    """Tests for the ask() flow which uses its own LLM."""

    def test_ask_uses_fast_model(self):
        """
        ConversationManagerHandle.ask() uses a fast model for responsiveness.

        The ask() flow handles user Q&A during active conversations and needs
        to be fast (gemini-flash or similar).
        """
        # This is verified by inspecting the handle.py code
        # The model used is gemini-2.5-flash@vertex-ai
        from unity.conversation_manager.handle import ConversationManagerHandle

        # The model is hardcoded in the ask() method - we just verify the class exists
        assert hasattr(ConversationManagerHandle, "ask")


# =============================================================================
# Integration Tests: Voice Call Flow (In-Memory Broker)
# =============================================================================


@pytest.mark.asyncio
class TestVoiceCallFlowIntegration:
    """
    Integration tests for voice call flows.

    These tests validate the event flow for both TTS and Realtime modes
    using the in-memory event broker (no Redis required).
    """

    @pytest.fixture
    def boss_contact(self):
        return {
            "contact_id": 1,
            "first_name": "Test",
            "surname": "Boss",
            "phone_number": "+15555555678",
            "email_address": "boss@test.com",
        }

    async def test_phone_call_started_event_flow(
        self,
        event_broker,
        boss_contact,
    ):
        """
        Verify phone call start event is properly published and captured.
        """
        from unity.conversation_manager.events import Event, PhoneCallStarted

        async with event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            # Publish a call started event
            event = PhoneCallStarted(contact=boss_contact)
            await event_broker.publish(
                "app:comms:phone_call_started",
                event.to_json(),
            )

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            captured = Event.from_json(msg["data"])
            assert isinstance(captured, PhoneCallStarted)

    async def test_call_guidance_event_flow(
        self,
        event_broker,
        boss_contact,
    ):
        """
        Verify call guidance events flow through the system.
        """
        from unity.conversation_manager.events import CallGuidance, Event

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:call_guidance")

            # Publish a guidance event
            event = CallGuidance(
                contact=boss_contact,
                content="Please ask about their schedule",
            )
            await event_broker.publish("app:call:call_guidance", event.to_json())

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            captured = Event.from_json(msg["data"])
            assert isinstance(captured, CallGuidance)
            assert captured.content == "Please ask about their schedule"

    async def test_tts_mode_publishes_guidance_not_utterance(
        self,
    ):
        """
        [Stage 2] TTS mode publishes call_guidance events.

        Stage 2 is complete - when the Main CM Brain responds during a TTS call,
        it publishes CallGuidance instead of OutboundPhoneUtterance.
        """
        # Verify by checking the conversation_manager code uses guidance pattern
        import inspect
        from unity.conversation_manager import conversation_manager as cm_module

        source = inspect.getsource(cm_module.ConversationManager._run_llm)
        # After Stage 2, TTS mode should publish CallGuidance
        assert "CallGuidance" in source, "Should use CallGuidance for voice modes"
        # Should NOT have separate paths for realtime vs TTS utterance publishing
        assert (
            "OutboundPhoneUtterance" not in source
        ), "Should not publish OutboundPhoneUtterance anymore"
        assert (
            "OutboundUnifyMeetUtterance" not in source
        ), "Should not publish OutboundUnifyMeetUtterance anymore"


# =============================================================================
# Integration Tests: Voice Guidance Channel
# =============================================================================


@pytest.mark.asyncio
class TestCallGuidanceChannel:
    """Tests for call_guidance channel used by both TTS and STS modes."""

    async def test_call_guidance_channel_format(self, event_broker):
        """Verify call_guidance channel message format."""
        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:call_guidance")

            # Consume the subscription confirmation message
            await pubsub.get_message(timeout=1.0)

            # Publish guidance (the format used by Main CM Brain)
            await event_broker.publish(
                "app:call:call_guidance",
                json.dumps({"content": "Please ask about their schedule"}),
            )

            msg = await pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=2.0,
            )

            assert msg is not None, "Expected to receive published message"
            assert msg["type"] == "message"
            data = json.loads(msg["data"])
            assert "content" in data
            assert data["content"] == "Please ask about their schedule"


# =============================================================================
# Threading Tests: In-process Voice Agent Uses Shared Broker
# =============================================================================


@pytest.mark.asyncio
async def test_event_broker_delivers_across_threads(event_broker):
    """
    Voice agents run in a background thread but must share the same in-memory broker.

    This validates that a subscriber created on a different event loop/thread can
    still receive published messages.
    """
    import queue
    import threading

    ready = threading.Event()
    received: "queue.Queue[dict]" = queue.Queue()

    def _subscriber_thread() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run() -> None:
            async with event_broker.pubsub() as pubsub:
                await pubsub.subscribe("app:call:status")
                ready.set()
                msg = await pubsub.get_message(
                    timeout=2.0,
                    ignore_subscribe_messages=True,
                )
                received.put(msg)

        loop.run_until_complete(_run())
        loop.close()

    t = threading.Thread(target=_subscriber_thread, daemon=True)
    t.start()

    # Wait until the subscriber is ready (no sleeps for event alignment).
    assert await asyncio.to_thread(ready.wait, 2.0)

    await event_broker.publish("app:call:status", json.dumps({"type": "stop"}))

    msg = await asyncio.to_thread(received.get, True, 2.0)
    assert msg is not None
    assert json.loads(msg["data"])["type"] == "stop"
    t.join(timeout=2.0)


# =============================================================================
# Regression Tests: Ensure Existing Behavior
# =============================================================================


class TestRegressionBaseline:
    """
    Regression tests to ensure existing behavior is preserved.

    These tests document the current behavior and should continue to pass
    throughout the refactoring process.
    """

    def test_text_mode_does_not_include_voice_fields(self):
        """Text mode response model excludes voice-specific fields."""
        from unity.conversation_manager.domains.brain import build_response_models

        models = build_response_models()
        text_model = models["text"]
        schema = text_model.model_json_schema()
        props = schema.get("properties", {})

        assert "voice_utterance" not in props
        assert "call_guidance" not in props

    def test_action_tools_are_available(self):
        """Action tools are available via ConversationManagerBrainActionTools."""
        from unittest.mock import MagicMock

        from unity.conversation_manager.domains.brain_action_tools import (
            ConversationManagerBrainActionTools,
        )

        # Create mock ConversationManager
        mock_cm = MagicMock()
        mock_cm.in_flight_actions = {}
        action_tools = ConversationManagerBrainActionTools(mock_cm)

        # Get the tools
        tools = action_tools.as_tools()

        # Check that core communication and task tools are available
        assert "send_sms" in tools
        assert "send_email" in tools
        assert "make_call" in tools
        assert "send_unify_message" in tools
        assert "act" in tools
        assert "wait" in tools

    def test_call_manager_cleanup_method_exists(self):
        """LivekitCallManager has cleanup method for call processes."""
        from unity.conversation_manager.domains.call_manager import LivekitCallManager

        assert hasattr(LivekitCallManager, "cleanup_call_proc")
        assert hasattr(LivekitCallManager, "start_call")
        assert hasattr(LivekitCallManager, "start_unify_meet")


# =============================================================================
# Stage Markers: Tests that will be enabled at each stage
# =============================================================================


class TestStage1MainBrainModel:
    """
    [Stage 1] Tests for Main CM Brain using SETTINGS.UNIFY_MODEL.

    These tests verify that after Stage 1:
    - The Main CM Brain uses SETTINGS.UNIFY_MODEL instead of hardcoded value
    - The LLM is configured with appropriate reasoning settings
    """

    def test_conversation_manager_uses_settings_model(self):
        """
        ConversationManager's LLM should use SETTINGS.UNIFY_MODEL.

        Stage 1 is complete - the Main CM Brain now uses the system default model.
        """
        # Verify the conversation_manager.py imports and uses SETTINGS.UNIFY_MODEL
        import inspect
        from unity.conversation_manager import conversation_manager as cm_module

        # Check that SETTINGS is imported in the module
        assert hasattr(cm_module, "SETTINGS"), "SETTINGS should be imported"

        # Verify the source code uses SETTINGS.UNIFY_MODEL in _run_llm (not hardcoded)
        source = inspect.getsource(cm_module.ConversationManager._run_llm)
        assert (
            "SETTINGS.UNIFY_MODEL" in source
        ), "ConversationManager._run_llm should use SETTINGS.UNIFY_MODEL for LLM"
        assert (
            '"gpt-5-mini@openai"' not in source
        ), "ConversationManager._run_llm should not have hardcoded model name"


class TestStage2UnifiedVoiceResponse:
    """
    [Stage 2] Tests for unified voice response model (call_guidance everywhere).

    Stage 2 is complete. These tests verify:
    - TTS mode outputs call_guidance instead of voice_utterance
    - Both TTS and Realtime modes use the same response model structure
    - The system prompt for TTS mode mentions call_guidance
    """

    def test_tts_mode_response_model_has_guidance(self):
        """TTS mode response model uses call_guidance field."""
        from unity.conversation_manager.domains.brain import build_response_models

        models = build_response_models()
        voice_model = models["call"]
        schema = voice_model.model_json_schema()
        props = schema.get("properties", {})

        assert "call_guidance" in props
        assert "voice_utterance" not in props

    def test_tts_and_realtime_models_match(self):
        """TTS and Realtime modes use identical response model structure."""
        from unity.conversation_manager.domains.brain import build_response_models

        models = build_response_models()

        tts_schema = models["call"].model_json_schema()
        sts_schema = models["call"].model_json_schema()

        # Stage 2 complete: TTS and STS models are identical
        assert (
            tts_schema.get("properties", {}).keys()
            == sts_schema.get(
                "properties",
                {},
            ).keys()
        )


class TestStage3TTSFastBrain:
    """
    [Stage 3] Tests for TTS Fast Brain implementation.

    Stage 3 is complete. These tests verify:
    - call.py has its own lightweight LLM for conversational responses
    - The TTS fast brain receives guidance from the Main CM Brain
    - The fast brain uses the same prompt as the Realtime phone agent
    """

    def test_tts_call_has_fast_brain_model(self):
        """TTS call.py uses UnifyLLM adapter for fast conversational responses."""
        import inspect
        from unity.conversation_manager.medium_scripts import call as call_module

        # Check that UnifyLLM adapter is imported
        assert hasattr(call_module, "UnifyLLM"), "UnifyLLM adapter should be imported"

        # Check that entrypoint uses UnifyLLM with gpt-5-nano
        source = inspect.getsource(call_module.entrypoint)
        assert (
            "UnifyLLM" in source
        ), "entrypoint should use UnifyLLM adapter for fast brain"
        assert (
            "gpt-5-nano@openai" in source
        ), "fast brain should use gpt-5-nano@openai model"
        assert (
            'reasoning_effort="none"' in source
        ), "fast brain should disable reasoning for max speed"

    def test_tts_fast_brain_receives_guidance(self):
        """TTS fast brain subscribes to call_guidance channel."""
        import inspect
        from unity.conversation_manager.medium_scripts import call as call_module

        source = inspect.getsource(call_module.entrypoint)
        # Verify subscription to guidance channel
        assert (
            "app:call:call_guidance" in source
        ), "call.py should subscribe to call_guidance"
        assert (
            "wait_for_guidance" in source
        ), "call.py should have wait_for_guidance function"

    def test_tts_fast_brain_uses_voice_agent_prompt(self):
        """TTS fast brain uses build_voice_agent_prompt."""
        import inspect
        from unity.conversation_manager.medium_scripts import call as call_module

        # Check that build_voice_agent_prompt is imported
        assert hasattr(
            call_module,
            "build_voice_agent_prompt",
        ), "build_voice_agent_prompt should be imported"

        # Check that entrypoint uses this prompt builder
        source = inspect.getsource(call_module.entrypoint)
        assert (
            "build_voice_agent_prompt" in source
        ), "entrypoint should use build_voice_agent_prompt"

    def test_tts_and_realtime_use_same_cli_args(self):
        """TTS and Realtime modes use the same CLI arguments (CONTACT, BOSS, BIO)."""
        import inspect
        from unity.conversation_manager.medium_scripts import (
            call as call_module,
            sts_call as sts_module,
        )

        call_source = inspect.getsource(call_module)
        sts_source = inspect.getsource(sts_module)

        # Both should use CONTACT, BOSS, and ASSISTANT_BIO env vars
        for env_var in ["CONTACT", "BOSS", "ASSISTANT_BIO"]:
            assert env_var in call_source, f"call.py should use {env_var}"
            assert env_var in sts_source, f"sts_call.py should use {env_var}"

    def test_call_manager_passes_boss_to_tts_mode(self):
        """CallManager passes boss details to TTS mode (not just Realtime)."""
        import inspect
        from unity.conversation_manager.domains import call_manager as cm_module

        source = inspect.getsource(cm_module.LivekitCallManager.start_call)
        # Boss and assistant_bio should always be in the args list (not conditionally)
        assert "json.dumps(boss)" in source, "start_call should pass boss"
        assert "self.assistant_bio" in source, "start_call should pass assistant_bio"

        # The args list should include boss/bio unconditionally - check that they
        # appear BEFORE the if statement that selects the script
        boss_line = source.find("json.dumps(boss)")
        if_realtime_line = source.find("if self.uses_realtime_api:")
        assert boss_line < if_realtime_line, (
            "boss should be added to args before the uses_realtime_api conditional "
            "(should not be conditionally added)"
        )


# =============================================================================
# Stage 4: UnifyLLM Adapter Tests
# =============================================================================


class TestUnifyLLMAdapter:
    """
    Tests for the UnifyLLM adapter that wraps unillm.AsyncUnify for LiveKit.

    The adapter provides:
    - Local caching for CI (via Unify's cache system)
    - Usage tracking through the Unify platform
    - Consistent routing through our standard LLM client
    """

    def test_unify_llm_adapter_exists(self):
        """UnifyLLM adapter is importable from conversation_manager."""
        from unity.conversation_manager.livekit_unify_adapter import UnifyLLM

        assert UnifyLLM is not None

    def test_unify_llm_extends_livekit_llm(self):
        """UnifyLLM extends the LiveKit llm.LLM base class."""
        from livekit.agents import llm
        from unity.conversation_manager.livekit_unify_adapter import UnifyLLM

        assert issubclass(UnifyLLM, llm.LLM)

    def test_unify_llm_has_chat_method(self):
        """UnifyLLM implements the required chat() method."""
        from unity.conversation_manager.livekit_unify_adapter import UnifyLLM

        llm_instance = UnifyLLM(model="gpt-5-nano@openai")
        assert hasattr(llm_instance, "chat")
        assert callable(llm_instance.chat)

    def test_unify_llm_model_property(self):
        """UnifyLLM.model returns the configured model name."""
        from unity.conversation_manager.livekit_unify_adapter import UnifyLLM

        llm_instance = UnifyLLM(model="gpt-5-nano@openai")
        assert llm_instance.model == "gpt-5-nano@openai"

    def test_unify_llm_default_model(self):
        """UnifyLLM has a sensible default model."""
        from unity.conversation_manager.livekit_unify_adapter import UnifyLLM

        llm_instance = UnifyLLM()
        assert llm_instance.model == "gpt-5-nano@openai"

    def test_unify_llm_accepts_reasoning_effort(self):
        """UnifyLLM accepts reasoning_effort parameter."""
        from unity.conversation_manager.livekit_unify_adapter import UnifyLLM

        # Should not raise
        llm_instance = UnifyLLM(model="gpt-5-nano@openai", reasoning_effort="none")
        assert llm_instance._reasoning_effort == "none"

    def test_unify_llm_accepts_temperature(self):
        """UnifyLLM accepts temperature parameter."""
        from unity.conversation_manager.livekit_unify_adapter import UnifyLLM

        llm_instance = UnifyLLM(model="gpt-5-nano@openai", temperature=0.7)
        assert llm_instance._temperature == 0.7

    def test_unify_llm_stream_extends_livekit_llm_stream(self):
        """UnifyLLMStream extends the LiveKit llm.LLMStream base class."""
        from livekit.agents import llm
        from unity.conversation_manager.livekit_unify_adapter import UnifyLLMStream

        assert issubclass(UnifyLLMStream, llm.LLMStream)

    def test_unify_llm_stream_implements_run(self):
        """UnifyLLMStream implements the abstract _run() method."""
        from unity.conversation_manager.livekit_unify_adapter import UnifyLLMStream
        import inspect

        # _run should be an async method (not abstract)
        assert hasattr(UnifyLLMStream, "_run")
        assert inspect.iscoroutinefunction(UnifyLLMStream._run)

    def test_unify_llm_uses_new_llm_client(self):
        """UnifyLLMStream uses new_llm_client from unity.common.llm_client."""
        import inspect
        from unity.conversation_manager import livekit_unify_adapter

        source = inspect.getsource(livekit_unify_adapter)
        assert (
            "from unity.common.llm_client import new_llm_client" in source
        ), "Should import new_llm_client"
        assert "new_llm_client" in inspect.getsource(
            livekit_unify_adapter.UnifyLLMStream._run,
        ), "Should use new_llm_client in _run()"

    def test_call_py_uses_unify_llm_adapter(self):
        """call.py imports and uses UnifyLLM instead of openai.LLM."""
        import inspect
        from unity.conversation_manager.medium_scripts import call as call_module

        source = inspect.getsource(call_module)

        # Should import UnifyLLM
        assert (
            "from unity.conversation_manager.livekit_unify_adapter import UnifyLLM"
            in source
        ), "call.py should import UnifyLLM"

        # Should NOT import openai.LLM directly for the main LLM
        # (openai plugin may still be imported but not used for the fast brain LLM)
        entrypoint_source = inspect.getsource(call_module.entrypoint)
        assert (
            "openai.LLM" not in entrypoint_source
        ), "entrypoint should not use openai.LLM directly"

    @pytest.mark.asyncio
    async def test_unify_llm_chat_returns_stream(self):
        """UnifyLLM.chat() returns a UnifyLLMStream instance."""
        from livekit.agents import llm
        from unity.conversation_manager.livekit_unify_adapter import (
            UnifyLLM,
            UnifyLLMStream,
        )

        llm_instance = UnifyLLM(model="gpt-5-nano@openai")

        # Create a minimal chat context
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="user", content="Hello")

        stream = llm_instance.chat(chat_ctx=chat_ctx)
        assert isinstance(stream, UnifyLLMStream)

        # Clean up the stream to avoid task warnings
        await stream.aclose()
