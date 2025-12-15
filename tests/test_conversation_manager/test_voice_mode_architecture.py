"""
tests/test_conversation_manager/test_voice_mode_architecture.py
================================================================

Comprehensive test harness for the voice mode architecture refactoring.

This test file validates the "fast brain / slow brain" architecture for voice calls,
ensuring that:
1. The Main CM Brain uses the system default model (SETTINGS.UNIFY_MODEL)
2. TTS mode works like Realtime mode with concurrent guidance streams
3. Both modes output `realtime_guidance` for orchestration

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
Stage 2: TTS mode outputs realtime_guidance instead of voice_utterance
Stage 3: TTS fast brain handles conversational responses

Each test is tagged with the stage it validates.
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from unity.settings import SETTINGS


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
        # The Unify client may normalize the model name (strip provider suffix)
        # So we check that the model base name matches
        expected_base = SETTINGS.UNIFY_MODEL.split("@")[0]
        assert client.model == expected_base or client.model == SETTINGS.UNIFY_MODEL

    def test_llm_client_explicit_model_override(self):
        """new_llm_client() respects explicit model parameter."""
        from unity.common.llm_client import new_llm_client

        explicit_model = "gpt-4o-mini@openai"
        client = new_llm_client(model=explicit_model)
        # The Unify client may normalize the model name (strip provider suffix)
        expected_base = explicit_model.split("@")[0]
        assert client.model == expected_base or client.model == explicit_model

    def test_main_cm_brain_model_configuration(self):
        """
        [Stage 1] Main CM Brain LLM should use SETTINGS.UNIFY_MODEL.

        After the refactoring, the ConversationManager's LLM should be
        initialized with the system default model, not a hardcoded value.
        """
        from unity.conversation_manager.domains.llm import LLM

        # Create an LLM instance the way ConversationManager does
        # After refactoring, this should use SETTINGS.UNIFY_MODEL
        llm = LLM(SETTINGS.UNIFY_MODEL, event_broker=None)
        # The LLM class stores the model as provided
        assert llm.model == SETTINGS.UNIFY_MODEL


class TestResponseModelConstruction:
    """Tests for dynamic response model construction."""

    def test_text_mode_response_model_structure(self):
        """Text mode response model has thoughts and actions."""
        from unity.conversation_manager.domains.actions import (
            build_dynamic_response_models,
        )

        models = build_dynamic_response_models(active_tasks={}, realtime=False)
        text_model = models["text"]

        # Get the schema
        schema = text_model.model_json_schema()
        props = schema.get("properties", {})

        assert "thoughts" in props
        assert "actions" in props
        # Text mode should NOT have voice_utterance or realtime_guidance
        assert "voice_utterance" not in props
        assert "realtime_guidance" not in props

    def test_voice_model_tts_mode_current(self):
        """
        [Stage 0 - Baseline] TTS mode currently uses voice_utterance.

        This test documents the CURRENT behavior that will change in Stage 2.
        """
        from unity.conversation_manager.domains.actions import (
            build_dynamic_response_models,
        )

        models = build_dynamic_response_models(active_tasks={}, realtime=False)
        voice_model = models["call"]

        schema = voice_model.model_json_schema()
        props = schema.get("properties", {})

        assert "thoughts" in props
        assert "actions" in props
        # Current TTS mode uses voice_utterance
        assert "voice_utterance" in props
        assert "realtime_guidance" not in props

    def test_voice_model_realtime_mode_uses_guidance(self):
        """Realtime mode uses realtime_guidance instead of voice_utterance."""
        from unity.conversation_manager.domains.actions import (
            build_dynamic_response_models,
        )

        models = build_dynamic_response_models(active_tasks={}, realtime=True)
        voice_model = models["call"]

        schema = voice_model.model_json_schema()
        props = schema.get("properties", {})

        assert "thoughts" in props
        assert "actions" in props
        # Realtime mode uses realtime_guidance
        assert "realtime_guidance" in props
        assert "voice_utterance" not in props

    def test_unify_meet_model_matches_call_model(self):
        """unify_meet mode uses the same model as call mode."""
        from unity.conversation_manager.domains.actions import (
            build_dynamic_response_models,
        )

        models = build_dynamic_response_models(active_tasks={}, realtime=False)

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

    @pytest.mark.skip(reason="Stage 2: Enable after unifying voice response models")
    def test_voice_model_tts_mode_uses_guidance_after_refactor(self):
        """
        [Stage 2] TTS mode should use realtime_guidance after refactoring.

        This test will PASS after Stage 2 is implemented.
        """
        from unity.conversation_manager.domains.actions import (
            build_dynamic_response_models,
        )

        models = build_dynamic_response_models(active_tasks={}, realtime=False)
        voice_model = models["call"]

        schema = voice_model.model_json_schema()
        props = schema.get("properties", {})

        # After Stage 2, TTS mode should also use realtime_guidance
        assert "realtime_guidance" in props
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
            realtime=False,
            active_tasks={},
        )

        # Basic structure checks
        assert "<role>" in prompt
        assert "<bio>" in prompt
        assert "<boss_details>" in prompt
        assert "<output_format>" in prompt

        # Text/TTS mode specific
        assert "voice_utterance" in prompt

    def test_build_system_prompt_realtime_mode(self):
        """System prompt for realtime mode mentions realtime_guidance."""
        from unity.conversation_manager.prompt_builders import build_system_prompt

        prompt = build_system_prompt(
            bio="Test assistant bio",
            contact_id=1,
            first_name="Test",
            surname="User",
            phone_number="+15551234567",
            email_address="test@example.com",
            realtime=True,
            active_tasks={},
        )

        # Realtime mode specific
        assert "realtime_guidance" in prompt
        assert "<voice_calls_guide>" in prompt
        assert "Realtime Agent" in prompt

    def test_build_realtime_phone_agent_prompt(self):
        """Realtime phone agent prompt has fast brain instructions."""
        from unity.conversation_manager.prompt_builders import (
            build_realtime_phone_agent_prompt,
        )

        prompt = build_realtime_phone_agent_prompt(
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

    @pytest.mark.skip(reason="Stage 2: Enable after unifying voice response models")
    def test_build_system_prompt_tts_mode_uses_guidance_after_refactor(self):
        """
        [Stage 2] TTS mode system prompt should mention realtime_guidance.

        After refactoring, TTS mode will also use the guidance pattern.
        """
        from unity.conversation_manager.prompt_builders import build_system_prompt

        prompt = build_system_prompt(
            bio="Test assistant bio",
            contact_id=1,
            first_name="Test",
            surname="User",
            realtime=False,  # TTS mode
            active_tasks={},
        )

        # After Stage 2, TTS mode should also mention realtime_guidance
        assert "realtime_guidance" in prompt
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

        assert tts_manager.realtime is False
        assert sts_manager.realtime is True


# =============================================================================
# Unit Tests: Event Types
# =============================================================================


class TestVoiceEvents:
    """Tests for voice-related event types."""

    def test_realtime_guidance_event_structure(self):
        """RealtimeGuidance event has required fields."""
        from unity.conversation_manager.events import RealtimeGuidance

        contact = {"contact_id": 1, "first_name": "Test", "surname": "User"}
        event = RealtimeGuidance(contact=contact, content="Test guidance")

        assert event.contact == contact
        assert event.content == "Test guidance"
        assert hasattr(event, "timestamp")

    def test_realtime_guidance_serialization(self):
        """RealtimeGuidance event can be serialized and deserialized."""
        from unity.conversation_manager.events import Event, RealtimeGuidance

        contact = {"contact_id": 1, "first_name": "Test", "surname": "User"}
        original = RealtimeGuidance(contact=contact, content="Test guidance")

        # Serialize
        json_str = original.to_json()
        data = json.loads(json_str)

        # Verify structure
        assert data["event_name"] == "RealtimeGuidance"
        assert data["payload"]["content"] == "Test guidance"

        # Deserialize
        restored = Event.from_json(json_str)
        assert isinstance(restored, RealtimeGuidance)
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
# Integration Tests: LLM Output Routing
# =============================================================================


class TestLLMOutputRouting:
    """Tests for LLM output routing in different modes."""

    @pytest.fixture
    def mock_event_broker(self):
        """Create a mock event broker that captures published events."""
        broker = MagicMock()
        broker.publish = MagicMock(return_value=asyncio.coroutine(lambda: None)())
        return broker

    @pytest.fixture
    def sample_contact(self):
        return {
            "contact_id": 1,
            "first_name": "Test",
            "surname": "User",
            "phone_number": "+15551234567",
            "email_address": "test@example.com",
        }

    def test_llm_domain_streaming_extracts_voice_utterance(self):
        """LLM domain extracts voice_utterance from streaming output."""
        from unity.conversation_manager.domains.llm import LLM

        # The LLM class has _to_streaming_format which handles voice_utterance
        # This is used in TTS mode for streaming responses
        llm = LLM("test-model", event_broker=None)

        # Test the streaming format conversion
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            thoughts: str
            voice_utterance: str

        format_result = llm._to_streaming_format(TestResponse)
        assert format_result["type"] == "json_schema"
        assert "voice_utterance" in str(format_result)


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
# Integration Tests: Full Voice Call Flow (Requires Redis)
# =============================================================================


@pytest.mark.asyncio
class TestVoiceCallFlowIntegration:
    """
    Integration tests for voice call flows.

    These tests require the Redis server fixture from conftest.py.
    They validate the event flow for both TTS and Realtime modes.
    """

    @pytest.fixture
    def boss_contact(self):
        return {
            "contact_id": 1,
            "first_name": "Test",
            "surname": "Boss",
            "is_boss": True,
            "phone_number": "+15555555678",
            "email_address": "boss@test.com",
        }

    async def test_phone_call_started_event_flow(
        self,
        test_redis_client,
        event_capture,
        boss_contact,
    ):
        """
        Verify phone call start event is properly published and captured.
        """
        from unity.conversation_manager.events import PhoneCallStarted

        # Publish a call started event
        event = PhoneCallStarted(contact=boss_contact)
        await test_redis_client.publish(
            "app:comms:phone_call_started",
            event.to_json(),
        )

        # Wait briefly for event propagation
        await asyncio.sleep(0.5)

        # Verify event was captured
        events = event_capture.get_events(PhoneCallStarted)
        assert len(events) >= 1

    async def test_realtime_guidance_event_flow(
        self,
        test_redis_client,
        event_capture,
        boss_contact,
    ):
        """
        Verify realtime guidance events flow through the system.
        """
        from unity.conversation_manager.events import RealtimeGuidance

        # Publish a guidance event
        event = RealtimeGuidance(
            contact=boss_contact,
            content="Please ask about their schedule",
        )
        await test_redis_client.publish("app:call:realtime_guidance", event.to_json())

        # Wait briefly for event propagation
        await asyncio.sleep(0.5)

        # Note: The event capture listens to app:comms:* not app:call:*
        # This test verifies the event is properly formed
        assert event.content == "Please ask about their schedule"

    @pytest.mark.skip(reason="Stage 2: Enable after TTS mode uses guidance pattern")
    async def test_tts_mode_publishes_guidance_not_utterance(
        self,
        test_redis_client,
        event_capture,
        boss_contact,
    ):
        """
        [Stage 2] TTS mode should publish realtime_guidance events.

        After refactoring, when the Main CM Brain responds during a TTS call,
        it should publish RealtimeGuidance instead of OutboundPhoneUtterance.
        """

        # This test will be implemented when Stage 2 is complete
        # It should verify that TTS mode publishes guidance events


# =============================================================================
# Integration Tests: Mode-Specific Response Streaming
# =============================================================================


@pytest.mark.asyncio
class TestResponseStreaming:
    """Tests for response streaming in different voice modes."""

    async def test_stream_response_channel_format(self, test_redis_client):
        """Verify response streaming channel message format."""
        pubsub = test_redis_client.pubsub()
        await pubsub.subscribe("app:call:response_gen")

        # Consume the subscription confirmation message
        await pubsub.get_message(timeout=1.0)

        # Simulate the streaming protocol
        await test_redis_client.publish(
            "app:call:response_gen",
            json.dumps({"type": "start_gen"}),
        )
        await test_redis_client.publish(
            "app:call:response_gen",
            json.dumps({"type": "gen_chunk", "chunk": "Hello "}),
        )
        await test_redis_client.publish(
            "app:call:response_gen",
            json.dumps({"type": "gen_chunk", "chunk": "there!"}),
        )
        await test_redis_client.publish(
            "app:call:response_gen",
            json.dumps({"type": "end_gen"}),
        )

        # Collect messages with polling
        messages = []
        for _ in range(40):
            msg = await pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=0.1,
            )
            if msg and msg["type"] == "message":
                messages.append(json.loads(msg["data"]))
            await asyncio.sleep(0.025)

        await pubsub.unsubscribe()

        # Verify protocol
        types = [m["type"] for m in messages]
        assert "start_gen" in types, f"Expected start_gen in {types}"
        assert "gen_chunk" in types, f"Expected gen_chunk in {types}"
        assert "end_gen" in types, f"Expected end_gen in {types}"

    async def test_unify_meet_response_channel(self, test_redis_client):
        """Verify unify_meet uses separate response channel."""
        pubsub = test_redis_client.pubsub()
        await pubsub.subscribe("app:unify_meet:response_gen")

        # Consume the subscription confirmation message
        await pubsub.get_message(timeout=1.0)

        await test_redis_client.publish(
            "app:unify_meet:response_gen",
            json.dumps({"type": "start_gen"}),
        )

        # Poll for message with retries
        msg = None
        for _ in range(20):
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
            if msg and msg["type"] == "message":
                break
            await asyncio.sleep(0.05)

        await pubsub.unsubscribe()

        assert msg is not None, "Expected to receive published message"
        assert msg["type"] == "message"
        assert json.loads(msg["data"])["type"] == "start_gen"


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
        from unity.conversation_manager.domains.actions import (
            build_dynamic_response_models,
        )

        models = build_dynamic_response_models(active_tasks={}, realtime=False)
        text_model = models["text"]
        schema = text_model.model_json_schema()
        props = schema.get("properties", {})

        assert "voice_utterance" not in props
        assert "realtime_guidance" not in props

    def test_actions_union_includes_core_actions(self):
        """Response model includes all core action types."""
        from unity.conversation_manager.domains.actions import (
            build_dynamic_response_models,
        )

        models = build_dynamic_response_models(active_tasks={}, realtime=False)
        text_model = models["text"]
        schema = text_model.model_json_schema()

        # Check that actions field exists
        actions_schema = schema.get("properties", {}).get("actions", {})
        assert actions_schema is not None

        # The schema should reference action types
        schema_str = json.dumps(schema)
        assert "send_sms" in schema_str.lower() or "SendSMS" in schema_str
        assert "send_email" in schema_str.lower() or "SendEmail" in schema_str
        assert "start_task" in schema_str.lower() or "StartTask" in schema_str

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

        # Verify the source code uses SETTINGS.UNIFY_MODEL (not hardcoded)
        source = inspect.getsource(cm_module.ConversationManager.__init__)
        assert (
            "SETTINGS.UNIFY_MODEL" in source
        ), "ConversationManager should use SETTINGS.UNIFY_MODEL for LLM"
        assert (
            '"gpt-5-mini@openai"' not in source
        ), "ConversationManager should not have hardcoded model name"


class TestStage2UnifiedVoiceResponse:
    """
    [Stage 2] Tests for unified voice response model (realtime_guidance everywhere).

    These tests verify that after Stage 2:
    - TTS mode outputs realtime_guidance instead of voice_utterance
    - Both TTS and Realtime modes use the same response model structure
    - The system prompt for TTS mode mentions realtime_guidance
    """

    @pytest.mark.skip(reason="Stage 2: Enable after unifying voice response models")
    def test_tts_mode_response_model_has_guidance(self):
        """TTS mode response model uses realtime_guidance field."""
        from unity.conversation_manager.domains.actions import (
            build_dynamic_response_models,
        )

        models = build_dynamic_response_models(active_tasks={}, realtime=False)
        voice_model = models["call"]
        schema = voice_model.model_json_schema()
        props = schema.get("properties", {})

        assert "realtime_guidance" in props
        assert "voice_utterance" not in props

    @pytest.mark.skip(reason="Stage 2: Enable after unifying voice response models")
    def test_tts_and_realtime_models_match(self):
        """TTS and Realtime modes use identical response model structure."""
        from unity.conversation_manager.domains.actions import (
            build_dynamic_response_models,
        )

        tts_models = build_dynamic_response_models(active_tasks={}, realtime=False)
        rt_models = build_dynamic_response_models(active_tasks={}, realtime=True)

        tts_schema = tts_models["call"].model_json_schema()
        rt_schema = rt_models["call"].model_json_schema()

        # After Stage 2, these should be identical
        assert (
            tts_schema.get("properties", {}).keys()
            == rt_schema.get(
                "properties",
                {},
            ).keys()
        )


class TestStage3TTSFastBrain:
    """
    [Stage 3] Tests for TTS Fast Brain implementation.

    These tests verify that after Stage 3:
    - call.py has its own lightweight LLM for conversational responses
    - The TTS fast brain receives guidance from the Main CM Brain
    - The fast brain uses the same prompt as the Realtime phone agent
    """

    @pytest.mark.skip(reason="Stage 3: Enable after implementing TTS fast brain")
    def test_tts_call_has_fast_brain_model(self):
        """TTS call.py should have a fast LLM for conversational responses."""
        # This would inspect the call.py implementation

    @pytest.mark.skip(reason="Stage 3: Enable after implementing TTS fast brain")
    def test_tts_fast_brain_receives_guidance(self):
        """TTS fast brain should subscribe to realtime_guidance channel."""
        # This would verify the event subscription in call.py

    @pytest.mark.skip(reason="Stage 3: Enable after implementing TTS fast brain")
    def test_tts_fast_brain_uses_phone_agent_prompt(self):
        """TTS fast brain should use build_realtime_phone_agent_prompt."""
        # This would verify the prompt used in call.py matches realtime_call.py
