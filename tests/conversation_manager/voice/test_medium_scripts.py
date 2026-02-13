"""
tests/conversation_manager/test_medium_scripts.py
=======================================================

Tests for the medium scripts (call.py and sts_call.py) that handle voice calls.

These scripts implement the "fast brain" Voice Agent that handles real-time
conversation while the Main CM Brain (slow brain) handles orchestration.

## Test Categories

### Unit Tests (no external dependencies)
- Assistant class initialization and state management
- Common helper functions (publish events, create_end_call, etc.)
- CLI argument parsing
- Voice prompt building
- Event type selection based on channel

### Integration Tests (require event broker)
- Event publishing flows
- Guidance subscription patterns
- Cross-thread event delivery

## Key Components Tested

1. **Assistant class** (call.py, sts_call.py):
   - Initialization with contact/boss/channel/instructions
   - set_call_received() state transitions
   - Utterance event type selection

2. **Common helpers** (common.py):
   - publish_call_started / publish_call_ended
   - create_end_call with pre_shutdown_callback
   - setup_inactivity_timeout
   - configure_from_cli argument parsing

3. **Voice Agent prompt**:
   - build_voice_agent_prompt output structure
"""

import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def event_broker():
    """Local in-memory broker for testing."""
    from unity.conversation_manager.event_broker import create_event_broker

    broker = create_event_broker()
    yield broker
    await broker.aclose()


@pytest.fixture
def boss_contact():
    """Standard boss contact for testing."""
    return {
        "contact_id": 1,
        "first_name": "Test",
        "surname": "Boss",
        "phone_number": "+15555555555",
        "email_address": "boss@test.com",
    }


@pytest.fixture
def external_contact():
    """Standard external contact for testing."""
    return {
        "contact_id": 2,
        "first_name": "External",
        "surname": "Contact",
        "phone_number": "+15555555556",
        "email_address": "contact@test.com",
    }


# =============================================================================
# Unit Tests: Assistant Class (call.py - TTS mode)
# =============================================================================


class TestTTSAssistantClass:
    """Tests for the Assistant class in call.py (TTS mode)."""

    def test_assistant_initialization_phone_channel(self, boss_contact):
        """Assistant initializes correctly for phone channel."""
        from unity.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="phone",
            instructions="Test instructions",
            outbound=False,
        )

        assert assistant.contact == boss_contact
        assert assistant.boss == boss_contact
        assert assistant.channel == "phone"
        assert assistant.call_received is True  # inbound call, already received

    def test_assistant_initialization_unify_meet_channel(self, boss_contact):
        """Assistant initializes correctly for unify_meet channel."""
        from unity.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="meet",
            instructions="Test instructions",
            outbound=False,
        )

        assert assistant.channel == "meet"
        assert assistant.call_received is True

    def test_assistant_outbound_call_not_received_initially(self, boss_contact):
        """Outbound calls start with call_received=False."""
        from unity.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="phone",
            instructions="Test instructions",
            outbound=True,
        )

        assert assistant.call_received is False

    def test_assistant_set_call_received(self, boss_contact):
        """set_call_received() updates state correctly."""
        from unity.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="phone",
            instructions="Test instructions",
            outbound=True,
        )

        assert assistant.call_received is False
        assistant.set_call_received()
        assert assistant.call_received is True

    def test_assistant_utterance_event_type_phone(self, boss_contact):
        """Phone channel uses InboundPhoneUtterance."""
        from unity.conversation_manager.events import InboundPhoneUtterance
        from unity.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="phone",
            instructions="Test instructions",
        )

        assert assistant.utterance_event == InboundPhoneUtterance

    def test_assistant_utterance_event_type_meet(self, boss_contact):
        """Meet channel uses InboundUnifyMeetUtterance."""
        from unity.conversation_manager.events import InboundUnifyMeetUtterance
        from unity.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="meet",
            instructions="Test instructions",
        )

        assert assistant.utterance_event == InboundUnifyMeetUtterance

    def test_assistant_outbound_utterance_event_type_phone(self, boss_contact):
        """Phone channel uses OutboundPhoneUtterance for assistant."""
        from unity.conversation_manager.events import OutboundPhoneUtterance
        from unity.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="phone",
            instructions="Test instructions",
        )

        assert assistant.assistant_utterance_event == OutboundPhoneUtterance

    def test_assistant_outbound_utterance_event_type_meet(self, boss_contact):
        """Meet channel uses OutboundUnifyMeetUtterance for assistant."""
        from unity.conversation_manager.events import OutboundUnifyMeetUtterance
        from unity.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="meet",
            instructions="Test instructions",
        )

        assert assistant.assistant_utterance_event == OutboundUnifyMeetUtterance


# =============================================================================
# Unit Tests: Assistant Class (sts_call.py - Realtime/STS mode)
# =============================================================================


class TestSTSAssistantClass:
    """Tests for the Assistant class in sts_call.py (STS mode)."""

    def test_sts_assistant_initialization(self, boss_contact):
        """STS Assistant initializes correctly."""
        from unity.conversation_manager.medium_scripts.sts_call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            instructions="Test instructions",
            outbound=False,
        )

        assert assistant.contact == boss_contact
        assert assistant.boss == boss_contact
        assert assistant.call_received is True

    def test_sts_assistant_outbound_not_received(self, boss_contact):
        """STS outbound calls start with call_received=False."""
        from unity.conversation_manager.medium_scripts.sts_call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            instructions="Test instructions",
            outbound=True,
        )

        assert assistant.call_received is False

    def test_sts_assistant_set_call_received(self, boss_contact):
        """STS set_call_received() updates state correctly."""
        from unity.conversation_manager.medium_scripts.sts_call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            instructions="Test instructions",
            outbound=True,
        )

        assert assistant.call_received is False
        assistant.set_call_received()
        assert assistant.call_received is True


# =============================================================================
# Unit Tests: Common Helpers
# =============================================================================


class TestCommonHelpers:
    """Tests for shared helper functions in common.py."""

    def test_should_dispatch_livekit_agent_with_dev_command(self, monkeypatch):
        """should_dispatch_livekit_agent returns True for 'dev' command."""
        from unity.conversation_manager.medium_scripts import common

        # Patch sys.argv in the common module's namespace
        monkeypatch.setattr(common.sys, "argv", ["call.py", "dev"])
        assert common.should_dispatch_livekit_agent() is True

    def test_should_dispatch_livekit_agent_with_connect_command(self, monkeypatch):
        """should_dispatch_livekit_agent returns True for 'connect' command."""
        from unity.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common.sys, "argv", ["call.py", "connect"])
        assert common.should_dispatch_livekit_agent() is True

    def test_should_not_dispatch_livekit_agent_for_download_files(self, monkeypatch):
        """should_dispatch_livekit_agent returns False for 'download-files' command."""
        from unity.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common.sys, "argv", ["call.py", "download-files"])
        assert common.should_dispatch_livekit_agent() is False

    def test_should_not_dispatch_livekit_agent_with_no_args(self, monkeypatch):
        """should_dispatch_livekit_agent returns False when no args provided."""
        from unity.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common.sys, "argv", ["call.py"])
        assert common.should_dispatch_livekit_agent() is False


# =============================================================================
# Integration Tests: STS Per-Turn Usage Logging via unillm.log_usage
# =============================================================================


class TestSTSPerTurnUsageLogging:
    """Tests that the sts_call metrics_collected handler calls unillm.log_usage
    with real token usage from the Realtime API, replacing the old
    duration-based heuristic."""

    def test_metrics_handler_calls_log_usage(self, tmp_path, monkeypatch):
        """The _on_metrics handler converts RealtimeModelMetrics to a
        unillm.log_usage call with the correct usage dict and transcript."""
        from unittest.mock import patch
        from livekit.agents.metrics import RealtimeModelMetrics

        # Build a realistic RealtimeModelMetrics (as LiveKit would emit)
        metrics = RealtimeModelMetrics(
            label="gpt-realtime",
            model="gpt-4o-realtime-preview",
            request_id="req_abc123",
            timestamp=1000.0,
            duration=2.5,
            ttft=0.3,
            cancelled=False,
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            tokens_per_second=40.0,
            input_token_details=RealtimeModelMetrics.InputTokenDetails(
                audio_tokens=170,
                text_tokens=30,
                image_tokens=0,
                cached_tokens=0,
                cached_tokens_details=None,
            ),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=15,
                audio_tokens=85,
                image_tokens=0,
            ),
        )

        # Simulate the handler logic from sts_call.py
        # (extracted here since sts_call.entrypoint requires a LiveKit room)
        usage = {
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "total_tokens": metrics.total_tokens,
            "input_token_details": {
                "audio_tokens": metrics.input_token_details.audio_tokens,
                "text_tokens": metrics.input_token_details.text_tokens,
                "cached_tokens": metrics.input_token_details.cached_tokens,
            },
            "output_token_details": {
                "audio_tokens": metrics.output_token_details.audio_tokens,
                "text_tokens": metrics.output_token_details.text_tokens,
            },
        }

        transcript = [
            {"role": "user", "content": "What time is the meeting?"},
            {"role": "assistant", "content": "Let me check on that."},
        ]

        # Configure unillm file logging
        from unillm import settings as unillm_settings
        from unillm import logger as unillm_logger

        monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
        monkeypatch.setattr(unillm_settings.SETTINGS, "UNILLM_LOG", True)
        monkeypatch.setattr(
            unillm_settings.SETTINGS,
            "UNILLM_LOG_DIR",
            str(tmp_path),
        )
        monkeypatch.setattr(unillm_logger, "_LOG_ENABLED", True)
        monkeypatch.setattr(unillm_logger, "_LOG_DIR_CHECKED", False)
        monkeypatch.setattr(unillm_logger, "_LOG_DIR", None)

        import unillm as _unillm

        with patch("unillm.logger.unify.deduct_credits") as mock_deduct:
            billed_cost = _unillm.log_usage(
                metrics.model,
                usage,
                transcript=transcript,
                label=metrics.model,
            )

        # Credits should have been deducted
        mock_deduct.assert_called_once()
        assert billed_cost > 0

        # A log file should exist with usage details
        log_files = list(tmp_path.glob("*_usage.txt"))
        assert len(log_files) == 1

        content = log_files[0].read_text()
        assert "gpt-4o-realtime-preview" in content
        assert "audio_tokens" in content
        assert "What time is the meeting?" in content
        assert "Let me check on that." in content

    def test_usage_dict_shape_matches_realtime_api(self):
        """The usage dict built from RealtimeModelMetrics has the shape
        that unillm.log_usage and compute_full_cost_from_usage expect."""
        from livekit.agents.metrics import RealtimeModelMetrics
        from unillm.costs import compute_full_cost_from_usage

        metrics = RealtimeModelMetrics(
            label="test",
            model="gpt-4o-realtime-preview",
            request_id="req_1",
            timestamp=0.0,
            duration=1.0,
            ttft=0.1,
            cancelled=False,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            tokens_per_second=50.0,
            input_token_details=RealtimeModelMetrics.InputTokenDetails(
                audio_tokens=80,
                text_tokens=20,
                image_tokens=0,
                cached_tokens=0,
                cached_tokens_details=None,
            ),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=10,
                audio_tokens=40,
                image_tokens=0,
            ),
        )

        usage = {
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "total_tokens": metrics.total_tokens,
            "input_token_details": {
                "audio_tokens": metrics.input_token_details.audio_tokens,
                "text_tokens": metrics.input_token_details.text_tokens,
                "cached_tokens": metrics.input_token_details.cached_tokens,
            },
            "output_token_details": {
                "audio_tokens": metrics.output_token_details.audio_tokens,
                "text_tokens": metrics.output_token_details.text_tokens,
            },
        }

        # compute_full_cost_from_usage should handle this without error
        cost = compute_full_cost_from_usage("gpt-4o-realtime-preview", usage)
        assert cost > 0

    def test_cancelled_response_still_logs(self, tmp_path, monkeypatch):
        """Interrupted/cancelled responses still produce a log file with
        partial usage (OpenAI still bills for partial audio tokens)."""
        from unittest.mock import patch
        from unillm import settings as unillm_settings
        from unillm import logger as unillm_logger

        monkeypatch.delenv("UNILLM_LOG_DIR", raising=False)
        monkeypatch.setattr(unillm_settings.SETTINGS, "UNILLM_LOG", True)
        monkeypatch.setattr(
            unillm_settings.SETTINGS,
            "UNILLM_LOG_DIR",
            str(tmp_path),
        )
        monkeypatch.setattr(unillm_logger, "_LOG_ENABLED", True)
        monkeypatch.setattr(unillm_logger, "_LOG_DIR_CHECKED", False)
        monkeypatch.setattr(unillm_logger, "_LOG_DIR", None)

        # Partial usage from a cancelled response (user interrupted)
        usage = {
            "input_tokens": 50,
            "output_tokens": 10,
            "total_tokens": 60,
            "input_token_details": {
                "audio_tokens": 40,
                "text_tokens": 10,
                "cached_tokens": 0,
            },
            "output_token_details": {
                "audio_tokens": 8,
                "text_tokens": 2,
            },
        }

        import unillm as _unillm

        with patch("unillm.logger.unify.deduct_credits"):
            billed_cost = _unillm.log_usage(
                "gpt-4o-realtime-preview",
                usage,
            )

        assert billed_cost > 0

        log_files = list(tmp_path.glob("*_usage.txt"))
        assert len(log_files) == 1


# =============================================================================
# Unit Tests: Event Publishing Helpers
# =============================================================================


@pytest.mark.asyncio
class TestEventPublishingHelpers:
    """Tests for event publishing helper functions.

    These tests patch common.event_broker to use our test fixture so we can
    verify the events are published correctly.
    """

    async def test_publish_call_started_phone_channel(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
    ):
        """publish_call_started publishes PhoneCallStarted for phone channel."""
        from unity.conversation_manager.events import Event, PhoneCallStarted
        from unity.conversation_manager.medium_scripts import common

        # Patch the event_broker in common module to use our test fixture
        monkeypatch.setattr(common, "event_broker", event_broker)

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:phone_call_started")

            await common.publish_call_started(boss_contact, "phone")

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            event = Event.from_json(msg["data"])
            assert isinstance(event, PhoneCallStarted)
            assert event.contact == boss_contact

    async def test_publish_call_started_meet_channel(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
    ):
        """publish_call_started publishes UnifyMeetStarted for meet channel."""
        from unity.conversation_manager.events import Event, UnifyMeetStarted
        from unity.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common, "event_broker", event_broker)

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:meet_call_started")

            await common.publish_call_started(boss_contact, "meet")

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            event = Event.from_json(msg["data"])
            assert isinstance(event, UnifyMeetStarted)
            assert event.contact == boss_contact

    async def test_publish_call_ended_phone_channel(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
    ):
        """publish_call_ended publishes PhoneCallEnded for phone channel."""
        from unity.conversation_manager.events import Event, PhoneCallEnded
        from unity.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common, "event_broker", event_broker)

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:phone_call_ended")

            await common.publish_call_ended(boss_contact, "phone")

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            event = Event.from_json(msg["data"])
            assert isinstance(event, PhoneCallEnded)
            assert event.contact == boss_contact

    async def test_publish_call_ended_meet_channel(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
    ):
        """publish_call_ended publishes UnifyMeetEnded for meet channel."""
        from unity.conversation_manager.events import Event, UnifyMeetEnded
        from unity.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common, "event_broker", event_broker)

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:meet_call_ended")

            await common.publish_call_ended(boss_contact, "meet")

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            event = Event.from_json(msg["data"])
            assert isinstance(event, UnifyMeetEnded)
            assert event.contact == boss_contact


# =============================================================================
# Unit Tests: End Call Helper
# =============================================================================


@pytest.mark.asyncio
class TestEndCallHelper:
    """Tests for create_end_call helper.

    These tests patch common.event_broker to use our test fixture so we can
    verify the events are published correctly.
    """

    async def test_create_end_call_publishes_ended_event(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
    ):
        """create_end_call returns function that publishes ended event."""
        from unity.conversation_manager.events import Event, PhoneCallEnded
        from unity.conversation_manager.medium_scripts import common

        # Patch the event_broker in common module to use our test fixture
        monkeypatch.setattr(common, "event_broker", event_broker)

        end_call = common.create_end_call(boss_contact, "phone")

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:phone_call_ended")

            await end_call()

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            event = Event.from_json(msg["data"])
            assert isinstance(event, PhoneCallEnded)

    async def test_create_end_call_runs_pre_shutdown_callback(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
    ):
        """create_end_call runs pre_shutdown_callback before shutdown."""
        from unity.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common, "event_broker", event_broker)

        callback_called = {"value": False}

        def pre_shutdown():
            callback_called["value"] = True

        end_call = common.create_end_call(
            boss_contact,
            "phone",
            pre_shutdown_callback=pre_shutdown,
        )

        await end_call()

        assert callback_called["value"] is True

    async def test_create_end_call_handles_callback_error(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
        capsys,
    ):
        """create_end_call continues even if callback raises."""
        from unity.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common, "event_broker", event_broker)

        def failing_callback():
            raise ValueError("Callback error")

        end_call = common.create_end_call(
            boss_contact,
            "phone",
            pre_shutdown_callback=failing_callback,
        )

        # Should not raise
        await end_call()

        captured = capsys.readouterr()
        assert "Error in pre-shutdown callback" in captured.out


# =============================================================================
# Unit Tests: CLI Argument Parsing
# =============================================================================


class TestCLIArgumentParsing:
    """Tests for configure_from_cli argument parsing.

    These tests patch sys.argv in the common module's namespace since that's
    where configure_from_cli reads it from.
    """

    def test_configure_from_cli_with_full_args(self, monkeypatch):
        """configure_from_cli parses all arguments correctly."""
        from unity.conversation_manager.medium_scripts import common
        from unity.session_details import SESSION_DETAILS

        # Reset SESSION_DETAILS before test
        SESSION_DETAILS.reset()

        contact_json = json.dumps(
            {
                "contact_id": 1,
                "first_name": "Test",
                "surname": "User",
            },
        )
        boss_json = json.dumps(
            {
                "contact_id": 1,
                "first_name": "Boss",
                "surname": "Person",
            },
        )

        # Simulate CLI args: script.py dev assistant_number VOICE_PROVIDER VOICE_ID OUTBOUND CHANNEL CONTACT BOSS ASSISTANT_BIO
        monkeypatch.setattr(
            common.sys,
            "argv",
            [
                "call.py",
                "dev",
                "12345",
                "elevenlabs",
                "voice123",
                "True",
                "phone",
                contact_json,
                boss_json,
                "Test assistant bio",
            ],
        )

        livekit_agent_name, room_name = common.configure_from_cli(
            extra_env=[
                ("CONTACT", True),
                ("BOSS", True),
                ("ASSISTANT_BIO", False),
            ],
        )

        assert livekit_agent_name == "unity_12345"
        assert room_name == "unity_12345"
        assert SESSION_DETAILS.voice.provider == "elevenlabs"
        assert SESSION_DETAILS.voice.id == "voice123"
        assert SESSION_DETAILS.voice_call.outbound is True
        assert SESSION_DETAILS.voice_call.channel == "phone"

    def test_configure_from_cli_livekit_agent_name_with_room(self, monkeypatch):
        """configure_from_cli handles livekit_agent_name:room_name format for UnifyMeet calls.

        For UnifyMeet, the caller (LivekitCallManager.start_unify_meet) passes
        "livekit_agent_name:room_name" where both are already prefixed with "unity_".
        The function splits on ":" and returns the two parts.
        """
        from unity.conversation_manager.medium_scripts import common
        from unity.session_details import SESSION_DETAILS

        SESSION_DETAILS.reset()

        contact_json = json.dumps({"contact_id": 1, "first_name": "Test"})
        boss_json = json.dumps({"contact_id": 1, "first_name": "Boss"})

        # Simulate UnifyMeet call with colon-separated livekit_agent_name:room_name
        # (This matches what LivekitCallManager.start_unify_meet passes)
        monkeypatch.setattr(
            common.sys,
            "argv",
            [
                "call.py",
                "dev",
                "unity_assistant_web:unity_assistant_web",  # Already prefixed by caller
                "cartesia",
                "voice456",
                "False",
                "meet",
                contact_json,
                boss_json,
                "Bio",
            ],
        )

        livekit_agent_name, room_name = common.configure_from_cli(
            extra_env=[
                ("CONTACT", True),
                ("BOSS", True),
                ("ASSISTANT_BIO", False),
            ],
        )

        # Colon triggers the split: "unity_assistant_web:unity_assistant_web"
        # becomes livekit_agent_name="unity_assistant_web", room_name="unity_assistant_web"
        assert livekit_agent_name == "unity_assistant_web"
        assert room_name == "unity_assistant_web"

    def test_configure_from_cli_defaults_none_voice_provider(self, monkeypatch):
        """configure_from_cli defaults 'None' voice provider to cartesia."""
        from unity.conversation_manager.medium_scripts import common
        from unity.session_details import SESSION_DETAILS

        SESSION_DETAILS.reset()

        contact_json = json.dumps({"contact_id": 1, "first_name": "Test"})
        boss_json = json.dumps({"contact_id": 1, "first_name": "Boss"})

        monkeypatch.setattr(
            common.sys,
            "argv",
            [
                "call.py",
                "dev",
                "12345",
                "None",  # Voice provider as "None" string
                "None",  # Voice ID as "None" string
                "False",
                "phone",
                contact_json,
                boss_json,
                "Bio",
            ],
        )

        common.configure_from_cli(
            extra_env=[
                ("CONTACT", True),
                ("BOSS", True),
                ("ASSISTANT_BIO", False),
            ],
        )

        assert SESSION_DETAILS.voice.provider == "cartesia"
        assert SESSION_DETAILS.voice.id == ""


# =============================================================================
# Integration Tests: Guidance Channel Subscription
# =============================================================================


@pytest.mark.asyncio
class TestGuidanceChannelSubscription:
    """Tests for guidance channel subscription patterns."""

    async def test_guidance_channel_receives_call_guidance(self, event_broker):
        """Guidance channel receives CallGuidance events."""
        from unity.conversation_manager.events import CallGuidance, Event

        contact = {"contact_id": 1, "first_name": "Test"}

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:call_guidance")

            event = CallGuidance(contact=contact, content="Test guidance")
            await event_broker.publish("app:call:call_guidance", event.to_json())

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            received = Event.from_json(msg["data"])
            assert isinstance(received, CallGuidance)
            assert received.content == "Test guidance"

    async def test_status_channel_receives_stop_signal(self, event_broker):
        """Status channel receives stop signals."""
        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:status")

            await event_broker.publish(
                "app:call:status",
                json.dumps({"type": "stop"}),
            )

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            data = json.loads(msg["data"])
            assert data["type"] == "stop"

    async def test_status_channel_receives_call_answered_signal(self, event_broker):
        """Status channel receives call_answered signals."""
        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:status")

            await event_broker.publish(
                "app:call:status",
                json.dumps({"type": "call_answered"}),
            )

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            data = json.loads(msg["data"])
            assert data["type"] == "call_answered"


# =============================================================================
# Integration Tests: Utterance Event Publishing
# =============================================================================


@pytest.mark.asyncio
class TestUtteranceEventPublishing:
    """Tests for utterance event publishing patterns."""

    async def test_phone_utterance_published_to_correct_channel(
        self,
        event_broker,
        boss_contact,
    ):
        """Phone utterances are published to app:comms:phone_utterance."""
        from unity.conversation_manager.events import (
            Event,
            InboundPhoneUtterance,
        )

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:phone_utterance")

            event = InboundPhoneUtterance(contact=boss_contact, content="Hello")
            await event_broker.publish("app:comms:phone_utterance", event.to_json())

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            received = Event.from_json(msg["data"])
            assert isinstance(received, InboundPhoneUtterance)
            assert received.content == "Hello"

    async def test_meet_utterance_published_to_correct_channel(
        self,
        event_broker,
        boss_contact,
    ):
        """Meet utterances are published to app:comms:meet_utterance."""
        from unity.conversation_manager.events import (
            Event,
            InboundUnifyMeetUtterance,
        )

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:meet_utterance")

            event = InboundUnifyMeetUtterance(contact=boss_contact, content="Hello")
            await event_broker.publish("app:comms:meet_utterance", event.to_json())

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            received = Event.from_json(msg["data"])
            assert isinstance(received, InboundUnifyMeetUtterance)
            assert received.content == "Hello"


# =============================================================================
# Unit Tests: Inactivity Timeout
# =============================================================================


@pytest.mark.asyncio
class TestInactivityTimeout:
    """Tests for inactivity timeout setup."""

    async def test_setup_inactivity_timeout_returns_touch_function(
        self,
        event_broker,
        boss_contact,
    ):
        """setup_inactivity_timeout returns a callable touch function."""
        from unittest.mock import AsyncMock

        from unity.conversation_manager.medium_scripts.common import (
            setup_inactivity_timeout,
        )

        end_call = AsyncMock()
        touch = setup_inactivity_timeout(end_call, timeout=300)

        assert callable(touch)

    async def test_touch_function_is_callable(self, event_broker, boss_contact):
        """Touch function can be called without error."""
        from unittest.mock import AsyncMock

        from unity.conversation_manager.medium_scripts.common import (
            setup_inactivity_timeout,
        )

        end_call = AsyncMock()
        touch = setup_inactivity_timeout(end_call, timeout=300)

        # Should not raise
        touch()


# =============================================================================
# Guidance tests: chat context and system messages
# =============================================================================


@pytest.mark.asyncio
class TestFastBrainGuidanceFlow:
    """Coverage for guidance delivery in the TTS fast brain path."""

    async def test_tts_guidance_reaches_agent_generation_context(self, monkeypatch):
        """Guidance should be injected into the chat context used for generation."""
        from livekit.agents import llm
        from unity.conversation_manager.medium_scripts import call as call_script

        contact = {
            "contact_id": 2,
            "first_name": "Caller",
            "surname": "Example",
            "phone_number": "+15550100002",
            "email_address": "caller@example.com",
        }
        boss = {
            "contact_id": 1,
            "first_name": "Manager",
            "surname": "Example",
            "phone_number": "+15550100001",
            "email_address": "manager@example.com",
        }

        class _ImmediateAwaitable:
            def __await__(self):
                async def _done():
                    return None

                return _done().__await__()

        class _FakeRoom:
            def on(self, *args, **kwargs):
                return None

        class _FakeJobContext:
            def __init__(self):
                self.room = _FakeRoom()

            async def connect(self):
                return None

        class _FakeEventBroker:
            def __init__(self):
                self.callbacks = {}

            def register_callback(self, channel, handler):
                self.callbacks[channel] = handler

            async def publish(self, channel, message):
                return 1

        fake_session_holder = {}

        class _FakeSession:
            def __init__(self, *args, **kwargs):
                self._chat_ctx = llm.ChatContext()
                self.current_agent = None
                self._events = {}
                self.generate_reply_calls = 0
                fake_session_holder["session"] = self

            def on(self, event_name):
                def _decorator(fn):
                    self._events[event_name] = fn
                    return fn

                return _decorator

            async def start(self, room, agent, room_input_options=None):
                self.current_agent = agent

            def generate_reply(self, **kwargs):
                self.generate_reply_calls += 1
                return _ImmediateAwaitable()

        class _FakeAssistant:
            def __init__(self, *args, **kwargs):
                self._chat_ctx = llm.ChatContext()
                self.call_received = True

            def set_call_received(self):
                self.call_received = True

        async def _noop_async(*args, **kwargs):
            return None

        async def _noop_end_call():
            return None

        fake_broker = _FakeEventBroker()
        fake_session_details = SimpleNamespace(
            populate_from_env=lambda: None,
            voice=SimpleNamespace(provider="cartesia", id=""),
            voice_call=SimpleNamespace(
                outbound=False,
                channel="meet",
                contact_json=json.dumps(contact),
                boss_json=json.dumps(boss),
            ),
            assistant=SimpleNamespace(about="Assistant bio", name="Ava"),
        )

        monkeypatch.setattr(call_script, "event_broker", fake_broker)
        monkeypatch.setattr(call_script, "SESSION_DETAILS", fake_session_details)
        monkeypatch.setattr(call_script, "AgentSession", _FakeSession)
        monkeypatch.setattr(call_script, "Assistant", _FakeAssistant)
        monkeypatch.setattr(call_script, "UnifyLLM", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            call_script,
            "build_voice_agent_prompt",
            lambda **kwargs: SimpleNamespace(flatten=lambda: "system prompt"),
        )
        monkeypatch.setattr(call_script, "start_event_broker_receive", _noop_async)
        monkeypatch.setattr(call_script, "publish_call_started", _noop_async)
        monkeypatch.setattr(
            call_script,
            "create_end_call",
            lambda *args, **kwargs: _noop_end_call,
        )
        monkeypatch.setattr(
            call_script,
            "setup_inactivity_timeout",
            lambda end_call: (lambda: None),
        )
        monkeypatch.setattr(
            call_script,
            "setup_participant_disconnect_handler",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(call_script, "RoomInputOptions", lambda **kwargs: object())
        monkeypatch.setattr(call_script, "EnglishModel", lambda: object())
        monkeypatch.setattr(call_script.cartesia, "TTS", lambda **kwargs: object())
        monkeypatch.setattr(call_script.elevenlabs, "TTS", lambda **kwargs: object())
        if hasattr(call_script, "noise_cancellation"):
            monkeypatch.setattr(call_script.noise_cancellation, "BVC", lambda: object())

        monkeypatch.setattr(call_script, "STT", object())
        monkeypatch.setattr(call_script, "VAD", object())

        await call_script.entrypoint(_FakeJobContext())

        guidance_cb = fake_broker.callbacks["app:call:call_guidance"]
        guidance_cb({"payload": {"content": "No, there is no contact named Bob."}})

        session = fake_session_holder["session"]
        mirror_texts = [
            item.text_content or ""
            for item in session._chat_ctx.items
            if getattr(item, "type", None) == "message"
        ]
        assert any("No, there is no contact named Bob." in txt for txt in mirror_texts)

        agent_texts = [
            item.text_content or ""
            for item in session.current_agent._chat_ctx.items
            if getattr(item, "type", None) == "message"
        ]
        assert any("No, there is no contact named Bob." in txt for txt in agent_texts)

    async def test_unify_llm_preserves_base_system_prompt_with_notification(
        self,
        monkeypatch,
    ):
        """System instructions and notifications should both survive conversion."""
        from livekit.agents import llm
        from unity.conversation_manager import livekit_unify_adapter as adapter_module

        captured = {}

        class _DummyClient:
            def set_stream(self, enabled):
                captured["set_stream"] = enabled

            async def generate(self, **kwargs):
                captured["generate_kwargs"] = kwargs

                async def _empty_stream():
                    if False:
                        yield ""

                return _empty_stream()

        def _fake_new_llm_client(model, **kwargs):
            captured["client_model"] = model
            captured["client_kwargs"] = kwargs
            return _DummyClient()

        monkeypatch.setattr(adapter_module, "new_llm_client", _fake_new_llm_client)

        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content="BASE_PROMPT")
        chat_ctx.add_message(
            role="user",
            content="Do I have a contact named Bob?",
        )
        chat_ctx.add_message(
            role="system",
            content="[notification] No, there is no contact named Bob.",
        )
        chat_ctx.add_message(role="assistant", content="Let me check on that.")

        stream = adapter_module.UnifyLLM(model="gpt-5-mini@openai").chat(
            chat_ctx=chat_ctx,
        )
        await stream._run()

        sent_system_message = captured["generate_kwargs"]["system_message"]
        assert "BASE_PROMPT" in sent_system_message
        assert (
            "[notification] No, there is no contact named Bob." in sent_system_message
        )
