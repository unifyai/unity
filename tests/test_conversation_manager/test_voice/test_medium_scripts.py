"""
tests/test_conversation_manager/test_medium_scripts.py
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
   - log_sts_usage billing heuristic

3. **Voice Agent prompt**:
   - build_voice_agent_prompt output structure
"""

import asyncio
import json

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

    def test_default_inactivity_timeout_value(self):
        """DEFAULT_INACTIVITY_TIMEOUT is 5 minutes (300 seconds)."""
        from unity.conversation_manager.medium_scripts import common

        assert common.DEFAULT_INACTIVITY_TIMEOUT == 300

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


class TestSTSUsageLogging:
    """Tests for STS usage logging and billing heuristics."""

    def test_log_sts_usage_skips_zero_duration(self, caplog):
        """log_sts_usage logs warning and skips for duration <= 0."""
        from unity.conversation_manager.medium_scripts.common import log_sts_usage

        log_sts_usage(call_duration_seconds=0)

        assert "Skipping STS usage logging" in caplog.text

    def test_log_sts_usage_skips_negative_duration(self, caplog):
        """log_sts_usage logs warning and skips for negative duration."""
        from unity.conversation_manager.medium_scripts.common import log_sts_usage

        log_sts_usage(call_duration_seconds=-10)

        assert "Skipping STS usage logging" in caplog.text

    def test_sts_billing_constants(self):
        """STS billing constants have expected values."""
        from unity.conversation_manager.medium_scripts.common import (
            _STS_BILLING_MODEL,
            _STS_TOKENS_PER_SECOND,
            _STS_SPEECH_RATIO,
            _STS_INPUT_OUTPUT_SPLIT,
        )

        assert _STS_BILLING_MODEL == "gpt-4o-realtime-preview"
        assert _STS_TOKENS_PER_SECOND == 150
        assert _STS_SPEECH_RATIO == 0.5
        assert _STS_INPUT_OUTPUT_SPLIT == 0.5


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
# Unit Tests: Voice Agent Prompt Building
# =============================================================================


class TestVoiceAgentPromptBuilding:
    """Tests for build_voice_agent_prompt function."""

    def test_prompt_contains_role_section(self, boss_contact):
        """Voice agent prompt contains role section."""
        from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

        prompt = build_voice_agent_prompt(
            bio="Test assistant",
            boss_first_name=boss_contact["first_name"],
            boss_surname=boss_contact["surname"],
            boss_phone_number=boss_contact["phone_number"],
            boss_email_address=boss_contact["email_address"],
            is_boss_user=True,
        )

        # Now uses Markdown headers
        assert "Role\n----" in prompt

    def test_prompt_contains_bio(self, boss_contact):
        """Voice agent prompt includes assistant bio."""
        from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

        prompt = build_voice_agent_prompt(
            bio="I am a helpful test assistant",
            boss_first_name=boss_contact["first_name"],
            boss_surname=boss_contact["surname"],
            is_boss_user=True,
        )

        assert "I am a helpful test assistant" in prompt

    def test_prompt_contains_boss_details(self, boss_contact):
        """Voice agent prompt includes boss details."""
        from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

        prompt = build_voice_agent_prompt(
            bio="Test assistant",
            boss_first_name="John",
            boss_surname="Smith",
            boss_phone_number="+15551234567",
            boss_email_address="john@test.com",
            is_boss_user=True,
        )

        # Now uses Markdown headers
        assert "Boss details\n------------" in prompt
        assert "John" in prompt
        assert "Smith" in prompt
        assert "+15551234567" in prompt
        assert "john@test.com" in prompt

    def test_prompt_for_non_boss_call_includes_contact_details(
        self,
        boss_contact,
        external_contact,
    ):
        """Non-boss calls include contact_details section."""
        from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

        prompt = build_voice_agent_prompt(
            bio="Test assistant",
            boss_first_name=boss_contact["first_name"],
            boss_surname=boss_contact["surname"],
            contact_first_name=external_contact["first_name"],
            contact_surname=external_contact["surname"],
            contact_phone_number=external_contact["phone_number"],
            contact_email=external_contact["email_address"],
            is_boss_user=False,
        )

        # Now uses Markdown headers
        assert "Contact details\n---------------" in prompt
        assert external_contact["first_name"] in prompt
        assert external_contact["surname"] in prompt

    def test_prompt_for_boss_call_excludes_contact_details(self, boss_contact):
        """Boss calls do not include contact_details section."""
        from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

        prompt = build_voice_agent_prompt(
            bio="Test assistant",
            boss_first_name=boss_contact["first_name"],
            boss_surname=boss_contact["surname"],
            is_boss_user=True,
        )

        # Now uses Markdown headers - section should not exist for boss calls
        assert "Contact details\n---------------" not in prompt

    def test_prompt_contains_conversation_manager_section(self, boss_contact):
        """Voice agent prompt explains conversation manager interaction."""
        from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

        prompt = build_voice_agent_prompt(
            bio="Test assistant",
            boss_first_name=boss_contact["first_name"],
            boss_surname=boss_contact["surname"],
            is_boss_user=True,
        )

        # Now uses Markdown headers
        assert "Conversation manager\n--------------------" in prompt
        assert "notification" in prompt.lower()

    def test_prompt_contains_communication_guidelines(self, boss_contact):
        """Voice agent prompt includes communication guidelines."""
        from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

        prompt = build_voice_agent_prompt(
            bio="Test assistant",
            boss_first_name=boss_contact["first_name"],
            boss_surname=boss_contact["surname"],
            is_boss_user=True,
        )

        # Now uses Markdown headers
        assert "Communication guidelines\n------------------------" in prompt


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
# Unit Tests: Module Structure
# =============================================================================


class TestModuleStructure:
    """Tests for medium script module structure."""

    def test_call_module_has_entrypoint(self):
        """call.py has an entrypoint function."""
        from unity.conversation_manager.medium_scripts import call as call_module

        assert hasattr(call_module, "entrypoint")
        assert asyncio.iscoroutinefunction(call_module.entrypoint)

    def test_call_module_has_prewarm(self):
        """call.py has a prewarm function for heavy initialization."""
        from unity.conversation_manager.medium_scripts import call as call_module

        assert hasattr(call_module, "prewarm")
        assert callable(call_module.prewarm)

    def test_sts_module_has_entrypoint(self):
        """sts_call.py has an entrypoint function."""
        from unity.conversation_manager.medium_scripts import sts_call as sts_module

        assert hasattr(sts_module, "entrypoint")
        assert asyncio.iscoroutinefunction(sts_module.entrypoint)


# =============================================================================
# Unit Tests: LiveKit Plugin Preloading
# =============================================================================


class TestLiveKitPluginPreloading:
    """Tests for LiveKit plugin preloading in call_manager.py.

    LiveKit requires plugins to be registered on the main thread, but when
    running in dev mode the voice agent script runs in a background thread.
    call_manager.py preloads these plugins on import to ensure registration
    happens before the thread is spawned.
    """

    def test_call_manager_preloads_openai_realtime_plugin(self):
        """call_manager.py preloads OpenAI Realtime plugin for STS mode."""
        import sys

        # Import call_manager (this triggers preloading)
        from unity.conversation_manager.domains import (
            call_manager as _call_manager_import,
        )  # noqa: F401

        # The OpenAI realtime plugin should be in sys.modules after import
        assert "livekit.plugins.openai.realtime" in sys.modules, (
            "OpenAI Realtime plugin not preloaded. "
            "call_manager.py must import livekit.plugins.openai.realtime at module level."
        )

    def test_call_manager_preloads_tts_plugins(self):
        """call_manager.py preloads TTS plugins (deepgram, elevenlabs, cartesia, silero)."""
        import sys

        # Import call_manager (this triggers preloading)
        from unity.conversation_manager.domains import (
            call_manager as _call_manager_import,
        )  # noqa: F401

        # Verify TTS plugins are actually loaded in sys.modules
        tts_plugins = [
            "livekit.plugins.cartesia",
            "livekit.plugins.deepgram",
            "livekit.plugins.elevenlabs",
            "livekit.plugins.silero",
        ]

        missing_plugins = [p for p in tts_plugins if p not in sys.modules]

        assert len(missing_plugins) == 0, (
            f"TTS plugins not preloaded: {missing_plugins}. "
            "call_manager.py must import these plugins at module level."
        )

    def test_call_manager_preloads_noise_cancellation_on_darwin(self):
        """call_manager.py preloads noise_cancellation plugin on macOS."""
        import sys

        # Import call_manager (this triggers preloading)
        from unity.conversation_manager.domains import (
            call_manager as _call_manager_import,
        )  # noqa: F401

        # On macOS, verify it's actually loaded
        if sys.platform == "darwin":
            assert "livekit.plugins.noise_cancellation" in sys.modules, (
                "noise_cancellation plugin not preloaded on macOS. "
                "call_manager.py must import this plugin at module level."
            )

    def test_call_manager_preloads_turn_detector(self):
        """call_manager.py preloads turn_detector.english plugin."""
        import sys

        # Import call_manager (this triggers preloading)
        from unity.conversation_manager.domains import (
            call_manager as _call_manager_import,
        )  # noqa: F401

        # Verify turn detector is actually loaded
        assert "livekit.plugins.turn_detector.english" in sys.modules, (
            "Turn detector plugin not preloaded. "
            "call_manager.py must import livekit.plugins.turn_detector.english at module level."
        )
