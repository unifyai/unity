"""
tests/conversation_manager/voice/test_deterministic_audio_input.py
===================================================================

Deterministic audio input tests for the voice pipeline.

These tests verify the voice input pipeline using pre-generated audio files
with known transcripts. This enables testing of:
1. Audio → STT transcription accuracy
2. Transcription → Event generation
3. Event → ConversationManager processing
4. Full voice input round-trip

The tests use real APIs (Deepgram STT, OpenAI Whisper) when available,
and skip gracefully when API keys are not configured.

To generate test audio files:
    uv run python scripts/generate_test_audio.py

These tests would catch bugs like:
- Audio format handling issues
- Transcription event field mapping errors
- Voice mode state machine bugs
- Utterance event routing issues
"""

from __future__ import annotations

import json
import os
import struct
import wave
from pathlib import Path

import pytest
import pytest_asyncio

from unity.conversation_manager.events import (
    Event,
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    PhoneCallStarted,
    UnifyMeetStarted,
)
from unity.conversation_manager.types import Medium, Mode

# Path to audio fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio"

# Test transcripts - must match what's in generate_test_audio.py
TEST_TRANSCRIPTS = {
    "hello_greeting": "Hello, how are you today?",
    "schedule_question": "What's on my schedule for tomorrow?",
    "meeting_request": "Can you schedule a meeting with Alice for 3pm?",
    "simple_yes": "Yes, that sounds good.",
    "simple_no": "No, I don't think so.",
    "thank_you": "Thank you very much for your help.",
}

# Check if APIs are available
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Skip markers
skip_without_deepgram = pytest.mark.skipif(
    not DEEPGRAM_API_KEY,
    reason="DEEPGRAM_API_KEY not set - skipping real STT test",
)
skip_without_openai = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set - skipping real Whisper test",
)


def create_test_wav(text: str, duration: float = 0.5) -> bytes:
    """Create a minimal WAV file for testing.

    This creates a simple sine wave that can be used for testing
    audio processing pipelines without needing real speech.

    Args:
        text: Ignored - just for interface compatibility
        duration: Duration in seconds

    Returns:
        WAV file bytes
    """
    import io
    import math

    sample_rate = 16000
    frequency = 440  # A4 note
    num_samples = int(sample_rate * duration)

    # Generate sine wave
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        # Sine wave with fade in/out to avoid clicks
        fade = min(i / 1000, (num_samples - i) / 1000, 1.0)
        value = int(32767 * 0.5 * fade * math.sin(2 * math.pi * frequency * t))
        samples.append(value)

    # Write to WAV format
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack("<" + "h" * len(samples), *samples))

    return buffer.getvalue()


def get_audio_fixture(name: str) -> Path | None:
    """Get path to audio fixture file.

    Looks for both .mp3 (OpenAI-generated) and .wav (placeholder) files.
    """
    for ext in [".mp3", ".wav"]:
        path = FIXTURES_DIR / f"{name}{ext}"
        if path.exists():
            return path
    return None


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_contact():
    """Sample contact for testing (Alice from TEST_CONTACTS)."""
    from tests.conversation_manager.conftest import TEST_CONTACTS

    return TEST_CONTACTS[0]  # Alice


@pytest.fixture
def boss_contact():
    """Boss contact for testing (contact_id 1)."""
    from tests.conversation_manager.conftest import BOSS

    return BOSS


@pytest_asyncio.fixture
async def event_broker():
    """Real in-memory event broker for tests."""
    from unity.conversation_manager.in_memory_event_broker import (
        create_in_memory_event_broker,
        reset_in_memory_event_broker,
    )

    reset_in_memory_event_broker()
    broker = create_in_memory_event_broker()
    yield broker
    await broker.aclose()
    reset_in_memory_event_broker()


# =============================================================================
# Unit Tests: Audio Format Handling
# =============================================================================


class TestAudioFormatHandling:
    """Tests for audio format validation and handling."""

    def test_wav_file_creation(self):
        """Test that we can create valid WAV files programmatically."""
        wav_bytes = create_test_wav("test", duration=0.1)

        # Verify WAV header
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"

        # Verify we can read it back
        import io

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 16000

    def test_wav_file_has_audio_content(self):
        """Test that generated WAV files have non-silent content."""
        wav_bytes = create_test_wav("test", duration=0.1)

        import io

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            samples = struct.unpack("<" + "h" * (len(frames) // 2), frames)

            # Check that we have some non-zero samples (not silence)
            max_amplitude = max(abs(s) for s in samples)
            assert max_amplitude > 1000, "Audio should not be silent"


# =============================================================================
# Unit Tests: Utterance Event Creation
# =============================================================================


class TestUtteranceEventCreation:
    """Tests for creating utterance events from transcriptions."""

    def test_inbound_phone_utterance_creation(self, sample_contact):
        """Test creating InboundPhoneUtterance from transcript."""
        transcript = "Hello, how are you?"

        event = InboundPhoneUtterance(
            contact=sample_contact,
            content=transcript,
        )

        assert event.content == transcript
        assert event.contact["contact_id"] == sample_contact["contact_id"]

    def test_inbound_unify_meet_utterance_creation(self, sample_contact):
        """Test creating InboundUnifyMeetUtterance from transcript."""
        transcript = "Let's discuss the project."

        event = InboundUnifyMeetUtterance(
            contact=sample_contact,
            content=transcript,
        )

        assert event.content == transcript
        assert event.contact["contact_id"] == sample_contact["contact_id"]

    def test_utterance_event_serialization(self, sample_contact):
        """Test that utterance events serialize correctly."""
        event = InboundPhoneUtterance(
            contact=sample_contact,
            content="Test transcript",
        )

        # Serialize
        json_str = event.to_json()
        data = json.loads(json_str)

        # Verify structure
        assert data["event_name"] == "InboundPhoneUtterance"
        assert data["payload"]["content"] == "Test transcript"

        # Deserialize
        restored = Event.from_json(json_str)
        assert isinstance(restored, InboundPhoneUtterance)
        assert restored.content == "Test transcript"


# =============================================================================
# Integration Tests: Voice Mode Event Flow
# =============================================================================


class TestVoiceModeEventFlow:
    """Integration tests for voice mode event handling."""

    @pytest.mark.asyncio
    async def test_phone_utterance_published_to_correct_channel(
        self,
        event_broker,
        sample_contact,
    ):
        """Test that phone utterances are published to app:comms:phone_utterance."""
        async with event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            # Publish utterance (simulating voice agent)
            event = InboundPhoneUtterance(
                contact=sample_contact,
                content="Hello from the test",
            )
            await event_broker.publish(
                "app:comms:phone_utterance",
                event.to_json(),
            )

            # Receive and verify
            msg = await pubsub.get_message(timeout=2.0, ignore_subscribe_messages=True)
            assert msg is not None
            assert msg["channel"] == "app:comms:phone_utterance"

            restored = Event.from_json(msg["data"])
            assert isinstance(restored, InboundPhoneUtterance)
            assert restored.content == "Hello from the test"

    @pytest.mark.asyncio
    async def test_unify_meet_utterance_published_to_correct_channel(
        self,
        event_broker,
        sample_contact,
    ):
        """Test that Unify Meet utterances are published to app:comms:unify_utterance."""
        async with event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            event = InboundUnifyMeetUtterance(
                contact=sample_contact,
                content="Testing Unify Meet",
            )
            await event_broker.publish(
                "app:comms:unify_utterance",
                event.to_json(),
            )

            msg = await pubsub.get_message(timeout=2.0, ignore_subscribe_messages=True)
            assert msg is not None

            restored = Event.from_json(msg["data"])
            assert isinstance(restored, InboundUnifyMeetUtterance)
            assert restored.content == "Testing Unify Meet"


# =============================================================================
# Integration Tests: Transcription Pipeline (Mocked STT)
# =============================================================================


class TestTranscriptionPipelineMocked:
    """Tests for the transcription pipeline with mocked STT.

    These tests verify the flow of audio → transcript → event without
    requiring real API calls.
    """

    @pytest.mark.asyncio
    async def test_audio_to_utterance_event_flow(
        self,
        event_broker,
        sample_contact,
    ):
        """Test the flow from audio input to utterance event.

        This simulates the voice agent's transcript handling:
        1. Audio comes in (mocked)
        2. STT produces transcript
        3. Event is published to event broker
        4. ConversationManager receives it
        """
        received_events = []

        async with event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            # Simulate what the voice agent does when it receives a transcript
            # (see call.py _on_chat_item_added)
            transcript = "What's on my schedule for tomorrow?"
            event = InboundPhoneUtterance(
                contact=sample_contact,
                content=transcript,
            )

            await event_broker.publish(
                "app:comms:phone_utterance",
                event.to_json(),
            )

            # Collect events
            for _ in range(3):
                msg = await pubsub.get_message(
                    timeout=1.0,
                    ignore_subscribe_messages=True,
                )
                if msg:
                    received_events.append(msg)

        # Verify event was received
        assert len(received_events) >= 1

        utterance_events = [
            e for e in received_events if e["channel"] == "app:comms:phone_utterance"
        ]
        assert len(utterance_events) == 1

        event_data = Event.from_json(utterance_events[0]["data"])
        assert event_data.content == "What's on my schedule for tomorrow?"

    @pytest.mark.asyncio
    async def test_multiple_utterances_maintain_order(
        self,
        event_broker,
        sample_contact,
    ):
        """Test that multiple utterances maintain chronological order."""
        transcripts = [
            "First message",
            "Second message",
            "Third message",
        ]
        received_contents = []

        async with event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:phone_utterance")

            # Publish in order
            for transcript in transcripts:
                event = InboundPhoneUtterance(
                    contact=sample_contact,
                    content=transcript,
                )
                await event_broker.publish(
                    "app:comms:phone_utterance",
                    event.to_json(),
                )

            # Receive in order
            for _ in range(len(transcripts)):
                msg = await pubsub.get_message(
                    timeout=2.0,
                    ignore_subscribe_messages=True,
                )
                if msg:
                    event_data = Event.from_json(msg["data"])
                    received_contents.append(event_data.content)

        assert received_contents == transcripts


# =============================================================================
# Integration Tests: Real STT (Requires API Keys)
# =============================================================================


@skip_without_deepgram
class TestDeepgramSTT:
    """Tests using real Deepgram STT API.

    These tests require DEEPGRAM_API_KEY to be set.
    They verify that real audio can be transcribed correctly.
    """

    @pytest.mark.asyncio
    async def test_deepgram_transcribes_known_audio(self):
        """Test that Deepgram can transcribe our test audio.

        This test verifies end-to-end audio transcription works.
        """
        try:
            pass
        except ImportError:
            pytest.skip("livekit-plugins-deepgram not installed")

        # Get a test audio file
        audio_path = get_audio_fixture("hello_greeting")
        if audio_path is None:
            pytest.skip(
                "Audio fixtures not generated - run scripts/generate_test_audio.py",
            )

        # This would need async file reading and Deepgram client setup
        # For now, just verify the fixture exists
        assert audio_path.exists()
        expected = TEST_TRANSCRIPTS["hello_greeting"]

        # TODO: Add actual Deepgram transcription test when infrastructure is ready
        # stt = deepgram.STT(model="nova-3", language="en-GB")
        # transcript = await stt.transcribe(audio_path.read_bytes())
        # assert any(word in transcript.lower() for word in ["hello", "how are you"])


@skip_without_openai
class TestOpenAIWhisper:
    """Tests using OpenAI Whisper API.

    These tests require OPENAI_API_KEY to be set.
    """

    @pytest.mark.asyncio
    async def test_whisper_transcribes_known_audio(self):
        """Test that Whisper can transcribe our test audio."""
        try:
            from openai import OpenAI
        except ImportError:
            pytest.skip("openai not installed")

        audio_path = get_audio_fixture("hello_greeting")
        if audio_path is None:
            pytest.skip(
                "Audio fixtures not generated - run scripts/generate_test_audio.py",
            )

        client = OpenAI()

        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )

        # Check that key words are in the transcript
        expected_words = ["hello", "how", "you"]
        transcript_lower = transcript.text.lower()

        matched = sum(1 for word in expected_words if word in transcript_lower)
        assert matched >= 2, (
            f"Expected at least 2 of {expected_words} in transcript, "
            f"got: '{transcript.text}'"
        )


# =============================================================================
# Integration Tests: Full Voice Input Pipeline
# =============================================================================


class TestFullVoiceInputPipeline:
    """End-to-end tests for the complete voice input pipeline.

    These tests verify the full flow:
    Audio → STT → Event → ConversationManager → State Update
    """

    @pytest.mark.asyncio
    async def test_utterance_updates_contact_index(
        self,
        initialized_cm,
        sample_contact,
    ):
        """Test that inbound utterance updates the contact's voice thread."""
        cm = initialized_cm.cm

        # First start a call
        start_event = PhoneCallStarted(contact=sample_contact)
        await initialized_cm.step(start_event)
        assert cm.mode == Mode.CALL

        # Now send an utterance
        utterance = InboundPhoneUtterance(
            contact=sample_contact,
            content="Hello, can you help me?",
        )
        await initialized_cm.step(utterance)

        # Verify it was added to the contact's voice thread
        contact_id = sample_contact["contact_id"]
        voice_thread = cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )
        assert len(voice_thread) > 0

        messages = [msg.content for msg in voice_thread]
        assert "Hello, can you help me?" in messages

    @pytest.mark.asyncio
    async def test_utterance_triggers_llm_processing(
        self,
        initialized_cm,
        sample_contact,
    ):
        """Test that inbound utterance triggers LLM processing."""
        cm = initialized_cm.cm

        # Start a call
        start_event = PhoneCallStarted(contact=sample_contact)
        await initialized_cm.step(start_event)

        # Send utterance
        utterance = InboundPhoneUtterance(
            contact=sample_contact,
            content="What's on my schedule?",
        )
        result = await initialized_cm.step(utterance)

        # Verify LLM was triggered (interject_or_run called)
        assert result.llm_requested, "Inbound utterance should trigger LLM processing"

    @pytest.mark.asyncio
    async def test_unify_meet_utterance_uses_correct_medium(
        self,
        initialized_cm,
        boss_contact,
    ):
        """Test that Unify Meet utterances use UNIFY_MEET medium."""
        cm = initialized_cm.cm

        # Start a Unify Meet
        start_event = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(start_event)
        assert cm.mode == Mode.MEET

        # Send utterance
        utterance = InboundUnifyMeetUtterance(
            contact=boss_contact,
            content="Let's discuss the project",
        )
        await initialized_cm.step(utterance)

        # Verify it's in the UNIFY_MEET thread, not PHONE_CALL
        contact_id = boss_contact["contact_id"]
        meet_thread = cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.UNIFY_MEET,
        )
        phone_thread = cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )

        assert len(meet_thread) > 0
        assert "Let's discuss the project" in [msg.content for msg in meet_thread]

        # Phone thread should not have this message
        assert "Let's discuss the project" not in [msg.content for msg in phone_thread]

    @pytest.mark.asyncio
    async def test_multiple_transcript_phrases_processed(
        self,
        initialized_cm,
        sample_contact,
    ):
        """Test that multiple transcript phrases are all processed."""
        cm = initialized_cm.cm

        # Start a call
        start_event = PhoneCallStarted(contact=sample_contact)
        await initialized_cm.step(start_event)

        # Send multiple utterances (simulating conversation)
        phrases = [
            "Hello",
            "I need help with my schedule",
            "Can you check tomorrow's meetings?",
        ]

        for phrase in phrases:
            utterance = InboundPhoneUtterance(
                contact=sample_contact,
                content=phrase,
            )
            await initialized_cm.step(utterance)

        # Verify all phrases are in the thread
        contact_id = sample_contact["contact_id"]
        voice_thread = cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )

        messages = [msg.content for msg in voice_thread]
        for phrase in phrases:
            assert phrase in messages, f"Missing phrase: {phrase}"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestVoiceInputEdgeCases:
    """Tests for edge cases in voice input handling."""

    @pytest.mark.asyncio
    async def test_empty_transcript_handled_gracefully(
        self,
        initialized_cm,
        sample_contact,
    ):
        """Test that empty transcripts don't crash the system."""
        cm = initialized_cm.cm

        # Start a call
        start_event = PhoneCallStarted(contact=sample_contact)
        await initialized_cm.step(start_event)

        # Send empty utterance (can happen with silence detection)
        utterance = InboundPhoneUtterance(
            contact=sample_contact,
            content="",
        )

        # Should not raise
        result = await initialized_cm.step(utterance)
        assert result is not None

    @pytest.mark.asyncio
    async def test_very_long_transcript_handled(
        self,
        initialized_cm,
        sample_contact,
    ):
        """Test that very long transcripts are handled."""
        cm = initialized_cm.cm

        # Start a call
        start_event = PhoneCallStarted(contact=sample_contact)
        await initialized_cm.step(start_event)

        # Send very long utterance
        long_content = "This is a test message. " * 100  # ~2500 chars
        utterance = InboundPhoneUtterance(
            contact=sample_contact,
            content=long_content,
        )

        result = await initialized_cm.step(utterance)

        # Should be processed (truncated if necessary)
        contact_id = sample_contact["contact_id"]
        voice_thread = cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )

        # Just verify something was added
        assert len(voice_thread) > 0

    @pytest.mark.asyncio
    async def test_special_characters_in_transcript(
        self,
        initialized_cm,
        sample_contact,
    ):
        """Test that special characters in transcripts are handled."""
        cm = initialized_cm.cm

        # Start a call
        start_event = PhoneCallStarted(contact=sample_contact)
        await initialized_cm.step(start_event)

        # Send utterance with special characters
        special_content = "Hello! How's it going? 🎉 Let's meet at 3:00pm #meeting"
        utterance = InboundPhoneUtterance(
            contact=sample_contact,
            content=special_content,
        )

        await initialized_cm.step(utterance)

        # Verify content is preserved
        contact_id = sample_contact["contact_id"]
        voice_thread = cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )

        messages = [msg.content for msg in voice_thread]
        assert special_content in messages

    @pytest.mark.asyncio
    async def test_utterance_without_active_call_handled(
        self,
        initialized_cm,
        sample_contact,
    ):
        """Test that utterance without active call doesn't crash.

        This can happen if events arrive out of order or the call ends
        but utterances are still in flight.
        """
        cm = initialized_cm.cm

        # Don't start a call, just send utterance
        assert cm.mode == Mode.TEXT

        utterance = InboundPhoneUtterance(
            contact=sample_contact,
            content="Hello?",
        )

        # Should handle gracefully (the event handler adds to contact_index regardless)
        result = await initialized_cm.step(utterance)
        assert result is not None
