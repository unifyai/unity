#!/usr/bin/env python3
"""
One-time script to generate test audio files using OpenAI TTS.

Run this script once to generate audio fixtures for deterministic audio input tests.
The generated files are saved to tests/conversation_manager/voice/fixtures/audio/

Usage:
    uv run python scripts/generate_test_audio.py

Requires:
    - OPENAI_API_KEY environment variable set
"""

import os
from pathlib import Path

# Try OpenAI first, fall back to a simple wave file generator if not available
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Audio fixtures directory
FIXTURES_DIR = (
    Path(__file__).parent.parent / "tests/conversation_manager/voice/fixtures/audio"
)

# Test phrases with known transcripts
# Format: (filename, text, expected_transcript_keywords)
TEST_PHRASES = [
    ("hello_greeting.mp3", "Hello, how are you today?", ["hello", "how are you"]),
    (
        "schedule_question.mp3",
        "What's on my schedule for tomorrow?",
        ["schedule", "tomorrow"],
    ),
    (
        "meeting_request.mp3",
        "Can you schedule a meeting with Alice for 3pm?",
        ["schedule", "meeting", "alice"],
    ),
    ("simple_yes.mp3", "Yes, that sounds good.", ["yes", "sounds good"]),
    ("simple_no.mp3", "No, I don't think so.", ["no", "don't think"]),
    ("thank_you.mp3", "Thank you very much for your help.", ["thank you", "help"]),
]


def generate_with_openai():
    """Generate audio files using OpenAI TTS API."""
    client = OpenAI()

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    for filename, text, _ in TEST_PHRASES:
        filepath = FIXTURES_DIR / filename
        if filepath.exists():
            print(f"  Skipping {filename} (already exists)")
            continue

        print(f"  Generating {filename}: '{text}'")

        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )

        response.stream_to_file(filepath)
        print(f"  ✓ Saved {filename}")

    # Also save the metadata file
    metadata_file = FIXTURES_DIR / "transcripts.txt"
    with open(metadata_file, "w") as f:
        f.write("# Audio file transcripts for testing\n")
        f.write("# Format: filename|exact_text|keywords\n\n")
        for filename, text, keywords in TEST_PHRASES:
            f.write(f"{filename}|{text}|{','.join(keywords)}\n")

    print(f"\n✓ Metadata saved to {metadata_file}")


def generate_silent_wav():
    """Generate minimal silent WAV files as placeholders."""
    import struct
    import wave

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    for filename, text, _ in TEST_PHRASES:
        # Convert .mp3 to .wav for simplicity
        wav_filename = filename.replace(".mp3", ".wav")
        filepath = FIXTURES_DIR / wav_filename
        if filepath.exists():
            print(f"  Skipping {wav_filename} (already exists)")
            continue

        print(f"  Generating placeholder {wav_filename}")

        # Create a minimal silent WAV file (0.1 seconds of silence)
        sample_rate = 16000
        duration = 0.1
        num_samples = int(sample_rate * duration)

        with wave.open(str(filepath), "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            # Write silence
            wav_file.writeframes(
                struct.pack("<" + "h" * num_samples, *([0] * num_samples)),
            )

        print(f"  ✓ Saved {wav_filename} (silent placeholder)")

    # Save metadata
    metadata_file = FIXTURES_DIR / "transcripts.txt"
    with open(metadata_file, "w") as f:
        f.write("# Audio file transcripts for testing\n")
        f.write("# Format: filename|exact_text|keywords\n")
        f.write(
            "# Note: These are placeholder files. Run with OPENAI_API_KEY to generate real audio.\n\n",
        )
        for filename, text, keywords in TEST_PHRASES:
            wav_filename = filename.replace(".mp3", ".wav")
            f.write(f"{wav_filename}|{text}|{','.join(keywords)}\n")

    print(f"\n✓ Metadata saved to {metadata_file}")
    print("\n⚠️  Note: Generated silent placeholder files.")
    print("   Set OPENAI_API_KEY and re-run to generate real audio.")


def main():
    print("Generating test audio fixtures...")
    print(f"Output directory: {FIXTURES_DIR}\n")

    if HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
        print("Using OpenAI TTS API...")
        generate_with_openai()
    else:
        if not HAS_OPENAI:
            print("OpenAI package not installed.")
        else:
            print("OPENAI_API_KEY not set.")
        print("Generating silent placeholder WAV files instead...\n")
        generate_silent_wav()

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
