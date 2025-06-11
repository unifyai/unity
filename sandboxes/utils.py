"""utils.py
Shared voice‑mode helpers for sandbox scripts: audio capture, Deepgram STT
and Cartesia TTS.  Extracted from the original sandbox implementations so
both transcript_sandbox.py and tasklist_sandbox.py can import them.
"""

from __future__ import annotations

import asyncio
import os
import platform
import select
import threading
import aiohttp
import sys
import time
import wave
from contextlib import contextmanager
from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
from typing import List, Optional, Tuple, Any, Coroutine
import pyaudio
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from livekit.plugins import cartesia

from dotenv import load_dotenv

# Import platform-specific modules for non-blocking input
if platform.system() == "Windows":
    import msvcrt

load_dotenv()


# ---------------------------------------------------------------------------
# Audio / PortAudio boilerplate
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
MAX_SCENARIO_LENGTH = 2048

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)


def _py_error_handler(
    filename,
    line,
    function,
    err,
    fmt,
):  # noqa: D401 – C callback sig
    pass


c_error_handler = ERROR_HANDLER_FUNC(_py_error_handler)


@contextmanager
def noalsaerr():
    "Temporarily suppress ALSA warnings (common on Linux CI containers)."
    try:
        asound = cdll.LoadLibrary("libasound.so")
        asound.snd_lib_error_set_handler(c_error_handler)
        yield
        asound.snd_lib_error_set_handler(None)
    except Exception:
        yield


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def record_until_enter() -> bytes:
    "Record audio between two ENTER presses and return WAV bytes."
    with noalsaerr():
        pa = pyaudio.PyAudio()

    frames: List[bytes] = []
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    def _capture():
        while not stop.is_set():
            frames.append(stream.read(CHUNK, exception_on_overflow=False))

    stop = threading.Event()
    thr = threading.Thread(target=_capture, daemon=True)

    input("\nPress ↵ to start recording…")
    thr.start()
    input("🎙️  Recording… press ↵ again to stop.")
    stop.set()
    thr.join()

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("✅ Recording captured.")

    wav_path = "/tmp/voice_input.wav"
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    with open(wav_path, "rb") as f:
        return f.read()


def transcribe_deepgram(audio_bytes: bytes) -> str:
    "Send *audio_bytes* to Deepgram SDK v4 and return the transcript."
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        print("[Voice] Deepgram key missing – fallback to CLI input.")
        return input("> ")

    dg = DeepgramClient(api_key=key)
    payload: FileSource = {"buffer": audio_bytes}
    opts = PrerecordedOptions(model="nova-3", smart_format=True, punctuate=True)

    try:
        response = dg.listen.rest.v("1").transcribe_file(payload, opts)
        return response.results.channels[0].alternatives[0].transcript.strip()
    except Exception as exc:
        print(f"[Voice] Deepgram error ({exc}) – fallback to CLI input.")
        return input("> ")


async def _speak_async(text: str) -> None:
    """Real async implementation – **must be run inside an event-loop**."""
    if "CARTESIA_API_KEY" not in os.environ:
        return

    print("🗣️ Assistant speaking… press ↵ to skip.")

    async def _gen() -> bytes:
        async with aiohttp.ClientSession() as s:
            tts = cartesia.TTS(http_session=s)
            frame = await tts.synthesize(text).collect()

            if hasattr(frame, "to_pcm_bytes"):  # newest SDK
                return frame.to_pcm_bytes()
            if hasattr(frame, "data"):  # older SDK
                return bytes(frame.data)
            if hasattr(frame, "to_wav_bytes"):  # WAV fallback → strip header
                return frame.to_wav_bytes()[44:]
            return bytes(frame)  # last-resort

    pcm = await _gen()

    if not pcm:
        return

    stop = threading.Event()

    def _wait_enter():
        sys.stdin.readline()
        stop.set()

    threading.Thread(target=_wait_enter, daemon=True).start()

    with noalsaerr():
        pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True,
    )

    chunk = 4800  # ≈0.1 s of audio
    i = 0
    while i < len(pcm) and not stop.is_set():
        stream.write(pcm[i : i + chunk])
        i += chunk

    stream.stop_stream()
    stream.close()
    pa.terminate()

    if stop.is_set():  # flush the newline the user pressed
        try:
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass


# ────────────────────────────── public shim ────────────────────────────────
def speak(text: str) -> None:
    """
    Safe *sync* wrapper around :pyfunc:`_speak_async`.

    • If *no* event-loop is running (classic script) → `asyncio.run`.
    • If we’re already **inside** a running loop (your sandbox) → schedule the
      coroutine with `loop.create_task` and return immediately (non-blocking).
    """
    try:
        loop = asyncio.get_running_loop()
        # we're inside an active loop → fire-and-forget
        loop.create_task(_speak_async(text))
    except RuntimeError:
        # no loop → normal synchronous context
        asyncio.run(_speak_async(text))


def input_with_timeout(timeout: float = 0.1) -> Tuple[bool, Optional[str]]:
    """Check for user input with a timeout, without blocking execution.

    This function allows sandboxes to poll for user input while waiting for
    async operations to complete, enabling interruption of long-running tasks.

    Args:
        timeout: Maximum time to wait for input in seconds (default: 0.1)

    Returns:
        Tuple of (has_input, input_value):
            - has_input: True if user provided input, False if timeout occurred
            - input_value: The string input if has_input is True, None otherwise

    Example usage in sandboxes:
        # Create and start the async operation
        handle = manager.ask(question)
        result_task = asyncio.create_task(handle.result())

        # Poll for user input while waiting for result
        while not result_task.done():
            has_input, text = input_with_timeout(0.1)
            if has_input:
                # User wants to interrupt
                await handle.interject(text)
            await asyncio.sleep(0.1)

        # Get the final result
        final_answer = await result_task
    """
    if platform.system() == "Windows":
        # Windows implementation using msvcrt
        start_time = time.time()
        input_chars = []

        while time.time() - start_time < timeout:
            if msvcrt.kbhit():
                char = msvcrt.getche().decode("utf-8")
                if char == "\r":  # Enter key
                    print()  # Move to next line after Enter
                    return True, "".join(input_chars)
                input_chars.append(char)

            time.sleep(0.01)  # Small sleep to prevent CPU hogging

        return False, None
    else:
        # Unix implementation using select
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return True, sys.stdin.readline().strip()
        return False, None


def get_custom_scenario(args) -> Optional[str]:
    """Get custom scenario from args, either text or voice input.

    Args:
        args: Parsed command line arguments with custom_scenario and custom_scenario_voice

    Returns:
        Custom scenario string if provided/captured, None otherwise
    """
    custom_scenario_flag = hasattr(args, "custom_scenario") and args.custom_scenario
    if not custom_scenario_flag:
        return
    voice_flag = hasattr(args, "voice") and args.voice
    # Check for text-based custom scenario first
    if not voice_flag:
        print(f"🧮 Let's build your custom scenario:\n{args.custom_scenario}")
        return args.custom_scenario

    try:
        # Record and transcribe audio
        print("🧮 Let's build your custom scenario using voice")
        audio_bytes = record_until_enter()
        transcript = transcribe_deepgram(audio_bytes)

        # Handle empty or failed transcription
        if not transcript or transcript.strip() == "":
            print("⚠️ Warning: No transcript received from voice input")
            return None

        # Truncate if too long
        if len(transcript) > MAX_SCENARIO_LENGTH:
            transcript = transcript[: MAX_SCENARIO_LENGTH - 3] + "..."
            print(
                f"⚠️ Warning: Scenario truncated to {MAX_SCENARIO_LENGTH} characters",
            )

        return transcript.strip()

    except Exception as exc:
        print(f"⚠️ Warning: Voice scenario capture failed ({exc})")
        return None


# ---------------------------------------------------------------------------
# Thread-safe helper to schedule coroutines from background threads
# ---------------------------------------------------------------------------

_MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _get_main_loop() -> asyncio.AbstractEventLoop:
    """
    Return the main asyncio loop.  If called from a background thread
    where ``asyncio.get_running_loop()`` fails, fall back to the loop that
    was running when this module was first imported.
    """
    global _MAIN_LOOP
    try:
        # We are already inside the loop’s thread
        return asyncio.get_running_loop()
    except RuntimeError:
        # Background thread – re-use cached loop
        if _MAIN_LOOP is None or _MAIN_LOOP.is_closed():
            _MAIN_LOOP = asyncio.get_event_loop_policy().get_event_loop()
        return _MAIN_LOOP


def run_in_loop(coro: Coroutine[Any, Any, Any]):
    """
    Schedule *coro* on the main event-loop from **any** thread.

    * If we are on the loop thread → just ``asyncio.create_task``.
    * Otherwise → ``asyncio.run_coroutine_threadsafe``.
    """
    loop = _get_main_loop()

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is loop:  # same thread
        return asyncio.create_task(coro)

    # another thread
    return asyncio.run_coroutine_threadsafe(coro, loop)
