"""utils.py
Shared voice‑mode helpers for sandbox scripts: audio capture, Deepgram STT
and Cartesia TTS.  Extracted from the original sandbox implementations so
both transcript_sandbox.py and tasklist_sandbox.py can import them.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Optional GNU readline support (improves in-line editing & command history)
# ---------------------------------------------------------------------------
#
# On some Python builds the built-in ``input()`` function lacks readline
# capabilities, meaning arrow keys emit escape sequences like ``^[[D`` instead
# of moving the cursor.  Simply importing the *readline* module (or its
# platform-specific shim) activates those features globally for the current
# process.  We do this **once**, right at the top-level of ``sandboxes.utils``
# so that every sandbox script benefits without further changes.
#
# The import safely degrades on platforms where readline isn't available.

try:
    import readline  # type: ignore  # noqa: F401 – bound for side-effects only
except ModuleNotFoundError:
    # macOS / Windows or custom builds – attempt the gnureadline shim first
    try:
        import gnureadline as readline  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        # Graceful fallback – arrow keys won't be fancy but everything else works
        pass

import asyncio
import os
import platform
import select
import threading
import socket
from queue import SimpleQueue
import aiohttp
import logging
import sys
import time
from datetime import datetime
import wave
from contextlib import contextmanager
from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
from typing import List, Optional, Tuple, Any, Coroutine, cast, Dict, Literal
from av import AudioFrame
import pyaudio
import math
import struct
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from livekit.plugins import cartesia
import argparse
from unity.common.llm_helpers import SteerableToolHandle
from pydantic import BaseModel, Field

# Added for direct logging of generated messages
from unity.transcript_manager.transcript_manager import TranscriptManager

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

# ---------------------------------------------------------------------------
# Global lock to guarantee sequential TTS playback
# (prevents overlapping audio when several `speak()` calls happen in rapid succession)
# ---------------------------------------------------------------------------

_TTS_LOCK = threading.Lock()
# Track the current TTS skip event to allow external cancellation
_CURRENT_TTS_SKIP: Optional[threading.Event] = None


def _wait_for_tts_end(start_timeout: float = 0.5, poll: float = 0.05) -> None:
    """
    Block until any Cartesia-TTS playback triggered by :pyfunc:`speak` has
    completed **or** been skipped.

    1.  We first give a short grace period (*start_timeout*) for a background
        TTS thread to *acquire* ``_TTS_LOCK`` – this covers the small race
        window where :pyfunc:`speak` returned but audio hasn't started yet.
    2.  Once the lock is held we simply wait until it is released, meaning
        playback ended or the user hit ↵ to skip.
    """
    waited = 0.0
    while not _TTS_LOCK.locked() and waited < start_timeout:
        time.sleep(poll)
        waited += poll
    while _TTS_LOCK.locked():
        time.sleep(poll)


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


# New: suppress low-level stderr (e.g. JACK 'server is not running' noise)
@contextmanager
def suppress_stderr_fd():
    """Redirect the process-level stderr FD to os.devnull within the context."""
    try:
        # Duplicate original stderr (fd 2)
        saved_stderr_fd = os.dup(2)
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    except Exception:
        # Best-effort – do not crash callers if redirection fails
        yield
    finally:
        try:
            if "saved_stderr_fd" in locals():
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stderr_fd)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Simple sine-wave beeps for recording cues
# ---------------------------------------------------------------------------


def _generate_tone(
    freq: int,
    duration: float = 0.15,
    *,
    sample_rate: int = SAMPLE_RATE,
    volume: float = 0.3,
) -> bytes:
    """Return raw 16-bit PCM bytes for a mono 16-bit sine-wave tone."""
    total = int(sample_rate * duration)
    amp = int(volume * 32767)
    frames = [
        struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * n / sample_rate)))
        for n in range(total)
    ]
    return b"".join(frames)


def _beep(freq: int, duration: float = 0.15) -> None:
    """Play a sine-wave *freq* Hz tone via PortAudio (same path as TTS)."""
    pcm = _generate_tone(freq, duration)
    with noalsaerr(), suppress_stderr_fd():
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=FORMAT,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
        )
        stream.write(pcm)
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def record_until_enter() -> bytes:
    """
    Interactive voice-capture helper.

    Controls
    --------
        ↵ once    → start recording   (high-pitch beep ↑)
        ↵ again   → accept recording  (low-pitch  beep ↓)
        cancel↵   → abort & discard   (mid-pitch  beep →)
    """
    while True:
        # Ensure any prior TTS playback has finished
        _wait_for_tts_end()

        # ───────────── prompt to start ─────────────
        input("\nPress ↵ to start recording…")

        # ───────────── PortAudio set-up ─────────────
        with noalsaerr(), suppress_stderr_fd():
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            sample_size = pa.get_sample_size(FORMAT)

        frames: List[bytes] = []
        stop = threading.Event()

        def _capture():
            while not stop.is_set():
                frames.append(stream.read(CHUNK, exception_on_overflow=False))

        # ───────────── begin capture ─────────────
        _beep(1000)  # start cue
        thr = threading.Thread(target=_capture, daemon=True)
        thr.start()

        user_cmd = (
            input(
                "🎙️  Recording… press ↵ again to finish " "(or type 'c' + ↵ to abort).",
            )
            .strip()
            .lower()
        )

        # ───────────── tear-down PortAudio ─────────────
        stop.set()
        thr.join()
        with suppress_stderr_fd():
            stream.stop_stream()
            stream.close()
            pa.terminate()

        # ───────────── cancellation branch ─────────────
        if user_cmd.lower() == "c":
            _beep(750)  # mid-pitch cue
            print("🚫 Cancelled.")
            continue  # back to the top → fresh prompt

        # ───────────── normal completion ─────────────
        _beep(500)  # stop cue
        print("✅ Recording captured.")

        wav_path = "/tmp/voice_input.wav"
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(sample_size)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))

        with open(wav_path, "rb") as f:
            return f.read()


# New: interruptible variant used for in-flight steering
def record_until_enter_interruptible(is_cancelled) -> Optional[bytes]:
    """
    Like record_until_enter but cancelable via the is_cancelled() predicate.

    Returns None if cancelled before completion.
    """

    def _read_line_nonblocking(timeout: float) -> Optional[str]:
        if platform.system() == "Windows":
            start_time = time.time()
            buf: list[str] = []
            while time.time() - start_time < timeout:
                if msvcrt.kbhit():  # type: ignore[name-defined]
                    ch = msvcrt.getche().decode("utf-8")  # type: ignore[name-defined]
                    if ch == "\r":
                        print()
                        return "".join(buf)
                    buf.append(ch)
                time.sleep(0.01)
            return None
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.readline().strip()
        return None

    # Ensure TTS has finished before prompting
    _wait_for_tts_end()

    print("\nPress ↵ to start recording… (type 'c' then ↵ to cancel)")
    # Wait for start or cancellation
    while True:
        if is_cancelled():
            print("⚠️  Recording cancelled – task finished.")
            return None
        ln = _read_line_nonblocking(0.1)
        if ln is None:
            continue
        if ln.strip().lower() == "c":
            print("🚫 Cancelled.")
            return None
        # Any Enter (incl. empty) starts recording
        break

    # Set up audio
    with noalsaerr(), suppress_stderr_fd():
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        sample_size = pa.get_sample_size(FORMAT)

    frames: List[bytes] = []
    stop = threading.Event()

    def _capture():
        while not stop.is_set():
            frames.append(stream.read(CHUNK, exception_on_overflow=False))

    _beep(1000)
    thr = threading.Thread(target=_capture, daemon=True)
    thr.start()
    print("🎙️  Recording… press ↵ to finish (or 'c' + ↵ to abort).")

    # Wait for finish or cancellation
    while True:
        if is_cancelled():
            # Tear down and abort
            stop.set()
            thr.join()
            with suppress_stderr_fd():
                stream.stop_stream()
                stream.close()
                pa.terminate()
            print("⚠️  Recording cancelled – task finished.")
            return None
        ln2 = _read_line_nonblocking(0.1)
        if ln2 is None:
            continue
        if ln2.strip().lower() == "c":
            stop.set()
            thr.join()
            with suppress_stderr_fd():
                stream.stop_stream()
                stream.close()
                pa.terminate()
            _beep(750)
            print("🚫 Cancelled.")
            return None
        # Any Enter ends recording
        break

    # Tear down on normal completion
    stop.set()
    thr.join()
    with suppress_stderr_fd():
        stream.stop_stream()
        stream.close()
        pa.terminate()

    _beep(500)
    print("✅ Recording captured.")

    wav_path = "/tmp/voice_input.wav"
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_size)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    with open(wav_path, "rb") as f:
        return f.read()


def transcribe_deepgram(audio_bytes: bytes) -> str:
    "Send *audio_bytes* to Deepgram SDK v4 and return the transcript."
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
    """
    Stream-out Cartesia audio as it is generated so playback starts almost
    immediately.  ↵ while speaking still skips the rest.
    """
    if "CARTESIA_API_KEY" not in os.environ:
        return

    # ─────────────── enter-to-skip listener ────────────────
    skip = threading.Event()  # raised when user hits ↵
    listener_done = threading.Event()  # tells the listener to exit
    # expose skip globally so other parts can cancel speech immediately
    global _CURRENT_TTS_SKIP
    _CURRENT_TTS_SKIP = skip

    def _listen_enter():
        """Poll stdin so we can shut the thread down cleanly."""
        while not listener_done.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                sys.stdin.readline()
                skip.set()
                break

    listener = threading.Thread(target=_listen_enter, daemon=True)
    listener.start()

    try:
        # ─────────────── streaming TTS ────────────────
        async with aiohttp.ClientSession() as session:
            tts = cartesia.TTS(http_session=session)

            # Open the PortAudio stream once, before the first frame arrives
            with noalsaerr(), suppress_stderr_fd():
                pa = pyaudio.PyAudio()
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=tts.sample_rate,  # usually 24 kHz
                    output=True,
                )

            # PortAudio initialisation (and its interaction with JACK) tends to
            # emit noisy warnings *right here*.  Briefly pause and then print the
            # skip hint so that it appears **after** those warnings.
            await asyncio.sleep(1.0)
            print(f'🗣️ Assistant speaking…\n"{text}"')
            print("🔇 Press ↵ to skip playback")

            def _frame_to_pcm(frame: "AudioFrame") -> bytes:
                """Return raw 16-bit PCM for *any* Cartesia AudioFrame flavour."""
                if hasattr(frame, "to_pcm_bytes"):  # newest SDK
                    return frame.to_pcm_bytes()
                if hasattr(frame, "data"):  # mid-2024 builds
                    return bytes(frame.data)
                if hasattr(frame, "to_wav_bytes"):  # old fallback → strip header
                    return cast(bytes, frame.to_wav_bytes())[44:]
                return bytes(frame)  # last-resort

            async with tts.synthesize(text) as synth_stream:
                async for audio in synth_stream:  # 10-50 ms frames
                    if skip.is_set():
                        break
                    stream.write(_frame_to_pcm(audio.frame))
            with suppress_stderr_fd():
                stream.stop_stream()
                stream.close()
                pa.terminate()
    finally:
        # ─────────────── clean-up ───────────────
        listener_done.set()
        listener.join(timeout=0.1)
        # clear global skip handle now that we're done
        _CURRENT_TTS_SKIP = None

        if skip.is_set():  # flush the newline the user pressed
            try:
                import termios

                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            except Exception:
                pass


# ────────────────────────────── public shim ────────────────────────────────
def speak(text: str) -> None:
    """
    Thread-safe synchronous wrapper around :pyfunc:`_speak_async`.

    Why change the original behaviour?
    ----------------------------------
    The former implementation queued the coroutine on **the current
    event-loop** if one was already running.  Immediately afterwards many
    callers block the main thread with `input()` or other synchronous work,
    starving the loop and delaying audio until the block ends.

    New strategy
    ------------
    • If **no** event-loop is running in this thread → fall back to the
      straightforward ``asyncio.run(_speak_async(text))``.
    • If a loop **is** running → spin up a **daemon thread** that owns its
      *own* event-loop and run the coroutine there.
      A global ``_TTS_LOCK`` ensures that only one utterance plays at once,
      so messages remain sequential.
    """

    def _run_in_thread() -> None:
        try:
            asyncio.run(_speak_async(text))
        finally:
            _TTS_LOCK.release()

    try:
        # Is there already an event-loop in *this* thread?
        asyncio.get_running_loop()
    except RuntimeError:
        # No → safe to run the coroutine synchronously here
        asyncio.run(_speak_async(text))
    else:
        # Yes → grab the lock *now* to freeze call order and then start
        # a worker that will release it when done.
        _TTS_LOCK.acquire()
        threading.Thread(target=_run_in_thread, daemon=True).start()


def speak_and_wait(text: str) -> None:
    """Speak *text* and block until TTS playback has finished or was skipped.

    Convenience wrapper for places that want an immediate audible affirmation
    before continuing with a longer-running task.
    """
    speak(text)
    _wait_for_tts_end()


def stop_speaking() -> None:
    """Cancel any in-flight TTS playback immediately if active."""
    try:
        # avoid races if called during transitions
        if _CURRENT_TTS_SKIP is not None:
            _CURRENT_TTS_SKIP.set()
    except Exception:
        pass


def is_speaking() -> bool:
    """Return True if a TTS utterance is currently playing."""
    try:
        return _TTS_LOCK.locked()
    except Exception:
        return False


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
                    # Return the typed characters exactly as entered (no trimming)
                    return True, "".join(input_chars)
                input_chars.append(char)

            time.sleep(0.01)  # Small sleep to prevent CPU hogging

        return False, None
    else:
        # Unix implementation using select
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            # Preserve user input exactly as typed, removing only the trailing newline
            line = sys.stdin.readline()
            if line.endswith("\n"):
                line = line[:-1]
            return True, line
        return False, None


def get_custom_scenario(args) -> Optional[str]:
    """Get custom scenario from args, either text or voice input."""
    voice_flag = hasattr(args, "voice") and args.voice
    # Check for text-based custom scenario first
    if not voice_flag:
        return input("🧮 Explain your custom scenario, press ↵ once you're done\n")
    try:
        # Record and transcribe audio
        print("🧮 Let's build your custom scenario using voice")
        speak("Let's build your custom scenario using voice. Press enter to start.")
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


# ===========================================================================
#  CLI boilerplate helper (used by every sandbox)
# ===========================================================================


def build_cli_parser(description: str) -> argparse.ArgumentParser:
    """
    Return an :pyclass:`argparse.ArgumentParser` pre-populated with the core
    command-line switches shared by every interactive sandbox:

    • ``--voice / -v``        – enable voice capture & TTS
    • ``--debug / -d``        – verbose tool logs (reasoning steps)
    • ``--traced / -t``       – wrap manager calls in Unify tracing
    • ``--project_name / -p`` – Unify *project / context* name (default: "Sandbox")
    • ``--overwrite / -o``    – delete any existing data for *project_name*
                                 before seeding
    • ``--project_version``  – version index to load (default -1 for latest)
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--voice",
        "-v",
        action="store_true",
        help="enable voice capture + TTS",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="verbose tool logs (reasoning steps)",
    )
    parser.add_argument(
        "--traced",
        "-t",
        action="store_true",
        help="include Unify tracing",
    )
    parser.add_argument(
        "--project_name",
        "-p",
        default="Sandbox",
        metavar="NAME",
        help="Unify project / context name (default: Sandbox)",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="overwrite existing data for the chosen project name",
    )
    parser.add_argument(
        "--project_version",
        type=int,
        default=-1,
        metavar="IDX",
        help="Project version index to load (default -1 for latest; supports positive and negative indexing)",
    )
    parser.add_argument(
        "--log_in_terminal",
        action="store_true",
        help="stream logs to terminal in addition to writing .logs.txt (default is file-only)",
    )
    parser.add_argument(
        "--no_clarifications",
        action="store_true",
        help="disable interactive clarification requests (both text and voice)",
    )
    parser.add_argument(
        "--log_tcp_port",
        type=int,
        default=-1,
        metavar="PORT",
        help="serve logs over TCP on localhost:PORT (default -1 auto-picks an available port; 0 disables; >0 binds requested port)",
    )
    parser.add_argument(
        "--http_log_tcp_port",
        type=int,
        default=-1,
        metavar="PORT",
        help=(
            "serve Unify Request logs (logger: 'unify_requests' only) over TCP on localhost:PORT "
            "(default -1 auto-picks when UNIFY_REQUESTS_DEBUG is set; 0 disables; >0 binds requested port)"
        ),
    )
    return parser


class _LogBroadcastServer:
    """Minimal TCP log broadcaster: accepts clients and writes each line to them.

    Designed for local development only (binds to 127.0.0.1). Not for production.
    """

    def __init__(self, port: int) -> None:
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._clients: list[socket.socket] = []
        self._queue: SimpleQueue[bytes] = SimpleQueue()
        self._running = threading.Event()

    def start(self) -> None:
        # Bind synchronously so the actual port is known before returning
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Auto-pick free port when port <= 0 (incl. -1 sentinel)
            bind_port = 0 if self._port <= 0 else self._port
            srv.bind(("127.0.0.1", bind_port))
            srv.listen(5)
            self._sock = srv
            # Store the actual chosen port (useful when bind_port was 0)
            self._port = srv.getsockname()[1]
        except Exception:
            return

        self._running.set()
        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._drain_loop, daemon=True).start()

    def stop(self) -> None:
        self._running.clear()
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        for c in list(self._clients):
            try:
                c.close()
            except Exception:
                pass
        self._clients.clear()

    def broadcast(self, line: str) -> None:
        if not self._running.is_set():
            return
        self._queue.put((line + "\n").encode("utf-8", errors="ignore"))

    def _accept_loop(self) -> None:
        srv = self._sock
        if srv is None:
            return
        while self._running.is_set():
            try:
                srv.settimeout(0.5)
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                conn.setblocking(False)
                self._clients.append(conn)
            except Exception:
                break

    def _drain_loop(self) -> None:
        while self._running.is_set():
            try:
                chunk = self._queue.get(timeout=0.5)
            except Exception:
                continue
            dead: list[socket.socket] = []
            for c in self._clients:
                try:
                    c.sendall(chunk)
                except Exception:
                    dead.append(c)
            for d in dead:
                try:
                    d.close()
                except Exception:
                    pass
                if d in self._clients:
                    self._clients.remove(d)


class _BroadcastLogHandler(logging.Handler):
    def __init__(self, server: _LogBroadcastServer) -> None:
        super().__init__()
        self._server = server

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._server.broadcast(msg)
        except Exception:
            pass


def configure_sandbox_logging(
    log_in_terminal: bool = False,
    log_file: Optional[str] = ".logs_main.txt",
    tcp_port: int = 0,
    http_tcp_port: int = 0,
    unify_requests_log_file: Optional[str] = ".logs_unify_requests.txt",
) -> None:
    """Configure logging to a file by default, with optional terminal streaming.

    - Overwrites the given log_file on each run.
    - Adds a StreamHandler to stdout when log_in_terminal is True.
    - Optionally serves logs over TCP on localhost:tcp_port for external viewing.
    - Supports a dedicated Unify Request log stream/file that captures only the 'unify_requests' logger.
    - Prints a short hint on how to watch the last 50 lines live.
    """
    import sys as _sys
    import logging as _logging

    root_logger = _logging.getLogger()
    root_logger.setLevel(_logging.INFO)

    # Clear any existing handlers to prevent duplicates
    for _h in list(root_logger.handlers):
        root_logger.removeHandler(_h)

    _fmt = _logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # Resolve the main log file path to an absolute path for clearer, clickable output
    _abs_main_log: Optional[str] = None
    if log_file:
        try:
            _abs_main_log = os.path.abspath(log_file)
        except Exception:
            _abs_main_log = log_file

    if _abs_main_log:
        _fh = _logging.FileHandler(_abs_main_log, mode="w", encoding="utf-8")
        _fh.setFormatter(_fmt)

        # Exclude Unify Request logs from the main log file to keep it high-level
        # (Unify Request logs have their own dedicated file and stream)
        class _LazyHTTPExcludeFilter(_logging.Filter):
            def filter(self, record: _logging.LogRecord) -> bool:
                name = record.name or ""
                return not any(name.startswith(p) for p in _HTTP_PREFIXES)

        _fh.addFilter(_LazyHTTPExcludeFilter())
        root_logger.addHandler(_fh)

    if log_in_terminal:
        _sh = _logging.StreamHandler(_sys.stdout)
        _sh.setFormatter(_fmt)
        root_logger.addHandler(_sh)

    # Helper: common filter to exclude/include HTTP-debug loggers
    class _NamePrefixFilter(_logging.Filter):
        def __init__(
            self,
            include_prefixes: Optional[list[str]] = None,
            exclude_prefixes: Optional[list[str]] = None,
        ) -> None:
            super().__init__()
            self._include = tuple(include_prefixes or [])
            self._exclude = tuple(exclude_prefixes or [])

        def filter(self, record: _logging.LogRecord) -> bool:  # noqa: D401
            name = record.name or ""
            if self._include and not any(name.startswith(p) for p in self._include):
                return False
            if self._exclude and any(name.startswith(p) for p in self._exclude):
                return False
            return True

    # Determine Unify Request logger prefixes (override via env if needed)
    _http_logger_env = os.getenv("HTTP_DEBUG_LOGGERS", "").strip()
    if _http_logger_env:
        _HTTP_PREFIXES = [p.strip() for p in _http_logger_env.split(",") if p.strip()]
    else:
        # Restrict to only Unify Request logs by default
        _HTTP_PREFIXES = [
            "unify_requests",  # Unify SDK dedicated request logger
        ]

    # Optional TCP broadcast for external terminals (main logs)
    # tcp_port semantics:
    #   -1 → auto-pick a free port and enable streaming by default
    #    0 → disabled
    #   >0 → bind requested port
    if tcp_port != 0:
        try:
            _srv = _LogBroadcastServer(tcp_port)
            _srv.start()
            _bh = _BroadcastLogHandler(_srv)
            _bh.setFormatter(_fmt)
            root_logger.addHandler(_bh)
            _actual = _srv._port
            # Also write a full-session copy to a hidden, timestamped file in CWD
            _ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            _hidden_name = f".logs_{_ts}.txt"
            # Resolve the hidden full-session log path to absolute for printing
            try:
                _abs_hidden = os.path.abspath(_hidden_name)
            except Exception:
                _abs_hidden = _hidden_name

            _fh_all = _logging.FileHandler(_abs_hidden, mode="w", encoding="utf-8")
            _fh_all.setFormatter(_fmt)
            root_logger.addHandler(_fh_all)
            print(
                f"📡 Log stream on 127.0.0.1:{_actual} – connect via: nc 127.0.0.1 {_actual} (Ctrl-C to detach)",
            )
            print(f"📝 Full session logs: {_abs_hidden}")
        except Exception as _exc:
            print(f"⚠️  Failed to start log TCP stream on port {tcp_port}: {_exc}")

    # Dedicated Unify Request debug stream (enabled when port provided or UNIFY_REQUESTS_DEBUG truthy and http_tcp_port == -1)
    _unify_debug_env = os.getenv("UNIFY_REQUESTS_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _start_http_stream = False
    _http_bind_port = http_tcp_port
    if http_tcp_port == -1:
        _start_http_stream = _unify_debug_env
        _http_bind_port = -1  # auto-pick if enabled
    elif http_tcp_port > 0:
        _start_http_stream = True

    if _start_http_stream:
        try:
            # Ensure 'unify_requests' logger emits DEBUG when UNIFY_REQUESTS_DEBUG is truthy
            if _unify_debug_env:
                for _name in _HTTP_PREFIXES:
                    try:
                        _logging.getLogger(_name).setLevel(_logging.DEBUG)
                    except Exception:
                        pass
            _srv_http = _LogBroadcastServer(_http_bind_port)
            _srv_http.start()
            _bh_http = _BroadcastLogHandler(_srv_http)
            _bh_http.setFormatter(_fmt)
            # Only include the Unify Request logger category
            _bh_http.addFilter(_NamePrefixFilter(include_prefixes=_HTTP_PREFIXES))

            # Attach to root but exclude these from main console/broadcast by filtering there
            root_logger.addHandler(_bh_http)
            _http_actual = _srv_http._port

            # Exclude Unify Request logs from the main stream and console if present
            for h in list(root_logger.handlers):
                if h is _bh_http:
                    continue
                if isinstance(h, (_logging.StreamHandler, _BroadcastLogHandler)):
                    h.addFilter(_NamePrefixFilter(exclude_prefixes=_HTTP_PREFIXES))

            print(
                f"📡 Unify Request debug stream on 127.0.0.1:{_http_actual} – connect via: nc 127.0.0.1 {_http_actual} (Ctrl-C to detach)",
            )
        except Exception as _exc:
            print(
                f"⚠️  Failed to start Unify Request debug TCP stream on port {http_tcp_port}: {_exc}",
            )

    # Dedicated Unify Request debug file
    _abs_http_log: Optional[str] = None
    if unify_requests_log_file:
        try:
            _abs_http_log = os.path.abspath(unify_requests_log_file)
        except Exception:
            _abs_http_log = unify_requests_log_file

    if _abs_http_log:
        try:
            _fh_http = _logging.FileHandler(_abs_http_log, mode="w", encoding="utf-8")
            _fh_http.setFormatter(_fmt)
            _fh_http.addFilter(_NamePrefixFilter(include_prefixes=_HTTP_PREFIXES))
            root_logger.addHandler(_fh_http)
            print(f"📝 Unify Request logs to {_abs_http_log}")
        except Exception as _exc:
            print(f"⚠️  Failed to open Unify Request log file {_abs_http_log}: {_exc}")

    # Friendly hints
    if _abs_main_log:
        print(
            f"📝 Logging to {_abs_main_log} (overwrites each run). "
            f"To follow live with scrollback: less +F {_abs_main_log} (Ctrl-C to pause, F to resume, q to quit). "
            "Pass --log_in_terminal to also stream logs here.",
        )


# ===========================================================================
# Minimal, cross-sandbox input / interrupt helpers
# ===========================================================================


def input_now(timeout: float = 0.1) -> Optional[str]:
    """
    Quick helper that returns the next *line* waiting on **stdin**
    (stripped) or ``None`` if nothing arrived within *timeout* s.

    It re-uses :pyfunc:`input_with_timeout`, which already handles the
    Windows vs Unix intricacies.
    """

    has_input, txt = input_with_timeout(timeout)
    return txt if has_input else None


def steering_controls_hint(
    pending_clarification: bool = False,
    *,
    voice_enabled: bool = False,
) -> str:
    """Return a one-line hint with available in-flight steering commands.

    Clarification controls are included only when a clarification is pending.
    Clarification-related commands are emphasized in bold to stand out.
    """
    base_parts: list[str] = [
        "/i <text>",
        "/pause",
        "/resume",
        "/ask <q>",
        "/freeform <text>",
    ]
    if voice_enabled:
        base_parts.append("/r (record voice)")
    base_parts.extend(["/stop", "/help"])

    hint = "Controls: " + ", ".join(base_parts)

    if pending_clarification:
        B = "\u001b[1m"
        R = "\u001b[0m"
        clar_parts = [
            f"{B}/c <answer> (clarify){R}",
            f"{B}/rc (record clarification){R}" if voice_enabled else None,
        ]
        clar_parts = [p for p in clar_parts if p is not None]
        hint = hint + ", " + ", ".join(clar_parts)

    return hint


# Shared steering intent model and router system prompt
class _SteeringIntent(BaseModel):
    action: str = Field(..., pattern="^(ask|interject|pause|resume|stop|status)$")


def _steering_router_sys() -> str:
    return (
        "You are a router that maps a user's free-form message to one of these steering commands: "
        "'ask', 'interject', 'pause', 'resume', 'stop', or 'status'.\n"
        "You will be given a short transcript with the latest user message at the end. Decide based on the user's intent given the conversation and whether a task is currently running.\n"
        "Definitions:\n"
        "- 'interject': Any directive that would change, add, remove, create, continue, or otherwise steer what the running task should do next. "
        "Treat polite or indirect phrasing (e.g., 'could you', 'please', 'let's', 'why don't we'), and requests containing action verbs (set/add/update/change/modify/remove/delete/replace/make/use/fill/assign/write/generate) as interjections. "
        "If a message mixes a question with a requested action, choose 'interject'.\n"
        "- 'ask': A read-only question about the running task or its data (progress, status, what has happened/what will happen, counts, why, how long). "
        "It must NOT request any change to behaviour or data.\n"
        "- 'pause'/'resume'/'stop'/'status': Direct control commands. Map common synonyms: 'continue' ⇒ 'resume'.\n"
        "Conversation-aware rules:\n"
        "- If a task is RUNNING and the latest user message indicates reverting, postponing, deferring, or otherwise NOT doing the task now (e.g., 'let's do it next week', 'don't worry about it now', 'we'll do it later', 'scrap this for now'), choose 'stop'.\n"
        "- If a task is RUNNING and the user explicitly asks to pause/hold/temporarily stop, choose 'pause'.\n"
        "- If a task is PAUSED (you may infer from recent 'Paused' announcements) and the user asks to continue, choose 'resume'.\n"
        "- If the request is to update/modify/steer the currently running task (without stopping/cancelling), then choose `interject`.\n"
        "Rules:\n"
        "- Only decide the action; do not rewrite, summarize, or clean the user's text.\n"
        "- Ignore pleasantries and judge the semantics.\n"
        "- When uncertain between 'ask' and 'interject', choose 'interject' (safer).\n"
        "Return ONLY JSON matching the response schema with an 'action' field."
    )


async def _apply_steering_action(
    handle: "SteerableToolHandle",
    action: str,
    text: str,
    enable_voice_steering: bool,
    HELP_TEXT: str,
) -> bool:
    """Apply a routed steering action. Returns True if the caller should break (on stop)."""
    try:
        if action == "ask":
            print(f"asking question: {text}")
            nested = await handle.ask(text)
            ans = await nested.result()
            print(f"[ask] → {ans}")
            if enable_voice_steering:
                speak(str(ans))
                _wait_for_tts_end()
                print(HELP_TEXT)
            return False
        if action == "interject":
            txt_to_inject = text.strip()
            if not txt_to_inject:
                print("⚠️  Router produced empty interjection – ignoring")
                return False
            print(f"interjecting: {txt_to_inject}")
            run_in_loop(handle.interject(txt_to_inject))
            print("✅ Interjection sent.")
            if enable_voice_steering:
                speak("Interjection sent")
                _wait_for_tts_end()
                print(HELP_TEXT)
            else:
                print(HELP_TEXT)
            return False
        if action == "pause":
            try:
                print("pausing…")
                handle.pause()
                print("⏸️  Paused")
                if enable_voice_steering:
                    speak("Paused")
                    _wait_for_tts_end()
                    print(HELP_TEXT)
                else:
                    print(HELP_TEXT)
            except Exception as exc:
                print(f"⚠️  Pause failed: {exc}")
            return False
        if action == "resume":
            try:
                print("resuming…")
                handle.resume()
                print("▶️  Resumed")
                if enable_voice_steering:
                    speak("Resumed")
                    _wait_for_tts_end()
                    print(HELP_TEXT)
                else:
                    print(HELP_TEXT)
            except Exception as exc:
                print(f"⚠️  Resume failed: {exc}")
            return False
        if action == "stop":
            print("stopping…")
            handle.stop()
            print("✅ Stop sent.")
            if enable_voice_steering:
                speak("Stop sent")
                _wait_for_tts_end()
                print(HELP_TEXT)
            else:
                print(HELP_TEXT)
            return True
        if action == "status":
            print("status requested")
            _state = "done" if handle.done() else "running"
            print(_state)
            if enable_voice_steering:
                speak(f"Status: {_state}")
                _wait_for_tts_end()
                print(HELP_TEXT)
            else:
                print(HELP_TEXT)
            return False
        # Fallback to interject if unknown
        print(f"interjecting: {text}")
        run_in_loop(handle.interject(text))
        print("✅ Interjection sent.")
        if enable_voice_steering:
            speak("Interjection sent")
            _wait_for_tts_end()
            print(HELP_TEXT)
        else:
            print(HELP_TEXT)
        return False
    except Exception as exc:
        print(f"⚠️  Freeform routing failed: {exc}")
        return False


async def _route_freeform_and_apply(
    handle: "SteerableToolHandle",
    text: str,
    enable_voice_steering: bool,
    HELP_TEXT: str,
    chat_context: Optional[list[dict]] = None,
    is_task_running: Optional[bool] = None,
) -> bool:
    import unify as _unify

    judge = _unify.Unify("gpt-4o@openai", response_format=_SteeringIntent)

    # Build a compact, recent-first transcript to provide conversation context
    def _format_ctx(ctx: list[dict], limit_chars: int = 2000) -> str:
        try:
            lines: list[str] = []
            total = 0
            for msg in reversed(ctx[-20:]):  # last 20 turns max
                role = str(msg.get("role", "")).strip() or "user"
                content = str(msg.get("content", "")).strip()
                line = f"{role}: {content}"
                if total + len(line) > limit_chars:
                    break
                lines.append(line)
                total += len(line)
            return "\n".join(reversed(lines)) if lines else "(no prior context)"
        except Exception:
            return "(no prior context)"

    ctx_block = _format_ctx(chat_context or [])
    running_hint = (
        "RUNNING"
        if (is_task_running is True)
        else ("UNKNOWN" if is_task_running is None else "NOT_RUNNING")
    )

    router_input = (
        "Conversation (most recent last):\n"
        f"{ctx_block}\n\n"
        f"Task state: {running_hint}.\n"
        "Latest user message:\n"
        f"{text}"
    )

    intent = _SteeringIntent.model_validate_json(
        judge.set_system_message(_steering_router_sys()).generate(router_input),
    )
    return await _apply_steering_action(
        handle,
        intent.action,
        text,
        enable_voice_steering,
        HELP_TEXT,
    )


async def await_with_interrupt(  # noqa: D401 – imperative helper
    handle: "SteerableToolHandle",
    poll: float = 0.05,
    *,
    enable_voice_steering: bool = False,
    clarification_up_q: Optional[asyncio.Queue[str]] = None,
    clarification_down_q: Optional[asyncio.Queue[str]] = None,
    clarifications_enabled: bool = True,
    chat_context: Optional[list[dict]] = None,
) -> str:
    """
    **Common wrapper** used by all interactive sandboxes.

    Waits on ``handle.result()`` but lets the user:
    • /i <text> or plain text     ⇒ interject via ``handle.interject``
    • /pause | /p                 ⇒ pause the running call
    • /resume | /r                ⇒ resume a paused call
    • /ask <question> | ? <q>     ⇒ ask a read-only question about the running call
    • /freeform <text>            ⇒ route free-form text to the best steering command via an LLM
    • /r | /record                ⇒ when enable_voice_steering=True, capture voice and route via freeform
    • /stop | /cancel             ⇒ abort the running call
    • /status                     ⇒ print whether the call is done
    • /help                       ⇒ show available controls

    Commands use a leading '/' prefix to avoid accidental interjections.
    """

    import asyncio  # local to avoid widening the public surface

    # State for handling a single pending clarification at a time
    pending_clar_q: Optional[str] = None
    has_clar_channels = bool(
        clarifications_enabled and clarification_up_q and clarification_down_q,
    )

    while not handle.done():
        # Non-blocking check for incoming clarification questions
        if has_clar_channels and pending_clar_q is None:
            try:
                # get_nowait raises when empty
                pending_clar_q = cast(Optional[str], clarification_up_q.get_nowait())  # type: ignore[arg-type]
                if pending_clar_q:
                    print()
                    print(f"❓ Clarification requested: {pending_clar_q}")
                    print(
                        "Reply with: /c <your answer> or just type your answer and press ↵. "
                        + (
                            "Use /rc to record by voice."
                            if enable_voice_steering
                            else ""
                        ),
                    )
                    if enable_voice_steering:
                        speak(f"Clarification requested. {pending_clar_q}")
                        _wait_for_tts_end()
                    # After announcing the clarification, print dynamic controls with clar commands visible
                    print(
                        steering_controls_hint(
                            pending_clarification=True,
                            voice_enabled=enable_voice_steering,
                        ),
                    )
            except Exception:
                pass

        txt = input_now(poll * 2)  # same cadence as old versions
        if txt is not None and txt != "":
            # Use a left-trimmed view only for recognizing commands, but keep the original text intact
            working = txt.lstrip()
            # Command mode with leading '/'
            if working.startswith("/"):
                # Parse command token while preserving the raw argument text
                cmd_line = working[1:]
                # Find first whitespace separating command and argument
                space_idx = -1
                for i, ch in enumerate(cmd_line):
                    if ch.isspace():
                        space_idx = i
                        break
                if space_idx == -1:
                    cmd = cmd_line.lower()
                    arg = ""
                else:
                    cmd = cmd_line[:space_idx].lower()
                    # Preserve the argument exactly as typed (post-separator substring)
                    arg = cmd_line[space_idx + 1 :]

                if cmd in {"stop", "cancel", "s"}:
                    print("stopping…")
                    handle.stop()
                    print("✅ Stop sent.")
                    if enable_voice_steering:
                        speak("Stop sent")
                        _wait_for_tts_end()
                        print(
                            steering_controls_hint(
                                pending_clarification=(pending_clar_q is not None),
                                voice_enabled=enable_voice_steering,
                            ),
                        )
                    else:
                        print(
                            steering_controls_hint(
                                pending_clarification=(pending_clar_q is not None),
                                voice_enabled=enable_voice_steering,
                            ),
                        )
                    break
                # Clarification commands (handled irrespective of other state if channels exist)
                if (
                    has_clar_channels
                    and (pending_clar_q is not None)
                    and cmd in {"c", "clarify"}
                ):
                    arg_to_send = arg if arg != "" else ""
                    if not arg_to_send.strip():
                        print("Usage: /c <answer>")
                    else:
                        try:
                            await clarification_down_q.put(arg_to_send)  # type: ignore[union-attr]
                            print("✅ Clarification sent.")
                            pending_clar_q = None
                            if enable_voice_steering:
                                speak("Clarification sent")
                                _wait_for_tts_end()
                                print(
                                    steering_controls_hint(
                                        pending_clarification=False,
                                        voice_enabled=enable_voice_steering,
                                    ),
                                )
                            else:
                                print(
                                    steering_controls_hint(
                                        pending_clarification=False,
                                        voice_enabled=enable_voice_steering,
                                    ),
                                )
                        except Exception as exc:
                            print(f"⚠️  Failed to send clarification: {exc}")
                    continue
                if (
                    has_clar_channels
                    and (pending_clar_q is None)
                    and cmd in {"c", "clarify"}
                ):
                    print(
                        "(no clarification pending) These commands are only available when a tool has requested clarification.",
                    )
                    continue
                if (
                    has_clar_channels
                    and (pending_clar_q is not None)
                    and cmd in {"rc"}
                    and enable_voice_steering
                ):
                    try:
                        print(
                            "🎙️  Clarification – press ↵ to start, ↵ again to send, 'c'+↵ to cancel",
                        )
                        audio = record_until_enter_interruptible(lambda: handle.done())
                        if audio is None:
                            continue
                        transcript = transcribe_deepgram(audio)
                        if not transcript or transcript.strip() == "":
                            print("⚠️  Empty transcript – ignoring")
                            continue
                        await clarification_down_q.put(transcript)  # type: ignore[union-attr]
                        print("✅ Clarification sent.")
                        pending_clar_q = None
                        speak("Clarification sent")
                        _wait_for_tts_end()
                        print(
                            steering_controls_hint(
                                pending_clarification=False,
                                voice_enabled=enable_voice_steering,
                            ),
                        )
                    except Exception as exc:
                        print(f"⚠️  Voice clarification failed: {exc}")
                    continue
                if (
                    has_clar_channels
                    and (pending_clar_q is None)
                    and cmd in {"rc"}
                    and enable_voice_steering
                ):
                    print(
                        "(no clarification pending) These commands are only available when a tool has requested clarification.",
                    )
                    continue
                # '/cs' (skip) removed – user can type a message if they wish not to clarify
                # '/cs' (skip) removed – ignore when no clarification is pending
                if cmd in {"pause", "p"}:
                    try:
                        print("pausing…")
                        handle.pause()
                        print("⏸️  Paused")
                        if enable_voice_steering:
                            speak("Paused")
                            _wait_for_tts_end()
                            print(
                                steering_controls_hint(
                                    pending_clarification=(pending_clar_q is not None),
                                    voice_enabled=enable_voice_steering,
                                ),
                            )
                        else:
                            print(
                                steering_controls_hint(
                                    pending_clarification=(pending_clar_q is not None),
                                    voice_enabled=enable_voice_steering,
                                ),
                            )
                    except Exception as exc:
                        print(f"⚠️  Pause failed: {exc}")
                    continue
                if cmd in {"resume", "play", "continue"}:
                    try:
                        print("resuming…")
                        handle.resume()
                        print("▶️  Resumed")
                        if enable_voice_steering:
                            speak("Resumed")
                            _wait_for_tts_end()
                            print(
                                steering_controls_hint(
                                    pending_clarification=(pending_clar_q is not None),
                                    voice_enabled=enable_voice_steering,
                                ),
                            )
                        else:
                            print(
                                steering_controls_hint(
                                    pending_clarification=(pending_clar_q is not None),
                                    voice_enabled=enable_voice_steering,
                                ),
                            )
                    except Exception as exc:
                        print(f"⚠️  Resume failed: {exc}")
                    continue
                if cmd in {"i", "interject"}:
                    if not arg.strip():
                        print("Usage: /i <text>")
                    else:
                        print(f"interjecting: {arg}")
                        # Forward the user's text exactly as provided
                        run_in_loop(handle.interject(arg))
                        print("✅ Interjection sent.")
                        if enable_voice_steering:
                            speak("Interjection sent")
                            _wait_for_tts_end()
                            print(
                                steering_controls_hint(
                                    pending_clarification=(pending_clar_q is not None),
                                    voice_enabled=enable_voice_steering,
                                ),
                            )
                        else:
                            print(
                                steering_controls_hint(
                                    pending_clarification=(pending_clar_q is not None),
                                    voice_enabled=enable_voice_steering,
                                ),
                            )
                    continue
                if cmd in {"ask", "?"}:
                    if not arg.strip():
                        print("Usage: /ask <question>")
                    else:
                        try:
                            print(f"asking question: {arg}")
                            # Forward the question exactly as provided
                            nested = await handle.ask(arg)
                            ans = await nested.result()
                            print(f"[ask] → {ans}")
                            if enable_voice_steering:
                                speak(str(ans))
                                _wait_for_tts_end()
                                print(
                                    steering_controls_hint(
                                        pending_clarification=(
                                            pending_clar_q is not None
                                        ),
                                        voice_enabled=enable_voice_steering,
                                    ),
                                )
                        except Exception as exc:
                            print(f"⚠️  Ask failed: {exc}")
                    continue
                if enable_voice_steering and cmd in {"record", "rec", "r"}:
                    try:
                        print(
                            "🎙️  Voice steering – press ↵ to start, ↵ again to send, 'c'+↵ to cancel",
                        )
                        audio = record_until_enter_interruptible(lambda: handle.done())
                        if audio is None:
                            continue
                        transcript = transcribe_deepgram(audio)
                        if not transcript or transcript.strip() == "":
                            print("⚠️  Empty transcript – ignoring")
                            continue
                        should_break = await _route_freeform_and_apply(
                            handle,
                            transcript,
                            enable_voice_steering,
                            steering_controls_hint(
                                pending_clarification=(pending_clar_q is not None),
                                voice_enabled=enable_voice_steering,
                            ),
                            chat_context=chat_context,
                            is_task_running=not handle.done(),
                        )
                        if should_break:
                            break
                    except Exception as exc:
                        print(f"⚠️  Voice steering failed: {exc}")
                    continue
                if cmd in {"freeform", "f"}:
                    if not arg.strip():
                        print("Usage: /freeform <text>")
                        continue
                    should_break = await _route_freeform_and_apply(
                        handle,
                        arg,
                        enable_voice_steering,
                        steering_controls_hint(
                            pending_clarification=(pending_clar_q is not None),
                            voice_enabled=enable_voice_steering,
                        ),
                        chat_context=chat_context,
                        is_task_running=not handle.done(),
                    )
                    if should_break:
                        break
                    continue
                if cmd in {"status", "st"}:
                    print("status requested")
                    state = "done" if handle.done() else "running"
                    print(state)
                    if enable_voice_steering:
                        speak(f"Status: {state}")
                        _wait_for_tts_end()
                        print(
                            steering_controls_hint(
                                pending_clarification=(pending_clar_q is not None),
                                voice_enabled=enable_voice_steering,
                            ),
                        )
                    else:
                        print(
                            steering_controls_hint(
                                pending_clarification=(pending_clar_q is not None),
                                voice_enabled=enable_voice_steering,
                            ),
                        )
                    continue
                if cmd in {"help", "h"}:
                    print(
                        steering_controls_hint(
                            pending_clarification=(pending_clar_q is not None),
                            voice_enabled=enable_voice_steering,
                        ),
                    )
                    continue
                # Unknown command → treat as interjection without the '/'
                unknown_text = working[1:]
                print(f"interjecting: {unknown_text}")
                run_in_loop(handle.interject(unknown_text))
                print("✅ Interjection sent.")
                if enable_voice_steering:
                    speak("Interjection sent")
                    _wait_for_tts_end()
                    print(
                        steering_controls_hint(
                            pending_clarification=(pending_clar_q is not None),
                            voice_enabled=enable_voice_steering,
                        ),
                    )
                else:
                    print(
                        steering_controls_hint(
                            pending_clarification=(pending_clar_q is not None),
                            voice_enabled=enable_voice_steering,
                        ),
                    )
            else:
                # Plain text: if a clarification is pending and channels exist, treat as clarification answer
                if has_clar_channels and pending_clar_q is not None:
                    try:
                        await clarification_down_q.put(txt)  # type: ignore[union-attr]
                        print("✅ Clarification sent.")
                        pending_clar_q = None
                        if enable_voice_steering:
                            speak("Clarification sent")
                            _wait_for_tts_end()
                            print(
                                steering_controls_hint(
                                    pending_clarification=False,
                                    voice_enabled=enable_voice_steering,
                                ),
                            )
                        else:
                            print(
                                steering_controls_hint(
                                    pending_clarification=False,
                                    voice_enabled=enable_voice_steering,
                                ),
                            )
                    except Exception as exc:
                        print(f"⚠️  Failed to send clarification: {exc}")
                else:
                    # Otherwise → interject (forward exactly as typed)
                    print(f"interjecting: {txt}")
                    run_in_loop(handle.interject(txt))
                    print("✅ Interjection sent.")
                    if enable_voice_steering:
                        speak("Interjection sent")
                        _wait_for_tts_end()
                        print(
                            steering_controls_hint(
                                pending_clarification=(pending_clar_q is not None),
                                voice_enabled=enable_voice_steering,
                            ),
                        )
                    else:
                        print(
                            steering_controls_hint(
                                pending_clarification=(pending_clar_q is not None),
                                voice_enabled=enable_voice_steering,
                            ),
                        )
        await asyncio.sleep(poll)

    # Task completed: cancel any ongoing TTS immediately and return result
    stop_speaking()
    return await handle.result()


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
        # We are already inside the loop's thread
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


# ===========================================================================
# Helper to invoke manager methods with optional clarification channels
# ===========================================================================


async def call_manager_with_optional_clarifications(
    fn: Any,
    text: str,
    *,
    parent_chat_context: list[dict],
    return_reasoning_steps: bool = False,
    clarifications_enabled: bool = True,
):
    """
    Call a manager method (e.g., ask/update) with context and, when supported,
    attach clarification queues automatically.

    Returns a tuple: (handle, clarification_up_q, clarification_down_q).
    """
    import inspect as _inspect
    import asyncio as _asyncio

    clar_up_q: Optional[asyncio.Queue[str]] = None  # type: ignore[name-defined]
    clar_down_q: Optional[asyncio.Queue[str]] = None  # type: ignore[name-defined]

    kwargs: Dict[str, Any] = {
        "parent_chat_context": parent_chat_context,
        "_return_reasoning_steps": return_reasoning_steps,
    }

    try:
        sig = _inspect.signature(fn)
    except Exception:
        sig = None

    if (
        clarifications_enabled
        and sig is not None
        and "clarification_up_q" in sig.parameters
        and "clarification_down_q" in sig.parameters
    ):
        clar_up_q = _asyncio.Queue()
        clar_down_q = _asyncio.Queue()
        kwargs["clarification_up_q"] = clar_up_q
        kwargs["clarification_down_q"] = clar_down_q

    handle = await fn(text, **kwargs)
    return handle, clar_up_q, clar_down_q


# ===========================================================================
# Synthetic transcript generation helper
# ===========================================================================


class TranscriptGenerator:
    """Generate synthetic multi-party chat transcripts via ScenarioBuilder.

    The generator orchestrates a self-contained tool-loop exposing a single
    ``log_messages`` tool that the LLM must use to incrementally append
    messages to an in-memory list.  Callers supply a free-form *description*
    of the desired conversation and the model produces a realistic transcript
    that satisfies that brief.

    Parameters
    ----------
    endpoint
        Chat-completion model identifier (same format used across sandboxes).
    traced
        Forwarded to :class:`unify.AsyncUnify` so unit-tests can inspect
        detailed traces when needed.
    stateful
        Re-use the underlying client across multiple ``generate`` calls –
        handy when chaining several transcripts together inside higher-level
        demos.
    """

    def __init__(
        self,
        *,
        endpoint: str = "gpt-5->o4-mini@openai",
        traced: bool = True,
        stateful: bool = True,
        in_conversation_manager: bool = False,
    ) -> None:
        self._endpoint = endpoint
        self._traced = traced
        self._stateful = stateful
        self._tm = TranscriptManager()
        self._in_cm = in_conversation_manager

    async def generate(
        self,
        description: str,
        *,
        min_messages: int = 40,
        max_messages: int = 60,
        batch_min: int = 3,
        batch_max: int = 8,
        delay_per_message: float = 0.0,
    ) -> List[dict]:
        """Return a synthetic transcript matching *description*.

        Each message dict contains:
        • ``timestamp`` – ISO-8601 string
        • ``sender``    – speaker name / alias
        • ``content``   – raw text
        """

        transcript: List[dict] = []

        from unity.contact_manager.types.contact import Contact  # local import

        # Cache of participant-label → Contact to ensure that two different
        # people who share the same first name (e.g. "Fred Smith" and
        # "Fred Taylor") are treated as distinct individuals during a single
        # generation run. Labels are normalised to lower-case full strings.
        _name_to_contact: dict[str, Contact] = {}
        # Track the Contact object of the previous message to infer receiver
        last_sender_contact: Contact | None = None

        # ------------------------------------------------------------------ #
        #  New, simpler input format                                        #
        # ------------------------------------------------------------------ #

        from datetime import datetime, timedelta, timezone  # local import

        def _build_contact(
            name: str,
            medium: str,
            details: dict[str, Any] | None,
        ) -> Contact:  # type: ignore[valid-type]
            """Return an existing Contact when the *first name* already exists.

            • If exactly one stored contact matches the first name, reuse it.
            • Otherwise create a *new* Contact instance (with contact_id = -1) so
              TranscriptManager will persist it on first use.
            """

            # 1️⃣  Attempt to reuse an existing contact
            try:
                cm = self._tm._contact_manager  # ContactManager instance
                parts = name.strip().split()
                first_name = (parts[0] if parts else "").lower()
                surname = " ".join(parts[1:]).strip().lower() if len(parts) > 1 else ""

                match: list[Contact] = []
                # Attempt 1: exact case-insensitive FULL-NAME match when a surname is present
                if first_name and surname:
                    match = cm._filter_contacts(
                        filter=(
                            "first_name is not None and surname is not None and "
                            f"first_name.lower() == '{first_name}' and surname.lower() == '{surname}'"
                        ),
                        limit=1,
                    )
                # Attempt 2: only when NO surname provided – reuse a unique first-name match
                if not match and first_name and not surname:
                    match = cm._filter_contacts(
                        filter=f"first_name.lower() == '{first_name}'",
                        limit=1,
                    )
                    if not match:
                        match = cm._filter_contacts(
                            filter=f"first_name.lower().startswith('{first_name}')",
                            limit=1,
                        )

                if match:
                    return match[0]
            except Exception:
                # Any backend/cycle issues → fall through to new contact generation
                pass

            # 2️⃣  No existing contact found → fabricate a new one
            details = details or {}
            # Robustly split *name* into first_name and (optional) surname so that
            # we never treat the full name as the first_name.  This fixes the issue
            # where "Daniel Lenton" was stored with first_name="Daniel Lenton" and
            # surname=None, leading to duplicate contact creation.

            first, *rest = name.strip().split()
            base_kwargs: dict[str, Any] = {"first_name": first.title()}
            if rest:
                base_kwargs["surname"] = " ".join(rest).title()

            # Preserve any recognised fields the LLM provided
            for fld in [
                "surname",
                "email_address",
                "phone_number",
                "whatsapp_number",
                "description",
                "bio",
                "rolling_summary",
            ]:
                if fld in details:
                    base_kwargs[fld] = details[fld]

            # Derive a synthetic identifier if none was given
            if "email_address" not in base_kwargs and "phone_number" not in base_kwargs:
                slug = name.lower().replace(" ", ".")
                idx = len(_name_to_contact) + 1
                if medium == "email":
                    base_kwargs["email_address"] = f"{slug}@example.com"
                else:
                    base_kwargs["phone_number"] = f"+155500{idx:04d}"

            return Contact(**base_kwargs)

        # Replace *create* helper so it accepts extra details
        def _contact_for(
            name: str,
            medium: str,
            details: dict[str, Any] | None = None,
        ) -> Contact:  # type: ignore[valid-type]
            def _norm_label(label: str) -> str:
                # Normalise by collapsing whitespace and lower-casing the full label
                return " ".join(label.split()).strip().lower()

            key = _norm_label(name)
            if key not in _name_to_contact:
                _name_to_contact[key] = _build_contact(name, medium, details)
            return _name_to_contact[key]

        class ConversationMessage(BaseModel):
            """Single utterance in the conversation."""

            sender: str = Field(..., description="Speaker display name")
            content: str = Field(..., description="Raw message text")

        class ConversationPayload(BaseModel):
            """Structured payload expected by `submit_conversation`.

            Fields:
            - medium: communication channel used
            - participants: map of participant display name → arbitrary details
              (e.g., phone_number, email_address, bio). Values are open‑schema.
            - conversation: ordered list of messages (sender/content pairs).
            """

            medium: (
                Literal[
                    "phone_call",
                    "sms_message",
                    "email",
                    "whatsapp_message",
                    "whatsapp_call",
                ]
                | str
            ) = Field(
                "sms_message",
                description=(
                    "Channel: phone_call | sms_message | email | whatsapp_message | whatsapp_call"
                ),
            )
            participants: Dict[str, Dict[str, Any]] = Field(
                default_factory=dict,
                description="Participant details keyed by display name",
            )
            conversation: List[ConversationMessage] = Field(
                ...,
                description="Ordered list of messages",
            )

        def submit_conversation(
            payload: ConversationPayload | dict | str | None = None,
            **tool_kwargs,
        ) -> str:  # noqa: C901 – complex but self-contained
            """Submit a complete conversation transcript for logging.

            Preferred call shape (validated):
            - payload: ConversationPayload

            Tolerated fallbacks (for robustness):
            - payload as JSON-serialisable dict or JSON string
            - flattened kwargs: medium=..., participants=..., conversation=[...]

            Extra kwargs (e.g. internal tool-loop params like parent_chat_context)
            are accepted and ignored.
            """

            nonlocal transcript, last_sender_contact

            # Normalise inputs → ConversationPayload
            model_payload: ConversationPayload
            if payload is None:
                # Check common LLM shapes
                if "payload" in tool_kwargs:
                    candidate = tool_kwargs["payload"]
                elif any(
                    k in tool_kwargs for k in ("medium", "participants", "conversation")
                ):
                    candidate = {
                        "medium": tool_kwargs.get("medium", "sms_message"),
                        "participants": tool_kwargs.get("participants", {}),
                        "conversation": tool_kwargs.get("conversation", []),
                    }
                else:
                    raise ValueError("submit_conversation requires a payload")

                if isinstance(candidate, str):
                    import json as _json

                    try:
                        model_payload = ConversationPayload.model_validate(
                            _json.loads(candidate),
                        )
                    except Exception as exc:
                        raise ValueError(
                            "submit_conversation: string payload must be valid JSON matching schema",
                        ) from exc
                elif isinstance(candidate, dict):
                    model_payload = ConversationPayload.model_validate(candidate)
                elif isinstance(candidate, ConversationPayload):
                    model_payload = candidate
                else:
                    raise ValueError("Unsupported payload type")
            else:
                if isinstance(payload, str):
                    import json as _json

                    try:
                        model_payload = ConversationPayload.model_validate(
                            _json.loads(payload),
                        )
                    except Exception as exc:
                        raise ValueError(
                            "submit_conversation: string payload must be valid JSON matching schema",
                        ) from exc
                elif isinstance(payload, dict):
                    model_payload = ConversationPayload.model_validate(payload)
                elif isinstance(payload, ConversationPayload):
                    model_payload = payload
                else:
                    raise ValueError("Unsupported payload type")

            medium = str(model_payload.medium)
            participants: dict[str, Any] = model_payload.participants or {}
            convo_raw = [
                {"sender": m.sender, "content": m.content}
                for m in model_payload.conversation
            ]

            # Support dict-format conversation {sender: message, ...} or list
            if isinstance(convo_raw, dict):
                convo_items = list(convo_raw.items())
            else:
                convo_items = convo_raw  # assume list-like

            if not convo_items:
                raise ValueError("'conversation' list cannot be empty")

            # Build contacts early so receiver heuristics work reliably
            for pname, pdetails in participants.items():
                _contact_for(pname, medium, pdetails)

            # Helper: extract (sender, content) from each entry while preserving order
            def _iter_messages():
                for entry in convo_items:
                    if isinstance(entry, str):
                        if ":" not in entry:
                            continue  # skip malformed string
                        sender, content = entry.split(":", 1)
                        yield sender.strip(), content.strip()
                    elif isinstance(entry, dict):
                        if "sender" in entry and "content" in entry:
                            yield str(entry["sender"]).strip(), str(
                                entry["content"],
                            ).strip()
                        elif len(entry) == 1:
                            sender, content = next(iter(entry.items()))
                            yield str(sender).strip(), str(content).strip()
                    # silently ignore anything else

            # Start time now, increment by one second per message to maintain order
            base_time = datetime.now(timezone.utc)

            for idx, (sender_name, content) in enumerate(_iter_messages()):
                sender_c = _contact_for(
                    sender_name,
                    medium,
                    participants.get(sender_name),
                )

                # Decide receiver – alternate between last speaker and fallback to first other participant / Assistant
                if last_sender_contact is not None and last_sender_contact != sender_c:
                    receiver_c = last_sender_contact
                else:
                    # Avoid mutating _name_to_contact during iteration which would
                    # raise `RuntimeError: dictionary changed size during iteration`.
                    _others = [c for c in _name_to_contact.values() if c != sender_c]
                    if _others:
                        receiver_c = _others[0]
                    else:
                        # Use the existing assistant contact (id == 0) instead of
                        # fabricating a new "Assistant" record.
                        receiver_c = 0

                last_sender_contact = sender_c

                timestamp = (base_time + timedelta(seconds=idx)).isoformat()

                msg_dict = {
                    "medium": medium,
                    "sender_id": sender_c,
                    "receiver_ids": [receiver_c],
                    "timestamp": timestamp,
                    "content": content,
                    "exchange_id": 0,
                }

                # Persist via TranscriptManager and local transcript list
                self._tm.log_messages([msg_dict], synchronous=True)
                # Optional stagger to visualise real-time callbacks
                if delay_per_message > 0:
                    import time  # local to avoid unnecessary global import at top

                    time.sleep(delay_per_message)
                # Store the *full* sender name so downstream maintenance commands
                # have unambiguous identifiers even when multiple contacts share
                # the same first name.
                transcript.append(
                    {
                        "sender": sender_name.strip(),
                        "content": content,
                        "timestamp": timestamp,
                        "medium": medium,
                    },
                )

                if self._in_cm:
                    # Emit comms messages following events.py schema
                    # Map medium to the appropriate Event class
                    from unity.conversation_manager.events import (
                        SMSMessageRecievedEvent,
                        EmailRecievedEvent,
                        WhatsappMessageRecievedEvent,
                        PhoneUtteranceEvent,
                    )
                    from pydantic import BaseModel
                    from unity.events.event_bus import Event
                    import unify

                    event_obj = None
                    if medium == "sms_message":
                        event_obj = SMSMessageRecievedEvent(
                            timestamp=timestamp,
                            content=content,
                            role="User",
                        )
                    elif medium == "email":
                        event_obj = EmailRecievedEvent(
                            timestamp=timestamp,
                            content=content,
                            role="User",
                        )
                    elif medium == "whatsapp_message":
                        event_obj = WhatsappMessageRecievedEvent(
                            timestamp=timestamp,
                            content=content,
                            role="User",
                        )
                    elif medium in ("phone_call", "whatsapp_call"):
                        # Log each utterance in a phone call context
                        event_obj = PhoneUtteranceEvent(
                            timestamp=timestamp,
                            role=sender_name.strip(),
                            content=content,
                        )
                    if event_obj:
                        ev_dict = event_obj.to_bus_event()
                        payload_dict = (
                            ev_dict.payload.model_dump(mode="json")
                            if isinstance(ev_dict.payload, BaseModel)
                            else Event._to_python(ev_dict.payload)
                        )
                        unify.create_logs(
                            project=unify.active_project(),
                            context="Assistant/Events/Comms",
                            params={},
                            entries={
                                "row_id": ev_dict.row_id,
                                "event_id": ev_dict.event_id,
                                "calling_id": ev_dict.calling_id,
                                "event_timestamp": ev_dict.timestamp.isoformat(),
                                "payload_cls": ev_dict.payload_cls,
                                "type": ev_dict.type,
                                **payload_dict,
                            },
                        )

            return f"{len(transcript)} messages logged"

        # ------------------------------------------------------------------ #
        #  Prompt that guides the LLM                                       #
        # ------------------------------------------------------------------ #

        prompt = (
            "You are a **Conversation Synthesis Assistant**. Your task is to fulfil the scenario description provided by the user.\n\n"
            "Tool usage policy:\n"
            "- If (and only if) the user explicitly asks to generate a conversation/transcript/messages/exchanges, then call the `submit_conversation` tool **exactly once** with a single JSON argument following the structure shown below.\n"
            "- If the user only asks to create or update contacts (and does not ask for a transcript), then use `update_contacts` as needed and finish without calling `submit_conversation`.\n"
            "- You may also use `update_contacts` before `submit_conversation` to ensure participants exist.\n\n"
            "`submit_conversation` payload shape:\n\n"
            "{\n"
            '  "medium": "phone_call|sms_message|email|whatsapp_message|whatsapp_call",\n'
            '  "participants": {\n'
            '      "Alice": { "phone_number": "+1555000001" },\n'
            '      "Bob":   { "email_address": "bob@example.com" }\n'
            "  },\n"
            '  "conversation": [\n'
            '      { "sender": "Alice", "content": "Hi Bob!" },\n'
            '      { "sender": "Bob",   "content": "Hi Alice, great to hear from you." }\n'
            "  ]\n"
            "}\n\n"
            f"When a transcript is requested and length is unspecified, aim for roughly {min_messages}-{max_messages} messages. "
            "Be concise – avoid unnecessary filler text. After you finish calling tools, do **not** output anything else."
        )

        # ------------------------------------------------------------------ #
        #  Inject existing contacts to discourage hallucinated surnames       #
        # ------------------------------------------------------------------ #

        try:
            cm = self._tm._contact_manager  # type: ignore[attr-defined]
            existing = cm._filter_contacts(limit=1000)
        except Exception:
            existing = []  # graceful fallback

        if existing:
            lines = []
            for c in existing:
                full = " ".join(p for p in [c.first_name, c.surname] if p)
                lines.append(f"• {full.strip()}")

            contact_block = (
                "\nExisting contacts (first names are unique):\n"
                + "\n".join(lines)
                + "\nAlways assume any participant whose first name appears in this list is the SAME person. "
                "Do NOT invent a different surname for them – reuse the exact full name provided (or omit the surname if unclear).\n"
            )
            prompt += contact_block

        prompt += f"The description is as follows:\n\n{description}."

        # Local import to avoid circular dependency: utils → scenario_builder → utils
        from sandboxes.scenario_builder import ScenarioBuilder  # noqa: WPS433

        builder = ScenarioBuilder(
            description=prompt,
            tools={
                "submit_conversation": submit_conversation,
                # Expose ContactManager.update so scenarios can explicitly create/update contacts
                "update_contacts": self._tm._contact_manager.update,
            },
            endpoint=self._endpoint,
            traced=self._traced,
            stateful=self._stateful,
        )

        await builder.create()

        # Allow empty transcripts when the user's request only involved contact creation/updates.
        return transcript


def activate_project(project_name: str, overwrite: bool = False) -> None:
    """
    Activate *project_name* and re-initialise the global EventBus singleton so
    that all subsequent Unify contexts (including those automatically created
    by EventBus) belong to that project.  Call this immediately after handling
    CLI arguments and before any manager instances are constructed.
    """
    import unity
    from unity.events.event_bus import EVENT_BUS

    # Force verbose Unify Request logging in sandbox runs
    try:
        os.environ["UNIFY_REQUESTS_DEBUG"] = "true"
    except Exception:
        pass

    unity.init(
        project_name,
        overwrite=("contexts" if overwrite else False),
    )
    # Clears all contexts in the EventBus
    EVENT_BUS.reset()

    # Set Trace Context
    import unify as _unify

    _unify.set_trace_context("Traces")
