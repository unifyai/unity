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
import wave
from contextlib import contextmanager
from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
from typing import List, Optional, Tuple, Any, Coroutine
from av import AudioFrame
import pyaudio
import math
import struct
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from livekit.plugins import cartesia
import argparse
from unity.common.llm_helpers import SteerableToolHandle

# Added for direct logging of generated messages
from unity.transcript_manager.transcript_manager import TranscriptManager
from sandboxes.scenario_builder import ScenarioBuilder

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
                return frame.to_wav_bytes()[44:]
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

    # ─────────────── clean-up ───────────────
    listener_done.set()
    listener.join(timeout=0.1)

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
        "--log_tcp_port",
        type=int,
        default=-1,
        metavar="PORT",
        help="serve logs over TCP on localhost:PORT (default -1 auto-picks an available port; 0 disables; >0 binds requested port)",
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
    log_file: Optional[str] = ".logs.txt",
    tcp_port: int = 0,
) -> None:
    """Configure logging to a file by default, with optional terminal streaming.

    - Overwrites the given log_file on each run.
    - Adds a StreamHandler to stdout when log_in_terminal is True.
    - Optionally serves logs over TCP on localhost:tcp_port for external viewing.
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

    if log_file:
        _fh = _logging.FileHandler(log_file, mode="w", encoding="utf-8")
        _fh.setFormatter(_fmt)
        root_logger.addHandler(_fh)

    if log_in_terminal:
        _sh = _logging.StreamHandler(_sys.stdout)
        _sh.setFormatter(_fmt)
        root_logger.addHandler(_sh)

    # Optional TCP broadcast for external terminals
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
            print(
                f"📡 Log stream on 127.0.0.1:{_actual} – connect via: nc 127.0.0.1 {_actual} (Ctrl-C to detach)",
            )
        except Exception as _exc:
            print(f"⚠️  Failed to start log TCP stream on port {tcp_port}: {_exc}")

    # Friendly hints
    if log_file:
        print(
            "📝 Logging to .logs.txt (overwrites each run). "
            "To follow live with scrollback: less +F .logs.txt (Ctrl-C to pause, F to resume, q to quit). "
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


def steering_controls_hint() -> str:
    """Return a one-line hint with available in-flight steering commands."""
    return (
        "Controls: /i <text>, /pause, /resume, /ask <q>, /freeform <text>, /stop, /help"
    )


async def await_with_interrupt(  # noqa: D401 – imperative helper
    handle: "SteerableToolHandle",
    poll: float = 0.05,
) -> str:
    """
    **Common wrapper** used by all interactive sandboxes.

    Waits on ``handle.result()`` but lets the user:
    • /i <text> or plain text     ⇒ interject via ``handle.interject``
    • /pause | /p                 ⇒ pause the running call
    • /resume | /r                ⇒ resume a paused call
    • /ask <question> | ? <q>     ⇒ ask a read-only question about the running call
    • /freeform <text>            ⇒ route free-form text to the best steering command via an LLM
    • /stop | /cancel             ⇒ abort the running call
    • /status                     ⇒ print whether the call is done
    • /help                       ⇒ show available controls

    Commands use a leading '/' prefix to avoid accidental interjections.
    """

    import asyncio  # local to avoid widening the public surface

    HELP_TEXT = steering_controls_hint()

    while not handle.done():
        txt = input_now(poll * 2)  # same cadence as old versions
        if txt:
            stripped = txt.strip()
            # Command mode with leading '/'
            if stripped.startswith("/"):
                parts = stripped[1:].split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in {"stop", "cancel", "s", "c"}:
                    print("stopping…")
                    handle.stop()
                    break
                if cmd in {"pause", "p"}:
                    try:
                        print("pausing…")
                        handle.pause()
                        print("⏸️  Paused")
                    except Exception as exc:
                        print(f"⚠️  Pause failed: {exc}")
                    continue
                if cmd in {"resume", "r"}:
                    try:
                        print("resuming…")
                        handle.resume()
                        print("▶️  Resumed")
                    except Exception as exc:
                        print(f"⚠️  Resume failed: {exc}")
                    continue
                if cmd in {"i", "interject"}:
                    if not arg:
                        print("Usage: /i <text>")
                    else:
                        # Log what is being interjected
                        print(f"interjecting: {arg}")
                        run_in_loop(handle.interject(arg))
                    continue
                if cmd in {"ask", "?"}:
                    if not arg:
                        print("Usage: /ask <question>")
                    else:
                        try:
                            # Log what is being asked
                            print(f"asking question: {arg}")
                            nested = await handle.ask(arg)
                            ans = await nested.result()
                            print(f"[ask] → {ans}")
                        except Exception as exc:
                            print(f"⚠️  Ask failed: {exc}")
                    continue
                if cmd in {"freeform", "f"}:
                    if not arg:
                        print("Usage: /freeform <text>")
                        continue
                    try:
                        # Lightweight LLM router to map free-form text to a steering action
                        from pydantic import BaseModel, Field
                        import unify as _unify

                        class _SteeringIntent(BaseModel):
                            action: str = Field(
                                ...,
                                pattern="^(ask|interject|pause|resume|stop|status)$",
                            )
                            question: str | None = None
                            cleaned_text: str | None = None

                        _SYS = (
                            "You are a router that maps a user's free-form message to one of these steering commands: "
                            "'ask' (include a specific, concise question in 'question'), "
                            "'interject' (include the text to inject in 'cleaned_text'), "
                            "'pause', 'resume', 'stop', or 'status'.\n"
                            "Rules:\n"
                            "- If the user requests progress, status, or 'how is it going', prefer 'ask' with a concrete question like 'what is the current task progress?'.\n"
                            "- If the user gives guidance or additional info to incorporate, choose 'interject' and pass it via 'cleaned_text'.\n"
                            "- Polite commands like 'please pause' → 'pause'. 'continue' → 'resume'. 'cancel/abort/stop' → 'stop'.\n"
                            "- Return ONLY JSON matching the response schema."
                        )

                        _judge = _unify.Unify(
                            "gpt-4o@openai",
                            response_format=_SteeringIntent,
                        )
                        _intent = _SteeringIntent.model_validate_json(
                            _judge.set_system_message(_SYS).generate(arg),
                        )

                        if _intent.action == "ask":
                            q = (_intent.question or _intent.cleaned_text or "").strip()
                            if not q:
                                q = "What is the current task progress?"
                            print(f"asking question: {q}")
                            nested = await handle.ask(q)
                            ans = await nested.result()
                            print(f"[ask] → {ans}")
                        elif _intent.action == "interject":
                            txt_to_inject = (_intent.cleaned_text or arg).strip()
                            if not txt_to_inject:
                                print(
                                    "⚠️  Router produced empty interjection – ignoring",
                                )
                            else:
                                print(f"interjecting: {txt_to_inject}")
                                run_in_loop(handle.interject(txt_to_inject))
                        elif _intent.action == "pause":
                            try:
                                print("pausing…")
                                handle.pause()
                                print("⏸️  Paused")
                            except Exception as exc:
                                print(f"⚠️  Pause failed: {exc}")
                        elif _intent.action == "resume":
                            try:
                                print("resuming…")
                                handle.resume()
                                print("▶️  Resumed")
                            except Exception as exc:
                                print(f"⚠️  Resume failed: {exc}")
                        elif _intent.action == "stop":
                            print("stopping…")
                            handle.stop()
                            break
                        elif _intent.action == "status":
                            print("status requested")
                            print("done" if handle.done() else "running")
                        else:
                            # Fallback to interject if unknown
                            print(f"interjecting: {arg}")
                            run_in_loop(handle.interject(arg))
                    except Exception as exc:
                        print(f"⚠️  Freeform routing failed: {exc}")
                    continue
                if cmd in {"status", "st"}:
                    print("status requested")
                    print("done" if handle.done() else "running")
                    continue
                if cmd in {"help", "h"}:
                    print(HELP_TEXT)
                    continue
                # Unknown command → treat as interjection without the '/'
                print(f"interjecting: {stripped[1:]}")
                run_in_loop(handle.interject(stripped[1:]))
            else:
                # Plain text → interject
                print(f"interjecting: {stripped}")
                run_in_loop(handle.interject(stripped))
        await asyncio.sleep(poll)

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
        endpoint: str = "o4-mini@openai",
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

        # Cache of *first_name → Contact* so repeated references to the same
        # person (even if the LLM adds/changes a surname) always map to the
        # same Contact instance.
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

            # 1️⃣  Attempt to reuse an existing contact (sandbox rule: first names are unique)
            try:
                cm = self._tm._contact_manager  # ContactManager instance
                first_name = name.split(" ")[0].lower()

                # Attempt 1: exact case-insensitive match
                match = cm._search_contacts(
                    filter=f"first_name.lower() == '{first_name}'",
                    limit=1,
                )

                # Attempt 2: prefix match (e.g. 'dan' → 'daniel') if nothing found
                if not match:
                    match = cm._search_contacts(
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
            key = name.split(" ")[0].lower()
            if key not in _name_to_contact:
                _name_to_contact[key] = _build_contact(name, medium, details)
            return _name_to_contact[key]

        def submit_conversation(
            payload: dict,
        ) -> str:  # noqa: C901 – complex but self-contained
            """Parse the high-level *conversation* JSON coming from the LLM.

            Expected schema (keys are case-sensitive):

            {
                "medium": "phone_call" | "sms_message" | "email" | "whatsapp_message" | "whatsapp_call",
                "participants": {
                    "Alice": {"phone_number": "+1…"},
                    "Bob":   {"email_address": "bob@example.com"}
                },
                "conversation": [
                    {"sender": "Alice", "content": "Hi Bob!"},
                    {"sender": "Bob",   "content": "Hi Alice, great to hear from you."}
                ]
            }
            """

            nonlocal transcript, last_sender_contact

            # Accept either a dict *object* or a JSON *string*
            if isinstance(payload, str):
                import json as _json

                try:
                    payload = _json.loads(payload)
                except Exception as exc:
                    raise ValueError(
                        "submit_conversation: string payload must be valid JSON",
                    ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    "submit_conversation expects a dict or JSON string argument",
                )

            medium = str(payload.get("medium", "sms_message"))
            participants: dict[str, Any] = payload.get("participants", {}) or {}
            convo_raw = payload.get("conversation", [])

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
                        receiver_c = _contact_for("Assistant", medium, {})

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
            "You are a **Conversation Synthesis Assistant**. Your task is to invent a realistic conversation that fulfils the scenario description provided by the user. "
            "When you are ready, call the `submit_conversation` tool *exactly once* with a single JSON argument following this structure:\n\n"
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
            f"If the scenario doesn't specify how long the chat should be, aim for roughly {min_messages}-{max_messages} messages. "
            "Be concise – avoid unnecessary filler text. After you have called the tool, do **not** output anything else."
        )

        # ------------------------------------------------------------------ #
        #  Inject existing contacts to discourage hallucinated surnames       #
        # ------------------------------------------------------------------ #

        try:
            cm = self._tm._contact_manager  # type: ignore[attr-defined]
            existing = cm._search_contacts(limit=1000)
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

        builder = ScenarioBuilder(
            description=prompt,
            tools={"submit_conversation": submit_conversation},
            endpoint=self._endpoint,
            traced=self._traced,
            stateful=self._stateful,
        )

        await builder.create()

        if not transcript:
            raise RuntimeError("TranscriptGenerator produced an empty transcript.")

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

    unity.init(
        project_name,
        overwrite=("contexts" if overwrite else False),
    )
    # Clears all contexts in the EventBus
    EVENT_BUS.reset()
