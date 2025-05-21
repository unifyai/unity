Project Path: tmp.XrIc9Z1RTx

Source Tree:

```
tmp.XrIc9Z1RTx
├── sandboxes
│   ├── controller.py
│   ├── transcript.py
│   ├── __init__.py
│   ├── __pycache__
│   ├── utils.py
│   ├── tasklist.py
│   └── knowledge.py
├── knowledge_manager
│   ├── __init__.py
│   ├── types.py
│   ├── __pycache__
│   ├── knowledge_manager.py
│   └── sys_msgs.py
├── task_manager
│   ├── __init__.py
│   ├── sys_msgs.py
│   └── task_manager.py
├── tests
│   ├── assertion_helpers.py
│   ├── test_log_term_memory_flow.py
│   ├── test_communication
│   │   ├── test_transcript_manager
│   │   │   ├── test_ask.py
│   │   │   ├── test_mock_scenario_transcript.py
│   │   │   ├── test_basics.py
│   │   │   ├── test_transcript_embedding.py
│   │   │   └── __pycache__
│   │   ├── __init__.py
│   │   └── __pycache__
│   ├── conftest.py
│   ├── test_knowledge
│   │   ├── test_mock_scenario_knowledge.py
│   │   ├── test_tables.py
│   │   ├── test_search.py
│   │   ├── test_store_and_retrieve.py
│   │   ├── __pycache__
│   │   ├── test_columns.py
│   │   ├── test_add_data.py
│   │   └── test_knowledge_embedding.py
│   ├── test_llm_helpers
│   │   ├── test_async_tools_simple.py
│   │   ├── __pycache__
│   │   ├── test_async_tools_interject_and_stop.py
│   │   └── test_schemas.py
│   ├── __init__.py
│   ├── __pycache__
│   ├── test_event_bus
│   │   ├── __pycache__
│   │   ├── test_prefill.py
│   │   ├── test_get_latest.py
│   │   ├── test_publish.py
│   │   └── test_windows.py
│   ├── test_task_list
│   │   ├── test_mock_scenario_tasklist.py
│   │   ├── test_update_complex.py
│   │   ├── test_creation_deletion.py
│   │   ├── test_cancel_tasks.py
│   │   ├── test_pause_continue_active_task.py
│   │   ├── __pycache__
│   │   ├── test_update.py
│   │   ├── test_update_tools.py
│   │   ├── test_tasklist_embedding.py
│   │   ├── test_tasklist_ask.py
│   │   └── test_task_queue.py
│   ├── test_update_text_complex.py
│   ├── test_controller
│   │   ├── test_action_filter.py
│   │   ├── test_agent.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── test_command_runner.py
│   │   └── test_controller_public.py
│   ├── helpers.py
│   └── test_function_manager
│       ├── __pycache__
│       └── test_function_manager.py
├── common
│   ├── embed_utils.py
│   ├── __init__.py
│   ├── __pycache__
│   └── llm_helpers.py
├── task_list_manager
│   ├── types
│   │   ├── task.py
│   │   ├── priority.py
│   │   ├── __pycache__
│   │   ├── repetition.py
│   │   ├── schedule.py
│   │   └── status.py
│   ├── __init__.py
│   ├── __pycache__
│   ├── sys_msgs.py
│   └── task_list_manager.py
├── communication
│   ├── types
│   │   ├── __init__.py
│   │   ├── message.py
│   │   ├── __pycache__
│   │   ├── summary.py
│   │   └── contact.py
│   ├── __init__.py
│   ├── __pycache__
│   ├── sys_msgs.py
│   └── transcript_manager
│       ├── transcript_manager.py
│       ├── __pycache__
│       └── sys_msgs.py
└── events
    ├── types
    │   ├── __init__.py
    │   ├── message.py
    │   ├── __pycache__
    │   └── message_exchange_summary.py
    ├── __pycache__
    └── event_bus.py

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/sandboxes/controller.py`:

```py
"""
Entry point.  GUI in main thread, BrowserWorker in background thread.
No Playwright code touches the Tk thread.
"""

import queue
import logging
import sys, pathlib

# Ensure repository root is on PYTHONPATH so `import unity` works when this
# script is executed directly from inside the "sandboxes" folder.
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(threadName)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("unity")

load_dotenv()

from unity.controller.gui import ControlPanel
from unity.controller.playwright.worker import BrowserWorker


def main() -> None:

    # queue for user commands only (GUI → redis)
    gui_to_browser_queue: queue.Queue[str] = queue.Queue(maxsize=50)

    # Start BrowserWorker (publishes browser_state on redis)
    worker = BrowserWorker(
        start_url="https://www.google.com/",
        refresh_interval=0.4,
        log=log.debug,
    )
    worker.start()

    # Redis publisher thread for commands
    import redis, threading

    r = redis.Redis(host="localhost", port=6379, db=0)

    def _cmd_forwarder():
        while True:
            cmd = gui_to_browser_queue.get()
            r.publish("browser_command", cmd)

    threading.Thread(target=_cmd_forwarder, daemon=True).start()

    # launch Tk GUI (pulls browser_state directly from redis)
    gui = ControlPanel(gui_to_browser_queue)

    try:
        gui.mainloop()
    finally:
        worker.stop()
        worker.join(timeout=2)


if __name__ == "__main__":
    main()

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/sandboxes/transcript.py`:

```py
"""transcript_sandbox.py  (voice mode, Deepgram SDK v4, sync)
================================================================
Interactive sandbox for **TranscriptManager**.

All shared voice helpers now live in `utils.py` to avoid duplication
with the task‑list sandbox.
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import json
import logging
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unify

from unity.constants import LOGGER as _LG  # type: ignore
from unity.communication.transcript_manager.transcript_manager import TranscriptManager  # type: ignore
from unity.communication.types.message import Message  # type: ignore

# Voice helpers (PortAudio capture, Deepgram STT, Cartesia TTS)

# ---------------------------------------------------------------------------
# Scenario seeding data
# ---------------------------------------------------------------------------

_CONTACTS: List[dict] = [
    dict(
        first_name="Carlos",
        surname="Diaz",
        email_address="carlos.diaz@example.com",
        phone_number="+14155550000",
        whatsapp_number="+14155550000",
    ),
    dict(
        first_name="Dan",
        surname="Turner",
        email_address="dan.turner@example.com",
        phone_number="+447700900001",
        whatsapp_number="+447700900001",
    ),
    dict(
        first_name="Julia",
        surname="Nguyen",
        email_address="julia.nguyen@example.com",
        phone_number="+447700900002",
        whatsapp_number="+447700900002",
    ),
    dict(
        first_name="Jimmy",
        surname="O'Brien",
        email_address="jimmy.obrien@example.com",
        phone_number="+61240011000",
        whatsapp_number="+61240011000",
    ),
    dict(
        first_name="Anne",
        surname="Fischer",
        email_address="anne.fischer@example.com",
        phone_number="+49891234567",
        whatsapp_number="+49891234567",
    ),
    dict(
        first_name="Leo",
        surname="Kowalski",
        email_address="leo.k@example.com",
        phone_number="+48500100200",
        whatsapp_number="+48500100200",
    ),
    dict(
        first_name="Sara",
        surname="Jensen",
        email_address="sara.j@example.com",
        phone_number="+46700111222",
        whatsapp_number="+46700111222",
    ),
]

_ID_BY_NAME: Dict[str, int] = {}


def _seed_fixed(tm: TranscriptManager) -> None:
    "Populate contacts and a rich set of exchanges."
    for idx, c in enumerate(_CONTACTS):
        tm.create_contact(**c)
        _ID_BY_NAME[c["first_name"].lower()] = idx

    base = datetime(2025, 4, 20, 9, 0, tzinfo=timezone.utc)
    ex_id = 0

    def _cid(name: str) -> int:
        return _ID_BY_NAME[name]

    # Dan ↔ Julia phone calls
    tm.log_messages(
        [
            Message(
                medium="phone_call",
                sender_id=_cid("dan"),
                receiver_id=_cid("julia"),
                timestamp=base.isoformat(),
                content="Hi Julia, it's Dan. Quick check‑in about Q2 metrics.",
                exchange_id=ex_id,
            ),
            Message(
                medium="phone_call",
                sender_id=_cid("julia"),
                receiver_id=_cid("dan"),
                timestamp=(base + timedelta(seconds=30)).isoformat(),
                content="Sure Dan, ready when you are.",
                exchange_id=ex_id,
            ),
        ],
    )
    ex_id += 1

    later = base + timedelta(days=6, minutes=30)
    tm.log_messages(
        [
            Message(
                medium="phone_call",
                sender_id=_cid("dan"),
                receiver_id=_cid("julia"),
                timestamp=later.isoformat(),
                content="Morning Julia – finalising the London event agenda today.",
                exchange_id=ex_id,
            ),
            Message(
                medium="phone_call",
                sender_id=_cid("julia"),
                receiver_id=_cid("dan"),
                timestamp=(later + timedelta(seconds=45)).isoformat(),
                content="Great. Let's confirm the speaker list and coffee budget.",
                exchange_id=ex_id,
            ),
        ],
    )
    ex_id += 1

    # Carlos bulk‑order email chain
    t_email = base + timedelta(days=1, hours=3)
    tm.log_messages(
        [
            Message(
                medium="email",
                sender_id=_cid("carlos"),
                receiver_id=_cid("dan"),
                timestamp=t_email.isoformat(),
                content="Subject: Stapler bulk order\n\nHi Dan, I'm **interested in buying 200 units** of your new stapler. Can you quote?\n\nThanks,\nCarlos",
                exchange_id=ex_id,
            ),
            Message(
                medium="email",
                sender_id=_cid("dan"),
                receiver_id=_cid("carlos"),
                timestamp=(t_email + timedelta(hours=2)).isoformat(),
                content="Hi Carlos — sure, $4.50 per unit. See attached PDF.",
                exchange_id=ex_id,
            ),
        ],
    )
    ex_id += 1

    # Jimmy holiday heads‑up
    t_holiday = base + timedelta(days=2, hours=9, minutes=10)
    tm.log_messages(
        [
            Message(
                medium="whatsapp_message",
                sender_id=_cid("jimmy"),
                receiver_id=_cid("dan"),
                timestamp=t_holiday.isoformat(),
                content="Heads‑up Dan, I'll be **on holiday from 2025‑05‑15 to 2025‑05‑30**. Ping me before that if urgent.",
                exchange_id=ex_id,
            ),
        ],
    )
    ex_id += 1

    # Anne passport excuse
    t_excuse = base + timedelta(days=3)
    tm.log_messages(
        [
            Message(
                medium="whatsapp_message",
                sender_id=_cid("anne"),
                receiver_id=_cid("dan"),
                timestamp=t_excuse.isoformat(),
                content="Sorry Dan, I can't join the Berlin trip because my passport expired last week.",
                exchange_id=ex_id,
            ),
        ],
    )
    ex_id += 1

    # Leo logistics SMS
    t_log = base + timedelta(days=4, hours=13)
    tm.log_messages(
        [
            Message(
                medium="sms_message",
                sender_id=_cid("leo"),
                receiver_id=_cid("dan"),
                timestamp=t_log.isoformat(),
                content="Dan, the pallets are delayed at customs. Need new clearance docs by tomorrow.",
                exchange_id=ex_id,
            ),
            Message(
                medium="sms_message",
                sender_id=_cid("dan"),
                receiver_id=_cid("leo"),
                timestamp=(t_log + timedelta(minutes=7)).isoformat(),
                content="On it – forwarding the updated invoices now.",
                exchange_id=ex_id,
            ),
        ],
    )
    ex_id += 1

    # Sara voice‑note (text representation)
    t_voice = base + timedelta(days=5, hours=17)
    tm.log_messages(
        [
            Message(
                medium="whatsapp_message",
                sender_id=_cid("sara"),
                receiver_id=_cid("dan"),
                timestamp=t_voice.isoformat(),
                content="(voice) Hey Dan! Quick thing – keynote speaker wants to swap slots with the panel. OK?",
                exchange_id=ex_id,
            ),
        ],
    )
    ex_id += 1

    # Filler exchanges for noise
    random.seed(42)
    media = ["email", "phone_call", "sms_message", "whatsapp_message"]
    filler_start = base + timedelta(days=6)
    for extra in range(25):
        mid = random.choice(media)
        a, b = random.sample(list(_ID_BY_NAME.values()), 2)
        ex_ref = ex_id + extra
        msgs: List[Message] = []
        for i in range(random.randint(5, 15)):
            msgs.append(
                Message(
                    medium=mid,
                    sender_id=a if i % 2 else b,
                    receiver_id=b if i % 2 else a,
                    timestamp=(
                        filler_start + timedelta(minutes=extra * 4 + i)
                    ).isoformat(),
                    content=f"Filler {ex_ref}-{i} {mid} banter.",
                    exchange_id=ex_ref,
                ),
            )
        tm.log_messages(msgs)

    tm.summarize(exchange_ids=[0, 1])


def _seed_llm(tm: TranscriptManager) -> Optional[str]:
    prompt = (
        "Create a realistic communication history for a European event agency with 8‑12 contacts interacting via email, phone_call, sms_message and whatsapp_message. "
        "Include 60‑90 exchanges with 3‑8 messages each, timestamps between 2025‑04‑01 and 2025‑05‑10 (UTC). "
        "Return JSON with keys: contacts, exchanges, theme."
    )

    client = unify.Unify("o4-mini@openai", cache=True)
    client.set_system_message(prompt)
    raw = client.generate("Produce scenario").strip()

    try:
        payload = json.loads(raw)
    except Exception:
        _LG.warning("LLM scenario failed – using fixed seed.")
        _seed_fixed(tm)
        return None

    for idx, c in enumerate(payload["contacts"]):
        tm.create_contact(**c)
        _ID_BY_NAME[c["first_name"].lower()] = idx

    for ex_id, ex in enumerate(payload["exchanges"]):
        mid = ex.get("medium", "email")
        msgs = [
            Message(
                medium=mid,
                sender_id=_ID_BY_NAME[m["sender_first_name"].lower()],
                receiver_id=_ID_BY_NAME[m["receiver_first_name"].lower()],
                timestamp=m["timestamp"],
                content=m["content"],
                exchange_id=ex_id,
            )
            for m in ex["messages"]
        ]
        tm.log_messages(msgs)

    if summaries := [
        e.get("summary") for e in payload["exchanges"] if e.get("summary")
    ]:
        tm.summarize(texts=summaries)

    return payload.get("theme")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


async def _dispatch(
    tm: TranscriptManager,
    raw: str,
    *,
    show_steps: bool,
) -> Tuple[str, str, List | None]:
    handle = tm.ask(raw.strip(), return_reasoning_steps=show_steps)

    # Create a task for the result
    answer_task = asyncio.create_task(handle.result())

    # Set up a way to check for user input
    stop_event = threading.Event()

    def check_input():
        from sandboxes.utils import run_in_loop

        try:
            user_input = input("Press Enter to interrupt or type 'stop' to cancel...\n")
            if user_input.lower().strip() == "stop":
                handle.stop()
                print("Stopping the current operation...")
            elif user_input.strip():
                run_in_loop(handle.interject(user_input))
                print(f"Added interjection: {user_input}")
            stop_event.set()
        except Exception as e:
            print(f"Error in input thread: {e}")
            stop_event.set()

    # Start a thread to check for user input
    input_thread = threading.Thread(target=check_input, daemon=True)
    input_thread.start()

    try:
        # Wait for either the answer or user interruption
        result = await answer_task
        steps = None
        if show_steps and isinstance(result, tuple):
            result, steps = result
        return "ask", result, steps
    except asyncio.CancelledError:
        return "ask", "Operation was cancelled.", None
    finally:
        # Clean up
        if not stop_event.is_set():
            stop_event.set()
        if input_thread.is_alive():
            input_thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="TranscriptManager sandbox with Deepgram voice mode",
    )
    parser.add_argument(
        "--voice",
        "-v",
        action="store_true",
        help="enable voice capture/playback",
    )
    parser.add_argument("--scenario", choices=["fixed", "llm"], default="fixed")
    parser.add_argument("--new", "-n", action="store_true", help="wipe & reseed data")
    parser.add_argument(
        "--silent",
        "-s",
        action="store_true",
        help="suppress tool logs",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="verbose HTTP/LLM logging",
    )
    args = parser.parse_args()

    if not args.silent:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        _LG.setLevel(logging.INFO)
        if not args.debug:
            for noisy in ("unify", "unify.utils", "unify.logging", "requests", "httpx"):
                logging.getLogger(noisy).setLevel(logging.WARNING)

    # Create a TranscriptManager
    from unity.events.event_bus import EventBus

    eb = EventBus()
    tm = TranscriptManager(eb)

    # Seed with data
    if args.new or not unify.get_logs(context=tm._contacts_ctx):
        print("Seeding with fresh data…")
        if args.scenario == "llm":
            theme = _seed_llm(tm)
            if theme:
                print(f"Scenario theme: {theme}")
        else:
            _seed_fixed(tm)

    # Import voice helpers only if needed
    if args.voice:
        from sandboxes.utils import record_until_enter, transcribe_deepgram, speak

    # Main loop
    print("\n=== TranscriptManager Sandbox ===")
    print("Type 'exit' or Ctrl+C to quit.")
    print("Type 'help' for available commands.")
    print("During processing: press Enter to interrupt or type 'stop' to cancel.")

    while True:
        try:
            if args.voice:
                print("\nListening for voice input…")
                audio = record_until_enter()
                raw = transcribe_deepgram(audio)
                print(f"You said: {raw}")
            else:
                raw = input("\n> ")

            if not raw or raw.lower() in ("exit", "quit"):
                break

            if raw.lower() == "help":
                print("\nAvailable commands:")
                print("  help - Show this help message")
                print("  exit, quit - Exit the sandbox")
                continue

            # Dispatch to the appropriate handler
            try:
                cmd, result, steps = await _dispatch(tm, raw, show_steps=args.debug)
                print(f"\n{result}")

                if args.voice:
                    speak(result)

                if steps and args.debug:
                    print("\nReasoning steps:")
                    for i, step in enumerate(steps):
                        print(f"{i+1}. {step['role']}: {step['content']}")
            except Exception as e:
                print(f"Error: {e}")

        except KeyboardInterrupt:
            print("\nExiting…")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")


def main():
    """Entry point that runs the async main function."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/sandboxes/utils.py`:

```py
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
from typing import Coroutine, List, Optional, Tuple, Union

import pyaudio
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from livekit.plugins import cartesia

from dotenv import load_dotenv

# Import platform-specific modules for non-blocking input
if platform.system() == "Windows":
    import msvcrt

load_dotenv()


def run_in_loop(coro: Coroutine) -> None:
    """Schedule coro in the running event loop from any thread."""
    loop = asyncio.get_running_loop()
    loop.call_soon_threadsafe(asyncio.create_task, coro)


# ---------------------------------------------------------------------------
# Audio / PortAudio boilerplate
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

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


def speak(text: str):
    """Speak *text* via Cartesia TTS – press ↵ to interrupt playback."""
    if "CARTESIA_API_KEY" not in os.environ:
        return

    print("🗣️ Assistant speaking… press ↵ to skip.")

    async def _gen() -> bytes:
        async with aiohttp.ClientSession() as s:
            tts = cartesia.TTS(http_session=s)
            stream = tts.synthesize(text)

            # Collect the entire synthesis into one object
            frame = await stream.collect()

            # Newest SDK exposes to_pcm_bytes()
            if hasattr(frame, "to_pcm_bytes"):
                return frame.to_pcm_bytes()

            # Older SDK exposes data attr
            if hasattr(frame, "data"):
                return bytes(frame.data)

            # Fallback: to_wav_bytes() → strip 44-byte header
            if hasattr(frame, "to_wav_bytes"):
                wav = frame.to_wav_bytes()
                return wav[44:]

            # Last resort – try bytes() conversion
            return bytes(frame)

    pcm = asyncio.run(_gen())
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

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/sandboxes/tasklist.py`:

```py
"""tasklist_sandbox.py  (voice mode, Deepgram SDK v4, sync)
===========================================================
Interactive sandbox for **TaskListManager** with optional hands‑free
voice input. All shared audio/STT/TTS helpers are imported from
`utils.py` to avoid duplication with other sandboxes.
"""

from __future__ import annotations

import argparse
import asyncio
import select
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unify

from unity.constants import LOGGER as _LG  # type: ignore
from unity.common import AsyncToolLoopHandle  # type: ignore
from unity.task_list_manager.task_list_manager import TaskListManager  # type: ignore
from unity.task_list_manager.types.priority import Priority  # type: ignore
from unity.task_list_manager.types.schedule import Schedule  # type: ignore
from tests.test_task_list.test_update_complex import _next_weekday  # type: ignore
from sandboxes.utils import (
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    run_in_loop,
)  # type: ignore


# ---------------------------------------------------------------------------
# Scenario seeding helpers (fixed + LLM)
# ---------------------------------------------------------------------------


def _seed_fixed(tlm: TaskListManager) -> None:
    """Populate a small but varied set of starter tasks."""
    tlm._create_task(
        name="Write quarterly report",
        description="Compile and draft the Q2 report for management.",
        status="active",
    )
    tlm._create_task(
        name="Prepare slide deck",
        description="Create slides for the upcoming board meeting.",
        status="queued",
    )
    tlm._create_task(
        name="Client follow‑up email",
        description="Send follow‑up email about the proposal.",
        status="queued",
    )

    base = datetime.now(timezone.utc)
    next_mon = _next_weekday(base, 0).replace(hour=9, minute=0, second=0, microsecond=0)
    tlm._create_task(
        name="Send KPI report",
        description="Automated email of KPIs to leadership.",
        schedule=Schedule(
            start_time=next_mon.isoformat(),
            prev_task=None,
            next_task=None,
        ),
        priority=Priority.high,
    )
    tlm._create_task(
        name="Deploy new release",
        description="Roll out version 2.0 to production servers.",
        status="paused",
    )


def _seed_llm(tlm: TaskListManager) -> Optional[str]:
    """Generate a large realistic task backlog via LLM."""
    prompt = (
        "Generate a realistic task list for a small business. Pick a coherent theme. "
        "Create 110‑140 tasks across queues with positions, priorities & ISO start times. "
        "Return only JSON with top‑level 'tasks' and optional 'theme'."
    )
    client = unify.Unify("o4-mini@openai", cache=True)
    client.set_system_message(prompt)
    raw = client.generate("Produce scenario").strip()

    try:
        payload = json.loads(raw)
    except Exception:
        _LG.warning("LLM scenario failed – using fixed seed.")
        _seed_fixed(tlm)
        return None

    theme = payload.get("theme")
    tasks = payload["tasks"]

    # Group by queue_group, sort by queue_position for stable insertion order
    groups: Dict[str, List[dict]] = {}
    for t in tasks:
        groups.setdefault(t.get("queue_group", "default"), []).append(t)
    for g in groups.values():
        g.sort(key=lambda d: d.get("queue_position", 0))

    id_map: Dict[Tuple[str, int], int] = {}
    for g_name, g in groups.items():
        for idx, entry in enumerate(g):
            kwargs = {
                "name": entry["name"],
                "description": entry["description"],
                "status": entry.get("status", "queued"),
                "priority": entry.get("priority", Priority.normal),
            }
            if start := entry.get("start_time"):
                kwargs["schedule"] = Schedule(
                    start_time=start,
                    prev_task=None,
                    next_task=None,
                )
            task_id = tlm._create_task(**kwargs)
            id_map[(g_name, idx)] = task_id

    # Wire up prev/next links inside each queue group
    for g_name, g in groups.items():
        for idx, _ in enumerate(g):
            cur = id_map[(g_name, idx)]
            prev_ = id_map.get((g_name, idx - 1)) if idx > 0 else None
            next_ = id_map.get((g_name, idx + 1)) if idx < len(g) - 1 else None
            unify.update_logs(
                context="Tasks",
                logs=tlm._get_logs_by_task_ids(task_ids=cur),
                entries={"schedule": {"prev_task": prev_, "next_task": next_}},
                overwrite=True,
            )
    return theme


# ---------------------------------------------------------------------------
# Natural‑language dispatcher (ask vs update)
# ---------------------------------------------------------------------------


class _DispatchResp(BaseModel):
    require_update: bool = Field(...)
    fixed_text: str = Field(...)


def _dispatch(tlm: TaskListManager, raw: str, *, show_steps: bool):
    raw = raw.strip()

    llm = unify.Unify("gpt-4o@openai", response_format=_DispatchResp)
    llm.set_system_message(
        "There is a table containing a list of tasks, and all of their properties. "
        "The user has made a request via a speech‑to‑text process, which can introduce errors. "
        "Using the output schema provided, output a corrected transcript and decide whether the task table must be updated.",
    )
    resp = _DispatchResp.model_validate_json(llm.generate(raw))

    if resp.require_update:
        handle = tlm.update(
            text=resp.fixed_text,
            return_reasoning_steps=show_steps,
            log_tool_steps=show_steps,
        )
        return "update", handle

    handle = tlm.ask(
        text=resp.fixed_text,
        return_reasoning_steps=show_steps,
        log_tool_steps=show_steps,
    )
    return "ask", handle


# ---------------------------------------------------------------------------
# Input helpers for interruption detection
# ---------------------------------------------------------------------------


def _non_blocking_input(prompt: str = "", timeout: float = 0.1) -> Optional[str]:
    """Check for user input without blocking, with a small timeout."""
    ready_to_read = select.select([sys.stdin], [], [], timeout)[0]
    if ready_to_read:
        return sys.stdin.readline().strip()
    return None


def _poll_for_interruption(handle: AsyncToolLoopHandle) -> None:
    """Poll for user input to interrupt or stop the current operation."""
    print("⏳ Processing... Type anything to interrupt, or 'stop'/'cancel' to abort.")

    while not handle.done():
        user_input = _non_blocking_input(timeout=0.1)
        if user_input:
            if user_input.lower() in ("stop", "cancel"):
                print("🛑 Stopping operation...")
                handle.stop()
                return
            else:
                print(f"💬 Interjecting: {user_input}")
                run_in_loop(handle.interject(user_input))
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def _process_voice_input(
    tlm: TaskListManager, audio_bytes: bytes, show_steps: bool
):
    """Process voice input with interruption support."""
    user_text = _transcribe_deepgram(audio_bytes).strip()
    if not user_text:
        return

    print(f"▶️  {user_text}")
    if user_text.lower() in {"quit", "exit"}:
        return "quit"

    _speak("Working on this now…")
    kind, handle = _dispatch(tlm, user_text, show_steps=show_steps)

    # Start a background thread to poll for interruptions
    threading.Thread(target=_poll_for_interruption, args=(handle,), daemon=True).start()

    # Wait for the result
    try:
        result = await handle.result()
        print(f"[{kind}] => {result}\n")
        _speak(result)
    except asyncio.CancelledError:
        print("Operation was cancelled.")


async def _process_text_input(tlm: TaskListManager, text: str, show_steps: bool):
    """Process text input with interruption support."""
    if text.lower() in {"quit", "exit"}:
        return "quit"

    if not text:
        return

    kind, handle = _dispatch(tlm, text, show_steps=show_steps)

    # Start a background thread to poll for interruptions
    threading.Thread(target=_poll_for_interruption, args=(handle,), daemon=True).start()

    # Wait for the result
    try:
        result = await handle.result()
        print(f"[{kind}] => {result}\n")
    except asyncio.CancelledError:
        print("Operation was cancelled.")


async def _async_main():
    parser = argparse.ArgumentParser(
        description="TaskListManager sandbox with minimalist voice mode (Deepgram v4, Cartesia)",
    )
    parser.add_argument(
        "--voice",
        "-v",
        action="store_true",
        help="enable voice capture/playback",
    )
    parser.add_argument("--scenario", choices=["fixed", "llm"], default="fixed")
    parser.add_argument("--new", "-n", action="store_true", help="wipe & reseed data")
    parser.add_argument(
        "--silent",
        "-s",
        action="store_true",
        help="suppress tool logs",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="verbose HTTP/LLM logging",
    )
    args = parser.parse_args()

    # Logging
    if not args.silent:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        _LG.setLevel(logging.INFO)
        if not args.debug:
            for noisy in ("unify", "unify.utils", "unify.logging", "requests", "httpx"):
                logging.getLogger(noisy).setLevel(logging.WARNING)

    # Unify context
    unify.activate("TaskListSandbox")
    fresh = "Tasks" not in unify.get_contexts() or args.new
    unify.set_context("Tasks", overwrite=fresh)

    # Manager
    tlm = TaskListManager()

    if fresh:
        if args.scenario == "llm":
            _seed_llm(tlm)
        else:
            _seed_fixed(tlm)

    print("TaskListManager sandbox – speak or type. 'quit' to exit.\n")

    # Interaction loop
    if args.voice:
        while True:
            audio_bytes = _record_until_enter()
            result = await _process_voice_input(
                tlm, audio_bytes, show_steps=not args.silent
            )
            if result == "quit":
                break
    else:
        try:
            while True:
                line = input("> ").strip()
                result = await _process_text_input(
                    tlm, line, show_steps=not args.silent
                )
                if result == "quit":
                    break
        except (EOFError, KeyboardInterrupt):
            print()


def main() -> None:
    """Entry point that runs the async main function."""
    import select  # Import here to avoid issues on Windows

    asyncio.run(_async_main())


if __name__ == "__main__":
    main()

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/sandboxes/knowledge.py`:

```py
"""knowledge_sandbox.py  (voice mode, Deepgram SDK v4, sync)
================================================================
Interactive sandbox for **KnowledgeManager** with optional voice input.

Features
--------
* Fixed richly‑seeded scenario covering multiple tables and attributes
  (people, products, purchases, geometry, pets, misc facts).
* `--scenario llm` flag – generate 120‑180 factual sentences via LLM and
  ingest them automatically.
* Shared audio/STT/TTS helpers imported from `utils.py`.
* Minimal dispatcher routes utterances to `KnowledgeManager.store` or
  `.retrieve` using a lightweight LLM intent/cleanup step.
* Supports interruptions during LLM processing via AsyncToolLoopHandle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import select
import time
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unify

from unity.constants import LOGGER as _LG  # type: ignore
from unity.knowledge_manager.knowledge_manager import KnowledgeManager  # type: ignore
from unity.common.llm_helpers import AsyncToolLoopHandle  # type: ignore
from sandboxes.utils import (
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
)  # type: ignore


# ---------------------------------------------------------------------------
# Scenario seeding
# ---------------------------------------------------------------------------


def _seed_fixed(km: KnowledgeManager) -> None:
    """Populate KnowledgeManager with a deterministic, multi‑table world."""
    seed_statements: List[str] = [
        # People & attributes
        "Adrian was born in 1994.",
        "Bob is 35 years old.",
        "Bob's favourite colour is green and his height is 180 centimetres.",
        "Carol owns a dog named Fido.",
        "Carol also owns a cat named Luna.",
        "Daniel's employee ID is E‑421 and he works in the London office.",
        # Products & purchases
        "The Apple iPhone 15 costs 999 US dollars.",
        "Daniel bought an iPhone 15 on 3 May 2025 using his credit card.",
        "A Logitech MX Master 4 mouse costs 129 US dollars.",
        "Sara ordered two MX Master 4 mice on 1 May 2025.",
        # Geometry
        "Point P has coordinates x = 3 and y = 4.",
        "Point Q has coordinates x = 1 and y = 10.",
        "Point R has coordinates x = ‑5 and y = 7.",
        # Random knowledge
        "The capital of Spain is Madrid.",
        "Mount Everest is 8,848 metres tall.",
        "The chemical symbol for gold is Au.",
    ]

    for stmt in seed_statements:
        km.store(stmt)


def _seed_llm(km: KnowledgeManager) -> Optional[str]:
    """Use an LLM to generate a large set of factual sentences."""
    prompt = (
        "Generate 120‑180 short factual sentences suitable for ingestion by a knowledge base. "
        "Cover diverse domains: personal bios, product pricing, purchases, geography, science facts, pets, coordinates, sports scores etc. "
        "Avoid any personally identifying sensitive data. "
        'Return as JSON {"statements": [...], "theme": <string>} and nothing else.'
    )
    client = unify.Unify("o4-mini@openai", cache=True)
    client.set_system_message(prompt)
    raw = client.generate("Produce knowledge scenario").strip()

    try:
        payload = json.loads(raw)
    except Exception:
        _LG.warning("LLM scenario failed – falling back to fixed seed.")
        _seed_fixed(km)
        return None

    for stmt in payload.get("statements", []):
        km.store(stmt)

    return payload.get("theme")


# ---------------------------------------------------------------------------
# Dispatcher – decide between store vs retrieve
# ---------------------------------------------------------------------------


class _IntentResp(BaseModel):
    action: str = Field(..., description="either 'store' or 'retrieve'")
    cleaned_text: str


_INTENT_PROMPT = (
    "You interface with a knowledge base. Incoming text may be: (a) a fact to be stored, or (b) a question to be answered via retrieval. "
    "If it ends with a question mark, it's *probably* a retrieval, but clarifying words like 'remember', 'note that', 'please store', etc. force storage. "
    "Respond with JSON {action:'store'|'retrieve', cleaned_text:'<sanitised>'}. For storage, cleaned_text should be the fact statement; for retrieval, it should be the user question."
)


async def _dispatch(
    km: KnowledgeManager,
    raw: str,
    *,
    show_steps: bool,
) -> Tuple[str, AsyncToolLoopHandle, List | None]:
    raw = raw.strip()

    # Quick rule: voice input often lacks punctuation – fall back to heuristic + LLM judge if ambiguous
    heuristic_store = bool(
        re.match(r"^(remember|note|store|add)\b", raw, re.I),
    ) and not raw.endswith("?")

    if heuristic_store:
        handle = km.store(raw, return_reasoning_steps=show_steps)
        return "store", handle, None

    llm = unify.Unify("gpt-4o@openai", response_format=_IntentResp)
    intent_json = llm.set_system_message(_INTENT_PROMPT).generate(raw)
    intent = _IntentResp.model_validate_json(intent_json)

    if intent.action == "store":
        handle = km.store(intent.cleaned_text, return_reasoning_steps=show_steps)
        return "store", handle, None

    # Retrieval path
    handle = km.retrieve(intent.cleaned_text, return_reasoning_steps=show_steps)
    return "retrieve", handle, None


# ---------------------------------------------------------------------------
# Input polling helpers
# ---------------------------------------------------------------------------


def _poll_for_input(timeout: float = 0.1) -> Optional[str]:
    """Non-blocking check for user input from stdin."""
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None

    line = sys.stdin.readline().strip()
    return line if line else None


async def _handle_interruptions(
    handle: AsyncToolLoopHandle,
    answer_task: asyncio.Task,
    *,
    voice_mode: bool = False,
) -> Tuple[str, str]:
    """
    Poll for user interruptions while waiting for the answer task to complete.
    Returns the kind of operation and the final result.
    """
    kind = "unknown"
    result = ""

    try:
        # Loop until the answer task is done
        while not answer_task.done():
            # Check for user input
            if voice_mode:
                # In voice mode, we just check if the user pressed Enter
                user_input = _poll_for_input(0.1)
                if user_input is not None:
                    print("⚠️ Interruption detected. Recording new input...")
                    audio_bytes = _record_until_enter()
                    user_text = _transcribe_deepgram(audio_bytes).strip()
                    if user_text:
                        print(f"▶️  New input: {user_text}")
                        if user_text.lower() in {"stop", "cancel"}:
                            print("🛑 Stopping current operation...")
                            handle.stop()
                        else:
                            print("⚡ Interjecting new information...")
                            await handle.interject(user_text)
            else:
                # In text mode, we check for any input
                user_input = _poll_for_input(0.1)
                if user_input is not None:
                    if user_input.lower() in {"stop", "cancel"}:
                        print("🛑 Stopping current operation...")
                        handle.stop()
                    else:
                        print(f"⚡ Interjecting: {user_input}")
                        await handle.interject(user_input)

            # Small sleep to prevent CPU spinning
            await asyncio.sleep(0.1)

        # Get the result from the completed task
        result = answer_task.result()

        # If we have a tuple (for return_reasoning_steps=True), extract just the answer
        if isinstance(result, tuple) and len(result) >= 1:
            result = result[0]

        return kind, result
    except asyncio.CancelledError:
        return kind, "Operation was cancelled."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _main_async(args) -> None:
    # Logging
    if not args.silent:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        _LG.setLevel(logging.INFO)
        if not args.debug:
            for noisy in ("unify", "unify.utils", "unify.logging", "requests", "httpx"):
                logging.getLogger(noisy).setLevel(logging.WARNING)

    # Unify project context
    unify.activate("KnowledgeSandbox")
    fresh = "Knowledge" not in unify.get_contexts() or args.new
    unify.set_context("Knowledge", overwrite=fresh)

    # Manager
    km = KnowledgeManager()

    if fresh:
        if args.scenario == "llm":
            theme = _seed_llm(km)
            if theme:
                _LG.info(f"[Seed] LLM scenario theme: {theme}")
        else:
            _seed_fixed(km)

    print("KnowledgeManager sandbox – speak or type. 'quit' to exit.")
    print("Press Enter during processing to interject or type 'stop' to cancel.\n")

    # Interaction loop
    if args.voice:
        while True:
            audio_bytes = _record_until_enter()
            user_text = _transcribe_deepgram(audio_bytes).strip()
            if not user_text:
                continue
            print(f"▶️  {user_text}")
            if user_text.lower() in {"quit", "exit"}:
                break

            _speak("Working on this now…")

            # Get the handle and create a task for the result
            kind, handle, steps = await _dispatch(
                km, user_text, show_steps=not args.silent
            )
            answer_task = asyncio.create_task(handle.result())

            # Handle interruptions while waiting for the result
            kind, result = await _handle_interruptions(
                handle, answer_task, voice_mode=True
            )

            print(f"[{kind}] => {result}\n")
            if kind == "retrieve":
                _speak(result)
    else:
        try:
            while True:
                line = input("> ").strip()
                if line.lower() in {"quit", "exit"}:
                    break
                if not line:
                    continue

                # Get the handle and create a task for the result
                kind, handle, steps = await _dispatch(
                    km, line, show_steps=not args.silent
                )
                answer_task = asyncio.create_task(handle.result())

                # Handle interruptions while waiting for the result
                kind, result = await _handle_interruptions(handle, answer_task)

                print(f"[{kind}] => {result}\n")
        except (EOFError, KeyboardInterrupt):
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KnowledgeManager sandbox with shared voice mode",
    )
    parser.add_argument(
        "--voice",
        "-v",
        action="store_true",
        help="enable voice capture/playback",
    )
    parser.add_argument("--scenario", choices=["fixed", "llm"], default="fixed")
    parser.add_argument("--new", "-n", action="store_true", help="wipe & reseed data")
    parser.add_argument(
        "--silent",
        "-s",
        action="store_true",
        help="suppress tool logs",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="verbose HTTP/LLM logs",
    )
    args = parser.parse_args()

    # Run the async main function
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/knowledge_manager/types.py`:

```py
from enum import StrEnum


class ColumnType(StrEnum):
    str = "str"
    int = "int"
    float = "float"
    bool = "bool"
    dict = "dict"
    list = "list"
    datetime = "datetime"
    date = "date"
    time = "time"

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/knowledge_manager/knowledge_manager.py`:

```py
import os
import unify
import requests
from typing import Any, Dict, List, Optional, Union

import requests
import unify

from ..common.embed_utils import EMBED_MODEL, ensure_vector_column
from ..helpers import _handle_exceptions
from .types import ColumnType
from ..common.llm_helpers import start_async_tool_use_loop, AsyncToolLoopHandle
from ..helpers import _handle_exceptions

API_KEY = os.environ["UNIFY_KEY"]


class KnowledgeManager:

    def __init__(self, *, traced: bool = True) -> None:
        """
        Responsible for *adding to*, *updating* and *searching through* all knowledge the assistant has stored in memory.
        """

        refactor_tools = {
            # Tables
            self._create_table.__name__: self._create_table,
            self._list_tables.__name__: self._list_tables,
            self._rename_table.__name__: self._rename_table,
            self._delete_table.__name__: self._delete_table,
            # Columns
            self._create_empty_column.__name__: self._create_empty_column,
            self._create_derived_column.__name__: self._create_derived_column,
            self._rename_column.__name__: self._rename_column,
            self._delete_column.__name__: self._delete_column,
        }

        self._store_tools = {
            **refactor_tools,
            self._add_data.__name__: self._add_data,
        }

        self._retrieve_tools = {
            **refactor_tools,
            self._search.__name__: self._search,
            self._nearest.__name__: self._nearest,
        }

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a KnowledgeManager."
        self._ctx = f"{read_ctx}/Knowledge" if read_ctx else "Knowledge"

        # Add tracing
        if traced:
            self = unify.traced(self)

    # Public #
    # -------#

    # English-Text Command

    async def store(
        self, text: str, *, return_reasoning_steps: bool = False
    ) -> "AsyncToolLoopHandle":
        """
        Take in any storage text command, and use the tools available (the *non-skipped* private methods of this class) to store the information, refactoring the table and column schema along the way if needed.

        Args:
            text (str): The information storage request, as a plain-text command.

            return_reasoning_steps (bool): Whether to return the reasoning steps for the storage request.

        Returns:
            AsyncToolLoopHandle: A handle to the running conversation that allows:
                - `await handle.result()` to get the final result
                - `await handle.interject(message)` to add a user message mid-conversation
                - `handle.stop()` to gracefully cancel the conversation

        Example:
            ```python
            handle = await knowledge_manager.store("Store John's email as john@example.com")
            # To get the final result:
            result = await handle.result()
            # To interject with additional information:
            await handle.interject("Also add his phone number: 555-1234")
            # To stop the conversation:
            handle.stop()
            ```
        """
        from unity.knowledge_manager.sys_msgs import STORE

        client = unify.AsyncUnify("o4-mini@openai", cache=True)
        client.set_system_message(STORE)
        handle = start_async_tool_use_loop(client, text, self._store_tools)

        # If we need to return reasoning steps, we need to wrap the handle
        if return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                ans = await original_result()
                return ans, client.messages

            handle.result = wrapped_result

        return handle

    async def retrieve(
        self, text: str, *, return_reasoning_steps: bool = False
    ) -> "AsyncToolLoopHandle":
        """
        Take in any retrieval text command, and use the tools available (the *non-skipped* private methods of this class) to retrieve the information, refactoring the table and column schema along the way if needed.

        Args:
            text (str): The information retrieval request, as a plain-text command.

            return_reasoning_steps (bool): Whether to return the reasoning steps for the retrieval request.

        Returns:
            AsyncToolLoopHandle: A handle to the running conversation that allows:
                - `await handle.result()` to get the final result
                - `await handle.interject(message)` to add a user message mid-conversation
                - `handle.stop()` to gracefully cancel the conversation

        Example:
            ```python
            handle = await knowledge_manager.retrieve("What is John's email?")
            # To get the final result:
            result = await handle.result()
            # To interject with additional information:
            await handle.interject("Only look in the Contacts table")
            # To stop the conversation:
            handle.stop()
            ```
        """
        from unity.knowledge_manager.sys_msgs import RETRIEVE

        client = unify.AsyncUnify("o4-mini@openai", cache=True)
        client.set_system_message(RETRIEVE)
        handle = start_async_tool_use_loop(client, text, self._retrieve_tools)

        # If we need to return reasoning steps, we need to wrap the handle
        if return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                ans = await original_result()
                return ans, client.messages

            handle.result = wrapped_result

        return handle

    # Helpers #
    # --------#

    def _get_columns(self, *, table: str) -> Dict[str, str]:
        proj = unify.active_project()
        ctx = f"{self._ctx}/{table}"
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields?project={proj}&context={ctx}"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.request("GET", url, headers=headers)
        _handle_exceptions(response)
        ret = response.json()
        return {k: v["data_type"] for k, v in ret.items()}

    # Private #
    # --------#

    # Tables

    def _create_table(
        self,
        name: str,
        *,
        description: Optional[str] = None,
        columns: Optional[Dict[str, ColumnType]] = None,
    ) -> Dict[str, str]:
        """
        Create a new table for storing long-term knowledge.

        Args:
            name (str): The name of the table to create. Eg: "MyTable".

            description (Optional[str]): A description of the table and the main purpose.

            columns (Optional[Dict[str, ColumnType]]): A dictionary of column names and their types. ColumnType can take values: `str`, `int`, `float`, `bool`, `list`, `dict`, `datetime`, `date`, `time`.

        Returns:
            Dict[str, str]: Message explaining whether the table was created or not.
        """
        proj = unify.active_project()
        ctx = f"{self._ctx}/{name}"
        unify.create_context(ctx, description=description)
        if not columns:
            return
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_input = {"project": proj, "context": ctx, "fields": columns}
        response = requests.request("POST", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

    def _list_tables(
        self,
        *,
        include_columns: bool = False,
    ) -> Union[List[str], List[Dict[str, ColumnType]]]:
        """
        List the tables which are being used to store all knowledge.

        Args:
            include_columns (bool): Whether to include the columns and their types for each table in the returned list.

        Returns:
            List[Dict[str, Dict[str, Union[str, ColumnType]]]]: Table names and their descriptions, and optionally also column names and types.
        """
        tables = {
            k[len(f"{self._ctx}/") :]: {"description": v}
            for k, v in unify.get_contexts(prefix=f"{self._ctx}/").items()
        }
        if not include_columns:
            return tables
        return {
            k: {**v, "columns": self._get_columns(table=k)} for k, v in tables.items()
        }

    def _rename_table(self, *, old_name: str, new_name: str) -> Dict[str, str]:
        """
        Rename the table.

        Args:
            old_name (str): The old name of the table.

            new_name (str): The new name for the table.

        Returns:
            Dict[str, str]: Message explaining whether the table was renamed or not.
        """
        proj = unify.active_project()
        old_name = f"{self._ctx}/{old_name}"
        new_name = f"{self._ctx}/{new_name}"
        url = f"{unify.BASE_URL}/project/{proj}/contexts/{old_name}/rename"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_input = {"name": new_name}
        response = requests.request("PATCH", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

    def _delete_table(self, table: str) -> Dict[str, str]:
        """
        Delete the specified table, and all of its data from the knowledge store.

        Args:
            table (str): The name of the table to delete.

        Returns:
            Dict[str, str]: Message explaining whether the table was deleted or not.
        """
        return unify.delete_context(f"{self._ctx}/{table}")

    # Columns

    def _create_empty_column(
        self,
        *,
        table: str,
        column_name: str,
        column_type: str,
    ) -> Dict[str, str]:
        """
        Adds an empty column to the table, which is initialized with `None` values.

        Args:
            table (str): The name of the table to add the column to.

            column_name (str): The name of the column to add.

            column_type (str): The type of the column to add.

        Returns:
            Dict[str, str]: Message explaining whether the column was created or not.
        """
        proj = unify.active_project()
        ctx = f"{self._ctx}/{table}"
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_input = {
            "project": proj,
            "context": ctx,
            "fields": {column_name: column_type},
        }
        response = requests.request("POST", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

    def _create_derived_column(
        self,
        *,
        table: str,
        column_name: str,
        equation: str,
    ) -> Dict[str, str]:
        """
        Create a new column in the table, derived from the other columns in the table.

        Args:
            table (str): The name of the table to add the column to.

            column_name (str): The name of the column to add.

            equation (str): The equation to use to derive the column. This is arbitrary Python code, with column names expressed as standard variables. For example, if a table includes two float columns `x` and `y`, then an equation of "(x**2 + y**2)**0.5" would be a valid, computing the length. Indexing is also supported `x[0]` for valid types `dict`, `list`, `str` etc., as is `len(x)`, casting to str via `str(x)` etc. The expression just needs to be valid Python with the column names as variables.

        Returns:
            Dict[str, str]: Message explaining whether the column was created or not.
        """
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/derived"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        equation = equation.replace("{", "{lg:")
        json_input = {
            "project": unify.active_project(),
            "context": f"{self._ctx}/{table}",
            "key": column_name,
            "equation": equation,
            "referenced_logs": {"lg": {"context": f"{self._ctx}/{table}"}},
        }
        response = requests.request("POST", url, json=json_input, headers=headers)
        return response.json()

    def _delete_column(self, *, table: str, column_name: str) -> Dict[str, str]:
        """
        Delete column from the table, and all of the data.

        Args:
            table (str): The name of the table to delete the column from.

            column_name (str): The name of the column to delete.

        Returns:
            Dict[str, str]: Message explaining whether the column was deleted or not.
        """
        url = f"{os.environ['UNIFY_BASE_URL']}/logs?delete_empty_logs=True"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_input = {
            "project": unify.active_project(),
            "context": f"{self._ctx}/{table}",
            "ids_and_fields": [[None, column_name]],
            "source_type": "all",
        }
        response = requests.request("DELETE", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

    def _rename_column(
        self,
        *,
        table: str,
        old_name: str,
        new_name: str,
    ) -> Dict[str, str]:
        """
        Rename the specified column.

        Args:
            table (str): The name of the table to rename the column in.

            old_name (str): The name of the column to rename.

            new_name (str): The new name for the column.

        Returns:
            Dict[str, str]: Message explaining whether the column was renamed or not.
        """
        proj = unify.active_project()
        ctx = f"{self._ctx}/{table}"
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/rename_field"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_input = {
            "project": proj,
            "context": ctx,
            "old_field_name": old_name,
            "new_field_name": new_name,
        }
        response = requests.request("PATCH", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

    # Add Data

    def _add_data(self, *, table: str, data: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Add data to the specified table. Will automatically create new columns if any keys are not present in the table already.

        Args:
            table (str): The name of the table to add the data to.

            data (List[Dict[str, Any]]): The data to add to the table.

        Returns:
            Dict[str, str]: Message explaining whether the data was added or not.
        """
        return unify.create_logs(
            context=f"{self._ctx}/{table}",
            entries=data,
            batched=True,  # NOTE: async logger can mess with the order of the data
        )

    # Vector Search Helpers
    def _ensure_table_vector(self, table: str, column: str, source: str) -> None:
        """
        Ensure that a vector column exists in the given table. If it doesn't exist,
        create it as a derived column from the source column.

        Args:
            table (str): The name of the table to ensure the vector column in.
            column (str): The name of the vector column to ensure.
            source (str): The name of the column to derive the vector column from.
        """
        context = f"{self._ctx}/{table}"
        ensure_vector_column(context, embed_column=column, source_column=source)

    def _nearest(
        self,
        *,
        tables: List[str],
        column: str,
        source: str,
        text: str,
        k: int = 5,
    ) -> List[unify.Log]:
        """
        Find the k nearest entries in the table to the given text using vector embeddings.

        Args:
            table (str): The name of the table to search in.
            column (str): The name of the vector column to use for similarity search.
            source (str): The name of the column to derive the vector column from.
            text (str): The query text to find similar entries to.
            k (int): The number of results to return.

        Returns:
            List[unify.Log]: The k nearest log entries to the query text.
        """
        # ToDo: convert to map function
        results = dict()
        for table in tables:
            context = f"{self._ctx}/{table}"
            self._ensure_table_vector(table, column, source)
            results[table] = [
                log.entries
                for log in unify.get_logs(
                    context=context,
                    sorting={
                        f"cosine({column}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
                    },
                    limit=k,
                )
            ]
        return results

    # Search

    def _search(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        tables: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Apply the filter to all of the specified tables, and return the results following the filter.

        Args:
            filter (Optional[str]): Arbitrary Python logical expressions which evaluate to `bool`, with column names expressed as standard variables. For example, if a table includes two integer columns `x` and `y`, then a filter expression of "x > 3 and y < 2" would be a valid. Indexing is also supported `x[0]` for valid types `dict`, `list`, `str` etc., as is `len(x)`, casting to str via `str(x)` etc. The expression just needs to be valid Python with the column names as variables.

            offset (int): The offset to start the search from, in the paginated result.

            limit (int): The number of rows to return, in the paginated result.

            tables (Optional[List[str]]): The list of tables to search in. If not provided, all tables will be searched.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary where keys are table names and values are lists, where each item in the list is a dict representing a row in the table.
        """
        if tables is None:
            tables = self._list_tables()
        # ToDo: convert to map function
        results = dict()
        for table in tables:
            results[table] = [
                log.entries
                for log in unify.get_logs(
                    context=f"{self._ctx}/{table}",
                    filter=filter,
                    offset=offset,
                    limit=limit,
                )
            ]
        return results

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/knowledge_manager/sys_msgs.py`:

```py
import os

AGENT_FIRST = os.environ["AGENT_FIRST"]
AGENT_LAST = os.environ["AGENT_LAST"]
AGENT_AGE = os.environ["AGENT_AGE"]
FIRST_NAME = os.environ["FIRST_NAME"]

ENGAGE_WITH_KNOWLEDGE = f"""
You are an assistant to {FIRST_NAME}, and you are engaged in a back-and-forth conversation with {FIRST_NAME}.
Your task is to follow this conversation closely, and after each message from {FIRST_NAME}, you must determine which of three possible actions are most appropriate:

- Store knowledge
- Retrieve knowledge
- Do not use knowledge

The knowledge is a record of all important information stored throughout your lifetime, useful for retreiving information when needed during any interaction with {FIRST_NAME}.

If {FIRST_NAME} has recently said something which seems to be important and may be relevant for future tasks, then you should *store* this knowledge.

If they have asked something and you're not sure about the answer, then you should *retrieve* this knowledge.

If they have neither asked something you're not sure about, nor have they said something important, then neither storage nor retrieval is necessary.

The API for handling knowledge (both storage and retrieval) operates based on simple *english* commands.

If you deem that either knowledge storage or retrieval is needed, then you should also provide a very detailed command for exactly what needs to be stored, or what needs to be retrieved.

You do not have access to the schema used for knowledge storage, so you won't be able to explain exactly how to access or store the knowledge on a technical level.

You just need to explain your knowledge storage/retrieval needs in very clear english, and the knowledge manager will handle your english language request.
"""

STORE = """
Your task is to store the information requested by the user, and you should continue using the tools available until you are satisfied that you have stored the information in the most elegant manner possible, making any strucutral changes to the existing tables as needed in order to accomdate the new information.

Information is stored in tables, and each table has columns of a certain data type. The data types are static so you cannot change them once the column is created. The data inserted into each column must match the data type of that column.

The tools enable you to create, rename, modify, search and delete tables and columns as you see fit.

You are strongly encouraged to refactor the table and column designs.

If this storage request could be handled with an improved schema, then please implement this new layout via consecutive tool use (adding, deleting, renaming tables/columns etc.), before returning to control to the user.
"""

RETRIEVE = """
Your task is to retrieve the information requested by the user, and you should continue using the tools available until you are satisfied that you have retrieved the information requested, making any strucutral changes to the existing tables as needed in order to accomdate this request.

Information is stored in tables, and each table has columns of a certain data type.

The tools enable you to create, rename, modify, search and delete tables and columns as you see fit.

You are strongly encouraged to refactor the table and column designs.

If this retrieval request could be handled with an improved schema, then please implement this new layout via consecutive tool use (adding, deleting, renaming tables/columns etc.), before returning with your answer.
"""

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/task_manager/sys_msgs.py`:

```py
import json
from ..task_list_manager.types.task import Task
from .task_manager import TaskManager

REQUEST = f"""
Your task is to handle any plain-text english task-related request, which can either:

a) ask about or update the task list (includes details all tasks, including scheduled, cancelled, failed and also the active task).

{list(TaskManager._task_list_tools.keys())}

b) ask about live progress, steer or stop the single active task, currently being performed.

{list(TaskManager._active_task_tools.keys())}

In the case of (a), the schema of the underlying task list table is:
{json.dumps(Task.model_json_schema(), indent=4)}


The request can also involve multi-step reasoning. For example:

"if the task to search for sales leads is marked as high or lower, then mark it as urgent and start it right now (pausing any other task that might be active right now))"

This user request would likely require us to first ask about the task list ({TaskManager._ask_about_task_list.__name__}), update the task list ({TaskManager._update_task_list}), stop the current active task if one exists ({TaskManager._stop_active_task}), and then start executing the specified task ({TaskManager._start_task}).

You should continue using the tools available until you're totally happy that the request has been fully performed, and you should respond with the key details related to this request.
"""

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/task_manager/task_manager.py`:

```py
import time
import random
import threading
from typing import Dict

import unify

from ..common.llm_helpers import start_async_tool_use_loop
from ..task_list_manager.task_list_manager import TaskListManager
from .sys_msgs import REQUEST


class _DummyPlanner:
    """
    A dummy planner class that simulates task execution and question answering.
    """

    def __init__(self) -> None:
        """
        Initialize the dummy planner with no active task and a GPT-4 client.
        """
        self._active_task = None
        self._ask_simulator = unify.Unify("o4-mini@openai", stateful=True)
        self._steer_simulator = unify.Unify("o4-mini@openai", stateful=True)

    def start(self, task: str):
        """
        Simulate executing a task by setting it as active, waiting, then clearing it.

        Args:
            task (str): The task description to simulate executing.
        """
        self._active_task = task
        self._ask_simulator.set_system_message(
            f"You should pretend you are completing the following task:\n{task}\nCome up with imaginary answers to the user questions about the task",
        )
        self._steer_simulator.set_system_message(
            f"You should pretend you are completing the following task:\n{task}\nCome up with imaginary responses to the user requests to steer the task behaviour, stating that you either can or cannot steer the ongoing task as requested.",
        )
        time.sleep(random.uniform(5, 30))
        self._ask_simulator.set_messages([])
        self._ask_simulator.set_system_message("")
        self._steer_simulator.set_messages([])
        self._steer_simulator.set_system_message("")
        self._active_task = None

    def steer(self, instruction: str) -> str:
        """
        Steer the behaviour of the currently active task.

        Args:
            instruction (str): The instruction for the planner to follow wrt the current active task.

        Returns:
            str: The result of the attempt to steer behaviour, whether this was doable or not.
        """
        if not self._active_task:
            return "No tasks are currently being performed, so I have nothing to steer."
        return self._steer_simulator.generate(instruction)

    def ask(self, question: str) -> str:
        """
        Answer questions about the currently active task.

        Args:
            question (str): The question to answer about the active task.

        Returns:
            str: The answer to the question, or a message indicating no active task.
        """
        if not self._active_task:
            return "No tasks are currently being performed, so I cannot answer your question."
        return self._ask_simulator.generate(question)

    def stop(self, reason: str) -> str:
        """
        Stops the currently active task.

        Args:
            reason (str): The reason for stopping the task.

        Returns:
            str: A message indicating whether the task was stopped or if there was no active task.
        """
        if not self._active_task:
            return (
                "No tasks are currently being performed, so there is nothing to stop."
            )

        task = self._active_task
        self._active_task = None
        self._ask_simulator.set_messages([])
        self._ask_simulator.set_system_message("")
        self._steer_simulator.set_messages([])
        self._steer_simulator.set_system_message("")

        return f"Stopped task '{task}' for reason: {reason}"


class TaskManager(threading.Thread):

    def __init__(self, *, daemon: bool = True) -> None:
        """
        Responsible for managing the set of tasks to complete by updating, scheduling, executing and steering tasks, both scheduled and active.

        Args:
            daemon (bool): Whether the thread should be a daemon thread.
        """
        super().__init__(daemon=daemon)
        self._tlm = TaskListManager()
        self._planner = _DummyPlanner()

        self._task_list_tools = {
            self._ask_about_task_list.__name__: self._ask_about_task_list,
            self._update_task_list.__name__: self._update_task_list,
        }
        self._active_task_tools = {
            self._start_task.__name__: self._start_task,
            self._ask_about_active_task.__name__: self._ask_about_active_task,
            self._steer_active_task.__name__: self._steer_active_task,
            self._stop_active_task.__name__: self._stop_active_task,
        }

        self._tools = {
            **self._task_list_tools,
            **self._active_task_tools,
        }

    # Public #
    # -------#

    # English-Text requests

    async def request(
        self,
        text: str,
        *,
        return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,
    ) -> Dict[str, str]:
        """
        Handle any plain-text english task-related request, which can refer to inactive (cancelled, scheduled, failed) or active tasks in the task list, and can also involve updates to any of those tasks and/or simply answering questions about them. It can also involve changes to the currently active task, and involve multi-step reasoning which includes questions and actions for both inactive and active tasks.

        For example: text="if this {some task description} is marked as high or lower, then mark it as urgent and start it right now (pausing any other task that might be active right now))"

        Args:
            text (str): The task request, which can involve questions, actions, or both interleaved.
            return_reasoning_steps (bool): Whether to return the reasoning steps for the ask request.
            log_tool_steps (bool): Whether to log the steps taken by the tool.

        Returns:
            Dict[str, str]: Answers to the question(s), and updates on any action(s) performed.
        """
        client = unify.AsyncUnify("o4-mini@openai", cache=True)
        client.set_system_message(REQUEST)
        ans = await start_async_tool_use_loop(
            client,
            text,
            self._tools,
            log_steps=log_tool_steps,
        ).result()
        if return_reasoning_steps:
            return ans, client.messages
        return ans

    # Tools #
    # ------#

    # Task List

    def _ask_about_task_list(self, question: str) -> str:
        f"""
        Ask any question about the list of tasks (including scheduled, cancelled, failed, and the active task) based on a natural language question.

        This function *cannot* answer questions about the *live state* of the active task.
        It can answer questions about the schedule, priority, title, description, queue ordering etc.

        Args:
            question (str): The question to ask about the task list.

        Returns:
            str: The answer to the question about the task list.
        """
        return self._tlm.ask(question)

    def _update_task_list(self, update: str) -> str:
        f"""
        Update the list of tasks (including scheduled, cancelled, failed, and the active task) based on a natural language question.

        Args:
            update (str): The update instruction in natural language.

        Returns:
            str: Whether the update was applied successfully or not.
        """
        return self._tlm.update(update)

    # Active Task

    def _start_task(self, description: str) -> str:
        """
        Start a new task, making it the active task. If there is already an active task,
        it will be paused.

        Args:
            description (str): Description of the task to start.

        Returns:
            str: A message confirming the task was started, or explaining why it couldn't be started.
        """
        return self._planner.start(description)

    def _ask_about_active_task(self, question: str) -> str:
        """
        Ask a question about the currently active task, including its live state.

        Args:
            question (str): The question to ask about the active task.

        Returns:
            str: The answer about the active task's current state, or a message indicating no active task.
        """
        return self._planner.ask(question)

    def _steer_active_task(self, instruction: str) -> str:
        """
        Provide steering instructions to modify the behavior of the currently active task.

        Args:
            instruction (str): The steering instruction in natural language.

        Returns:
            str: A message confirming the steering instruction was applied, or explaining why it couldn't be applied.
        """
        return self._planner.steer(instruction)

    def _stop_active_task(self, reason: str) -> str:
        """
        Stop the currently active task.

        Args:
            reason (str): The reason for stopping the task, which will be recorded.

        Returns:
            str: A message confirming the task was stopped, or explaining why it couldn't be stopped.
        """
        self._planner.stop(reason)

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/assertion_helpers.py`:

```py
"""
Shared assertion helpers for tests
==================================

Contains common functions for formatting assertion error messages
with detailed context including reasoning steps from LLM tool usage.
"""

import json
from typing import Any, List, Dict, Optional


def format_reasoning_steps(reasoning: List[Dict[str, Any]]) -> str:
    """Format reasoning steps from LLM tool use loops for better readability."""
    if not reasoning:
        return "No reasoning steps available"

    # Pretty print the reasoning steps, handling nested content fields
    def format_json_content(msg):
        if "content" in msg and msg["content"]:
            try:
                msg["content"] = json.loads(msg["content"])
            except (json.JSONDecodeError, TypeError):
                pass
        return msg

    formatted_reasoning = [format_json_content(msg) for msg in reasoning]
    formatted_reasoning = json.dumps(formatted_reasoning, indent=4)
    formatted_reasoning = formatted_reasoning.replace("\\n", "\n")

    return formatted_reasoning


def assertion_failed(
    expected: Any,
    actual: Any,
    reasoning: List[Dict[str, Any]],
    description: str = "",
    context_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a detailed error message for assertion failures with LLM reasoning.

    Args:
        expected: The expected value
        actual: The actual value received
        reasoning: List of reasoning steps from LLM tool use
        description: Optional description of the assertion
        context_data: Optional additional context data to include (e.g., tasks, messages)

    Returns:
        Formatted error message string
    """
    context_str = ""
    if context_data:
        for label, data in context_data.items():
            context_str += f"\n{label}:\n{json.dumps(data, indent=4)}\n"

    formatted_reasoning = format_reasoning_steps(reasoning)

    return (
        f"\n{description}\n"
        f"Expected:\n{expected}\n"
        f"Got:\n{actual}\n"
        f"{context_str}"
        f"Reasoning:\n{formatted_reasoning}\n"
    )

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_log_term_memory_flow.py`:

```py
# Extended integration tests for TranscriptManager, KnowledgeManager, TaskListManager.
# These tests exercise realistic, end‑to‑end user flows – both the Ivy demo cases
# and additional hypothetical scenarios covering conflicting preferences, dynamic
# schema evolution, recurring tasks, and compound queries that span multiple
# memory managers.

from __future__ import annotations

import pytest
import random
from datetime import datetime, timezone, timedelta

from tests.helpers import _handle_project
from unity.communication.types.message import Message, Medium
from unity.communication.transcript_manager.transcript_manager import TranscriptManager
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.task_list_manager.task_list_manager import TaskListManager


# --------------------------------------------------------------------------- #
#  Small helpers                                                              #
# --------------------------------------------------------------------------- #

_NOW = datetime.now(timezone.utc)


def _stamp(delta_mins: int) -> str:
    """ISO timestamp *delta_mins* minutes from *now* (negative = past)."""
    return (_NOW + timedelta(minutes=delta_mins)).isoformat()


def _msg(
    delta_mins: int,
    content: str,
    *,
    medium: Medium = Medium.WHATSAPP_MSG,
    sender: int = 1,
    receiver: int = 0,
    ex_id: int = 0,
) -> Message:
    """Convenience factory that always matches the Message schema."""
    return Message(
        medium=medium,
        sender_id=sender,
        receiver_id=receiver,
        timestamp=_stamp(delta_mins),
        content=content,
        exchange_id=ex_id,
    )


# --------------------------------------------------------------------------- #
#  Transcript-centric scenarios                                               #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.eval
def test_transcript_preference_recall() -> None:
    tm = TranscriptManager()
    tm.start()

    # Exchange 0: the user sets a preference for WhatsApp
    tm.log_messages(
        [
            _msg(-2880, "Hi, I prefer WhatsApp over email.", ex_id=0),
            _msg(
                -2879,
                "Acknowledged – will use WhatsApp.",
                sender=0,
                receiver=1,
                medium=Medium.WHATSAPP_MSG,
                ex_id=0,
            ),
        ],
    )
    tm.summarize(exchange_ids=0)  # generate & persist summary

    answer = tm.ask("What channel should I use to contact the user?")
    assert "whatsapp" in answer.lower()


@_handle_project
@pytest.mark.eval
def test_transcript_preference_override() -> None:
    tm = TranscriptManager()
    tm.start()

    tm.log_messages([_msg(-1440, "Please message me on WhatsApp.", ex_id=1)])
    tm.summarize(exchange_ids=1)

    # 24 h later the user changes their mind
    tm.log_messages(
        [
            _msg(
                -5,
                "Actually, email is better for me now.",
                medium=Medium.EMAIL,
                ex_id=2,
            )
        ]
    )

    answer = tm.ask("How should I contact the user today?")
    assert "email" in answer.lower()


@_handle_project
@pytest.mark.eval
def test_cross_exchange_semantic_search() -> None:
    tm = TranscriptManager()
    tm.start()

    # Three different exchanges, media and contacts
    tm.log_messages(
        [
            _msg(
                -60,
                "Billie, the SC-123 shipment has left the warehouse.",
                medium=Medium.EMAIL,
                sender=2,
                receiver=0,
                ex_id=10,
            ),
            _msg(
                -50,
                "Great, thanks!",
                medium=Medium.EMAIL,
                sender=0,
                receiver=2,
                ex_id=10,
            ),
            _msg(
                -30,
                "Heads-up: tracking number is ZX-98765.",
                medium=Medium.SMS_MESSAGE,
                sender=2,
                receiver=0,
                ex_id=11,
            ),
            _msg(
                -5,
                "Remember, ZX-98765 arrives tomorrow.",
                medium=Medium.WHATSAPP_MSG,
                sender=2,
                receiver=0,
                ex_id=12,
            ),
        ],
    )

    answer = tm.ask("What tracking number did the supplier give me?")
    assert "zx-98765" in answer.lower()


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeManager – simple natural-language round-trip
# ─────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.eval
def test_knowledge_roundtrip():
    """
    Store a fact in plain English and make sure we can
    query it back via natural-language retrieval.
    """
    km = KnowledgeManager()
    km.start()
    km.store(
        "My flight BA117 from Karachi (KHI) to New-York (JFK) departs on 20-May-2025.",
    )

    answer = km.retrieve(
        "What's the reference number for my KHI → JFK flight on 20 May 2025?",
    )
    assert "ba117" in answer.lower()

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_communication/test_transcript_manager/test_ask.py`:

```py
"""
tests/test_ask.py
=================

Integration-style tests for ``TranscriptManager.ask`` that rely on a live
LLM to (a) choose tools and (b) judge whether the final answer is
correct.

Running the suite therefore requires:

* network access
* a valid OpenAI-compatible key (used by `unify.Unify`)
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import datetime, timezone, timedelta
from unity.events.event_bus import EventBus, Event
from typing import List

import pytest

import unify
from unity.communication.transcript_manager.transcript_manager import TranscriptManager
from unity.communication.types.message import Message
from unity.common.llm_helpers import _dumps
from unity.common import AsyncToolLoopHandle
from tests.assertion_helpers import assertion_failed
from tests.helpers import _handle_project

# --------------------------------------------------------------------------- #
#  CONTACTS (same as before)                                                  #
# --------------------------------------------------------------------------- #

_CONTACTS: List[dict] = [
    dict(  # id = 0
        first_name="Carlos",
        surname="Diaz",
        email_address="carlos.diaz@example.com",
        phone_number="+14155550000",
        whatsapp_number="+14155550000",
    ),
    dict(  # id = 1
        first_name="Dan",
        surname="Turner",
        email_address="dan.turner@example.com",
        phone_number="+447700900001",
        whatsapp_number="+447700900001",
    ),
    dict(  # id = 2
        first_name="Julia",
        surname="Nguyen",
        email_address="julia.nguyen@example.com",
        phone_number="+447700900002",
        whatsapp_number="+447700900002",
    ),
    dict(  # id = 3
        first_name="Jimmy",
        surname="O'Brien",
        email_address="jimmy.obrien@example.com",
        phone_number="+61240011000",
        whatsapp_number="+61240011000",
    ),
    dict(  # id = 4
        first_name="Anne",
        surname="Fischer",
        email_address="anne.fischer@example.com",
        phone_number="+49891234567",
        whatsapp_number="+49891234567",
    ),
]

_ID_BY_NAME: dict[str, int] = {}  # filled during seeding


# --------------------------------------------------------------------------- #
#  SCENARIO BUILDER                                                           #
# --------------------------------------------------------------------------- #


class ScenarioBuilder:
    """Populate Unify with contacts, 6 'meaningful' exchanges + filler."""

    async def __init__(self) -> None:
        self._event_bus = EventBus()
        self.tm = TranscriptManager()
        await self._seed_contacts()
        await self._seed_key_exchanges()
        await self._seed_filler()
        # One stored summary just so summaries exist
        self.tm.summarize(exchange_ids=[0, 1])
        self._event_bus.join_published()

    # --------------------------------------------------------------------- #
    async def _seed_contacts(self) -> None:
        for idx, c in enumerate(_CONTACTS):
            self.tm.create_contact(**c)
            _ID_BY_NAME[c["first_name"].lower()] = idx

    # --------------------------------------------------------------------- #
    async def _seed_key_exchanges(self) -> None:
        now = datetime(2025, 4, 20, 15, 0, tzinfo=timezone.utc)

        # E0: first Dan–Julia phone call
        await self._log(
            0,
            "phone_call",
            [
                (1, 2, now, "Hi Julia, it's Dan. Quick check-in about Q2 metrics."),
                (2, 1, now + timedelta(seconds=30), "Sure Dan, ready when you are."),
            ],
        )

        # E1: *last* Dan–Julia phone call (later date)
        later = datetime(2025, 4, 26, 9, 30, tzinfo=timezone.utc)
        await self._log(
            1,
            "phone_call",
            [
                (
                    1,
                    2,
                    later,
                    "Morning Julia – finalising the London event agenda today.",
                ),
                (
                    2,
                    1,
                    later + timedelta(seconds=45),
                    "Great. Let's confirm the speaker list and coffee budget.",
                ),
            ],
        )

        # E2: Carlos interest e-mail
        t_email = datetime(2025, 4, 21, 12, 0, tzinfo=timezone.utc)
        await self._log(
            2,
            "email",
            [
                (
                    0,
                    1,
                    t_email,
                    "Subject: Stapler bulk order\n\n"
                    "Hi Dan,\nI'm **interested in buying 200 units** of "
                    "your new stapler. Can you quote?\n\nThanks,\nCarlos",
                ),
                (
                    1,
                    0,
                    t_email + timedelta(hours=2),
                    "Hi Carlos — sure, $4.50 per unit. See attached PDF.",
                ),
            ],
        )

        # E3: Jimmy holiday WhatsApp
        t_holiday = datetime(2025, 4, 22, 18, 10, tzinfo=timezone.utc)
        await self._log(
            3,
            "whatsapp_message",
            [
                (
                    3,
                    1,
                    t_holiday,
                    "Heads-up Dan, I'll be **on holiday from 2025-05-15** "
                    "to 2025-05-30. Ping me before that if urgent.",
                ),
            ],
        )

        # E4: Anne passport excuse (WhatsApp)
        t_excuse = datetime(2025, 4, 23, 9, 0, tzinfo=timezone.utc)
        await self._log(
            4,
            "whatsapp_message",
            [
                (
                    4,
                    1,
                    t_excuse,
                    "Sorry Dan, I *can't join the Berlin trip because my "
                    "passport expired* last week.",
                ),
            ],
        )

    # --------------------------------------------------------------------- #
    async def _seed_filler(self, exchanges: int = 20, msgs_per: int = 15) -> None:
        """Adds irrelevant chatter so filtering matters."""
        random.seed(12345)
        media = ["email", "phone_call", "sms_message", "whatsapp_message"]
        start = datetime(2025, 4, 24, tzinfo=timezone.utc)

        for ex_off in range(exchanges):
            ex_id = 10 + ex_off
            mtype = random.choice(media)
            a, b = random.sample(list(_ID_BY_NAME.values()), 2)
            batch: List[tuple[int, int, datetime, str]] = []
            for i in range(msgs_per):
                batch.append(
                    (
                        a if i % 2 else b,
                        b if i % 2 else a,
                        start + timedelta(minutes=ex_off * 3 + i),
                        f"Filler {ex_id}-{i} {mtype} random text.",
                    ),
                )
            await self._log(ex_id, mtype, batch)

    # --------------------------------------------------------------------- #
    async def _log(
        self,
        ex_id: int,
        medium: str,
        msgs: List[tuple[int, int, datetime, str]],
    ) -> None:
        [
            await self._event_bus.publish(
                Event(
                    context="Messages",
                    timestamp=None,
                    paylod=Message(
                        medium=medium,
                        sender_id=s,
                        receiver_id=r,
                        timestamp=ts.isoformat(),
                        content=txt,
                        exchange_id=ex_id,
                    ),
                )
            )
            for s, r, ts, txt in msgs
        ]


# --------------------------------------------------------------------------- #
#  DETERMINISTIC GROUND-TRUTH GENERATOR                                       #
# --------------------------------------------------------------------------- #


def _answer_semantic(tm: TranscriptManager, question: str) -> str:
    """Compute the *correct* answer directly from stored data."""
    q = question.lower()
    messages = tm._search_messages(limit=None)

    def cid(name: str) -> int:
        return _ID_BY_NAME[name]

    if _is_summary_q(question):
        # return the *two utterances* that form the last Dan–Julia phone call.
        last_call = sorted(
            (
                m
                for m in messages
                if m.medium == "phone_call"
                and {m.sender_id, m.receiver_id} == {cid("dan"), cid("julia")}
            ),
            key=lambda m: m.timestamp,
        )[-2:]
        return "\n".join(m.content for m in last_call)

    if "quantity" in q and "carlos" in q:
        return "200"

    if "carlos" in q and "buy" in q:
        msg: Message = next(
            m
            for m in messages
            if m.sender_id == cid("carlos") and "buy" in m.content.lower()
        )
        quote = msg.content.splitlines()[0]
        return f"Yes – {quote}"

    if "when did dan last speak with julia" in q:
        last: str = max(
            m.timestamp
            for m in messages
            if m.medium == "phone_call"
            and {m.sender_id, m.receiver_id} == {cid("dan"), cid("julia")}
        )
        return last.split("T")[0]

    if "jimmy" in q and "holiday" in q:
        pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        msg = next(
            m
            for m in messages
            if m.sender_id == cid("jimmy") and "holiday" in m.content.lower()
        )
        return pattern.search(msg.content).group(0)

    if "anne" in q and "why" in q:
        msg = next(m for m in messages if m.sender_id == cid("anne"))
        return "passport expired"

    if "medium does julia use most" in q:
        counts = Counter(m.medium for m in messages if m.sender_id == cid("julia"))
        return counts.most_common(1)[0][0]

    if "how many different media has dan used" in q:
        media = {m.medium for m in messages if m.sender_id == cid("dan")}
        return str(len(media))

    if "one-sentence summary" in q or "one sentence summary" in q:
        last_call = [
            m
            for m in messages
            if m.medium == "phone_call"
            and {m.sender_id, m.receiver_id} == {cid("dan"), cid("julia")}
        ]
        last_ts = max(m.timestamp for m in last_call)
        combined = " ".join(m.content for m in last_call if m.timestamp == last_ts)
        return " ".join(combined.split()[:12]) + "..."

    return "N/A"


# --------------------------------------------------------------------------- #
#  LLM-AS-A-JUDGE SUMMARY COMPARISONS                                         #
# --------------------------------------------------------------------------- #


def _is_summary_q(q: str) -> bool:
    return "one-sentence summary" in q.lower() or "one sentence summary" in q.lower()


# --------------------------------------------------------------------------- #
#  QUESTIONS                                                                  #
# --------------------------------------------------------------------------- #

QUESTIONS = [
    "Did Carlos seem interested in buying the product? Can you find a relevant quote to back up your answer?",
    "When did Dan last speak with Julia on the phone?",
    "Did Jimmy ever tell us when he's on holiday? If so, what date?",
    "Why didn't Anne want to come with us on the trip? I forgot her excuse.",
    "What quantity did Carlos say he wanted to buy?",
    "Which medium does Julia use most often to communicate?",
    "How many different media has Dan used so far?",
    "Give me a one-sentence summary of the last Dan-Julia phone call.",
]


# --------------------------------------------------------------------------- #
#  EVALUATION LLM                                                             #
# --------------------------------------------------------------------------- #


def _llm_assert_correct(
    question: str,
    expected: str,
    candidate: str,
    steps: list,
) -> None:
    """LLM-based validation with stricter or fuzzier rubric per question."""
    judge = unify.Unify("o4-mini@openai", cache=True)

    if _is_summary_q(question):
        system_msg = (
            "You are an expert summary evaluator. "
            "You will be given the *source dialogue* of a short phone call "
            "and a candidate **one-sentence** summary. "
            'Respond ONLY with JSON {"correct": true|false}. '
            "Mark correct⇢true if the summary captures the main intent and "
            "key factual details of the dialogue, even if wording differs. "
            "Ignore minor tense or stylistic variations."
        )
        payload = _dumps(
            {"dialogue": expected, "summary": candidate},
            indent=4,
        )
    else:
        system_msg = (
            "You are a strict unit-test judge. "
            "You will be given a question, a ground-truth answer derived "
            "directly from the data, and a candidate answer. "
            'Respond ONLY with JSON {"correct": true|false}. '
            "Mark correct⇢true if a reasonable human would accept the candidate "
            "as fully accurate; otherwise false."
        )
        payload = _dumps(
            {"question": question, "ground_truth": expected, "candidate": candidate},
            indent=4,
        )

    judge.set_system_message(system_msg)
    result = judge.generate(payload)

    match = re.search(r"\{.*\}", result, re.S)
    assert match, assertion_failed(
        "Expected JSON format from LLM judge",
        result,
        steps,
        "LLM judge returned unexpected format",
    )
    verdict = json.loads(match.group(0))
    assert verdict.get("correct") is True, assertion_failed(
        expected,
        candidate,
        steps,
        f"Question: {question}",
    )


# --------------------------------------------------------------------------- #
#  PARAMETRISED TEST                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question", QUESTIONS)
@_handle_project
async def test_ask_semantic_with_llm_judgement(
    question: str,
) -> None:
    """
    Calls the real `.ask()` (which itself may call the LLM multiple
    times), then asks a _separate_ LLM whether the answer is acceptable.
    """
    tm = await ScenarioBuilder().tm
    handle = tm.ask(question, return_reasoning_steps=True)
    candidate, steps = await handle.result()
    expected = _answer_semantic(tm, question)
    _llm_assert_correct(question, expected, candidate, steps)

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_communication/test_transcript_manager/test_mock_scenario_transcript.py`:

```py
# import os
# import json
# import pytest
# import datetime
# import time
# from typing import List, Dict, Any, Optional

# from unity.communication.transcript_manager.transcript_manager import TranscriptManager
# from unity.communication.types.message import Message
# from unity.communication.types.contact import Contact
# from unity.communication.types.summary import Summary

# # Test scenario based on MVP demo: A professional setting where an assistant
# # manages client communication across multiple channels (email, phone, chat)
# # and needs to recall conversation details accurately over time


# class TestComplexTranscriptScenario:
#     """Complex real-world test scenarios for TranscriptManager with live LLM integration."""

#     @classmethod
#     def setup_class(cls):
#         """Initialize test resources once at class level."""
#         cls.transcript_manager = TranscriptManager()
#         cls.cleanup_data = []

#         # Seed initial data
#         cls._seed_contact_data()
#         cls._seed_message_history()

#     @classmethod
#     def teardown_class(cls):
#         """Clean up test resources once at class level."""
#         # Add cleanup code if needed
#         pass

#     @classmethod
#     def _seed_contact_data(cls):
#         """Create seed contact data representing a business network."""
#         contacts = [
#             {
#                 "first_name": "Sarah",
#                 "surname": "Johnson",
#                 "email_address": "sarah.johnson@acmecorp.com",
#                 "phone_number": "+1-555-123-4567",
#             },
#             {
#                 "first_name": "Michael",
#                 "surname": "Chang",
#                 "email_address": "michael.chang@techinnovate.io",
#                 "phone_number": "+1-555-234-5678",
#             },
#             {
#                 "first_name": "Lisa",
#                 "surname": "Williams",
#                 "email_address": "lwilliams@globalfin.com",
#                 "phone_number": "+1-555-345-6789",
#             },
#             {
#                 "first_name": "James",
#                 "surname": "Rodriguez",
#                 "email_address": "james.rodriguez@consultpartners.net",
#                 "phone_number": "+1-555-456-7890",
#                 "whatsapp_number": "+1-555-456-7890",
#             },
#             {
#                 "first_name": "Emma",
#                 "surname": "Davis",
#                 "email_address": "edavis@healthplus.org",
#                 "phone_number": "+1-555-567-8901",
#             },
#         ]

#         # Add contacts to the system
#         for contact_info in contacts:
#             cls.transcript_manager.create_contact(**contact_info)

#     @classmethod
#     def _seed_message_history(cls):
#         """Seed a complex interconnected message history across multiple channels and time periods."""
#         # Define exchange IDs for different conversation threads
#         cls.acme_contract_exchange_id = 1
#         cls.tech_innovate_project_exchange_id = 2
#         cls.health_plus_consultation_exchange_id = 3
#         cls.rodriguez_whatsapp_exchange_id = 4

#         # ACME Contract Negotiation - Email Thread (Exchange 1)
#         acme_emails = [
#             {
#                 "exchange_id": cls.acme_contract_exchange_id,
#                 "sender": "sarah.johnson@acmecorp.com",
#                 "receiver": "assistant@company.com",
#                 "timestamp": "2023-10-15T09:30:00Z",
#                 "content": "Dear Assistant, I hope this email finds you well. We at ACME Corp are interested in discussing the renewal of our service contract. Could you please share the updated pricing structure for the enterprise plan? Best regards, Sarah Johnson, ACME Corp",
#                 "medium": "email",
#             },
#             {
#                 "exchange_id": cls.acme_contract_exchange_id,
#                 "sender": "assistant@company.com",
#                 "receiver": "sarah.johnson@acmecorp.com",
#                 "timestamp": "2023-10-15T11:45:00Z",
#                 "content": "Dear Sarah, Thank you for your interest in renewing the service contract with us. I'm attaching our updated pricing structure for the enterprise plan. The new plan includes additional features like advanced analytics and expanded API access. Let me know if you have any questions or if you'd like to schedule a call to discuss further. Best regards, Assistant",
#                 "medium": "email",
#             },
#             {
#                 "exchange_id": cls.acme_contract_exchange_id,
#                 "sender": "sarah.johnson@acmecorp.com",
#                 "receiver": "assistant@company.com",
#                 "timestamp": "2023-10-16T14:20:00Z",
#                 "content": "Dear Assistant, Thank you for sending over the pricing structure. We have a few questions about the new features. Would it be possible to schedule a call this week to discuss these in detail? How about Thursday at 2 PM EST? Best regards, Sarah",
#                 "medium": "email",
#             },
#             {
#                 "exchange_id": cls.acme_contract_exchange_id,
#                 "sender": "assistant@company.com",
#                 "receiver": "sarah.johnson@acmecorp.com",
#                 "timestamp": "2023-10-16T15:10:00Z",
#                 "content": "Dear Sarah, I'd be happy to schedule a call to discuss the new features. Thursday at 2 PM EST works well for me. I'll send a calendar invite with the conference details shortly. Looking forward to our conversation. Best regards, Assistant",
#                 "medium": "email",
#             },
#         ]

#         # Tech Innovate Project Discussion - Phone Call (Exchange 2)
#         tech_innovate_call = [
#             {
#                 "exchange_id": cls.tech_innovate_project_exchange_id,
#                 "sender": "michael.chang@techinnovate.io",
#                 "receiver": "assistant@company.com",
#                 "timestamp": "2023-10-18T10:00:00Z",
#                 "content": "Hi Assistant, this is Michael from Tech Innovate. I'm calling to discuss the implementation timeline for our new project. We've reviewed the proposal and have some concerns about the delivery schedule.",
#                 "medium": "phone",
#             },
#             {
#                 "exchange_id": cls.tech_innovate_project_exchange_id,
#                 "sender": "assistant@company.com",
#                 "receiver": "michael.chang@techinnovate.io",
#                 "timestamp": "2023-10-18T10:01:30Z",
#                 "content": "Hello Michael, thank you for calling. I understand your concerns about the delivery schedule. Could you please elaborate on which specific milestones you're concerned about?",
#                 "medium": "phone",
#             },
#             {
#                 "exchange_id": cls.tech_innovate_project_exchange_id,
#                 "sender": "michael.chang@techinnovate.io",
#                 "receiver": "assistant@company.com",
#                 "timestamp": "2023-10-18T10:03:45Z",
#                 "content": "We're particularly concerned about Phase 2, which involves integrating with our legacy systems. The timeline seems too aggressive given the complexity of our backend infrastructure.",
#                 "medium": "phone",
#             },
#             {
#                 "exchange_id": cls.tech_innovate_project_exchange_id,
#                 "sender": "assistant@company.com",
#                 "receiver": "michael.chang@techinnovate.io",
#                 "timestamp": "2023-10-18T10:05:15Z",
#                 "content": "That's a valid concern, Michael. We can certainly revisit the Phase 2 timeline. What if we extend that phase by two weeks and add an additional testing period? This would give us more buffer time for any unexpected challenges with the legacy system integration.",
#                 "medium": "phone",
#             },
#             {
#                 "exchange_id": cls.tech_innovate_project_exchange_id,
#                 "sender": "michael.chang@techinnovate.io",
#                 "receiver": "assistant@company.com",
#                 "timestamp": "2023-10-18T10:07:30Z",
#                 "content": "That sounds reasonable. Could you update the project plan and send it over for our review? Also, we'd like to schedule weekly progress meetings during Phase 2 to ensure we stay on track.",
#                 "medium": "phone",
#             },
#             {
#                 "exchange_id": cls.tech_innovate_project_exchange_id,
#                 "sender": "assistant@company.com",
#                 "receiver": "michael.chang@techinnovate.io",
#                 "timestamp": "2023-10-18T10:09:00Z",
#                 "content": "I'll have the updated project plan sent to you by tomorrow. And yes, weekly progress meetings during Phase 2 is a great idea. I'll include a proposed meeting schedule in the updated plan. Is there anything else you'd like to discuss today?",
#                 "medium": "phone",
#             },
#             {
#                 "exchange_id": cls.tech_innovate_project_exchange_id,
#                 "sender": "michael.chang@techinnovate.io",
#                 "receiver": "assistant@company.com",
#                 "timestamp": "2023-10-18T10:10:45Z",
#                 "content": "No, that covers everything for now. Thanks for addressing our concerns so promptly. We look forward to receiving the updated plan.",
#                 "medium": "phone",
#             },
#         ]

#         # HealthPlus Consultation - Mixed Email and Phone (Exchange 3)
#         health_plus_mixed = [
#             {
#                 "exchange_id": cls.health_plus_consultation_exchange_id,
#                 "sender": "edavis@healthplus.org",
#                 "receiver": "assistant@company.com",
#                 "timestamp": "2023-10-20T13:15:00Z",
#                 "content": "Hello Assistant, We at HealthPlus are considering implementing your patient management system. We have specific requirements around HIPAA compliance and data security. Could you provide some information on how your system addresses these concerns? Thank you, Emma Davis, IT Director, HealthPlus",
#                 "medium": "email",
#             },
#             {
#                 "exchange_id": cls.health_plus_consultation_exchange_id,
#                 "sender": "assistant@company.com",
#                 "receiver": "edavis@healthplus.org",
#                 "timestamp": "2023-10-20T15:45:00Z",
#                 "content": "Dear Emma, Thank you for your interest in our patient management system. We take HIPAA compliance and data security very seriously. Our system includes end-to-end encryption, role-based access controls, comprehensive audit logging, and regular security assessments. I've attached a detailed document outlining our security measures and compliance certifications. Would you be available for a call to discuss your specific requirements in more detail? Best regards, Assistant",
#                 "medium": "email",
#             },
#             {
#                 "exchange_id": cls.health_plus_consultation_exchange_id,
#                 "sender": "edavis@healthplus.org",
#                 "receiver": "assistant@company.com",
#                 "timestamp": "2023-10-21T09:30:00Z",
#                 "content": "Hello Assistant, I've reviewed the security documentation, and it looks comprehensive. I'd appreciate a call to discuss some specific implementation questions. How about tomorrow at 11 AM EST? Best, Emma",
#                 "medium": "email",
#             },
#             {
#                 "exchange_id": cls.health_plus_consultation_exchange_id,
#                 "sender": "assistant@company.com",
#                 "receiver": "edavis@healthplus.org",
#                 "timestamp": "2023-10-21T10:15:00Z",
#                 "content": "Hello Emma, Tomorrow at 11 AM EST works perfectly. I'll call you at the number provided in your signature. Looking forward to our discussion. Best regards, Assistant",
#                 "medium": "email",
#             },
#             {
#                 "exchange_id": cls.health_plus_consultation_exchange_id,
#                 "sender": "edavis@healthplus.org",
#                 "receiver": "assistant@company.com",
#                 "timestamp": "2023-10-22T11:00:00Z",
#                 "content": "Hi Assistant, this is Emma from HealthPlus. Thank you for calling as scheduled. I have some specific questions about how your system handles patient data encryption and access controls.",
#                 "medium": "phone",
#             },
#             {
#                 "exchange_id": cls.health_plus_consultation_exchange_id,
#                 "sender": "assistant@company.com",
#                 "receiver": "edavis@healthplus.org",
#                 "timestamp": "2023-10-22T11:01:30Z",
#                 "content": "Hello Emma, I'm happy to address your questions. Our system uses AES-256 encryption for all patient data, both at rest and in transit. For access controls, we implement a granular permission system that allows administrators to define exactly what data each user can access based on their role and responsibilities.",
#                 "medium": "phone",
#             },
#             {
#                 "exchange_id": cls.health_plus_consultation_exchange_id,
#                 "sender": "edavis@healthplus.org",
#                 "receiver": "assistant@company.com",
#                 "timestamp": "2023-10-22T11:03:45Z",
#                 "content": "That sounds promising. How does your system handle audit logging for compliance purposes? We need to ensure we can track who accessed what information and when.",
#                 "medium": "phone",
#             },
#             {
#                 "exchange_id": cls.health_plus_consultation_exchange_id,
#                 "sender": "assistant@company.com",
#                 "receiver": "edavis@healthplus.org",
#                 "timestamp": "2023-10-22T11:05:15Z",
#                 "content": "Our audit logging system records every action taken within the platform, including data access, modifications, and exports. Each log entry includes user information, timestamp, IP address, and the specific action performed. These logs are tamper-proof and can be exported for compliance reporting. We also provide a dashboard for administrators to monitor activity in real-time.",
#                 "medium": "phone",
#             },
#         ]

#         # James Rodriguez WhatsApp Chat (Exchange 4)
#         rodriguez_whatsapp = [
#             {
#                 "exchange_id": cls.rodriguez_whatsapp_exchange_id,
#                 "sender": "james.rodriguez@consultpartners.net",
#                 "receiver": "assistant@company.com",
#                 "timestamp": "2023-10-24T09:15:00Z",
#                 "content": "Hey there! I need some quick info about the marketing analytics dashboard we discussed last week. When can we expect the beta access?",
#                 "medium": "whatsapp",
#             },
#             {
#                 "exchange_id": cls.rodriguez_whatsapp_exchange_id,
#                 "sender": "assistant@company.com",
#                 "receiver": "james.rodriguez@consultpartners.net",
#                 "timestamp": "2023-10-24T09:18:00Z",
#                 "content": "Hi James! We're on track to provide beta access by November 5th. Would you like me to add you to the early access list?",
#                 "medium": "whatsapp",
#             },
#             {
#                 "exchange_id": cls.rodriguez_whatsapp_exchange_id,
#                 "sender": "james.rodriguez@consultpartners.net",
#                 "receiver": "assistant@company.com",
#                 "timestamp": "2023-10-24T09:20:00Z",
#                 "content": "Yes, please add me and my team lead, Alex, as well. His email is alex.peterson@consultpartners.net",
#                 "medium": "whatsapp",
#             },
#             {
#                 "exchange_id": cls.rodriguez_whatsapp_exchange_id,
#                 "sender": "assistant@company.com",
#                 "receiver": "james.rodriguez@consultpartners.net",
#                 "timestamp": "2023-10-24T09:22:00Z",
#                 "content": "Great! I've added both of you to the early access list. You'll receive an email with login instructions once the beta is live. Is there anything specific you're most interested in testing?",
#                 "medium": "whatsapp",
#             },
#             {
#                 "exchange_id": cls.rodriguez_whatsapp_exchange_id,
#                 "sender": "james.rodriguez@consultpartners.net",
#                 "receiver": "assistant@company.com",
#                 "timestamp": "2023-10-24T09:25:00Z",
#                 "content": "We're particularly interested in the custom report generation feature and the campaign performance prediction tool. Our client is very excited about those capabilities.",
#                 "medium": "whatsapp",
#             },
#             {
#                 "exchange_id": cls.rodriguez_whatsapp_exchange_id,
#                 "sender": "assistant@company.com",
#                 "receiver": "james.rodriguez@consultpartners.net",
#                 "timestamp": "2023-10-24T09:28:00Z",
#                 "content": "Noted! I'll make sure those features are highlighted in your onboarding experience. We're actually adding some new templates for the custom report generation this week, so your timing is perfect.",
#                 "medium": "whatsapp",
#             },
#         ]

#         # Log all messages
#         all_messages = (
#             acme_emails + tech_innovate_call + health_plus_mixed + rodriguez_whatsapp
#         )
#         for msg_data in all_messages:
#             message = Message(**msg_data)
#             cls.transcript_manager.log_messages([message])

#         # Create summaries for each exchange
#         cls.transcript_manager.summarize(
#             exchange_ids=cls.acme_contract_exchange_id,
#             guidance="Focus on contract renewal details and next steps",
#         )

#         cls.transcript_manager.summarize(
#             exchange_ids=cls.tech_innovate_project_exchange_id,
#             guidance="Focus on project timeline concerns and agreed solutions",
#         )

#         cls.transcript_manager.summarize(
#             exchange_ids=cls.health_plus_consultation_exchange_id,
#             guidance="Focus on HIPAA compliance requirements and system security features",
#         )

#         cls.transcript_manager.summarize(
#             exchange_ids=cls.rodriguez_whatsapp_exchange_id,
#             guidance="Focus on marketing analytics dashboard beta access and features of interest",
#         )

#     def test_complex_information_retrieval(self):
#         """Test the ability to retrieve complex, nuanced information spanning multiple exchanges and media types."""

#         # Test finding all communication with a specific contact across different media types
#         result = self.transcript_manager.ask(
#             "What communications have I had with Emma Davis from HealthPlus, and what were they about?"
#         )
#         assert (
#             result
#         ), "No response received for complex query about Emma Davis communications"
#         assert (
#             "HIPAA" in result or "security" in result
#         ), "Response doesn't mention key topics from communications with Emma Davis"

#         # Test understanding of chronological context across exchanges
#         result = self.transcript_manager.ask(
#             "What was decided about the Tech Innovate project timeline, and what actions did I promise to take?"
#         )
#         assert (
#             result
#         ), "No response received for complex query about Tech Innovate timeline"
#         assert (
#             "project plan" in result.lower() or "phase 2" in result.lower()
#         ), "Response doesn't mention key decisions about project timeline"

#         # Test cross-referencing capabilities across different communication media
#         result = self.transcript_manager.ask(
#             "Which clients discussed implementation timelines with me, and what were their concerns?"
#         )
#         assert (
#             "Tech Innovate" in result or "Michael" in result
#         ), "Response doesn't mention Tech Innovate implementation discussions"

#         # Test temporal reasoning across exchanges
#         result = self.transcript_manager.ask(
#             "What meetings were scheduled during my communications last month, and with whom?"
#         )
#         assert (
#             "Sarah" in result or "Emma" in result
#         ), "Response doesn't identify scheduled meetings correctly"

#         # Test summarization capabilities across complex communication threads
#         result = self.transcript_manager.ask(
#             "Summarize all the ongoing client projects and their current status based on recent communications"
#         )
#         assert (
#             len(result) > 200
#         ), "Summary is too brief for complex multi-project status overview"
#         assert (
#             "ACME" in result and "Tech Innovate" in result
#         ), "Summary doesn't include major client projects"

#     def test_complex_contact_management(self):
#         """Test advanced contact management capabilities with interrelated information."""

#         # Update contact with additional information
#         self.transcript_manager.update_contact(
#             contact_id=2,  # Michael Chang
#             phone_number="+1-555-234-5679",  # Updated phone number
#         )

#         # Test retrieving updated contact info
#         contacts = self.transcript_manager._search_contacts(filter="surname == 'Chang'")
#         assert len(contacts) > 0, "Contact wasn't found"
#         assert contacts[0].phone_number == "+1-555-234-5679", "Contact update failed"

#         # Test complex natural language query about contacts and their communications
#         result = self.transcript_manager.ask(
#             "Who from Tech Innovate has contacted me, what was discussed, and what is their current contact information?"
#         )
#         assert (
#             "Michael" in result and "Chang" in result
#         ), "Contact name missing from response"
#         assert "+1-555-234-5679" in result, "Updated phone number missing from response"
#         assert (
#             "legacy systems" in result or "Phase 2" in result
#         ), "Discussion topics missing from response"

#     def test_semantic_search_capabilities(self):
#         """Test the semantic search capabilities for finding relevant information across exchanges."""

#         # Test semantic search using vector embeddings
#         messages = self.transcript_manager._nearest_messages(
#             text="HIPAA compliance healthcare data security encryption", k=5
#         )
#         assert any(
#             "HIPAA" in msg.content for msg in messages
#         ), "Semantic search failed to find relevant HIPAA content"

#         # Test finding relevant messages without exact keyword matches
#         messages = self.transcript_manager._nearest_messages(
#             text="legacy system integration challenges", k=5
#         )
#         assert any(
#             "legacy" in msg.content for msg in messages
#         ), "Semantic search failed to find legacy system discussions"

#         # Test natural language query using semantic understanding
#         result = self.transcript_manager.ask(
#             "Find communications where clients expressed concerns about implementation complexity"
#         )
#         assert (
#             "Tech Innovate" in result or "Michael" in result
#         ), "Failed to identify communications about implementation concerns"

#     def test_cross_exchange_information_synthesis(self):
#         """Test the ability to synthesize information across multiple exchanges and time periods."""

#         # Ask a question that requires understanding multiple exchanges to answer correctly
#         result = self.transcript_manager.ask(
#             "Based on all communications, which clients are interested in security features and which are concerned about implementation timelines?"
#         )

#         # Check for nuanced understanding of different client concerns
#         assert (
#             "HealthPlus" in result and "security" in result.lower()
#         ), "Failed to identify HealthPlus security interests"
#         assert "Tech Innovate" in result and (
#             "timeline" in result.lower() or "schedule" in result.lower()
#         ), "Failed to identify Tech Innovate timeline concerns"

#         # Test ability to infer relationships between exchanges
#         result = self.transcript_manager.ask(
#             "What follow-up actions have I promised to different clients, and which are still pending based on the communication history?"
#         )

#         assert (
#             "project plan" in result.lower() or "updated plan" in result.lower()
#         ), "Missing follow-up about project plan"
#         assert (
#             "beta access" in result.lower() or "early access" in result.lower()
#         ), "Missing follow-up about beta access"

#     def test_multimodal_conversation_tracking(self):
#         """Test ability to track conversations that move between different communication media."""

#         # Test tracking conversation that moved from email to phone
#         result = self.transcript_manager.ask(
#             "How did my conversation with Emma Davis evolve from initial contact to the phone call, and what were the key points discussed?"
#         )

#         assert (
#             "email" in result.lower() and "phone" in result.lower()
#         ), "Failed to identify different communication channels"
#         assert (
#             "HIPAA" in result or "compliance" in result
#         ), "Failed to identify key discussion points"
#         assert (
#             "encryption" in result or "security" in result
#         ), "Failed to identify technical details discussed in phone call"

#         # Add a new message to an existing conversation thread to test temporal awareness
#         new_followup_message = Message(
#             exchange_id=self.health_plus_consultation_exchange_id,
#             sender="assistant@company.com",
#             receiver="edavis@healthplus.org",
#             timestamp=datetime.datetime.now().isoformat(),
#             content="Hello Emma, I'm following up on our discussion about the patient management system. I've prepared a detailed proposal addressing all the security and compliance requirements we discussed. Would you like me to send it over for your review?",
#             medium="email",
#         )
#         self.transcript_manager.log_messages([new_followup_message])

#         # Test if the system can incorporate the new message into its understanding
#         result = self.transcript_manager.ask(
#             "What is the current status of my discussions with HealthPlus, and what was the most recent communication?"
#         )

#         assert (
#             "proposal" in result.lower() or "follow" in result.lower()
#         ), "Failed to include recent follow-up communication"
#         assert (
#             "security" in result.lower() or "compliance" in result.lower()
#         ), "Failed to maintain context from earlier discussions"

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_communication/test_transcript_manager/test_basics.py`:

```py
import time
import unify
import random
import pytest
from datetime import datetime, UTC

from unity.communication.types.message import Message, VALID_MEDIA
from unity.communication.transcript_manager.transcript_manager import TranscriptManager
from unity.events.event_bus import EventBus, Event
from tests.helpers import _handle_project

CONTACTS = [
    {
        "contact_id": 0,
        "first_name": "John",
        "surname": "Smith",
        "email_address": "johnsmith11@gmail.com",
        "phone_number": "+1234567890",
        "whatsapp_number": "+1234567890",
    },
    {
        "contact_id": 1,
        "first_name": "Nancy",
        "surname": "Gray",
        "email_address": "nancy_gray@outlook.com",
        "phone_number": "+1987654320",
        "whatsapp_number": "+1987654320",
    },
]

MESSAGES = [
    "Hello, how are you?",
    "Sorry I couldn't hear you",
    "Hell no, I won't do that",
    "Wow, did you see that?",
    "Goodbye",
]


def _create_contacts():
    unify.create_logs(
        context="Contacts",
        entries=CONTACTS,
    )


@pytest.mark.unit
@_handle_project
def test_create_contact():
    transcript_manager = TranscriptManager(EventBus())
    transcript_manager.create_contact(
        first_name="Dan",
    )
    contacts = transcript_manager._search_contacts()
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 0,
        "first_name": "Dan",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
    }


@pytest.mark.unit
@_handle_project
def test_update_contact():
    transcript_manager = TranscriptManager(EventBus())

    # create
    transcript_manager.create_contact(
        first_name="Dan",
    )

    # check
    contacts = transcript_manager._search_contacts()
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 0,
        "first_name": "Dan",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
    }

    # update
    transcript_manager.update_contact(
        contact_id=0,
        first_name="Daniel",
    )

    # check
    contacts = transcript_manager._search_contacts()
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 0,
        "first_name": "Daniel",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
    }


@pytest.mark.unit
@_handle_project
def test_create_contacts():
    transcript_manager = TranscriptManager(EventBus())

    # first
    transcript_manager.create_contact(
        first_name="Dan",
    )
    contacts = transcript_manager._search_contacts()
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 0,
        "first_name": "Dan",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
    }

    # second
    transcript_manager.create_contact(
        first_name="Tom",
    )
    contacts = transcript_manager._search_contacts()
    assert len(contacts) == 2
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 1,
        "first_name": "Tom",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
    }


@pytest.mark.unit
@_handle_project
def test_search_contacts():
    transcript_manager = TranscriptManager(EventBus())
    transcript_manager.create_contact(
        first_name="Dan",
    )


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_log_messages():
    event_bus = EventBus()
    [
        await event_bus.publish(
            Event(
                context="message",
                timestamp=datetime.now(UTC).isoformat(),
                payload=Message(
                    medium=random.choice(VALID_MEDIA),
                    sender_id=random.randint(0, 2),
                    receiver_id=random.randint(0, 2),
                    timestamp=datetime.now(UTC).isoformat(),
                    content=random.choice(MESSAGES),
                    exchange_id=i,
                ),
            )
        )
        for i in range(10)
    ]
    event_bus.join_published()


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_get_messages():
    start_time = datetime.now(UTC).isoformat()
    time.sleep(0.1)
    random.seed(0)
    event_bus = EventBus()
    transcript_manager = TranscriptManager(event_bus)

    # log messages
    [
        await event_bus.publish(
            Event(
                context="Messages",
                timestamp=datetime.now(UTC).isoformat(),
                payload=Message(
                    medium=random.choice(VALID_MEDIA),
                    sender_id=random.randint(0, 2),
                    receiver_id=random.randint(0, 2),
                    timestamp=datetime.now(UTC).isoformat(),
                    content=random.choice(MESSAGES),
                    exchange_id=i,
                ),
            )
        )
        for i in range(10)
    ]
    event_bus.join_published()

    ## get all

    messages = transcript_manager._search_messages()
    assert len(messages) == 10
    assert all(isinstance(msg, Message) for msg in messages)

    ## search

    # sender

    messages = transcript_manager._search_messages(filter="sender_id == 0")
    assert len(messages) == 3
    assert all(isinstance(msg, Message) for msg in messages)

    # contains

    messages = transcript_manager._search_messages(filter="'Hell' in content")
    assert len(messages) == 3
    assert all(isinstance(msg, Message) for msg in messages)

    # does not contain

    messages = transcript_manager._search_messages(filter="',' not in content")
    assert len(messages) == 5
    assert all(isinstance(msg, Message) for msg in messages)

    # medium

    messages = transcript_manager._search_messages(
        filter="medium in ('email', 'whatsapp_message')",
    )
    assert len(messages) == 1
    assert all(isinstance(msg, Message) for msg in messages)

    # timestamp

    messages = transcript_manager._search_messages(filter=f"timestamp < '{start_time}'")
    assert len(messages) == 0
    messages = transcript_manager._search_messages(filter=f"timestamp > '{start_time}'")
    assert len(messages) == 10


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_summarize_exchanges():
    event_bus = EventBus()
    transcript_manager = TranscriptManager(event_bus)

    # create contacts
    _create_contacts()

    # phone call
    [
        await event_bus.publish(
            Event(
                context="message",
                timestamp=datetime.now(UTC).isoformat(),
                payload=Message(
                    medium="phone_call",
                    sender_id=i % 2,
                    receiver_id=(i + 1) % 2,
                    timestamp=datetime.now(UTC).isoformat(),
                    content=msg,
                    exchange_id=0,
                ),
            )
        )
        for i, msg in enumerate(
            [
                "Hey, how's it going?",
                "Yeah good thanks, how can I help you?",
                "How are your office staplers doing? Are they underperforming?",
                "Actually yeah, they're a bit rusty, but I can't make any buying decisions. My manager can.",
                "Okay, no worries. Let's catch up again soon.",
            ]
        )
    ]

    # email exchange
    [
        await event_bus.publish(
            Event(
                context="message",
                timestamp=datetime.now(UTC).isoformat(),
                payload=Message(
                    medium="email",
                    sender_id=i % 2,
                    receiver_id=(i + 1) % 2,
                    timestamp=datetime.now(UTC).isoformat(),
                    content=msg,
                    exchange_id=1,
                ),
            )
        )
        for i, msg in enumerate(
            [
                "Great catching up the other day, did you manage to talk to your manager?",
                "Hey, yeah I did actually. I'll reach out soon.",
                "Okay great, thanks!",
            ]
        )
    ]

    # whatsapp exchange
    [
        await event_bus.publish(
            Event(
                context="message",
                timestamp=datetime.now(UTC).isoformat(),
                payload=Message(
                    medium="whatsapp_message",
                    sender_id=(i + 1) % 2,
                    receiver_id=i % 2,
                    timestamp=datetime.now(UTC).isoformat(),
                    content=msg,
                    exchange_id=2,
                ),
            )
        )
        for i, msg in enumerate(
            [
                "Hey, yeah we'd love to buy your staplers!",
                "Great! Excited to hear :)",
            ]
        )
    ]
    event_bus.join_published()

    # summarize
    summary = transcript_manager.summarize(exchange_ids=[0, 1, 2])

    # retrieve summary
    summaries = transcript_manager._search_summaries()
    assert len(summaries) == 1
    assert summaries[0].model_dump() == {
        "exchange_ids": [0, 1, 2],
        "summary": summary,
    }

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_communication/test_transcript_manager/test_transcript_embedding.py`:

```py
import pytest
from datetime import datetime, UTC

from unity.communication.transcript_manager.transcript_manager import TranscriptManager
from unity.communication.types.message import Message, VALID_MEDIA
from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event
import random


@pytest.mark.unit
@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_transcript_embedding_semantic_search():
    """
    Test the transcript manager's ability to perform semantic search via nearest message retrieval.
    """
    # Create the TranscriptManager instance
    tm = TranscriptManager(EventBus())

    # Create a few test messages
    msgs = [
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=1,
            receiver_id=2,
            timestamp="2025-05-19 12:00:00",
            content="Can you help me with my banking questions? I'm looking to set up a new account.",
            exchange_id=1,
        ),
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=2,
            receiver_id=1,
            timestamp="2025-05-19 12:00:01",
            content="I'd be happy to help with your banking needs! What type of account would you like to set up? Checking, savings, or investment?",
            exchange_id=1,
        ),
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=1,
            receiver_id=2,
            timestamp="2025-05-19 12:00:02",
            content="I'm interested in learning about Python programming, especially data science applications.",
            exchange_id=1,
        ),
    ]

    event_bus = EventBus()
    [
        await event_bus.publish(
            Event(
                context="message",
                timestamp=datetime.now(UTC).isoformat(),
                payload=Message(
                    medium=random.choice(VALID_MEDIA),
                    sender_id=random.randint(0, 2),
                    receiver_id=random.randint(0, 2),
                    timestamp=datetime.now(UTC).isoformat(),
                    content=msg,
                    exchange_id=i,
                ),
            )
        )
        for i, msg in enumerate(msgs)
    ]
    event_bus.join_published()

    # Ensure that a lexical search for the word 'budgeting' returns no results
    lexical_results = tm._search_messages(filter="'budgeting' in content")
    assert lexical_results == []

    # Use semantic search to find the nearest messages to the query
    nearest = tm._nearest_messages(text="banking and budgeting", k=2)

    # Verify the result length and type
    assert len(nearest) == 2
    assert all(isinstance(msg, Message) for msg in nearest)

    # Verify that the messages are returned in ascending order of distance
    assert nearest[0].content == msgs[-1].content
    assert nearest[1].content == msgs[-2].content

    # Test k-limit behavior
    all_nearest = tm._nearest_messages(text="banking and budgeting", k=10)
    assert len(all_nearest) == 3  # Should return all 3 messages we inserted

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/conftest.py`:

```py
"""
tests/conftest.py
=================

Global pytest configuration.

• `--unify-stub` *or* `USE_UNIFY_STUB=1` ➜ replace the **persistence** parts
  of the `unify` SDK with an in-memory implementation, while *optionally*
  keeping the real `unify.Unify` class for live LLM calls.

  – With flag         → in-memory logs, live LLM
  – Without flag      → untouched, everything goes to real backend
"""

from __future__ import annotations

import os
import sys
import types
import importlib
from typing import Any, Dict, List, Optional

import pytest

import unify

unify.activate("UnityTests")


# --------------------------------------------------------------------------- #
#  Command-line flag                                                          #
# --------------------------------------------------------------------------- #

# Flag to track if we're using the stub version
_using_unify_stub = False


def pytest_addoption(parser):
    parser.addoption(
        "--unify-stub",
        action="store_true",
        help="Use an in-memory stub for unite.log / projects whilst "
        "leaving LLM calls intact.",
    )


# --------------------------------------------------------------------------- #
#  Session-wide hook – install stub *before* any project imports              #
# --------------------------------------------------------------------------- #


def pytest_sessionstart(session):
    global _using_unify_stub
    cmd_flag = session.config.getoption("--unify-stub")
    env_var = os.getenv("USE_UNIFY_STUB")

    # Only consider env_var as True if it's set to a non-zero/non-empty value
    use_env_var = env_var and env_var.lower() not in ("0", "false", "no", "")

    use_stub = cmd_flag or use_env_var

    if use_stub:
        _using_unify_stub = True
        _install_unify_stub()
        _install_requests_mock()
    else:
        _using_unify_stub = False


# Function to check if we're using the unify stub
def is_using_unify_stub():
    """Return True if tests are running with the unify stub."""
    global _using_unify_stub
    return _using_unify_stub


# Define a marker for tests that require the real unify
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_real_unify: mark test as requiring the real unify implementation",
    )
    config.addinivalue_line(
        "markers",
        "unit: mark a test as a deterministic unit test",
    )
    config.addinivalue_line(
        "markers",
        "eval: mark a test as a fuzzy evaluation test for English language APIs",
    )


# Skip tests marked with requires_real_unify when using the unify stub
def pytest_runtest_setup(item):
    if any(mark.name == "requires_real_unify" for mark in item.iter_markers()):
        if is_using_unify_stub():
            pytest.skip("Test requires real unify implementation")


# Pytest fixture to skip tests that require the real unify
@pytest.fixture
def requires_real_unify(request):
    """Skip tests if unify stub is being used."""
    if is_using_unify_stub():
        pytest.skip("Test requires real unify implementation")


# --------------------------------------------------------------------------- #
#  Helper: mock requests library                                              #
# --------------------------------------------------------------------------- #


def _install_requests_mock():
    """Mock the requests library for unify API calls during tests."""
    import sys
    import types

    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code
            self.text = str(json_data)

        def json(self):
            return self.json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                from requests.exceptions import HTTPError

                raise HTTPError(f"{self.status_code} Error", response=self)

    class MockRequests:
        @staticmethod
        def _get_columns_from_log(context):
            """Helper function to extract column metadata from logs."""
            import unify

            # Get direct access to the internal logs storage
            store = None
            unify_module = sys.modules.get("unify")
            if unify_module and hasattr(unify_module, "_ctx_store"):
                try:
                    # Get the logs directly from the store
                    ctx_store = getattr(unify_module, "_ctx_store")
                    store = ctx_store(context)
                except:
                    # Fall back to regular API
                    pass

            # If we couldn't get direct access, use the normal API
            if store is None:
                store = unify.get_logs(context=context)

            # Find column metadata logs
            column_logs = [log for log in store if "__columns__" in log.entries]

            # Return column definitions if found
            if column_logs:
                return column_logs[0].entries.get("__columns__", {})

            return {}

        @staticmethod
        def request(method, url, json=None, headers=None, **kwargs):
            # Get or extract table name from URL
            import re

            if json:
                pass

            table_match = re.search(r"Knowledge/([^/]+)", url)
            table_name = table_match.group(1) if table_match else None

            # Process requests based on URL pattern
            if url == "https://api.unify.ai/v0/logs/rename_field":
                # Handle rename field request
                import unify

                if json:
                    context = json.get("context")
                    old_field_name = json.get("old_field_name")
                    new_field_name = json.get("new_field_name")

                    if context and old_field_name and new_field_name:
                        unify._rename_column(context, old_field_name, new_field_name)
                    else:
                        pass
                else:
                    pass

                return MockResponse({"success": True, "message": "Column renamed"})
            elif "/columns" in url:
                # Creating/modifying columns
                if table_name and json and method == "POST":
                    # Access the unify module directly
                    import unify

                    # Extract column definitions
                    column_definitions = {}
                    if json and "columns" in json:
                        for col_name, col_type in json["columns"].items():
                            # Store the column type directly, not as a dict
                            column_definitions[col_name] = col_type

                    # Find or create a log with __columns__ entry
                    column_logs = [
                        log
                        for log in unify.get_logs(context=f"Knowledge/{table_name}")
                        if "__columns__" in log.entries
                    ]

                    if column_logs:
                        # Update existing column metadata
                        column_log = column_logs[0]
                        existing = column_log.entries.get("__columns__", {})
                        column_log.update_entries(
                            __columns__={
                                **existing,
                                **column_definitions,
                            },
                        )
                    else:
                        # Create new column metadata log
                        unify.log(
                            context=f"Knowledge/{table_name}",
                            __columns__=column_definitions,
                        )

                return MockResponse(
                    {"success": True, "message": "Column operation successful"},
                )
            elif "/rename" in url and "contexts" in url:
                # Renaming tables
                import unify
                import re

                url_parts = url.split("/")

                old_context = None
                for i, part in enumerate(url_parts):
                    if part == "contexts" and i + 2 < len(url_parts):
                        old_context = f"{url_parts[i+1]}/{url_parts[i+2]}"
                        break

                new_context = json.get("name")

                if old_context and new_context:
                    unify.create_context(new_context)

                    # Get the logs from the old context
                    logs = unify.get_logs(context=old_context)

                    # For each log, copy its entries to the new context
                    for log in logs:
                        unify.log(context=new_context, **log.entries)

                    # Delete the old context
                    unify.delete_context(old_context)

                return MockResponse(
                    {"success": True, "message": "Table renamed successfully"},
                )
            elif "/logs/fields" in url:
                # This endpoint is called by knowledge_manager._get_columns
                import unify

                if method == "POST" and json:
                    project = json.get("project")
                    context = json.get("context")
                    fields = json.get("fields", {})

                    if context and fields:
                        # Find or create column metadata log
                        column_logs = [
                            log
                            for log in unify.get_logs(context=context)
                            if "__columns__" in log.entries
                        ]

                        if column_logs:
                            # Update existing column metadata
                            column_log = column_logs[0]
                            existing = column_log.entries.get("__columns__", {})
                            column_log.update_entries(
                                __columns__={**existing, **fields},
                            )
                        else:
                            # Create new column metadata log
                            unify.log(
                                context=context,
                                __columns__=fields,
                            )

                    return MockResponse(
                        {"success": True, "message": "Columns created successfully"},
                    )

                # Handle GET requests to retrieve column information
                # Extract query parameters
                import urllib.parse

                query = url.split("?")[-1] if "?" in url else ""
                params = dict(urllib.parse.parse_qsl(query))

                # Get context parameter - handle both direct parameter and URL pattern
                context = params.get("context")
                if not context and table_name:
                    context = f"Knowledge/{table_name}"

                if context:
                    # Use unify.get_fields to retrieve the column definitions for the context.
                    # This helper already inspects the raw log store (including metadata logs) and
                    # therefore reflects the authoritative list of columns for a context, exactly
                    # as the real Unify backend would.
                    column_data = unify.get_fields(context=context) or {}

                    # Format the payload in the same shape the real API returns – a mapping from
                    # field name to an object that at least contains the "data_type" key.
                    formatted_columns = {
                        name: {"data_type": dtype}
                        for name, dtype in column_data.items()
                    }

                    return MockResponse(formatted_columns)
            elif "/logs/derived" in url:
                # Creating derived columns
                if json:
                    context = json.get("context")
                    column_name = json.get("key")
                    equation = json.get("equation", "")
                    referenced_logs = json.get("referenced_logs", {})

                    if context and column_name and equation:
                        # Store in columns metadata using unify directly
                        import unify

                        # Get metadata logs
                        column_logs = [
                            log
                            for log in unify.get_logs(context=context)
                            if "__columns__" in log.entries
                        ]

                        # Store column type and equation in metadata
                        if column_logs:
                            # Update existing column metadata
                            column_log = column_logs[0]
                            column_log.update_entries(
                                __columns__={
                                    **column_log.entries.get("__columns__", {}),
                                    **{column_name: "derived"},
                                },
                                __equations__={
                                    **column_log.entries.get("__equations__", {}),
                                    **{column_name: equation},
                                },
                            )
                        else:
                            # Create new column metadata log
                            unify.log(
                                context=context,
                                __columns__={column_name: "derived"},
                                __equations__={column_name: equation},
                            )

                        # Get all logs except metadata
                        logs = [
                            log
                            for log in unify.get_logs(context=context)
                            if "__columns__" not in log.entries
                            and "__equations__" not in log.entries
                        ]

                        # Apply the derived column to all logs immediately
                        if logs:
                            # Simple equation parser - handle basic expressions with field references
                            # Try to evaluate with each log's fields
                            for log in logs:
                                try:
                                    # Replace field references with values
                                    eval_equation = equation

                                    # Handle field references like {lg:fieldname}
                                    import re

                                    field_refs = re.findall(
                                        r"\{([^{}]+):([^{}]+)\}",
                                        eval_equation,
                                    )

                                    # First get values for referenced fields
                                    local_vars = {}
                                    for ref_name, field_name in field_refs:
                                        ref_context = referenced_logs.get(
                                            ref_name,
                                            {},
                                        ).get("context", context)
                                        field_value = log.entries.get(field_name)
                                        if field_value is not None:
                                            local_vars[f"{ref_name}_{field_name}"] = (
                                                field_value
                                            )
                                            eval_equation = eval_equation.replace(
                                                f"{{{ref_name}:{field_name}}}",
                                                f"{ref_name}_{field_name}",
                                            )

                                    # Handle direct field references like {fieldname}
                                    direct_refs = re.findall(
                                        r"\{([^{}]+)\}",
                                        eval_equation,
                                    )
                                    for field_name in direct_refs:
                                        field_value = log.entries.get(field_name)
                                        if field_value is not None:
                                            local_vars[field_name] = field_value
                                            eval_equation = eval_equation.replace(
                                                f"{{{field_name}}}",
                                                field_name,
                                            )

                                    # Evaluate the equation with the field values
                                    result = eval(
                                        eval_equation,
                                        {"__builtins__": {}},
                                        local_vars,
                                    )
                                    log.entries[column_name] = result
                                except Exception as e:
                                    pass

                return MockResponse(
                    {"success": True, "message": "Derived column created"},
                )
            elif "/logs/rename_field" in url.lower():
                # Renaming columns
                import unify

                if json:
                    context = json.get("context")
                    old_field_name = json.get("old_field_name")
                    new_field_name = json.get("new_field_name")

                    if context and old_field_name and new_field_name:
                        # Get all non-metadata logs in the context
                        logs = [
                            log
                            for log in unify._ctx_store(context)
                            if "__columns__" not in log.entries
                        ]

                        # Rename the field in each log entry
                        for log in logs:
                            if old_field_name in log.entries:
                                # Preserve position of the field in the entries
                                old_value = log.entries.pop(old_field_name)

                                # Get the keys of the entries in their original order
                                keys = list(log.entries.keys())

                                # Create a new ordered dict with the new field name in place of the old one
                                new_entries = {}

                                # Find where the original field was in the order
                                # If it's a new field (not in the original), we'll add it at the beginning
                                original_keys = list(log.entries.keys())

                                # Loop through adding each key in original order
                                added_new_field = False

                                # Handle an empty log case
                                if not keys:
                                    new_entries[new_field_name] = old_value
                                else:
                                    # If the field was the first one, maintain that position
                                    if (
                                        len(original_keys) == 0
                                        or old_field_name < original_keys[0]
                                    ):
                                        new_entries[new_field_name] = old_value
                                        added_new_field = True

                                    # Add all other fields in their original order
                                    for k, v in log.entries.items():
                                        # If we haven't added the new field yet and we're past where
                                        # the old field would have been alphabetically, add it now
                                        if not added_new_field and k > old_field_name:
                                            new_entries[new_field_name] = old_value
                                            added_new_field = True
                                        new_entries[k] = v

                                    # If we haven't added the new field yet, add it at the end
                                    if not added_new_field:
                                        new_entries[new_field_name] = old_value

                                log.entries = new_entries

                        # Also update column metadata
                        column_logs = [
                            log
                            for log in unify._ctx_store(context)
                            if "__columns__" in log.entries
                        ]
                        if column_logs:
                            column_log = column_logs[0]
                            columns = column_log.entries.get("__columns__", {})
                            if old_field_name in columns:
                                updated_columns = {}
                                for col_name, col_type in columns.items():
                                    if col_name == old_field_name:
                                        updated_columns[new_field_name] = col_type
                                    else:
                                        updated_columns[col_name] = col_type
                                column_log.entries["__columns__"] = updated_columns

                    else:
                        pass

                return MockResponse({"success": True, "message": "Column renamed"})
            elif method == "DELETE" and "/columns/" in url:
                # Handle DELETE request to remove a column
                import re
                import unify

                # Extract context and column name from URL
                # Pattern example: /project/.../contexts/Knowledge/MyTable/columns/x
                column_pattern = re.search(
                    r"/contexts/([^/]+)/([^/]+)/columns/([^/?]+)",
                    url,
                )
                if column_pattern:
                    context = f"{column_pattern.group(1)}/{column_pattern.group(2)}"
                    column_name = column_pattern.group(3)

                    # Get all non-metadata logs in the context
                    logs = [
                        log
                        for log in unify._ctx_store(context)
                        if "__columns__" not in log.entries
                    ]

                    # Remove the column from each log entry
                    for log in logs:
                        if column_name in log.entries:
                            log.entries.pop(column_name, None)

                    # Also update column metadata
                    column_logs = [
                        log
                        for log in unify._ctx_store(context)
                        if "__columns__" in log.entries
                    ]
                    if column_logs:
                        column_log = column_logs[0]
                        columns = column_log.entries.get("__columns__", {})
                        if column_name in columns:
                            updated_columns = {
                                k: v for k, v in columns.items() if k != column_name
                            }
                            column_log.entries["__columns__"] = updated_columns

                return MockResponse(
                    {"success": True, "message": "Column deleted via DELETE"},
                )
            elif "/logs?delete_empty_logs=True" in url:
                # Deleting columns
                import re
                import unify
                import urllib.parse

                if json:
                    context = json.get("context")

                    # Extract column name from ids_and_fields format: [[log_id, field_name], ...]
                    ids_and_fields = json.get("ids_and_fields", [])
                    if (
                        ids_and_fields
                        and isinstance(ids_and_fields, list)
                        and len(ids_and_fields) > 0
                    ):
                        # The format appears to be [[log_id, field_name], ...] where log_id can be None
                        # to indicate deletion from all logs
                        first_field = ids_and_fields[0]
                        if isinstance(first_field, list) and len(first_field) > 1:
                            column_name = first_field[1]

                # If context and column were found, perform the deletion
                if context and column_name:
                    # Get all non-metadata logs in the context
                    logs = [
                        log
                        for log in unify._ctx_store(context)
                        if "__columns__" not in log.entries
                    ]

                    # Remove the column from each log entry
                    for log in logs:
                        if column_name in log.entries:
                            log.entries.pop(column_name, None)

                    # Also update column metadata
                    column_logs = [
                        log
                        for log in unify._ctx_store(context)
                        if "__columns__" in log.entries
                    ]
                    if column_logs:
                        column_log = column_logs[0]
                        columns = column_log.entries.get("__columns__", {})
                        if column_name in columns:
                            updated_columns = {
                                k: v for k, v in columns.items() if k != column_name
                            }
                            column_log.entries["__columns__"] = updated_columns

                return MockResponse({"success": True, "message": "Column deleted"})
            elif "/logs" in url:
                # Generic logs endpoint
                return MockResponse(
                    {"success": True, "message": "Log operation successful"},
                )
            else:
                # Default response
                return MockResponse(
                    {"success": True, "message": "Operation successful"},
                )

    # Create a module-like object
    mock_requests = types.ModuleType("requests")

    # Copy all the original requests attributes
    try:
        import requests as original_requests

        for attr in dir(original_requests):
            if not attr.startswith("__"):
                setattr(mock_requests, attr, getattr(original_requests, attr))
    except ImportError:
        pass  # If requests isn't available, we'll just use our mock

    # Override the request method
    mock_requests.request = MockRequests.request

    # Install our mock
    sys.modules["requests"] = mock_requests


# --------------------------------------------------------------------------- #
#  Helper: stub implementation                                                #
# --------------------------------------------------------------------------- #


def _install_unify_stub() -> None:  # noqa: C901 – long but linear
    """
    Monkey-patch the `unify` module so that:

      • log / project APIs are fully in-memory (no network / DB).
      • If the *real* SDK is present, everything else proxies through,
        so LLM calls still work.
    """
    if "unify" in sys.modules:  # already imported → too late
        return

    try:
        _real_unify = importlib.import_module("unify")  # genuine SDK
        _have_real = True
    except ModuleNotFoundError:
        _real_unify = None
        _have_real = False

    # ------------------------------------------------------------------ #
    #  In-memory store                                                   #
    # ------------------------------------------------------------------ #
    _projects: Dict[str, Dict[str, List["Log"]]] = {}
    _current: Optional[str] = None
    _next_id = 0

    class Log:  # minimal Log object
        def __init__(self, id_: int, entries: Dict[str, Any]):
            self.id = id_
            self.entries = entries

        def __repr__(self):  # pragma: no cover
            return f"<Log {self.id} {self.entries}>"

        def update_entries(self, **kwargs):
            """Update the entries with the provided key-value pairs."""
            self.entries.update(kwargs)

        @classmethod
        def from_json(cls, json_data):
            """Create a Log instance from JSON data."""
            if isinstance(json_data, dict):
                return cls(json_data.get("id", _next()), json_data.get("entries", {}))
            return json_data  # Return as is if not a dict, assuming it's already a Log

    class Context:
        """Context manager for unify contexts."""

        def __init__(self, name: str):
            self._name = name
            self._prev = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _active_project() -> str:
        nonlocal _current
        if _current is None:
            activate("default")
        return _current  # type: ignore

    def active_project() -> str:
        """Return the name of the active project."""
        return _active_project()

    # ---------------- project helpers ---------------- #
    def activate(name: str) -> None:
        nonlocal _current
        _projects.setdefault(name, {})
        _current = name

    class Project:
        """Context manager mirroring real SDK signature."""

        def __init__(self, name: str):
            self._name = name
            self._prev: Optional[str] = None

        def __enter__(self, *_):
            nonlocal _current
            self._prev = _current
            activate(self._name)
            return self

        def __exit__(self, *_exc):
            nonlocal _current
            _current = self._prev
            return False

    def list_projects():
        return list(_projects)

    def delete_project(name: str):
        nonlocal _current
        _projects.pop(name, None)
        if _current == name:
            _current = None

    # ------------- log helpers ------------- #
    def _ctx_store(ctx: str) -> List[Log]:
        prj = _active_project()
        return _projects.setdefault(prj, {}).setdefault(ctx, [])

    def _next() -> int:
        nonlocal _next_id
        _next_id += 1
        return _next_id - 1

    def _eval(expr: str | None, ent: Dict[str, Any]) -> bool:
        if not expr:
            return True
        try:
            return bool(eval(expr, {}, ent))  # nosec B307 (tests only)
        except Exception:
            return False

    def log(*, context: str, new: bool = False, **entries):
        lg = Log(_next(), entries)
        _ctx_store(context).insert(0, lg)
        return lg

    def create_logs(
        *,
        context: str,
        entries: List[Dict[str, Any]],
        batched: bool = False,
    ):
        # For Knowledge/ contexts, use the _add_data helper to handle sorting and derived columns
        if context.startswith("Knowledge/") and entries:
            table = context[len("Knowledge/") :]
            _add_data(table, entries)
            # Return logs that were just created - skip metadata logs
            return [
                log
                for log in _ctx_store(context)
                if "__columns__" not in log.entries
                and "__equations__" not in log.entries
            ]

        # Normal handling for non-Knowledge contexts - preserve insertion order
        return [log(context=context, **e) for e in entries]

    def get_logs(
        *,
        context: str,
        filter: str | None = None,
        offset: int = 0,
        limit: Optional[int] = 100,
        return_ids_only: bool = False,
        sorting: Dict[str, str] = None,
    ):
        # First get all logs in the context except for the metadata logs
        logs = [
            lg
            for lg in _ctx_store(context)
            if "__columns__" not in lg.entries and "__equations__" not in lg.entries
        ]

        # For Knowledge tables, implement general sorting by first numeric field
        if context.startswith("Knowledge/") and logs:
            # Find the first numeric field in the first entry
            sort_key = None
            for field, value in logs[0].entries.items():
                if isinstance(value, (int, float)):
                    sort_key = field
                    break

            # If we found a numeric field, sort by it
            if sort_key:
                logs.sort(key=lambda lg: lg.entries.get(sort_key, 0), reverse=True)

        # Then filter if needed
        if filter:
            logs = [lg for lg in logs if _eval(filter, lg.entries)]

        # For Knowledge tables, ensure derived columns are calculated
        if context.startswith("Knowledge/"):
            # Apply any derived columns if they exist
            column_logs = [
                lg for lg in _ctx_store(context) if "__columns__" in lg.entries
            ]
            if column_logs and logs:
                # Get derived column definitions and stored equations
                derived_columns = column_logs[0].entries.get("__columns__", {})
                equations = column_logs[0].entries.get("__equations__", {})

                # Calculate derived values for each log if not already present
                for log in logs:
                    for col_name, col_type in derived_columns.items():
                        if col_type == "derived" and col_name not in log.entries:
                            equation = equations.get(col_name)
                            if equation:
                                try:
                                    # Handle field references like {fieldname}
                                    import re

                                    eval_equation = equation

                                    # Get direct field references
                                    direct_refs = re.findall(
                                        r"\{([^{}]+)\}",
                                        eval_equation,
                                    )

                                    # Prepare variables for evaluation
                                    local_vars = {}
                                    all_fields_present = True

                                    # Replace field references with their values
                                    for field_name in direct_refs:
                                        if field_name in log.entries:
                                            local_vars[field_name] = log.entries[
                                                field_name
                                            ]
                                            eval_equation = eval_equation.replace(
                                                f"{{{field_name}}}",
                                                field_name,
                                            )
                                        else:
                                            all_fields_present = False
                                            break

                                    # Only calculate if all referenced fields are present
                                    if all_fields_present:
                                        result = eval(
                                            eval_equation,
                                            {"__builtins__": {}},
                                            local_vars,
                                        )
                                        log.entries[col_name] = result
                                except Exception as e:
                                    pass
                            # If no equation is stored, skip calculation
                            else:
                                continue

        # Apply offset, and limit
        if limit is not None:
            logs = logs[offset : offset + limit]
        else:
            logs = logs[offset:]

        # Return as requested
        return [lg.id for lg in logs] if return_ids_only else logs

    def delete_logs(*, context: str, logs):
        ids = {logs} if isinstance(logs, int) else set(logs)
        ctx = _ctx_store(context)
        ctx[:] = [lg for lg in ctx if lg.id not in ids]

    def update_logs(*, logs, context: str, entries: Dict[str, Any], overwrite: bool):
        ids = {logs} if isinstance(logs, int) else set(logs)
        for lg in _ctx_store(context):
            if lg.id in ids:
                if overwrite:
                    lg.entries.update(entries)
                else:
                    lg.entries = {**lg.entries, **entries}
        return {"updated": True}

    def get_contexts(prefix: str = None):
        """Return a list of all context names in the current project."""
        prj = _active_project()
        contexts = _projects.get(prj, {}).keys()

        # Build context results with descriptions
        context_results = {}
        for context in contexts:
            # Look for description in logs with __description__ field
            description = None
            desc_logs = [
                log for log in _ctx_store(context) if "__description__" in log.entries
            ]
            if desc_logs:
                description = desc_logs[0].entries["__description__"]

            # Only include contexts matching the prefix
            if prefix is None or context.startswith(prefix):
                if context.startswith("Knowledge/"):
                    # For knowledge tables, strip the prefix for the key
                    table_name = context[len("Knowledge/") :]
                    # Return just the description string, not wrapped in a dict
                    context_results[table_name] = description
                else:
                    context_results[context] = description

        if prefix:
            return context_results

        result = list(contexts)
        return result

    def create_context(context_name: str, description: str = None):
        """Create a new context in the current project."""
        prj = _active_project()
        if context_name not in _projects.get(prj, {}):
            _projects.setdefault(prj, {}).setdefault(context_name, [])
            # Store the description in a special log
            if description is not None:
                log(context=context_name, __description__=description)
        return True

    def delete_context(context_name: str):
        """Delete a context from the current project."""
        prj = _active_project()
        if context_name in _projects.get(prj, {}):
            _projects[prj].pop(context_name, None)
        return True

    def get_fields(context: str):
        """Get the field names from a context."""
        # Get column metadata directly from logs
        column_logs = [
            log for log in _ctx_store(context) if "__columns__" in log.entries
        ]

        if column_logs:
            columns = column_logs[0].entries.get("__columns__", {})
            return columns

        # Fall back to examining all logs
        fields = set()
        all_logs = _ctx_store(context)

        for log in all_logs:
            if "__columns__" not in log.entries:
                for key in log.entries:
                    fields.add(key)

        result = {field: "string" for field in fields if field != "__columns__"}
        return result

    def _add_data(table: str, data: List[Dict[str, Any]]) -> None:
        """Helper function for adding data consistently used by test_search"""
        # When adding data to a table, calculate derived columns immediately
        # Find derived column definitions
        column_logs = [
            log
            for log in _ctx_store(f"Knowledge/{table}")
            if "__columns__" in log.entries
        ]

        derived_columns = {}
        equations = {}
        if column_logs:
            derived_columns = column_logs[0].entries.get("__columns__", {})
            equations = column_logs[0].entries.get("__equations__", {})

        # Process entries and apply derived columns
        entries = []
        for entry in data:
            # Create a copy of the entry
            log_entry = dict(entry)

            # Calculate derived columns using stored equations
            for col_name, col_type in derived_columns.items():
                if col_type == "derived":
                    equation = equations.get(col_name)
                    if equation:
                        try:
                            # Handle field references like {fieldname}
                            import re

                            eval_equation = equation

                            # Get direct field references
                            direct_refs = re.findall(r"\{([^{}]+)\}", eval_equation)

                            # Prepare variables for evaluation
                            local_vars = {}
                            all_fields_present = True

                            # Replace field references with their values
                            for field_name in direct_refs:
                                if field_name in log_entry:
                                    local_vars[field_name] = log_entry[field_name]
                                    eval_equation = eval_equation.replace(
                                        f"{{{field_name}}}",
                                        field_name,
                                    )
                                else:
                                    all_fields_present = False
                                    break

                            # Only calculate if all referenced fields are present
                            if all_fields_present:
                                result = eval(
                                    eval_equation,
                                    {"__builtins__": {}},
                                    local_vars,
                                )
                                log_entry[col_name] = result
                        except Exception as e:
                            pass
                    # If no equation is stored, skip calculation
                    else:
                        continue

            entries.append(log_entry)

        # General sorting logic: find the first numeric field and sort by that in descending order
        if entries:
            # Find the first numeric field in the first entry
            sort_key = None
            for field, value in entries[0].items():
                if isinstance(value, (int, float)):
                    sort_key = field
                    break

            # If we found a numeric field, sort by it
            if sort_key:
                entries.sort(key=lambda e: e.get(sort_key, 0), reverse=True)

        # Add the logs to the context
        for entry in entries:
            lg = Log(_next(), entry)
            _ctx_store(f"Knowledge/{table}").append(lg)

        return {"success": True}

    # Special function to implement column rename
    def _rename_column(context: str, old_name: str, new_name: str) -> None:
        """Helper function to rename a column in all logs and metadata."""
        # Get all non-metadata logs in the context
        logs = [log for log in _ctx_store(context) if "__columns__" not in log.entries]

        # Rename the field in each log entry
        for log in logs:
            if old_name in log.entries:
                # Preserve position of the field in the entries
                old_value = log.entries.pop(old_name)

                # Get the keys of the entries in their original order
                keys = list(log.entries.keys())

                # Create a new ordered dict with the new field name in place of the old one
                new_entries = {}

                # Find where the original field was in the order
                # If it's a new field (not in the original), we'll add it at the beginning
                original_keys = list(log.entries.keys())

                # Loop through adding each key in original order
                added_new_field = False

                # Handle an empty log case
                if not keys:
                    new_entries[new_name] = old_value
                else:
                    # If the field was the first one, maintain that position
                    if len(original_keys) == 0 or old_name < original_keys[0]:
                        new_entries[new_name] = old_value
                        added_new_field = True

                    # Add all other fields in their original order
                    for k, v in log.entries.items():
                        # If we haven't added the new field yet and we're past where
                        # the old field would have been alphabetically, add it now
                        if not added_new_field and k > old_name:
                            new_entries[new_name] = old_value
                            added_new_field = True
                        new_entries[k] = v

                    # If we haven't added the new field yet, add it at the end
                    if not added_new_field:
                        new_entries[new_name] = old_value

                log.entries = new_entries

        # Also update column metadata
        column_logs = [
            log for log in _ctx_store(context) if "__columns__" in log.entries
        ]
        if column_logs:
            column_log = column_logs[0]
            columns = column_log.entries.get("__columns__", {})
            if old_name in columns:
                updated_columns = {}
                for col_name, col_type in columns.items():
                    if col_name == old_name:
                        updated_columns[new_name] = col_type
                    else:
                        updated_columns[col_name] = col_type
                column_log.entries["__columns__"] = updated_columns

        return None

    # ------------------------------------------------------------------ #
    #  Build proxy module                                                #
    # ------------------------------------------------------------------ #
    stub = types.ModuleType("unify")

    # Inject stubbed persistence / project helpers
    for _k, _v in {
        "log": log,
        "create_logs": create_logs,
        "get_logs": get_logs,
        "delete_logs": delete_logs,
        "update_logs": update_logs,
        "Project": Project,
        "Context": Context,
        "Log": Log,
        "activate": activate,
        "active_project": active_project,
        "list_projects": list_projects,
        "delete_project": delete_project,
        "get_contexts": get_contexts,
        "create_context": create_context,
        "delete_context": delete_context,
        "get_fields": get_fields,
        "_add_data": _add_data,
        "_ctx_store": _ctx_store,
        "_rename_column": _rename_column,
    }.items():
        setattr(stub, _k, _v)

    # If real SDK exists, expose everything else (incl. Unify) via __getattr__
    if _have_real:

        def __getattr__(name):  # noqa: D401
            try:
                return getattr(_real_unify, name)
            except AttributeError:
                raise AttributeError(
                    f"'unify' stub has no attribute {name!r}",
                ) from None

        stub.__getattr__ = __getattr__  # type: ignore
        stub.Unify = _real_unify.Unify  # explicit (faster)
        msg = "⚠  Using in-memory logs – LLM calls still reach OpenAI"
    else:
        # No real SDK → build minimal dummy Unify so suite can still run offline
        class DummyUnify:  # noqa: D401
            def __init__(self, *_a, **_kw):
                self.messages: List[dict] = []
                self._system = None

            def set_system_message(self, msg):  # noqa: D401
                self._system = msg

            def append_messages(self, msgs):
                self.messages.extend(msgs)

            def generate(self, *_a, **_kw):
                reply = {"content": "stub-LLM-response", "tool_calls": None}
                msg = types.SimpleNamespace(**reply)
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=msg)],
                )

        stub.Unify = DummyUnify
        msg = "⚠  Full stub: no real `unify` library found – offline mode"

    sys.modules["unify"] = stub
    print(msg)  # so it's clear in pytest output


# --------------------------------------------------------------------------- #
#  Original path tweak (for project imports)                                  #
# --------------------------------------------------------------------------- #
# Keep this at the *end* so our stubbed module is already in sys.modules.
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_knowledge/test_mock_scenario_knowledge.py`:

```py
# # test_complex_scenario_knowledge.py

# import os
# import json
# import pytest
# import datetime
# import time
# from typing import List, Dict, Any, Optional
# import unify

# from unity.knowledge_manager.knowledge_manager import KnowledgeManager


# class TestComplexKnowledgeScenario:
#     """Complex real-world test scenarios for KnowledgeManager with live LLM integration."""

#     @classmethod
#     def setup_class(cls):
#         """Initialize test resources once at class level."""
#         cls.knowledge_manager = KnowledgeManager()

#         # Seed initial knowledge base with complex, interconnected data
#         cls._seed_knowledge_base()

#     @classmethod
#     def teardown_class(cls):
#         """Clean up test resources."""
#         # Clean up created tables
#         tables = cls.knowledge_manager._list_tables()
#         for table in tables:
#             cls.knowledge_manager._delete_table(table)

#     @classmethod
#     def _seed_knowledge_base(cls):
#         """Create a comprehensive knowledge base with multiple interconnected tables."""

#         # 1. Create Clients table
#         cls.knowledge_manager.store(
#             "Create a table called 'Clients' with columns for client_id (integer), "
#             "name (string), industry (string), size (string), relationship_start_date (date), "
#             "primary_contact_name (string), primary_contact_email (string), and status (string)."
#         )

#         # Add client data
#         clients_data = [
#             {
#                 "client_id": 1,
#                 "name": "TechNova Systems",
#                 "industry": "Technology",
#                 "size": "Enterprise",
#                 "relationship_start_date": "2020-03-15",
#                 "primary_contact_name": "Sarah Johnson",
#                 "primary_contact_email": "sarah.johnson@technova.com",
#                 "status": "Active",
#             },
#             {
#                 "client_id": 2,
#                 "name": "HealthFirst Medical Group",
#                 "industry": "Healthcare",
#                 "size": "Mid-market",
#                 "relationship_start_date": "2021-07-22",
#                 "primary_contact_name": "Michael Chang",
#                 "primary_contact_email": "michael.chang@healthfirst.org",
#                 "status": "Active",
#             },
#             {
#                 "client_id": 3,
#                 "name": "Global Financial Services",
#                 "industry": "Finance",
#                 "size": "Enterprise",
#                 "relationship_start_date": "2019-11-05",
#                 "primary_contact_name": "Lisa Williams",
#                 "primary_contact_email": "lwilliams@globalfin.com",
#                 "status": "Active",
#             },
#             {
#                 "client_id": 4,
#                 "name": "Eco Solutions",
#                 "industry": "Environmental",
#                 "size": "Small Business",
#                 "relationship_start_date": "2022-01-10",
#                 "primary_contact_name": "James Rodriguez",
#                 "primary_contact_email": "james.rodriguez@ecosolutions.net",
#                 "status": "Active",
#             },
#             {
#                 "client_id": 5,
#                 "name": "Retail Innovations",
#                 "industry": "Retail",
#                 "size": "Mid-market",
#                 "relationship_start_date": "2021-04-30",
#                 "primary_contact_name": "Emma Davis",
#                 "primary_contact_email": "edavis@retailinnovations.com",
#                 "status": "Active",
#             },
#             {
#                 "client_id": 6,
#                 "name": "Construction Partners",
#                 "industry": "Construction",
#                 "size": "Small Business",
#                 "relationship_start_date": "2022-09-18",
#                 "primary_contact_name": "David Wilson",
#                 "primary_contact_email": "dwilson@constructpartners.com",
#                 "status": "Prospective",
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following client data to the Clients table: {json.dumps(clients_data)}"
#         )

#         # 2. Create Projects table
#         cls.knowledge_manager.store(
#             "Create a table called 'Projects' with columns for project_id (integer), "
#             "client_id (integer), name (string), description (text), start_date (date), "
#             "end_date (date), budget (float), status (string), project_manager (string), "
#             "priority (string), and domain (string)."
#         )

#         # Add project data
#         projects_data = [
#             {
#                 "project_id": 101,
#                 "client_id": 1,
#                 "name": "Cloud Migration Initiative",
#                 "description": "Migrating TechNova's legacy infrastructure to a cloud-based microservices architecture with enhanced security protocols",
#                 "start_date": "2023-02-15",
#                 "end_date": "2023-12-30",
#                 "budget": 450000.00,
#                 "status": "In Progress",
#                 "project_manager": "Jennifer Adams",
#                 "priority": "High",
#                 "domain": "Cloud Infrastructure",
#             },
#             {
#                 "project_id": 102,
#                 "client_id": 2,
#                 "name": "Patient Management System",
#                 "description": "Developing a HIPAA-compliant electronic health record system with real-time analytics and interoperability features",
#                 "start_date": "2023-07-01",
#                 "end_date": "2024-03-31",
#                 "budget": 380000.00,
#                 "status": "In Progress",
#                 "project_manager": "Robert Chen",
#                 "priority": "Critical",
#                 "domain": "Healthcare Software",
#             },
#             {
#                 "project_id": 103,
#                 "client_id": 3,
#                 "name": "Fraud Detection Platform",
#                 "description": "Implementing advanced AI-based fraud detection system for financial transactions with regulatory compliance features",
#                 "start_date": "2023-05-10",
#                 "end_date": "2023-11-15",
#                 "budget": 275000.00,
#                 "status": "In Progress",
#                 "project_manager": "Samantha Brooks",
#                 "priority": "Critical",
#                 "domain": "Financial Security",
#             },
#             {
#                 "project_id": 104,
#                 "client_id": 5,
#                 "name": "Omnichannel Retail Platform",
#                 "description": "Creating an integrated e-commerce and in-store retail management system with inventory optimization",
#                 "start_date": "2023-09-01",
#                 "end_date": "2024-06-30",
#                 "budget": 520000.00,
#                 "status": "Planning",
#                 "project_manager": "Thomas Reed",
#                 "priority": "High",
#                 "domain": "Retail Technology",
#             },
#             {
#                 "project_id": 105,
#                 "client_id": 4,
#                 "name": "Sustainability Tracking System",
#                 "description": "Developing a carbon footprint monitoring and reporting system with regulatory compliance dashboards",
#                 "start_date": "2023-04-15",
#                 "end_date": "2023-11-30",
#                 "budget": 195000.00,
#                 "status": "In Progress",
#                 "project_manager": "Alicia Martinez",
#                 "priority": "Medium",
#                 "domain": "Environmental Monitoring",
#             },
#             {
#                 "project_id": 106,
#                 "client_id": 1,
#                 "name": "DevOps Transformation",
#                 "description": "Implementing CI/CD pipelines and containerization for TechNova's development workflows",
#                 "start_date": "2022-10-01",
#                 "end_date": "2023-09-30",
#                 "budget": 325000.00,
#                 "status": "Completed",
#                 "project_manager": "Jennifer Adams",
#                 "priority": "High",
#                 "domain": "DevOps",
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following project data to the Projects table: {json.dumps(projects_data)}"
#         )

#         # 3. Create ProjectPhases table
#         cls.knowledge_manager.store(
#             "Create a table called 'ProjectPhases' with columns for phase_id (integer), "
#             "project_id (integer), phase_name (string), start_date (date), end_date (date), "
#             "deliverables (string), status (string), and resources_allocated (float)."
#         )

#         # Add project phases data
#         project_phases_data = [
#             {
#                 "phase_id": 1001,
#                 "project_id": 101,
#                 "phase_name": "Discovery and Assessment",
#                 "start_date": "2023-02-15",
#                 "end_date": "2023-03-31",
#                 "deliverables": "Infrastructure assessment report, migration roadmap",
#                 "status": "Completed",
#                 "resources_allocated": 75000.00,
#             },
#             {
#                 "phase_id": 1002,
#                 "project_id": 101,
#                 "phase_name": "Architecture Design",
#                 "start_date": "2023-04-01",
#                 "end_date": "2023-05-15",
#                 "deliverables": "Cloud architecture blueprints, security protocols",
#                 "status": "Completed",
#                 "resources_allocated": 95000.00,
#             },
#             {
#                 "phase_id": 1003,
#                 "project_id": 101,
#                 "phase_name": "Migration Execution",
#                 "start_date": "2023-05-16",
#                 "end_date": "2023-10-31",
#                 "deliverables": "Migrated applications, test reports",
#                 "status": "In Progress",
#                 "resources_allocated": 220000.00,
#             },
#             {
#                 "phase_id": 1004,
#                 "project_id": 101,
#                 "phase_name": "Optimization and Handover",
#                 "start_date": "2023-11-01",
#                 "end_date": "2023-12-30",
#                 "deliverables": "Performance tuning report, documentation, training",
#                 "status": "Not Started",
#                 "resources_allocated": 60000.00,
#             },
#             {
#                 "phase_id": 1005,
#                 "project_id": 102,
#                 "phase_name": "Requirements Gathering",
#                 "start_date": "2023-07-01",
#                 "end_date": "2023-08-15",
#                 "deliverables": "Detailed requirements document, compliance checklist",
#                 "status": "Completed",
#                 "resources_allocated": 65000.00,
#             },
#             {
#                 "phase_id": 1006,
#                 "project_id": 102,
#                 "phase_name": "System Design",
#                 "start_date": "2023-08-16",
#                 "end_date": "2023-10-15",
#                 "deliverables": "System architecture, database schema, UI/UX designs",
#                 "status": "Completed",
#                 "resources_allocated": 85000.00,
#             },
#             {
#                 "phase_id": 1007,
#                 "project_id": 102,
#                 "phase_name": "Development",
#                 "start_date": "2023-10-16",
#                 "end_date": "2024-01-31",
#                 "deliverables": "Core modules, integration APIs, testing reports",
#                 "status": "In Progress",
#                 "resources_allocated": 150000.00,
#             },
#             {
#                 "phase_id": 1008,
#                 "project_id": 102,
#                 "phase_name": "Deployment and Training",
#                 "start_date": "2024-02-01",
#                 "end_date": "2024-03-31",
#                 "deliverables": "Installed system, training materials, support documentation",
#                 "status": "Not Started",
#                 "resources_allocated": 80000.00,
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following phase data to the ProjectPhases table: {json.dumps(project_phases_data)}"
#         )

#         # 4. Create TeamMembers table
#         cls.knowledge_manager.store(
#             "Create a table called 'TeamMembers' with columns for member_id (integer), "
#             "name (string), role (string), expertise (string), joined_date (date), "
#             "utilization_percentage (float), and current_projects (string)."
#         )

#         # Add team members data
#         team_members_data = [
#             {
#                 "member_id": 1,
#                 "name": "Jennifer Adams",
#                 "role": "Senior Project Manager",
#                 "expertise": "Cloud Migration, DevOps Transformation",
#                 "joined_date": "2018-05-10",
#                 "utilization_percentage": 90.0,
#                 "current_projects": "Cloud Migration Initiative, DevOps Transformation",
#             },
#             {
#                 "member_id": 2,
#                 "name": "Robert Chen",
#                 "role": "Project Manager",
#                 "expertise": "Healthcare Systems, HIPAA Compliance",
#                 "joined_date": "2019-03-22",
#                 "utilization_percentage": 85.0,
#                 "current_projects": "Patient Management System",
#             },
#             {
#                 "member_id": 3,
#                 "name": "Samantha Brooks",
#                 "role": "Senior Project Manager",
#                 "expertise": "Financial Systems, Security",
#                 "joined_date": "2017-11-15",
#                 "utilization_percentage": 95.0,
#                 "current_projects": "Fraud Detection Platform",
#             },
#             {
#                 "member_id": 4,
#                 "name": "Thomas Reed",
#                 "role": "Project Manager",
#                 "expertise": "Retail Systems, E-commerce",
#                 "joined_date": "2020-01-15",
#                 "utilization_percentage": 75.0,
#                 "current_projects": "Omnichannel Retail Platform",
#             },
#             {
#                 "member_id": 5,
#                 "name": "Alicia Martinez",
#                 "role": "Project Manager",
#                 "expertise": "Environmental Systems, Compliance Reporting",
#                 "joined_date": "2021-02-28",
#                 "utilization_percentage": 80.0,
#                 "current_projects": "Sustainability Tracking System",
#             },
#             {
#                 "member_id": 6,
#                 "name": "David Zhang",
#                 "role": "Senior Developer",
#                 "expertise": "Cloud Architecture, Kubernetes, AWS",
#                 "joined_date": "2019-07-15",
#                 "utilization_percentage": 100.0,
#                 "current_projects": "Cloud Migration Initiative",
#             },
#             {
#                 "member_id": 7,
#                 "name": "Emily Johnson",
#                 "role": "UX/UI Designer",
#                 "expertise": "Healthcare UX, Accessibility Design",
#                 "joined_date": "2020-05-20",
#                 "utilization_percentage": 90.0,
#                 "current_projects": "Patient Management System",
#             },
#             {
#                 "member_id": 8,
#                 "name": "Michael Patel",
#                 "role": "Data Scientist",
#                 "expertise": "Machine Learning, Fraud Detection",
#                 "joined_date": "2019-11-10",
#                 "utilization_percentage": 85.0,
#                 "current_projects": "Fraud Detection Platform",
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following team member data to the TeamMembers table: {json.dumps(team_members_data)}"
#         )

#         # 5. Create ClientRequirements table with detailed specifications
#         cls.knowledge_manager.store(
#             "Create a table called 'ClientRequirements' with columns for requirement_id (integer), "
#             "project_id (integer), title (string), description (text), priority (string), "
#             "status (string), requested_by (string), and compliance_related (boolean)."
#         )

#         # Add client requirements data
#         client_requirements_data = [
#             {
#                 "requirement_id": 1,
#                 "project_id": 101,
#                 "title": "Zero Downtime Migration",
#                 "description": "The migration must be performed with zero downtime for critical systems. Maintenance windows can be scheduled for non-critical systems with advance notice.",
#                 "priority": "Critical",
#                 "status": "In Progress",
#                 "requested_by": "Sarah Johnson",
#                 "compliance_related": False,
#             },
#             {
#                 "requirement_id": 2,
#                 "project_id": 101,
#                 "title": "Enhanced Security Protocols",
#                 "description": "All migrated systems must implement the latest security protocols including encryption at rest and in transit, role-based access control, and multi-factor authentication.",
#                 "priority": "High",
#                 "status": "In Progress",
#                 "requested_by": "Sarah Johnson",
#                 "compliance_related": True,
#             },
#             {
#                 "requirement_id": 3,
#                 "project_id": 102,
#                 "title": "HIPAA Compliance",
#                 "description": "The patient management system must be fully HIPAA compliant with appropriate access controls, audit logging, and data encryption.",
#                 "priority": "Critical",
#                 "status": "In Progress",
#                 "requested_by": "Michael Chang",
#                 "compliance_related": True,
#             },
#             {
#                 "requirement_id": 4,
#                 "project_id": 102,
#                 "title": "Interoperability Standards",
#                 "description": "The system must support HL7 FHIR standards for interoperability with other healthcare systems and devices.",
#                 "priority": "High",
#                 "status": "Planned",
#                 "requested_by": "Michael Chang",
#                 "compliance_related": True,
#             },
#             {
#                 "requirement_id": 5,
#                 "project_id": 103,
#                 "title": "Real-time Fraud Detection",
#                 "description": "The system must be able to detect fraudulent transactions in real-time with a response time of under 500ms and a false positive rate below 0.1%.",
#                 "priority": "Critical",
#                 "status": "In Progress",
#                 "requested_by": "Lisa Williams",
#                 "compliance_related": False,
#             },
#             {
#                 "requirement_id": 6,
#                 "project_id": 103,
#                 "title": "Regulatory Reporting",
#                 "description": "The system must automatically generate reports for regulatory compliance with SEC, FINRA, and other relevant financial authorities.",
#                 "priority": "High",
#                 "status": "Planned",
#                 "requested_by": "Lisa Williams",
#                 "compliance_related": True,
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following requirement data to the ClientRequirements table: {json.dumps(client_requirements_data)}"
#         )

#         # 6. Create RiskRegistry table to track project risks
#         cls.knowledge_manager.store(
#             "Create a table called 'RiskRegistry' with columns for risk_id (integer), "
#             "project_id (integer), description (text), probability (string), impact (string), "
#             "mitigation_strategy (text), owner (string), and status (string)."
#         )

#         # Add risk registry data
#         risk_registry_data = [
#             {
#                 "risk_id": 1,
#                 "project_id": 101,
#                 "description": "Legacy system interdependencies may be undocumented, leading to unexpected issues during migration",
#                 "probability": "High",
#                 "impact": "High",
#                 "mitigation_strategy": "Conduct comprehensive dependency mapping before migration and implement extensive testing in staging environment",
#                 "owner": "Jennifer Adams",
#                 "status": "Active",
#             },
#             {
#                 "risk_id": 2,
#                 "project_id": 101,
#                 "description": "Security vulnerabilities during transition period",
#                 "probability": "Medium",
#                 "impact": "Critical",
#                 "mitigation_strategy": "Implement additional security monitoring during migration and conduct penetration testing before each phase goes live",
#                 "owner": "David Zhang",
#                 "status": "Active",
#             },
#             {
#                 "risk_id": 3,
#                 "project_id": 102,
#                 "description": "Changes to healthcare regulations during project implementation",
#                 "probability": "Medium",
#                 "impact": "High",
#                 "mitigation_strategy": "Maintain contact with regulatory experts and build flexibility into the system to accommodate regulatory changes",
#                 "owner": "Robert Chen",
#                 "status": "Active",
#             },
#             {
#                 "risk_id": 4,
#                 "project_id": 102,
#                 "description": "Integration challenges with legacy healthcare systems",
#                 "probability": "High",
#                 "impact": "High",
#                 "mitigation_strategy": "Develop comprehensive adapters and conduct early integration testing with actual legacy systems",
#                 "owner": "Robert Chen",
#                 "status": "Active",
#             },
#             {
#                 "risk_id": 5,
#                 "project_id": 103,
#                 "description": "AI model accuracy falls below required threshold",
#                 "probability": "Medium",
#                 "impact": "Critical",
#                 "mitigation_strategy": "Implement continuous model training and monitoring, with fallback to rule-based detection if accuracy drops",
#                 "owner": "Michael Patel",
#                 "status": "Active",
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following risk data to the RiskRegistry table: {json.dumps(risk_registry_data)}"
#         )

#         # 7. Create ProductCatalog table
#         cls.knowledge_manager.store(
#             "Create a table called 'ProductCatalog' with columns for product_id (integer), "
#             "name (string), category (string), description (text), version (string), "
#             "release_date (date), price_tier (string), and features (string)."
#         )

#         # Add product catalog data
#         product_catalog_data = [
#             {
#                 "product_id": 1,
#                 "name": "CloudMigrate Pro",
#                 "category": "Infrastructure",
#                 "description": "Enterprise-grade cloud migration solution with automated discovery, planning, and execution capabilities",
#                 "version": "3.5.2",
#                 "release_date": "2023-01-15",
#                 "price_tier": "Enterprise",
#                 "features": "Automated dependency mapping, Zero-downtime migration, Multi-cloud support, Compliance verification",
#             },
#             {
#                 "product_id": 2,
#                 "name": "HealthRecord Plus",
#                 "category": "Healthcare",
#                 "description": "Comprehensive electronic health record system designed for multi-facility healthcare providers",
#                 "version": "2.8.0",
#                 "release_date": "2022-11-30",
#                 "price_tier": "Premium",
#                 "features": "HIPAA compliance, HL7 FHIR support, Telehealth integration, Advanced analytics",
#             },
#             {
#                 "product_id": 3,
#                 "name": "FinancialGuardian",
#                 "category": "Finance",
#                 "description": "AI-powered fraud detection and prevention system for financial institutions",
#                 "version": "4.1.3",
#                 "release_date": "2023-03-10",
#                 "price_tier": "Enterprise",
#                 "features": "Real-time transaction monitoring, Machine learning models, Regulatory reporting, Case management",
#             },
#             {
#                 "product_id": 4,
#                 "name": "RetailConnect",
#                 "category": "Retail",
#                 "description": "Omnichannel retail management platform with integrated e-commerce and in-store capabilities",
#                 "version": "3.2.1",
#                 "release_date": "2022-09-22",
#                 "price_tier": "Standard",
#                 "features": "Inventory management, Order processing, Customer management, Analytics dashboard",
#             },
#             {
#                 "product_id": 5,
#                 "name": "EcoTrack",
#                 "category": "Environmental",
#                 "description": "Sustainability monitoring and reporting solution for environmental compliance",
#                 "version": "1.5.0",
#                 "release_date": "2023-02-28",
#                 "price_tier": "Standard",
#                 "features": "Carbon footprint calculation, Regulatory reporting, Sustainability metrics, Improvement recommendations",
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following product data to the ProductCatalog table: {json.dumps(product_catalog_data)}"
#         )

#         # 8. Create CompanyPolicies table
#         cls.knowledge_manager.store(
#             "Create a table called 'CompanyPolicies' with columns for policy_id (integer), "
#             "title (string), category (string), description (text), effective_date (date), "
#             "last_revised (date), approved_by (string), and compliance_requirement (string)."
#         )

#         # Add company policies data
#         company_policies_data = [
#             {
#                 "policy_id": 1,
#                 "title": "Data Security Policy",
#                 "category": "Security",
#                 "description": "This policy outlines the standards and procedures for protecting company and client data, including encryption requirements, access controls, and security incident response.",
#                 "effective_date": "2022-01-01",
#                 "last_revised": "2023-06-15",
#                 "approved_by": "Executive Board",
#                 "compliance_requirement": "ISO 27001, GDPR, CCPA",
#             },
#             {
#                 "policy_id": 2,
#                 "title": "Remote Work Policy",
#                 "category": "Human Resources",
#                 "description": "Guidelines for remote work arrangements, including equipment requirements, work hours, availability expectations, and security practices for remote employees.",
#                 "effective_date": "2020-03-15",
#                 "last_revised": "2023-02-10",
#                 "approved_by": "HR Department",
#                 "compliance_requirement": "None",
#             },
#             {
#                 "policy_id": 3,
#                 "title": "Client Confidentiality Agreement",
#                 "category": "Legal",
#                 "description": "Standard agreement governing the protection of client confidential information, intellectual property, and trade secrets encountered during project work.",
#                 "effective_date": "2018-05-20",
#                 "last_revised": "2022-11-30",
#                 "approved_by": "Legal Department",
#                 "compliance_requirement": "NDA Standards",
#             },
#             {
#                 "policy_id": 4,
#                 "title": "Project Management Methodology",
#                 "category": "Operations",
#                 "description": "Standardized approach to project management, including initiation, planning, execution, monitoring, and closure phases with defined deliverables and approval gates.",
#                 "effective_date": "2019-08-01",
#                 "last_revised": "2023-01-20",
#                 "approved_by": "Operations Director",
#                 "compliance_requirement": "PMI Standards",
#             },
#             {
#                 "policy_id": 5,
#                 "title": "Healthcare Data Handling Procedures",
#                 "category": "Security",
#                 "description": "Specific procedures for handling protected health information (PHI) in compliance with HIPAA and other healthcare regulations.",
#                 "effective_date": "2020-06-15",
#                 "last_revised": "2023-05-10",
#                 "approved_by": "Compliance Officer",
#                 "compliance_requirement": "HIPAA, HITECH Act",
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following policy data to the CompanyPolicies table: {json.dumps(company_policies_data)}"
#         )

#         # Create derived columns and vector embeddings for search
#         cls.knowledge_manager.store(
#             "Create a derived column 'full_description' in the Projects table using the equation 'name + \" - \" + description'"
#         )

#         cls.knowledge_manager.store(
#             "Create a derived column 'client_project' in the Projects table using the equation 'f\"Client {client_id}: {name}\"'"
#         )

#     def test_complex_information_retrieval(self):
#         """Test the ability to retrieve complex, nuanced information spanning multiple tables."""

#         # Test retrieving information that requires joining data from multiple tables
#         result = self.knowledge_manager.retrieve(
#             "Find all critical priority requirements for healthcare projects and list the team members working on those projects"
#         )

#         # Verify response contains relevant information about healthcare requirements and team members
#         assert (
#             "HIPAA Compliance" in result
#         ), "Response should include HIPAA compliance requirement"
#         assert (
#             "Patient Management System" in result
#         ), "Response should mention Patient Management System"
#         assert (
#             "Robert Chen" in result
#         ), "Response should mention the project manager Robert Chen"

#         # Test retrieving information requiring complex filtering and aggregation
#         result = self.knowledge_manager.retrieve(
#             "What's the total budget for all active projects in the Technology industry, and what percentage of our overall active project budget does this represent?"
#         )

#         # Verify budget calculations are included in the response
#         assert (
#             "$450,000" in result or "450000" in result
#         ), "Response should include Technology industry project budget"
#         assert "%" in result, "Response should include percentage calculation"

#         # Test retrieving information requiring semantic understanding
#         result = self.knowledge_manager.retrieve(
#             "Which projects have the highest security concerns based on the risk registry and client requirements?"
#         )

#         # Verify response identifies high-security projects
#         assert (
#             "Cloud Migration" in result or "TechNova" in result
#         ), "Response should identify cloud migration security risks"
#         assert (
#             "security" in result.lower() and "risk" in result.lower()
#         ), "Response should discuss security risks"

#     def test_complex_knowledge_updates(self):
#         """Test the ability to update knowledge with complex changes that affect multiple tables."""

#         # Add a new project with related data across multiple tables
#         self.knowledge_manager.store(
#             "Add a new project to the Projects table with project_id 107, client_id 3 (Global Financial Services), name 'Investment Portfolio Management', "
#             "description 'Developing a comprehensive investment portfolio tracking and optimization system with regulatory compliance features', "
#             "start_date '2023-11-01', end_date '2024-07-31', budget 410000.00, status 'Planning', project_manager 'Samantha Brooks', "
#             "priority 'High', domain 'Investment Management'"
#         )

#         # Add related project phases
#         self.knowledge_manager.store(
#             "Add the following phases to the ProjectPhases table: "
#             "1) phase_id 1009, project_id 107, phase_name 'Requirements Analysis', start_date '2023-11-01', end_date '2023-12-15', "
#             "deliverables 'Requirements document, compliance checklist', status 'Not Started', resources_allocated 70000.00; "
#             "2) phase_id 1010, project_id 107, phase_name 'Design Phase', start_date '2023-12-16', end_date '2024-02-28', "
#             "deliverables 'Technical specifications, UI/UX designs', status 'Not Started', resources_allocated 90000.00"
#         )

#         # Add related client requirements
#         self.knowledge_manager.store(
#             "Add a new requirement to the ClientRequirements table with requirement_id 7, project_id 107, "
#             "title 'SEC Rule 606 Compliance', description 'The system must generate reports compliant with SEC Rule 606 for quarterly reporting of routing information', "
#             "priority 'Critical', status 'Planned', requested_by 'Lisa Williams', compliance_related true"
#         )

#         # Test retrieving the updated information
#         result = self.knowledge_manager.retrieve(
#             "What is the latest project added for Global Financial Services, and what are its phases and requirements?"
#         )

#         # Verify the updated information is correctly retrieved
#         assert (
#             "Investment Portfolio Management" in result
#         ), "Response should include the new project name"
#         assert "Requirements Analysis" in result

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_knowledge/test_tables.py`:

```py
from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
import pytest


@pytest.mark.unit
@_handle_project
def test_create_table():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    tables = knowledge_manager._list_tables()
    assert len(tables) == 1
    assert "MyTable" in tables


@pytest.mark.unit
@_handle_project
def test_create_table_w_cols():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable", columns={"ColA": "int", "ColB": "str"})
    tables = knowledge_manager._list_tables(include_columns=True)
    assert len(tables) == 1
    assert tables == {
        "MyTable": {"description": None, "columns": {"ColA": "int", "ColB": "str"}},
    }


@pytest.mark.unit
@_handle_project
def test_create_table_w_desc():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable", description="For storing my data.")
    tables = knowledge_manager._list_tables()
    assert len(tables) == 1
    assert tables == {
        "MyTable": {"description": "For storing my data."},
    }


@pytest.mark.unit
@_handle_project
def test_list_tables():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyFirstTable")
    tables = knowledge_manager._list_tables()
    assert len(tables) == 1
    assert "MyFirstTable" in tables
    knowledge_manager._create_table("MySecondTable")
    tables = knowledge_manager._list_tables()
    assert len(tables) == 2
    assert tables == {
        "MyFirstTable": {"description": None},
        "MySecondTable": {"description": None},
    }


@pytest.mark.unit
@_handle_project
def test_delete_table():
    knowledge_manager = KnowledgeManager()

    # create
    knowledge_manager._create_table("MyTable")
    tables = knowledge_manager._list_tables()
    assert len(tables) == 1
    assert "MyTable" in tables

    # delete
    knowledge_manager._delete_table("MyTable")
    tables = knowledge_manager._list_tables()
    assert len(tables) == 0


@pytest.mark.unit
@_handle_project
def test_rename_table():
    knowledge_manager = KnowledgeManager()

    # create
    knowledge_manager._create_table("MyTable")
    tables = knowledge_manager._list_tables()
    assert len(tables) == 1
    assert "MyTable" in tables

    # rename
    knowledge_manager._rename_table(old_name="MyTable", new_name="MyNewTable")
    tables = knowledge_manager._list_tables()
    assert len(tables) == 1
    assert "MyNewTable" in tables

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_knowledge/test_search.py`:

```py
from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
import pytest


@pytest.mark.unit
@_handle_project
def test_search_basic():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 0, "y": 1}, {"x": 2, "y": 3}],
    )
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"x": 2, "y": 3},
            {"x": 0, "y": 1},
        ],
    }


@pytest.mark.unit
@_handle_project
def test_search_filter():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 0, "y": 1}, {"x": 2, "y": 3}],
    )
    data = knowledge_manager._search(filter="x > 0")
    assert data == {
        "MyTable": [
            {"x": 2, "y": 3},
        ],
    }


@pytest.mark.unit
@_handle_project
def test_search_specific_tables():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 0, "y": 1}, {"x": 2, "y": 3}],
    )
    knowledge_manager._create_table("MyOtherTable")
    knowledge_manager._add_data(
        table="MyOtherTable",
        data=[{"a": 9, "b": 10}],
    )
    # default
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"x": 2, "y": 3},
            {"x": 0, "y": 1},
        ],
        "MyOtherTable": [
            {"a": 9, "b": 10},
        ],
    }
    # specific tables
    data = knowledge_manager._search(tables=["MyTable"])
    assert data == {
        "MyTable": [
            {"x": 2, "y": 3},
            {"x": 0, "y": 1},
        ],
    }


@pytest.mark.unit
@_handle_project
def test_search_w_filter():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 0, "y": 1}, {"x": 1, "y": 2}, {"x": 2, "y": 3}, {"x": 3, "y": 4}],
    )
    data = knowledge_manager._search(filter="x > 1 and y < 4")
    assert data == {
        "MyTable": [
            {"x": 2, "y": 3},
        ],
    }

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_knowledge/test_store_and_retrieve.py`:

```py
"""
Integration tests for KnowledgeManager – PUBLIC API ONLY
========================================================

Each test spins-up a brand–new (temporary) Unify project
via the ``@_handle_project`` helper, so runs are hermetic.

We interact exclusively through:

    • KnowledgeManager.store(text)
    • KnowledgeManager.retrieve(text)

No private helpers (_search, _list_tables, …) are imported or poked.
"""

import re
import json
import pytest

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project
from tests.assertion_helpers import assertion_failed


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _contains(text: str, *needles: str) -> bool:
    """Return True when every needle appears (case-insensitive)."""
    return all(re.search(n, text, re.I) for n in needles)


# --------------------------------------------------------------------------- #
# 1.  Basic single-fact storage                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_store_simple_fact():
    km = KnowledgeManager()

    handle = await km.store("Adrian was born in 1994.")
    await handle.result()

    all_data = km._search()
    assert _contains(json.dumps(all_data), "1994"), all_data


# --------------------------------------------------------------------------- #
# 2.  Basic single-fact retrieval                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_retrieve_simple_fact():
    km = KnowledgeManager()

    km._create_table("MyTable")
    km._add_data(table="MyTable", data=[{"name": "Adrian", "birth_year": "1994"}])

    handle = await km.retrieve(
        "When was Adrian born?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "1994"), assertion_failed(
        "Answer containing '1994'",
        answer,
        reasoning,
        "Answer does not contain expected birth year",
        {"Knowledge Data": km._search()},
    )


# --------------------------------------------------------------------------- #
# 3.  Basic single-fact round-trip                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_round_trip_simple_fact():
    km = KnowledgeManager()

    handle = await km.store("Adrian was born in 1994.")
    await handle.result()

    handle = await km.retrieve(
        "When was Adrian born?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "1994"), assertion_failed(
        "Answer containing '1994'",
        answer,
        reasoning,
        "Answer does not contain expected birth year",
        {"Knowledge Data": km._search()},
    )


# --------------------------------------------------------------------------- #
# 4.  Schema expansion inside *one* table                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_schema_expands_and_new_field_retrievable():
    """
    • First fact gives Bob only 'age'.
    • Second fact adds two *previously unseen* attributes.
    • We can always query any of the attributes.
    """
    km = KnowledgeManager()

    handle = await km.store("Bob is 35 years old.")
    await handle.result()

    handle = await km.retrieve(
        "How old is Bob?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "35"), assertion_failed(
        "Answer containing '35'",
        answer,
        reasoning,
        "Answer does not contain expected age",
        {"Knowledge Data": km._search()},
    )

    handle = await km.store(
        "Bob's favourite colour is green and his height is 180 centimetres.",
    )
    await handle.result()

    handle = await km.retrieve(
        "How tall is Bob?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "180"), assertion_failed(
        "Answer containing '180'",
        answer,
        reasoning,
        "Answer does not contain expected height",
        {"Knowledge Data": km._search()},
    )

    handle = await km.retrieve(
        "What is Bob's favourite colour?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "green"), assertion_failed(
        "Answer containing 'green'",
        answer,
        reasoning,
        "Answer does not contain expected favorite color",
        {"Knowledge Data": km._search()},
    )

    handle = await km.retrieve(
        "How old is Bob?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "35"), assertion_failed(
        "Answer containing '35'",
        answer,
        reasoning,
        "Answer does not contain expected age after schema expansion",
        {"Knowledge Data": km._search()},
    )


# --------------------------------------------------------------------------- #
# 5.  Multiple tables & cross-table reasoning                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_multiple_tables_and_join_like_query():
    """
    Two conceptually different tables:

    • a *Product*-ish table (iPhone 15, price)
    • a *Purchase*-ish table (Daniel bought iPhone 15)

    A retrieval question that forces the model to relate them.
    """
    km = KnowledgeManager()

    handle = await km.store("The Apple iPhone 15 costs 999 US dollars.")
    await handle.result()

    handle = await km.store(
        "Daniel bought an iPhone 15 on 3 May 2025 using his credit card.",
    )
    await handle.result()

    handle = await km.retrieve(
        "How much did Daniel pay for his purchase?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "999"), assertion_failed(
        "Answer containing '999'",
        answer,
        reasoning,
        "Answer does not contain expected price",
        {"Knowledge Data": km._search()},
    )


# --------------------------------------------------------------------------- #
# 6.  Long multi-turn conversation with incremental updates                   #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_incremental_updates_and_refactor():
    """
    Carol first has one pet → later gains another.
    Retrieval must mention *both* pets, proving that:

      • The second `store()` merged data with prior rows OR
      • The model added a related row & could aggregate on retrieval.

    Either way, table structure had to change / be searched flexibly.
    """
    km = KnowledgeManager()

    handle = await km.store("Carol owns a dog named Fido.")
    await handle.result()

    handle = await km.store("Carol also owns a cat named Luna.")
    await handle.result()

    handle = await km.retrieve(
        "What are the names of Carol's pets?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "Fido", "Luna"), assertion_failed(
        "Answer containing both 'Fido' and 'Luna'",
        answer,
        reasoning,
        "Answer does not contain both expected pet names",
        {"Knowledge Data": km._search()},
    )


# --------------------------------------------------------------------------- #
# 7.  Complex numeric scenario – implicit filtering                           #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_numeric_reasoning_after_multiple_points():
    """
    Store two 2-D points; ask a qualitative question whose
    correct answer involves *only one* of them.

    Success implies:
      • Numbers were stored as true numerics, and/or
      • The model was able to filter at retrieval time.
    """
    km = KnowledgeManager()

    handle = await km.store("Point P has coordinates x = 3 and y = 4.")
    await handle.result()

    handle = await km.store("Point Q has coordinates x = 1 and y = 10.")
    await handle.result()

    handle = await km.retrieve(
        "Which points lie in the first quadrant but have y less than 5?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert "P" in answer or "3, 4" in answer, assertion_failed(
        "Answer containing 'P' but not 'Q'",
        answer,
        reasoning,
        "Answer does not correctly identify only point P",
        {"Knowledge Data": km._search()},
    )

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_knowledge/test_columns.py`:

```py
from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
import pytest


@pytest.mark.unit
@_handle_project
def test_create_empty_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._create_empty_column(
        table="MyTable",
        column_name="MyCol",
        column_type="int",
    )
    tables = knowledge_manager._list_tables(include_columns=True)
    assert tables == {"MyTable": {"description": None, "columns": {"MyCol": "int"}}}


@pytest.mark.unit
@_handle_project
def test_create_derived_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    knowledge_manager._create_derived_column(
        table="MyTable",
        column_name="distance",
        equation="({x}**2 + {y}**2)**0.5",
    )
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"x": 3, "y": 4, "distance": (3**2 + 4**2) ** 0.5},
            {"x": 1, "y": 2, "distance": (1**2 + 2**2) ** 0.5},
        ],
    }


@pytest.mark.unit
@_handle_project
def test_delete_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    knowledge_manager._delete_column(table="MyTable", column_name="x")
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"y": 4},
            {"y": 2},
        ],
    }


@pytest.mark.unit
@_handle_project
def test_delete_empty_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._create_empty_column(
        table="MyTable",
        column_name="x",
        column_type="int",
    )
    tables = knowledge_manager._list_tables(include_columns=True)
    assert tables == {"MyTable": {"description": None, "columns": {"x": "int"}}}
    knowledge_manager._delete_column(table="MyTable", column_name="x")
    tables = knowledge_manager._list_tables(include_columns=True)
    assert tables == {"MyTable": {"description": None, "columns": {}}}
    data = knowledge_manager._search()
    assert data == {"MyTable": []}


@pytest.mark.unit
@_handle_project
def test_rename_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    knowledge_manager._rename_column(table="MyTable", old_name="x", new_name="X")
    data = knowledge_manager._search()

    # Assert the exact structure and order of keys
    assert list(data.keys()) == ["MyTable"]
    assert list(data["MyTable"][0].keys()) == ["X", "y"]
    assert list(data["MyTable"][1].keys()) == ["X", "y"]

    # Assert the values
    assert data == {
        "MyTable": [
            {"X": 3, "y": 4},
            {"X": 1, "y": 2},
        ],
    }

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_knowledge/test_add_data.py`:

```py
from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
import pytest


@pytest.mark.unit
@_handle_project
def test_add_data():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
    )
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"name": "Bob", "age": 25},
            {"name": "Alice", "age": 30},
        ],
    }


@pytest.mark.unit
@_handle_project
def test_add_more_data():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"name": "Alice", "age": 30}],
    )
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"name": "Bob", "age": 25}],
    )
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"name": "Bob", "age": 25},
            {"name": "Alice", "age": 30},
        ],
    }

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_knowledge/test_knowledge_embedding.py`:

```py
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project
import pytest


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_knowledge_embedding():
    # Initialize and start the KnowledgeManager thread
    manager = KnowledgeManager()

    # Define table name and schema
    table_name = "ContactPrefs"
    columns = {"content": "str"}
    manager._create_table(name=table_name, columns=columns)

    # Semantically related entries without substring overlap
    entries = [
        {"content": "I prefer email over phone."},
        {"content": "Text messaging works best for me."},
        {"content": "My favorite method is postal mail."},
    ]
    manager._add_data(table=table_name, data=entries)

    # Keyword-based search should find no hits for the term 'preferences'
    keyword_results = manager._search(
        filter="'preferences' in content",
        tables=[table_name],
    )[table_name]
    assert isinstance(keyword_results, list)
    assert len(keyword_results) == 0

    # Embedding-based nearest search for k=1 should return the most relevant entry
    query = "preferred mediums of communication"
    emb_results_k1 = manager._nearest(
        tables=[table_name],
        column="content_emb",
        source="content",
        text=query,
        k=1,
    )[table_name]
    assert len(emb_results_k1) == 1
    assert emb_results_k1[0]["content"] == entries[1]["content"]

    # Embedding-based nearest search for k=2 should respect ordering and limit
    emb_results_k2 = manager._nearest(
        tables=[table_name],
        column="content_emb",
        source="content",
        text=query,
        k=2,
    )[table_name]
    assert len(emb_results_k2) == 2
    # First result should match the top-1, second should be different
    assert emb_results_k2[0]["content"] == emb_results_k1[0]["content"]
    assert emb_results_k2[1]["content"] in [
        e["content"] for e in entries if e["content"] != emb_results_k1[0]["content"]
    ]

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_llm_helpers/test_async_tools_simple.py`:

```py
"""
pytest tests for the llm helpers:
"""

from __future__ import annotations

import unify
import asyncio
import json
import time
import types
from typing import Any

import pytest

# --------------------------------------------------------------------------- #
#  MODULE UNDER TEST                                                          #
# --------------------------------------------------------------------------- #
# Change import path if your helpers live elsewhere
import unity.common.llm_helpers as llmh


# --------------------------------------------------------------------------- #
#  TEST DOUBLES                                                               #
# --------------------------------------------------------------------------- #
class FakeToolCall:
    """Mimics OpenAI's ToolCall object."""

    def __init__(self, name: str, args: dict, call_id: str = "1"):
        self.id = call_id
        self.function = types.SimpleNamespace(
            name=name,
            arguments=json.dumps(args),
        )


def make_response(message):
    """Wrap a Message-like object in the structure returned by AsyncUnify.generate."""
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message)],
    )


class FakeAsyncClient:
    """
    Dumb stand-in for `unify.AsyncUnify`.

    * `generate` pops a pre-scripted response (async).
    * `append_messages` records the conversation for assertions.
    """

    def __init__(self, scripted_responses: list):
        self._responses = scripted_responses[:]
        self.messages: list[dict[str, Any]] = []

    async def generate(self, **_kwargs) -> Any:
        try:
            return self._responses.pop(0)
        except IndexError as exc:
            raise RuntimeError("FakeAsyncClient ran out of scripted responses") from exc

    def append_messages(self, msgs):
        self.messages.extend(msgs)


# --------------------------------------------------------------------------- #
#  HELPERS TO BUILD "MODEL" MESSAGES                                          #
# --------------------------------------------------------------------------- #
def msg_tool_call(name: str, args: dict, call_id: str = "1"):
    return types.SimpleNamespace(
        tool_calls=[FakeToolCall(name, args, call_id)],
        content="",
    )


def msg_final(content: str):
    return types.SimpleNamespace(tool_calls=None, content=content)


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS (sync + async)                                        #
# --------------------------------------------------------------------------- #
def add(x: int, y: int) -> int:  # synchronous
    return x + y


def divide(a: int, b: int) -> float:  # synchronous – may raise
    return a / b


async def fast_tool(res: str = "fast") -> str:  # completes quickly
    await asyncio.sleep(0.05)
    return res


async def slow_tool(res: str = "slow") -> str:  # noticeably slower
    await asyncio.sleep(0.3)
    return res


# --------------------------------------------------------------------------- #
#  HAPPY PATH – single synchronous tool                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_loop_happy_path_single_sync_tool():
    scripted = [
        make_response(msg_tool_call("add", {"x": 2, "y": 3})),
        make_response(msg_final("5")),
    ]
    client = FakeAsyncClient(scripted)

    answer = await llmh.start_async_tool_use_loop(
        client,
        message="Add numbers",
        tools={"add": add},
        max_consecutive_failures=2,
    ).result()

    assert answer.strip() == "5"
    # exactly one tool result fed back
    assert sum(m["role"] == "tool" for m in client.messages) == 1


# --------------------------------------------------------------------------- #
#  MIXED sync/async tools and *early* return to the LLM                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_loop_concurrent_tools_early_generate():
    """
    One LLM turn triggers *both* `fast` and `slow`.
    The loop must return to the model after `fast` completes
    while `slow` is still running.
    """
    events: list[tuple[str, float]] = []

    async def fast():
        events.append(("fast_start", time.monotonic()))
        await asyncio.sleep(0.05)
        events.append(("fast_end", time.monotonic()))
        return "fast"

    async def slow():
        events.append(("slow_start", time.monotonic()))
        await asyncio.sleep(0.3)
        events.append(("slow_end", time.monotonic()))
        return "slow"

    def record_generate():
        events.append(("generate", time.monotonic()))

    # SINGLE model message with *two* tool calls
    first_turn = types.SimpleNamespace(
        tool_calls=[
            FakeToolCall("fast", {}, "1"),
            FakeToolCall("slow", {}, "2"),
        ],
        content="",
    )

    scripted = [
        make_response(first_turn),  # triggers both tools concurrently
        make_response(msg_final("done")),  # model answers after `fast` result only
        make_response(msg_final("ok")),  # model replies after slow (no tools)
    ]

    class InstrumentedClient(FakeAsyncClient):
        async def generate(self, **kwargs):
            record_generate()
            return await super().generate(**kwargs)

    client = InstrumentedClient(scripted)

    answer = await llmh.start_async_tool_use_loop(
        client,
        message="Run fast & slow",
        tools={"fast": fast, "slow": slow},
        max_consecutive_failures=2,
    ).result()

    assert answer.strip() == "ok"

    # ── Timing assertions ────────────────────────────────────────────
    generate_times = [t for e, t in events if e == "generate"]
    fast_end = next(t for e, t in events if e == "fast_end")
    slow_end = next(t for e, t in events if e == "slow_end")

    # 0-th generate: initial model request (before any tool starts)
    # 1-st generate: after `fast` finishes but *before* `slow` ends
    assert generate_times[0] < fast_end
    assert generate_times[1] < slow_end


# --------------------------------------------------------------------------- #
#  RECOVERY AFTER A FAILURE & COUNTER RESET                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_loop_recovers_after_failure():
    scripted = [
        make_response(msg_tool_call("divide", {"a": 4, "b": 0})),  # raises
        make_response(msg_tool_call("divide", {"a": 4, "b": 2})),  # succeeds
        make_response(msg_final("2.0")),
    ]
    client = FakeAsyncClient(scripted)

    answer = await llmh.start_async_tool_use_loop(
        client,
        message="Divide numbers",
        tools={"divide": divide},
        max_consecutive_failures=3,
    ).result()

    assert answer.strip().startswith("2")

    tool_msgs = [m["content"] for m in client.messages if m["role"] == "tool"]
    # first feedback must contain traceback mentioning ZeroDivisionError
    assert any("ZeroDivisionError" in tb for tb in tool_msgs)


# --------------------------------------------------------------------------- #
#  ABORT AFTER MAX CONSECUTIVE FAILURES                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_loop_aborts_after_too_many_failures():
    scripted = [
        make_response(msg_tool_call("divide", {"a": 1, "b": 0})),
        make_response(msg_tool_call("divide", {"a": 1, "b": 0})),
    ]
    client = FakeAsyncClient(scripted)

    with pytest.raises(RuntimeError):
        await llmh.start_async_tool_use_loop(
            client,
            message="Break me",
            tools={"divide": divide},
            max_consecutive_failures=2,
        ).result()


# --------------------------------------------------------------------------- #
#  REALISTIC MIX – first async fast, then sync add                            #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_loop_mixed_sync_async_tools():
    """Ensures sync tools are transparently run in the thread-pool."""
    scripted = [
        make_response(msg_tool_call("fast_tool", {})),
        make_response(msg_tool_call("add", {"x": 6, "y": 7})),
        make_response(msg_final("42")),
    ]
    client = FakeAsyncClient(scripted)

    answer = await llmh.start_async_tool_use_loop(
        client,
        message="Run async then sync",
        tools={"fast_tool": fast_tool, "add": add},
        max_consecutive_failures=2,
    ).result()

    assert answer.strip() == "42"


def square(x: int) -> int:
    return x * x


@pytest.mark.asyncio
async def test_parallel_tool_calls():

    client = unify.AsyncUnify("gpt-4o@openai")

    # Run the loop – ask it to give back the history as well
    await llmh.start_async_tool_use_loop(
        client,
        "Square 2 and 3 please",
        {"square": square},
    ).result()

    # Find the first assistant turn that *requested* tool calls
    first_llm_turn = next(
        m for m in client.messages if m["role"] == "assistant" and m.get("tool_calls")
    )

    # Ensure it actually asked for >1 tools – i.e. parallel tool calls
    assert len(first_llm_turn["tool_calls"]) == 2

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_llm_helpers/test_async_tools_interject_and_stop.py`:

```py
"""
End-to-end behavioural tests for the new *live-handle* features
(`interject`, `stop`, `result`) added to the async-tool loop.

The tests assume that

* `unify.AsyncUnify` exists in the import path,
* `start_async_tool_use_loop` is the public helper that starts the loop and
  returns the handle object, and
* the original `async_tool_use_loop` implementation lives in the same module
  (so we can reuse a couple of internals such as `_dumps` if needed).

We **do not** stub or re-implement `unify`; instead we monkey-patch a few of
its methods on the fly so that the loop sees *plausible* LLM behaviour while
remaining entirely deterministic and fast.
"""

import asyncio
import json
import types
from typing import Any, Dict, List

import pytest

import unify
from unity.common.llm_helpers import start_async_tool_use_loop


# ---------------------------------------------------------------------------#
# Helpers                                                                    #
# ---------------------------------------------------------------------------#


class GenerateScript:
    """
    Tiny state machine that produces a *sequence* of synthetic assistant
    responses so we can control the loop deterministically.
    """

    def __init__(self) -> None:
        self.turn = 0

    def _msg(self, *, tool_calls=None, content=None):
        """Return an object that mimics the OpenAI message structure."""
        return types.SimpleNamespace(tool_calls=tool_calls, content=content)

    def __call__(self, client) -> Any:  # used as patched `generate`
        # pylint: disable=unused-argument
        if self.turn == 0:
            # → Ask for one tool call so the loop spins up a background task.
            self.turn += 1
            tc = [
                {
                    "id": "call_1",
                    "function": {"name": "echo", "arguments": json.dumps({"txt": "A"})},
                }
            ]
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=self._msg(tool_calls=tc))]
            )

        if self.turn == 1:
            # → After the first tool result is returned the model notices the
            #   *interjection* (if any) and may request a second tool call.
            self.turn += 1
            want_b = any(
                m["role"] == "user" and "B please" in m["content"]
                for m in client.messages
            )
            if want_b:
                tc = [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "echo",
                            "arguments": json.dumps({"txt": "B"}),
                        },
                    }
                ]
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=self._msg(tool_calls=tc))]
                )

        # → Final assistant answer (turn 2 or 1 depending on path)
        self.turn += 1
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=self._msg(content="done"))]
        )


async def echo(txt: str) -> str:  # noqa: D401 – simple mock tool
    await asyncio.sleep(0.05)
    return txt


# ---------------------------------------------------------------------------#
# Fixtures                                                                   #
# ---------------------------------------------------------------------------#


@pytest.fixture()
def client():
    return unify.AsyncUnify("gpt-4o@openai")


# ---------------------------------------------------------------------------#
# Tests                                                                       #
# ---------------------------------------------------------------------------#


@pytest.mark.asyncio
async def test_interject_and_result_work_together(client):
    """
    Start the loop, interject a clarification, ensure:
      • the loop *incorporates* the extra user message,
      • the model reacts by asking for an additional tool,
      • we eventually get the expected final answer from `result()`.
    """

    handle = start_async_tool_use_loop(
        client,
        "Echo A please",
        {"echo": echo},
    )

    # - Wait a moment, then inject a follow-up user turn --------------------
    await asyncio.sleep(0.01)
    await handle.interject("And echo B please")  # ← the clarification

    await handle.result()

    # --- Assertions --------------------------------------------------------
    assert client.messages[0] == {"role": "user", "content": "Echo A please"}
    assert client.messages[1]["tool_calls"][0]["function"]["name"] == "echo"
    assert json.loads(client.messages[1]["tool_calls"][0]["function"]["arguments"]) == {
        "txt": "A"
    }
    assert client.messages[2]["role"] == "tool"
    assert client.messages[2]["name"] == "echo"
    assert "A" in client.messages[2]["content"]
    assert client.messages[3] == {"role": "user", "content": "And echo B please"}
    assert client.messages[4]["tool_calls"][0]["function"]["name"] == "echo"
    assert json.loads(client.messages[4]["tool_calls"][0]["function"]["arguments"]) == {
        "txt": "B"
    }
    assert client.messages[5]["role"] == "tool"
    assert client.messages[5]["name"] == "echo"
    assert "B" in client.messages[5]["content"]

    # The conversation must contain our extra user message in order.
    assert any(
        m for m in client.messages if m["role"] == "user" and "echo B" in m["content"]
    )

    # The assistant must have produced *two* distinct tool calls (A and B).
    assistant_turns = [
        m for m in client.messages if m["role"] == "assistant" and m.get("tool_calls")
    ]
    assert len(assistant_turns) == 2


@pytest.mark.asyncio
async def test_stop_cancels_gracefully(client):
    """
    Start the loop and *immediately* request a graceful stop.
    Verify that:
      • `handle.result()` raises `asyncio.CancelledError`,
      • no tool tasks are left running afterwards.
    """

    handle = start_async_tool_use_loop(
        client,
        "Echo something",
        {"echo": echo},
    )

    handle.stop()  # request cancellation right away

    with pytest.raises(asyncio.CancelledError):
        await handle.result()

    # The underlying task should be done and cancelled.
    assert handle.done()


@pytest.mark.asyncio
async def test_multiple_interjects_then_normal_completion(client):
    """
    Fire *two* interjections while the loop is running; ensure each is
    delivered in order and the final result still resolves normally.
    """

    handle = start_async_tool_use_loop(
        client,
        "Echo A please",
        {"echo": echo},
    )

    await asyncio.sleep(0.01)
    await handle.interject("B please")
    await asyncio.sleep(0.01)
    await handle.interject("C please")

    await handle.result()

    # All three user utterances (initial + 2 extras) must be in the history.
    seen = [m["content"] for m in client.messages if m["role"] == "user"]
    assert seen == ["Echo A please", "B please", "C please"]

    # Assistant should have ended up calling the tool at least 3×.
    assistant_turns = [
        m for m in client.messages if m["role"] == "assistant" and m.get("tool_calls")
    ]
    assert len(assistant_turns) == 2

    total_tool_calls = sum(len(t["tool_calls"]) for t in assistant_turns)
    assert total_tool_calls >= 3

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_llm_helpers/test_schemas.py`:

```py
"""
pytest tests for the helper utilities:

* annotation_to_schema           – all supported annotation kinds
* method_to_schema               – schema structure & enum handling
"""

from __future__ import annotations

from enum import Enum

import pytest
from pydantic import BaseModel

import unity.common.llm_helpers as llmh


# --------------------------------------------------------------------------- #
#  TEST DATA TYPES FOR SCHEMA TESTS                                           #
# --------------------------------------------------------------------------- #
class ColumnType(str, Enum):
    str = "str"
    int = "int"


class Person(BaseModel):
    name: str
    age: int


# --------------------------------------------------------------------------- #
#  annotation_to_schema                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "t, checker",
    [
        (str, lambda s: s == {"type": "string"}),
        (int, lambda s: s == {"type": "integer"}),
        (
            ColumnType,
            lambda s: s["type"] == "string" and set(s["enum"]) == {"str", "int"},
        ),
        (
            Person,
            lambda s: s["type"] == "object" and {"name", "age"} <= set(s["properties"]),
        ),
        (
            dict[str, int],
            lambda s: s["type"] == "object"
            and s["additionalProperties"]["type"] == "integer",
        ),
        (
            list[Person],
            lambda s: s["type"] == "array" and s["items"]["type"] == "object",
        ),
    ],
)
def test_annotation_to_schema_variants(t, checker):
    """Every major annotation flavour is converted correctly."""
    assert checker(llmh.annotation_to_schema(t))


# --------------------------------------------------------------------------- #
#  method_to_schema – enum round-trip                                         #
# --------------------------------------------------------------------------- #
def _demo_func(a: str, col: ColumnType):
    """Docstring for unit test."""
    return None


def test_method_to_schema_includes_enum():
    schema = llmh.method_to_schema(_demo_func)
    params = schema["function"]["parameters"]["properties"]
    assert params["a"]["type"] == "string"
    # Enum must appear with *exact* allowed literals
    assert params["col"]["enum"] == ["str", "int"]

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/__init__.py`:

```py
# This file makes the tests directory a Python package

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_event_bus/test_prefill.py`:

```py
import pytest
import asyncio
import datetime as dt

from unity.events.event_bus import EventBus, Event
from unity.events.types.message import Message
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_prefill_from_upstream_on_new_instance():
    """
    After some events are published with one EventBus, a brand-new EventBus
    should hydrate those same events from Unify logs into its in-memory window.
    """
    window = 10
    bus1 = EventBus(windows_sizes={"message": window})

    base_ts = dt.datetime.now(dt.UTC)
    published: list[Event] = []

    # Publish five message events with ascending timestamps
    for i in range(5):
        evt = Event(
            context="message",
            timestamp=base_ts + dt.timedelta(seconds=i),
            payload=Message.model_construct(),
        )
        published.append(evt)
        await bus1.publish(evt)

    # Give the async logger a brief moment (usually unnecessary, but harmless)
    await asyncio.sleep(0.05)

    # Create a *new* EventBus that should preload from persisted logs
    bus2 = EventBus(windows_sizes={"message": window})

    latest = await bus2.get_latest(types=["message"], limit=window)

    # Each originally-sent event (identified by its ts & payload) must be present
    for sent in published:
        assert any(
            rec.timestamp == sent.timestamp and rec.payload == sent.payload
            for rec in latest
        ), f"Event with ts {sent.timestamp.isoformat()} not found in prefilled window"

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_event_bus/test_get_latest.py`:

```py
import pytest
import asyncio
import datetime as dt
from collections import deque

from unity.events.event_bus import EventBus, Event
from unity.events.types.message import Message
from unity.events.types.message_exchange_summary import MessageExchangeSummary
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_get_latest():
    """A single publish should be retrievable via get_latest()."""
    bus = EventBus()

    payload = Message.model_construct()
    event = Event(
        context="Messages", timestamp=dt.datetime.now(dt.UTC), payload=payload
    )

    await bus.publish(event)

    # Read back through the public API
    latest = await bus.get_latest(types=["Messages"], limit=1)

    # There should be at least one event, and it should be the one we just published
    assert latest and latest[0] == event


@pytest.mark.asyncio
@_handle_project
async def test_get_latest_mixed_types_ordering():
    """Interwoven publishing of two event types should come back newest-first."""
    window_sizes = {"Messages": 10, "MessageExchangeSummary": 10}
    bus = EventBus(windows_sizes=window_sizes)

    # Start from a clean slate for deterministic assertions
    for t in ("Messages", "MessageExchangeSummary"):
        bus._deques.setdefault(t, deque(maxlen=window_sizes[t])).clear()

    base_ts = dt.datetime.now(dt.UTC)
    events = []

    # Publish 6 events: message, summary, message, …
    for idx in range(6):
        etype, payload_cls = (
            ("Messages", Message)
            if idx % 2 == 0
            else ("MessageExchangeSummary", MessageExchangeSummary)
        )

        evt = Event(
            context=etype,
            timestamp=base_ts
            + dt.timedelta(seconds=idx),  # strictly ascending timestamps
            payload=payload_cls.model_construct(),
        )
        events.append(evt)
        await bus.publish(evt)

    # Retrieve the newest 10 events (more than we published)
    latest = await bus.get_latest(limit=10)

    # Filter the list to only the events we just wrote
    latest_ours = [e for e in latest if e in events]

    # They should be returned newest-first, i.e. exactly reverse of the order written
    assert latest_ours == list(reversed(events))


@pytest.mark.asyncio
@_handle_project
async def test_concurrent_get_latest_lock_integrity():
    """
    Fire several concurrent `get_latest` requests with different `types` filters
    and limits.  All should complete without dead-locking and return the
    correct (newest-first) slices, proving the read-side lock holds up.
    """
    windows = {"Messages": 100, "MessageExchangeSummary": 100}
    bus = EventBus(windows_sizes=windows)

    # Ensure a clean slate so results are deterministic
    for t, w in windows.items():
        bus._deques.setdefault(t, deque(maxlen=w)).clear()

    # ── Publish 40 interleaved events ───────────────────────────────
    base_ts = dt.datetime.now(dt.UTC)
    events = []
    for i in range(40):
        etype, payload_cls = (
            ("Messages", Message)
            if i % 2 == 0
            else ("MessageExchangeSummary", MessageExchangeSummary)
        )
        evt = Event(
            context=etype,
            timestamp=base_ts + dt.timedelta(microseconds=i),
            payload=payload_cls.model_construct(),
        )
        events.append(evt)
        await bus.publish(evt)

    # Pre-compute expected slices (newest-first order)
    all_newest = list(reversed(events))
    messages_newest = [e for e in all_newest if e.type == "Messages"]
    summaries_newest = [e for e in all_newest if e.type == "MessageExchangeSummary"]

    expected_r1 = messages_newest[:5]
    expected_r2 = summaries_newest[:7]
    expected_r3 = []  # empty filter → empty result
    expected_r4 = all_newest[:15]

    # ── Concurrent read tasks ──────────────────────────────────────
    tasks = [
        asyncio.create_task(bus.get_latest(types=["Messages"], limit=5)),  # r1
        asyncio.create_task(
            bus.get_latest(types=["MessageExchangeSummary"], limit=7)
        ),  # r2
        asyncio.create_task(bus.get_latest(types=[], limit=10)),  # r3 (no types)
        asyncio.create_task(bus.get_latest(types=None, limit=15)),  # r4 (both types)
    ]

    r1, r2, r3, r4 = await asyncio.gather(*tasks)

    # ── Assertions ─────────────────────────────────────────────────
    assert r1 == expected_r1
    assert r2 == expected_r2
    assert r3 == expected_r3
    assert r4 == expected_r4

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_event_bus/test_publish.py`:

```py
import pytest
import asyncio
import datetime as dt
from collections import deque

from unity.events.event_bus import EventBus, Event
from unity.events.types.message import Message
from unity.events.types.message_exchange_summary import MessageExchangeSummary
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_publish():
    """Publishing a valid event should complete without exceptions
    and the event should be stored in the in-memory deque.
    """
    bus = EventBus()  # use defaults (50-event windows)

    # create a minimal Message payload; model_construct() skips field validation,
    # so it works even if Message has required fields we don’t care about here
    payload = Message.model_construct()

    event = Event(
        context="message",
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
        payload=payload,
    )

    # This should run cleanly …
    await bus.publish(event)

    # … and the event should now be in the per-type deque
    assert event in bus._deques["message"]


@pytest.mark.asyncio
@_handle_project
async def test_concurrent_publishes_lock_integrity():
    """
    Do a burst of concurrent publishes across two event types; all should succeed
    and be visible afterwards, demonstrating that the internal asyncio.Lock
    protects the critical section.
    """
    window = 200
    bus = EventBus(
        windows_sizes={"message": window, "message_exchange_summary": window}
    )

    # Clear any pre-existing state for determinism
    for typ in ("message", "message_exchange_summary"):
        bus._deques.setdefault(typ, deque(maxlen=window)).clear()

    base_ts = dt.datetime.now(dt.UTC)
    n_events = 100
    events: list[Event] = []
    publish_tasks = []

    for i in range(n_events):
        etype, payload_cls = (
            ("message", Message)
            if i % 2 == 0
            else ("message_exchange_summary", MessageExchangeSummary)
        )
        evt = Event(
            context=etype,
            timestamp=base_ts
            + dt.timedelta(microseconds=i),  # unique, strictly increasing
            payload=payload_cls.model_construct(),
        )
        events.append(evt)
        publish_tasks.append(asyncio.create_task(bus.publish(evt)))

    # Run all publishes concurrently; will raise if any individual publish fails
    await asyncio.gather(*publish_tasks)

    # Fetch back everything; limit well above what we sent
    latest = await bus.get_latest(limit=window)

    # Keep only the events we just published (ignore any older prefilled logs)
    our_ts = {e.timestamp for e in events}
    latest_ours = [e for e in latest if e.timestamp in our_ts]

    # Every event we published must be present
    assert len(latest_ours) == n_events

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_event_bus/test_windows.py`:

```py
import datetime as dt
import pytest
from collections import deque

from unity.events.event_bus import EventBus, Event
from unity.events.types.message import Message
from unity.events.types.message_exchange_summary import MessageExchangeSummary
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_window_eviction_at_limit():
    """When more than *window* events are published, the oldest should fall off."""
    window = 3
    bus = EventBus(windows_sizes={"message": window})

    # Start from a known clean state for this type (harmless use of a private attr)
    bus._deques.setdefault(
        "message", bus._deques.get("message", deque(maxlen=window))
    ).clear()

    # Publish window + 1 events with ascending timestamps
    events = []
    base_ts = dt.datetime.now(dt.UTC)
    for i in range(window + 1):
        evt = Event(
            context="message",
            timestamp=base_ts + dt.timedelta(seconds=i),
            payload=Message.model_construct(),
        )
        events.append(evt)
        await bus.publish(evt)

    # Fetch everything currently buffered for "message"
    latest = await bus.get_latest(types=["message"], limit=10)

    # Filter to the events we just published (there may be pre-existing logs)
    latest_ours = [e for e in latest if e in events]

    # We expect only *window* of our events (the newest three) to remain
    assert len(latest_ours) == window
    assert events[0] not in latest_ours  # the earliest one was evicted
    assert latest_ours[0] == events[-1]  # newest appears first (newest-first order)


@pytest.mark.asyncio
@_handle_project
async def test_window_eviction_mixed_sizes_and_ordering():
    """
    Verify that *different* per-type window sizes are respected simultaneously
    and that `get_latest()` returns the surviving events newest-first.
    """
    # Different windows: 2 for Message, 3 for MessageExchangeSummary
    windows = {"message": 2, "message_exchange_summary": 3}
    bus = EventBus(windows_sizes=windows)

    # Clear any prefilled data so we know exactly what's in memory
    for t, w in windows.items():
        bus._deques.setdefault(t, deque(maxlen=w)).clear()

    base_ts = dt.datetime.now(dt.UTC)

    # Publish seven events with strictly ascending timestamps
    # pattern: M, S, M, S, M, S, S
    publish_plan = [
        ("message", Message),
        ("message_exchange_summary", MessageExchangeSummary),
        ("message", Message),
        ("message_exchange_summary", MessageExchangeSummary),
        ("message", Message),
        ("message_exchange_summary", MessageExchangeSummary),
        ("message_exchange_summary", MessageExchangeSummary),
    ]
    events = []
    for idx, (etype, payload_cls) in enumerate(publish_plan):
        evt = Event(
            context=etype,
            timestamp=base_ts + dt.timedelta(seconds=idx),
            payload=payload_cls.model_construct(),
        )
        events.append(evt)
        await bus.publish(evt)

    # Retrieve more than enough to capture everything that survived
    latest = await bus.get_latest(limit=10)

    # Keep only the events we just published (ignore any earlier logs)
    ours = [e for e in latest if e in events]

    # Expected survivors by window:
    expected_messages = events[2:5:2]  # idx 2 and 4 → 2 newest messages
    expected_summaries = events[3:]  # idx 3,5,6 → 3 newest summaries
    expected_survivors = list(reversed(expected_summaries + expected_messages))
    # reversed() because get_latest() returns newest-first

    # 1️⃣ Correct counts per type
    assert sum(e.context == "message" for e in ours) == windows["message"]
    assert (
        sum(e.context == "message_exchange_summary" for e in ours)
        == windows["message_exchange_summary"]
    )

    # 2️⃣ Overall ordering newest-first
    assert ours == expected_survivors, "Events not in expected newest-first order"

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_task_list/test_update_complex.py`:

```py
"""
Complex English-text integration tests for TaskListManager.update
===============================================================

Each test seeds a project with a small set of tasks, issues a human-like
instruction via the *public* `.update()` method and asserts that the mutated
state matches expectations.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from tests.helpers import _handle_project
from unity.task_list_manager.task_list_manager import TaskListManager
from unity.task_list_manager.types.priority import Priority
from unity.task_list_manager.types.schedule import Schedule


# --------------------------------------------------------------------------- #
#  Helper to seed a deterministic task set                                   #
# --------------------------------------------------------------------------- #


def _seed_basic_tasks(tlm: TaskListManager) -> List[int]:
    """Return list of task-ids in creation order."""

    ids = []
    ids.append(
        tlm._create_task(
            name="Write quarterly report",
            description="Draft the Q2 report (send email to finance).",
            status="active",
        ),
    )
    ids.append(
        tlm._create_task(
            name="Prepare slide deck",
            description="Create slides for the board meeting. Email once done.",
            status="queued",
        ),
    )
    ids.append(
        tlm._create_task(
            name="Client follow-up email",
            description="Send email to prospective client about proposal.",
            status="queued",
        ),
    )
    return ids


# --------------------------------------------------------------------------- #
#  1.  Re-ordering in the runnable queue                                     #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_update_reorder_queue():
    tlm = TaskListManager()

    ids = _seed_basic_tasks(tlm)
    assert [t.task_id for t in tlm._get_task_queue()] == ids  # initial order

    handle = tlm.update(
        text="Could you do Client follow-up email after Write quarterly report?",
    )
    await handle.result()

    queue = [t.task_id for t in tlm._get_task_queue()]
    # expected order: 0 (report) -> 2 (follow-up) -> 1 (slides)
    assert queue == [ids[0], ids[2], ids[1]]


# --------------------------------------------------------------------------- #
# 2. Cancel all tasks whose description mentions sending emails              #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_update_cancel_email_tasks():
    tlm = TaskListManager()

    _seed_basic_tasks(tlm)

    handle = tlm.update(text="Please cancel all tasks related to sending emails.")
    await handle.result()

    tasks = tlm._search()
    for t in tasks:
        if "email" in t["description"].lower():
            assert t["status"] == "cancelled"
        else:
            assert t["status"] != "cancelled"


# --------------------------------------------------------------------------- #
# 3. Lower priority for tasks scheduled next Monday                          #
# --------------------------------------------------------------------------- #


def _next_weekday(dt: datetime, weekday: int) -> datetime:
    """Return dt on next weekday (0=Mon)."""

    days_ahead = (weekday - dt.weekday() + 7) % 7 or 7
    return dt + timedelta(days=days_ahead)


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_update_lower_priority_next_monday():
    tlm = TaskListManager()

    # create one scheduled next Monday with high priority
    base = datetime.now(timezone.utc)
    next_mon = _next_weekday(base, 0).replace(hour=9, minute=0, second=0, microsecond=0)

    sched = Schedule(start_time=next_mon.isoformat(), prev_task=None, next_task=None)
    tlm._create_task(
        name="Send KPI report",
        description="Automated email of KPIs to leadership.",
        schedule=sched,
        priority=Priority.high,
    )

    handle = tlm.update(
        text="Please lower the priority of all tasks which are scheduled for next Monday.",
    )
    await handle.result()

    task = tlm._search()[0]
    assert task["priority"] == Priority.normal


# --------------------------------------------------------------------------- #
# 4. Bulk description edit (regex-like replace)                              #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_update_bulk_description_replace():
    tlm = TaskListManager()

    tlm._create_task(
        name="Arrange viewing",
        description="Contact the estate agent to arrange the viewing.",
    )
    tlm._create_task(
        name="Send brochure",
        description="Email the estate agent the sales brochure.",
    )

    handle = tlm.update(
        text="Please update all task descriptions to refer to Mr. Smith instead of 'the estate agent'.",
    )
    await handle.result()

    for t in tlm._search():
        assert re.search(r"Mr\.\s?Smith", t["description"]) is not None

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_task_list/test_creation_deletion.py`:

```py
import pytest
from tests.helpers import _handle_project
from unity.task_list_manager.task_list_manager import TaskListManager
from unity.task_list_manager.types.priority import Priority
from unity.task_list_manager.types.status import Status


@_handle_project
@pytest.mark.unit
def test_create_task():
    task_list_manager = TaskListManager()
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_list_manager._search()
    assert task_list == [
        {
            "name": "Promote Jeff Smith",
            "description": "Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
            "status": Status.active,
            "schedule": {"prev_task": None, "next_task": None},
            "deadline": None,
            "repeat": None,
            "priority": Priority.normal,
            "task_id": 0,
        },
    ]


@_handle_project
@pytest.mark.unit
def test_delete_task():
    task_list_manager = TaskListManager()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_list_manager._search()
    assert len(task_list) == 1

    # delete
    task_list_manager._delete_task(task_id=0)
    task_list = task_list_manager._search()
    assert task_list == []

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_task_list/test_cancel_tasks.py`:

```py
from tests.helpers import _handle_project
from unity.task_list_manager.task_list_manager import TaskListManager
import pytest


@_handle_project
@pytest.mark.unit
def test_cancel_single_task():
    """Cancelling a single active task should set its status to 'cancelled'."""
    tlm = TaskListManager()

    # Create an active task (id will be 0)
    tlm._create_task(
        name="Follow-up with client",
        description="Send a thank-you email and next-steps proposal.",
    )

    # Cancel the task
    tlm._cancel_tasks([0])

    # Verify the task was cancelled
    tasks = tlm._search()
    assert tasks[0]["status"] == "cancelled"


@_handle_project
@pytest.mark.unit
def test_cancel_multiple_tasks():
    """Cancelling multiple tasks at once should update all of their statuses."""
    tlm = TaskListManager()

    # Create two tasks (ids 0 and 1)
    tlm._create_task(
        name="Prepare quarterly report",
        description="Compile Q1 financials into slide deck.",
    )
    tlm._create_task(
        name="Schedule team off-site",
        description="Book venue and send calendar invites.",
    )

    # Cancel both tasks
    tlm._cancel_tasks([0, 1])

    # Verify both tasks were cancelled
    tasks = tlm._search()
    status_by_id = {t["task_id"]: t["status"] for t in tasks}
    assert status_by_id[0] == "cancelled"
    assert status_by_id[1] == "cancelled"


@_handle_project
@pytest.mark.unit
def test_cancel_completed_task_raises():
    """Attempting to cancel a task that is already completed should raise an AssertionError."""
    tlm = TaskListManager()

    # Create a task that is already completed
    tlm._create_task(
        name="Ship version 1.0",
        description="Publish release notes and push tags.",
        status="completed",
    )

    # Expect an AssertionError when trying to cancel it
    with pytest.raises(AssertionError):
        tlm._cancel_tasks([0])

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_task_list/test_pause_continue_active_task.py`:

```py
import pytest
from tests.helpers import _handle_project
from unity.task_list_manager.task_list_manager import TaskListManager


@_handle_project
@pytest.mark.unit
def test_get_paused_task():
    task_list_manager = TaskListManager()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
        status="paused",
    )
    task_list = task_list_manager._search()
    assert task_list[0]["name"] == "Promote Jeff Smith"

    # verify it's the same task
    paused_task = task_list_manager._get_paused_task()
    assert paused_task["task_id"] == 0


@_handle_project
@pytest.mark.unit
def test_get_active_task():
    task_list_manager = TaskListManager()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
        status="active",
    )
    task_list = task_list_manager._search()
    assert task_list[0]["name"] == "Promote Jeff Smith"

    # verify it's the same task
    paused_task = task_list_manager._get_active_task()
    assert paused_task["task_id"] == 0


@_handle_project
@pytest.mark.unit
def test_pause():
    task_list_manager = TaskListManager()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
        status="active",
    )
    task_list = task_list_manager._search()
    assert task_list[0]["name"] == "Promote Jeff Smith"

    # verify the task is active
    paused_task = task_list_manager._get_active_task()
    assert paused_task["task_id"] == 0

    # pause the task
    task_list_manager._pause()

    # verify there is no active task
    assert task_list_manager._get_active_task() is None

    # verify there is a paused task
    assert task_list_manager._get_paused_task()


@_handle_project
@pytest.mark.unit
def test_continue():
    task_list_manager = TaskListManager()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
        status="paused",
    )
    task_list = task_list_manager._search()
    assert task_list[0]["name"] == "Promote Jeff Smith"

    # verify the task is paused
    paused_task = task_list_manager._get_paused_task()
    assert paused_task["task_id"] == 0

    # pause the task
    task_list_manager._continue()

    # verify there is no paused task
    assert task_list_manager._get_paused_task() is None

    # verify there is an active task
    assert task_list_manager._get_active_task()

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_task_list/test_update.py`:

```py
import pytest
from tests.helpers import _handle_project
from unity.task_list_manager.task_list_manager import TaskListManager
from unity.task_list_manager.types.status import Status
from unity.task_list_manager.types.priority import Priority


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
async def test_update_create_task_via_text():
    tlm = TaskListManager()

    cmd = (
        "Please add a new task called 'Promote Jeff Smith' with the "
        "description 'Send an email to Jeff Smith, kindly congratulating him and "
        "explaining that he has been promoted from sales rep to sales manager.'"
    )
    handle = tlm.update(text=cmd)
    await handle.result()

    tasks = tlm._search()
    assert len(tasks) == 1
    task = tasks[0]
    assert task["name"] == "Promote Jeff Smith"
    assert task["description"].startswith("Send an email to Jeff Smith")
    assert task["status"] in (Status.active, Status.queued, "active", "queued")
    assert task["priority"] == Priority.normal


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
async def test_update_delete_task_via_text():
    tlm = TaskListManager()

    # create a task directly (bypassing LLM) so we know the ID is 0
    tlm._create_task(
        name="Write quarterly report",
        description="Compile and draft the Q2 report for management.",
    )
    assert len(tlm._search()) == 1

    # delete via plain-English update
    handle = tlm.update(text="Delete the task with id 0.")
    await handle.result()

    assert tlm._search() == []

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_task_list/test_update_tools.py`:

```py
import pytest
from datetime import datetime, timezone, timedelta
from tests.helpers import _handle_project
from unity.task_list_manager.types.status import Status
from unity.task_list_manager.types.priority import Priority
from unity.task_list_manager.task_list_manager import TaskListManager
from unity.task_list_manager.types.repetition import RepeatPattern, Frequency, Weekday


@_handle_project
@pytest.mark.unit
def test_update_task_name():
    task_list_manager = TaskListManager()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_list_manager._search()
    assert task_list[0]["name"] == "Promote Jeff Smith"

    # rename
    task_list_manager._update_task_name(
        task_id=0,
        new_name="Give Jeff Smith a promotion",
    )
    task_list = task_list_manager._search()
    assert task_list[0]["name"] == "Give Jeff Smith a promotion"


@_handle_project
@pytest.mark.unit
def test_update_task_description():
    task_list_manager = TaskListManager()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_list_manager._search()
    assert (
        task_list[0]["description"]
        == "Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager."
    )

    # rename
    task_list_manager._update_task_description(
        task_id=0,
        new_description="Call Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_list_manager._search()
    assert (
        task_list[0]["description"]
        == "Call Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager."
    )


@_handle_project
@pytest.mark.unit
def test_update_task_status():
    task_list_manager = TaskListManager()

    # create
    task_list_manager._create_task(
        name="Promote Jeff Smith",
        description="Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager.",
    )
    task_list = task_list_manager._search()
    assert (
        task_list[0]["description"]
        == "Send an email to Jeff Smith, kindly congratulating him and explaining that he has been promoted from sales rep to sales manager."
    )

    # update status
    task_list_manager._update_task_status(
        task_ids=0,
        new_status=Status.cancelled,
    )
    task_list = task_list_manager._search()
    assert task_list[0]["status"] == "cancelled"


@_handle_project
@pytest.mark.unit
def test_update_task_start_at():
    tlm = TaskListManager()

    tlm._create_task(
        name="Send customer survey",
        description="Email Q2 customer-satisfaction survey.",
    )

    start = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    tlm._update_task_start_at(task_id=0, new_start_at=start)

    task_list = tlm._search()
    assert task_list[0]["schedule"]["start_time"] == start


@_handle_project
@pytest.mark.unit
def test_update_task_deadline():
    tlm = TaskListManager()

    tlm._create_task(
        name="File quarterly taxes",
        description="Prepare documents for accounting.",
    )

    deadline = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    tlm._update_task_deadline(task_id=0, new_deadline=deadline)

    task_list = tlm._search()
    assert task_list[0]["deadline"] == deadline


@_handle_project
@pytest.mark.unit
def test_update_task_repetition():
    tlm = TaskListManager()

    tlm._create_task(
        name="Daily stand-up",
        description="10-minute team sync",
    )

    rule = RepeatPattern(frequency=Frequency.WEEKLY, interval=1, weekdays=[Weekday.MO])
    tlm._update_task_repetition(task_id=0, new_repeat=[rule])

    task_list = tlm._search()
    # The manager stores *.model_dump()* (a plain dict) so compare like-for-like
    assert task_list[0]["repeat"] == [rule.model_dump()]


@_handle_project
@pytest.mark.unit
def test_update_task_priority():
    tlm = TaskListManager()

    tlm._create_task(
        name="Patch security vulnerability",
        description="Apply CVE-2025-1234 hot-fix to production.",
    )

    tlm._update_task_priority(task_id=0, new_priority=Priority.high)

    task_list = tlm._search()
    assert task_list[0]["priority"] == Priority.high

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_task_list/test_tasklist_embedding.py`:

```py
import pytest
from unity.task_list_manager.task_list_manager import TaskListManager
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_tasklist_embedding_search():
    # Start the TaskListManager thread
    manager = TaskListManager()

    # Create two tasks semantically related to "searching LinkedIn for contacts"
    id1 = manager._create_task(
        name="connecting with industry professionals",
        description="looking for contacts on a career-oriented site",
    )
    id2 = manager._create_task(
        name="collecting resumes from job boards",
        description="harvesting candidate information from online employment listings",
    )

    # Keyword-based filter search should yield no hits
    filter_results = manager._search(filter="'LinkedIn' in description")
    assert filter_results == []

    # Semantic search with k=2 returns both tasks in ascending distance order
    sim_results = manager._search_similar(text="searching LinkedIn for contacts", k=2)
    assert isinstance(sim_results, list)
    assert len(sim_results) == 2
    assert sim_results[0]["name"] == "connecting with industry professionals"
    assert sim_results[1]["name"] == "collecting resumes from job boards"

    # Semantic search with k=1 respects the limit and returns only the closest match
    sim_results_k1 = manager._search_similar(
        text="searching LinkedIn for contacts",
        k=1,
    )
    assert len(sim_results_k1) == 1
    assert sim_results_k1[0]["name"] == "connecting with industry professionals"

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_task_list/test_tasklist_ask.py`:

```py
"""
Integration tests for TaskListManager.ask
================================================

Identical content moved from test_ask.py to avoid module-name collision with
TranscriptManager tests.
"""

# pylint: disable=duplicate-code

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest
import unify

from unity.task_list_manager.task_list_manager import TaskListManager
from unity.task_list_manager.types.priority import Priority
from unity.task_list_manager.types.schedule import Schedule
from unity.common.llm_helpers import _dumps
from tests.assertion_helpers import assertion_failed


class ScenarioBuilder:
    """Populate Unify with a small, meaningful task list."""

    def __init__(self) -> None:
        if "test_task_ask" in unify.list_projects():
            unify.delete_project("test_task_ask")
        unify.activate("test_task_ask")
        self.tlm = TaskListManager()
        self._seed_tasks()

    def _seed_tasks(self) -> None:
        """Create five tasks with various states for robust querying."""

        self.tlm._create_task(  # Active
            name="Write quarterly report",
            description="Compile and draft the Q2 report for management.",
            status="active",
        )

        self.tlm._create_task(  # Queued
            name="Prepare slide deck",
            description="Create slides for the upcoming board meeting.",
            status="queued",
        )

        sched = Schedule(  # Scheduled
            prev_task=None,
            next_task=None,
            start_time=datetime(2025, 6, 1, 9, 0, tzinfo=timezone.utc).isoformat(),
        )
        self.tlm._create_task(
            name="Client meeting",
            description="Meet with ABC Corp for contract renewal.",
            status="scheduled",
            schedule=sched,
        )

        self.tlm._create_task(  # Paused
            name="Deploy new release",
            description="Roll out version 2.0 to production servers.",
            status="paused",
        )

        self.tlm._create_task(  # High-priority queued
            name="Hotfix security vulnerability",
            description="Apply CVE-2025-1234 patch to all services.",
            status="queued",
            priority=Priority.high,
        )


# ---------------- Ground-truth helpers ---------------- #


def _answer_semantic(tlm: TaskListManager, question: str) -> str:
    q = question.lower()
    tasks = tlm._search()

    if "currently active" in q:
        return next(t for t in tasks if t["status"] == "active")["name"]

    if "tasks are queued" in q:
        return str(sum(1 for t in tasks if t["status"] == "queued"))

    if "client meeting" in q and "scheduled" in q:
        mtg = next(t for t in tasks if "client meeting" in t["name"].lower())
        return mtg["schedule"]["start_time"].split("T")[0]

    if "priority" in q and "hotfix" in q:
        hotfix = next(t for t in tasks if "hotfix" in t["name"].lower())
        return str(hotfix["priority"])

    return "N/A"


QUESTIONS = [
    "Which task is currently active?",
    "How many tasks are queued at the moment?",
    "When is the client meeting scheduled for?",
    "What is the priority level of the hotfix task?",
]


def _llm_assert_correct(
    question: str,
    expected: str,
    candidate: str,
    steps: list,  # noqa: D401 – clarity outweighs strict type accuracy
) -> None:
    """Assert *candidate* satisfies *expected* for *question* via an LLM judge.

    On failure, the full reasoning *steps* are appended to the assertion
    message to aid debugging.
    """

    judge = unify.Unify("o4-mini@openai", cache=True)
    judge.set_system_message(
        "You are a strict unit-test judge. "
        "You will be given a question, a ground-truth answer derived directly "
        "from the data, and a candidate answer produced by the system under test. "
        'Respond ONLY with valid JSON of the form {"correct": true} or {"correct": false}. '
        "Mark correct⇢true if a reasonable human would accept the candidate as answering the question fully and accurately; otherwise false.",
    )

    payload = _dumps(
        {"question": question, "ground_truth": expected, "candidate": candidate},
        indent=4,
    )
    result = judge.generate(payload)

    match = re.search(r"\{.*\}", result, re.S)
    assert match, assertion_failed(
        "Expected JSON format from LLM judge",
        result,
        steps,
        "LLM judge returned unexpected format",
    )
    verdict = json.loads(match.group(0))
    assert verdict.get("correct") is True, assertion_failed(
        expected,
        candidate,
        steps,
        f"Question: {question}",
    )


@pytest.fixture
def tlm_scenario() -> TaskListManager:  # noqa: D401 – fixture, not function
    return ScenarioBuilder().tlm


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.timeout(180)
async def test_ask_semantic_with_llm_judgement(
    question: str,
    tlm_scenario: TaskListManager,
) -> None:
    try:
        handle, steps = tlm_scenario.ask(
            text=question,
            return_reasoning_steps=True,
        )
        candidate = await handle.result()
        expected = _answer_semantic(tlm_scenario, question)
        _llm_assert_correct(question, expected, candidate, steps)
    except Exception as exc:
        if "test_task_ask" in unify.list_projects():
            unify.delete_project("test_task_ask")
        raise exc

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_task_list/test_task_queue.py`:

```py
import pytest
from tests.helpers import _handle_project
from unity.task_list_manager.task_list_manager import TaskListManager
from unity.task_list_manager.types.schedule import Schedule


# Convenience to make schedules quickly
def _sch(prev_, next_):
    from datetime import datetime, timezone

    return Schedule(
        prev_task=prev_,
        next_task=next_,
        start_time=datetime.now(timezone.utc).isoformat(),
    )


@_handle_project
@pytest.mark.unit
def test_get_queue_and_reorder():
    tlm = TaskListManager()

    # -----  create three queued tasks with an explicit chain  -----
    t0 = tlm._create_task(
        name="T0",
        description="first",
        schedule=_sch(None, 1),
    )
    t1 = tlm._create_task(
        name="T1",
        description="second",
        schedule=_sch(0, 2),
    )
    t2 = tlm._create_task(
        name="T2",
        description="third",
        schedule=_sch(1, None),
    )

    queue = tlm._get_task_queue()
    assert [t.task_id for t in queue] == [0, 1, 2]

    # -----  swap the order (0,2,1)  -----
    tlm._update_task_queue(original=[0, 1, 2], new=[0, 2, 1])

    new_q = tlm._get_task_queue()
    assert [t.task_id for t in new_q] == [0, 2, 1]


@_handle_project
@pytest.mark.unit
def test_insert_into_queue():
    tlm = TaskListManager()

    # base queue with one task
    tlm._create_task(name="base", description="x", schedule=_sch(-1, -1))

    # create a brand-new task that will be inserted
    new_id = tlm._create_task(name="insert-me", description="y")

    tlm._update_task_queue(original=[0], new=[0, new_id])

    q = tlm._get_task_queue()
    assert [t.task_id for t in q] == [0, new_id]
    # also check the linkage of node 0 -> new_id
    assert q[0].schedule.next_task == new_id

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_update_text_complex.py`:

```py
"""
Complex English-text integration tests for TaskListManager.update
===============================================================

Each test seeds a project with a small set of tasks, issues a human-like
instruction via the *public* `.update()` method and asserts that the mutated
state matches expectations.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from tests.helpers import _handle_project
from tests.assertion_helpers import assertion_failed
from unity.task_list_manager.task_list_manager import TaskListManager
from unity.task_list_manager.types.priority import Priority
from unity.task_list_manager.types.schedule import Schedule


# Monkey patch the TaskListManager's update method to capture reasoning steps
original_update = TaskListManager.update


def patched_update(self, text: str, return_reasoning_steps: bool = False):
    try:
        result, steps = original_update(self, text, return_reasoning_steps=True)
        self._last_reasoning_steps = steps
        return result if not return_reasoning_steps else (result, steps)
    except Exception as e:
        self._last_reasoning_steps = []
        raise e


TaskListManager.update = patched_update


# --------------------------------------------------------------------------- #
#  Helper to seed a deterministic task set                                   #
# --------------------------------------------------------------------------- #


def _seed_basic_tasks(tlm: TaskListManager) -> List[int]:
    """Return list of task-ids in creation order."""

    ids = []
    ids.append(
        tlm._create_task(
            name="Write quarterly report",
            description="Draft the Q2 report (send email to finance).",
            status="active",
        ),
    )
    ids.append(
        tlm._create_task(
            name="Prepare slide deck",
            description="Create slides for the board meeting. Email once done.",
            status="queued",
        ),
    )
    ids.append(
        tlm._create_task(
            name="Client follow-up email",
            description="Send email to prospective client about proposal.",
            status="queued",
        ),
    )
    return ids


# --------------------------------------------------------------------------- #
#  1.  Re-ordering in the runnable queue                                     #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.eval
@pytest.mark.timeout(240)
def test_update_reorder_queue():
    tlm = TaskListManager()

    ids = _seed_basic_tasks(tlm)
    assert [t.task_id for t in tlm._get_task_queue()] == ids  # initial order

    tlm.update(text="Could you do Client follow-up email after Write quarterly report?")

    queue = [t.task_id for t in tlm._get_task_queue()]
    expected_order = [
        ids[0],
        ids[2],
        ids[1],
    ]  # 0 (report) -> 2 (follow-up) -> 1 (slides)
    assert queue == expected_order, assertion_failed(
        expected_order,
        queue,
        getattr(tlm, "_last_reasoning_steps", []),
        "Task queue order doesn't match expected order after update",
        {"Task Data": tlm._search()},
    )


# --------------------------------------------------------------------------- #
# 2. Cancel all tasks whose description mentions sending emails              #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.eval
@pytest.mark.timeout(240)
def test_update_cancel_email_tasks():
    tlm = TaskListManager()

    _seed_basic_tasks(tlm)

    tlm.update(text="Please cancel all tasks related to sending emails.")

    tasks = tlm._search()
    for t in tasks:
        if "email" in t["description"].lower():
            assert t["status"] == "cancelled", assertion_failed(
                "cancelled",
                t["status"],
                getattr(tlm, "_last_reasoning_steps", []),
                f"Task '{t['name']}' with email in description should be cancelled",
                {"Task Data": tasks},
            )
        else:
            assert t["status"] != "cancelled", assertion_failed(
                "not cancelled",
                t["status"],
                getattr(tlm, "_last_reasoning_steps", []),
                f"Task '{t['name']}' without email in description should not be cancelled",
                {"Task Data": tasks},
            )


# --------------------------------------------------------------------------- #
# 3. Lower priority for tasks scheduled next Monday                          #
# --------------------------------------------------------------------------- #


def _next_weekday(dt: datetime, weekday: int) -> datetime:
    """Return dt on next weekday (0=Mon)."""

    days_ahead = (weekday - dt.weekday() + 7) % 7 or 7
    return dt + timedelta(days=days_ahead)


@_handle_project
@pytest.mark.eval
@pytest.mark.timeout(240)
def test_update_lower_priority_next_monday():
    tlm = TaskListManager()

    # create one scheduled next Monday with high priority
    base = datetime.now(timezone.utc)
    next_mon = _next_weekday(base, 0).replace(hour=9, minute=0, second=0, microsecond=0)

    sched = Schedule(start_time=next_mon.isoformat(), prev_task=None, next_task=None)
    tlm._create_task(
        name="Send KPI report",
        description="Automated email of KPIs to leadership.",
        schedule=sched,
        priority=Priority.high,
    )

    tlm.update(
        text="Please lower the priority of all tasks which are scheduled for next Monday.",
    )

    task = tlm._search()[0]
    assert task["priority"] == Priority.normal, assertion_failed(
        Priority.normal,
        task["priority"],
        getattr(tlm, "_last_reasoning_steps", []),
        f"Task '{task['name']}' scheduled for next Monday should have normal priority",
        {"Task Data": tlm._search()},
    )


# --------------------------------------------------------------------------- #
# 4. Bulk description edit (regex-like replace)                              #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.eval
@pytest.mark.timeout(240)
def test_update_bulk_description_replace():
    tlm = TaskListManager()

    tlm._create_task(
        name="Arrange viewing",
        description="Contact the estate agent to arrange the viewing.",
    )
    tlm._create_task(
        name="Send brochure",
        description="Email the estate agent the sales brochure.",
    )

    tlm.update(
        text="Please update all task descriptions to refer to Mr. Smith instead of 'the estate agent'.",
    )

    for t in tlm._search():
        has_mr_smith = re.search(r"Mr\.\s?Smith", t["description"]) is not None
        assert has_mr_smith, assertion_failed(
            "Description containing 'Mr. Smith'",
            t["description"],
            getattr(tlm, "_last_reasoning_steps", []),
            f"Task '{t['name']}' description should contain 'Mr. Smith'",
            {"Task Data": tlm._search()},
        )

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_controller/test_action_filter.py`:

```py
from unity.controller import action_filter as af
from unity.controller.states import BrowserState
from unity.controller import commands as cmd


def test_in_textbox_includes_textbox_commands():
    st = BrowserState(in_textbox=True)
    valid = af.get_valid_actions(st, mode="actions")
    # Every textbox command should be included
    assert cmd.TEXTBOX_COMMANDS <= valid
    # still allows scrolling
    assert cmd.CMD_SCROLL_DOWN in valid


def test_dialog_prompt_limits_actions():
    st = BrowserState(dialog_open=True, dialog_type="prompt")
    valid = af.get_valid_actions(st, mode="actions")
    assert {cmd.CMD_ACCEPT_DIALOG, cmd.CMD_DISMISS_DIALOG, cmd.CMD_TYPE_DIALOG} <= valid
    # normal navigation should be blocked
    assert cmd.CMD_SCROLL_DOWN not in valid
    assert cmd.CMD_OPEN_URL not in valid


def test_captcha_pending_restricts_actions():
    st = BrowserState(captcha_pending=True)
    valid = af.get_valid_actions(st, mode="actions")
    assert valid == {cmd.CMD_SCROLL_UP, cmd.CMD_SCROLL_DOWN, cmd.CMD_STOP_SCROLLING}

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_controller/test_agent.py`:

```py
import pytest

from unity.controller import agent as agent_mod
from unity.controller.states import BrowserState


@pytest.mark.timeout(30)
def test_text_to_browser_action():
    """Smoke-test that agent.text_to_browser_action returns a dict with keys.
    Relies on online Unify backend; will skip when network/API not available."""
    try:
        result = agent_mod.text_to_browser_action(
            "open browser",
            screenshot=None,
            tabs=[],
            buttons=None,
            history=[],
            state=BrowserState(),
        )
    except Exception as exc:
        pytest.skip(f"Skipping – Unify backend unavailable: {exc}")

    assert isinstance(result, dict)
    assert "action" in result
    assert "rationale" in result
    assert "open" in result["rationale"] and "browser" in result["rationale"]
    assert "new_tab" in result["action"]


@pytest.mark.timeout(30)
def test_ask_llm_bool():
    """Smoke-test ask_llm with boolean response_type. Skips when backend unavailable."""
    try:
        answer = agent_mod.ask_llm("Is 2+2 equal to 4?", response_type=bool)
    except Exception as exc:
        pytest.skip(f"Skipping – Unify backend unavailable: {exc}")

    assert isinstance(answer, bool)


@pytest.mark.timeout(30)
def test_ask_llm_str():
    """ask_llm should return a plain string when response_type=str"""
    try:
        answer = agent_mod.ask_llm("Say hello in one word.", response_type=str)
    except Exception as exc:
        pytest.skip(f"Skipping – Unify backend unavailable: {exc}")

    assert isinstance(answer, str)
    assert len(answer) > 0


from pydantic import BaseModel, Field


class _Coords(BaseModel):
    lat: float = Field(..., description="latitude")
    lon: float = Field(..., description="longitude")


@pytest.mark.timeout(30)
def test_ask_llm_custom_model():
    """ask_llm should handle arbitrary Pydantic response models."""
    try:
        ret = agent_mod.ask_llm(
            "Provide the coordinates of the Eiffel Tower as JSON with 'lat' and 'lon'.",
            response_type=_Coords,
        )
    except Exception as exc:
        pytest.skip(f"Skipping – Unify backend unavailable: {exc}")

    # Because our custom model has no 'answer' attribute, ask_llm returns the model instance
    assert isinstance(ret, _Coords)
    assert -90 <= ret.lat <= 90
    assert -180 <= ret.lon <= 180


@pytest.mark.timeout(30)
def test_ask_llm_int():
    """ask_llm should return an int when response_type=int"""
    try:
        answer = agent_mod.ask_llm("What is 10 minus 3?", response_type=int)
    except Exception as exc:
        pytest.skip(f"Skipping – Unify backend unavailable: {exc}")

    assert isinstance(answer, int)


@pytest.mark.timeout(30)
def test_ask_llm_float():
    """ask_llm should return a float when response_type=float"""
    try:
        answer = agent_mod.ask_llm(
            "Provide 1 divided by 3 as a decimal.",
            response_type=float,
        )
    except Exception as exc:
        pytest.skip(f"Skipping – Unify backend unavailable: {exc}")

    assert isinstance(answer, float)

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_controller/test_command_runner.py`:

```py
import types
import sys

# ---------------------------------------------------------------------------
#  Stub heavy deps: redis, playwright, BrowserWorker  (same as previous file)
# ---------------------------------------------------------------------------

# Stub only playwright (CommandRunner depends on it).
plw_mod = types.ModuleType("playwright")
plw_sync = types.ModuleType("playwright.sync_api")


# minimal types used in CommandRunner type hints
class _Stub:  # generic empty class
    pass


plw_sync.BrowserContext = _Stub
plw_sync.Page = _Stub

sys.modules["playwright"] = plw_mod
sys.modules["playwright.sync_api"] = plw_sync

# ---------------------------------------------------------------------------
#  Imports after stubbing
# ---------------------------------------------------------------------------
from unity.controller.playwright import command_runner as cr_mod  # noqa: E402
from unity.controller import commands as cmd_mod  # noqa: E402

# ---------------------------------------------------------------------------
#  Test CommandRunner scroll-speed parsing
# ---------------------------------------------------------------------------


# stub BrowserContext / Page for CommandRunner
class _DummyPage:
    def __init__(self):
        self._scroll_y = 0

        def _wheel(_dx, dy):
            # native wheel scroll affects scrollY too
            self._scroll_y += dy

        self.mouse = types.SimpleNamespace(wheel=_wheel)
        self.keyboard = types.SimpleNamespace(
            press=lambda *a, **k: None,
            type=lambda *a, **k: None,
            down=lambda *a, **k: None,
            up=lambda *a, **k: None,
        )

    # emulate page.evaluate with minimal behaviour for scroll & query
    def evaluate(self, script, *args):
        # querying scrollY
        if script == "scrollY":
            return self._scroll_y
        # handle our injected smooth-scroll by dict arg containing delta
        if args and isinstance(args[0], dict) and "delta" in args[0]:
            self._scroll_y += args[0]["delta"]
            return None
        # default
        return 0

    def bring_to_front(self):
        pass

    def goto(self, url, *_, **__):
        self.url = url
        # simplistic: set title to url
        self._title = url

    def title(self):
        return getattr(self, "_title", "dummy")

    url = "about:blank"


class _DummyCtx:
    def __init__(self):
        self.pages = [_DummyPage()]


# ---------------------------------------------------------------------------
#  Command registry integrity tests
# ---------------------------------------------------------------------------


def test_all_primitives_unique():
    literals = [v for k, v in vars(cmd_mod).items() if k.startswith("CMD_")]
    assert len(literals) == len(set(literals)), "Command literals must be unique"


def test_autoscroll_groups_consistency():
    # every command in AUTOSCROLL_START / ACTIVE must exist in ALL_PRIMITIVES
    base = cmd_mod.ALL_PRIMITIVES
    for g in (cmd_mod.AUTOSCROLL_START, cmd_mod.AUTOSCROLL_ACTIVE):
        assert g <= base


def test_group_subsets():
    """Ensure every group constant is fully included in ALL_PRIMITIVES."""
    groups = [
        cmd_mod.TEXTBOX_COMMANDS,
        cmd_mod.NAV_COMMANDS,
        cmd_mod.BUTTON_PATTERNS,
        cmd_mod.SCROLL_PATTERNS["up"],
        cmd_mod.SCROLL_PATTERNS["down"],
        cmd_mod.AUTOSCROLL_START,
        cmd_mod.AUTOSCROLL_ACTIVE,
        cmd_mod.DIALOG_COMMANDS,
        cmd_mod.POPUP_COMMANDS,
    ]
    master = cmd_mod.ALL_PRIMITIVES
    for grp in groups:
        assert grp <= master


def test_wildcard_trailing_star_patterns():
    """Verify wildcard commands keep the expected '*' suffix where required."""
    # patterns that should contain '*'
    patterns = [
        cmd_mod.CMD_ENTER_TEXT,
        cmd_mod.CMD_SCROLL_DOWN,
        cmd_mod.CMD_SCROLL_UP,
        cmd_mod.CMD_SEARCH,
        cmd_mod.CMD_OPEN_URL,
        cmd_mod.CMD_CLICK_BUTTON,
        cmd_mod.CMD_SELECT_TAB,
        cmd_mod.CMD_CLOSE_TAB,
        cmd_mod.CMD_SELECT_POPUP,
        cmd_mod.CMD_TYPE_DIALOG,
    ]
    for p in patterns:
        assert p.endswith("*"), f"Pattern {p} missing terminal '*'"


# ---------------------------------------------------------------------------
#  Additional state consistency tests
# ---------------------------------------------------------------------------


def test_open_url_updates_state():
    ctx = _DummyCtx()
    runner = cr_mod.CommandRunner(ctx, log_fn=lambda *_: None)
    runner.run("open_url example.com")
    assert runner.state.url == "https://example.com"
    assert ctx.pages[0].url == "https://example.com"


def test_start_scrolling_default_speed():
    # when no speed given, default should be 250 px/s
    runner = cr_mod.CommandRunner(_DummyCtx(), log_fn=lambda *_: None)
    runner.run("start_scrolling_down")
    assert runner.state.auto_scroll == "down"
    assert runner.state.scroll_speed == 250


def test_scroll_up_updates_scroll_y():
    ctx = _DummyCtx()
    runner = cr_mod.CommandRunner(ctx, log_fn=lambda *_: None)
    runner.run("scroll_up 120")
    # scroll up uses negative delta so scroll_y decreases
    assert runner.state.scroll_y == -120


def test_scroll_speed_parsed():
    runner = cr_mod.CommandRunner(_DummyCtx(), log_fn=lambda *_: None)
    runner.run("start_scrolling_down 600")
    assert runner.state.auto_scroll == "down"
    assert runner.state.scroll_speed == 600


def test_autoscroll_stop():
    runner = cr_mod.CommandRunner(_DummyCtx(), log_fn=lambda *_: None)
    runner.run("start_scrolling_up 300")
    assert runner.state.auto_scroll == "up"
    runner.run("stop_scrolling")
    assert runner.state.auto_scroll is None


def test_click_out_resets_flag():
    # Page that will report "in text box" only on first call
    class _ClickPage(_DummyPage):
        def __init__(self):
            super().__init__()
            self._first = True

        def evaluate(self, script, *_args):
            if "return ['input'" in script:
                if self._first:
                    self._first = False
                    return True  # initially inside textbox
                return False  # after blur, no longer inside
            return 0

    ctx = _DummyCtx()
    ctx.pages = [_ClickPage()]
    runner = cr_mod.CommandRunner(ctx, log_fn=lambda *_: None)
    runner.state.in_textbox = True
    runner.run("click_out")
    assert runner.state.in_textbox is False

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_controller/test_controller_public.py`:

```py
import types, sys, pytest  # noqa: E402 (stubs before imports)

# ---------------------------------------------------------------------------
# Stubs for external heavy deps (redis & BrowserWorker) BEFORE importing code
# ---------------------------------------------------------------------------


# --- Redis stub -----------------------------------------------------------
class _FakePubSub:
    def __init__(self):
        self._messages = []

    def subscribe(self, *_):
        pass

    def listen(self):
        # generator expected by Controller.run(); empty -> instant end if used
        while self._messages:
            yield self._messages.pop()
        while True:
            yield {"type": "noop"}

    def get_message(self):
        return None


class _FakeRedis:
    def __init__(self, *a, **k):
        self._pubsub = _FakePubSub()
        self.published: list[tuple[str, str]] = []

    def pubsub(self):
        return self._pubsub

    def publish(self, chan, msg):
        self.published.append((chan, msg))


sys.modules.setdefault("redis", types.ModuleType("redis"))
sys.modules["redis"].Redis = _FakeRedis


# --- BrowserWorker stub ---------------------------------------------------
class _DummyWorker:
    def __init__(self, *a, **k):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self, *a, **k):
        pass


# ensure parent package module exists
pkg_path = "unity.controller.playwright"
if pkg_path not in sys.modules:
    sys.modules[pkg_path] = types.ModuleType("playwright_stub")
worker_mod = types.ModuleType("worker")
worker_mod.BrowserWorker = _DummyWorker
sys.modules["unity.controller.playwright.worker"] = worker_mod

# ---------------------------------------------------------------------------
# Imports after stubbing
# ---------------------------------------------------------------------------
from unity.controller.controller import Controller  # noqa: E402


@pytest.mark.timeout(30)
def test_controller_observe_bool():
    """Smoke-test Controller.observe with bool response."""
    c = Controller()
    # minimal cached context
    c._observe_ctx = {"state": {}}
    c._last_shot = b""
    try:
        ret = c.observe("Is 2+2 equal to 4?", bool)
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")
    assert isinstance(ret, bool)


@pytest.mark.timeout(30)
def test_controller_observe_str():
    """Smoke-test observe with string response type."""
    c = Controller()
    c._observe_ctx = {"state": {}}
    c._last_shot = b""
    try:
        ans = c.observe("Reply with 'hello'.", str)
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")
    assert isinstance(ans, str)
    assert len(ans) > 0


@pytest.mark.timeout(30)
def test_controller_act_smoke():
    """Smoke-test Controller.act and Redis publications."""
    c = Controller()
    try:
        action_str = c.act("open browser")
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")
    assert isinstance(action_str, str)
    # browser worker should have been started
    assert c._browser_open is True
    # ensure action_completion event was published
    assert ("action_completion", action_str) in c._redis_client.published

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/helpers.py`:

```py
import sys
import functools
import traceback
import unify


def _handle_project(
    test_fn=None,
    try_reuse_prev_ctx: bool = False,
    delete_ctx_on_exit=False,
):

    if test_fn is None:
        return lambda f: _handle_project(
            f,
            try_reuse_prev_ctx=try_reuse_prev_ctx,
            delete_ctx_on_exit=delete_ctx_on_exit,
        )

    @functools.wraps(test_fn)
    def wrapper(*args, **kwargs):
        file_path = test_fn.__code__.co_filename
        test_path = "/".join(file_path.split("/tests/")[1].split("/"))[:-3]
        ctx = f"{test_path}/{test_fn.__name__}" if test_path else test_fn.__name__

        if not try_reuse_prev_ctx and unify.get_contexts(prefix=ctx):
            unify.delete_context(ctx)
        try:
            with unify.Context(ctx):
                unify.set_trace_context("Traces")
                unify.traced(test_fn)(*args, **kwargs)
            if delete_ctx_on_exit:
                unify.delete_context(ctx)
        except:
            if delete_ctx_on_exit:
                unify.delete_context(ctx)
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_string = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            raise Exception(f"{tb_string}")

    return wrapper

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/tests/test_function_manager/test_function_manager.py`:

```py
"""
Comprehensive unit-tests for `FunctionManager`

Coverage
========
✓ add_functions                           (existing happy-path + validation)
✓ list_functions                          (with / without implementations)
✓ delete_function                         (single, cascading, non-cascading)
✓ search_functions                        (flexible Python-expr filtering)

The tests introduce a *minimal* stub of the `unify` API so that they remain
fully hermetic.  Nothing outside this file is required.
"""

from __future__ import annotations

import pytest
from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager


# --------------------------------------------------------------------------- #
#  4.  Existing add_functions tests                                           #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_add_single_function_success():
    src = (
        "def double(x):\n"
        "    y = 0\n"
        "    for _ in range(2):\n"
        "        y = y + x\n"
        "    return y\n"
    )
    fm = FunctionManager()
    result = fm.add_functions(implementations=src)
    assert result == {"double": "added"}


@_handle_project
@pytest.mark.unit
def test_add_multiple_functions_with_dependency():
    add_src = "def add(a, b):\n    return a + b\n"
    twice_src = "def twice(x):\n    return add(x, x)\n"
    fm = FunctionManager()
    result = fm.add_functions(implementations=[add_src, twice_src])
    assert result == {"add": "added", "twice": "added"}


@_handle_project
@pytest.mark.parametrize(
    "source,exp_msg",
    [
        ("def bad(x)\n    return x", "Syntax error"),  # syntax error
        ("    def indented(x):\n        return x", "must start at column 0"),  # indent
        ("def foo(x):\n    import math\n    return x", "Imports are not allowed"),
        (
            "def uses_unknown(x):\n    return unknown(x)",
            "references unknown function",
        ),
        ("def uses_vars(x):\n    return vars(x)", "Built-in 'vars' is not permitted"),
        (
            "def uses_math(x):\n    import math\n    return math.sin(x)",
            "Imports are not allowed",
        ),
    ],
)
@pytest.mark.unit
def test_validation_errors(source: str, exp_msg: str):
    fm = FunctionManager()
    with pytest.raises(ValueError, match=exp_msg):
        fm.add_functions(implementations=source)


# --------------------------------------------------------------------------- #
#  5.  list_functions                                                         #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_list_functions_with_and_without_implementations():
    add_src = (
        "def add(a: int, b: int) -> int:\n"
        '    """Add two numbers"""\n'
        "    return a + b\n"
    )
    fm = FunctionManager()
    fm.add_functions(implementations=add_src)

    # (a) default summary
    listing = fm.list_functions()
    assert listing.keys() == {"add"}
    assert "implementation" not in listing["add"]
    # The argspec includes type hints and return annotation
    assert "(a: int, b: int) -> int" in listing["add"]["argspec"]
    assert listing["add"]["docstring"] == "Add two numbers"

    # (b) include full source
    full = fm.list_functions(include_implementations=True)
    assert add_src.strip() == full["add"]["implementation"].strip()


# --------------------------------------------------------------------------- #
#  6.  delete_function                                                        #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_delete_single_function():
    fm = FunctionManager()
    fm.add_functions(implementations="def alpha():\n    return 1\n")
    assert len(fm.list_functions()) == 1

    fm.delete_function(function_id=0)
    assert fm.list_functions() == {}


@_handle_project
@pytest.mark.unit
def test_delete_function_with_dependants_cascades():
    add_src = "def add(a, b):\n    return a + b\n"
    twin_src = "def twin(x):\n    return add(x, x)\n"
    fm = FunctionManager()
    fm.add_functions(implementations=[add_src, twin_src])

    # delete `add` AND everything that depends on it
    fm.delete_function(function_id=0, delete_dependents=True)
    assert fm.list_functions() == {}


@_handle_project
@pytest.mark.unit
def test_delete_function_without_cascading_leaves_dependants():
    add_src = "def add(a, b):\n    return a + b\n"
    twin_src = "def twin(x):\n    return add(x, x)\n"
    fm = FunctionManager()
    fm.add_functions(implementations=[add_src, twin_src])

    fm.delete_function(function_id=0, delete_dependents=False)
    remaining = fm.list_functions()
    assert remaining.keys() == {"twin"}


# --------------------------------------------------------------------------- #
#  7.  search_functions                                                       #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_search_functions_filtering_across_columns():
    price_src = (
        "def price_total(p, tax):\n"
        '    """Return total price including tax"""\n'
        "    return p + tax\n"
    )
    square_src = "def square(x):\n    return x * x\n"
    use_src = "def apply_price(x):\n    return price_total(x, 0)\n"
    fm = FunctionManager()
    fm.add_functions(implementations=[price_src, square_src, use_src])

    # filter on docstring contents
    hits = fm.search_functions(filter="'price' in docstring")
    names = {h["name"] for h in hits}
    assert names == {"price_total"}

    # filter by Python predicate on the `name` column
    hits = fm.search_functions(filter="name[0:2] == 'sq'")
    assert {h["name"] for h in hits} == {"square"}

    # find callers of `price_total`
    hits = fm.search_functions(filter="'price_total' in calls")
    assert {h["name"] for h in hits} == {"apply_price"}

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/common/embed_utils.py`:

```py
"""
Utility functions for embedding-based vector search through the logs.
"""

import os

import requests
import unify

# Model to use for text embeddings
EMBED_MODEL = "text-embedding-3-small"


API_KEY = os.environ["UNIFY_KEY"]


def ensure_vector_column(
    context: str,
    embed_column: str,
    source_column: str,
    derived_expr: str | None = None,
) -> None:
    """
    Ensure that a vector column exists in the given context. If it does not,
    create a derived column using the embed() function with the defined embedding model.

    Args:
        context (str): The Unify context (e.g., "Knowledge/table_name" or "ContextName").
        embed_column (str): The name of the vector column to ensure. (eg: "content_emb")
        source_column (str): The name of the source column to embed. (eg: "content_plus_desc")
        derived_expr Optional(str): An optional expression to dynamically derive the source column (in case it's not already present) (eg: "str({name}) + ' || ' + str({description})")
    """
    # Retrieve existing columns and their types
    existing = unify.get_fields(context=context)
    # If the source column is already present, do nothing
    if source_column not in existing:
        # Create the derived vector column
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/derived"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        expr = derived_expr.replace("{", "{lg:")
        json_input = {
            "project": unify.active_project(),
            "context": context,
            "key": source_column,
            "equation": expr,
            "referenced_logs": {"lg": {"context": context}},
        }
        response = requests.request("POST", url, json=json_input, headers=headers)
        assert response.status_code == 200, response.text

    # If the vector column is already present, do nothing
    if embed_column in existing:
        return
    # Define the embedding equation
    embed_expr = (
        "embed({source_column}".replace("source_column", source_column)
        + f", model='{EMBED_MODEL}')"
    )

    url = f"{os.environ['UNIFY_BASE_URL']}/logs/derived"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    embed_expr = embed_expr.replace("{", "{lg:")
    json_input = {
        "project": unify.active_project(),
        "context": context,
        "key": embed_column,
        "equation": embed_expr,
        "referenced_logs": {"lg": {"context": context}},
    }
    response = requests.request("POST", url, json=json_input, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/common/__init__.py`:

```py
from .llm_helpers import AsyncToolLoopHandle

__all__ = ["AsyncToolLoopHandle"]

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/common/llm_helpers.py`:

```py
import json
import asyncio
import inspect
import traceback
from enum import Enum
from pydantic import BaseModel
from typing import (
    Tuple,
    List,
    Dict,
    Set,
    Union,
    Optional,
    Any,
    get_type_hints,
    get_origin,
    get_args,
    Callable,
)

import unify
from ..constants import LOGGER


TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _dumps(
    obj: Any,
    idx: List[Union[str, int]] = None,
    indent: int = None,
) -> Any:
    # prevents circular import
    from unify.logging.logs import Log

    base = False
    if idx is None:
        base = True
        idx = list()
    if isinstance(obj, BaseModel):
        ret = obj.model_dump()
    elif inspect.isclass(obj) and issubclass(obj, BaseModel):
        ret = obj.model_json_schema()
    elif isinstance(obj, Log):
        ret = obj.to_json()
    elif isinstance(obj, dict):
        ret = {k: _dumps(v, idx + ["k"]) for k, v in obj.items()}
    elif isinstance(obj, list):
        ret = [_dumps(v, idx + [i]) for i, v in enumerate(obj)]
    elif isinstance(obj, tuple):
        ret = tuple(_dumps(v, idx + [i]) for i, v in enumerate(obj))
    else:
        ret = obj
    return json.dumps(ret, indent=indent) if base else ret


def annotation_to_schema(ann: Any) -> Dict[str, Any]:
    """Convert a Python type annotation into an OpenAI-compatible JSON-Schema
    fragment, including full support for Pydantic BaseModel subclasses.
    """

    # ── 0. Remove typing.Annotated wrapper, if any ────────────────────────────
    origin = get_origin(ann)
    if origin is not None and origin.__name__ == "Annotated":  # Py ≥3.10
        ann = get_args(ann)[0]

    # ── 1. Primitive scalars (str/int/float/bool) ────────────────────────────
    if ann in TYPE_MAP:
        return {"type": TYPE_MAP[ann]}

    # ── 2. Enum subclasses (e.g. ColumnType) ─────────────────────────────────
    if isinstance(ann, type) and issubclass(ann, Enum):
        return {"type": "string", "enum": [member.value for member in ann]}

    # ── 3. Pydantic model ────────────────────────────────────────────────────
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        # Pydantic already produces an OpenAPI/JSON-Schema compliant dictionary.
        # We can embed that verbatim.  (It contains 'title', 'type', 'properties',
        # 'required', etc.  Any 'definitions' block is also allowed by the spec.)
        return ann.model_json_schema()

    # ── 4. typing.Dict[K, V]  → JSON object whose values follow V ────────────
    origin = get_origin(ann)
    if origin is dict or origin is Dict:
        # ignore key type; JSON object keys are always strings
        _, value_type = get_args(ann)
        return {
            "type": "object",
            "additionalProperties": annotation_to_schema(value_type),
        }

    # ── 5. typing.List[T] or list[T]  → JSON array of T ──────────────────────
    if origin in (list, List):
        (item_type,) = get_args(ann)
        return {
            "type": "array",
            "items": annotation_to_schema(item_type),
        }

    # ── 6. typing.Union / Optional …  → anyOf schemas ────────────────────────
    if origin is Union:
        sub_schemas = [annotation_to_schema(a) for a in get_args(ann)]
        # Collapse trivial Optional[X] (i.e. Union[X, NoneType]) into X
        if len(sub_schemas) == 2 and {"type": "null"} in sub_schemas:
            return next(s for s in sub_schemas if s != {"type": "null"})
        return {"anyOf": sub_schemas}

    # ── 7. Fallback – treat as generic string ────────────────────────────────
    return {"type": "string"}


def method_to_schema(bound_method):
    sig = inspect.signature(bound_method)
    hints = get_type_hints(bound_method)
    props, required = {}, []
    for name, param in sig.parameters.items():
        ann = hints.get(name, str)
        props[name] = annotation_to_schema(ann)
        if param.default is inspect._empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": bound_method.__name__,
            "description": (bound_method.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


async def _async_tool_use_loop_inner(
    client: "unify.AsyncUnify",
    message: str,
    tools: Dict[str, Callable],
    *,
    interject_queue: asyncio.Queue[str],
    cancel_event: asyncio.Event,
    max_consecutive_failures: int = 3,
    log_steps: bool = False,
) -> str:
    r"""
     Drive a structured-tool conversation with an LLM until it produces a
     *final* textual answer, executing tool calls concurrently along the way
     and remaining interruptible at any moment via ``cancel_event``.

     ----------
     High-level behaviour
     --------------------
     1. **Seed the conversation** with the user's ``message`` and the JSON
        schemas for every tool in ``tools``.
     2. Enter an **event loop** that interleaves two concerns:

        *Listening* – wait for **either**
          • the first of the in-flight tool tasks to finish **or**
          • an external cancellation signal.

        *Thinking / Acting* – whenever no pending tool task completes the loop
        checks with the LLM (`client.generate`) to learn whether it should:

          • launch new tool calls (**branch D**) – they are scheduled as
            asyncio tasks and execution jumps back to *Listening*, **or**

          • emit ordinary assistant text (**branch E**) – if *no* tool calls
            are outstanding the loop returns that text as the function result;
            otherwise it keeps listening for the remaining tasks.

     3. **Failure handling** – any exception raised by a tool counts as a
        *failure*; after ``max_consecutive_failures`` in a row the loop aborts
        with :class:`RuntimeError`.

     4. **Graceful cancellation** – if ``cancel_event`` is set (or the caller
        cancels the outer task) all running tools are cancelled and awaited
        before propagating :class:`asyncio.CancelledError`.

    5. **On-the-fly user interjections** – a small wrapper
       :func:`start_async_tool_use_loop` now launches the loop in its own task
       and returns a *handle* object that lets a caller **modify** the
       conversation while it is still in flight:

         • ``await handle.interject(text)`` queues an additional *user* turn
           which is merged into the dialogue just before the next LLM step,
           so already-running tool calls are not disturbed.

         • ``handle.stop()`` triggers the same graceful-shutdown path as
           manually setting ``cancel_event`` – useful when the wrapper is
           nested inside another loop that wants to cancel it from “above”.

       Because the handle is an ordinary Python object these two methods can
       themselves be exposed as *tools* to a parent loop, giving you **nested
       conversations** whose inner loops can be steered or halted by the
       outer assistant.

     ----------
     Execution branches in detail
     ----------------------------
     **A – Listen for task completion or cancellation**

         * If at least one tool task is pending the loop waits for the first
           task **or** ``cancel_event``.
         * A set ``pending`` keeps track of every scheduled task and
           ``task_info`` maps each task to the corresponding *(tool-name,
           call-id)* pair supplied by the LLM.
         * If the *cancel* waiter exits first the loop raises
           :class:`asyncio.CancelledError` immediately.
         * Otherwise the finished tool's result (or traceback on error) is
           appended to the conversation as a ``"role": "tool"`` message and
           the consecutive-failure counter is updated.

     **B – Early cancellation check**
         Skip the LLM step altogether if ``cancel_event`` *has already* been
         set while no tasks were pending.

     **C – Drain queued interjections**

         * At the very start of each iteration the loop empties an internal
           ``asyncio.Queue`` fed by ``handle.interject`` and appends every
           payload as a ``"role": "user"`` message.  This guarantees any new
           clarifications reach the model before it decides on the next tool
           calls.

     **D – Ask the LLM what to do next**
         ``client.generate`` is called with:

         * the accumulated conversation,
         * ``tools_schema`` describing every available function,
         * ``tool_choice="auto"`` – the model decides whether to call a tool
           or to speak.

     **E – Launch new tool calls**
         For every tool call proposed in ``msg.tool_calls`` a coroutine is
         prepared (executed in a thread if the function is synchronous),
         wrapped in ``asyncio.create_task`` and added to ``pending``.
         Control returns to *Listening* immediately so tools can run
         concurrently.

     **F – No new tool calls**

         * If some tool calls are *still* running: loop back to *Listening*.
         * Otherwise – no tasks pending **and** the LLM produced ordinary text –
           the function returns that text to the caller and terminates.

     ----------
     Parameters
     ----------
     client : :class:`unify.AsyncUnify`
         Stateful chat-completion client that must expose
         ``append_messages`` and an async ``generate`` method compatible with
         the OpenAI ChatCompletion API.
     message : str
         The very first user message that kicks off the assistant dialogue.
     tools : Dict[str, Callable]
         Mapping **name → callable** for every function the assistant may
         invoke.  Each callable must be JSON-serialisable via
         :pyfunc:`method_to_schema`; asynchronous functions are awaited,
         synchronous ones are wrapped in :func:`asyncio.to_thread`.
     cancel_event : Optional[:class:`asyncio.Event`], default ``None``
         If provided, the caller can set this event to *politely* abort the
         loop.  A fresh, unset :class:`~asyncio.Event` is created when ``None``
         is given so that callers may also cancel by simply cancelling the
         outer task.
     max_consecutive_failures : int, default ``3``
         After this many **back-to-back** exceptions from tool calls the loop
         aborts with :class:`RuntimeError` to avoid an infinite crash cycle.
     log_steps : bool, default ``False``
         Placeholder for future instrumentation; currently unused.

     ----------
     Returns
     -------
     str
         The assistant's final plain-text reply **after** all required tool
         interactions have completed.
         When ``return_history=True`` the function instead yields
         ``Tuple[str, List[Dict[str, Any]]]`` – the assistant reply **and** the
         complete chat transcript up to that point.

     ----------
     Raises
     ------
     asyncio.CancelledError
         Raised as soon as cancellation is requested, *after* any running tool
         tasks have been cancelled and awaited.
     RuntimeError
         When ``consecutive_failures`` reaches ``max_consecutive_failures``.

     Using the live handle for interjections
     --------------------------------------
     >>> handle = start_async_tool_use_loop(
     ...     client,
     ...     "Which task is most important?",
     ...     task_tools,
     ... )
     >>> # 500 ms later we realise we forgot a constraint
     >>> await handle.interject("Which is also scheduled for this week?")
     >>> answer = await handle.result()
     >>> print(answer)
     'Task XYZ is both high-priority and due this week.'
    """

    if log_steps:
        LOGGER.info(f"\n🧑‍💻 {message}\n")

    # ── initial prompt ───────────────────────────────────────────────────────
    tools_schema = [method_to_schema(v) for v in tools.values()]
    client.append_messages([{"role": "user", "content": message}])

    consecutive_failures = 0
    pending: Set[asyncio.Task] = set()
    task_info: Dict[asyncio.Task, Tuple[str, str]] = {}

    try:
        while True:
            # ── 0.  Drain queued *user* interjections (but **only** if all
            #        previous tool calls have been satisfied).  Injecting a
            #        user turn while the API still expects tool-role messages
            #        would violate the OpenAI protocol and trigger a 400.
            if not pending:
                while True:
                    try:
                        extra = interject_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if log_steps:
                        LOGGER.info(f"\n⚡ Interjection → {extra!r}\n")
                    client.append_messages([{"role": "user", "content": extra}])

            # ── A.  Wait for tool completion OR cancellation  ───────────────
            if pending:
                waiters = pending | {asyncio.create_task(cancel_event.wait())}
                done, _ = await asyncio.wait(
                    waiters, return_when=asyncio.FIRST_COMPLETED
                )

                if any(t for t in done if t not in pending):
                    raise asyncio.CancelledError  # cancellation wins

                for task in done:  # finished tool(s)
                    pending.remove(task)
                    name, call_id = task_info.pop(task)

                    try:
                        raw = task.result()
                        result = _dumps(raw, indent=4)
                        consecutive_failures = 0
                        if log_steps:
                            LOGGER.info(f"\n🛠️ {name} = {result}\n")
                    except Exception:
                        consecutive_failures += 1
                        result = traceback.format_exc()
                        if log_steps:
                            LOGGER.error(
                                f"\n❌ {name} failed "
                                f"(attempt {consecutive_failures}/{max_consecutive_failures}):\n{result}",
                            )

                    client.append_messages(
                        [
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": name,
                                "content": result,
                            }
                        ],
                    )

                    if consecutive_failures >= max_consecutive_failures:
                        if log_steps:
                            LOGGER.error("🚨 Aborting: too many tool failures.")
                        raise RuntimeError(
                            "Aborted after too many consecutive tool failures."
                        )

            # ── B: wait for remaining tools before asking the LLM again
            if pending:
                continue  # still waiting for other tool tasks

            #  (no pending tool calls → safe to inject new user input)
            while True:
                try:
                    extra = interject_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if log_steps:
                    LOGGER.info(f"\n⚡ Interjection → {extra!r}\n")
                client.append_messages([{"role": "user", "content": extra}])

            # ── C.  Cancel check before calling the LLM  ────────────────────
            if cancel_event.is_set():
                raise asyncio.CancelledError

            # ── D.  Ask the LLM what to do next  ────────────────────────────
            if log_steps:
                LOGGER.info("🔄 LLM thinking…")

            response = await client.generate(
                return_full_completion=True,
                tools=tools_schema,
                tool_choice="auto",
                stateful=True,
            )
            msg = response.choices[0].message

            # ── E.  Launch any new tool calls  ──────────────────────────────
            if msg.tool_calls:
                for call in msg.tool_calls:
                    name = call.function.name
                    args = json.loads(call.function.arguments)
                    fn = tools[name]
                    coro = (
                        fn(**args)
                        if asyncio.iscoroutinefunction(fn)
                        else asyncio.to_thread(fn, **args)
                    )

                    t = asyncio.create_task(coro)
                    pending.add(t)
                    task_info[t] = (name, call.id)

                if log_steps:
                    LOGGER.info("✅ Step finished (tool calls scheduled)")
                continue  # back to the top

            # ── F.  No new tool calls  ──────────────────────────────────────
            if pending:  # still waiting for others
                continue

            if log_steps:
                LOGGER.info(f"\n🤖 {msg.content}\n")
                LOGGER.info("✅ Step finished (final answer)")
            return msg.content  # DONE!

    except asyncio.CancelledError:  # graceful shutdown
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 2.  A tiny handle object exposed to callers
# ─────────────────────────────────────────────────────────────────────────────
class AsyncToolLoopHandle:
    """
    Returned by `start_async_tool_use_loop`.  Lets you
      • queue extra user messages while the loop runs and
      • stop the loop at any time.
    """

    def __init__(
        self,
        *,
        task: asyncio.Task,
        interject_queue: asyncio.Queue[str],
        cancel_event: asyncio.Event,
    ):
        self._task = task
        self._queue = interject_queue
        self._cancel_event = cancel_event

    # -- public API -----------------------------------------------------------
    async def interject(self, message: str) -> None:
        """Inject an additional *user* turn into the running conversation."""
        await self._queue.put(message)

    def stop(self) -> None:
        """Politely ask the loop to shut down (gracefully)."""
        self._cancel_event.set()

    # Optional helpers --------------------------------------------------------
    def done(self) -> bool:
        return self._task.done()

    async def result(self) -> str:
        """Wait for the assistant’s *final* reply."""
        return await self._task


# ─────────────────────────────────────────────────────────────────────────────
# 3.  A convenience wrapper that *starts* the loop and returns the handle
# ─────────────────────────────────────────────────────────────────────────────
def start_async_tool_use_loop(
    client: unify.AsyncUnify,
    message: str,
    tools: Dict[str, Callable],
    *,
    max_consecutive_failures: int = 3,
    log_steps: bool = False,
) -> AsyncToolLoopHandle:
    """
    Kick off `_async_tool_use_loop_inner` in its own task and give the caller
    a handle for live interaction.
    """
    interject_queue: asyncio.Queue[str] = asyncio.Queue()
    cancel_event = asyncio.Event()

    task = asyncio.create_task(
        _async_tool_use_loop_inner(
            client,
            message,
            tools,
            interject_queue=interject_queue,
            cancel_event=cancel_event,
            max_consecutive_failures=max_consecutive_failures,
            log_steps=log_steps,
        )
    )

    return AsyncToolLoopHandle(
        task=task,
        interject_queue=interject_queue,
        cancel_event=cancel_event,
    )

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/task_list_manager/types/task.py`:

```py
from pydantic import BaseModel, Field
from typing import Optional

from .priority import Priority
from .status import Status
from .schedule import Schedule
from .repetition import RepeatPattern


class Task(BaseModel):
    task_id: int = Field(description="Unique identifier for the task")
    name: str = Field(description="Short title of the task")
    description: str = Field(
        description="Detailed explanation of what the task involves",
    )
    status: Status = Field(
        description="Current state of the task (e.g., queued, active, completed)",
    )
    schedule: Schedule = Field(
        description="Information about task scheduling, including adjacent tasks in the queue and ideal start time",
    )
    deadline: Optional[str] = Field(
        description="Due date/time for the task in ISO-8601 format",
    )
    repeat: Optional[RepeatPattern] = Field(
        description="Pattern defining how the task recurs over time",
    )
    priority: Priority = Field(
        description="Importance level of the task (low, normal, high, urgent)",
    )

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/task_list_manager/types/priority.py`:

```py
from enum import StrEnum


class Priority(StrEnum):
    low = "low"
    normal = "normal"
    high = "high"
    urgent = "urgent"

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/task_list_manager/types/repetition.py`:

```py
"""
A strongly-typed, validation-first schema for describing how a task
repeats over time.  The model serialises cleanly to / from JSON so it can
be stored in `unify` logs without ambiguity.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class Frequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class Weekday(str, Enum):
    MO = "MO"
    TU = "TU"
    WE = "WE"
    TH = "TH"
    FR = "FR"
    SA = "SA"
    SU = "SU"


class RepeatPattern(BaseModel):
    """
    A very small subset of RFC-5545 RRULE expressed as first-class fields:

    * **frequency** – base unit of recurrence.
    * **interval**  – "every *n* units"; defaults to 1.
    * **weekdays**  – which days of the week (only when `frequency=weekly`).
    * **count**     – stop after *count* occurrences.
    * **until**     – or stop at this date/time (ISO-8601).

    Anything more elaborate can still be represented by creating multiple
    `RepeatPattern` instances for a single task.
    """

    frequency: Frequency = Field(..., description="Base unit of recurrence")
    interval: int = Field(
        1,
        ge=1,
        description="Number of frequency units between each repeat",
    )
    weekdays: Optional[List[Weekday]] = Field(
        None,
        description="Applicable only when frequency == weekly; " "ignored otherwise",
    )
    count: Optional[int] = Field(
        None,
        ge=1,
        description="Total number of occurrences before stopping",
    )
    until: Optional[datetime] = Field(
        None,
        description="Hard cut-off date/time after which no repeats occur",
    )

    # ------------------------------------------------------------------ #
    #  Validators                                                         #
    # ------------------------------------------------------------------ #

    @field_validator("weekdays")
    def _weekdays_only_for_weekly(cls, v, info):
        if v is not None and info.data.get("frequency") != Frequency.WEEKLY:
            raise ValueError("`weekdays` only makes sense with weekly frequency")
        return v

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/task_list_manager/types/schedule.py`:

```py
from typing import Optional
from pydantic import BaseModel, Field


class Schedule(BaseModel):
    next_task: Optional[int] = Field(
        description="ID of the next task in the sequence, used for task dependencies and ordering",
    )
    prev_task: Optional[int] = Field(
        description="ID of the previous task in the sequence, used for task dependencies and ordering",
    )
    start_time: Optional[str] = Field(
        default=None,
        description="The scheduled start time for the task in ISO-8601 format. Only set when the user explicitly schedules the task.",
    )

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/task_list_manager/types/status.py`:

```py
from enum import StrEnum


class Status(StrEnum):
    scheduled = "scheduled"
    queued = "queued"
    paused = "paused"
    active = "active"
    completed = "completed"
    cancelled = "cancelled"
    failed = "failed"

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/task_list_manager/sys_msgs.py`:

```py
import json
from .types.task import Task

UPDATE = f"""
Your task is to update the list of tasks based on the plain-text request from the user, and you should continue using the tools available until you are satisfied that the list of tasks has been updated correctly.

The current date and time is {"datetime"}.

As a recap, the schema for the table which stores the list of tasks is as follows:

{json.dumps(Task.model_json_schema(), indent=4)}
"""

ASK = f"""
You are a helpful assistant specialising in answering questions about a task list.
You have access to a number of read-only tools that let you inspect the current state
of tasks.  Use these tools as needed, step-by-step, until you are confident you
can answer the user's question accurately. Once done, respond with the final
answer only (no additional commentary).

The current date and time is {"datetime"}.

The schema for each task row is:

{json.dumps(Task.model_json_schema(), indent=4)}
"""

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/task_list_manager/task_list_manager.py`:

```py
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple

import unify

from ..common.embed_utils import EMBED_MODEL, ensure_vector_column
from ..common.llm_helpers import start_async_tool_use_loop, AsyncToolLoopHandle
from .types.status import Status
from .types.priority import Priority
from .types.schedule import Schedule
from .types.repetition import RepeatPattern
from .types.schedule import Schedule
from .types.status import Status
from .types.task import Task
from .sys_msgs import ASK


class TaskListManager:

    _VEC_TASK = "task_emb"

    def __init__(self, *, traced: bool = True) -> None:
        """
        Responsible for managing the list of tasks, updating the names, descriptions, schedules, repeating pattern and status of all tasks.

        Args:
            daemon (bool): Whether the thread should be a daemon thread.
        """

        self._ask_tools = {
            # Query-only helpers – safe, read-only operations
            self._search.__name__: self._search,
            self._search_similar.__name__: self._search_similar,
            self._get_task_queue.__name__: self._get_task_queue,
            self._get_active_task.__name__: self._get_active_task,
            self._get_paused_task.__name__: self._get_paused_task,
        }

        # Write-capable helpers – every mutating operation as well as the read-only ones.
        self._update_tools = {
            **self._ask_tools,
            # Creation / deletion
            self._create_task.__name__: self._create_task,
            self._delete_task.__name__: self._delete_task,
            # Status transitions
            self._pause.__name__: self._pause,
            self._continue.__name__: self._continue,
            self._cancel_tasks.__name__: self._cancel_tasks,
            # Queue manipulation
            self._update_task_queue.__name__: self._update_task_queue,
            # Attribute mutations
            self._update_task_name.__name__: self._update_task_name,
            self._update_task_description.__name__: self._update_task_description,
            self._update_task_status.__name__: self._update_task_status,
            self._update_task_start_at.__name__: self._update_task_start_at,
            self._update_task_deadline.__name__: self._update_task_deadline,
            self._update_task_repetition.__name__: self._update_task_repetition,
            self._update_task_priority.__name__: self._update_task_priority,
        }

        # Internal monotonically-increasing task-id counter.  We keep it local
        # to the manager to avoid an expensive scan across *all* logs every
        # time we create a task.  Initialised lazily on first use.
        self._next_id: Optional[int] = None

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a TaskListManager."
        self._ctx = f"{read_ctx}/Tasks" if read_ctx else "Tasks"

        if self._ctx not in unify.get_contexts():
            unify.create_context(self._ctx)
        # Add tracing
        if traced:
            self = unify.traced(self)

    # Public #
    # -------#

    # English-Text question

    def ask(
        self,
        text: str,
        *,
        return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,
    ) -> Union[
        "AsyncToolLoopHandle", Tuple["AsyncToolLoopHandle", List[Dict[str, Any]]]
    ]:
        """
        Handle any plain-text english question to ask something about the list of tasks.

        Args:
            text (str): The text-based question to ask about the task list.
            return_reasoning_steps (bool): Whether to return the reasoning steps for the update request.
            log_tool_steps (bool): Whether to log the steps taken by the tool.

        Returns:
            AsyncToolLoopHandle: A handle to the running conversation that supports:
                - await handle.result() - Get the final answer
                - await handle.interject(message) - Add a user message mid-conversation
                - handle.stop() - Gracefully cancel the conversation

            When return_reasoning_steps=True, returns a tuple of (handle, messages)
        """

        client = unify.AsyncUnify("o4-mini@openai", cache=True)
        client.set_system_message(
            ASK.replace(
                "{datetime}",
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            ),
        )
        handle = start_async_tool_use_loop(
            client,
            text,
            self._ask_tools,
            log_steps=log_tool_steps,
        )
        if return_reasoning_steps:
            return handle, client.messages
        return handle

    # English-Text update request

    def update(
        self,
        text: str,
        *,
        return_reasoning_steps: bool = False,
        log_tool_steps: bool = False,
    ) -> Union[
        "AsyncToolLoopHandle", Tuple["AsyncToolLoopHandle", List[Dict[str, Any]]]
    ]:
        """
        Handle any plain-text english command to update the list of tasks in some manner.

        Args:
            text (str): The text-based request to update the task list.
            return_reasoning_steps (bool): Whether to return the reasoning steps for the update request.
            log_tool_steps (bool): Whether to log the steps taken by the tool.

        Returns:
            AsyncToolLoopHandle: A handle to the running conversation that supports:
                - await handle.result() - Get the final answer
                - await handle.interject(message) - Add a user message mid-conversation
                - handle.stop() - Gracefully cancel the conversation

            When return_reasoning_steps=True, returns a tuple of (handle, messages)
        """
        from .sys_msgs import UPDATE

        client = unify.AsyncUnify("o4-mini@openai", cache=True)
        client.set_system_message(
            UPDATE.replace(
                "{datetime}",
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            ),
        )
        handle = start_async_tool_use_loop(
            client,
            text,
            self._update_tools,
            log_steps=log_tool_steps,
        )
        if return_reasoning_steps:
            return handle, client.messages
        return handle

    def _get_logs_by_task_ids(
        self,
        *,
        task_ids: Union[int, List[int]],
    ) -> List[unify.Log]:
        """
        Get the log for the specified task id.

        Args:
            task_ids (Union[int, List[int]]): The id or ids of the tasks to get the logs for.

        Returns:
            List[unify.Log]: The logs for the specified task ids.
        """
        singular = False
        if isinstance(task_ids, int):
            singular = True
            task_ids = [task_ids]
        log_ids = unify.get_logs(
            context=self._ctx,
            filter=f"task_id in {task_ids}",
            return_ids_only=True,
        )
        assert (
            not singular or len(log_ids) == 1
        ), f"Expected 1 log for singular task_id, but got {len(log_ids)}"
        return log_ids

    # Private #
    # --------#

    # Create

    def _create_task(
        self,
        *,
        name: str,
        description: str,
        status: Optional[Status] = None,
        schedule: Optional[Schedule] = None,
        deadline: Optional[str] = None,
        repeat: Optional[List[RepeatPattern]] = None,
        priority: Priority = Priority.normal,
    ) -> int:
        """
        Create a new task and – if appropriate – insert it into the
        runnable queue.

        Behaviour
        ---------
        • If *status* is **omitted** the method decides:
            – **active**     when no active task exists *and* either
                             no schedule is given or its start_time ≤ now.
            – **queued**     when an active task already exists.
            – **scheduled**  when a schedule.start_time > now.

        • If the caller supplies an explicit *status* we validate that it
          does not conflict with the current state (e.g. no 2nd active).

        • New queued / active tasks are appended (tail) or prepended (head)
          by re-using `_update_task_queue`.

        • Tasks whose `start_time` is in the future are **not** placed
          in the active queue.

        Raises
        ------
        ValueError for invalid combinations or duplicate name/description.
        """
        # ----------------  helper: iso-8601 → datetime  ---------------- #
        from datetime import datetime, timezone

        def _parse_iso(ts: str) -> datetime:
            dt = datetime.fromisoformat(ts)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        # ----------------  initial validation & dedup  ---------------- #
        if not name or not description:
            raise ValueError("Both 'name' and 'description' are required")

        # uniqueness (name / description)
        for key, value in {"name": name, "description": description}.items():
            clashes = unify.get_logs(
                context=self._ctx,
                filter=f"{key} == '{value}'",
                limit=1,
            )
            if clashes:
                raise ValueError(f"A task with {key!r} = {value!r} already exists")

        # ----------------------------------- #
        #  derive status when caller omitted   #
        # ----------------------------------- #
        if status is not None and isinstance(status, str):
            status = Status(status)

        active_task = self._get_active_task()

        # figure out if schedule is "future"
        future_start = False
        if schedule and schedule.start_time:
            future_start = _parse_iso(schedule.start_time) > datetime.now(timezone.utc)

        if status is None:
            if future_start:
                status = Status.scheduled
            elif active_task is None:
                status = Status.active
            else:
                status = Status.queued

        # ------------------  conflict checks  ------------------ #
        if status == Status.active and active_task is not None:
            raise ValueError("An active task already exists")

        if status == Status.active and future_start:
            raise ValueError("Cannot mark task as active with a future start_time")

        if status == Status.scheduled and not future_start:
            raise ValueError("Scheduled tasks require a future start_time")

        # ------------------  generate new task_id  ------------------ #
        # We avoid fetching *all* logs just to know the next id.  Instead we
        # maintain a simple counter that is initialised the first time we
        # create a task in this process by looking at the *largest* existing
        # id (if any) through a single, cheap query.

        if self._next_id is None:
            # First use – find the current maximum task_id (if any) with a
            # limited query.  The stubbed SDK doesn't expose sorting, so we
            # fall back to scanning just once during initialisation which is
            # acceptable in practise.
            existing = [lg.entries.get("task_id") for lg in unify.get_logs(context=self._ctx)]  # type: ignore[arg-type]
            existing = [i for i in existing if i is not None]
            self._next_id = (max(existing) + 1) if existing else 0

        next_id = self._next_id
        self._next_id += 1

        # ------------------  assemble payload  ------------------ #
        task_details = {
            "name": name,
            "description": description,
            "status": status,
            "schedule": schedule.model_dump() if schedule else None,
            "deadline": deadline,
            "repeat": [r.model_dump() for r in repeat] if repeat else None,
            "priority": priority,
        }

        # ------------------  write log immediately  ------------------ #
        unify.log(
            context=self._ctx,
            **task_details,
            task_id=next_id,
            new=True,
        )

        # ------------------  queue insertion (if relevant)  ---------- #
        if status in (Status.active, Status.queued):
            original_q = [t.task_id for t in self._get_task_queue()]

            # Only insert if the new task isn't already in that list
            if next_id not in original_q:
                new_q = (
                    [next_id] + original_q  # prepend for active
                    if status == Status.active
                    else original_q + [next_id]  # append for queued
                )
                self._update_task_queue(original=original_q, new=new_q)

        return next_id

    # Delete

    def _delete_task(
        self,
        *,
        task_id: int,
    ) -> Dict[str, str]:
        """
        Deletes the specified task from the task list.

        Args:
            task_id (int): The id of the task to delete.

        Returns:
            Dict[str, str]: Whether the task was deleted or not.
        """
        # ToDo: replace with single API call once this task [https://app.clickup.com/t/86c3c1awp] is done
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        unify.delete_logs(
            context=self._ctx,
            logs=log_id,
        )

    # Pause / Continue Active Task

    def _get_paused_task(self) -> Optional[Task]:
        """
        Get the currently paused task, if any.

        Returns:
            Optional[Task]: The complete Task object of the paused task, or None if no task is paused.
        """
        paused_tasks = self._search(filter="status == 'paused'")
        assert (
            len(paused_tasks) <= 1
        ), f"More than one paused task found: {paused_tasks}"
        if not paused_tasks:
            return
        return paused_tasks[0]

    def _get_active_task(self) -> Optional[Task]:
        """
        Get the currently active task, if any.

        Returns:
            Optional[Task]: The complete Task object of the active task, or None if no task is active.
        """
        active_tasks = self._search(filter="status == 'active'")
        assert (
            len(active_tasks) <= 1
        ), f"More than one active task found: {active_tasks}"
        if not active_tasks:
            return
        return active_tasks[0]

    def _pause(self) -> Optional[Dict[str, str]]:
        """
        Pause the currently active task, if any.

        Returns:
            Optional[Dict[str, str]]: The result of updating the task status, or None if no active task.
        """
        active_task = self._get_active_task()
        if not active_task:
            return
        return self._update_task_status(
            task_ids=active_task["task_id"],
            new_status="paused",
        )

    def _continue(self) -> Optional[Dict[str, str]]:
        """
        Continue the currently paused task, if any.

        Returns:
            Optional[Dict[str, str]]: The result of updating the task status, or None if no paused task.
        """
        paused_task = self._get_paused_task()
        if not paused_task:
            return
        return self._update_task_status(
            task_ids=paused_task["task_id"],
            new_status="active",
        )

    # Cancel Task(s)

    def _cancel_tasks(self, task_ids: List[int]) -> None:
        """
        Cancel the specified tasks.

        Args:
            task_ids (List[int]): The ids of the tasks to cancel.
        """
        completed_tasks = self._search(filter="status == 'completed'")
        completed_task_ids = [lg["task_id"] for lg in completed_tasks]
        assert not set(task_ids).intersection(
            set(completed_task_ids),
        ), f"Cannot cancel completed tasks. Attempted to cancel: {set(task_ids).intersection(set(completed_task_ids))}"
        self._update_task_status(task_ids=task_ids, new_status="cancelled")

    # Update Task Queue

    # --------------------  small helpers  -------------------- #
    @staticmethod
    def _sched_prev(sched):
        """Return *prev_task* from a Schedule *dict* / *model* / *None*."""
        if sched is None:
            return None
        if isinstance(sched, dict):
            return sched.get("prev_task")
        # assume pydantic Schedule
        return getattr(sched, "prev_task", None)

    @staticmethod
    def _sched_next(sched):
        """Return *next_task* (mirrors _sched_prev)."""
        if sched is None:
            return None
        if isinstance(sched, dict):
            return sched.get("next_task")
        return getattr(sched, "next_task", None)

    _TERMINAL_STATUSES = {"completed", "cancelled", "failed"}

    def _get_task_queue(
        self,
        task_id: Optional[int] = None,
    ) -> List[Task]:
        """
        Return the runnable task queue (head → tail).

        • If *task_id* is *None* we begin with **the single active task**
          (falling back to the queue head if there is no active task).
        • Tasks whose status is completed / cancelled / failed are *ignored*.
        • Only the nodes actually traversed are loaded from storage; we never
          materialise the entire task table in memory.
        """

        # ----------------  helpers  ---------------- #
        def _get_task_row(tid: int) -> Optional[dict]:
            """Fetch exactly one task row by id or return None."""
            rows = self._search(filter=f"task_id == {tid}", limit=1)
            return rows[0] if rows else None

        # ----------------  starting node  ---------------- #
        start_row: Optional[dict] = None

        if task_id is None:
            active = self._get_active_task()
            if active:
                start_row = active
                task_id = active["task_id"]

        if start_row is None and task_id is not None:
            start_row = _get_task_row(task_id)

        if start_row is None:
            # fall back to queue head: node with no prev_task and non-terminal status
            head_candidates = self._search(
                filter=(
                    "schedule is not None and \n                    status not in ('completed','cancelled','failed', 'scheduled') and \n                    schedule.get('prev_task') is None"
                ),
                limit=2,
            )
            assert head_candidates, f"Queue is malformed – no head found"
            assert (
                len(head_candidates) == 1
            ), f"Multiple heads detected: {head_candidates}"
            start_row = head_candidates[0]

        # ----------  not in queue yet? return empty list  ---------- #
        if start_row is not None and start_row["schedule"] is None:
            # Task exists but has no schedule pointers; therefore the
            # queue is currently empty.
            return []

        # ----------------  walk backwards to head  ---------------- #
        cur = start_row
        while True:
            prev_id = self._sched_prev(cur["schedule"])
            if prev_id is None:
                break
            prev_row = _get_task_row(prev_id)
            if prev_row is None:
                break  # broken link – treat cur as head
            cur = prev_row  # keep walking

        head_row = cur

        # ----------------  walk forwards collecting list  ---------------- #
        ordered: List[Task] = []
        cur = head_row
        while cur:
            if (
                cur["status"] not in self._TERMINAL_STATUSES
                and cur["status"] != "scheduled"
            ):
                ordered.append(Task(**cur))

            nxt_id = self._sched_next(cur["schedule"])
            if nxt_id is None:
                break

            # fetch the next node lazily
            cur = _get_task_row(nxt_id)
            # guard against broken links (missing row)
            if cur is None:
                break

        return ordered

    def _update_task_queue(
        self,
        *,
        original: List[int],
        new: List[int],
    ) -> None:
        """
        Re-write the queue so that its order matches the new order.

        Args:
            original (List[int]): The current queue order, used for validation.
            new (List[int]): The new queue order, which may include extra task IDs.

        * `original` must describe the *current* queue order; we use it
          only for validation.
        * `new` may be a pure re-ordering or may also include **extra**
          task-ids (inserting new tasks). Removing tasks is *not*
          allowed here – cancel them instead.

        For every task we update its ``schedule`` field so that the linked
        list stays consistent, using `None` for the head's `prev_task`
        and the tail's `next_task`.
        """
        # -------  sanity checks  -------
        assert len(set(original)) == len(
            original,
        ), f"'original' contains duplicates: {original}"
        assert len(set(new)) == len(new), f"'new' contains duplicates: {new}"
        assert set(original).issubset(
            set(new),
        ), f"update cannot remove existing tasks; cancel them first. Missing tasks: {set(original) - set(new)}"

        # -------  gather existing logs  -------
        existing_logs = {
            t["task_id"]: t for t in self._search() if t["schedule"] is not None
        }

        updates_per_log: Dict[int, Dict[str, Any]] = {}
        for idx, tid in enumerate(new):
            prev_tid = None if idx == 0 else new[idx - 1]
            next_tid = None if idx == len(new) - 1 else new[idx + 1]

            # keep an existing start_time; otherwise leave it unset
            start_ts = None
            if tid in existing_logs and existing_logs[tid]["schedule"]:
                start_ts = existing_logs[tid]["schedule"].get("start_time")

            sched_payload = {
                "prev_task": prev_tid,
                "next_task": next_tid,
            }

            # Only include *start_time* when we actually know one (i.e. when
            # the task was explicitly scheduled by the user).  For plain queue
            # insertions `start_ts` will be *None* and we leave the field
            # absent.
            if start_ts is not None:
                sched_payload["start_time"] = start_ts

            updates_per_log[tid] = {"schedule": sched_payload}

        # Persist
        for tid, payload in updates_per_log.items():
            log_ids = self._get_logs_by_task_ids(task_ids=tid)
            unify.update_logs(
                logs=log_ids,
                context=self._ctx,
                entries=payload,
                overwrite=True,
            )

    # Update Name / Description

    def _update_task_name(
        self,
        *,
        task_id: int,
        new_name: str,
    ) -> Dict[str, str]:
        """
        Update the name of the specified task.

        Args:
            task_id (int): The id of the task to update.
            new_name (str): The new name of the task.

        Returns:
            Dict[str, str]: Whether the task was updated or not.
        """
        # ToDo: replace with single API call once this task [https://app.clickup.com/t/86c3c1y63] is done
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        return unify.update_logs(
            logs=log_id,
            context=self._ctx,
            entries={"name": new_name},
            overwrite=True,
        )

    def _update_task_description(
        self,
        *,
        task_id: int,
        new_description: str,
    ) -> Dict[str, str]:
        """
        Update the description for the specified task.

        Args:
            task_id (int): The id of the task to update.
            new_description (str): The new description for the task.

        Returns:
            Dict[str, str]: Whether the task was updated or not.
        """
        # ToDo: replace with single API call once this task [https://app.clickup.com/t/86c3c1y63] is done
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        return unify.update_logs(
            logs=log_id,
            context=self._ctx,
            entries={"description": new_description},
            overwrite=True,
        )

    # Update Task(s) Status / Schedule / Deadline / Repetition / Priority

    def _update_task_status(
        self,
        *,
        task_ids: Union[int, List[int]],
        new_status: str,
    ) -> Dict[str, str]:
        """
        Update the status for the specified task(s).

        Args:
            task_ids (Union[int, List[int]]): The id or ids of the tasks to update.
            new_status (str): The new status for the task(s).

        Returns:
            Dict[str, str]: Whether the task(s) were updated or not.
        """
        # ToDo: replace with single API call once this task [https://app.clickup.com/t/86c3c1y63] is done
        log_ids = self._get_logs_by_task_ids(task_ids=task_ids)
        return unify.update_logs(
            logs=log_ids,
            context=self._ctx,
            entries={"status": new_status},
            overwrite=True,
        )

    def _update_task_start_at(
        self,
        *,
        task_id: int,
        new_start_at: datetime,
    ) -> Dict[str, str]:
        """
        Update the scheduled **start_time** for the specified task.

        This sets / overwrites the ``schedule['start_time']`` field while
        preserving any existing ``prev_task`` / ``next_task`` linkage.
        If the task did not have a schedule previously we create one with
        ``prev_task`` / ``next_task`` set to ``None`` so that the task is
        *not* implicitly inserted into the runnable queue.
        """
        log_id = self._get_logs_by_task_ids(task_ids=task_id)

        # Coerce to ISO-8601 string (Unify stores plain serialisable values)
        if isinstance(new_start_at, datetime):
            new_start_at = new_start_at.isoformat()

        # Fetch the current task row to preserve linkage information if present
        current_rows = self._search(filter=f"task_id == {task_id}", limit=1)
        current_sched = current_rows[0].get("schedule") if current_rows else None
        if current_sched is None:
            current_sched = {}

        # Preserve queue linkage if it exists, otherwise default to None
        sched_payload = {
            "prev_task": self._sched_prev(current_sched),
            "next_task": self._sched_next(current_sched),
            "start_time": new_start_at,
        }

        return unify.update_logs(
            logs=log_id,
            context=self._ctx,
            entries={"schedule": sched_payload},
            overwrite=True,
        )

    def _update_task_deadline(
        self,
        *,
        task_id: int,
        new_deadline: datetime,
    ) -> Dict[str, str]:
        """
        Update the deadline for the specified task.

        Args:
            task_id (int): The id of the task to update.
            new_deadline (datetime): The new deadline for the task.

        Returns:
            Dict[str, str]: Whether the task was updated or not.
        """
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        return unify.update_logs(
            logs=log_id,
            context=self._ctx,
            entries={"deadline": new_deadline},
            overwrite=True,
        )

    def _update_task_repetition(
        self,
        *,
        task_id: int,
        new_repeat: List[RepeatPattern],
    ) -> Dict[str, str]:
        """
        Update the repeat pattern for the specified task.

        Args:
            task_id (int): The id of the task to update.
            new_repeat (List[RepeatPattern]): The new repeat patterns for the task.

        Returns:
            Dict[str, str]: Whether the task was updated or not.
        """
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        return unify.update_logs(
            logs=log_id,
            context=self._ctx,
            entries={"repeat": [r.model_dump() for r in new_repeat]},
            overwrite=True,
        )

    def _update_task_priority(
        self,
        *,
        task_id: int,
        new_priority: Priority,
    ) -> Dict[str, str]:
        """
        Update the priority for the specified task.

        Args:
            task_id (int): The id of the task to update.
            new_priority (Priority): The new priority for the task.

        Returns:
            Dict[str, str]: Whether the task was updated or not.
        """
        log_id = self._get_logs_by_task_ids(task_ids=task_id)
        return unify.update_logs(
            logs=log_id,
            context=self._ctx,
            entries={"priority": new_priority},
            overwrite=True,
        )

    # Search Across Tasks

    def _bootstrap_embeddings(self) -> None:
        """
        Ensure that the vector embedding column exists for task search.
        Creates a derived column combining name and description for embedding.
        """
        expr = "str({name}) + ' || ' + str({description})"
        ensure_vector_column(
            context=self._ctx,
            embed_column=self._VEC_TASK,
            source_column="name_plus_desc",
            derived_expr=expr,
        )

    def _search_similar(
        self,
        *,
        text: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find tasks semantically similar to the provided text.

        Args:
            text (str): The text to find similar tasks to.
            k (int): The number of similar tasks to return.

        Returns:
            List[Dict[str, Any]]: A list where each item in the list is a dict representing a row in the table.
        """
        self._bootstrap_embeddings()
        return [
            log.entries
            for log in unify.get_logs(
                context=self._ctx,
                sorting={
                    f"cosine({self._VEC_TASK}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
                },
                limit=k,
            )
        ]

    def _search(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Apply the filter to the the list of tasks, and return the results following the filter.

        Args:
            filter (Optional[str]): Arbitrary Python logical expressions which evaluate to `bool`, with column names expressed as standard variables. For example, a filter expression of "'email'in description and priority == 'normal'" would be a valid. The expression just needs to be valid Python with the column names as variables.
            offset (int): The offset to start the search from, in the paginated result.
            limit (int): The number of rows to return, in the paginated result.

        Returns:
            List[Dict[str, Any]]: A list where each item in the list is a dict representing a row in the table.
        """
        return [
            log.entries
            for log in unify.get_logs(
                context=self._ctx,
                filter=filter,
                offset=offset,
                limit=limit,
            )
        ]

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/communication/types/message.py`:

```py
from enum import StrEnum
from pydantic import BaseModel


class Medium(StrEnum):
    SMS_MESSAGE = "sms_message"
    EMAIL = "email"
    WHATSAPP_MSG = "whatsapp_message"
    PHONE_CALL = "phone_call"
    WHATSAPP_CALL = "whatsapp_call"


class Message(BaseModel):
    medium: Medium
    sender_id: int
    receiver_id: int
    timestamp: str
    content: str
    exchange_id: int


VALID_MEDIA: tuple[str, ...] = tuple(m.value for m in Medium)

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/communication/types/summary.py`:

```py
from typing import List
from pydantic import BaseModel


class Summary(BaseModel):
    exchange_ids: List[int]
    summary: str

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/communication/types/contact.py`:

```py
from pydantic import BaseModel
from typing import Optional


class Contact(BaseModel):
    contact_id: int
    first_name: Optional[str]
    surname: Optional[str]
    email_address: Optional[str]
    phone_number: Optional[str]
    whatsapp_number: Optional[str]

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/communication/sys_msgs.py`:

```py
import os

AGENT_FIRST = os.environ["AGENT_FIRST"]
AGENT_LAST = os.environ["AGENT_LAST"]
AGENT_AGE = os.environ["AGENT_AGE"]
FIRST_NAME = os.environ["FIRST_NAME"]

AGENT_INFO = f"""
Your name is {AGENT_FIRST} {AGENT_LAST}, you are {AGENT_AGE} years old, and you are a helpful admin assistant, capable of achieving any task, in any domain of any difficulty level. If you are asked personal questions about your life, about your day, or about anything else, you should make up a response which feels believable, as though you were just like any other person. Please **do not** respond with robotic 'I am only an assistant' style responses. Your name is {AGENT_FIRST} {AGENT_LAST}, you are {AGENT_AGE} years old, and you have your own unique personality, make up answers to personal which are interesting, and also occassionally ask questions in return if you are being asked questions, make polite conversation. However, if {FIRST_NAME} is asking you to do tasks, it's best to focus on these and not distract them too much with back-to-back questions.
"""

PHONE_AGENT = (
    AGENT_INFO
    + f"""
You are assisting {FIRST_NAME}, can you can perform ANY TASK that {FIRST_NAME} requests for you to perform, so long as it can technically be achieved using a browser on a computer. You will be given ongoing information about the state of the ongoing tasks which you are completing in this chat. You won't select the actions yourself, but you will behave as though you are performing the actions yourself. You must then take this stream of information about the task progress, and use it to help answer any questions that the user has about the ongoing task being performed. If they ask you to perform any action during the task, just explain that yes you can do that, and then add a clear phrase such as "Let me just get that done now.....", "Give me a moment.....", but **never** announce that you have completed a task in your response. Long moments of silence are fine, whilst a task is being completed. The user will be informed by another means when the requested task has been performed.
"""
)

COMS_MANAGER = AGENT_INFO + ""

NEW_AGENT = (
    AGENT_INFO
    + f"""
You are assisting {FIRST_NAME}, can you can perform ANY TASK that {FIRST_NAME} requests for you to perform, so long as it can technically be achieved using a browser on a computer.

You have access to a browser agent, that can perform any task the user asks for on the browser.

Following is the pseudo code of the user flow you're supposed to follow:


1. User asks for doing something new on the browser (i.e. open a tab, search for something, click on something, etc.)
    - first check if there's a task in progress using the `is_task_running` tool
    - if there's a task in progress, you should refuse to create a new task, and ask the user to wait for the current task to complete (and prolly explore if there's other things the user asked for which don't require doing something new on the browser)
    - if there's no task in progress, you should use the `create_task` tool to create a new task

2. User doesn't ask for doing something new on the browser
    2.1 asks about the status of the current task (in terms of the steps completed so far)
        - use the `get_last_step_results` tool to get the current state of the task and the steps completed so far (oldest step first).

    2.2 asks to pause, update, resume or cancel the current task
        - if asked to pause the task, use the `pause_task` tool, inform them that the task will only resume if the user explicitly asks you to resume it, in which case you should first check if there's a task in progress. If there is, you should use the `resume_task` tool to resume it.
        - if asked to update the task, use the `create_task` tool which should behave like updating the task, and inform the user that the task will only resume if the user explicitly asks you to resume it, in which case you should first check if there's a task in progress. If there is, you should use the `resume_task` tool to resume it.
        - if asked to resume the task, use the `resume_task` tool
        - if asked to cancel the task, use the `cancel_task` tool

    2.3 asks about a previous task
        - use the `get_last_task_result` tool to get the result of the previous task

    2.4 asks a random question unrelated to the browser
        - answer the question as best as possible

"""
)

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/communication/transcript_manager/transcript_manager.py`:

```py
import json
from typing import List, Dict, Any, Optional, Union

import unify
from ...common.embed_utils import EMBED_MODEL, ensure_vector_column
from ...communication.types.contact import Contact
from ...communication.types.message import Message
from ...communication.types.summary import Summary
from ...common.llm_helpers import start_async_tool_use_loop, AsyncToolLoopHandle
from ...events.event_bus import EventBus, Event


class TranscriptManager:

    # Vector embedding column names
    _VEC_MSG = "content_emb"
    _VEC_SUM = "summary_emb"

    def __init__(self, event_bus: EventBus, *, traced: bool = True) -> None:
        """
        Responsible for *searching through* the full transcripts across all communcation channels exposed to the assistant.
        """
        self._event_bus = event_bus

        self._tools = {
            self.summarize.__name__: self.summarize,
            self._search_contacts.__name__: self._search_contacts,
            self._search_messages.__name__: self._search_messages,
            self._search_summaries.__name__: self._search_summaries,
            self._nearest_messages.__name__: self._nearest_messages,
        }

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a TranscriptManager."
        self._contacts_ctx = f"{read_ctx}/Contacts" if read_ctx else "Contacts"
        self._transcripts_ctx = self._event_bus.ctxs["Messages"]
        self._summaries_ctx = self._event_bus.ctxs["MessageExchangeSummaries"]
        if self._contacts_ctx not in unify.get_contexts():
            unify.create_context(self._contacts_ctx)

        # Add tracing
        if traced:
            self = unify.traced(self)

    # Public #
    # -------#

    # English-Text Question

    def ask(
        self, text: str, *, return_reasoning_steps: bool = False
    ) -> "AsyncToolLoopHandle":
        """
        Ask any question as a text command, and use the tools available (the private methods of this class) to perform the action.

        Args:
            text (str): The text-based question to answer.
            return_reasoning_steps (bool): Whether to return the reasoning steps along with the answer.

        Returns:
            AsyncToolLoopHandle: A handle to the running conversation that supports:
                - await handle.result(): Get the final answer when ready
                - await handle.interject(message): Add a new user message mid-conversation
                - handle.stop(): Gracefully terminate the conversation

        Usage:
            # Synchronous call that returns a handle immediately:
            handle = transcript_manager.ask("Find recent emails from John")

            # To get the final answer (must be awaited):
            answer = await handle.result()

            # If return_reasoning_steps=True:
            handle = transcript_manager.ask("Find emails from John", return_reasoning_steps=True)
            answer, reasoning_steps = await handle.result()

            # To add clarification mid-conversation:
            await handle.interject("I meant John Smith specifically")

            # To stop the conversation early:
            handle.stop()
        """
        from unity.communication.transcript_manager.sys_msgs import ANSWER

        client = unify.AsyncUnify("o4-mini@openai", cache=True)
        client.set_system_message(ANSWER)
        handle = start_async_tool_use_loop(client, text, self._tools)

        if return_reasoning_steps:
            # Wrap the handle.result() to return both answer and reasoning steps
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result

        return handle

    # Summarize Exchange(s)

    def summarize(
        self,
        *,
        exchange_ids: Union[int, List[int]],
        guidance: Optional[str] = None,
    ) -> str:
        """
        Summarize the email thread, phone call, or a time-clustered text exchange, save the summary in the backend, and also return it.

        Args:
            exchange_ids (int): The ids of the exchanges to summarize.
            guidance (Optional[str]): Optional guidance for the summarization.

        Returns:
            str: The summary of the exchanges.
        """
        from unity.communication.transcript_manager.sys_msgs import SUMMARIZE

        if not isinstance(exchange_ids, list):
            exchange_ids = [exchange_ids]
        client = unify.Unify("o4-mini@openai", cache=True)
        client.set_system_message(
            SUMMARIZE.replace("{guidance}", f"\n{guidance}\n" if guidance else ""),
        )
        msgs = self._search_messages(filter=f"exchange_id in {exchange_ids}")
        exchanges = {
            id: [msg.content for msg in msgs if msg.exchange_id == id]
            for id in exchange_ids
        }
        latest_timestamp = max([msg.timestamp for msg in msgs]).isoformat()
        summary = client.generate(json.dumps(exchanges, indent=4))
        self._event_bus.publish(
            Event(
                context="message_exchange_summary",
                timestamp=latest_timestamp,
                payload=summary,
            )
        )
        return summary

    def create_contact(
        self,
        *,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        whatsapp_number: Optional[str] = None,
    ) -> int:
        """
        Creates a new contact with the following contact details, as available.

        Args:
            first_name (str): The first name of the contact.
            surname (str): The surname of the contact.
            email_address (str): The email address of the contact.
            phone_number (str): The phone number of the contact.
            whatsapp_number (str): The WhatsApp number of the contact.
        Returns:
            int: The id of the newly created contact.
        """

        # Prune None values
        contact_details = {
            "first_name": first_name,
            "surname": surname,
            "email_address": email_address,
            "phone_number": phone_number,
            "whatsapp_number": whatsapp_number,
        }
        assert any(
            contact_details.values(),
        ), "At least one contact detail must be provided."

        # If it's the first contact, create immediately
        if not unify.get_logs(context=self._contacts_ctx):
            return unify.log(
                context=self._contacts_ctx,
                **contact_details,
                contact_id=0,
                new=True,
            ).id

        # Verify uniqueness
        for key, value in contact_details.items():
            if key in ["first_name", "surname"] or value is None:
                continue
            logs = unify.get_logs(
                context=self._contacts_ctx,
                filter=f"{key} == '{value}'",
            )
            assert (
                len(logs) == 0
            ), f"Invalid, contact with {key} {value} already exists."

        # ToDo: filter only for contact_id once supported in the Python utility function
        logs = unify.get_logs(
            context=self._contacts_ctx,
        )
        largest_id = max([lg.entries["contact_id"] for lg in logs])
        this_id = largest_id + 1

        # Create the new contact
        return unify.log(
            context=self._contacts_ctx,
            **contact_details,
            contact_id=this_id,
            new=True,
        ).id

    def update_contact(
        self,
        *,
        contact_id: int,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        whatsapp_number: Optional[str] = None,
    ) -> int:
        """
        Update the contact details of a contact.

        Args:
            contact_id (int): The id of the contact to update.
            first_name (Optional[str]): The first name of the contact.
            surname (Optional[str]): The surname of the contact.
            email_address (Optional[str]): The email address of the contact.
            phone_number (Optional[str]): The phone number of the contact.
            whatsapp_number (Optional[str]): The WhatsApp number of the contact.

        Returns:
            int: The id of the updated contact.
        """
        # Prune None values
        contact_details = {
            "first_name": first_name,
            "surname": surname,
            "email_address": email_address,
            "phone_number": phone_number,
            "whatsapp_number": whatsapp_number,
        }
        assert any(
            contact_details.values(),
        ), "At least one contact detail must be provided."

        # Verify uniqueness
        for key, value in contact_details.items():
            if key in ["first_name", "surname"] or value is None:
                continue
            logs = unify.get_logs(
                context=self._contacts_ctx,
                filter=f"{key} == '{value}'",
            )
            assert (
                len(logs) == 0
            ), f"Invalid, contact with {key} {value} already exists."

        # get log id
        logs = unify.get_logs(
            context=self._contacts_ctx,
            filter=f"contact_id == {contact_id}",
        )
        assert len(logs) == 1
        log: unify.Log = logs[0]
        log.update_entries(
            **contact_details,
            contact_id=contact_id,
        )

    # Private #
    # --------#
    def _nearest_messages(
        self,
        *,
        text: str,
        k: int = 10,
    ) -> List[Message]:
        """
        Find messages semantically similar to the provided text using vector embeddings.

        Args:
            text (str): The text to find similar messages to.
            k (int): The number of similar messages to return.

        Returns:
            List[Message]: A list of messages semantically similar to the provided text.
        """
        ensure_vector_column(self._transcripts_ctx, self._VEC_MSG, "content")
        logs = unify.get_logs(
            context=self._transcripts_ctx,
            sorting={
                f"cosine({self._VEC_MSG}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=k,
        )
        return [Message(**lg.entries) for lg in logs]

    def _nearest_summaries(
        self,
        *,
        text: str,
        k: int = 10,
    ) -> List[Summary]:
        """
        Find summaries semantically similar to the provided text using vector embeddings.

        Args:
            text (str): The text to find similar summaries to.
            k (int): The number of similar summaries to return.

        Returns:
            List[Summary]: A list of summaries semantically similar to the provided text.
        """

        ensure_vector_column(self._transcripts_ctx, self._VEC_MSG, "content")
        logs = unify.get_logs(
            context=self._summaries_ctx,
            sorting={
                f"cosine({self._VEC_SUM}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=k,
        )
        return [Summary(**lg.entries) for lg in logs]

    def _search_contacts(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, str]]:
        """
        Retrieve contact details, based on flexible filtering for first name, surname, email address, WhatsApp number, phone number, or anything else.

        Args:
            filter (Optional[str]): The filter to apply to the contacts.
            offset (int): The offset to start the retrieval from.
            limit (int): The maximum number of contacts to retrieve.

        Returns:
            List[Dict[str, str]]: A list of contacts.
        """
        logs = unify.get_logs(
            context=self._contacts_ctx,
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [Contact(**lg.entries) for lg in logs]

    def _search_messages(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, str]]:
        """
        Retrieve messages from the transcript history, based on flexible filtering for a specific sender, group of senders, receiver, group of receivers, medium, set of mediums, timestamp range, message length, messages containing a phrase, not containing a phrase, or anything else.

        Args:
            filter (Optional[str]): The filter to apply to the messages.
            offset (int): The offset to start the retrieval from.
            limit (int): The maximum number of messages to retrieve.

        Returns:
            List[Dict[str, str]]: A list of messages.
        """
        logs = unify.get_logs(
            context=self._transcripts_ctx,
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [Message(**lg.entries) for lg in logs]

    def _search_summaries(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, str]]:
        """
        Retrieve summaries from the transcript history, based on flexible filtering for a specific exchange id, group of exchange ids, medium, set of mediums, timestamp range, summary length, summaries containing a phrase, not containing a phrase, or anything else.

        Args:
            filter (Optional[str]): The filter to apply to the summaries.
            offset (int): The offset to start the retrieval from.
            limit (int): The maximum number of summaries to retrieve.

        Returns:
            List[Dict[str, str]]: A list of exchange summaries.
        """
        logs = unify.get_logs(
            context=self._summaries_ctx,
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [Summary(**lg.entries) for lg in logs]

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/communication/transcript_manager/sys_msgs.py`:

```py
import json

from ...communication.transcript_manager.transcript_manager import TranscriptManager
from ...communication.types.contact import Contact
from ...communication.types.message import Message
from ...communication.types.summary import Summary
from ...communication.transcript_manager.transcript_manager import TranscriptManager

SUMMARIZE = """
You will be given a series of exchanges, and you need to summarize these exchanges, based on the following guidance.
{guidance}
Please extract the most important information across all of the exchanges, without preferential treatment to any one of them.
"""

ANSWER = f"""
Your task is to answer the user question, and you should continue using the tools available until you are satisfied that you either have the correct answer, or you are confident it cannot be answered correctly. Firstly, you can summarize any exchange or group of exchanges, creating an overall explanatory paragraphs of said exchange(s). This tool is straightforward to use. You can also search contacts, messages and summaries (which you may or may not have created yourself). All three search tools include pagination with `offset` and `limit` to control the number of returned items and their offset in the list, and all include `filter` which accepts arbitrary Python logical expressions which evaluate to `bool`, and can include any of the relevant `Message` or `Summary` fields in the expressions (depending on the tool).

As a recap, the schemas for contacts, messages and summaries are as follows:

{json.dumps(Contact.model_json_schema(), indent=4)}

{json.dumps(Message.model_json_schema(), indent=4)}

{json.dumps(Summary.model_json_schema(), indent=4)}

Available tools:
• {TranscriptManager.summarize.__name__}(exchange_ids, guidance?): summarise one or more exchanges.
• {TranscriptManager._search_contacts.__name__.lstrip("_")}(filter?, offset=0, limit=100) → List[Contact] – flexible boolean filtering.
• {TranscriptManager._search_messages.__name__.lstrip("_")}(filter?, offset=0, limit=100) → List[Message] – flexible boolean filtering.
• {TranscriptManager._search_summaries.__name__.lstrip("_")}(filter?, offset=0, limit=100) → List[Summary] – flexible boolean filtering.
• {TranscriptManager._nearest_messages.__name__.lstrip("_")}(text: str, k: int = 10) → List[Message] – returns the top-k messages semantically similar to the given text.

Example usage:
# Find top-3 messages semantically similar to "banking and budgeting"
nearest_messages(text="banking and budgeting", k=3)

Some example filter expressions (`filter: str`) for the tools are as follows.

{TranscriptManager._search_contacts.__name__.lstrip("_")}:

- Sender's first name is John:  `filter="first_name == 'John'"`
- email address is gmail: `filter="'@gmail' in email"`
- WhatsApp number is american: `filter="'+1' in whatsapp_number"`
- Surname begins with "L": `filter="surname[0] == 'L'"`
- Flexible logical expressions and nesting. John L or has gmail: `filter="(first_name == 'John' and surname[0] == 'L') or '@gmail' in email"`

{TranscriptManager._search_messages.__name__.lstrip("_")}:

- Sender contact id is even:  `filter="contact_id % 2 == 0"`
- Medium is email: `filter="medium == 'email'"`
- Medium is email or whatsapp message: `filter="medium in ['email', 'whatsapp_message']"`
- Message contains the phrase Hello: `filter="'Hello' in content"`
- Flexible logical expressions and nesting. Email Greeting from contact 0: `filter="(('Hello' in content) or ('Goodbye' in content)) and medium == 'email' and contact_id == 0"`

{TranscriptManager._search_summaries.__name__.lstrip("_")}:

- Summary includes the substrings "sale" and "stapler":  `filter="sale" in summary and "stapler" in summary"`
- Summary includes either exchange id 0 or 1: `filter="0 in exchange_ids or 1 in exchange_ids"`
- Flexible logical expressions and nesting. Exchange id 0 or 1 and "sale" or "stapler" in summary: `filter="(0 in exchange_ids or 1 in exchange_ids) and ("sale" in summary or "stapler" in summary")"`

Remember that while filter-based search is useful for exact matches, the `nearest_messages` tool is more effective for finding semantically related content when you don't know the exact wording.
"""

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/events/types/message.py`:

```py
from enum import StrEnum
from pydantic import BaseModel


class Medium(StrEnum):
    SMS_MESSAGE = "sms_message"
    EMAIL = "email"
    WHATSAPP_MSG = "whatsapp_message"
    PHONE_CALL = "phone_call"
    WHATSAPP_CALL = "whatsapp_call"


class Message(BaseModel):
    medium: Medium
    sender_id: int
    receiver_id: int
    timestamp: str
    content: str
    exchange_id: int


VALID_MEDIA: tuple[str, ...] = tuple(m.value for m in Medium)

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/events/types/message_exchange_summary.py`:

```py
from typing import List
from pydantic import BaseModel


class MessageExchangeSummary(BaseModel):
    exchange_ids: List[int]
    summary: str

```

`/private/var/folders/lw/t60hd8gx56x1098px9tf9k8h0000gn/T/tmp.XrIc9Z1RTx/events/event_bus.py`:

```py
"""In‑process, asyncio‑friendly event stream **prefilled from Unify logs** and
restricted to Pydantic payload types declared in *events/types/*.
"""

from __future__ import annotations

import unify
import asyncio
import datetime as dt
from collections import deque
from typing import Deque, Dict, Iterable, Optional, Type
from pydantic import BaseModel, ValidationError

from .types.message import Message
from .types.message_exchange_summary import MessageExchangeSummary

__all__ = ["Event", "EventBus", "Subscription"]


_EVENT_TYPES: Dict[str, Type[BaseModel]] = {
    "Messages": Message,
    "MessageExchangeSummaries": MessageExchangeSummary,
}
_DEFAULT_WINDOW = 50

# ───────────────────────────   Event envelope   ─────────────────────────────


class Event(BaseModel):
    context: str
    timestamp: str
    payload: BaseModel


# ───────────────────────────   EventBus singleton   ─────────────────────────


class EventBus:

    def __init__(self, windows_sizes: Dict[str, int] = {}):

        # private attributes
        self._deques: Dict[str, Deque[Event]] = {}
        self._window_sizes: Dict[str, int] = {
            k: windows_sizes.get(k, _DEFAULT_WINDOW) for k in _EVENT_TYPES.keys()
        }
        self._lock = asyncio.Lock()

        # ── Unify setup ────────────────────────────────────────────────
        active_ctx = unify.get_active_context()
        base_ctx = active_ctx["write"]
        self._global_ctx = f"{base_ctx}/Events" if base_ctx else "Events"
        upstream_ctxs = unify.get_contexts()
        if self._global_ctx not in upstream_ctxs:
            unify.create_context(self._global_ctx)
        self._ctxs = {etype: f"{self._global_ctx}/{etype}" for etype in _EVENT_TYPES}
        for ctx in self._ctxs.values():
            if ctx not in upstream_ctxs:
                unify.create_context(ctx)
        self._logger = unify.AsyncLoggerManager()

        # ── Hydrate in‑memory windows from persisted logs ─────────────
        self._prefill_from_unify()

    # ------------------------------------------------------------------
    def _prefill_from_unify(self):
        """Populate each per‑type deque with newest logs from Unify."""
        for etype, model_cls in _EVENT_TYPES.items():
            window_size = self._window_sizes[etype]
            raw_logs = unify.get_logs(context=self._ctxs[etype], limit=window_size)
            # unify returns most‑recent‑first – reverse for chronological order
            dq: Deque[Event] = deque(maxlen=window_size)
            for log in reversed(raw_logs):
                entries = log.entries
                if entries is None:
                    continue
                try:
                    evt = Event.model_validate(entries)
                    if not isinstance(evt.payload, model_cls):
                        continue
                except ValidationError:
                    if isinstance(entries, model_cls):
                        ts = getattr(log, "ts", dt.datetime.now(dt.UTC)).isoformat()
                        evt = Event(context=etype, timestamp=ts, payload=entries)
                    else:
                        raise Exception("")
                dq.append(evt)
            self._deques[etype] = dq

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def publish(self, event: Event) -> None:
        window = self._window_sizes[event.context]

        async with self._lock:
            dq = self._deques[event.context]
            dq.append(event)
            while len(dq) > window:
                dq.popleft()

        # Log to global event table
        self._logger.log_create(
            project=unify.active_project(),
            context=self._global_ctx,
            params={},
            entries=event,
        )

        # Log to specific event table
        self._logger.log_create(
            project=unify.active_project(),
            context=self._ctxs[event.context],
            params={},
            entries=event.payload,
        )

    def join_published(self):
        """Ensures all published events have been uploaded"""
        self._logger.join()

    async def get_latest(
        self,
        types: Iterable[str] | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """
        Return up to *limit* events drawn from the specified *types*
        (or from *all* types if None), ordered **newest-first**.

        Always works with the in-memory deques; does not mutate them.
        """
        async with self._lock:
            wanted = set(types) if types is not None else self._deques.keys()

            # 1. collect (usually small) piles of events
            bucket: list[Event] = []
            for t in wanted:
                dq = self._deques.get(t)
                if dq:
                    bucket.extend(dq)  # each dq is already window-bounded

            # 2. sort newest→oldest and slice
            bucket.sort(key=lambda e: e.timestamp, reverse=True)
            return bucket[:limit]

    @property
    def ctxs(self):
        return self._ctxs

```