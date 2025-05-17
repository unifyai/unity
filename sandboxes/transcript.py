"""transcript_sandbox.py  (voice mode, Deepgram SDK v4, sync)
================================================================
Single‑file interactive sandbox for **TranscriptManager** with a
simple voice mode that relies *only* on Deepgram (v4) for STT and
Cartesia for TTS.  No LiveKit Agents and no async gymnastics.

This mirrors **tasklist_sandbox.py** but focuses on conversational
history: contacts, multi‑party exchanges, search/QA and summarisation.

Voice‑mode flow
---------------
1. **Press ↵** → speak → **press ↵** to stop.  Audio captured via
   PortAudio (`pyaudio`).
2. WAV bytes are sent to **Deepgram SDK v4** (`listen.rest.v('1').transcribe_file`).
   Transcript is printed as though typed.
3. Script immediately speaks *“Working on this now…”* with **Cartesia TTS**.
4. After TranscriptManager finishes, the full answer is printed **and**
   read aloud.

Environment variables
---------------------
* `DEEPGRAM_API_KEY`  – required for STT
* `CARTESIA_API_KEY`  – required for TTS (audio playback is skipped if
  missing)

Extra dependencies
------------------
```bash
pip install deepgram-sdk>=4 cartesia-sdk pyaudio
```
(Install `portaudio` dev libs from your package manager if `pyaudio` build
fails.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import threading
import aiohttp
import wave
from contextlib import contextmanager
from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import asyncio  # cross‑platform audio I/O (links to PortAudio)
import pyaudio
from livekit.plugins import cartesia  # TTS only
from dotenv import load_dotenv
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Local project imports (TranscriptManager & helpers)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unify
from constants import LOGGER as _LG
from communication.transcript_manager.transcript_manager import TranscriptManager
from communication.types.message import Message

# ---------------------------------------------------------------------------
# Utility functions (seeding, dispatch) – adapted for TranscriptManager
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
    # A couple of extra contacts to make things interesting
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
    """Populate contacts plus an involved web of 10 exchanges."""

    # --- contacts
    for idx, c in enumerate(_CONTACTS):
        tm.create_contact(**c)
        _ID_BY_NAME[c["first_name"].lower()] = idx

    base = datetime(2025, 4, 20, 9, 0, tzinfo=timezone.utc)

    def _cid(name: str) -> int:
        return _ID_BY_NAME[name]

    ex_id = 0

    # Exchange 0 – Dan ⇄ Julia phone call (early)
    tm.log_messages(
        [
            Message(
                medium="phone_call",
                sender_id=_cid("dan"),
                receiver_id=_cid("julia"),
                timestamp=(base).isoformat(),
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
        ]
    )
    ex_id += 1

    # Exchange 1 – Dan ⇄ Julia phone call (later)
    later = base + timedelta(days=6, hours=0, minutes=30)  # 2025‑04‑26 09:30Z
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
        ]
    )
    ex_id += 1

    # Exchange 2 – Carlos interest email chain
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
        ]
    )
    ex_id += 1

    # Exchange 3 – Jimmy holiday WhatsApp
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
            )
        ]
    )
    ex_id += 1

    # Exchange 4 – Anne passport excuse
    t_excuse = base + timedelta(days=3, hours=0)
    tm.log_messages(
        [
            Message(
                medium="whatsapp_message",
                sender_id=_cid("anne"),
                receiver_id=_cid("dan"),
                timestamp=t_excuse.isoformat(),
                content="Sorry Dan, I *can't join the Berlin trip because my passport expired* last week.",
                exchange_id=ex_id,
            )
        ]
    )
    ex_id += 1

    # Exchange 5 – Leo logistics SMS
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
        ]
    )
    ex_id += 1

    # Exchange 6 – Sara WhatsApp voice note (represented as text)
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
            )
        ]
    )
    ex_id += 1

    # More filler exchanges for realism
    random.seed(42)
    media = ["email", "phone_call", "sms_message", "whatsapp_message"]
    start = base + timedelta(days=6)
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
                    timestamp=(start + timedelta(minutes=extra * 4 + i)).isoformat(),
                    content=f"Filler {ex_ref}-{i} {mid} banter.",
                    exchange_id=ex_ref,
                )
            )
        tm.log_messages(msgs)

    # Seed one stored summary to imitate production state
    tm.summarize(exchange_ids=[0, 1])


def _seed_llm(tm: TranscriptManager) -> Optional[str]:
    """Use an LLM to generate an elaborate multi‑contact scenario.

    The expected JSON schema (top‑level keys):
        * contacts – list of contacts with the same fields as _CONTACTS
        * exchanges – list of exchanges, each with:
            - medium (email / phone_call / sms_message / whatsapp_message)
            - messages (ordered list of dicts with sender_first_name,
              receiver_first_name, timestamp ISO, content)
            - optional summary
        * theme (optional) – free‑text scenario label
    """
    prompt = (
        "Create a realistic communication history for a mid‑size European event "
        "agency with 8‑12 contacts interacting via email, phone_call, "
        "sms_message and whatsapp_message. Include 60‑90 exchanges with 3‑8 "
        "messages each, timestamps ranging from 2025‑04‑01 to 2025‑05‑10 (UTC). "
        "Return JSON with keys: contacts, exchanges, theme. Contacts must have "
        "first_name, surname, email_address, phone_number, whatsapp_number. "
        "Each exchange has medium and messages with sender_first_name, "
        "receiver_first_name, timestamp and content. Keep text concise."
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

    contacts = payload["contacts"]
    for idx, c in enumerate(contacts):
        tm.create_contact(**c)
        _ID_BY_NAME[c["first_name"].lower()] = idx

    for ex_id, ex in enumerate(payload["exchanges"]):
        mid = ex.get("medium", "email")
        msgs: List[Message] = []
        for m in ex["messages"]:
            msgs.append(
                Message(
                    medium=mid,
                    sender_id=_ID_BY_NAME[m["sender_first_name"].lower()],
                    receiver_id=_ID_BY_NAME[m["receiver_first_name"].lower()],
                    timestamp=m["timestamp"],
                    content=m["content"],
                    exchange_id=ex_id,
                )
            )
        tm.log_messages(msgs)

    # Optionally store summaries if provided
    if summaries := [i.get("summary") for i in payload["exchanges"] if i.get("summary")]:
        tm.summarize(texts=summaries)

    return payload.get("theme")


# ---------------------------------------------------------------------------
# Natural‑language dispatch helper
# ---------------------------------------------------------------------------

def _dispatch(tm: TranscriptManager, raw: str, *, show_steps: bool) -> Tuple[str, str, List | None]:
    """Very light dispatcher: answers all queries via `.ask()`.

    Future: could incorporate intent classification (eg. logging a new
    message vs asking). For now anything typed/spoken is treated as a
    question about stored transcripts.
    """
    raw = raw.strip()

    ans, steps = tm.ask(raw, return_reasoning_steps=show_steps)
    return "ask", ans, steps


# ---------------------------------------------------------------------------
# Voice‑mode helpers (audio capture, STT, TTS) – identical to tasklist version
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
_CHUNK = 1024
_FORMAT = pyaudio.paInt16
_CHANNELS = 1

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)


def py_error_handler(filename, line, function, err, fmt):  # noqa: D401 – C callback
    pass


c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)


@contextmanager
def noalsaerr():
    try:
        asound = cdll.LoadLibrary("libasound.so")
        asound.snd_lib_error_set_handler(c_error_handler)
        yield
        asound.snd_lib_error_set_handler(None)
    except Exception:
        yield


def _record_until_enter() -> bytes:
    """Record between two ENTER presses and return WAV bytes."""
    with noalsaerr():
        pa = pyaudio.PyAudio()
    frames: List[bytes] = []
    stream = pa.open(
        format=_FORMAT,
        channels=_CHANNELS,
        rate=_SAMPLE_RATE,
        input=True,
        frames_per_buffer=_CHUNK,
    )

    def _capture():
        while not stop.is_set():
            frames.append(stream.read(_CHUNK, exception_on_overflow=False))

    stop = threading.Event()
    thr = threading.Thread(target=_capture, daemon=True)

    input("\nPress ↵ to start recording…")
    print("🎙️  Recording… press ↵ again to stop.")
    thr.start()
    input()
    stop.set()
    thr.join()

    stream.stop_stream()
    stream.close()
    pa.terminate()

    wav_path = "/tmp/voice_input.wav"
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(pa.get_sample_size(_FORMAT))
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    with open(wav_path, "rb") as f:
        return f.read()


def _transcribe_deepgram(audio_bytes: bytes) -> str:
    """Synchronously transcribe WAV bytes using Deepgram SDK v4."""
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        print("[Voice] Deepgram key missing – falling back to CLI input.")
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


def _speak(text: str):
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TranscriptManager sandbox with minimalist voice mode (Deepgram v4, Cartesia)",
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
        help="include all logs for unify, requests and httpx",
    )
    args = parser.parse_args()

    # --- logging setup
    if not args.silent:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        _LG.setLevel(logging.INFO)
        if not args.debug:
            for noisy in ("unify", "unify.utils", "unify.logging", "requests", "httpx"):
                logging.getLogger(noisy).setLevel(logging.WARNING)

    # --- Unify project context
    unify.activate("TranscriptSandbox")
    new_project = "Messages" not in unify.get_contexts() or args.new
    unify.set_context("Messages", overwrite=new_project)

    # --- Initialise manager & seed data
    tm = TranscriptManager()
    tm.start()

    if new_project:
        if args.scenario == "llm":
            theme = _seed_llm(tm)
            if theme:
                _LG.info(f"[Seed] LLM scenario theme: {theme}")
        else:
            _seed_fixed(tm)

    print("TranscriptManager sandbox – speak or type. 'quit' to exit.\n")

    if args.voice:
        while True:
            audio_bytes = _record_until_enter()
            user_text = _transcribe_deepgram(audio_bytes).strip()
            if not user_text:
                continue
            print(f"▶️  {user_text}")
            if user_text.lower() in {"quit", "exit"}:
                break
            _speak("Working on this now…")
            kind, result, _ = _dispatch(tm, user_text, show_steps=not args.silent)
            print(f"[{kind}] => {result}\n")
            _speak(result)
    else:
        try:
            while True:
                line = input("> ").strip()
                if line.lower() in {"quit", "exit"}:
                    break
                if not line:
                    continue
                kind, result, _ = _dispatch(tm, line, show_steps=not args.silent)
                print(f"[{kind}] => {result}\n")
        except (EOFError, KeyboardInterrupt):
            print()


if __name__ == "__main__":
    main()
