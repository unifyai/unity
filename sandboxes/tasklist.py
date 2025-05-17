"""tasklist_sandbox.py  (voice mode, Deepgram SDK v4, sync)
==============================================================
Single‑file interactive sandbox for **TaskListManager** with a simple
voice mode that relies *only* on Deepgram (v4) for STT and Cartesia for
TTS.  No LiveKit Agents and no async gymnastics.

Voice‑mode flow
---------------
1. **Press ↵** → speak → **press ↵** to stop.  Audio captured via PortAudio
   (`pyaudio`).
2. WAV bytes are sent to **Deepgram SDK v4** (`listen.rest.v('1').transcribe_file`).
   Transcript is printed as though typed.
3. Script immediately speaks *“Working on this now…”* with **Cartesia TTS**.
4. After TaskListManager finishes, the full answer is printed **and** read
   aloud.

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
import sys
import threading
import wave
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pyaudio
import asyncio  # cross‑platform audio I/O (links to PortAudio)
from dotenv import load_dotenv
from livekit.plugins import cartesia  # TTS only
from deepgram import DeepgramClient, PrerecordedOptions, FileSource  # SDK v4

load_dotenv()

# ---------------------------------------------------------------------------
# Local project imports (TaskListManager & helpers)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unify
from constants import LOGGER as _LG
from task_list_manager.task_list_manager import TaskListManager
from task_list_manager.types.priority import Priority
from task_list_manager.types.schedule import Schedule
from tests.test_task_list.test_update_text_complex import _next_weekday

# ---------------------------------------------------------------------------
# Utility functions (project name, seeding, dispatch) – mostly unchanged
# ---------------------------------------------------------------------------


def _generate_project_name(scenario_type: str, theme: Optional[str] = None) -> str:
    timestamp = datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    base = (
        "SimpleTaskList"
        if scenario_type == "fixed"
        else (theme or "LLMGeneratedTaskList")
    )
    base = "".join(c for c in base if c.isalnum() or c in [" ", "-", "_"]).replace(
        " ",
        "_",
    )
    return f"{base}/{timestamp}"


def _seed_fixed(tlm: TaskListManager) -> None:
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
        description="Roll out version 2.0 to production servers.",
        status="paused",
    )


def _seed_llm(tlm: TaskListManager) -> Optional[str]:
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

    for g_name, g in groups.items():
        for idx, _ in enumerate(g):
            cur = id_map[(g_name, idx)]
            prev_ = id_map.get((g_name, idx - 1)) if idx > 0 else None
            next_ = id_map.get((g_name, idx + 1)) if idx < len(g) - 1 else None
            tlm._update_task_status(
                task_ids=cur,
                new_status=tlm._search(filter=f"task_id=={cur}")[0]["status"],
            )
            unify.update_logs(
                context="Tasks",
                logs=tlm._get_logs_by_task_ids(task_ids=cur),
                entries={"schedule": {"prev_task": prev_, "next_task": next_}},
                overwrite=True,
            )
    return theme


def _dispatch(
    tlm: TaskListManager,
    raw: str,
    *,
    show_steps: bool,
) -> Tuple[str, str, List | None]:
    raw = raw.strip()
    if raw.lower().startswith("ask:"):
        ans, steps = tlm.ask(
            text=raw[4:].strip(),
            return_reasoning_steps=show_steps,
            log_tool_steps=show_steps,
        )
        return "ask", ans, steps
    if raw.lower().startswith("update:"):
        ans, steps = tlm.update(
            text=raw[7:].strip(),
            return_reasoning_steps=show_steps,
            log_tool_steps=show_steps,
        )
        return "update", ans, steps
    if raw.endswith("?"):
        ans, steps = tlm.ask(
            text=raw,
            return_reasoning_steps=show_steps,
            log_tool_steps=show_steps,
        )
        return "ask", ans, steps
    ans, steps = tlm.update(
        text=raw,
        return_reasoning_steps=show_steps,
        log_tool_steps=show_steps,
    )
    return "update", ans, steps


# ---------------------------------------------------------------------------
# Voice‑mode helpers (audio capture, STT, TTS)
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
_CHUNK = 1024
_FORMAT = pyaudio.paInt16
_CHANNELS = 1


def _record_until_enter() -> bytes:
    """Record between two ENTER presses and return WAV bytes."""
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
    """Speak *text* via Cartesia TTS – manages its own aiohttp session."""
    key = os.getenv("CARTESIA_API_KEY")
    if not key:
        return  # silently skip if no key

    async def _gen() -> bytes:
        import aiohttp  # local to avoid hard dep when not in voice mode

        async with aiohttp.ClientSession() as sess:
            tts = cartesia.TTS(
                http_session=sess,
            )  # hand‑rolled session sidesteps worker context
            stream = tts.synthesize(text)
            frame = await stream.collect()
            return frame.to_wav_bytes()

    wav_bytes = asyncio.run(_gen())

    duration = len(wav_bytes) / (24000 * 2)
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream.write(wav_bytes)
    stream.stop_stream()
    time.sleep(max(0, duration - stream.get_output_latency()))
    stream.close()
    pa.terminate()


def main() -> None:
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
    args = parser.parse_args()

    if not args.silent:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        _LG.setLevel(logging.INFO)
        for noisy in ("unify", "unify.utils", "unify.logging", "requests", "httpx"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    project = _generate_project_name(args.scenario)
    unify.activate(project)
    unify.set_context("Tasks", overwrite=args.new)

    tlm = TaskListManager()
    tlm.start()

    if args.new:
        if args.scenario == "llm":
            theme = _seed_llm(tlm)
            if theme:
                unify.activate(_generate_project_name("llm", theme))
        else:
            _seed_fixed(tlm)

    print(
        "TaskListManager sandbox – speak or type. Prefix with 'ask:'/'update:'. 'quit' to exit.\n",
    )

    if args.voice:
        while True:
            audio_bytes = _record_until_enter()
            user_text = _transcribe_deepgram(audio_bytes).strip()
            if not user_text:
                continue
            print(f"▶️  {user_text}")
            if user_text.lower() in {"quit", "exit"}:
                break
            _speak("Working on this now")
            kind, result, _ = _dispatch(tlm, user_text, show_steps=not args.silent)
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
                kind, result, _ = _dispatch(tlm, line, show_steps=not args.silent)
                print(f"[{kind}] => {result}\n")
        except (EOFError, KeyboardInterrupt):
            print()


if __name__ == "__main__":
    main()
