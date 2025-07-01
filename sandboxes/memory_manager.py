"""
sandboxes/memory_manager.py
===========================

Sandbox for **MemoryManager** maintenance tasks.
Supports plain-text *or* voice capture of the initial transcript
description via the ``--voice/-v`` flag (same UX as the other sandboxes).

┌────────────── 8 accepted commands ──────────────┐
│ uc  X-Y   –– update_contacts                    │
│ ucb X-Y   –– update_contact_bio                 │
│ ucrs X-Y  –– update_contact_rolling_summary     │
│ uk  X-Y   –– update_knowledge                   │
│ cc        –– clear Contacts store              │
│ ccb       –– clear Contact bios      (alias cc) │
│ ccrs      –– clear Rolling summaries (alias cc) │
│ ck        –– clear Knowledge store             │
└─────────────────────────────────────────────────┘

• *X* and *Y* are **inclusive**, 0-based indices into the transcript
  (0 ≤ X ≤ Y < num_messages).
• Type **help** to show the table again, **quit/exit** to leave.
"""

from __future__ import annotations

import os
import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

import unify
from dotenv import load_dotenv

# ───────── project-local imports ─────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenario_builder import ScenarioBuilder
from unity.memory_manager.memory_manager import MemoryManager  # type: ignore[attr-defined]
from sandboxes.utils import (
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
LG = logging.getLogger("memory_manager_sandbox")


# ═════════════════════════════════ transcript seeding ═══════════════════════


async def _build_transcript(description: str) -> List[Dict[str, Any]]:
    """
    Use :class:`ScenarioBuilder` to create a *realistic* multi-party transcript
    (≈ 40-60 messages) according to *description* and return it as a list of
    message dicts.
    """
    transcript: List[Dict[str, Any]] = []

    # tool exposed to the LLM
    def log_messages(messages: list[dict]) -> str:
        """
        Append a batch of messages to the in-memory transcript.
        Expected keys per message: timestamp, sender, content.
        """
        nonlocal transcript
        transcript.extend(messages)
        return f"{len(messages)} messages logged"

    # Encourage the model to send messages in *batches*.
    prompt = (
        description.strip()
        + "\n\nGenerate 40-60 chronological chat messages that fit the "
        "scenario above.  Each dict **must** include 'timestamp' (ISO 8601), "
        "'sender' and 'content'.  Use the `log_messages` tool in batches of "
        "3-8 messages until the full transcript is logged, then stop."
    )

    builder = ScenarioBuilder(description=prompt, tools={"log_messages": log_messages})
    await builder.create()
    if not transcript:
        raise RuntimeError("ScenarioBuilder produced an empty transcript.")

    return transcript


# ═════════════════════════════════ helper utilities ═════════════════════════


def _chunk_to_text(messages: List[Dict[str, Any]]) -> str:
    """
    Convert a slice of message dicts to a plain-text transcript that the
    MemoryManager methods expect.
    """
    lines = [f"{m.get('sender', '')}: {m.get('content', '')}" for m in messages]
    return "\n".join(lines)


def _clear_contacts() -> None:
    ctxs = unify.get_contexts()
    if "Contacts" in ctxs:
        unify.delete_context("Contacts")


def _clear_knowledge() -> None:
    for name in unify.get_contexts(prefix="Knowledge").keys():
        unify.delete_context(name)


_CMD_RE = re.compile(r"^(uc|ucb|ucrs|uk)\s+(\d+)-(\d+)$", re.I)


def _explain_commands() -> None:
    print(__doc__.split("Text-only sandbox")[0].rstrip())


# ═════════════════════════════════ main async loop ══════════════════════════


async def _main_async() -> None:
    parser = argparse.ArgumentParser(description="MemoryManager sandbox")
    parser.add_argument(
        "--voice",
        "-v",
        action="store_true",
        help="enable voice capture + TTS for the initial scenario",
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
        help="wrap MemoryManager calls in Unify tracing",
    )
    args = parser.parse_args()

    # Unify context
    unify.activate("MemorySandbox")
    unify.set_trace_context("Traces")
    if args.traced:
        LG.info("[trace] Unify tracing enabled")
        os.environ["UNIFY_TRACED"] = "true"

    # ── Step 1: obtain scenario (voice or text), build transcript ────────────
    if not args.voice:
        scenario = input(
            "\n🧮  Describe the conversation you'd like to simulate (one or two "
            "sentences, e.g. *“A product-design chat between Alice, Bob and their "
            "client Carol discussing a new smartwatch”*)\n> ",
        ).strip()
    else:
        _speak(
            "Describe the conversation you'd like to simulate.  "
            "Press enter to start recording and again to finish.",
        )
        audio = _record_until_enter()
        scenario = _transcribe_deepgram(audio).strip()
        if not scenario:
            _speak("I didn't catch that, please type your description instead.")
            scenario = input("> ").strip()

    if not scenario:
        print("Nothing entered – exiting.")
        return

    print("\n[seed] Generating transcript – this can take a minute…")
    transcript = await _build_transcript(scenario)
    num_messages = len(transcript)
    print(
        f"[seed] Done.  Generated {num_messages} messages (indices 0-{num_messages-1}).\n",
    )

    # ── MemoryManager instance ──────────────────────────────────────────────
    mm: MemoryManager = MemoryManager()

    _explain_commands()

    # ── Interactive loop ───────────────────────────────────────────────────
    while True:
        try:
            raw = input("\n>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break

        if raw in {"quit", "exit"}:
            break
        if raw in {"help", "h", "?"}:
            _explain_commands()
            continue

        # ─── clearing commands ────────────────────────────────────────────
        if raw in {"cc", "ccb", "ccrs"}:
            _clear_contacts()
            mm = MemoryManager()  # new instance, clean slate
            print("✅ Contacts store cleared.")
            continue
        if raw == "ck":
            _clear_knowledge()
            mm = MemoryManager()
            print("✅ Knowledge store cleared.")
            continue

        # ─── functional commands ─────────────────────────────────────────
        m = _CMD_RE.match(raw)
        if not m:
            print("⚠️  Unrecognised command. Type 'help' for guidance.")
            continue

        cmd, xs, ys = m.groups()
        start, end = int(xs), int(ys)
        if not (0 <= start <= end < num_messages):
            print(f"⚠️  Indices must satisfy 0 ≤ x ≤ y < {num_messages}.")
            continue

        chunk_txt = _chunk_to_text(transcript[start : end + 1])

        print(f"[{cmd}] Running on messages {start}-{end} …")
        try:
            if cmd == "uc":
                result = await mm.update_contacts(chunk_txt)
            elif cmd == "ucb":
                result = await mm.update_contact_bio(chunk_txt)
            elif cmd == "ucrs":
                result = await mm.update_contact_rolling_summary(chunk_txt)
            else:  # uk
                result = await mm.update_knowledge(chunk_txt)

            print(f"→ {result}")
        except Exception as exc:
            LG.error("Error during MemoryManager call: %s", exc, exc_info=True)
            print(f"❌  {exc}")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
