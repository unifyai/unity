"""
sandboxes/memory_manager.py
===========================

Sandbox for **MemoryManager** maintenance tasks.
Supports plain-text *or* voice capture of the initial transcript
description via the ``--voice/-v`` flag (same UX as the other sandboxes).

┌────────────── 8 accepted commands ──────────────┐
│ uc  [X-Y] –– update_contacts  (no range → full) │
│ ucb [X-Y] –– update_contact_bio                 │
│ ucrs[X-Y] –– update_contact_rolling_summary     │
│ uk  [X-Y] –– update_knowledge                   │
│ cc        –– clear Contacts store               │
│ ccb       –– clear Contact bios      (alias cc) │
│ ccrs      –– clear Rolling summaries (alias cc) │
│ ck        –– clear Knowledge store              │
└─────────────────────────────────────────────────┘

• *X* and *Y* are **inclusive**, 0-based indices into the transcript
  (0 ≤ X ≤ Y < num_messages).  Omitting the range processes **all**
  messages.
• Type **help** to show the table again, **quit/exit** to leave.

After choosing any *u** command you can now add **extra guidance**
that steers what the MemoryManager should prioritise (e.g. *"Focus on
project-related facts only"*).  In `--voice` mode this prompt is captured
with the microphone; otherwise just type it.
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
from datetime import datetime

import unify
from dotenv import load_dotenv

# ───────── project-local imports ─────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unity.memory_manager.memory_manager import MemoryManager  # type: ignore[attr-defined]
from sandboxes.utils import (
    TranscriptGenerator,
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
LG = logging.getLogger("memory_manager_sandbox")


# ═════════════════════════════════ transcript seeding ═══════════════════════


async def _build_transcript(description: str) -> List[Dict[str, Any]]:
    """Generate a synthetic transcript via the shared TranscriptGenerator."""
    generator = TranscriptGenerator()
    return await generator.generate(description)


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


# Bare command or "cmd  X-Y"
_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")


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
    parser.add_argument(
        "--project_name",
        "-p",
        default="Sandbox",
        help="Unify project / context name (default: Sandbox)",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="overwrite existing data for the chosen project",
    )
    parser.add_argument(
        "--project_version",
        type=int,
        default=-1,
        metavar="IDX",
        help="Project version index to load (default -1 for latest; supports positive and negative indexing)",
    )
    args = parser.parse_args()

    # Unify context
    unify.activate(args.project_name)
    unify.set_trace_context("Traces")
    if args.traced:
        LG.info("[trace] Unify tracing enabled")
        os.environ["UNIFY_TRACED"] = "true"

    # ─────────────────── project version handling ────────────────────
    if args.project_version != -1:
        commits = unify.get_project_commits(args.project_name)
        if commits:
            try:
                target = commits[args.project_version]
                unify.rollback_project(args.project_name, target["commit_hash"])
                LG.info("[version] Rolled back to commit %s", target["commit_hash"])
            except IndexError:
                LG.warning(
                    "[version] project_version index %s out of range, ignoring",
                    args.project_version,
                )

    # Optionally wipe existing data first
    if args.overwrite:
        _clear_contacts()
        _clear_knowledge()

    # ── Step 1: obtain scenario (voice or text), build transcript ────────────
    if not args.voice:
        scenario = input(
            "\n🧮  Describe the conversation you'd like to simulate (one or two "
            "sentences, e.g. *'A product-design chat between Alice, Bob and their "
            "client Carol discussing a new smartwatch'*)\n> ",
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
    _speak(
        "That's now been generated for you. From this point forward, just use the commands in the terminal to use tools on the transcript.",
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
        parts = raw.split(maxsplit=1)
        cmd = parts[0]
        if cmd not in {"uc", "ucb", "ucrs", "uk"}:
            print("⚠️  Unrecognised command. Type 'help' for guidance.")
            continue

        # Default slice – entire transcript
        start, end = 0, num_messages - 1
        if len(parts) == 2:
            rng = parts[1].strip()
            m = _RANGE_RE.match(rng)
            if not m:
                print("⚠️  Range must be of the form X-Y (e.g. 4-18).")
                continue
            start, end = map(int, m.groups())
            if not (0 <= start <= end < num_messages):
                print(f"⚠️  Indices must satisfy 0 ≤ x ≤ y < {num_messages}.")
                continue

        chunk_txt = _chunk_to_text(transcript[start : end + 1])

        # ─── optional GUIDANCE capture ──────────────────────────────────
        guidance_txt: str | None = None
        yn = input("Add guidance for this run? [y/N] ").strip().lower()
        if yn in {"y", "yes"}:
            if args.voice:
                _speak(
                    "Please dictate your guidance now. Press enter to start "
                    "recording and again to finish.",
                )
                audio = _record_until_enter()
                guidance_txt = _transcribe_deepgram(audio).strip()
                if not guidance_txt:
                    _speak("I didn't catch that, please type your guidance.")
            if guidance_txt is None:  # fallback for voice or non-voice
                guidance_txt = input("Guidance> ").strip()
            if not guidance_txt:
                guidance_txt = None  # treat empty as absent

        print(f"[{cmd}] Running on messages {start}-{end} …")
        try:
            if cmd == "uc":
                result = await mm.update_contacts(
                    chunk_txt,
                    guidance=guidance_txt,
                )
            elif cmd == "ucb":
                result = await mm.update_contact_bio(
                    chunk_txt,
                    guidance=guidance_txt,
                )
            elif cmd == "ucrs":
                result = await mm.update_contact_rolling_summary(
                    chunk_txt,
                    guidance=guidance_txt,
                )
            else:  # uk
                result = await mm.update_knowledge(
                    chunk_txt,
                    guidance=guidance_txt,
                )

            print(f"→ {result}")
            _speak(str(result))
        except Exception as exc:
            LG.error("Error during MemoryManager call: %s", exc, exc_info=True)
            print(f"❌  {exc}")
            _speak("There was an error running that command.")

        # ─────────────── save project snapshot ────────────────
        if raw in {"save_project", "sp"}:
            commit_hash = unify.commit_project(
                args.project_name,
                commit_message=f"Sandbox save {datetime.utcnow().isoformat()}",
            ).get("commit_hash")
            print(f"💾 Project saved at commit {commit_hash}")
            _speak("Project saved")
            continue


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
