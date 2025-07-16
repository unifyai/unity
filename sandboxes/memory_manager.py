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
    activate_project,
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
    # NEW ------------------------------------------------------------------
    parser.add_argument(
        "--manual_summaries",
        action="store_true",
        help="Disable automatic rolling-activity snapshot generation (MemoryManager._setup_rolling_callbacks).",
    )
    parser.add_argument(
        "--manual_updates",
        action="store_true",
        help="Disable automatic memory updates triggered by message chunks (MemoryManager._setup_message_callbacks).",
    )
    args = parser.parse_args()

    # Unify context
    activate_project(args.project_name, args.overwrite)
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

    # ── Monkey-patch MemoryManager behaviour based on CLI flags ─────────────
    async def _noop(self, *_, **__):
        return None

    if args.manual_summaries:
        setattr(MemoryManager, "_setup_rolling_callbacks", _noop)  # type: ignore[arg-type]
    if args.manual_updates:
        setattr(MemoryManager, "_setup_message_callbacks", _noop)  # type: ignore[arg-type]

    # Helper to create a fresh, patched MemoryManager instance --------------
    def _create_mm() -> MemoryManager:
        inst = MemoryManager()
        try:
            inst._CHUNK_SIZE = int(os.getenv("MM_CHUNK_SIZE", "10"))  # type: ignore[attr-defined]
        except Exception:
            pass
        return inst

    mm = _create_mm()

    tm = mm._transcript_manager  # use the TranscriptManager owned by MemoryManager

    # Helper: convert *TranscriptGenerator* dict → Message-schema dict ----------
    _name_to_id: dict[str, int] = {}
    # Keep track of the sender of the previous message so we can infer the receiver
    last_sender_id: int | None = None

    def _normalise_msg(raw: dict) -> dict:
        """Return a dict that satisfies TranscriptManager.log_message schema."""

        nonlocal last_sender_id

        sender_name = str(raw.get("sender", "User"))

        # Keep **contact-id 0** reserved for the assistant / system persona so
        # that MemoryManager logic (which skips contact_id 0) continues to work.
        if sender_name.lower() in {"assistant", "system", "agent", "bot"}:
            sender_id = 0
        else:
            sender_id = _name_to_id.setdefault(sender_name, len(_name_to_id) + 1)

        # 1️⃣ Prefer explicit receiver information if provided by the generator
        receiver_name = raw.get("receiver")
        if receiver_name is not None:
            receiver_id = _name_to_id.setdefault(receiver_name, len(_name_to_id) + 1)
        # 2️⃣ Otherwise assume a back-and-forth pattern so the receiver is the
        #    previous sender (when different from the current sender).
        elif last_sender_id is not None and last_sender_id != sender_id:
            receiver_id = last_sender_id
        # 3️⃣ Fallback for very first message or single-speaker transcripts
        else:
            receiver_id = 0  # assistant / unknown contact by convention

        # Update the tracker for the next message
        last_sender_id = sender_id

        return {
            "medium": raw.get("medium", "sms_message"),
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "timestamp": raw.get("timestamp"),
            "content": raw.get("content", ""),
            "exchange_id": raw.get("exchange_id", 0),
        }

    # ── Interactive REPL ------------------------------------------------------
    print(
        "\nMemoryManager sandbox – enter a conversation *description* to generate "
        "and log synthetic messages.  Type 'summary' to display the latest "
        "rolling-activity overview or 'quit' to exit.\n",
    )

    # Voice-mode greeting so behaviour matches other sandboxes
    if args.voice:
        _speak(
            "Welcome to the Memory Manager sandbox. Describe your conversation scenario or choose a maintenance command. Press enter to start recording.",
        )

    while True:
        try:
            # Voice or text capture for the scenario / command prompt
            if args.voice:
                audio = _record_until_enter()
                prompt = _transcribe_deepgram(audio).strip()
                if not prompt:
                    continue
                print(f"▶️  {prompt}")
            else:
                prompt = input("scenario> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break

        if not prompt:
            continue

        if prompt.lower() in {"quit", "exit"}:
            break

        if prompt.lower() in {"summary", "s"}:
            overview = mm.get_rolling_activity()
            print("\n──────── Historic Activity ────────\n")
            print(overview or "<no activity yet>")
            print("\n──────────────────────────────────\n")
            continue

        # ------------------------------------------------------------------
        #  Manual command loop (uc, uk, …) – executed *before* scenario mode
        # ------------------------------------------------------------------

        lower = prompt.lower()

        if lower in {"help", "h", "?"}:
            _explain_commands()
            continue

        if lower in {"cc", "ccb", "ccrs"}:
            _clear_contacts()
            mm = _create_mm()
            tm = mm._transcript_manager
            print("✅ Contacts store cleared.")
            continue
        if lower == "ck":
            _clear_knowledge()
            mm = _create_mm()
            tm = mm._transcript_manager
            print("✅ Knowledge store cleared.")
            continue

        # Functional uc/ucb/ucrs/uk commands --------------------------------
        parts = prompt.split(maxsplit=1)
        cmd = parts[0]
        if cmd in {"uc", "ucb", "ucrs", "uk"}:
            if not last_transcript:
                print("⚠️  No transcript available yet – generate one first.")
                continue

            num_messages = len(last_transcript)
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

            chunk_txt = _chunk_to_text(last_transcript[start : end + 1])

            # Optional guidance -------------------------------------------
            guidance_txt: str | None = None
            yn = input("Add guidance for this run? [y/N] ").strip().lower()
            if yn in {"y", "yes"}:
                guidance_txt = input("Guidance> ").strip()
                guidance_txt = guidance_txt or None

            print(f"[{cmd}] Running on messages {start}-{end} …")
            try:
                if cmd == "uc":
                    result = await mm.update_contacts(chunk_txt, guidance=guidance_txt)
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
                    result = await mm.update_knowledge(chunk_txt, guidance=guidance_txt)

                print(f"→ {result}")
            except Exception as exc:
                LG.error("Error during MemoryManager call: %s", exc, exc_info=True)
                print(f"❌  {exc}")

            continue  # back to REPL

        # ------------------------------------------------------------------
        #  Scenario generation branch (default)
        # ------------------------------------------------------------------

        # Otherwise treat the input as a new transcript scenario description.
        print("[generate] Building synthetic transcript – this can take a moment…")
        try:
            transcript = await _build_transcript(prompt)
        except Exception as exc:
            LG.error("Transcript generation failed: %s", exc, exc_info=True)
            print(f"❌  Failed to generate transcript: {exc}")
            continue

        print(f"[log] Ingesting {len(transcript)} messages …")

        # Save as latest for manual commands
        last_transcript = transcript

        # Give EventBus a chance to upload + process subscriptions --------
        from unity.events.event_bus import EVENT_BUS

        EVENT_BUS.join_published()
        EVENT_BUS.join_callbacks(cascade=True)

        # Show updated rolling activity -----------------------------------
        overview = mm.get_rolling_activity()
        print("\n──────── Updated Historic Activity ────────\n")
        print(overview or "<no activity captured>")
        print("\n──────────────────────────────────────────\n")

    print("Goodbye! 👋")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
