"""
sandboxes/memory_manager.py
===========================

Sandbox for **MemoryManager** maintenance tasks.
Supports plain-text *or* voice capture of the initial transcript
description via the ``--voice/-v`` flag (same UX as the other sandboxes).

┌────────────── 10 accepted commands ─────────────┐
│ uc   –– update_contacts                         │
│ ucb  –– update_contact_bio                      │
│ ucrs –– update_contact_rolling_summary          │
│ uk   –– update_knowledge                        │
│ ut   –– update_tasks                           │
│ cc        –– clear Contacts store               │
│ ccb       –– clear Contact bios      (alias cc) │
│ ccrs      –– clear Rolling summaries (alias cc) │
│ ck        –– clear Knowledge store              │
│ r         –– record scenario description (voice)│
└─────────────────────────────────────────────────┘

After typing **uc / ucb / ucrs / uk / ut** you will be *asked* for the message
**range** in a second prompt.  Use Python-slice style notation just like
lists are indexed:

    0:10   –– messages 0 through 10 (inclusive)
   -5:     –– the **last** 5 messages
     4:    –– from 4 to the end

Press ↵ with an empty response to process **all** messages.
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


async def _build_transcript(
    description: str,
    *,
    delay_per_message: float = 0.0,
) -> List[Dict[str, Any]]:
    """Generate a synthetic transcript via the shared TranscriptGenerator.

    The optional *delay_per_message* argument allows callers to throttle the
    rate at which each message is logged so that EventBus subscribers fire in
    real-time, making it easier to observe behaviour when certain thresholds
    are crossed.
    """
    generator = TranscriptGenerator()
    return await generator.generate(
        description,
        delay_per_message=delay_per_message,
    )


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


# ---------------------------------------------------------------------------
#  Range-parsing helpers (slice-style "start:end")
# ---------------------------------------------------------------------------


def _parse_range(range_str: str, num_messages: int) -> tuple[int, int]:
    """Return an **inclusive** (start, end) tuple based on *range_str*.

    The accepted syntax mirrors Python slice notation but is easier:

    "start:end"  – start *and/or* end may be omitted; negative indices
    count from the end.  The *end* index is treated **inclusive** so
    the intuitive "0:10" captures the first eleven messages just like
    it did with the previous "0-10" syntax.
    """

    if ":" not in range_str:
        # Single value → treat as one-element range
        idx = int(range_str)
        if idx < 0:
            idx += num_messages
        return idx, idx

    left, right = range_str.split(":", 1)

    def _to_idx(val: str | None, default: int) -> int:
        if val is None or val == "":
            return default
        iv = int(val)
        return iv + num_messages if iv < 0 else iv

    start = _to_idx(left, 0)
    end = _to_idx(right, num_messages - 1)

    if not (0 <= start <= end < num_messages):
        raise ValueError(
            f"Indices must satisfy 0 ≤ start ≤ end < {num_messages} (got {start}:{end})",
        )

    return start, end


# Map long-form command names to their short aliases for convenience
_CMD_ALIASES: dict[str, str] = {
    "update_contacts": "uc",
    "update_contact_bio": "ucb",
    "update_contact_rolling_summary": "ucrs",
    "update_knowledge": "uk",
    "update_tasks": "ut",
}


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
    # ------------------------------------------------------------------
    # NEW: Custom window / chunk configuration via JSON file
    # ------------------------------------------------------------------
    parser.add_argument(
        "--windows_config",
        "-w",
        metavar="PATH",
        type=str,
        help=(
            "Path to a JSON file that overrides the default rolling-activity "
            "time/count windows and/or the transcript chunk size. Structure: "
            '{\n  "time_windows": { "past_day": 86400, ... },\n  "count_windows": { "past_interaction": 1, ... },\n  "chunk_size": 25\n}. '
            "Units: time windows are *seconds*; count windows are raw integers. "
            "See tests/test_memory/_patch_memory_manager_windows for examples."
        ),
    )

    # ──────────────────────────────────────────────────────────────────
    # Optional: throttle message logging so callbacks can be observed
    # ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--stagger_seconds",
        "-s",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Delay SECONDS between each TranscriptManager.log_messages() call "
            "when seeding the synthetic transcript. Set to 0 for immediate "
            "logging (default).",
        ),
    )
    args = parser.parse_args()

    # Voice feedback helper
    voice_enabled = args.voice

    def _maybe_speak(message: str) -> None:  # noqa: D401 – helper
        """Vocalise *message* when --voice mode is active."""
        if voice_enabled:
            try:
                _speak(str(message))
            except Exception:
                # TTS should never break core sandbox functionality
                LG.warning("[voice] Failed to speak feedback.")

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

    # ─────────────────── apply custom windows / chunk size ────────────────────
    def _apply_custom_windows(cfg: dict[str, Any]) -> None:  # noqa: D401 – helper name
        """Patch *class-level* window constants on MemoryManager.

        The helper rebuilds the auxiliary *_ORDER* and *_PARENT* sequences so that
        internal hierarchy calculations remain consistent with the overrides.
        """

        cls = MemoryManager

        # ---- 1.  Time-based windows -------------------------------------
        time_windows = cfg.get("time_windows")
        if isinstance(time_windows, dict) and time_windows:
            cls._TIME_WINDOWS = time_windows  # type: ignore[attr-defined]
            # Sort by ascending duration for deterministic parent mapping
            cls._TIME_ORDER = sorted(time_windows, key=time_windows.get)  # type: ignore[attr-defined]

            cls._TIME_PARENT = {}  # type: ignore[attr-defined]
            for i in range(1, len(cls._TIME_ORDER)):  # type: ignore[attr-defined]
                child, parent = cls._TIME_ORDER[i], cls._TIME_ORDER[i - 1]  # type: ignore[attr-defined]
                cls._TIME_PARENT[child] = (
                    parent,
                    time_windows[child] // time_windows[parent],
                )  # type: ignore[attr-defined]

        # ---- 2.  Count-based windows ------------------------------------
        count_windows = cfg.get("count_windows")
        if isinstance(count_windows, dict) and count_windows:
            cls._COUNT_WINDOWS = count_windows  # type: ignore[attr-defined]
            cls._COUNT_ORDER = sorted(count_windows, key=count_windows.get)  # type: ignore[attr-defined]

            cls._COUNT_PARENT = {}  # type: ignore[attr-defined]
            for i in range(1, len(cls._COUNT_ORDER)):  # type: ignore[attr-defined]
                child, parent = cls._COUNT_ORDER[i], cls._COUNT_ORDER[i - 1]  # type: ignore[attr-defined]
                cls._COUNT_PARENT[child] = (
                    parent,
                    count_windows[child] // count_windows[parent],
                )  # type: ignore[attr-defined]

        # ---- 3.  Rolling column list needs refresh so new windows appear
        _cols: list[str] = []
        for nick in cls._MANAGERS.values():  # type: ignore[attr-defined]
            for window in list(cls._TIME_WINDOWS) + list(cls._COUNT_WINDOWS):  # type: ignore[attr-defined]
                _cols.append(f"{nick}/{window}")
        _cols.extend([cls._SUMMARY_TIME_COL, cls._SUMMARY_COUNT_COL])  # type: ignore[attr-defined]
        cls._ROLLING_COLUMNS = tuple(_cols)  # type: ignore[attr-defined]

    # Read JSON config (only when automatic summaries/updates are enabled)
    _custom_chunk_size: int | None = None
    if args.windows_config and not (args.manual_summaries and args.manual_updates):
        try:
            import json, pathlib

            cfg_path = pathlib.Path(args.windows_config).expanduser()
            if cfg_path.is_file():
                with cfg_path.open("r", encoding="utf-8") as fp:
                    cfg_json = json.load(fp)
                _apply_custom_windows(cfg_json)
                _custom_chunk_size = cfg_json.get("chunk_size")
                LG.info(
                    "[config] Applied custom windows from %s%s",
                    cfg_path,
                    f" (chunk_size={_custom_chunk_size})" if _custom_chunk_size else "",
                )
            else:
                LG.warning("[config] JSON file %s not found – ignoring", cfg_path)
        except Exception as exc:
            LG.error("[config] Failed to parse windows_config: %s", exc)

    # Helper to create a fresh, patched MemoryManager instance --------------
    def _create_mm() -> MemoryManager:
        inst = MemoryManager()
        try:
            inst._CHUNK_SIZE = int(os.getenv("MM_CHUNK_SIZE", "10"))  # type: ignore[attr-defined]
        except Exception:
            pass
        # Override via JSON config (only when automatic updates are active)
        if _custom_chunk_size and not args.manual_updates:
            try:
                inst._CHUNK_SIZE = int(_custom_chunk_size)  # type: ignore[attr-defined]
            except Exception:
                LG.warning(
                    "[config] Invalid chunk_size (%s) – must be int; keeping %s",
                    _custom_chunk_size,
                    inst._CHUNK_SIZE,
                )
        return inst

    mm = _create_mm()

    tm = mm._transcript_manager  # use the TranscriptManager owned by MemoryManager

    # Helper: convert *TranscriptGenerator* dict → Message-schema dict ----------
    from unity.contact_manager.types.contact import Contact

    _name_to_contact: dict[str, Contact] = {}
    last_sender_contact: Contact | None = None

    def _create_contact(name: str, medium: str) -> Contact:
        slug = name.lower().replace(" ", ".")
        idx = len(_name_to_contact) + 1
        if medium == "email":
            return Contact(first_name=name.title(), email_address=f"{slug}@example.com")
        return Contact(first_name=name.title(), phone_number=f"+155509{idx:04d}")

    def _contact_for(name: str, medium: str) -> Contact:
        if name not in _name_to_contact:
            _name_to_contact[name] = _create_contact(name, medium)
        return _name_to_contact[name]

    # ── Interactive REPL ------------------------------------------------------
    print(
        "\nMemoryManager sandbox – enter a conversation *description* to generate "
        "and log synthetic messages, *or* type one of the maintenance commands "
        "below.  Type 'summary' to display the latest rolling-activity overview "
        "or 'quit' to exit.\n",
    )

    # Show the full list of commands immediately so the user knows the options
    _explain_commands()

    # Track latest generated transcript for subsequent maintenance commands
    last_transcript: List[Dict[str, Any]] = []

    # Voice-mode greeting so behaviour matches other sandboxes
    if args.voice:
        _speak(
            "Welcome to the Memory Manager sandbox. Describe your conversation scenario or enter one of the maintenance commands. Type R and press Enter whenever you'd like to record a scenario description using your voice.",
        )

    while True:
        # Reprint the command list so it's always visible just before the prompt
        print()
        _explain_commands()
        print()
        try:
            # Voice or text capture for the scenario / command prompt
            if args.voice:
                # Offer the user a choice instead of immediately starting voice capture
                prompt = input(
                    "scenario/command (see command list above)> ",
                ).strip()
                if prompt.lower() in {"r", "record"}:
                    audio = _record_until_enter()
                    prompt = _transcribe_deepgram(audio).strip()
                    if not prompt:
                        continue
                    print(f"▶️  {prompt}")
            else:
                prompt = input(
                    "scenario/command (see command list above)> ",
                ).strip()
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
            _maybe_speak("Contacts store cleared.")
            continue
        if lower == "ck":
            _clear_knowledge()
            mm = _create_mm()
            tm = mm._transcript_manager
            print("✅ Knowledge store cleared.")
            _maybe_speak("Knowledge store cleared.")
            continue

        # Functional uc/ucb/ucrs/uk commands --------------------------------
        parts = prompt.split(maxsplit=1)
        cmd = _CMD_ALIASES.get(parts[0], parts[0])

        if cmd in {"uc", "ucb", "ucrs", "uk", "ut"}:
            if not last_transcript:
                print("⚠️  No transcript available yet – generate one first.")
                continue

            # ─────────────── prompt for range ────────────────
            num_messages = len(last_transcript)
            try:
                range_input = input(
                    f"Message range [start:end] (default: all {num_messages} messages, 'b' to go back)> ",
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print()  # newline for clean prompt
                continue

            # Allow user to go back to the main prompt
            if range_input.lower() in {"b", "back"}:
                print("↩️  Returning to main menu…")
                continue

            try:
                if range_input == "":
                    start, end = 0, num_messages - 1
                else:
                    start, end = _parse_range(range_input, num_messages)
            except ValueError as exc:
                print(f"⚠️  {exc}")
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
                elif cmd in {"ucb", "ucrs"}:
                    cid_str = input(
                        "Target contact_id (press Enter to cancel)> ",
                    ).strip()
                    if cid_str == "":
                        print("⚠️  Command cancelled (no contact_id provided).")
                        continue
                    try:
                        cid_val = int(cid_str)
                    except ValueError:
                        print("⚠️  contact_id must be a valid integer.")
                        continue

                    if cmd == "ucb":
                        result = await mm.update_contact_bio(
                            chunk_txt,
                            contact_id=cid_val,
                            guidance=guidance_txt,
                        )
                    else:  # ucrs
                        result = await mm.update_contact_rolling_summary(
                            chunk_txt,
                            contact_id=cid_val,
                            guidance=guidance_txt,
                        )
                elif cmd == "ut":
                    result = await mm.update_tasks(chunk_txt, guidance=guidance_txt)
                else:  # uk
                    result = await mm.update_knowledge(chunk_txt, guidance=guidance_txt)

                print(f"→ {result}")
                _maybe_speak(result)
            except Exception as exc:
                LG.error("Error during MemoryManager call: %s", exc, exc_info=True)
                print(f"❌  {exc}")

            continue  # back to REPL

        # ------------------------------------------------------------------
        #  Scenario generation branch (default)
        # ------------------------------------------------------------------

        # Otherwise treat the input as a new transcript scenario description.
        print("[generate] Building synthetic transcript – this can take a moment…")
        if args.voice:
            _speak("Sure thing, building your custom scenario now.")
        try:
            transcript = await _build_transcript(
                prompt,
                delay_per_message=args.stagger_seconds,
            )
            if args.voice:
                _speak("All done, your custom scenario is built and ready to go.")
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
        EVENT_BUS.join_callbacks()

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
