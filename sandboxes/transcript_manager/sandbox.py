"""
======================================================================
Interactive sandbox for **TranscriptManager**.

Features
--------
• Fixed or LLM-generated seed data via :class:`ScenarioBuilder`.
• Voice or plain-text input (shared helpers).
• Automatic dispatch to `ask` depending on intent.
• Mid-conversation interruption (pause / interject / cancel).
• Scenario builder tool-loop exposes **private** ``_log_messages`` alongside
  the public `ask` method so that the LLM can inject raw transcripts directly.
"""

from __future__ import annotations

# ─────────────────────────────── stdlib / vendored ──────────────────────────
import os
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime

import unify

from dotenv import load_dotenv
from sandboxes.scenario_builder import ScenarioBuilder

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

# ────────────────────────────────  unity imports  ───────────────────────────
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.common.llm_helpers import SteerableToolHandle
from sandboxes.utils import (  # shared helpers reused in other sandboxes
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    await_with_interrupt as _await_with_interrupt,
    build_cli_parser,
    activate_project,
    _wait_for_tts_end as _wait_tts_end,
)

LG = logging.getLogger("transcript_sandbox")

# ═════════════════════════════════ seed helpers ═════════════════════════════


async def _build_scenario(custom: Optional[str] = None) -> Optional[str]:
    """Populate the transcript store **via the official tools** using
    :class:`ScenarioBuilder`.

    The tool-loop exposes:
    • ``_log_messages`` – for inserting raw transcript messages.
    • ``ask`` – so the LLM can sanity-check its inserts.
    """

    tm = TranscriptManager()

    description = (
        custom.strip()
        if custom
        else (
            "Generate 15 realistic message exchanges across email, Slack and "
            "WhatsApp between 5 colleagues over the last two weeks. Vary the "
            "topics (project updates, meeting scheduling, casual banter). "
            "Provide rich, time-ordered message content so that questions "
            "about context, participants and timing are interesting."
        )
    )
    description += (
        "\nYou have two tools:\n"
        "• `_log_messages` — write raw messages (you *must* supply all schema "
        "  fields).\n"
        "• `ask` — query what you've created.\n"
        "Work in batches: insert several related messages at once, then call "
        "`ask` to confirm everything looks good."
    )

    def log_messages(messages: list[dict]) -> str:
        for msg in messages:
            tm.log_messages(msg)
        tm.join_published()
        return "messages logged successfully"

    builder = ScenarioBuilder(
        description=description,
        tools={
            "log_messages": log_messages,
            "ask": tm.ask,
        },
    )

    try:
        await builder.create()
    except Exception as exc:
        raise RuntimeError(f"LLM seeding via ScenarioBuilder failed. {exc}")

    return None  # kept for signature parity with other sandboxes


# ═════════════════════════════ intent dispatcher ════════════════════════════


async def _dispatch_with_context(
    tm: TranscriptManager,
    raw: str,
    *,
    show_steps: bool,
    parent_chat_context: List[Dict[str, str]],
) -> Tuple[str, SteerableToolHandle]:
    """Always route *raw* to `ask`, forwarding parent chat context."""

    handle = await tm.ask(
        raw,
        parent_chat_context=parent_chat_context,
        _return_reasoning_steps=show_steps,
    )
    return "ask", handle


# ══════════════════════════════════  CLI  ═══════════════════════════════════


async def _main_async() -> None:
    parser = build_cli_parser("TranscriptManager sandbox")
    args = parser.parse_args()

    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    activate_project(args.project_name, args.overwrite)
    base_ctx = unify.get_active_context().get("write")
    traces_ctx = f"{base_ctx}/Traces" if base_ctx else "Traces"
    unify.set_trace_context(traces_ctx)
    if args.overwrite:
        ctxs = unify.get_contexts()
        for tbl in (
            "Transcripts",
            "Contacts",
            traces_ctx,
        ):
            if tbl in ctxs:
                unify.delete_context(tbl)
        unify.create_context(traces_ctx)

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

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LG.setLevel(logging.INFO)

    tm: TranscriptManager = TranscriptManager()
    if args.traced:
        tm = unify.traced(tm)

    # ─────────────────── optional initial seeding ─────────────────────────
    # No automatic seeding – users can invoke 'us' / 'usv' commands to populate transcripts when desired.

    # ─────────────────── command helper output ────────────────────

    _COMMANDS_HELP = (
        "\nTranscriptManager sandbox – type commands below (press ↵ with an empty "
        "line to dictate via voice when --voice mode is active – type 'r' to record).  'quit' to exit.\n\n"
        "┌────────────────── accepted commands ─────────────────────┐\n"
        "│ us  {description}     – update_scenario (text)           │\n"
        "│ usv                   – update_scenario_vocally          │\n"
        "│ r / free text         – freeform ask (auto)              │\n"
        "│ save_project | sp     – save project snapshot            │\n"
        "│ help | h              – show this help                   │\n"
        "└──────────────────────────────────────────────────────────┘\n"
    )

    def _explain_commands() -> None:  # noqa: D401 – helper
        print(_COMMANDS_HELP)

    if args.voice:
        _speak(
            "Sandbox ready. You can type commands, or press enter on an empty line "
            "to record a voice query.  Use 'u-s-v' to build a new scenario vocally.",
        )
        _wait_tts_end()

    # running memory of the dialogue (passed back into tm.ask for context)
    chat_history: List[Dict[str, str]] = []

    # interaction loop
    while True:
        # Keep command list visible similar to MemoryManager sandbox
        print()
        _explain_commands()
        print()

        try:
            if args.voice:
                # Ensure any ongoing TTS playback has finished before showing prompt
                _wait_tts_end()
            if args.voice:
                # Voice mode prompt with 'r' option
                raw = input("command ('r' to record)> ").strip()
                if raw.lower() == "r":
                    audio = _record_until_enter()
                    raw = _transcribe_deepgram(audio).strip()
                    if not raw:
                        continue
                    print(f"▶️  {raw}")
            else:
                raw = input("command> ").strip()

            # Show help table
            if raw.lower() in {"help", "h", "?"}:
                _explain_commands()
                continue

            if raw.lower() in {"quit", "exit"}:
                break
            if not raw:
                continue

            # ─────────────── save project snapshot ────────────────
            if raw.lower() in {"save_project", "sp"}:
                commit_hash = unify.commit_project(
                    args.project_name,
                    commit_message=f"Sandbox save {datetime.utcnow().isoformat()}",
                ).get("commit_hash")
                print(f"💾 Project saved at commit {commit_hash}")
                if args.voice:
                    _speak("Project saved")
                continue

            # ─────────────── scenario (re)seeding commands ────────────────
            parts = raw.split(maxsplit=1)
            cmd_lower = parts[0].lower()

            if cmd_lower in {"us", "update_scenario"}:
                # Text-based scenario description supplied after the command, if any
                description = parts[1].strip() if len(parts) > 1 else ""
                if not description:
                    description = input(
                        "🧮 Describe the transcript scenario you want to build > ",
                    ).strip()
                    if not description:
                        print("⚠️  No description provided – cancelled.")
                        continue

                print(
                    "[generate] Building synthetic transcripts – this can take a moment…",
                )
                if args.voice:
                    _speak("Sure thing, building your custom scenario now.")
                try:
                    await _build_scenario(description)
                    if args.voice:
                        _speak(
                            "All done, your custom scenario is built and ready to go.",
                        )
                except Exception as exc:
                    LG.error("Scenario generation failed: %s", exc, exc_info=True)
                    print(f"❌  Failed to generate scenario: {exc}")
                continue  # back to REPL

            if cmd_lower in {"usv", "update_scenario_vocally"}:
                if not args.voice:
                    print(
                        "⚠️  Voice mode not enabled – restart with --voice or use 'us' instead.",
                    )
                    continue

                audio = _record_until_enter()
                description = _transcribe_deepgram(audio).strip()
                if not description:
                    print("⚠️  Transcription was empty – please try again.")
                    continue
                print(f"▶️  {description}")

                print(
                    "[generate] Building synthetic transcripts – this can take a moment…",
                )
                try:
                    await _build_scenario(description)
                    if args.voice:
                        _speak(
                            "All done, your custom scenario is built and ready to go.",
                        )
                except Exception as exc:
                    LG.error("Scenario generation failed: %s", exc, exc_info=True)
                    print(f"❌  Failed to generate scenario: {exc}")
                continue  # back to REPL

            # ───────────── remember the user's utterance before dispatch ──────
            _kind, result = await _dispatch_with_context(
                tm,
                raw,
                show_steps=args.debug,
                parent_chat_context=list(chat_history),
            )
            chat_history.append({"role": "user", "content": raw})
            if args.voice:
                _speak("Let me take a look, give me a moment")

            # ───────────── process result (handle or immediate string) ─────────
            if isinstance(result, SteerableToolHandle):
                answer = await _await_with_interrupt(result)
                if isinstance(answer, tuple):  # reasoning steps requested
                    answer, _steps = answer
            else:  # already a string (unlikely path)
                answer = result

            if args.voice:
                _speak("Okay, that's all done")
            print(f"[{_kind}] → {answer}\n")

            # ───────────── remember assistant's reply for follow-up ────────────
            chat_history.append({"role": "assistant", "content": answer})
            if args.voice:
                _speak(f"{answer} Anything else I can help with?")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break
        except Exception as exc:
            LG.error("[error] %s", exc)

    # end while


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
