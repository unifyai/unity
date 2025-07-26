"""contact_sandbox.py  (optional voice mode, Deepgram SDK v4, sync)
===================================================================
Interactive sandbox for **ContactManager**.

It supports:
• Fixed or LLM‑generated seed data.
• Voice or plain‑text input (same helpers as the other sandboxes).
• Automatic dispatch to `ask` *or* `update` depending on intent.
• Mid‑conversation interruption (pause / interject / cancel).
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
from pydantic import BaseModel, Field
from sandboxes.scenario_builder import ScenarioBuilder

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

# ────────────────────────────────  unity imports  ───────────────────────────
from unity.contact_manager.contact_manager import ContactManager
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

LG = logging.getLogger("contact_sandbox")

# ═════════════════════════════════ seed helpers ═════════════════════════════


async def _build_scenario(
    custom: Optional[str] = None,
) -> Optional[str]:
    """
    Populate the contact store **through the official tools** using
    :class:`ScenarioBuilder`.  Falls back to the fixed seed on any error.
    """
    cm = ContactManager()
    description = (
        custom.strip()
        if custom
        else (
            "Generate 10 realistic business contacts across EMEA, APAC and AMER. "
            "Each contact needs first_name, surname, email_address and phone_number. "
            "Also create custom columns with varying industries and locations."
        )
    )
    description += (
        "\nTry to get as much done as you can with each `update` and `ask` call. "
        "They can deal with complex multi-step requests just fine."
    )

    builder = ScenarioBuilder(
        description=description,
        tools={  # expose only the public surface
            "update": cm.update,
            "ask": cm.ask,  # allows the LLM to check for duplicates if it wishes
        },
    )

    try:
        await builder.create()
    except Exception as exc:
        raise (f"LLM seeding via ScenarioBuilder failed. {exc}")

    # The new flow doesn't produce a structured "theme"; preserve signature.
    return None


# ═════════════════════════════ intent dispatcher ════════════════════════════


class _Intent(BaseModel):
    action: str = Field(..., pattern="^(ask|update)$")
    cleaned_text: str


_INTENT_SYS_MSG = (
    "Decide whether the user input is a *query* about existing contacts "
    "or a *mutation* (create / update).  "
    "Return JSON {'action':'ask'|'update','cleaned_text':<fixed_input>}."
)


async def _dispatch_with_context(
    cm: ContactManager,
    raw: str,
    *,
    show_steps: bool,
    parent_chat_context: List[Dict[str, str]],
) -> Tuple[str, SteerableToolHandle]:
    """
    Same as :pyfunc:`_dispatch` but forwards *parent_chat_context* to the CM
    methods.  This indirection keeps the diff minimal.
    """

    # quick heuristic – verbs that virtually always imply an update
    if raw.lower().startswith(("add ", "create ", "update ", "change ", "delete ")):
        handle = await cm.update(
            raw,
            parent_chat_context=parent_chat_context,
            _return_reasoning_steps=show_steps,
        )
        return "update", handle

    # ask an LLM if less obvious
    judge = unify.Unify("gpt-4o@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )
    fn = cm.update if intent.action == "update" else cm.ask
    handle = await fn(
        raw,
        parent_chat_context=parent_chat_context,
        _return_reasoning_steps=show_steps,
    )
    return intent.action, handle


# ══════════════════════════════════  CLI  ═══════════════════════════════════


async def _main_async() -> None:
    parser = build_cli_parser("ContactManager sandbox")

    # No automatic seeding – users can invoke 'us' / 'usv' commands to populate contacts when desired.

    args = parser.parse_args()

    # tracing flag
    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    # ─────────────────── Unify context ────────────────────
    activate_project(args.project_name, args.overwrite)
    unify.set_trace_context("Traces")
    if args.overwrite:
        ctxs = unify.get_contexts()
        if "Contacts" in ctxs:
            unify.delete_context("Contacts")
        if "Traces" in ctxs:
            unify.delete_context("Traces")
        unify.create_context("Traces")

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

    # logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LG.setLevel(logging.INFO)

    # manager
    cm = ContactManager()
    if args.traced:
        cm = unify.traced(cm)

    # ─────────────────── optional initial seeding ─────────────────────────
    # No automatic seeding – users can invoke 'us' / 'usv' commands to populate contacts when desired.

    # ─────────────────── command helper output ────────────────────

    _COMMANDS_HELP = (
        "\nContactManager sandbox – type commands below (press ↵ with an empty "
        "line to dictate via voice when --voice mode is active – type 'r' to record).  'quit' to exit.\n\n"
        "┌────────────────── accepted commands ──────────────────┐\n"
        "│ us  {description}      – update_scenario (text)         │\n"
        "│ usv                   – update_scenario_vocally        │\n"
        "│ save_project | sp     – save project snapshot          │\n"
        "│ help | h              – show this help                  │\n"
        "└─────────────────────────────────────────────────────────┘\n"
    )

    def _explain_commands() -> None:  # noqa: D401 – helper
        print(_COMMANDS_HELP)

    if args.voice:
        _speak(
            "Sandbox ready. You can type commands, or press enter on an empty line "
            "to record a voice query.  Use 'u-s-v' to build a new scenario vocally.",
        )
        _wait_tts_end()

    # running memory of the dialogue
    chat_history: List[Dict[str, str]] = []

    # interaction loop
    while True:
        # Reprint the commands so they remain visible, mirroring MemoryManager sandbox
        print()
        _explain_commands()
        print()

        try:
            if args.voice:
                # Ensure any ongoing TTS playback has finished before showing prompt
                _wait_tts_end()
            if args.voice:
                # Voice mode: explicit prompt shows 'r' option
                raw = input("command ('r' to record)> ").strip()
                if raw.lower() == "r":
                    audio = _record_until_enter()
                    raw = _transcribe_deepgram(audio).strip()
                    if not raw:
                        continue
                    print(f"▶️  {raw}")
            else:
                raw = input("command> ").strip()

            # User can ask for the help table at any time
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
                    # Fallback to interactive prompt for description
                    description = input(
                        "🧮 Describe the contact scenario you want to build > ",
                    ).strip()
                    if not description:
                        print("⚠️  No description provided – cancelled.")
                        continue

                print(
                    "[generate] Building synthetic contacts – this can take a moment…",
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
                    "[generate] Building synthetic contacts – this can take a moment…",
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

            # ──────────────── remember the user's utterance ────────────────
            _kind, _handle = await _dispatch_with_context(
                cm,
                raw,
                show_steps=args.debug,
                parent_chat_context=list(chat_history),
            )
            chat_history.append({"role": "user", "content": raw})
            if args.voice:
                _speak("Let me take a look, give me a moment")

            answer = await _await_with_interrupt(_handle)
            if args.voice:
                _speak("Okay that's all done")
            if isinstance(answer, tuple):  # reasoning steps requested
                answer, _steps = answer
            print(f"[{_kind}] → {answer}\n")

            # ──────────────── remember the assistant's reply ───────────────
            chat_history.append({"role": "assistant", "content": answer})
            if args.voice:
                _speak(f"{answer} Anything else I can help with?")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
