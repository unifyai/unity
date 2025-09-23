"""knowledge_sandbox.py  (optional voice mode, Deepgram SDK v4, sync)
====================================================================
Interactive sandbox for **KnowledgeManager**.

It supports:
• Fixed or LLM‑generated seed data.
• Voice or plain‑text input (same helpers as the other sandboxes).
• Automatic dispatch to `ask`, `update` or `refactor` depending on intent.
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

from dotenv import load_dotenv

load_dotenv()

import unify
from pydantic import BaseModel, Field
from sandboxes.scenario_builder import ScenarioBuilder

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ────────────────────────────────  unity imports  ───────────────────────────
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.common.async_tool_loop import SteerableToolHandle
from sandboxes.utils import (  # shared helpers reused in other sandboxes
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    speak_and_wait as _speak_wait,
    await_with_interrupt as _await_with_interrupt,
    steering_controls_hint as _steer_hint,
    build_cli_parser,
    activate_project,
    _wait_for_tts_end as _wait_tts_end,
    configure_sandbox_logging,
    call_manager_with_optional_clarifications,
)

LG = logging.getLogger("knowledge_sandbox")

# ═════════════════════════════════ seed helpers ═════════════════════════════


async def _build_scenario(custom: Optional[str] = None) -> Optional[str]:
    """
    Populate the knowledge store **via the official tools** (`update` / `ask`)
    using :class:`ScenarioBuilder`.  Falls back to the fixed seed on error.
    """
    km = KnowledgeManager()

    description = (
        custom.strip()
        if custom
        else (
            "Generate 20 diverse facts about electric-vehicle manufacturers. "
            "Cover launch years, battery capacities, warranty terms and sales "
            "figures in different regions.  Include numbers, dates and named "
            "entities so the schema has to evolve."
        )
    )
    description += (
        "\nTry to batch actions – each `store` can add multiple rows/columns "
        "and `retrieve` can verify to avoid duplication."
    )

    builder = ScenarioBuilder(
        description=description,
        tools={
            "update": km.update,
            "ask": km.ask,
        },
    )

    try:
        await builder.create()
    except Exception as exc:
        raise RuntimeError(f"LLM seeding via ScenarioBuilder failed. {exc}")

    return None


# ═════════════════════════════ intent dispatcher ════════════════════════════


class _Intent(BaseModel):
    action: str = Field(..., pattern="^(ask|update|refactor)$")
    cleaned_text: str


_INTENT_SYS_MSG = (
    "Classify the user's message into exactly one of: 'ask' | 'update' | 'refactor'.\n"
    "- ask: read-only retrieval or analysis over existing knowledge.\n"
    "- update: add or modify rows/columns/tables.\n"
    "- refactor: schema normalization or structural changes (rename/split/move columns, joins migration).\n"
    "Return ONLY JSON: {'action': 'ask'|'update'|'refactor'}"
)


async def _dispatch_with_context(
    km: KnowledgeManager,
    raw: str,
    *,
    show_steps: bool,
    parent_chat_context: List[Dict[str, str]],
    clarifications_enabled: bool,
    enable_voice: bool,
) -> Tuple[
    str,
    SteerableToolHandle,
    Optional[asyncio.Queue[str]],
    Optional[asyncio.Queue[str]],
]:
    """
    Figure out whether to call `store`, `retrieve` or `refactor`, forwarding
    *parent_chat_context* to the KnowledgeManager methods.
    """

    lowered = raw.lower()

    # ───── quick heuristics (fast-path) ───────────────────────────────
    if lowered.startswith(
        (
            "add ",
            "create ",
            "update ",
            "change ",
            "delete ",
            "store ",
            "remember ",
            "note ",
        ),
    ):
        handle, clar_up_q, clar_down_q = (
            await call_manager_with_optional_clarifications(
                km.update,
                raw,
                parent_chat_context=parent_chat_context,
                return_reasoning_steps=show_steps,
                clarifications_enabled=clarifications_enabled,
            )
        )
        # Speak an acknowledgement if voice mode is on
        if enable_voice:
            try:
                _speak("Working on it.")
            except Exception:
                pass
        return "update", handle, clar_up_q, clar_down_q

    if lowered.startswith(
        (
            "refactor ",
            "restructure ",
            "normalize ",
            "normalise ",
            "schema ",
        ),
    ):
        handle, clar_up_q, clar_down_q = (
            await call_manager_with_optional_clarifications(
                km.refactor,
                raw,
                parent_chat_context=parent_chat_context,
                return_reasoning_steps=show_steps,
                clarifications_enabled=clarifications_enabled,
            )
        )
        if enable_voice:
            try:
                _speak("Working on it.")
            except Exception:
                pass
        return "refactor", handle, clar_up_q, clar_down_q

    # ───── everything else – ask an LLM judge ────────────────────────
    judge = unify.Unify("gpt-5@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )

    fn = (
        km.ask
        if intent.action == "ask"
        else km.update if intent.action == "update" else km.refactor
    )
    handle, clar_up_q, clar_down_q = await call_manager_with_optional_clarifications(
        fn,
        raw,
        parent_chat_context=parent_chat_context,
        return_reasoning_steps=show_steps,
        clarifications_enabled=clarifications_enabled,
    )
    if enable_voice:
        try:
            _speak("Working on it.")
        except Exception:
            pass
    return intent.action, handle, clar_up_q, clar_down_q


# ══════════════════════════════════  CLI  ═══════════════════════════════════


async def _main_async() -> None:
    parser = build_cli_parser("KnowledgeManager sandbox")
    args = parser.parse_args()

    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    activate_project(args.project_name, args.overwrite)

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

    configure_sandbox_logging(
        log_in_terminal=args.log_in_terminal,
        log_file=".logs_main.txt",
        tcp_port=args.log_tcp_port,
        http_tcp_port=args.http_log_tcp_port,
        unify_requests_log_file=".logs_unify_requests.txt",
    )
    LG.setLevel(logging.INFO)

    km = KnowledgeManager()
    if args.traced:
        km = unify.traced(km)

    _COMMANDS_HELP = (
        "\nKnowledgeManager sandbox – type commands below (press ↵ with an empty "
        "line to dictate via voice when --voice mode is active – type 'r' to record).  'quit' to exit.\n\n"
        "┌────────────────── accepted commands ─────────────────────┐\n"
        "│ us  {description}     – update_scenario (text)           │\n"
        "│ usv                   – update_scenario_vocally          │\n"
        "│ r / free text         – freeform ask / update (auto)     │\n"
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

    # running memory of the dialogue
    chat_history: List[Dict[str, str]] = []

    # interaction loop
    while True:
        try:
            print()
            _explain_commands()
            print()

            if args.voice:
                _wait_tts_end()
            if args.voice:
                raw = input("command ('r' to record)> ").strip()
                if raw.lower() == "r":
                    audio = _record_until_enter()
                    raw = _transcribe_deepgram(audio).strip()
                    if not raw:
                        continue
                    print(f"▶️  {raw}")
            else:
                raw = input("command> ").strip()

            if raw.lower() in {"help", "h", "?"}:
                _explain_commands()
                continue

            if raw.lower() in {"quit", "exit"}:
                break
            if not raw:
                continue

            if raw.lower() in {"save_project", "sp"}:
                commit_hash = unify.commit_project(
                    args.project_name,
                    commit_message=f"Sandbox save {datetime.utcnow().isoformat()}",
                ).get("commit_hash")
                print(f"💾 Project saved at commit {commit_hash}")
                if args.voice:
                    _speak("Project saved")
                continue

            parts = raw.split(maxsplit=1)
            cmd_lower = parts[0].lower()

            if cmd_lower in {"us", "update_scenario"}:
                description = parts[1].strip() if len(parts) > 1 else ""
                if not description:
                    description = input(
                        "🧮 Describe the knowledge scenario you want to build > ",
                    ).strip()
                    if not description:
                        print("⚠️  No description provided – cancelled.")
                        continue

                if args.voice:
                    task = asyncio.create_task(_build_scenario(description))
                    _speak_wait("Got it, working on your custom scenario now.")
                    print(
                        "[generate] Building synthetic knowledge – this can take a moment…",
                    )
                    try:
                        await task
                        _speak_wait(
                            "All done, your custom scenario is built and ready to go.",
                        )
                    except Exception as exc:
                        LG.error("Scenario generation failed: %s", exc, exc_info=True)
                        print(f"❌  Failed to generate scenario: {exc}")
                else:
                    print(
                        "[generate] Building synthetic knowledge – this can take a moment…",
                    )
                    try:
                        await _build_scenario(description)
                    except Exception as exc:
                        LG.error("Scenario generation failed: %s", exc, exc_info=True)
                        print(f"❌  Failed to generate scenario: {exc}")
                continue

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

                task = asyncio.create_task(_build_scenario(description))
                _speak_wait("Got it, working on your custom scenario now.")
                print(
                    "[generate] Building synthetic knowledge – this can take a moment…",
                )
                try:
                    await task
                    _speak_wait(
                        "All done, your custom scenario is built and ready to go.",
                    )
                except Exception as exc:
                    LG.error("Scenario generation failed: %s", exc, exc_info=True)
                    print(f"❌  Failed to generate scenario: {exc}")
                continue

            if raw.startswith("/"):
                print(
                    "(no active request) Steering commands are only available while a call is running.",
                )
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

            # ──────────────── remember the user's utterance ────────────────
            _kind, _handle, _clar_up, _clar_down = await _dispatch_with_context(
                km,
                raw,
                show_steps=args.debug,
                parent_chat_context=list(chat_history),
                clarifications_enabled=not args.no_clarifications,
                enable_voice=bool(args.voice),
            )
            chat_history.append({"role": "user", "content": raw})
            if args.voice:
                _speak("Let me take a look, give me a moment")
                _wait_tts_end()

            print(_steer_hint(voice_enabled=bool(args.voice)))
            answer = await _await_with_interrupt(
                _handle,
                enable_voice_steering=bool(args.voice),
                clarification_up_q=_clar_up,
                clarification_down_q=_clar_down,
                clarifications_enabled=not args.no_clarifications,
                chat_context=list(chat_history),
            )
            if args.voice:
                _speak("Okay that's all done")
                _wait_tts_end()
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
