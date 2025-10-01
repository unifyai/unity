"""secret_sandbox.py (optional voice mode, Deepgram SDK v4, sync)
=================================================================
Interactive sandbox for SecretManager.

It supports:
- Free text input auto-routed to ask or update (intent router)
- Explicit from_placeholder/to_placeholder commands
- Optional voice input/output (same helpers as other sandboxes)
- In-flight steering via SteerableToolHandle
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

# Always enable detailed request logging for sandbox runs BEFORE importing unify
os.environ["UNIFY_REQUESTS_DEBUG"] = "true"

from dotenv import load_dotenv

load_dotenv()

import unify
from pydantic import BaseModel, Field

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ────────────────────────────────  unity imports  ───────────────────────────
from unity.secret_manager.secret_manager import SecretManager
from unity.common.async_tool_loop import SteerableToolHandle
from sandboxes.utils import (
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    await_with_interrupt as _await_with_interrupt,
    steering_controls_hint as _steer_hint,
    build_cli_parser,
    activate_project,
    _wait_for_tts_end as _wait_tts_end,
    configure_sandbox_logging,
    call_manager_with_optional_clarifications,
)

LG = logging.getLogger("secret_sandbox")


# ═════════════════════════════ intent dispatcher ════════════════════════════


class _Intent(BaseModel):
    action: str = Field(..., pattern="^(ask|update)$")


_INTENT_SYS_MSG = (
    "You are an intent router for the SecretManager.\n"
    "Decide if the user's input is a read-only question about existing secrets ('ask') "
    "or a write/mutation that creates, updates, or deletes secrets ('update').\n"
    "Return ONLY JSON with this shape: {'action':'ask'|'update'}. Do not rewrite or summarize the user's input.\n"
    "- Classify as 'update' when the user asks to set, add, create, change, update, delete, write, generate, populate, assign or otherwise modify secrets (names/values/descriptions), including bulk operations.\n"
    "- Classify as 'ask' when the user is requesting information/lookup/reporting without modifying data (e.g., 'show placeholders', 'list secret keys', 'what placeholders exist').\n"
    "Examples:\n"
    " - 'Create a secret unify_key with this value ...' → update\n"
    " - 'Update db_password to a new value ...' → update\n"
    " - 'List secret keys' → ask\n"
    " - 'Show placeholder for the staging DB password' → ask"
)


async def _dispatch_with_context(
    sm: SecretManager,
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
    Route free text to SecretManager.ask or SecretManager.update and forward
    parent_chat_context. Mirrors the ContactManager sandbox style.
    """

    judge = unify.Unify("gpt-5@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )
    fn = sm.update if intent.action == "update" else sm.ask
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
    parser = build_cli_parser("SecretManager sandbox")

    args = parser.parse_args()

    # tracing flag
    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    # ─────────────────── Unify context ────────────────────
    activate_project(args.project_name, args.overwrite)

    # ─────────────────── project version handling ────────────────────
    if getattr(args, "project_version", -1) != -1:
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

    # logging via shared helper
    configure_sandbox_logging(
        log_in_terminal=args.log_in_terminal,
        log_file=".logs_secret_main.txt",
        tcp_port=args.log_tcp_port,
        http_tcp_port=args.http_log_tcp_port,
        unify_requests_log_file=".logs_unify_requests.txt",
    )
    LG.setLevel(logging.INFO)

    # manager
    sm = SecretManager()
    if args.traced:
        sm = unify.traced(sm)

    # ─────────────────── command helper output ────────────────────
    _COMMANDS_HELP = (
        "\nSecretManager sandbox – type commands below (press ↵ with an empty "
        "line to dictate via voice when --voice mode is active – type 'r' to record).  'quit' to exit.\n\n"
        "┌──────────────────── accepted commands ─────────────────────┐\n"
        "│ from_placeholder {text} – resolve ${NAME} placeholders      │\n"
        "│ to_placeholder {text}   – redact known values → ${NAME}     │\n"
        "│ r / free text      – freeform ask / update (auto)           │\n"
        "│ save_project | sp  – save project snapshot                  │\n"
        "│ help | h           – show this help                         │\n"
        "└────────────────────────────────────────────────────────────┘\n"
    )

    def _explain_commands() -> None:  # noqa: D401 – helper
        print(_COMMANDS_HELP)

    if args.voice:
        _speak(
            "Sandbox ready. You can type commands, or press enter on an empty line "
            "to record a voice query.",
        )
        _wait_tts_end()

    # running memory of the dialogue
    chat_history: List[Dict[str, str]] = []

    # interaction loop
    while True:
        print()
        _explain_commands()
        print()

        try:
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

            # save project snapshot
            if raw.lower() in {"save_project", "sp"}:
                commit_hash = unify.commit_project(
                    args.project_name,
                    commit_message=f"Secret sandbox save {datetime.utcnow().isoformat()}",
                ).get("commit_hash")
                print(f"💾 Project saved at commit {commit_hash}")
                if args.voice:
                    _speak("Project saved")
                continue

            parts = raw.split(maxsplit=1)
            cmd_lower = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""

            # explicit from_placeholder/to_placeholder helpers (no LLM)
            if cmd_lower == "from_placeholder":
                if not rest:
                    print("⚠️  Provide text containing ${NAME} placeholders to resolve.")
                    continue
                try:
                    out = asyncio.get_event_loop().run_until_complete(
                        sm.from_placeholder(rest),
                    )
                except RuntimeError:
                    out = await sm.from_placeholder(rest)
                print(f"[from_placeholder] → {out}\n")
                chat_history.append({"role": "assistant", "content": out})
                continue

            if cmd_lower == "to_placeholder":
                if not rest:
                    print(
                        "⚠️  Provide text containing raw secret values to redact to ${NAME}.",
                    )
                    continue
                try:
                    out = asyncio.get_event_loop().run_until_complete(
                        sm.to_placeholder(rest),
                    )
                except RuntimeError:
                    out = await sm.to_placeholder(rest)
                print(f"[to_placeholder] → {out}\n")
                chat_history.append({"role": "assistant", "content": out})
                continue

            # Ignore steering commands when no request is running
            if raw.startswith("/"):
                print(
                    "(no active request) Steering commands are only available while a call is running.",
                )
                continue

            # free-form ask/update via intent router
            _kind, _handle, _clar_up, _clar_down = await _dispatch_with_context(
                sm,
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
            if isinstance(answer, tuple):
                answer, _steps = answer
            print(f"[{_kind}] → {answer}\n")

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
