"""web_searcher/sandbox.py – interactive sandbox for WebSearcher.
Mirrors design and flow of the ContactManager sandbox.
"""

from __future__ import annotations

# stdlib / env
import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Enable request logging for sandbox runs before importing unify
os.environ["UNIFY_REQUESTS_DEBUG"] = "true"

from dotenv import load_dotenv

load_dotenv()

import unify

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# unity imports
from unity.web_searcher.web_searcher import WebSearcher
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

LG = logging.getLogger("web_searcher_sandbox")


async def _dispatch(
    ws: WebSearcher,
    raw: str,
    *,
    show_steps: bool,
    parent_chat_context: List[Dict[str, str]],
    clarifications_enabled: bool,
    enable_voice: bool,
) -> tuple[
    str,
    SteerableToolHandle,
    Optional[asyncio.Queue[str]],
    Optional[asyncio.Queue[str]],
]:
    """Dispatch a web research question to WebSearcher.ask (with optional clarifications)."""
    handle, clar_up_q, clar_down_q = await call_manager_with_optional_clarifications(
        ws.ask,
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
    return "ask", handle, clar_up_q, clar_down_q


async def _main_async() -> None:
    parser = build_cli_parser("WebSearcher sandbox")

    args = parser.parse_args()

    # tracing flag
    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    # Unify context
    activate_project(args.project_name, args.overwrite)

    # Optional rollback to previous commit
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

    # logging via shared helper
    configure_sandbox_logging(
        log_in_terminal=args.log_in_terminal,
        log_file=".logs_web_searcher.txt",
        tcp_port=args.log_tcp_port,
        http_tcp_port=args.http_log_tcp_port,
        unify_requests_log_file=".logs_unify_requests.txt",
    )
    LG.setLevel(logging.INFO)

    ws = WebSearcher()
    if args.traced:
        ws = unify.traced(ws)  # type: ignore[assignment]

    _COMMANDS_HELP = (
        "\nWebSearcher sandbox – type queries below (press ↵ with an empty "
        "line to dictate via voice when --voice mode is active – type 'r' to record).  'quit' to exit.\n\n"
        "┌────────────────── accepted commands ─────────────────────┐\n"
        "│ r / free text   – freeform ask (web research)            │\n"
        "│ save_project | sp – save project snapshot                │\n"
        "│ help | h         – show this help                        │\n"
        "└──────────────────────────────────────────────────────────┘\n"
    )

    def _explain_commands() -> None:
        print(_COMMANDS_HELP)

    if args.voice:
        _speak(
            "Sandbox ready. You can type queries, or press enter on an empty line to record a voice query.",
        )
        _wait_tts_end()

    chat_history: List[Dict[str, str]] = []

    while True:
        print()
        _explain_commands()
        print()

        try:
            if args.voice:
                _wait_tts_end()
            if args.voice:
                raw = input("query ('r' to record)> ").strip()
                if raw.lower() == "r":
                    audio = _record_until_enter()
                    raw = _transcribe_deepgram(audio).strip()
                    if not raw:
                        continue
                    print(f"▶️  {raw}")
            else:
                raw = input("query> ").strip()

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
                    commit_message="WebSearcher sandbox save",
                ).get("commit_hash")
                print(f"💾 Project saved at commit {commit_hash}")
                if args.voice:
                    _speak("Project saved")
                continue

            kind, handle, _clar_up, _clar_down = await _dispatch(
                ws,
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
                handle,
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
            print(f"[{kind}] → {answer}\n")

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
