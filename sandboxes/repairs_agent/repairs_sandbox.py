from __future__ import annotations

import os
import sys
import asyncio
import signal
import threading
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Ensure repository root on path for local execution
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intranet.scripts.utils import initialize_script_environment, activate_project

if not initialize_script_environment():
    sys.exit(1)

from dotenv import load_dotenv

load_dotenv()

# Always enable detailed request logging for sandbox runs BEFORE importing unify
os.environ["UNIFY_REQUESTS_DEBUG"] = "false"

import unify
from intranet.core.repairs_agent import RepairsAgent
from sandboxes.utils import (
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    await_with_interrupt as _await_with_interrupt,
    steering_controls_hint as _steer_hint,
    build_cli_parser,
    _wait_for_tts_end as _wait_tts_end,
    configure_sandbox_logging,
    call_manager_with_optional_clarifications,
)

LG = logging.getLogger("repairs_sandbox")

_shutdown_requested = False
_loop_ref: asyncio.AbstractEventLoop | None = None
_shutdown_event: asyncio.Event | None = None


def _signal_handler(signum, _frame):
    global _shutdown_requested
    _shutdown_requested = True
    sig_names = {signal.SIGINT: "SIGINT (CtrlC)", signal.SIGTERM: "SIGTERM"}
    name = sig_names.get(signum, f"Signal {signum}")
    print(f"\n🛑 Received {name} – requesting shutdown…")
    if _loop_ref and _shutdown_event:
        try:
            _loop_ref.call_soon_threadsafe(_shutdown_event.set)
        except Exception:
            pass

    def _force_exit():
        if _shutdown_requested:
            print("⏳ Graceful shutdown timed out – forcing exit.")
            sys.exit(1)

    _t = threading.Timer(10.0, _force_exit)
    _t.daemon = True
    _t.start()


for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, _signal_handler)


def _commands_help() -> str:
    return (
        "\nRepairsAgent sandbox – type commands below (press ↵ with an empty line "
        "to dictate via voice when --voice mode is active – type 'r' to record).  'quit' to exit.\n\n"
        "┌────────────────── accepted commands ─────────────────────┐\n"
        "│ us  {description}     – update_scenario (text) [not supported] │\n"
        "│ usv                   – update_scenario_vocally [not supported] │\n"
        "│ r / free text         – freeform ask                     │\n"
        "│ save_project | sp     – save project snapshot            │\n"
        "│ help | h              – show this help                   │\n"
        "└──────────────────────────────────────────────────────────┘\n"
    )


async def _main_async() -> None:
    parser = build_cli_parser("RepairsAgent sandbox")
    args = parser.parse_args()

    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"
    # Activate project per requirement
    activate_project("Repairs", overwrite=False)

    # Optional version rollback (mirror intranet sandbox semantics)
    if args.project_version != -1:
        commits = unify.get_project_commits("Repairs")
        if commits:
            try:
                target = commits[args.project_version]
                unify.rollback_project("Repairs", target["commit_hash"])
                LG.info("[version] Rolled back to commit %s", target["commit_hash"])
            except IndexError:
                LG.warning(
                    "[version] project_version index %s out of range, ignoring",
                    args.project_version,
                )

    configure_sandbox_logging(
        log_in_terminal=args.log_in_terminal,
        log_file=".logs_repairs_agent.txt",
        tcp_port=args.log_tcp_port,
        http_tcp_port=args.http_log_tcp_port,
        unify_requests_log_file=".logs_unify_requests.txt",
    )
    LG.setLevel(logging.INFO)

    # Agent in sandbox mode (returns handle)
    agent = RepairsAgent(sandbox_mode=True)
    if args.traced:
        agent = unify.traced(agent)  # type: ignore

    if args.voice:
        _speak(
            "Repairs Agent ready. You can type your question, or press enter on an empty line "
            "to record a voice query with 'r'.",
        )
        _wait_tts_end()

    print(_commands_help())
    chat_history: List[Dict[str, str]] = []
    global _loop_ref, _shutdown_event
    _loop_ref = asyncio.get_running_loop()
    _shutdown_event = asyncio.Event()

    while True:
        try:
            print()
            print(_commands_help())
            print()
            if args.voice:
                # Ensure any ongoing TTS playback has finished before showing prompt
                _wait_tts_end()
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
                print(_commands_help())
                continue
            if raw.lower() in {"quit", "exit"}:
                break
            if not raw:
                continue

            # Ignore steering commands when no request is running
            if raw.startswith("/"):
                print(
                    "(no active request) Steering commands are only available while a call is running.",
                )
                continue

            # Save project snapshot
            if raw.lower() in {"save_project", "sp"}:
                commit_hash = unify.commit_project(
                    "Repairs",
                    commit_message=f"Repairs sandbox save {datetime.utcnow().isoformat()}",
                ).get("commit_hash")
                print(f"💾 Project saved at commit {commit_hash}")
                if args.voice:
                    _speak("Project saved")
                continue

            # Not supported in RepairsAgent (no update flow)
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            if cmd in {"us", "update_scenario", "usv", "update_scenario_vocally"}:
                print(
                    "⚠️  Scenario update is not supported in RepairsAgent. Please ask questions (read-only) instead.",
                )
                continue

            # Ask flow
            if args.voice:
                _speak("Working on it.")
                _wait_tts_end()
            handle, clar_up, clar_down = (
                await call_manager_with_optional_clarifications(
                    agent.ask,
                    raw,
                    parent_chat_context=list(chat_history),
                    return_reasoning_steps=args.debug,
                    clarifications_enabled=not args.no_clarifications,
                )
            )
            # Remember the user's utterance before awaiting result
            chat_history.append({"role": "user", "content": raw})
            print(_steer_hint(voice_enabled=bool(args.voice)))
            answer = await _await_with_interrupt(
                handle,
                enable_voice_steering=bool(args.voice),
                clarification_up_q=clar_up,
                clarification_down_q=clar_down,
                clarifications_enabled=not args.no_clarifications,
                chat_context=list(chat_history),
            )
            if args.voice:
                _speak("Okay that's all done")
                _wait_tts_end()
            if isinstance(answer, tuple):
                answer, _steps = answer
            print(f"[ask] → {answer}\n")
            chat_history.append({"role": "assistant", "content": str(answer)})
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
