"""
Starts the conversation manager.

This script can be run as a CLI with the following arguments:
    --local         Enable local GUI mode (default).
    --full          Disable local GUI mode (real comms and no GUI).
    --enabled_tools Comma-separated list of enabled tools (choices: conductor, contact, transcript, knowledge, scheduler, comms). Default: None
"""

import asyncio
import signal
import time
import os
from dotenv import load_dotenv
import logging, unify
import unity.conversation_manager
from sandboxes.utils import build_cli_parser, activate_project
from datetime import datetime
from sandboxes.utils import (
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    _wait_for_tts_end,
    TranscriptGenerator,
)

LG = logging.getLogger("contact_sandbox")

load_dotenv(override=True)


# Graceful shutdown handler
def signal_handler(signum, frame):
    print("Shutting down convo manager...")
    unity.conversation_manager.stop("signal_shutdown")
    exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


async def interaction_loop(args):
    # CLI interaction loop for sending events and scenario seeding
    _COMMANDS_HELP = (
        "\nConversationManager sandbox – type commands below ('r' to record voice with --voice). 'quit' to exit.\n\n"
        "┌────────────────── accepted commands ─────────────────────┐\n"
        "│ us <description>         – update scenario (text)        │\n"
        "│ usv                      – update scenario vocally       │\n"
        "│ save_project | sp        – save project snapshot         │\n"
        "│ help | h                 – show this help                │\n"
        "└──────────────────────────────────────────────────────────┘\n"
    )

    def _explain_commands():
        print(_COMMANDS_HELP)

    if args.voice:
        _speak("Sandbox ready. Type commands or press enter on empty for voice input.")
        _wait_for_tts_end()

    while True:
        print()
        _explain_commands()
        print()
        try:
            if args.voice:
                _wait_for_tts_end()
                raw = input("command ('r' to record)> ").strip()
                if raw.lower() == "r":
                    audio = _record_until_enter()
                    raw = _transcribe_deepgram(audio).strip()
                    if not raw:
                        continue
                    print(f"▶️  {raw}")
            else:
                raw = input("command> ").strip()
            cmd_lower = raw.lower()
            if cmd_lower in {"help", "h", "?"}:
                _explain_commands()
                continue
            if cmd_lower in {"quit", "exit"}:
                break
            if not raw:
                continue
            if cmd_lower in {"save_project", "sp"}:
                commit_hash = unify.commit_project(
                    args.project_name,
                    commit_message=f"Sandbox save {datetime.utcnow().isoformat()}",
                ).get("commit_hash")
                print(f"💾 Project saved at commit {commit_hash}")
                if args.voice:
                    _speak("Project saved")
                continue
            # Scenario seeding (text)
            if cmd_lower.startswith("us "):
                description = raw[3:].strip()
                if not description:
                    description = input("🧮 Describe scenario > ").strip()
                    if not description:
                        print("⚠️ No description provided – cancelled.")
                        continue
                print("[generate] Building scenario – please wait…")
                if args.voice:
                    _speak("Building scenario now.")
                gen = TranscriptGenerator()
                try:
                    messages = await gen.generate(description)
                    print(f"✓ Transcript generated: {len(messages)} messages")
                    if args.voice:
                        _speak("Scenario generation complete.")
                except Exception as exc:
                    print(f"❌ Failed to generate scenario: {exc}")
                continue
            # Scenario seeding (voice)
            if cmd_lower == "usv":
                if not args.voice:
                    print("⚠️ Voice mode not enabled – restart with --voice.")
                    continue
                audio = _record_until_enter()
                description = _transcribe_deepgram(audio).strip()
                if not description:
                    print("⚠️ No transcript – please try again.")
                    continue
                print(f"▶️  {description}")
                print("[generate] Building scenario – please wait…")
                gen = TranscriptGenerator()
                try:
                    messages = await gen.generate(description)
                    print(f"✓ Transcript generated: {len(messages)} messages")
                except Exception as exc:
                    print(f"❌ Failed to generate scenario: {exc}")
                if args.voice:
                    _speak("Scenario generation complete.")
                continue
            print(f"⚠️ Unknown command: {raw}")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break
        except Exception as exc:
            LG.error("[error] %s", exc)
            continue


async def main():
    # CLI flags (local/gui, tools) + unified project, tracing, debug
    parser = build_cli_parser("ConversationManager sandbox")
    parser.add_argument(
        "--local",
        dest="start_local",
        action="store_true",
        default=True,
        help="Enable local GUI mode",
    )
    parser.add_argument(
        "--full",
        dest="start_local",
        action="store_false",
        help="Disable local GUI mode (real comms and no GUI)",
    )
    parser.add_argument(
        "--enabled_tools",
        dest="enabled_tools",
        type=lambda s: [t.strip() for t in s.split(",")],
        default=None,
        help="Comma-separated list of enabled tools: conductor, contact, transcript, knowledge, scheduler, comms",
    )
    args = parser.parse_args()

    if args.start_local:
        # tracing flag
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

    # from unity.helpers import run_script
    # proc = run_script("sandboxes/conversation_manager/gui.py", terminal=True)
    # proc.wait()
    # exit(0)

    # Start the convo manager
    print("Starting convo manager...")
    if unity.conversation_manager.start(
        start_local=args.start_local,
        enabled_tools=(
            ",".join(args.enabled_tools)
            if isinstance(args.enabled_tools, list)
            else args.enabled_tools
        ),
    ):
        print("Convo manager started successfully...")

        from unity.helpers import run_script

        if args.start_local:
            proc = run_script(
                "sandboxes/conversation_manager/gui.py",
                args.project_name,
                terminal=True,
            )

            await interaction_loop(args)

            proc.wait()
            unity.conversation_manager.stop("signal_shutdown")

        # Keep running until the convo manager process is dead
        while unity.conversation_manager.is_running():
            time.sleep(1)  # Check every second

        # Final status
        status = unity.conversation_manager.get_status()
        print(
            f"Convo manager has stopped. Reason: {status.get('shutdown_reason', 'unknown')}",
        )
        if "message" in status:
            print(f"Details: {status['message']}")
    else:
        print("Failed to start convo manager")
        # exit(1)


if __name__ == "__main__":
    asyncio.run(main())
