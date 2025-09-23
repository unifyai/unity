"""intranet_sandbox.py  (optional voice mode, Deepgram SDK v4, sync)
====================================================================
Interactive sandbox for **IntranetRAGAgent**.

It supports:
• Fixed or LLM‑generated seed data via :class:`ScenarioBuilder`.
• Voice or plain‑text input (same helpers as the other sandboxes).
• Automatic dispatch to `ask` or `update` depending on intent.
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

# Always enable detailed request logging for sandbox runs BEFORE importing unify
os.environ["UNIFY_REQUESTS_DEBUG"] = "true"

# Added for graceful shutdown handling
import signal
import threading

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import intranet initialization
from intranet.scripts.utils import initialize_script_environment

if not initialize_script_environment():
    sys.exit(1)

from dotenv import load_dotenv

load_dotenv()

# # Honour LOG_LEVEL env var – if set to OFF/NONE/0 disable logging entirely
# if os.getenv("LOG_LEVEL", "INFO").upper() in {"OFF", "NONE", "0"}:
#     logging.disable(logging.CRITICAL)

# # Optional: silence INFO logs on stdout while retaining them in rag_agent.log
# if os.getenv("SILENT_CONSOLE", "false").lower() == "true":
#     logging.getLogger("UnifyAsyncLogger.EventBus").setLevel(logging.CRITICAL)
#     root_logger = logging.getLogger()
#     for _h in root_logger.handlers:
#         if isinstance(_h, logging.StreamHandler):
#             root_logger.removeHandler(_h)

# logging.getLogger("UnifyAsyncLogger").setLevel(logging.INFO)

import unify
from pydantic import BaseModel, Field
from sandboxes.scenario_builder import ScenarioBuilder

# ────────────────────────────────  unity imports  ───────────────────────────
from intranet.core.rag_agent import IntranetRAGAgent
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

LG = logging.getLogger("intranet_sandbox")

# ═════════════════════════════════ seed helpers ═════════════════════════════

_shutdown_requested = False


def _signal_handler(signum, _frame):
    """Catch SIGINT / SIGTERM and request a clean exit."""
    global _shutdown_requested
    _shutdown_requested = True

    sig_names = {signal.SIGINT: "SIGINT (CtrlC)", signal.SIGTERM: "SIGTERM"}
    name = sig_names.get(signum, f"Signal {signum}")
    print(f"\n🛑 Received {name} – requesting shutdown…")
    # Wake any waiter in the running loop
    if _loop_ref and _shutdown_event:
        try:
            _loop_ref.call_soon_threadsafe(_shutdown_event.set)
        except Exception:
            pass

    # Hard-exit after 10s if tasks don't complete
    def _force_exit():
        if _shutdown_requested:
            print("⏳ Graceful shutdown timed out – forcing exit.")
            # Avoid async teardown deadlocks
            sys.exit(1)

    _t = threading.Timer(10.0, _force_exit)
    _t.daemon = True
    _t.start()


# Register handlers early
for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, _signal_handler)


async def _build_scenario(
    custom: Optional[str] = None,
    *,
    clarifications_enabled: bool = True,
    enable_voice: bool = False,
) -> Optional[str]:
    """
    Populate the RAG knowledge base **through the official tools** using
    :class:`ScenarioBuilder`.  Falls back to the fixed seed on any error.
    """
    rag_agent = IntranetRAGAgent()
    description = (
        custom.strip()
        if custom
        else (
            "Generate a realistic knowledge base for an intranet with documents about "
            "company policies, technical documentation, project updates, and FAQ items. "
            "Create diverse content that would be found in a typical corporate intranet."
        )
    )
    description += (
        "\nTry to get as much done as you can with each `update` and `ask` call. "
        "They can deal with complex multi-step requests just fine."
    )

    builder = ScenarioBuilder(
        description=description,
        tools={  # expose only the public surface
            "update": rag_agent.update,
            "ask": rag_agent.ask,  # allows the LLM to check for duplicates if it wishes
        },
        enable_voice=enable_voice,
        clarifications_enabled=clarifications_enabled,
    )

    try:
        await builder.create()
    except Exception as exc:
        raise RuntimeError(f"LLM seeding via ScenarioBuilder failed. {exc}")

    # The new flow doesn't produce a structured "theme"; preserve signature.
    return None


# ═════════════════════════════ intent dispatcher ════════════════════════════


class _Intent(BaseModel):
    action: str = Field(..., pattern="^(ask|update)$")


_INTENT_SYS_MSG = (
    "You are an intent router for the IntranetRAGAgent.\n"
    "Decide if the user's input is a read-only question about existing knowledge ('ask') "
    "or a write/mutation that creates, updates, or modifies knowledge data ('update').\n"
    "Return ONLY JSON with this shape: {'action':'ask'|'update'}. Do not rewrite or summarize the user's input.\n"
    "- Classify as 'update' when the user asks to add, create, update, delete, write, store, insert, or otherwise produce/modify data.\n"
    "- Classify as 'ask' when the user is requesting information/lookup/search without modifying data (e.g., 'tell me about', 'what is', 'find information on', 'search for').\n"
    "Examples:\n"
    " - 'Add a new policy about remote work' → update\n"
    " - 'Store this document in the knowledge base' → update\n"
    " - 'What is our vacation policy?' → ask\n"
    " - 'Find information about the Q4 budget' → ask\n"
    " - 'Update the employee handbook with new guidelines' → update"
)


async def _dispatch_with_context(
    rag_agent: IntranetRAGAgent,
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
    Same as :pyfunc:`_dispatch` but forwards *parent_chat_context* to the RAG agent
    methods.  This indirection keeps the diff minimal.
    """

    judge = unify.Unify("gpt-5@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )
    fn = rag_agent.update if intent.action == "update" else rag_agent.ask
    handle, clar_up_q, clar_down_q = await call_manager_with_optional_clarifications(
        fn,
        raw,
        parent_chat_context=parent_chat_context,
        return_reasoning_steps=show_steps,
        clarifications_enabled=clarifications_enabled,
    )

    # Speak an acknowledgement if voice mode is on so users know work began
    if enable_voice:
        try:
            _speak("Working on it.")
        except Exception:
            pass

    return intent.action, handle, clar_up_q, clar_down_q


# ══════════════════════════════════  CLI  ═══════════════════════════════════


async def _main_async() -> None:
    parser = build_cli_parser("IntranetRAGAgent sandbox")

    # No automatic seeding – users can invoke 'us' / 'usv' commands to populate knowledge when desired.

    args = parser.parse_args()

    # tracing flag
    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    # ─────────────────── Unify context ────────────────────
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

    # logging via shared helper
    configure_sandbox_logging(
        log_in_terminal=args.log_in_terminal,
        log_file=".logs_main.txt",
        tcp_port=args.log_tcp_port,
        http_tcp_port=args.http_log_tcp_port,
        unify_requests_log_file=".logs_unify_requests.txt",
    )
    LG.setLevel(logging.INFO)

    # Attach to the existing "Intranet" project and instantiate the RAG agent
    os.environ["RAG_SKIP_INIT"] = (
        "true"  # assume intranet/scripts/01_initialize_system.py already ran
    )

    # manager
    rag_agent = IntranetRAGAgent(sandbox_mode=True)
    if args.traced:
        rag_agent = unify.traced(rag_agent)

    # ─────────────────── optional initial seeding ─────────────────────────
    # No automatic seeding – users can invoke 'us' / 'usv' commands to populate knowledge when desired.

    # ─────────────────── command helper output ────────────────────

    _COMMANDS_HELP = (
        "\nIntranetRAGAgent sandbox – type commands below (press ↵ with an empty "
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
                        "🧮 Describe the knowledge scenario you want to build > ",
                    ).strip()
                    if not description:
                        print("⚠️  No description provided – cancelled.")
                        continue

                if args.voice:
                    task = asyncio.create_task(
                        _build_scenario(
                            description,
                            clarifications_enabled=not args.no_clarifications,
                            enable_voice=bool(args.voice),
                        ),
                    )
                    _speak_wait("Got it, working on your custom scenario now.")
                    print(
                        "[generate] Building synthetic knowledge base – this can take a moment…",
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
                        "[generate] Building synthetic knowledge base – this can take a moment…",
                    )
                    try:
                        await _build_scenario(
                            description,
                            clarifications_enabled=not args.no_clarifications,
                            enable_voice=False,
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

                task = asyncio.create_task(
                    _build_scenario(
                        description,
                        clarifications_enabled=not args.no_clarifications,
                        enable_voice=bool(args.voice),
                    ),
                )
                _speak_wait("Got it, working on your custom scenario now.")
                print(
                    "[generate] Building synthetic knowledge base – this can take a moment…",
                )
                try:
                    await task
                    _speak_wait(
                        "All done, your custom scenario is built and ready to go.",
                    )
                except Exception as exc:
                    LG.error("Scenario generation failed: %s", exc, exc_info=True)
                    print(f"❌  Failed to generate scenario: {exc}")
                continue  # back to REPL

            # Ignore steering commands when no request is running
            if raw.startswith("/"):
                print(
                    "(no active request) Steering commands are only available while a call is running.",
                )
                continue

            # ──────────────── remember the user's utterance ────────────────
            _kind, _handle, _clar_up, _clar_down = await _dispatch_with_context(
                rag_agent,
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
