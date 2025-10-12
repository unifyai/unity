"""file_manager_sandbox.py  (optional voice mode, Deepgram SDK v4, sync)
====================================================================
Interactive sandbox for **FileManager**.

It supports:
• File import from local filesystem or URLs.
• Voice or plain‑text input (same helpers as the other sandboxes).
• Automatic dispatch to `ask`, `ask_about_file`, or `organize` depending on intent.
• Mid‑conversation interruption (pause / interject / cancel).
• File parsing and content analysis.
• Support for multiple filesystem adapters (Local, CodeSandbox, Interact, GoogleDrive).
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
from sandboxes.scenario_builder import ScenarioBuilder

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ────────────────────────────────  unity imports  ───────────────────────────
from unity.file_manager.managers.file_manager import FileManager
from unity.file_manager.managers.local import LocalFileManager
from unity.file_manager.fs_adapters.local_adapter import LocalFileSystemAdapter
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

LG = logging.getLogger("file_manager_sandbox")

# ═════════════════════════════════ seed helpers ═════════════════════════════


async def _build_scenario(
    fm: FileManager,
    custom: Optional[str] = None,
    *,
    clarifications_enabled: bool = True,
    enable_voice: bool = False,
) -> Optional[str]:
    """
    Populate the file store with sample files and demonstrate FileManager capabilities.
    """
    # Manually populate sample files
    sample_dir = (
        Path(__file__).parent.parent.parent / "tests" / "test_file_manager" / "sample"
    )

    if sample_dir.exists():
        try:
            # Import sample files directly using the internal method
            imported_files = fm.import_directory(sample_dir)
            print(
                f"📁 Pre-populated {len(imported_files)} sample files: {', '.join(imported_files)}",
            )
        except Exception as e:
            print(f"⚠️  Could not import sample files: {e}")
            return None
    else:
        print(f"⚠️  Sample directory not found: {sample_dir}")
        return None

    description = (
        custom.strip()
        if custom
        else (
            f"Demonstrate FileManager's file analysis and organization capabilities. "
            f"The FileManager currently contains {len(imported_files)} files: {', '.join(imported_files)}. "
            f"Show content analysis, semantic search, filtering, organization, and information extraction."
        )
    )
    description += (
        f"\nAvailable capabilities:\n"
        f"- Use 'ask' to query the filesystem and file contents\n"
        f"- Use 'ask_about_file' for file-specific questions\n"
        f"- Use 'organize' to rename, move, or delete files\n"
        f"- Demonstrate semantic search and filtering"
    )

    builder = ScenarioBuilder(
        description=description,
        tools={
            "ask": fm.ask,
            "organize": fm.organize,
        },
        enable_voice=enable_voice,
        clarifications_enabled=clarifications_enabled,
    )

    try:
        await builder.create()
    except Exception as exc:
        print(f"⚠️  LLM scenario building failed: {exc}")
        # Don't raise - we already have sample files imported

    return None


# ═════════════════════════════ intent dispatcher ════════════════════════════


class _Intent(BaseModel):
    action: str = Field(..., pattern="^(ask|ask_about_file|organize)$")
    filename: Optional[str] = None


_INTENT_SYS_MSG = (
    "You are an intent router for the FileManager.\n"
    "Decide if the user's input is:\n"
    " - a general question about files or filesystem ('ask'),\n"
    " - a question about a specific file ('ask_about_file'), or\n"
    " - a request to organize/rename/move/delete files ('organize').\n"
    "Return ONLY JSON with this shape: {'action':'ask'|'ask_about_file'|'organize', 'filename': str|null}.\n"
    "Rules:\n"
    "- Choose 'ask_about_file' when the user asks about a specific file mentioned by name.\n"
    "- Choose 'organize' for rename, move, delete, or restructuring operations.\n"
    "- Choose 'ask' for general queries about the filesystem or content across files.\n"
    "- If action is 'ask_about_file', extract the filename and include it in the response.\n"
    "Examples:\n"
    " - 'What's in report.pdf?' → {'action':'ask_about_file', 'filename':'report.pdf'}\n"
    " - 'Rename document.docx to final.docx' → {'action':'organize', 'filename':null}\n"
    " - 'What files do I have?' → {'action':'ask', 'filename':null}\n"
    " - 'Search for files about Python' → {'action':'ask', 'filename':null}"
)


async def _dispatch_with_context(
    fm: FileManager,
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
    Decide whether to call `ask`, `ask_about_file`, or `organize`, forwarding
    *parent_chat_context* to the FileManager methods.
    """

    judge = unify.Unify("gpt-5@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )

    if intent.action == "ask_about_file" and intent.filename:
        # Validate file exists
        if not fm.exists(intent.filename):
            # Fall back to general ask if file doesn't exist
            intent.action = "ask"

    if intent.action == "organize":
        handle, clar_up_q, clar_down_q = (
            await call_manager_with_optional_clarifications(
                fm.organize,
                raw,
                parent_chat_context=parent_chat_context,
                return_reasoning_steps=show_steps,
                clarifications_enabled=clarifications_enabled,
            )
        )
    elif intent.action == "ask_about_file" and intent.filename:
        handle, clar_up_q, clar_down_q = (
            await call_manager_with_optional_clarifications(
                lambda q: fm.ask_about_file(intent.filename, q),
                raw,
                parent_chat_context=parent_chat_context,
                return_reasoning_steps=show_steps,
                clarifications_enabled=clarifications_enabled,
            )
        )
    else:  # ask
        handle, clar_up_q, clar_down_q = (
            await call_manager_with_optional_clarifications(
                fm.ask,
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

    return intent.action, handle, clar_up_q, clar_down_q


# ══════════════════════════════════  CLI  ═══════════════════════════════════


async def _main_async() -> None:
    parser = build_cli_parser("FileManager sandbox")
    parser.add_argument(
        "--adapter",
        "-a",
        type=str,
        choices=["local", "codesandbox", "interact", "google_drive"],
        default="local",
        help="Select the filesystem adapter to use (default: local).",
    )
    parser.add_argument(
        "--root",
        "-r",
        type=str,
        default=None,
        help="Root directory for local adapter (default: temp directory).",
    )

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
        log_file=".logs_file_manager.txt",
        tcp_port=args.log_tcp_port,
        http_tcp_port=args.http_log_tcp_port,
        unify_requests_log_file=".logs_unify_requests.txt",
    )
    LG.setLevel(logging.INFO)

    # Create appropriate file manager based on adapter choice
    if args.adapter == "local":
        if args.root:
            root_path = Path(args.root).expanduser().resolve()
            adapter = LocalFileSystemAdapter(root_dir=root_path)
            fm = FileManager(adapter=adapter)
        else:
            fm = LocalFileManager()
    elif args.adapter == "codesandbox":
        from unity.file_manager.managers.codesandbox import CodeSandboxFileManager

        fm = CodeSandboxFileManager()
    elif args.adapter == "interact":
        from unity.file_manager.managers.interact import InteractFileManager

        fm = InteractFileManager()
    elif args.adapter == "google_drive":
        from unity.file_manager.managers.google_drive import GoogleDriveFileManager

        fm = GoogleDriveFileManager()
    else:
        fm = LocalFileManager()

    if args.traced:
        fm = unify.traced(fm)

    print(f"📁 FileManager initialized with {args.adapter} adapter")

    # ─────────────────── command helper output ────────────────────

    _COMMANDS_HELP = (
        "\nFileManager sandbox – type commands below (press ↵ with an empty "
        "line to dictate via voice when --voice mode is active – type 'r' to record).  'quit' to exit.\n\n"
        "┌────────────────── accepted commands ─────────────────────┐\n"
        "│ us  {description}     – update_scenario (text)           │\n"
        "│ usv                   – update_scenario_vocally          │\n"
        "│ r / free text         – freeform ask / organize (auto)   │\n"
        "│ save_project | sp     – save project snapshot            │\n"
        "│ help | h              – show this help                   │\n"
        "└──────────────────────────────────────────────────────────┘\n"
    )

    def _explain_commands() -> None:  # noqa: D401 – helper
        print(_COMMANDS_HELP)

    if args.voice:
        _speak(
            "Sandbox ready. You can type commands, or press enter on an empty line "
            "to record a voice query. Use 'u-s-v' to build a new scenario vocally.",
        )
        _wait_tts_end()

    # running memory of the dialogue
    chat_history: List[Dict[str, str]] = []

    # interaction loop
    while True:
        # Reprint the commands so they remain visible
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
                        "🧮 Describe the file scenario you want to build > ",
                    ).strip()
                    if not description:
                        print("⚠️  No description provided – cancelled.")
                        continue

                if args.voice:
                    task = asyncio.create_task(
                        _build_scenario(
                            fm,
                            description,
                            clarifications_enabled=not args.no_clarifications,
                            enable_voice=bool(args.voice),
                        ),
                    )
                    _speak_wait("Got it, working on your custom scenario now.")
                    print(
                        "[generate] Building synthetic file scenario – this can take a moment…",
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
                        "[generate] Building synthetic file scenario – this can take a moment…",
                    )
                    try:
                        await _build_scenario(
                            fm,
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
                        fm,
                        description,
                        clarifications_enabled=not args.no_clarifications,
                        enable_voice=bool(args.voice),
                    ),
                )
                _speak_wait("Got it, working on your custom scenario now.")
                print(
                    "[generate] Building synthetic file scenario – this can take a moment…",
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
                fm,
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
