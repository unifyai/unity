"""file_manager_sandbox.py  (optional voice mode, Deepgram SDK v4, sync)
====================================================================
Interactive sandbox for **FileManager**.

It supports:
• File import from local filesystem or URLs.
• Voice or plain‑text input (same helpers as the other sandboxes).
• Automatic dispatch to `ask` depending on intent.
• Mid‑conversation interruption (pause / interject / cancel).
• File parsing and content analysis.
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
from sandboxes.scenario_builder import ScenarioBuilder

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ────────────────────────────────  unity imports  ───────────────────────────
from unity.file_manager.file_manager import FileManager
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
    custom: Optional[str] = None,
    *,
    clarifications_enabled: bool = True,
    enable_voice: bool = False,
) -> Optional[str]:
    """
    Populate the file store with sample files and demonstrate FileManager capabilities.
    FileManager is read-only, so we manually add sample files then use ScenarioBuilder
    to showcase content analysis capabilities.
    """
    fm = FileManager()

    # Manually populate sample files since FileManager is read-only
    from pathlib import Path

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
            f"Demonstrate FileManager's file analysis capabilities using the pre-loaded sample files. "
            f"The FileManager currently contains {len(imported_files)} files: {', '.join(imported_files)}. "
            f"Show content analysis, semantic search, filtering, and information extraction across multiple files. "
            f"Demonstrate both individual file queries and cross-file analysis using the search and filter tools."
        )
    )
    description += (
        f"\nAvailable capabilities to demonstrate:\n"
        f"- Files: {', '.join(imported_files)}\n"
        f"- Use 'ask' to query specific file contents\n"
        f"- Use 'search_files' to find files by content similarity\n"
        f"- Use 'filter_files' to find files by exact criteria\n"
        f"- Demonstrate semantic search across multiple documents\n"
        f"- Show how to extract and compare information across files\n"
        f"- Parse additional files if needed for analysis"
    )

    builder = ScenarioBuilder(
        description=description,
        tools={  # expose the full read-only FileManager interface
            "ask": fm.ask,  # allows the LLM to query file contents
            "search_files": fm._search_files,  # semantic search over files
            "filter_files": fm._filter_files,  # exact filtering of files
            "list_columns": fm._list_columns,  # inspect file table schema
            "list": fm.list,  # list all available files
            "exists": fm.exists,  # check if file exists
            "parse": fm.parse,  # parse additional files if needed
            "import_file": fm.import_file,  # import single file
            "import_directory": fm.import_directory,  # import directory
        },
        enable_voice=enable_voice,
        clarifications_enabled=clarifications_enabled,
    )

    try:
        await builder.create()
    except Exception as exc:
        print(f"⚠️  LLM scenario building failed: {exc}")
        # Don't raise - we already have sample files imported

    # The new flow doesn't produce a structured "theme"; preserve signature.
    return None


# ═════════════════════════════ intent dispatcher ════════════════════════════


# FileManager is read-only - all operations are queries about file contents


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
    Route user input to FileManager ask method with automatic file selection.
    FileManager is read-only - all operations are content queries.
    """
    files = fm.list()

    if not files:
        raise ValueError(
            "No files available. FileManager is read-only - files are automatically added when received as email attachments or browser downloads. For this sandbox, use 'us' to populate sample files.",
        )

    # Extract filename if present in the query
    filename = None

    # Simple heuristic to find mentioned filenames
    for file in files:
        if file.lower() in raw.lower():
            filename = file
            break

    if not filename:
        # If no specific file mentioned, use heuristics to select appropriate file
        if len(files) == 1:
            filename = files[0]
        else:
            # For multi-file questions, let the ask method handle it with tool selection
            # We'll just pick the first file and let the LLM use the list() tool to see others
            filename = files[0]
            raw = f"Available files: {', '.join(files)}. {raw}"

    handle, clar_up_q, clar_down_q = await call_manager_with_optional_clarifications(
        lambda query: fm.ask(filename, query),
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

    return "ask", handle, clar_up_q, clar_down_q


# ══════════════════════════════════  CLI  ═══════════════════════════════════


async def _main_async() -> None:
    parser = build_cli_parser("FileManager sandbox")

    # No automatic seeding – users can invoke 'us' / 'usv' commands to populate files when desired.

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

    # manager
    fm = FileManager()
    if args.traced:
        fm = unify.traced(fm)

    # ─────────────────── optional initial seeding ─────────────────────────
    # No automatic seeding – users can invoke 'us' / 'usv' commands to populate files when desired.

    # ─────────────────── command helper output ────────────────────

    _COMMANDS_HELP = (
        "\nFileManager sandbox – type commands below (press ↵ with an empty "
        "line to dictate via voice when --voice mode is active – type 'r' to record).  'quit' to exit.\n\n"
        "┌────────────────── accepted commands ─────────────────────┐\n"
        "│ us  {description}     – build_scenario (populate files)  │\n"
        "│ usv                   – build_scenario_vocally           │\n"
        "│ if  {file_path}       – import_file (from filesystem)    │\n"
        "│ id  {directory_path}  – import_directory                 │\n"
        "│ lf                    – list_files                       │\n"
        "│ search {query}        – search_files (semantic search)   │\n"
        "│ filter {filter}       – filter_files (exact filtering)   │\n"
        "│ lc                    – list_columns (table schema)      │\n"
        "│ pf  {filename}        – parse_file                       │\n"
        "│ r / free text         – ask questions about files       │\n"
        "│ save_project | sp     – save project snapshot            │\n"
        "│ help | h              – show this help                   │\n"
        "└──────────────────────────────────────────────────────────┘\n"
        "\n"
        "FileManager provides powerful file analysis capabilities:\n"
        "• Import: if/id commands to add files to registry\n"
        "• Search: search command for semantic content search\n"
        "• Filter: filter command for exact criteria filtering\n"
        "• Query: ask questions about specific file contents\n"
        "• Parse: process additional files for analysis\n"
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
        # Reprint the commands so they remain visible, mirroring other sandboxes
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

            # ─────────────── file management commands ────────────────
            parts = raw.split(maxsplit=1)
            cmd_lower = parts[0].lower()

            # Import file command
            if cmd_lower in {"if", "import_file"}:
                if len(parts) < 2:
                    file_path = input("📁 Enter file path to import > ").strip()
                    if not file_path:
                        print("⚠️  No file path provided – cancelled.")
                        continue
                else:
                    file_path = parts[1].strip()

                try:
                    file_path_obj = Path(file_path).expanduser().resolve()
                    if not file_path_obj.exists():
                        print(f"⚠️  File not found: {file_path}")
                        continue
                    if not file_path_obj.is_file():
                        print(f"⚠️  Path is not a file: {file_path}")
                        continue

                    display_name = fm._add_file(file_path_obj)
                    print(f"✅ File imported as: {display_name}")
                    if args.voice:
                        _speak(f"File imported successfully as {display_name}")
                except Exception as exc:
                    print(f"❌ Failed to import file: {exc}")
                    if args.voice:
                        _speak("Failed to import file")
                continue

            # Import directory command
            if cmd_lower in {"id", "import_directory"}:
                if len(parts) < 2:
                    dir_path = input("📁 Enter directory path to import > ").strip()
                    if not dir_path:
                        print("⚠️  No directory path provided – cancelled.")
                        continue
                else:
                    dir_path = parts[1].strip()

                try:
                    imported_files = fm.import_directory(dir_path)
                    if imported_files:
                        print(
                            f"✅ Imported {len(imported_files)} files: {', '.join(imported_files)}",
                        )
                        if args.voice:
                            _speak(f"Imported {len(imported_files)} files successfully")
                    else:
                        print("⚠️  No files found in directory or import failed")
                        if args.voice:
                            _speak("No files were imported")
                except Exception as exc:
                    print(f"❌ Failed to import directory: {exc}")
                    if args.voice:
                        _speak("Failed to import directory")
                continue

            # List files command
            if cmd_lower in {"lf", "list_files"}:
                files = fm.list()
                if files:
                    print(f"📁 Available files ({len(files)}):")
                    for i, file in enumerate(files, 1):
                        print(f"  {i}. {file}")
                    if args.voice:
                        _speak(f"You have {len(files)} files available")
                else:
                    print("📁 No files imported yet")
                    if args.voice:
                        _speak("No files available")
                continue

            # Show supported formats command
            if cmd_lower in {"sf", "show_formats", "supported_formats"}:
                formats = fm.supported_formats
                print(f"📄 Supported formats ({len(formats)}):")
                for fmt in formats:
                    print(f"  • {fmt}")
                if args.voice:
                    _speak(f"Supports {len(formats)} file formats")
                continue

            # Parse file command
            if cmd_lower in {"pf", "parse_file"}:
                if len(parts) < 2:
                    files = fm.list()
                    if not files:
                        print("⚠️  No files available to parse")
                        continue
                    print("Available files:")
                    for i, file in enumerate(files, 1):
                        print(f"  {i}. {file}")
                    filename = input("📄 Enter filename to parse > ").strip()
                    if not filename:
                        print("⚠️  No filename provided – cancelled.")
                        continue
                else:
                    filename = parts[1].strip()

                if not fm.exists(filename):
                    print(f"⚠️  File not found: {filename}")
                    continue

                try:
                    print(f"🔍 Parsing {filename}...")
                    results = fm.parse(filename)
                    result = results.get(filename, {})

                    if result.get("status") == "success":
                        records = result.get("records", [])
                        metadata = result.get("metadata", {})

                        print(f"✅ Parsing successful!")
                        print(f"   • Records: {len(records)}")
                        print(f"   • File type: {metadata.get('file_type', 'unknown')}")
                        print(
                            f"   • File size: {metadata.get('file_size', 'unknown')} bytes",
                        )
                        print(
                            f"   • Processing time: {metadata.get('processing_time', 'unknown')}s",
                        )

                        if args.voice:
                            _speak(
                                f"Parsed {filename} successfully. Found {len(records)} records.",
                            )
                    else:
                        error = result.get("error", "Unknown error")
                        print(f"❌ Parsing failed: {error}")
                        if args.voice:
                            _speak("Parsing failed")

                except Exception as exc:
                    print(f"❌ Failed to parse file: {exc}")
                    if args.voice:
                        _speak("Failed to parse file")
                continue

            # Search files command (semantic search)
            if cmd_lower in {"search", "search_files"}:
                if len(parts) < 2:
                    query = input("🔍 Enter search query > ").strip()
                    if not query:
                        print("⚠️  No query provided – cancelled.")
                        continue
                else:
                    query = " ".join(parts[1:]).strip()

                try:
                    print(f"🔍 Searching files for: '{query}'...")
                    results = fm._search_files(references={"full_text": query}, k=5)

                    if results:
                        print(f"📄 Found {len(results)} matching files:")
                        for i, file_obj in enumerate(results, 1):
                            print(
                                f"  {i}. {file_obj.filename} (status: {file_obj.status})",
                            )
                        if args.voice:
                            _speak(f"Found {len(results)} matching files")
                    else:
                        print("📄 No files found matching your query")
                        if args.voice:
                            _speak("No matching files found")

                except Exception as exc:
                    print(f"❌ Search failed: {exc}")
                    if args.voice:
                        _speak("Search failed")
                continue

            # Filter files command (exact filtering)
            if cmd_lower in {"filter", "filter_files"}:
                if len(parts) < 2:
                    filter_expr = input("🔍 Enter filter expression > ").strip()
                    if not filter_expr:
                        print("⚠️  No filter provided – cancelled.")
                        continue
                else:
                    filter_expr = " ".join(parts[1:]).strip()

                try:
                    print(f"🔍 Filtering files with: '{filter_expr}'...")
                    results = fm._filter_files(filter=filter_expr, limit=20)

                    if results:
                        print(f"📄 Found {len(results)} matching files:")
                        for i, file_obj in enumerate(results, 1):
                            print(
                                f"  {i}. {file_obj.filename} (status: {file_obj.status})",
                            )
                        if args.voice:
                            _speak(f"Found {len(results)} matching files")
                    else:
                        print("📄 No files found matching your filter")
                        if args.voice:
                            _speak("No matching files found")

                except Exception as exc:
                    print(f"❌ Filter failed: {exc}")
                    if args.voice:
                        _speak("Filter failed")
                continue

            # List columns command (show table schema)
            if cmd_lower in {"lc", "list_columns"}:
                try:
                    print("📊 File table schema:")
                    columns = fm._list_columns(include_types=True)
                    for col_name, col_type in columns.items():
                        print(f"  • {col_name}: {col_type}")
                    if args.voice:
                        _speak(f"File table has {len(columns)} columns")
                except Exception as exc:
                    print(f"❌ Failed to list columns: {exc}")
                    if args.voice:
                        _speak("Failed to list columns")
                continue

            # ─────────────── scenario (re)seeding commands ────────────────
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
                            description,
                            clarifications_enabled=not args.no_clarifications,
                            enable_voice=bool(args.voice),
                        ),
                    )
                    _speak_wait("Got it, working on your custom scenario now.")
                    print(
                        "[generate] Building synthetic files – this can take a moment…",
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
                        "[generate] Building synthetic files – this can take a moment…",
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
                    "[generate] Building synthetic files – this can take a moment…",
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

            # ──────────────── handle file-related questions ────────────────
            files = fm.list()
            if not files:
                print(
                    "⚠️  No files available. Please import some files first using 'if' or 'id' commands.",
                )
                if args.voice:
                    _speak("No files available. Please import some files first.")
                continue

            # ──────────────── remember the user's utterance ────────────────
            try:
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
            except ValueError as ve:
                print(f"⚠️  {ve}")
                if args.voice:
                    _speak(str(ve))
            except Exception as exc:
                LG.error("Error during FileManager operation: %s", exc, exc_info=True)
                print(f"❌  {exc}")
                if args.voice:
                    _speak("An error occurred")

        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
