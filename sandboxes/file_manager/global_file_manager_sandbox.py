"""global_file_manager_sandbox.py  (optional voice mode, Deepgram SDK v4, sync)
=================================================================================
Interactive sandbox for **GlobalFileManager**.

It supports:
• Multiple filesystem adapters simultaneously (Local, CodeSandbox, Interact, GoogleDrive).
• Voice or plain‑text input (same helpers as the other sandboxes).
• Automatic dispatch to `ask` or `organize` depending on intent.
• Mid‑conversation interruption (pause / interject / cancel).
• Cross-filesystem file operations and queries.
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
from unity.common.llm_client import new_llm_client

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ────────────────────────────────  unity imports  ───────────────────────────
from unity.file_manager.global_file_manager import GlobalFileManager
from unity.file_manager.managers.file_manager import FileManager
from unity.file_manager.managers.local import LocalFileManager
from unity.file_manager.filesystem_adapters.local_adapter import LocalFileSystemAdapter
from unity.file_manager.types.config import FilePipelineConfig
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

LG = logging.getLogger("global_file_manager_sandbox")

# ═════════════════════════════════ seed helpers ═════════════════════════════


async def _build_scenario(
    gfm: GlobalFileManager,
    custom: Optional[str] = None,
    *,
    clarifications_enabled: bool = True,
    enable_voice: bool = False,
) -> Optional[str]:
    """
    Populate file stores across filesystems and demonstrate GlobalFileManager capabilities.
    """
    # Get list of available filesystems
    filesystems = gfm.list_filesystems()

    # Manually populate sample files for each filesystem
    sample_dir = (
        Path(__file__).parent.parent.parent / "tests" / "file_manager" / "sample"
    )

    total_imported = 0
    if sample_dir.exists():
        for fs_alias in filesystems:
            try:
                mgr = gfm._managers.get(fs_alias)
                if mgr and hasattr(mgr, "import_directory"):
                    imported_files = mgr.import_directory(sample_dir)
                    total_imported += len(imported_files)
                    print(
                        f"📁 Populated {len(imported_files)} files in '{fs_alias}': {', '.join(imported_files)}",
                    )
            except Exception as e:
                print(f"⚠️  Could not import sample files to '{fs_alias}': {e}")
                continue
    else:
        print(f"⚠️  Sample directory not found: {sample_dir}")

    if total_imported == 0:
        print("⚠️  No files were imported to any filesystem")
        return None

    description = (
        custom.strip()
        if custom
        else (
            f"Demonstrate GlobalFileManager's cross-filesystem capabilities. "
            f"The GlobalFileManager manages {len(filesystems)} filesystem(s): {', '.join(filesystems)}. "
            f"Total files imported: {total_imported}. "
            f"Show content analysis, semantic search across filesystems, filtering, and organization."
        )
    )
    description += (
        f"\nAvailable capabilities:\n"
        f"- Use 'ask' to query across all filesystems\n"
        f"- Use 'organize' to manage files across filesystems\n"
        f"- Demonstrate cross-filesystem search and filtering\n"
        f"- Show filesystem-specific operations"
    )

    builder = ScenarioBuilder(
        description=description,
        tools={
            "ask": gfm.ask,
            "organize": gfm.organize,
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


# ═════════════════════════════ helpers & dispatcher ═════════════════════════


async def _seed_sample_all(
    gfm: GlobalFileManager,
    *,
    return_mode: str = "compact",
) -> None:
    sample_dir = (
        Path(__file__).resolve().parents[2] / "tests" / "file_manager" / "sample"
    )
    if not sample_dir.exists():
        print(f"⚠️  Sample directory not found: {sample_dir}")
        return
    total = 0
    for alias, mgr in getattr(gfm, "_managers", {}).items():
        try:
            try:
                added = mgr.import_directory(str(sample_dir))
            except Exception:
                added = []
                for p in sample_dir.iterdir():
                    if p.is_file():
                        try:
                            added.append(mgr.import_file(str(p)))
                        except Exception:
                            continue
            print(f"📁 Imported {len(added)} sample files into '{alias}'.")
            cfg = FilePipelineConfig(output={"return_mode": return_mode})
            mgr.ingest_files(
                [str(p) for p in sample_dir.iterdir() if p.is_file()],
                config=cfg,
            )
            total += len(added)
        except Exception as exc:
            print(f"⚠️  Seeding failed for '{alias}': {exc}")
    if total == 0:
        print("⚠️  No files imported across managers")


class _Intent(BaseModel):
    action: str = Field(..., pattern="^(ask|organize)$")


_INTENT_SYS_MSG = (
    "You are an intent router for the GlobalFileManager.\n"
    "Decide if the user's input is:\n"
    " - a read-only question about files across filesystems ('ask'), or\n"
    " - a request to organize/rename/move/delete files ('organize').\n"
    "Return ONLY JSON with this shape: {'action':'ask'|'organize'}.\n"
    "Rules:\n"
    "- Choose 'ask' for questions, searches, and information retrieval.\n"
    "- Choose 'organize' for rename, move, delete, or restructuring operations.\n"
    "Examples:\n"
    " - 'What files do I have across all filesystems?' → {'action':'ask'}\n"
    " - 'Search for Python files' → {'action':'ask'}\n"
    " - 'Rename local/document.docx to final.docx' → {'action':'organize'}\n"
    " - 'Move files from local to remote' → {'action':'organize'}"
)


async def _dispatch_with_context(
    gfm: GlobalFileManager,
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
    Decide whether to call `ask` or `organize`, forwarding
    *parent_chat_context* to the GlobalFileManager methods.
    """

    # REPL explicit commands
    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "list_fms":
        try:
            print(
                ", ".join(getattr(gfm, "_managers", {}).keys()) or "(no managers)",
            )
        except Exception as exc:
            print(f"⚠️  list_fms failed: {exc}")

        class _Noop(SteerableToolHandle):
            async def result(self):
                return "ok"

        return "noop", _Noop(), None, None

    if cmd == "add_local":
        try:
            args = rest.split()
            root = None
            rootless = False
            for a in args:
                if a.startswith("--root="):
                    root = a[len("--root=") :]
                if a == "--rootless":
                    rootless = True
            if rootless:
                mgr = FileManager(adapter=LocalFileSystemAdapter(root_dir=None))
            elif root:
                mgr = FileManager(
                    adapter=LocalFileSystemAdapter(
                        root_dir=Path(root).expanduser().resolve(),
                    ),
                )
            else:
                mgr = LocalFileManager()
            alias = f"local_{len(getattr(gfm, '_managers', {})) + 1}"
            getattr(gfm, "_managers", {})[alias] = mgr
            print(f"➕ Added manager '{alias}'")
        except Exception as exc:
            print(f"⚠️  add_local failed: {exc}")

        class _Noop(SteerableToolHandle):
            async def result(self):
                return "ok"

        return "noop", _Noop(), None, None

    if cmd == "use_fm":
        # Stash selection in a side-channel attribute for this process
        try:
            sel = rest.strip()
            if not sel:
                print("Usage: use_fm <alias>")
            elif sel in getattr(gfm, "_managers", {}):
                setattr(gfm, "_current_alias", sel)
                print(f"✅ Current FM set to '{sel}'")
            else:
                print(f"⚠️  Unknown alias '{sel}'")
        except Exception as exc:
            print(f"⚠️  use_fm failed: {exc}")

        class _Noop(SteerableToolHandle):
            async def result(self):
                return "ok"

        return "noop", _Noop(), None, None

    if cmd == "seed-sample":
        try:
            await _seed_sample_all(gfm)
        except Exception as exc:
            print(f"⚠️  seed-sample failed: {exc}")

        class _Noop(SteerableToolHandle):
            async def result(self):
                return "ok"

        return "noop", _Noop(), None, None

    if cmd in {"gask", "gorganize"}:
        fn = gfm.organize if cmd == "gorganize" else gfm.ask
        handle, cu, cd = await call_manager_with_optional_clarifications(
            fn,
            rest,
            parent_chat_context=parent_chat_context,
            return_reasoning_steps=show_steps,
            clarifications_enabled=clarifications_enabled,
        )
        if enable_voice:
            try:
                _speak("Working on it.")
            except Exception:
                pass
        return ("organize" if cmd == "gorganize" else "ask"), handle, cu, cd

    # Fallback to natural-language intent for global ask/organize
    judge = new_llm_client(response_format=_Intent)
    intent = _Intent.model_validate_json(
        await judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )
    fn = gfm.organize if intent.action == "organize" else gfm.ask
    handle, clar_up_q, clar_down_q = await call_manager_with_optional_clarifications(
        fn,
        raw,
        parent_chat_context=parent_chat_context,
        return_reasoning_steps=show_steps,
        clarifications_enabled=clarifications_enabled,
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
    parser = build_cli_parser("GlobalFileManager sandbox")
    parser.add_argument(
        "--filesystems",
        "-f",
        type=str,
        default="local",
        help="Comma-separated list of filesystem adapters to use (e.g., 'local,interact'). Options: local, codesandbox, interact, google_drive (default: local).",
    )
    parser.add_argument(
        "--local-root",
        type=str,
        default=None,
        help="Root directory for local adapter (default: temp directory).",
    )

    args = parser.parse_args()

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
        log_file=".logs_global_file_manager.txt",
        tcp_port=args.log_tcp_port,
        http_tcp_port=args.http_log_tcp_port,
        unify_requests_log_file=".logs_unify_requests.txt",
    )
    LG.setLevel(logging.INFO)

    # Parse filesystem list and create managers
    fs_list = [fs.strip() for fs in args.filesystems.split(",")]
    managers_by_alias: Dict[str, FileManager] = {}

    for fs_type in fs_list:
        try:
            if fs_type == "local":
                if args.local_root:
                    root_path = Path(args.local_root).expanduser().resolve()
                    adapter = LocalFileSystemAdapter(root_dir=root_path)
                    mgr = FileManager(adapter=adapter)
                else:
                    mgr = LocalFileManager()
                managers_by_alias["local"] = mgr
            elif fs_type == "codesandbox":
                from unity.file_manager.managers.codesandbox import (
                    CodeSandboxFileManager,
                )

                managers_by_alias["codesandbox"] = CodeSandboxFileManager()
            elif fs_type == "interact":
                from unity.file_manager.managers.interact import InteractFileManager

                managers_by_alias["interact"] = InteractFileManager()
            elif fs_type == "google_drive":
                from unity.file_manager.managers.google_drive import (
                    GoogleDriveFileManager,
                )

                managers_by_alias["google_drive"] = GoogleDriveFileManager()
            else:
                print(f"⚠️  Unknown filesystem type: {fs_type}, skipping")
                continue
        except Exception as e:
            print(f"⚠️  Failed to initialize {fs_type} manager: {e}")
            continue

    if not managers_by_alias:
        print("❌  No valid filesystem managers could be initialized. Exiting.")
        return

    gfm = GlobalFileManager(managers_by_alias)

    print(
        f"📁 GlobalFileManager initialized with {len(managers_by_alias)} filesystem(s): {', '.join(managers_by_alias.keys())}",
    )

    # ─────────────────── command helper output ────────────────────

    _COMMANDS_HELP = (
        "\nGlobalFileManager sandbox – type commands below (press ↵ with an empty "
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
                            gfm,
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
                            gfm,
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
                        gfm,
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
                gfm,
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
