"""file_manager_sandbox.py  (optional voice mode, Deepgram SDK v4, sync)
============================================================================
Interactive sandbox for FileManager exposing only its public entry points.

Public surface exercised via natural language:
- ask(text)
- ask_about_file(filename, question, response_format=Optional[Pydantic|JSON Schema])
- organize(text)

Additionally provides:
- seed-sample: import and parse sample files from tests/file_manager/sample
- list, stat: light helpers for inspection

Construction controls via CLI:
- adapter selection (local)
- local root vs rootless
- default parse return mode (compact|full|none)
- optional default response_format (schema/model) for ask_about_file
"""

from __future__ import annotations

# ─────────────────────────────── stdlib / vendored ──────────────────────────
import os
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Always enable detailed request logging for sandbox runs BEFORE importing unify
os.environ["UNIFY_REQUESTS_DEBUG"] = "true"

from dotenv import load_dotenv

load_dotenv()

import json
import unify
from pydantic import BaseModel, Field
from sandboxes.scenario_builder import ScenarioBuilder
from unity.common.llm_client import new_llm_client

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ────────────────────────────────  unity imports  ───────────────────────────
from unity.file_manager.managers.file_manager import FileManager
from unity.file_manager.managers.local import LocalFileManager
from unity.file_manager.filesystem_adapters.local_adapter import LocalFileSystemAdapter
from unity.file_manager.types.config import FilePipelineConfig
from unity.common.async_tool_loop import SteerableToolHandle
from sandboxes.utils import (
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


# ───────────────────────────── Built-in response models ─────────────────────
class ReportFacts(BaseModel):
    period: Optional[str] = Field(None, description="e.g., 'Q1 2025'")
    mentions_revenue: bool = Field(..., description="Whether revenue is mentioned")


_BUILTIN_MODELS: Dict[str, type[BaseModel]] = {
    "report_facts": ReportFacts,
}


def _load_response_format(schema_path: Optional[str], model_name: Optional[str]) -> Any:
    """Return a response_format object for ask_about_file.

    Priority: explicit schema path → built-in model name → None.
    """
    if schema_path:
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"⚠️  Failed to load schema from {schema_path}: {exc}")
    if model_name:
        m = _BUILTIN_MODELS.get(model_name.strip().lower())
        if m is not None:
            return m
        print(f"⚠️  Unknown model '{model_name}'. Known: {', '.join(_BUILTIN_MODELS)}")
    return None


# ═════════════════════════════════ seed helpers ═════════════════════════════
async def _seed_sample(
    fm: FileManager,
    *,
    parse_return_mode: str = "compact",
) -> None:
    sample_dir = ROOT / "tests" / "file_manager" / "sample"
    if not sample_dir.exists():
        print(f"⚠️  Sample directory not found: {sample_dir}")
        return
    try:
        added = []
        try:
            added = fm.import_directory(str(sample_dir))
        except Exception:
            # import_directory is adapter-specific; fall back to manual loop
            added = []
            for p in sample_dir.iterdir():
                if p.is_file():
                    try:
                        added.append(fm.import_file(str(p)))
                    except Exception:
                        continue
        print(f"📁 Imported {len(added)} sample files.")
    except Exception as exc:
        print(f"⚠️  Could not import sample files: {exc}")
        return
    # Optionally parse to ingest
    try:
        cfg = FilePipelineConfig(output={"return_mode": parse_return_mode})
        # ingest in batches to ensure per-file contexts exist
        fm.ingest_files(
            [str(p) for p in sample_dir.iterdir() if p.is_file()],
            config=cfg,
        )
        print(f"🧩 Ingested sample files with return_mode={parse_return_mode}.")
    except Exception as exc:
        print(f"⚠️  Parsing failed (continuing): {exc}")


# ═════════════════════════════════ scenario builder ═════════════════════════
async def _build_scenario(
    fm: FileManager,
    custom: Optional[str] = None,
    *,
    clarifications_enabled: bool = True,
    enable_voice: bool = False,
) -> Optional[str]:
    """Populate a realistic file environment via public tools using ScenarioBuilder.

    If there are no files, attempts to seed from sample fixtures first.
    """
    try:
        if not fm.list():
            await _seed_sample(fm, parse_return_mode="compact")
    except Exception:
        # Continue even if list/import fails
        pass

    description = (
        custom.strip()
        if custom
        else (
            "Organize files by sensible categories (type/date), then answer a few "
            "natural-language questions about their contents. Prefer compact parse "
            "artifacts and reference-heavy reasoning. Demonstrate structured extraction "
            "via ask→ask_about_file when appropriate. Do not attempt to write new files."
        )
    )

    builder = ScenarioBuilder(
        description=description,
        tools={
            # expose only the public surface
            "organize": fm.organize,
            "ask": fm.ask,
            "ask_about_file": fm.ask_about_file,
        },
        enable_voice=enable_voice,
        clarifications_enabled=clarifications_enabled,
    )

    try:
        await builder.create()
    except Exception as exc:
        raise (f"LLM seeding via ScenarioBuilder failed. {exc}")

    return None


# ═════════════════════════════ intent dispatcher ════════════════════════════
async def _dispatch_with_context(
    fm: FileManager,
    raw: str,
    *,
    show_steps: bool,
    parent_chat_context: List[Dict[str, str]],
    clarifications_enabled: bool,
    enable_voice: bool,
    default_response_format: Any,
) -> Tuple[
    str,
    SteerableToolHandle,
    Optional[asyncio.Queue[str]],
    Optional[asyncio.Queue[str]],
]:
    """Route raw to ask, ask_about_file, or organize based on simple prefixes.

    Natural-language first: if no explicit prefix, use intent router for ask/organize.
    Convenience explicit commands supported:
      - askf <filename> <question>  (uses default_response_format if set)
      - list, stat <path>, seed-sample
    """

    # Explicit light helpers
    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "list":
        try:
            names = fm.list()
            print("\n".join(names) if names else "(empty)")
        except Exception as exc:
            print(f"⚠️  list failed: {exc}")

        # Return a no-op handle
        class _Noop(SteerableToolHandle):
            async def result(self):
                return "ok"

        return "noop", _Noop(), None, None

    if cmd in ("stat", "info", "file_info"):
        try:
            info = fm.file_info(identifier=rest)
            # FileInfo is a Pydantic model; convert to dict for display
            print(json.dumps(info.model_dump(), indent=2))
        except Exception as exc:
            print(f"⚠️  file_info failed: {exc}")

        class _Noop(SteerableToolHandle):
            async def result(self):
                return "ok"

        return "noop", _Noop(), None, None

    if cmd == "seed-sample":
        try:
            await _seed_sample(fm)
        except Exception as exc:
            print(f"⚠️  seed-sample failed: {exc}")

        class _Noop(SteerableToolHandle):
            async def result(self):
                return "ok"

        return "noop", _Noop(), None, None

    if cmd == "parse":
        try:
            targets = rest.split()
            if not targets:
                print("Usage: parse <path1> [path2 ...]")

                class _Noop(SteerableToolHandle):
                    async def result(self):
                        return "invalid"

                return "noop", _Noop(), None, None
            cfg = FilePipelineConfig()
            res = fm.ingest_files(targets, config=cfg)
            # Print a compact summary
            try:
                print(
                    json.dumps(
                        {
                            k: (
                                v if isinstance(v, dict) else getattr(v, "status", "ok")
                            )
                            for k, v in res.items()
                        },
                        indent=2,
                        default=str,
                    ),
                )
            except Exception:
                print("parsed")
        except Exception as exc:
            print(f"⚠️  parse failed: {exc}")

        class _Noop(SteerableToolHandle):
            async def result(self):
                return "ok"

        return "noop", _Noop(), None, None

    if cmd == "askf":
        try:
            # Format: askf <filename> <question>
            toks = rest.split(maxsplit=1)
            if len(toks) < 2:
                print("Usage: askf <filename> <question>")

                class _Noop(SteerableToolHandle):
                    async def result(self):
                        return "invalid"

                return "noop", _Noop(), None, None
            filename, question = toks[0], toks[1]
            handle, up, down = await call_manager_with_optional_clarifications(
                lambda q: fm.ask_about_file(
                    filename,
                    q,
                    response_format=default_response_format,
                ),
                question,
                parent_chat_context=parent_chat_context,
                return_reasoning_steps=show_steps,
                clarifications_enabled=clarifications_enabled,
            )
            if enable_voice:
                try:
                    _speak("Working on it.")
                except Exception:
                    pass
            return "ask_about_file", handle, up, down
        except Exception as exc:
            print(f"⚠️  askf failed: {exc}")

            class _Noop(SteerableToolHandle):
                async def result(self):
                    return "error"

            return "noop", _Noop(), None, None

    # Natural-language path using a tiny intent router (JSON mode)
    class _Intent(BaseModel):
        action: str = Field(..., pattern="^(ask|ask_about_file|organize)$")
        filename: Optional[str] = None

    _INTENT_SYS_MSG = (
        "You are an intent router for the FileManager.\n"
        "Decide if the user's input is:\n"
        " - a general question about files or filesystem ('ask'),\n"
        " - a question about a specific file ('ask_about_file'), or\n"
        " - a request to organize/rename/move files ('organize').\n"
        "Return ONLY JSON with this shape: {'action':'ask'|'ask_about_file'|'organize', 'filename': str|null}.\n"
        "Choose 'ask_about_file' when an explicit filename is referenced."
    )

    judge = new_llm_client(response_format=_Intent)
    intent = _Intent.model_validate_json(
        await judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )

    if (
        intent.action == "ask_about_file"
        and intent.filename
        and fm.exists(intent.filename)
    ):
        fn = lambda q: fm.ask_about_file(
            intent.filename or "",
            q,
            response_format=default_response_format,
        )
    elif intent.action == "organize":
        fn = fm.organize
    else:
        fn = fm.ask

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
    parser = build_cli_parser("FileManager sandbox")
    parser.add_argument(
        "--adapter",
        "-a",
        type=str,
        choices=["local"],
        default="local",
    )
    parser.add_argument(
        "--root",
        "-r",
        type=str,
        default=None,
        help="Root directory for local adapter",
    )
    parser.add_argument(
        "--rootless",
        action="store_true",
        help="Use Local adapter without a root (absolute paths)",
    )
    parser.add_argument(
        "--return-mode",
        type=str,
        choices=["compact", "full", "none"],
        default="compact",
        help="Default parse return mode for seeding",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="JSON schema file for ask_about_file structured extraction",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Built-in model for ask_about_file ({', '.join(_BUILTIN_MODELS)})",
    )

    args = parser.parse_args()

    # Unify context
    activate_project(args.project_name, args.overwrite)

    # Optional rollback to a specific project commit (align with other sandboxes)
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

    # Construct FileManager per CLI
    if args.adapter == "local":
        if args.rootless:
            fm = FileManager(adapter=LocalFileSystemAdapter(root_dir=None))
        elif args.root:
            root_path = Path(args.root).expanduser().resolve()
            fm = FileManager(adapter=LocalFileSystemAdapter(root_dir=root_path))
        else:
            fm = LocalFileManager()
    else:
        fm = LocalFileManager()

    default_response_format = _load_response_format(args.schema, args.model)

    print(
        f"📁 FileManager initialized (adapter={args.adapter}, rootless={bool(args.rootless)}, return_mode={args.return_mode})",
    )

    # REPL commands
    _COMMANDS_HELP = (
        "\nFileManager sandbox – type commands below (press ↵ with an empty line to record voice when --voice is enabled). 'quit' to exit.\n\n"
        "┌────────────────────────── accepted commands ─────────────────────────┐\n"
        "│ us  {description}          – update_scenario (text)                  │\n"
        "│ usv                         – update_scenario_vocally                 │\n"
        "│ seed-sample                 – import and parse sample files           │\n"
        "│ list                        – list files (adapter scope)              │\n"
        "│ info <path|id>              – show file info (status, ingest layout)  │\n"
        "│ askf <filename> <question>  – file-scoped Q&A (uses --schema/--model) │\n"
        "│ r / free text               – freeform ask/ask_about_file/organize    │\n"
        "│ save_project | sp           – save project snapshot                    │\n"
        "│ help | h                    – show this help                           │\n"
        "└──────────────────────────────────────────────────────────────────────┘\n"
    )

    def _explain_commands() -> None:  # noqa: D401 – helper
        print(_COMMANDS_HELP)

    if args.voice:
        _speak("Sandbox ready. Press enter on an empty line to record a voice query.")
        _wait_tts_end()

    chat_history: List[Dict[str, str]] = []

    while True:
        print()
        _explain_commands()
        print()

        try:
            if args.voice:
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
                _explain_commands()
                continue
            if raw.lower() in {"quit", "exit"}:
                break
            if not raw:
                continue

            if raw.lower() in {"save_project", "sp"}:
                commit_hash = unify.commit_project(
                    args.project_name,
                    commit_message="Sandbox save",
                ).get("commit_hash")
                print(f"💾 Project saved at commit {commit_hash}")
                if args.voice:
                    _speak("Project saved")
                continue

            # Scenario (re)seeding commands
            parts = raw.split(maxsplit=1)
            cmd_lower = parts[0].lower()

            if cmd_lower in {"us", "update_scenario"}:
                description = parts[1].strip() if len(parts) > 1 else ""
                if not description:
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
                continue

            # Ignore steering commands when no request is running
            if raw.startswith("/"):
                print(
                    "(no active request) Steering commands are only available while a call is running.",
                )
                continue

            # Dispatch
            kind, handle, clar_up, clar_down = await _dispatch_with_context(
                fm,
                raw,
                show_steps=args.debug,
                parent_chat_context=list(chat_history),
                clarifications_enabled=not args.no_clarifications,
                enable_voice=bool(args.voice),
                default_response_format=default_response_format,
            )
            chat_history.append({"role": "user", "content": raw})
            if args.voice:
                _speak("Working on it.")
                _wait_tts_end()

            print(_steer_hint(voice_enabled=bool(args.voice)))
            answer = await _await_with_interrupt(
                handle,
                enable_voice_steering=bool(args.voice),
                clarification_up_q=clar_up,
                clarification_down_q=clar_down,
                clarifications_enabled=not args.no_clarifications,
                chat_context=list(chat_history),
            )
            if isinstance(answer, tuple):
                answer, _ = answer
            print(f"[{kind}] → {answer}\n")

            chat_history.append({"role": "assistant", "content": str(answer)})
            if args.voice:
                _speak(f"{answer} Anything else I can help with?")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break
        except Exception as exc:
            LG.error("[error] %s", exc, exc_info=True)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
