"""
===================================================================
Repairs Agent Sandbox - Interactive CodeActActor for repairs/telematics analysis.

This is a specialized sandbox pre-configured with:
- Business context from FilePipelineConfig (column definitions, business rules)
- FunctionManager access for pre-built metric functions
- FileManager primitives via `primitives.files`
- No computer environment (data analysis only)

Usage:
    python -m sandboxes.actor.repairs_agent_sandbox -p RepairsAgent5M

    # Then ask questions like:
    repairs> What is the first time fix rate?
    repairs> Show me jobs completed per day by operative
    repairs> Compare no-access rates between North and South regions
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PATH AND ENVIRONMENT SETUP - MUST HAPPEN BEFORE ANY intranet/unity IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

# Initialize script environment for intranet imports
from intranet.scripts.utils import initialize_script_environment

if not initialize_script_environment():
    print("ERROR: Failed to initialize script environment", file=sys.stderr)
    sys.exit(1)

import unify

from sandboxes.utils import (
    activate_project,
    await_with_interrupt,
    build_cli_parser,
    call_manager_with_optional_clarifications,
    configure_sandbox_logging,
    get_custom_scenario,
    speak,
    steering_controls_hint,
)
from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments import StateManagerEnvironment
from unity.function_manager.primitives import Primitives
from unity.manager_registry import ManagerRegistry
from intranet.repairs_agent.config.prompt_builder import build_repairs_system_prompt

LG = logging.getLogger("repairs_agent_sandbox")


# Help text displayed to the user in the REPL
_COMMANDS_HELP = """
Repairs Agent Sandbox
---------------------
Ask natural language questions about repairs and telematics data.

┌─────────────────────────── Commands ───────────────────────────┐
│ <your question>     - Ask about repairs/telematics data         │
│ custom              - Interactively provide a multi-line query  │
│ save_project | sp   - Save project snapshot with current state  │
│ help | h            - Show this help message                    │
│ quit | exit         - Exit the sandbox                          │
└─────────────────────────────────────────────────────────────────┘

Example queries:
  - What is the first time fix rate?
  - Show me jobs completed per day by operative
  - What is the no-access rate by region?
  - Compare performance between operatives

Steering controls (while a task is running):
  /pause    - Pause execution
  /resume   - Resume execution
  /stop     - Stop execution
  /i <msg>  - Interject with a message
"""


def _create_repairs_actor(config_path: Optional[Path] = None) -> CodeActActor:
    """Create a CodeActActor configured for repairs analysis."""
    LG.info("Creating CodeActActor for repairs analysis...")

    # Initialize primitives for sandbox execution
    primitives = Primitives()

    # Get FunctionManager if available
    function_manager = None
    try:
        function_manager = ManagerRegistry.get_function_manager()
        LG.info("FunctionManager loaded - metric functions available")
    except Exception as e:
        LG.warning(f"FunctionManager not available: {e}")

    # Create actor with only StateManagerEnvironment (no computer)
    # Only expose 'files' primitives - this is a data analysis sandbox,
    # so we don't need contacts, tasks, knowledge, etc.
    actor = CodeActActor(
        function_manager=function_manager,
        environments=[StateManagerEnvironment(primitives, exposed_managers={"files"})],
    )

    return actor


async def _main_async() -> None:
    """Main asynchronous function to run the repairs agent sandbox REPL."""
    # 1. Standard Sandbox Setup
    parser = build_cli_parser(
        "Repairs Agent Sandbox - Interactive repairs/telematics analysis",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to FilePipelineConfig JSON for business context",
    )
    args = parser.parse_args()

    # Default to RepairsAgent5M project
    project_name = args.project_name
    if project_name == "Sandbox":
        project_name = "RepairsAgent5M"

    activate_project(project_name, args.overwrite)

    # ─────────────────── project version handling ────────────────────
    if args.project_version != -1:
        commits = unify.get_project_commits(project_name)
        if commits:
            try:
                target = commits[args.project_version]
                unify.rollback_project(project_name, target["commit_hash"])
                LG.info("[version] Rolled back to commit %s", target["commit_hash"])
            except IndexError:
                LG.warning(
                    "[version] project_version index %s out of range, ignoring",
                    args.project_version,
                )

    configure_sandbox_logging(
        log_in_terminal=args.log_in_terminal,
        log_file=".logs_repairs_agent_sandbox.txt",
    )
    LG.setLevel(logging.DEBUG)

    # 2. Build system prompt extension from config
    config_path = Path(args.config) if args.config else None
    system_prompt_extension = build_repairs_system_prompt(config_path)

    # 3. Initialize the CodeActActor
    actor = _create_repairs_actor(config_path)
    LG.info("Repairs agent initialized successfully")
    print(f"\n🔧 Repairs Agent initialized (project: {project_name})")
    print("   Business context loaded from FilePipelineConfig")

    # 4. Main REPL (Read-Eval-Print Loop)
    print(_COMMANDS_HELP)
    chat_history: List[Dict[str, str]] = []

    try:
        while True:
            try:
                query = input(f"\nrepairs> ").strip()
                if not query:
                    continue

                if query.lower() in {"quit", "exit"}:
                    break
                elif query.lower() in {"help", "h", "?"}:
                    print(_COMMANDS_HELP)
                    continue
                elif query.lower() == "custom":
                    query = get_custom_scenario(args)
                    if not query:
                        print("Custom query cancelled.")
                        continue
                    print(f"Using custom query: {query}")
                elif query.lower() in {"save_project", "sp"}:
                    commit_hash = unify.commit_project(
                        project_name,
                        commit_message=f"Repairs sandbox save {datetime.utcnow().isoformat()}",
                    ).get("commit_hash")
                    print(f"💾 Project saved at commit {commit_hash}")
                    if args.voice:
                        speak("Project saved")
                    continue

                # Prepend business context to the query (like DynamicRepairsAgent does)
                full_description = (
                    f"{system_prompt_extension}\n\n### User Query\n{query}"
                )

                print(
                    f'▶️  Analyzing: "{query[:80]}{"..." if len(query) > 80 else ""}"',
                )
                if args.voice:
                    speak("Analyzing your query.")

                # A. DISPATCH: Call actor.act() to get the steerable handle
                handle, clar_up_q, clar_down_q = (
                    await call_manager_with_optional_clarifications(
                        actor.act,
                        full_description,
                        parent_chat_context=list(chat_history),
                        clarifications_enabled=not args.no_clarifications,
                    )
                )

                chat_history.append({"role": "user", "content": query})

                # B. AWAIT: Pass the handle to the interactive waiter
                print(steering_controls_hint(voice_enabled=args.voice))
                final_result = await await_with_interrupt(
                    handle,
                    enable_voice_steering=args.voice,
                    clarification_up_q=clar_up_q,
                    clarification_down_q=clar_down_q,
                    clarifications_enabled=not args.no_clarifications,
                    chat_context=list(chat_history),
                )

                # C. PROCESS RESULT: Print the final outcome
                print("\n" + "=" * 60)
                print("✅ Analysis Complete:")
                print("=" * 60)
                print(final_result)
                print("=" * 60 + "\n")
                chat_history.append({"role": "assistant", "content": final_result})

                if args.voice:
                    speak("Analysis complete.")

            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break
            except Exception as e:
                LG.error("An error occurred: %s", e, exc_info=True)
                print(f"❌ Error: {e}")

    finally:
        print("Shutting down repairs agent...")
        if hasattr(actor, "close") and asyncio.iscoroutinefunction(actor.close):
            await actor.close()
        print("Shutdown complete.")


def main() -> None:
    """Synchronous entry point for the sandbox."""
    try:
        asyncio.run(_main_async())
    except Exception as e:
        print(f"A critical error forced the sandbox to exit: {e}")
        LG.critical(
            "Sandbox forced to exit due to unhandled exception.",
            exc_info=True,
        )


if __name__ == "__main__":
    main()
