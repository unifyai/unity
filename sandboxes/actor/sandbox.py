"""
===================================================================
An interactive, steerable sandbox for running and testing Actor implementations.

This sandbox serves as a sophisticated command-line environment to launch,
monitor, and interact with any of the core actor classes (Hierarchical and
CodeAct). It fully supports advanced interactive features like in-flight
interjection, clarification, and steering commands.

Usage examples:
    # CodeAct without computer tools (data analysis, state managers only)
    python -m sandboxes.actor.sandbox --actor code_act --no-computer -p MyProject

    # CodeAct with web automation
    python -m sandboxes.actor.sandbox --actor code_act -p MyProject

    # Hierarchical actor with web automation
    python -m sandboxes.actor.sandbox --actor hierarchical -p MyProject
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()
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
from unity.actor.base import BaseActor
from unity.actor.code_act_actor import CodeActActor
from unity.actor.hierarchical_actor import HierarchicalActor

LG = logging.getLogger("actor_sandbox")


# Help text displayed to the user in the REPL
_COMMANDS_HELP = """
Actor Sandbox
-------------
Enter a high-level goal for the selected actor to execute.

┌─────────────────────────── Commands ───────────────────────────┐
│ <your goal>         - A high-level task for the actor           │
│ custom              - Interactively provide a multi-line goal   │
│ save_project | sp   - Save project snapshot with current state  │
│ help | h            - Show this help message                    │
│ quit | exit         - Exit the sandbox                          │
└─────────────────────────────────────────────────────────────────┘

Steering controls (while a task is running):
  /pause    - Pause execution
  /resume   - Resume execution
  /stop     - Stop execution
  /i <msg>  - Interject with a message
"""


def _create_actor(args) -> BaseActor:
    """Factory function to instantiate the selected actor based on CLI args."""
    actor_choice = args.actor.lower()
    LG.info(f"Instantiating actor: {actor_choice}")

    if actor_choice == "hierarchical":
        if args.no_computer:
            LG.warning(
                "HierarchicalActor requires computer tools - ignoring --no-computer flag",
            )
        return HierarchicalActor(
            headless=args.headless,
            agent_server_url=args.agent_url,
            computer_mode="magnitude",
        )
    elif actor_choice == "code_act":
        if args.no_computer:
            # No computer tools - just state managers via Primitives
            from unity.actor.environments import StateManagerEnvironment
            from unity.function_manager.primitives import Primitives
            from unity.manager_registry import ManagerRegistry

            primitives = Primitives()
            environments = [StateManagerEnvironment(primitives)]

            # Optionally inject FunctionManager if available
            function_manager = None
            try:
                function_manager = ManagerRegistry.get_function_manager()
            except Exception:
                LG.debug("FunctionManager not available, continuing without it")

            return CodeActActor(
                environments=environments,
                function_manager=function_manager,
            )
        else:
            # Full computer mode (web/desktop)
            return CodeActActor(
                headless=args.headless,
                agent_server_url=args.agent_url,
                computer_mode="magnitude",
            )
    else:
        raise ValueError(f"Unknown actor type: {actor_choice}")


async def _main_async() -> None:
    """Main asynchronous function to run the sandbox REPL."""
    # 1. Standard Sandbox Setup
    parser = build_cli_parser("Interactive Actor Sandbox")
    parser.add_argument(
        "--actor",
        "-a",
        type=str,
        choices=["hierarchical", "code_act"],
        default="code_act",
        help="Select the actor implementation to run.",
    )
    parser.add_argument(
        "--no-computer",
        action="store_true",
        help="Disable computer environment (CodeAct only - uses state managers only).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the web view in headless mode (no visible UI).",
    )
    parser.add_argument(
        "--agent-url",
        type=str,
        default="http://localhost:3000",
        help="URL of the agent service (default: http://localhost:3000).",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Enable persistent, long-running sessions that wait for interjections.",
    )
    args = parser.parse_args()

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

    configure_sandbox_logging(
        log_in_terminal=args.log_in_terminal,
        log_file=".logs_actor_sandbox.txt",
    )
    LG.setLevel(logging.DEBUG)

    # 2. Initialize the selected Actor
    actor = _create_actor(args)
    mode_desc = "no-computer" if args.no_computer else "computer-enabled"
    LG.info(f"Actor '{args.actor}' initialized successfully ({mode_desc}).")
    print(f"\n🎭 Actor '{args.actor}' initialized ({mode_desc})")

    # 3. Main REPL (Read-Eval-Print Loop)
    print(_COMMANDS_HELP)
    chat_history: List[Dict[str, str]] = []

    try:
        while True:
            try:
                goal = input(f"\n{args.actor}> ").strip()
                if not goal:
                    continue

                if goal.lower() in {"quit", "exit"}:
                    break
                elif goal.lower() in {"help", "h", "?"}:
                    print(_COMMANDS_HELP)
                    continue
                elif goal.lower() == "custom":
                    goal = get_custom_scenario(args)
                    if not goal:
                        print("Custom scenario cancelled.")
                        continue
                    print(f"Using custom goal: {goal}")
                elif goal.lower() in {"save_project", "sp"}:
                    commit_hash = unify.commit_project(
                        args.project_name,
                        commit_message=f"Actor sandbox save {datetime.utcnow().isoformat()}",
                    ).get("commit_hash")
                    print(f"💾 Project saved at commit {commit_hash}")
                    if args.voice:
                        speak("Project saved")
                    continue

                # This is the core "dispatch and await" pattern for actors
                print(
                    f'▶️  Starting task: "{goal[:80]}{"..." if len(goal) > 80 else ""}"',
                )
                if args.voice:
                    speak("On it.")

                # A. DISPATCH: Call actor.act() to get the steerable handle
                handle, clar_up_q, clar_down_q = (
                    await call_manager_with_optional_clarifications(
                        actor.act,
                        goal,
                        parent_chat_context=list(chat_history),
                        clarifications_enabled=not args.no_clarifications,
                        persist=args.persist,
                    )
                )

                chat_history.append({"role": "user", "content": goal})

                # B. AWAIT: Pass the handle to the interactive waiter from utils
                print(steering_controls_hint(voice_enabled=args.voice))
                final_result = await await_with_interrupt(
                    handle,
                    enable_voice_steering=args.voice,
                    clarification_up_q=clar_up_q,
                    clarification_down_q=clar_down_q,
                    clarifications_enabled=not args.no_clarifications,
                    chat_context=list(chat_history),
                    persist_mode=args.persist,
                )

                # C. PROCESS RESULT: Print the final outcome
                print("\n" + "=" * 60)
                print("✅ Task Completed. Result:")
                print("=" * 60)
                print(final_result)
                print("=" * 60 + "\n")
                chat_history.append({"role": "assistant", "content": final_result})

                if args.voice:
                    speak("Task complete.")

            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break
            except Exception as e:
                LG.error("An error occurred in the main loop: %s", e, exc_info=True)
                print(f"❌ An unexpected error occurred: {e}")

    finally:
        # Ensure resources like the computer backend are closed gracefully
        print("Shutting down actor resources...")
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
            "Sandbox forced to exit due to unhandled exception in main.",
            exc_info=True,
        )


if __name__ == "__main__":
    main()
