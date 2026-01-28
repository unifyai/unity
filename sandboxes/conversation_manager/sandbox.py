"""
ConversationManager sandbox entrypoint.

This module wires together:
- project activation + logging
- in-process ConversationManager startup (simulated or real-comms)
- outbound event subscription (prints CM responses)
- either REPL mode (default) or Textual GUI mode (`--gui`)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from contextlib import suppress

from dotenv import load_dotenv

load_dotenv(override=True)

import unify

from sandboxes.conversation_manager.cm_init import initialize_cm, shutdown_cm
from sandboxes.conversation_manager.event_subscriber import subscribe_to_responses
from sandboxes.conversation_manager.gui import run_gui_mode
from sandboxes.conversation_manager.repl import SandboxState, run_repl
from sandboxes.utils import (
    activate_project,
    build_cli_parser,
    configure_sandbox_logging,
)

LG = logging.getLogger("conversation_manager_sandbox")


def _suppress_litellm_noise() -> None:
    """
    LiteLLM prints a provider list to stdout when it cannot infer a provider from a
    model string. This is helpful in isolation but very noisy in an interactive REPL.

    Sandbox runs may intentionally use model strings that are normalized by unillm,
    so we suppress these debug prints while preserving actual exceptions.
    """
    try:
        import litellm  # type: ignore

        litellm.suppress_debug_info = True
    except Exception:
        pass


async def _main_async() -> None:
    parser = build_cli_parser("ConversationManager sandbox")
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Enable the Textual GUI (optional).",
    )
    parser.add_argument(
        "--real-comms",
        dest="real_comms",
        action="store_true",
        default=False,
        help="Use real comms (SMS/email/calls). Requires external infrastructure and prompts for confirmation.",
    )
    parser.add_argument(
        "--auto-confirm",
        dest="auto_confirm",
        action="store_true",
        default=False,
        help="(real-comms) Auto-confirm all outbound actions (DANGEROUS).",
    )
    args = parser.parse_args()

    # Unify project activation
    activate_project(args.project_name, args.overwrite)

    # Optional project version rollback (0-indexed)
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

    # Logging via shared helper
    configure_sandbox_logging(
        log_in_terminal=args.log_in_terminal,
        log_file=".logs_conversation_sandbox.txt" if args.debug else None,
        tcp_port=getattr(args, "log_tcp_port", 0) or 0,
        http_tcp_port=getattr(args, "http_log_tcp_port", 0) or 0,
        unify_requests_log_file=".logs_unify_requests.txt" if args.debug else None,
    )
    LG.setLevel(logging.DEBUG if args.debug else logging.INFO)

    # Keep sandbox logs readable by default. Full traces are still available via --debug.
    if not args.debug:
        for name in ("unify", "unify_requests", "unillm", "UnifyAsyncLogger"):
            try:
                logging.getLogger(name).setLevel(logging.WARNING)
            except Exception:
                pass

    _suppress_litellm_noise()

    cm = await initialize_cm(args=args)

    # Attach cm onto args so downstream UI layers can access it without additional plumbing.
    setattr(args, "_cm", cm)

    state = SandboxState()

    # Start outbound event subscription (prints responses as they arrive).
    stop_sub = asyncio.Event()

    async def _display(line: str) -> None:
        print(line)

    sub_task = asyncio.create_task(
        subscribe_to_responses(
            cm=cm,
            sandbox_state=state,
            display_callback=_display,
            include_call_guidance=bool(args.debug),
            voice_enabled=bool(getattr(args, "voice", False)),
            stop_event=stop_sub,
        ),
    )

    # Exit triggers:
    # - Ctrl+C / SIGTERM
    # - CM inactivity timeout (cm.stop is set by ConversationManager.check_inactivity)
    shutdown_requested = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_shutdown() -> None:
        shutdown_requested.set()

    with suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGINT, _request_shutdown)
        loop.add_signal_handler(signal.SIGTERM, _request_shutdown)

    inactivity_shutdown = False
    try:
        if args.gui and args.real_comms:
            print("⚠️ Real-comms mode requires REPL. Starting REPL instead.")
            args.gui = False
        if args.gui and getattr(args, "voice", False):
            print("⚠️ Voice mode runs in REPL. Starting REPL instead.")
            args.gui = False

        async def _run_ui() -> None:
            if args.gui:
                ran = False
                try:
                    ran = await run_gui_mode(cm=cm, args=args, state=state)
                except Exception as exc:
                    LG.warning(
                        "GUI mode failed; falling back to REPL: %s",
                        exc,
                        exc_info=True,
                    )
                    ran = False
                if not ran:
                    print("⚠️ GUI mode unavailable/failed; falling back to REPL.")
                    await run_repl(args=args, state=state)
            else:
                await run_repl(args=args, state=state)

        ui_task = asyncio.create_task(_run_ui())
        cm_stop_task = None
        try:
            cm_stop = getattr(cm, "stop", None)
            if cm_stop is not None and hasattr(cm_stop, "wait"):
                cm_stop_task = asyncio.create_task(cm_stop.wait())
        except Exception:
            cm_stop_task = None

        done, pending = await asyncio.wait(
            {ui_task, asyncio.create_task(shutdown_requested.wait())}
            | ({cm_stop_task} if cm_stop_task else set()),
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cm_stop_task and cm_stop_task in done:
            inactivity_shutdown = True
            print("\n⏲️ Inactivity timeout reached — shutting down.")
        if shutdown_requested.is_set():
            print("\nShutting down…")

        for t in pending:
            t.cancel()
        if not ui_task.done():
            ui_task.cancel()
        with suppress(asyncio.CancelledError):
            await ui_task
    finally:
        try:
            stop_sub.set()
        except Exception:
            pass
        try:
            sub_task.cancel()
        except Exception:
            pass
        with suppress(asyncio.CancelledError):
            await sub_task
        await shutdown_cm(cm)

        # If any background asyncio.to_thread() calls are still running, Python can
        # hang for minutes while shutting down the loop's default executor.
        #
        # For inactivity-triggered shutdown we prefer a fast exit, since this is a
        # developer sandbox (not a long-lived service). We attempt a best-effort
        # executor shutdown and if it's still stuck, force-exit.
        try:
            if hasattr(loop, "shutdown_default_executor"):
                await asyncio.wait_for(loop.shutdown_default_executor(), timeout=2.0)
        except Exception:
            if inactivity_shutdown:
                os._exit(0)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
