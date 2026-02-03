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
import time
from contextlib import suppress
from multiprocessing import get_context
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

import unify

from pathlib import Path

from sandboxes.conversation_manager.cm_init import initialize_cm, shutdown_cm
from sandboxes.conversation_manager.config_manager import (
    ActorConfig,
    ConfigurationManager,
)
from sandboxes.conversation_manager.agent_service_bootstrap import (
    free_agent_service_port,
    get_agent_service_log_path,
    try_start_agent_service_direct,
    try_auto_bootstrap_agent_service,
)
from sandboxes.conversation_manager.event_subscriber import subscribe_to_responses
from sandboxes.conversation_manager.event_tree_display import EventTreeDisplay
from sandboxes.conversation_manager.log_aggregator import LogAggregator
from sandboxes.conversation_manager.repl import SandboxState, run_repl
from sandboxes.conversation_manager.trace_display import TraceDisplay
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


def _terminate_process_tree(proc: Any, *, timeout_s: float = 2.0) -> None:
    """Terminate then kill a multiprocessing.Process best-effort."""

    try:
        if proc is None or not hasattr(proc, "is_alive"):
            return
        if not proc.is_alive():
            return
    except Exception:
        # If we can't determine, still try terminate.
        pass
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.join(timeout=float(timeout_s))
    except Exception:
        pass
    try:
        if hasattr(proc, "is_alive") and proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.join(timeout=0.5)
            except Exception:
                pass
    except Exception:
        pass


def _build_worker_config(*, args: Any, actor_config: ActorConfig) -> dict:
    """
    Build the stable config dict passed to both UI + worker processes.

    Keep this payload small and explicit. Extra keys are allowed but the UI/worker
    should only rely on the documented fields.
    """

    cfg = {
        # Primary mode selection
        "actor_type": actor_config.actor_type,
        "managers_mode": actor_config.managers_mode,
        "computer_backend_mode": actor_config.computer_backend_mode,
        # Computer backend / agent-service
        "agent_server_url": getattr(args, "agent_server_url", "http://localhost:3000"),
        "agent_mode": getattr(args, "agent_mode", "web"),
        "headless": bool(getattr(args, "headless", False)),
        # UX
        "voice": bool(getattr(args, "voice", False)),
        "debug": bool(getattr(args, "debug", False)),
        # Project
        "project_name": getattr(args, "project_name", "unity"),
        "overwrite": bool(getattr(args, "overwrite", False)),
        # Worker-only flags (still part of the stable config contract)
        "agent_service_bootstrap": (
            getattr(args, "agent_service_bootstrap", "guide") == "auto"
        ),
        "real_comms": bool(getattr(args, "real_comms", False)),
        "auto_confirm": bool(getattr(args, "auto_confirm", False)),
        # Nested copy for future-proofing (UI already prefers this when present).
        "actor_config": actor_config.to_json_obj(),
    }
    return cfg


async def _run_gui_mode_multiprocess(*, args: Any, config: dict) -> bool:
    """
    Run Textual GUI as a dedicated UI process + worker process (spawn context).

    This function blocks until either process exits or shutdown is requested.
    """

    # Avoid importing Textual in environments where it isn't installed.
    try:
        from sandboxes.conversation_manager import gui as _gui_mod

        if not bool(getattr(_gui_mod, "_TEXTUAL_AVAILABLE", False)):
            print(
                "⚠️ GUI mode unavailable (Textual not installed); falling back to REPL.",
            )
            args.gui = False
            return False
    except Exception:
        print("⚠️ GUI mode unavailable (Textual import failed); falling back to REPL.")
        args.gui = False
        return False

    from sandboxes.conversation_manager import gui_main, gui_worker
    from sandboxes.conversation_manager.ipc_protocol import (
        MessageType,
        create_message,
        new_message_id,
    )

    ctx = get_context("spawn")
    ui_to_worker = ctx.Queue(maxsize=100)
    worker_to_ui = ctx.Queue(maxsize=5000)

    worker_process = ctx.Process(
        target=gui_worker.main,
        args=(ui_to_worker, worker_to_ui, config),
    )

    # Start worker first so UI can connect quickly.
    worker_process.start()
    ui_cfg = dict(config or {})
    try:
        ui_cfg["worker_pid"] = int(worker_process.pid or 0) or None
    except Exception:
        ui_cfg["worker_pid"] = None
    ui_process = ctx.Process(
        target=gui_main.main,
        args=(ui_to_worker, worker_to_ui, ui_cfg),
    )
    ui_process.start()

    # Exit triggers:
    # - Ctrl+C / SIGTERM
    shutdown_requested = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_shutdown() -> None:
        shutdown_requested.set()

    with suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGINT, _request_shutdown)
        loop.add_signal_handler(signal.SIGTERM, _request_shutdown)

    async def _try_graceful_shutdown() -> None:
        try:
            ui_to_worker.put_nowait(
                create_message(MessageType.SHUTDOWN, payload={}, id=new_message_id()),
            )
        except Exception:
            pass

    try:
        while True:
            if shutdown_requested.is_set():
                break

            ui_alive = ui_process.is_alive()
            worker_alive = worker_process.is_alive()

            if not ui_alive or not worker_alive:
                # If the worker died, give the UI a moment to show an error state.
                if ui_alive and (not worker_alive):
                    try:
                        worker_to_ui.put_nowait(
                            create_message(
                                MessageType.WORKER_EXIT,
                                payload={"restart": False, "config": None},
                                id=None,
                            ),
                        )
                    except Exception:
                        pass
                    await asyncio.sleep(0.8)
                break

            await asyncio.sleep(0.3)
    finally:
        # Ask the worker to shutdown first (best-effort) so it can cleanup agent-service.
        await _try_graceful_shutdown()
        # Give a short grace period for clean exit.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if (not ui_process.is_alive()) and (not worker_process.is_alive()):
                break
            await asyncio.sleep(0.1)

        # Hard cleanup to avoid orphans.
        _terminate_process_tree(ui_process)
        _terminate_process_tree(worker_process)

        # Best-effort cleanup: ensure agent-service is not left running if the
        # runtime process was terminated before it could shut down.
        try:
            await asyncio.to_thread(
                free_agent_service_port,
                repo_root=Path(__file__).resolve().parents[2],
                agent_server_url=str(
                    config.get("agent_server_url") or "http://localhost:3000",
                ),
                progress=(lambda _m: None),
            )
        except Exception:
            pass

    # Restart detection: UI exits with a special code when it wants sandbox restart.
    try:
        return int(getattr(ui_process, "exitcode", 0) or 0) == 23
    except Exception:
        return False


async def _main_async() -> None:
    # Used for best-effort executor shutdown at the end of the run.
    main_loop = asyncio.get_running_loop()
    inactivity_shutdown = False

    parser = build_cli_parser("ConversationManager sandbox")
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Enable the Textual GUI (optional).",
    )
    parser.add_argument(
        "--agent-server-url",
        dest="agent_server_url",
        default="http://localhost:3000",
        metavar="URL",
        help="agent-service base URL (default: http://localhost:3000)",
    )
    parser.add_argument(
        "--agent-mode",
        dest="agent_mode",
        default="web",
        choices=["web", "desktop"],
        help="agent-service mode (default: web)",
    )
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=False,
        help="(real web mode) launch Chromium in headless mode",
    )
    parser.add_argument(
        "--agent-service-bootstrap",
        dest="agent_service_bootstrap",
        default="guide",
        choices=["off", "guide", "auto"],
        help=(
            "Mode 3 helper: "
            "'guide' prints step-by-step setup instructions if agent-service is missing; "
            "'auto' also tries to install/build/start agent-service automatically."
        ),
    )
    parser.add_argument(
        "--show-trace",
        dest="show_trace",
        action="store_true",
        default=False,
        help="(CodeAct) auto-print execution trace after each code turn (REPL only)",
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
        help="(real-comms) Auto-confirm all outbound actions (use with care).",
    )
    args = parser.parse_args()

    # Best-effort sink for computer activity lines (used by sandbox-only wrappers).
    def _computer_log_sink(line: str) -> None:
        sink = getattr(args, "_gui_line_sink", None)
        try:
            if callable(sink):
                sink(line)
                return
        except Exception:
            pass
        print(line)

    setattr(args, "_computer_log_sink", _computer_log_sink)

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
        log_file=".logs_conversation_sandbox.txt",
        tcp_port=getattr(args, "log_tcp_port", 0) or 0,
        http_tcp_port=getattr(args, "http_log_tcp_port", 0) or 0,
        unify_requests_log_file=".logs_unify_requests.txt" if args.debug else None,
    )
    LG.setLevel(logging.DEBUG if args.debug else logging.INFO)
    try:
        LG.info(
            "Sandbox starting pid=%s project=%s gui=%s",
            os.getpid(),
            args.project_name,
            bool(getattr(args, "gui", False)),
        )
    except Exception:
        pass

    # Keep sandbox logs readable by default. Full traces are still available via --debug.
    if not args.debug:
        for name in ("unify", "unify_requests", "unillm", "UnifyAsyncLogger"):
            try:
                logging.getLogger(name).setLevel(logging.WARNING)
            except Exception:
                pass

    _suppress_litellm_noise()

    # Project-local config manager (also reused by `config` command).
    project_root = Path(__file__).resolve().parents[2]
    cfg_mgr = ConfigurationManager(
        project_name=args.project_name,
        project_root=project_root,
    )
    setattr(args, "_config_manager", cfg_mgr)

    selected: ActorConfig | None = None

    def _prompt() -> ActorConfig:
        last_used = cfg_mgr.load_config()
        print("ConversationManager Sandbox")
        print("═══════════════════════════════════════════════════════════")
        print("")
        print("Select Actor Configuration:")
        print("")
        print("1. SandboxSimulatedActor (simulated managers, no computer interface)")
        print("2. CodeActActor + Simulated Managers (mock computer backend)")
        print("3. CodeActActor + Real Managers + Real Computer Interface")
        print("")
        print(
            f"Last used: [{_to_choice(last_used.actor_type)}] {_label(last_used.actor_type)}",
        )
        print("")
        raw = input("Enter choice (1-3) or press Enter for last used: ").strip()
        if not raw:
            return last_used
        if raw in {"1", "2", "3"}:
            return ActorConfig(actor_type=_from_choice(raw))
        print("⚠️ Invalid choice, using last used.")
        return last_used

    def _to_choice(actor_type: str) -> str:
        return {"simulated": "1", "codeact_simulated": "2", "codeact_real": "3"}.get(
            actor_type,
            "1",
        )

    def _from_choice(choice: str) -> str:
        return {"1": "simulated", "2": "codeact_simulated", "3": "codeact_real"}[choice]

    def _label(actor_type: str) -> str:
        return {
            "simulated": "SandboxSimulatedActor (simulated managers, no computer interface)",
            "codeact_simulated": "CodeActActor + Simulated Managers (mock computer backend)",
            "codeact_real": "CodeActActor + Real Managers + Real Computer Interface",
        }.get(
            actor_type,
            "SandboxSimulatedActor (simulated managers, no computer interface)",
        )

    # Outer loop supports runtime config switching (REPL command `config`).
    while True:
        if selected is None:
            selected = await asyncio.to_thread(_prompt)
        # Validate infra with retry/switch/exit loop.
        while True:

            def _should_offer_agent_help() -> bool:
                return (
                    getattr(selected, "actor_type", None) == "codeact_real"
                    and getattr(args, "agent_service_bootstrap", "guide") != "off"
                )

            async def _attempt_agent_service_recovery() -> None:
                """
                One consolidated action:
                1) try starting agent-service directly (no installs/builds)
                2) if that fails, fall back to auto-bootstrap (install/build/start)
                """
                # Avoid spawning multiple subprocesses in the same sandbox run.
                existing = getattr(args, "_agent_service_process", None)
                if (
                    existing is not None
                    and hasattr(existing, "poll")
                    and existing.poll() is None
                ):
                    return

                agent_server_url = getattr(
                    args,
                    "agent_server_url",
                    "http://localhost:3000",
                )
                try:
                    setattr(
                        args,
                        "_agent_service_log_path",
                        str(
                            get_agent_service_log_path(
                                repo_root=project_root,
                                agent_server_url=str(agent_server_url),
                            ),
                        ),
                    )
                except Exception:
                    pass

                try:
                    # Best-effort port cleanup before attempting a restart.
                    await asyncio.to_thread(
                        free_agent_service_port,
                        repo_root=project_root,
                        agent_server_url=agent_server_url,
                        progress=(lambda m: print(m)),
                    )
                    print("\n[agent-service] Attempting to start (direct)...\n")
                    res1 = await asyncio.to_thread(
                        try_start_agent_service_direct,
                        repo_root=project_root,
                        agent_server_url=agent_server_url,
                        progress=(lambda m: print(m)),
                    )
                    if res1.ok and res1.process is not None:
                        setattr(args, "_agent_service_process", res1.process)
                        print(f"[agent-service] {res1.summary}\n")
                        return
                    if res1.process is not None:
                        try:
                            res1.process.terminate()
                            res1.process.wait(timeout=2.0)
                        except Exception:
                            try:
                                res1.process.kill()
                            except Exception:
                                pass
                    print(f"[agent-service] {res1.summary}\n")
                except Exception as exc:
                    print(f"[agent-service] Direct start failed: {exc}\n")

                try:
                    print("[agent-service] Falling back to auto-bootstrap...\n")
                    res2 = await asyncio.to_thread(
                        try_auto_bootstrap_agent_service,
                        repo_root=project_root,
                        agent_server_url=agent_server_url,
                        progress=(lambda m: print(m)),
                    )
                    if res2.ok and res2.process is not None:
                        setattr(args, "_agent_service_process", res2.process)
                        print(f"[agent-service] {res2.summary}\n")
                        return
                    if res2.process is not None:
                        try:
                            res2.process.terminate()
                            res2.process.wait(timeout=2.0)
                        except Exception:
                            try:
                                res2.process.kill()
                            except Exception:
                                pass
                    print(f"[agent-service] {res2.summary}\n")
                except Exception as exc:
                    print(f"[agent-service] Auto-bootstrap failed: {exc}\n")

            # In REPL mode, we can optionally attempt agent-service recovery before
            # validation. In multi-process GUI mode, the worker owns this lifecycle.
            if not bool(getattr(args, "gui", False)):
                if (
                    getattr(selected, "actor_type", None) == "codeact_real"
                    and getattr(args, "agent_service_bootstrap", "guide") == "auto"
                ):
                    await _attempt_agent_service_recovery()

            vr = await asyncio.to_thread(
                cfg_mgr.validate_config,
                selected,
                agent_server_url=getattr(
                    args,
                    "agent_server_url",
                    "http://localhost:3000",
                ),
                require_agent_service_running=(
                    not (
                        bool(getattr(args, "gui", False))
                        and (not bool(getattr(args, "real_comms", False)))
                    )
                ),
            )
            if vr.ok:
                break

            print("❌ Configuration Error")
            print("═══════════════════════════════════════════════════════════")
            print("")
            if vr.failed_component:
                print(f"Failed to initialize: {vr.failed_component}")
            if vr.error:
                print(f"Reason: {vr.error}")
            if getattr(vr, "help_text", None):
                print("")
                print("How to fix:")
                print(getattr(vr, "help_text"))
            print("")
            print("Options:")
            if (not bool(getattr(args, "gui", False))) and _should_offer_agent_help():
                print(
                    "1. Retry (attempt start agent-service, then bootstrap if needed)",
                )
            else:
                print("1. Retry (after fixing infrastructure)")
            print("2. Switch to different configuration")
            print("3. Exit sandbox")
            print("")
            choice = (
                await asyncio.to_thread(
                    input,
                    "Enter choice (1-3): ",
                )
            ).strip()
            if choice == "1":
                if (
                    not bool(getattr(args, "gui", False))
                ) and _should_offer_agent_help():
                    await _attempt_agent_service_recovery()
                continue
            if choice == "2":
                selected = await asyncio.to_thread(_prompt)
                continue
            raise SystemExit(1)

        cfg_mgr.save_config(selected)
        setattr(args, "_actor_config", selected)

        # GUI mode: do not initialize CM in this process.
        if bool(getattr(args, "gui", False)) and not bool(
            getattr(args, "real_comms", False),
        ):
            cfg = _build_worker_config(args=args, actor_config=selected)
            # Run UI + worker processes; returns when they exit.
            restart = await _run_gui_mode_multiprocess(args=args, config=cfg)
            if restart:
                # The runtime process persists the selected configuration to the
                # project-local config file before requesting a restart.
                selected = cfg_mgr.load_config()
                continue
            break

        cm = await initialize_cm(args=args)
        setattr(args, "_cm", cm)

        # Display components (trace/tree/logs) are instantiated here and wired into
        # the event subscriber and command router.
        trace_display = TraceDisplay()
        event_tree_display = EventTreeDisplay()
        log_aggregator = LogAggregator()
        setattr(args, "_trace_display", trace_display)
        setattr(args, "_event_tree_display", event_tree_display)
        setattr(args, "_log_aggregator", log_aggregator)

        # Wire trace capture into CodeActActor execution boundary (SessionExecutor).
        # This keeps trace capture local to the sandbox UI surface.
        try:
            actor = getattr(cm, "actor", None)
            executor = getattr(actor, "_session_executor", None)
            if executor is not None and trace_display is not None:
                auto_print = bool(getattr(args, "show_trace", False)) and (
                    not bool(args.gui)
                )

                def _after_capture(_entry: object) -> None:
                    if not auto_print:
                        # In GUI mode, refresh the trace panel when a new entry arrives.
                        try:
                            if bool(getattr(args, "gui", False)):
                                req = getattr(args, "_gui_refresh_request", None)
                                if callable(req):
                                    req(trace=True)
                        except Exception:
                            pass
                        return
                    try:
                        print(trace_display.render_recent(1))
                    except Exception:
                        pass

                # IMPORTANT: the sandbox can "restart" within the same Python process.
                # If we only wrap once, the wrapper will keep capturing into the previous
                # TraceDisplay instance, and the active UI will show "(no trace entries yet)".
                # Store the original execute once and re-wrap against the current display
                # every time we (re)initialize the sandbox.
                orig = getattr(executor, "_cm_sandbox_execute_orig", None)
                if not callable(orig):
                    cand = getattr(executor, "execute", None)
                    if callable(cand):
                        setattr(executor, "_cm_sandbox_execute_orig", cand)
                        orig = cand
                if callable(orig):
                    setattr(
                        executor,
                        "execute",
                        trace_display.install_executor_wrapper(
                            execute_fn=orig,
                            after_capture=_after_capture,
                        ),
                    )
                    setattr(executor, "_cm_sandbox_trace_wrapped", True)
        except Exception:
            pass

        state = SandboxState()

        # Start outbound event subscription (prints responses as they arrive).
        stop_sub = asyncio.Event()

        async def _display(line: str) -> None:
            # In GUI mode, the Textual app installs a line sink so the subscriber
            # can append to the conversation/log panes instead of printing.
            sink = getattr(args, "_gui_line_sink", None)
            try:
                if callable(sink):
                    sink(line)
                    return
            except Exception:
                pass
            print(line)

        sub_task = asyncio.create_task(
            subscribe_to_responses(
                cm=cm,
                sandbox_state=state,
                display_callback=_display,
                include_call_guidance=bool(args.debug)
                or bool(getattr(args, "voice", False)),
                voice_enabled=bool(getattr(args, "voice", False)),
                stop_event=stop_sub,
                trace_display=trace_display,
                event_tree_display=event_tree_display,
                log_aggregator=log_aggregator,
                ui_refresh_callback=(
                    (
                        lambda: (
                            getattr(args, "_gui_refresh_request", None)
                            or (lambda **_kw: None)
                        )(tree=True, logs=True)
                    )
                    if bool(getattr(args, "gui", False))
                    else None
                ),
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
        # Clear any prior restart flags.
        setattr(args, "_restart_requested", False)
        setattr(args, "_restart_actor_config", None)

        try:
            if args.gui and args.real_comms:
                print("⚠️ Real-comms mode requires REPL. Starting REPL instead.")
                args.gui = False

            ui_task = asyncio.create_task(run_repl(args=args, state=state))
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
                print("\nShutting down...")

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
            # If we auto-started agent-service, stop it on exit.
            try:
                proc = getattr(args, "_agent_service_process", None)
                if proc is not None and hasattr(proc, "terminate"):
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
            except Exception:
                pass
            # Best-effort: ensure the configured port isn't left bound by a sandbox-started
            # agent-service instance (or a stale previous one from this repo).
            try:
                await asyncio.to_thread(
                    free_agent_service_port,
                    repo_root=project_root,
                    agent_server_url=getattr(
                        args,
                        "agent_server_url",
                        "http://localhost:3000",
                    ),
                    progress=(lambda _m: None),
                )
            except Exception:
                pass

        # Restart requested by REPL `config`.
        if bool(getattr(args, "_restart_requested", False)):
            nxt = getattr(args, "_restart_actor_config", None)
            selected = nxt if isinstance(nxt, ActorConfig) else None
            continue

        # Normal exit (no restart)
        break

    # If any background asyncio.to_thread() calls are still running, Python can
    # hang for minutes while shutting down the loop's default executor.
    #
    # For inactivity-triggered shutdown we prefer a fast exit, since this is a
    # developer sandbox (not a long-lived service). We attempt a best-effort
    # executor shutdown and if it's still stuck, force-exit.
    try:
        if hasattr(main_loop, "shutdown_default_executor"):
            await asyncio.wait_for(main_loop.shutdown_default_executor(), timeout=2.0)
    except Exception:
        if inactivity_shutdown:
            os._exit(0)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
