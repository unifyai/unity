"""
Worker process entrypoint for the multi-process ConversationManager sandbox.

This worker hosts:
- ConversationManager + Actor + ManagerRegistry singletons (in-process)
- EventBroker subscriptions (app:comms:* + app:actor:*)
- EventBus subscriptions (ManagerMethod)
- Command execution via `CommandRouter.execute_raw()`

The UI runs in a separate process and communicates with this worker via
`multiprocessing.Queue` using the schemas in `ipc_protocol.py`.

Notes
-----
- This module is sandbox-only glue code; it must not modify production modules.
- Blocking operations are *allowed* here (they won't freeze the UI).
- Messages sent to UI must be JSON-serializable.
"""

from __future__ import annotations

import asyncio
import logging
import queue as _queue
import traceback as _traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from dotenv import load_dotenv

from sandboxes.utils import configure_sandbox_logging

from sandboxes.conversation_manager.agent_service_bootstrap import (
    try_auto_bootstrap_agent_service,
    try_start_agent_service_direct,
    free_agent_service_port,
)
from sandboxes.conversation_manager.cm_init import initialize_cm, shutdown_cm
from sandboxes.conversation_manager.command_router import CommandRouter
from sandboxes.conversation_manager.config_manager import (
    ActorConfig,
    ConfigurationManager,
)
from sandboxes.conversation_manager.event_publisher import EventPublisher
from sandboxes.conversation_manager.event_subscriber import subscribe_to_responses
from sandboxes.conversation_manager.event_tree_display import EventTreeDisplay
from sandboxes.conversation_manager.log_aggregator import LogAggregator
from sandboxes.conversation_manager.trace_display import TraceDisplay
from sandboxes.conversation_manager.ipc_protocol import (
    MessageType,
    create_message,
    parse_message,
    serialize_event,
)
from unity.events.event_bus import EVENT_BUS, Event as BusEvent

LG = logging.getLogger("conversation_manager_sandbox")

_RESTART_EXIT_CODE = 23


def _simplify_execute_code_result(result: Any) -> dict[str, Any]:
    """
    Convert CodeActActor execute_code() results into a JSON-safe dict for IPC.

    The in-process Python executor can return rich stdout/stderr as a list of parts
    (text/images). For the sandbox Trace panel we keep a readable text summary.
    """

    def _parts_to_text(v: Any) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            chunks: list[str] = []
            img_count = 0
            for p in v:
                try:
                    # TextPart/ImagePart may come as dicts, pydantic models, or objects.
                    if hasattr(p, "model_dump"):
                        try:
                            p = p.model_dump(mode="python")  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    if isinstance(p, dict):
                        typ = str(p.get("type") or "")
                        if typ == "image":
                            img_count += 1
                            continue
                        t = p.get("text")
                        if t is not None:
                            s = str(t)
                            if s:
                                chunks.append(s)
                            continue
                    if hasattr(p, "text"):
                        s = str(getattr(p, "text") or "")
                        if s:
                            chunks.append(s)
                        continue
                    # Unknown object: represent conservatively.
                    s = str(p)
                    if s and s != "None":
                        chunks.append(s)
                except Exception:
                    continue
            if img_count:
                chunks.append(f"\n[images: {img_count}]")
            return "".join(chunks)
        try:
            return str(v)
        except Exception:
            return ""

    out: dict[str, Any] = {}
    try:
        if hasattr(result, "model_dump"):
            d = result.model_dump(mode="python")  # type: ignore[attr-defined]
        elif isinstance(result, dict):
            d = dict(result)
        else:
            d = {"stdout": str(result)}
    except Exception:
        d = {"stdout": ""}

    try:
        out["stdout"] = _parts_to_text(d.get("stdout"))
    except Exception:
        out["stdout"] = ""
    try:
        out["stderr"] = _parts_to_text(d.get("stderr"))
    except Exception:
        out["stderr"] = ""
    try:
        out["error"] = (
            d.get("error")
            if isinstance(d.get("error"), str)
            else (str(d.get("error")) if d.get("error") else None)
        )
    except Exception:
        out["error"] = None
    for k in (
        "language",
        "state_mode",
        "session_id",
        "session_name",
        "venv_id",
        "duration_ms",
        "computer_used",
    ):
        try:
            v = d.get(k)
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            else:
                out[k] = str(v)
        except Exception:
            continue
    return out


def _filter_kwargs_for_callable(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Filter keyword args to those accepted by `fn`.

    The async tool-loop infrastructure may pass internal/private kwargs (e.g.
    `_parent_chat_context`) which are not part of the tool's public signature.
    Sandbox wrappers must drop them before delegating to the real tool.
    """

    try:
        import inspect as _inspect

        sig = _inspect.signature(fn)
        # If fn accepts **kwargs, keep all.
        for p in sig.parameters.values():
            if p.kind == _inspect.Parameter.VAR_KEYWORD:
                return dict(kwargs)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        # Conservative fallback: drop private keys.
        return {k: v for k, v in kwargs.items() if not str(k).startswith("_")}


@dataclass
class WorkerSandboxState:
    """
    Minimal sandbox state needed by `CommandRouter` + `subscribe_to_responses`.

    UI-specific state (e.g. "steering hint visible") is intentionally omitted.
    """

    chat_history: list[dict] = field(default_factory=list)
    in_call: bool = False
    brain_run_in_flight: bool = False
    paused: bool = False
    last_event_published_at: float = 0.0
    queued_events: list[Any] = field(default_factory=list)
    awaiting_config_choice: bool = False
    pending_clarification: bool = False

    def reset_ephemeral(self) -> None:
        self.chat_history.clear()
        self.in_call = False
        self.brain_run_in_flight = False
        self.paused = False
        self.last_event_published_at = 0.0
        self.queued_events.clear()
        self.awaiting_config_choice = False
        self.pending_clarification = False


class _Sender:
    """Best-effort, drop-aware sender into `worker_to_ui` queue."""

    def __init__(self, *, worker_to_ui) -> None:
        self._q = worker_to_ui
        self._dropped_noncritical = 0

    def send(self, msg: Dict[str, Any], *, critical: bool) -> None:
        try:
            if critical:
                # For critical messages, we prefer delivery even if it blocks.
                self._q.put(msg)
                return
            self._q.put_nowait(msg)
        except _queue.Full:
            # Non-critical drops are allowed by design.
            if not critical:
                self._dropped_noncritical += 1
                if self._dropped_noncritical % 200 == 0:
                    LG.warning(
                        "worker_to_ui queue full; dropped %d non-critical messages",
                        self._dropped_noncritical,
                    )
                return
            # Critical fallback: block.
            try:
                self._q.put(msg)
            except Exception:
                return
        except Exception:
            return

    def send_lines(self, lines: list[str], *, id: str | None = None) -> None:
        if not lines:
            return
        self.send(
            create_message(
                MessageType.LINES,
                id=id,
                payload={"lines": [str(l) for l in lines if str(l).strip() != ""]},
            ),
            critical=True,
        )

    def send_error(self, message: str, *, tb: str = "", id: str | None = None) -> None:
        self.send(
            create_message(
                MessageType.ERROR,
                id=id,
                payload={"message": str(message), "traceback": str(tb or "")},
            ),
            critical=True,
        )

    def send_event(self, *, channel: str, event: dict, critical: bool = False) -> None:
        self.send(
            create_message(
                MessageType.EVENT,
                id=None,
                payload={"channel": str(channel), "event": event},
            ),
            critical=bool(critical),
        )

    def send_state(self, *, active: bool, in_call: bool, pending_clarification: bool):
        self.send(
            create_message(
                MessageType.STATE,
                id=None,
                payload={
                    "active": bool(active),
                    "in_call": bool(in_call),
                    "pending_clarification": bool(pending_clarification),
                },
            ),
            critical=False,
        )

    def send_worker_exit(
        self,
        *,
        restart: bool,
        config: Dict[str, Any] | None = None,
    ) -> None:
        self.send(
            create_message(
                MessageType.WORKER_EXIT,
                id=None,
                payload={
                    "restart": bool(restart),
                    "config": config,
                },
            ),
            critical=True,
        )


def _forward_broker_event(*, sender: _Sender, channel: str, event: dict) -> None:
    """
    Forward a raw broker event (`Event.to_dict()`) to the UI.

    This is used by the IPC GUI. Any exception here would otherwise be swallowed
    inside `subscribe_to_responses()`.
    """

    try:
        sender.send_event(
            channel=f"broker:{channel}",
            event=event,
            critical=False,
        )
    except Exception as exc:
        try:
            name = ""
            if isinstance(event, dict):
                name = str(event.get("event_name") or "")
            LG.warning(
                "[runtime] failed forwarding broker event channel=%s name=%s (%s: %s)",
                channel,
                name,
                type(exc).__name__,
                exc,
            )
        except Exception:
            pass


def _terminate_process_best_effort(proc: Any, *, timeout_s: float = 2.0) -> None:
    """Best-effort terminate+kill for a subprocess-like object."""

    # If the subprocess was started with `start_new_session=True` (as we do for
    # agent-service), `terminate()` will only signal the parent (often `npm exec`)
    # and can leave the child `node` process running. Prefer killing the whole
    # process group when possible.
    try:
        import os as _os
        import signal as _signal

        pid = int(getattr(proc, "pid", 0) or 0)
        if pid > 0:
            try:
                pgid = _os.getpgid(pid)
            except Exception:
                pgid = None
            if pgid:
                try:
                    _os.killpg(pgid, _signal.SIGTERM)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        if getattr(proc, "poll", None) and proc.poll() is not None:  # type: ignore[truthy-bool]
            return
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        return
    try:
        proc.wait(timeout=float(timeout_s))
    except Exception:
        try:
            # Try to hard-kill the process group first.
            try:
                import os as _os
                import signal as _signal

                pid = int(getattr(proc, "pid", 0) or 0)
                if pid > 0:
                    try:
                        pgid = _os.getpgid(pid)
                    except Exception:
                        pgid = None
                    if pgid:
                        _os.killpg(pgid, _signal.SIGKILL)
                        return
            except Exception:
                pass
            proc.kill()
        except Exception:
            pass


def _coerce_actor_config(config: dict) -> ActorConfig:
    """
    Extract `ActorConfig` from a config dict.

    Supported shapes:
    - {"actor_type": "...", ...} (flat)
    - {"actor_config": {...}} (nested)
    - {"_actor_config": {...}} (internal)
    """

    raw = None
    if isinstance(config.get("actor_config"), dict):
        raw = config.get("actor_config")
    elif isinstance(config.get("_actor_config"), dict):
        raw = config.get("_actor_config")
    else:
        raw = config

    try:
        return ActorConfig.from_json_obj(raw if isinstance(raw, dict) else {})
    except Exception:
        # Safe default.
        return ActorConfig(actor_type="simulated")


def _build_args_namespace(*, config: dict, sender: _Sender) -> Any:
    """
    Convert the worker config dict into an args-like object expected by sandbox code.
    """

    cfg = dict(config or {})
    args = SimpleNamespace(**cfg)
    # Ensure required defaults.
    if not hasattr(args, "project_name"):
        setattr(args, "project_name", "unity")
    if not hasattr(args, "agent_server_url"):
        setattr(args, "agent_server_url", None)
    if not hasattr(args, "headless"):
        setattr(args, "headless", False)
    if not hasattr(args, "agent_mode"):
        setattr(args, "agent_mode", "web")
    if not hasattr(args, "real_comms"):
        setattr(args, "real_comms", False)
    if not hasattr(args, "auto_confirm"):
        setattr(args, "auto_confirm", False)

    # ActorConfig is expected under `_actor_config` by existing sandbox code.
    actor_cfg = _coerce_actor_config(cfg)
    setattr(args, "_actor_config", actor_cfg)

    # Let the computer backend emit useful lines into the UI.
    setattr(args, "_computer_log_sink", lambda line: sender.send_lines([str(line)]))

    return args


async def _subscribe_event_bus(*, sender: _Sender, stop_event: asyncio.Event) -> None:
    """
    Subscribe to selected EventBus event types and forward to the UI.

    We forward `ManagerMethod` only (high-signal, low-volume).
    """

    import time as _time

    async def _forward(evts: list[BusEvent], *, bus_type: str) -> None:
        # ManagerMethod events are low-volume and high-signal; prefer delivery.
        critical = bus_type == "ManagerMethod"
        for e in evts:
            try:
                sender.send_event(
                    channel=f"eventbus:{bus_type}",
                    event=serialize_event(e),
                    critical=critical,
                )
            except Exception:
                continue

    def _mk_async_callback(bus_type: str):
        async def _cb(evts: list[BusEvent]) -> None:
            await _forward(evts, bus_type=bus_type)

        return _cb

    # Wait for EventBus initialization.
    #
    # In this sandbox, `unity.init()` is performed as part of ConversationManager's
    # manager initialization (inside its own initialization workflow). If we try to
    # register callbacks too early, EVENT_BUS is still a proxy and will raise.
    import time as _time

    started_at = _time.monotonic()
    while not stop_event.is_set():
        try:
            if bool(EVENT_BUS):
                break
        except Exception:
            pass
        # After a while, warn the UI but keep waiting; initialization can be slow
        # on fresh environments.
        if (_time.monotonic() - started_at) > 30.0:
            try:
                sender.send_error(
                    "EventBus not initialized yet (still waiting). "
                    "Trace/Event Tree panes will remain empty until it becomes available.",
                    tb="",
                )
            except Exception:
                pass
            started_at = _time.monotonic()  # rate-limit warnings
        await asyncio.sleep(0.15)

    if stop_event.is_set():
        return

    # Register callbacks (retry until success or shutdown).
    registered = False
    while (not stop_event.is_set()) and (not registered):
        try:
            await EVENT_BUS.register_callback(
                event_type="ManagerMethod",
                callback=_mk_async_callback("ManagerMethod"),
                every_n=1,
            )
            registered = True
            try:
                LG.info("[runtime] EventBus callbacks registered")
            except Exception:
                pass
        except Exception as exc:
            # If EventBus isn't available yet, keep the worker running; the UI will
            # still show conversation lines and broker-derived logs.
            try:
                LG.warning(
                    "[runtime] EventBus subscription not ready (%s: %s); retrying...",
                    type(exc).__name__,
                    exc,
                )
            except Exception:
                pass
            await asyncio.sleep(0.5)

    # Keep task alive until shutdown.
    await stop_event.wait()


async def _computer_status_streamer(
    *,
    sender: _Sender,
    args: Any,
    stop_event: asyncio.Event,
    every_s: float = 0.75,
) -> None:
    """
    Stream computer activity snapshots to the UI (best-effort).

    The runtime process owns the computer backend. The UI renders this snapshot
    in its Computer panel.
    """

    last: dict[str, Any] | None = None
    while not stop_event.is_set():
        try:
            activity = getattr(args, "_computer_activity", None)
            if activity is None:
                await asyncio.sleep(every_s)
                continue
            snap = activity.snapshot_sync()
            # Convert ComputerAction objects to plain dicts.
            actions = []
            for a in (snap.get("actions") or [])[:]:
                try:
                    actions.append(
                        {
                            "ts": float(getattr(a, "ts", 0.0) or 0.0),
                            "kind": str(getattr(a, "kind", "") or ""),
                            "detail": str(getattr(a, "detail", "") or ""),
                        },
                    )
                except Exception:
                    continue
            out = {
                "connected": snap.get("connected", None),
                "last_error": snap.get("last_error", None),
                "last_url": snap.get("last_url", None),
                "actions": actions[-50:],
            }
            # Only emit if changed (cheap coalesce).
            if last != out:
                sender.send_event(channel="computer:status", event=out, critical=False)
                last = out
        except Exception:
            pass
        await asyncio.sleep(every_s)


async def _ipc_loop(
    *,
    ui_to_worker,
    sender: _Sender,
    router: CommandRouter,
    state: WorkerSandboxState,
    args: Any,
    cfg_mgr: ConfigurationManager,
    stop_event: asyncio.Event,
) -> None:
    """
    Receive UI→Worker messages and execute them.
    """

    while not stop_event.is_set():
        try:
            # `multiprocessing.Queue.get()` is blocking and not awaitable; run in thread.
            raw_msg = await asyncio.to_thread(ui_to_worker.get)
        except Exception:
            await asyncio.sleep(0.05)
            continue

        if not isinstance(raw_msg, dict):
            continue

        try:
            msg = parse_message(raw_msg)
        except Exception:
            # Ignore malformed messages.
            continue

        if msg.type == MessageType.SHUTDOWN:
            stop_event.set()
            return

        if msg.type != MessageType.EXECUTE_RAW:
            continue

        cmd_id = msg.id
        payload = msg.payload
        raw = payload.raw
        raw_trimmed = (raw or "").strip()
        raw_lower = raw_trimmed.lower()

        # Config switching is handled here (CommandRouter's config switch is REPL-only).
        if state.awaiting_config_choice:
            choice = raw_lower
            if choice in {"cancel", "c", "n", "no"}:
                state.awaiting_config_choice = False
                sender.send_lines(["(config switch cancelled)"], id=cmd_id)
                continue
            actor_type = {
                "1": "simulated",
                "2": "codeact_simulated",
                "3": "codeact_real",
            }.get(
                choice,
            )
            if actor_type is None and choice in {
                "simulated",
                "codeact_simulated",
                "codeact_real",
            }:
                actor_type = choice
            if actor_type is None:
                sender.send_lines(
                    [
                        "⚠️ Please choose 1, 2, or 3 (or type 'cancel').",
                    ],
                    id=cmd_id,
                )
                continue

            new_cfg = ActorConfig(actor_type=actor_type)  # type: ignore[arg-type]
            try:
                vr = await asyncio.to_thread(
                    cfg_mgr.validate_config,
                    new_cfg,
                    agent_server_url=getattr(args, "agent_server_url", None),
                    require_agent_service_running=False,
                )
            except Exception as exc:
                sender.send_error(
                    f"Failed to validate configuration: {type(exc).__name__}: {exc}",
                    tb=_traceback.format_exc(),
                    id=cmd_id,
                )
                continue

            if not vr.ok:
                lines = [
                    "❌ Configuration Error",
                ]
                if vr.failed_component:
                    lines.append(f"Failed to initialize: {vr.failed_component}")
                if vr.error:
                    lines.append(f"Reason: {vr.error}")
                if vr.help_text:
                    lines.append("")
                    lines.append("How to fix:")
                    lines.append(str(vr.help_text))
                lines.append("")
                lines.append(
                    "Choose a different configuration (1-3), or type 'cancel'.",
                )
                sender.send_lines(lines, id=cmd_id)
                continue

            try:
                await asyncio.to_thread(cfg_mgr.save_config, new_cfg)
            except Exception:
                # Not fatal; still restart.
                pass

            sender.send_lines(
                [
                    f"✓ Selected: {actor_type}",
                    "Restarting...",
                ],
                id=cmd_id,
            )
            sender.send_worker_exit(restart=True, config={"actor_type": actor_type})
            stop_event.set()
            return

        if raw_lower in {"config", "switch_actor"}:
            state.awaiting_config_choice = True
            sender.send_lines(
                [
                    "Select actor configuration:",
                    "1. Simulated (no computer interface)",
                    "2. CodeAct + simulated managers (mock computer backend)",
                    "3. CodeAct + real managers + real computer interface",
                    "",
                    "Reply with 1, 2, or 3 (or type 'cancel').",
                ],
                id=cmd_id,
            )
            continue

        try:
            res = await router.execute_raw(
                payload.raw,
                prompt_text=None,
                in_call=payload.in_call,
            )
            if res.lines:
                sender.send_lines(res.lines, id=cmd_id)
            else:
                # Still send an empty `lines` message to acknowledge completion so the
                # UI can clear per-command timeouts.
                sender.send(
                    create_message(
                        MessageType.LINES,
                        id=cmd_id,
                        payload={"lines": []},
                    ),
                    critical=False,
                )
            if res.should_exit:
                stop_event.set()
                return
        except Exception as exc:
            sender.send_error(
                f"Command execution failed: {type(exc).__name__}: {exc}",
                tb=_traceback.format_exc(),
                id=cmd_id,
            )


async def _state_broadcaster(
    *,
    sender: _Sender,
    state: WorkerSandboxState,
    stop_event: asyncio.Event,
) -> None:
    """
    Best-effort state broadcasting with coalescing.

    We poll at a small interval; the state is tiny and this avoids needing to
    thread state-change notifications through all components.
    """

    last: tuple[bool, bool, bool] | None = None
    while not stop_event.is_set():
        try:
            active = bool(getattr(state, "brain_run_in_flight", False))
            in_call = bool(getattr(state, "in_call", False))
            pending = bool(getattr(state, "pending_clarification", False))
            cur = (active, in_call, pending)
            if last != cur:
                sender.send_state(
                    active=active,
                    in_call=in_call,
                    pending_clarification=pending,
                )
                last = cur
        except Exception:
            pass
        await asyncio.sleep(0.15)


async def _run_worker(*, ui_to_worker, worker_to_ui, config: dict) -> None:
    sender = _Sender(worker_to_ui=worker_to_ui)

    # Load .env early (matches sandbox.py behavior).
    try:
        load_dotenv(override=True)
    except Exception:
        pass

    # Ensure this process writes to the sandbox log file (append mode).
    try:
        configure_sandbox_logging(
            log_in_terminal=False,
            log_file=".logs_conversation_sandbox.txt",
            log_file_mode="a",
            tcp_port=0,
            http_tcp_port=0,
            unify_requests_log_file=None,
        )
        LG.setLevel(logging.INFO)
        try:
            import os as _os

            LG.info("[runtime] started pid=%s", _os.getpid())
        except Exception:
            LG.info("[runtime] started")
    except Exception:
        pass

    args = _build_args_namespace(config=config, sender=sender)
    actor_cfg: ActorConfig = getattr(args, "_actor_config", ActorConfig())
    cfg_mgr = ConfigurationManager(
        project_name=str(getattr(args, "project_name", "unity")),
        project_root=Path(__file__).resolve().parents[2],
    )

    # agent-service bootstrap (best-effort) for real computer interface.
    repo_root = Path(__file__).resolve().parents[2]
    agent_proc = None
    try:
        if actor_cfg.actor_type == "codeact_real":
            from unity.function_manager.primitives import DEFAULT_AGENT_SERVER_URL

            agent_server_url = (
                getattr(args, "agent_server_url", None) or DEFAULT_AGENT_SERVER_URL
            )
            # Always try to free the port first (only kills repo-owned agent-service).
            try:
                free_agent_service_port(
                    repo_root=repo_root,
                    agent_server_url=agent_server_url,
                    progress=lambda m: sender.send_lines([str(m)]),
                )
            except Exception:
                pass

            do_bootstrap = bool(getattr(args, "agent_service_bootstrap", False))
            if do_bootstrap:
                res = await asyncio.to_thread(
                    try_auto_bootstrap_agent_service,
                    repo_root=repo_root,
                    agent_server_url=agent_server_url,
                    progress=lambda m: sender.send_lines([str(m)]),
                )
            else:
                res = await asyncio.to_thread(
                    try_start_agent_service_direct,
                    repo_root=repo_root,
                    agent_server_url=agent_server_url,
                    progress=lambda m: sender.send_lines([str(m)]),
                )
            if not res.ok:
                sender.send_error(res.summary)
                sender.send_worker_exit(restart=False, config=None)
                return
            agent_proc = res.process
            setattr(args, "_agent_service_process", agent_proc)
    except Exception as exc:
        sender.send_error(
            f"agent-service bootstrap failed: {type(exc).__name__}: {exc}",
            tb=_traceback.format_exc(),
        )
        sender.send_worker_exit(restart=False, config=None)
        return

    # Initialize CM + managers + actor.
    try:
        cm = await initialize_cm(
            args=args,
            progress_callback=lambda m: sender.send_lines([str(m)]),
        )
    except Exception as exc:
        sender.send_error(
            f"Failed to initialize ConversationManager: {type(exc).__name__}: {exc}",
            tb=_traceback.format_exc(),
        )
        sender.send_worker_exit(restart=False, config=None)
        return

    # Build command plumbing.
    state = WorkerSandboxState()
    publisher = EventPublisher(cm=cm, state=state)

    # Create display components for logs/traces/event-tree.
    # These are populated by subscribe_to_responses and used by save_state.
    trace_display = TraceDisplay()
    event_tree_display = EventTreeDisplay()
    log_aggregator = LogAggregator()

    # Sandbox-only: wrap CodeActActor's `execute_code` tool to stream a readable
    # trace entry over IPC (code + simplified stdout/stderr/error), and also
    # populate the local trace_display for save_state.
    try:
        actor = getattr(cm, "actor", None)
        if actor is not None and not bool(
            getattr(actor, "_sandbox_trace_wrapped", False),
        ):
            # CodeActActor registers tools under method "act".
            tools = None
            try:
                tools = actor.get_tools("act")  # type: ignore[attr-defined]
            except Exception:
                tools = None
            exec_fn = None
            if isinstance(tools, dict):
                exec_fn = tools.get("execute_code")
            if callable(exec_fn):

                async def _wrapped_execute_code(*a: Any, **kw: Any) -> Any:
                    # Best-effort extraction of code for display.
                    code = ""
                    try:
                        if "code" in kw:
                            code = str(kw.get("code") or "")
                        elif len(a) >= 2:
                            code = str(a[1] or "")
                    except Exception:
                        code = ""
                    # Drop internal/private kwargs injected by tool-loop plumbing.
                    kw2 = _filter_kwargs_for_callable(exec_fn, dict(kw))
                    res = await exec_fn(*a, **kw2)  # type: ignore[misc]
                    try:
                        simp = _simplify_execute_code_result(res)
                        # Avoid emitting noisy/empty placeholder trace entries.
                        has_code = bool(str(code or "").strip())
                        has_out = bool(str(simp.get("stdout") or "").strip()) or bool(
                            str(simp.get("stderr") or "").strip(),
                        )
                        has_err = bool(str(simp.get("error") or "").strip())
                        if has_code or has_out or has_err:
                            # Send to UI via IPC.
                            sender.send_event(
                                channel="trace:entry",
                                event={
                                    "code": code,
                                    "result": simp,
                                },
                                critical=False,
                            )
                            # Also populate local trace_display for save_state.
                            try:
                                trace_display.capture_execution(code=code, result=simp)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    return res

                # Patch the underlying tool table so the LLM uses our wrapped function.
                try:
                    actor._tools["act"]["execute_code"] = _wrapped_execute_code  # type: ignore[attr-defined]
                    setattr(actor, "_sandbox_trace_wrapped", True)
                except Exception:
                    pass
    except Exception:
        pass

    router = CommandRouter(
        cm=cm,
        args=args,
        state=state,
        publisher=publisher,
        chat_history=state.chat_history,
        allow_voice=False,
        allow_save_project=False,
        config_manager=None,
        trace_display=trace_display,
        event_tree_display=event_tree_display,
        log_aggregator=log_aggregator,
    )

    stop_event = asyncio.Event()

    # Start broker subscriber (app:comms:* and app:actor:*).
    async def _display_callback(line: str) -> None:
        sender.send_lines([str(line)])

    subscriber_task = asyncio.create_task(
        subscribe_to_responses(
            cm=cm,
            sandbox_state=state,
            display_callback=_display_callback,
            event_callback=lambda ch, ev: _forward_broker_event(
                sender=sender,
                channel=str(ch),
                event=ev,
            ),
            include_call_guidance=True,
            voice_enabled=False,
            stop_event=stop_event,
            trace_display=trace_display,
            event_tree_display=event_tree_display,
            log_aggregator=log_aggregator,
        ),
    )

    # Start EventBus stream.
    event_bus_task = asyncio.create_task(
        _subscribe_event_bus(sender=sender, stop_event=stop_event),
    )

    computer_task = asyncio.create_task(
        _computer_status_streamer(sender=sender, args=args, stop_event=stop_event),
    )

    # Start IPC loop and state broadcaster.
    ipc_task = asyncio.create_task(
        _ipc_loop(
            ui_to_worker=ui_to_worker,
            sender=sender,
            router=router,
            state=state,
            args=args,
            cfg_mgr=cfg_mgr,
            stop_event=stop_event,
        ),
    )
    state_task = asyncio.create_task(
        _state_broadcaster(sender=sender, state=state, stop_event=stop_event),
    )

    # Supervise background tasks so failures are visible in logs (and optionally UI).
    def _supervise(name: str, task: asyncio.Task) -> None:
        def _done(_t: asyncio.Task) -> None:
            try:
                if _t.cancelled():
                    return
                exc = _t.exception()
                if exc is None:
                    return
                LG.error(
                    "[runtime] background task failed: %s (%s: %s)",
                    name,
                    type(exc).__name__,
                    exc,
                )
                try:
                    sender.send_error(
                        f"Background task failed: {name} ({type(exc).__name__}: {exc})",
                        tb=_traceback.format_exc(),
                    )
                except Exception:
                    pass
            except Exception:
                return

        try:
            task.add_done_callback(_done)
        except Exception:
            return

    _supervise("subscriber_task", subscriber_task)
    _supervise("event_bus_task", event_bus_task)
    _supervise("computer_task", computer_task)
    _supervise("ipc_task", ipc_task)
    _supervise("state_task", state_task)

    # Signal readiness.
    sender.send(create_message(MessageType.READY, payload={}), critical=True)

    try:
        await stop_event.wait()
    finally:
        # Stop background tasks.
        for t in (ipc_task, subscriber_task, event_bus_task, computer_task, state_task):
            try:
                t.cancel()
            except Exception:
                pass
        await asyncio.gather(
            ipc_task,
            subscriber_task,
            event_bus_task,
            computer_task,
            state_task,
            return_exceptions=True,
        )

        # Shutdown CM best-effort.
        try:
            await shutdown_cm(cm)
        except Exception:
            pass

        # Terminate agent-service if we started it.
        try:
            if agent_proc is not None:
                _terminate_process_best_effort(agent_proc)
        except Exception:
            pass

        # Best-effort: free port after shutdown (only repo agent-service).
        try:
            if actor_cfg.actor_type == "codeact_real":
                from unity.function_manager.primitives import DEFAULT_AGENT_SERVER_URL

                free_agent_service_port(
                    repo_root=repo_root,
                    agent_server_url=getattr(args, "agent_server_url", None)
                    or DEFAULT_AGENT_SERVER_URL,
                    progress=lambda _m: None,
                )
        except Exception:
            pass


def main(ui_to_worker, worker_to_ui, config: dict):
    """Worker process entrypoint."""

    # The UI process owns the terminal. Suppress stdout/stderr in the runtime process
    # to avoid corrupting the Textual display (subprocesses inherit these by default).
    try:
        import os as _os
        import sys as _sys

        _null = open(_os.devnull, "w", encoding="utf-8")  # noqa: P201
        _sys.stdout = _null
        _sys.stderr = _null
    except Exception:
        pass

    try:
        asyncio.run(
            _run_worker(
                ui_to_worker=ui_to_worker,
                worker_to_ui=worker_to_ui,
                config=config or {},
            ),
        )
    except KeyboardInterrupt:
        return
    except Exception as exc:
        # Last-resort crash reporting: try to notify UI, then exit.
        try:
            sender = _Sender(worker_to_ui=worker_to_ui)
            sender.send_error(
                f"Worker crashed: {type(exc).__name__}: {exc}",
                tb=_traceback.format_exc(),
            )
            sender.send_worker_exit(restart=False, config=None)
        except Exception:
            pass
        raise
