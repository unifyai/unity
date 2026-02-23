"""
Shared command execution for the ConversationManager sandbox (REPL + GUI).

The parser (`commands.parse_command`) is intentionally UI-agnostic; this module
implements the shared execution semantics so both the REPL and Textual GUI route
commands identically.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

import unify

from sandboxes.conversation_manager.commands import (
    HELP_TEXT,
    ParsedCommand,
    parse_command,
)
from sandboxes.conversation_manager.event_publisher import EventPublisher
from unity.conversation_manager.events import (
    AssistantScreenShareStarted,
    AssistantScreenShareStopped,
    UserRemoteControlStarted,
    UserRemoteControlStopped,
    UserScreenShareStarted,
    UserScreenShareStopped,
    UserWebcamStarted,
    UserWebcamStopped,
)
from sandboxes.conversation_manager.io_gate import gated_input
from sandboxes.conversation_manager.scenario_generator import ScenarioGenerator
from sandboxes.conversation_manager.steering import SteeringController, is_active
from sandboxes.conversation_manager.config_manager import (
    ConfigurationManager,
    ActorConfig,
)
from sandboxes.conversation_manager.trace_display import TraceDisplay
from sandboxes.conversation_manager.event_tree_display import EventTreeDisplay
from sandboxes.conversation_manager.log_aggregator import LogAggregator
from sandboxes.conversation_manager.agent_service_bootstrap import (
    get_agent_service_log_path,
)
from sandboxes.conversation_manager.state_snapshot import (
    capture_snapshot,
    save_snapshot,
    render_snapshot_text,
)

LG = logging.getLogger("conversation_manager_sandbox")

_MEET_INTERACTION_EVENTS: dict[str, type] = {
    "assistant_screen_share_start": AssistantScreenShareStarted,
    "assistant_screen_share_stop": AssistantScreenShareStopped,
    "user_screen_share_start": UserScreenShareStarted,
    "user_screen_share_stop": UserScreenShareStopped,
    "user_webcam_start": UserWebcamStarted,
    "user_webcam_stop": UserWebcamStopped,
    "user_remote_control_start": UserRemoteControlStarted,
    "user_remote_control_stop": UserRemoteControlStopped,
}

PromptFn = Callable[[str], Awaitable[str]]


@dataclass
class RouterResult:
    lines: list[str]
    should_exit: bool = False


@dataclass
class CommandRouter:
    """
    Executes a single input line against a running CM sandbox session.

    The router is stateful only via the provided `state` and `chat_history`
    references; it does not persist internal state.
    """

    cm: Any
    args: Any
    state: Any
    publisher: EventPublisher
    chat_history: list[dict]
    allow_voice: bool = True
    allow_save_project: bool = True
    config_manager: ConfigurationManager | None = None
    trace_display: TraceDisplay | None = None
    event_tree_display: EventTreeDisplay | None = None
    log_aggregator: LogAggregator | None = None
    conversation_lines: list[str] = field(default_factory=list)

    async def execute_raw(
        self,
        raw: str,
        *,
        prompt_text: Optional[PromptFn] = None,
        in_call: Optional[bool] = None,
    ) -> RouterResult:
        """
        Execute a single raw line.

        prompt_text:
          Optional async callback used to request additional user input
          (e.g., missing scenario description) in REPL mode. GUI callers
          typically omit it.
        """
        st = self.state
        in_call_now = (
            bool(getattr(st, "in_call", False)) if in_call is None else bool(in_call)
        )

        active_now = is_active(self.cm, st)
        cmd: ParsedCommand = parse_command(
            text=raw,
            in_call=in_call_now,
            active=active_now,
        )

        # Keep a minimal chat history for steering context.
        if cmd.kind not in {"unknown", "help"} and (raw or "").strip():
            try:
                self.chat_history.append({"role": "user", "content": raw.strip()})
            except Exception:
                pass

        # Unknown / error
        if cmd.kind == "unknown":
            if cmd.error and cmd.error != "empty":
                return RouterResult(lines=[cmd.error])
            return RouterResult(lines=[])

        # Meta
        if cmd.kind == "help":
            return RouterResult(lines=["\n" + HELP_TEXT + "\n"])
        if cmd.kind == "quit":
            return RouterResult(lines=["Exiting..."], should_exit=True)
        if cmd.kind == "reset":
            await self._reset_best_effort()
            # Reset display state.
            try:
                if self.trace_display is not None:
                    self.trace_display.reset_history()
            except Exception:
                pass
            try:
                if self.event_tree_display is not None:
                    self.event_tree_display.reset_tree()
            except Exception:
                pass
            try:
                if self.log_aggregator is not None:
                    self.log_aggregator.reset_expansion()
            except Exception:
                pass
            return RouterResult(lines=["✅ Reset complete."])
        if cmd.kind == "save_project":
            if not self.allow_save_project:
                return RouterResult(
                    lines=["⚠️ save_project is not available in this mode."],
                )
            try:
                commit_hash = unify.commit_project(
                    self.args.project_name,
                    commit_message=f"ConversationManager sandbox save {datetime.utcnow().isoformat()}",
                ).get("commit_hash")
                return RouterResult(lines=[f"💾 Project saved at commit {commit_hash}"])
            except Exception as exc:
                LG.error("save_project failed: %s", exc, exc_info=True)
                return RouterResult(lines=[f"❌ Failed to save project: {exc}"])
        if cmd.kind == "save_state":
            return await self._handle_save_state(cmd.args)

        # Configuration
        if cmd.kind == "config":
            return await self._handle_config_switch(prompt_text=prompt_text)

        # Display
        if cmd.kind == "trace":
            return await self._handle_trace_display(cmd.args)
        if cmd.kind == "tree":
            return await self._handle_tree_display()
        if cmd.kind == "show_logs":
            return await self._handle_log_expansion(cmd.args, expand=True)
        if cmd.kind == "collapse_logs":
            return await self._handle_log_expansion(cmd.args, expand=False)
        if cmd.kind == "agent_logs":
            return await self._handle_agent_logs(cmd.args)

        # Scenario seeding
        if cmd.kind in {"scenario_seed", "scenario_seed_voice"}:
            return await self._handle_scenario(cmd, prompt_text=prompt_text)

        # Event / utterance
        if cmd.kind in {"event", "utterance"}:
            return await self._handle_event(cmd)

        # Steering
        if cmd.kind == "steering":
            ctrl = SteeringController(
                cm=self.cm,
                state=st,
                publisher=self.publisher,
                chat_history=self.chat_history,
                args=self.args,
            )
            out = await ctrl.handle(cmd.args)
            return RouterResult(lines=[out] if out else [])

        return RouterResult(lines=[f"⚠️ Unhandled command kind: {cmd.kind}"])

    async def _handle_trace_display(self, args: str) -> RouterResult:
        td = self.trace_display
        cfg = getattr(self.args, "_actor_config", None)
        if td is None:
            return RouterResult(lines=["⚠️ Trace display is not initialized."])
        if getattr(cfg, "actor_type", "simulated") == "simulated":
            return RouterResult(
                lines=[
                    "⚠️ Trace display only available for CodeActActor configurations.",
                ],
            )
        n = 3
        try:
            if (args or "").strip():
                n = int((args or "").strip())
        except Exception:
            n = 3
        return RouterResult(lines=[td.render_recent(n)])

    async def _handle_tree_display(self) -> RouterResult:
        tree = self.event_tree_display
        if tree is None:
            return RouterResult(lines=["⚠️ Event tree display is not initialized."])
        return RouterResult(lines=[tree.render_tree()])

    async def _handle_save_state(self, args: str) -> RouterResult:
        """Save structured state snapshot to a file."""
        # Capture the snapshot
        snapshot = capture_snapshot(
            log_aggregator=self.log_aggregator,
            event_tree_display=self.event_tree_display,
            trace_display=self.trace_display,
            conversation_lines=self.conversation_lines,
        )

        repo_root = Path(__file__).resolve().parents[2]

        # Determine output path
        if args and args.strip():
            json_path = repo_root / args.strip()
        else:
            # Auto-generate filename with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            json_path = repo_root / f".sandbox_state_{timestamp}.json"

        # Save JSON
        try:
            save_snapshot(snapshot, json_path)
        except Exception as exc:
            LG.error("save_state failed: %s", exc, exc_info=True)
            return RouterResult(lines=[f"❌ Failed to save state: {exc}"])

        # Also save human-readable text version
        try:
            text_path = json_path.with_suffix(".txt")
            text_content = render_snapshot_text(snapshot)
            with open(text_path, "w") as f:
                f.write(text_content)
        except Exception as exc:
            LG.warning("Failed to save text snapshot: %s", exc)
            return RouterResult(
                lines=[
                    f"💾 State saved to: {json_path}",
                    f"⚠️ Failed to save text version: {exc}",
                ],
            )

        result_lines = [
            f"💾 State saved:",
            f"   JSON: {json_path}",
            f"   Text: {text_path}",
            f"   Summary: {snapshot.summary.get('total_conversation_lines', 0)} conversation lines, "
            f"{snapshot.summary['total_cm_logs']} CM logs, "
            f"{snapshot.summary['total_actor_logs']} actor logs, "
            f"{snapshot.summary['total_manager_logs']} manager logs, "
            f"{snapshot.summary['total_traces']} traces, "
            f"{snapshot.summary['total_event_trees']} trees",
        ]

        # Generate call transcript from the voice agent log if available.
        import os

        _launch_cwd = os.environ.get("UNITY_SANDBOX_LAUNCH_CWD", "").strip()
        _voice_root = Path(_launch_cwd).resolve() if _launch_cwd else repo_root
        voice_log = _voice_root / ".logs_voice_agent.txt"
        if voice_log.exists():
            try:
                from sandboxes.conversation_manager.call_transcript import (
                    build_timeline,
                    format_timeline,
                    parse_voice_log,
                )

                voice_data = parse_voice_log(voice_log)
                if voice_data.utterances:
                    timeline = build_timeline(voice_data)
                    transcript_path = json_path.with_name(
                        json_path.stem + "_transcript.txt",
                    )
                    with open(transcript_path, "w") as f:
                        f.write(format_timeline(timeline, verbose=True))
                    result_lines.append(f"   Transcript: {transcript_path}")
            except Exception as exc:
                LG.warning("Failed to generate call transcript: %s", exc)

        return RouterResult(lines=result_lines)

    async def _handle_log_expansion(self, args: str, *, expand: bool) -> RouterResult:
        lg = self.log_aggregator
        if lg is None:
            return RouterResult(lines=["⚠️ Log aggregator is not initialized."])

        raw = (args or "").strip().lower()
        cats = []
        if raw in {"cm", "actor", "manager"}:
            cats = [raw]
        elif raw == "all":
            cats = ["cm", "actor", "manager"]
        else:
            return RouterResult(
                lines=[
                    "⚠️ Usage: show_logs <cm|actor|manager|all>  or  collapse_logs <cm|actor|manager|all>",
                ],
            )

        if expand:
            for c in cats:
                lg.expand(c)  # type: ignore[arg-type]
            blocks = []
            for c in cats:
                blocks.append(lg.render_expanded(c))  # type: ignore[arg-type]
            return RouterResult(lines=[("\n\n".join(blocks)).rstrip()])

        for c in cats:
            lg.collapse(c)  # type: ignore[arg-type]
        return RouterResult(lines=[lg.render_summary()])

    def _tail_text_file(self, path: Path, *, max_lines: int = 80) -> str:
        """
        Return the last N lines of a text file, best-effort.

        This is a sandbox UX helper; we keep it safe and bounded.
        """
        max_lines = int(max_lines)
        if max_lines <= 0:
            max_lines = 80
        if max_lines > 400:
            max_lines = 400
        try:
            if not path.exists():
                return ""
        except Exception:
            return ""
        try:
            # Bounded approach: keep only the last N lines without loading everything into memory.
            dq: deque[str] = deque(maxlen=max_lines)
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    dq.append(line.rstrip("\n"))
            return "\n".join(dq).rstrip()
        except Exception:
            return ""

    async def _handle_agent_logs(self, args: str) -> RouterResult:
        n = 80
        try:
            if (args or "").strip():
                n = int((args or "").strip())
        except Exception:
            n = 80

        # Determine repo root (same strategy as the sandbox entrypoint).
        repo_root = Path(__file__).resolve().parents[2]
        agent_server_url = getattr(
            self.args,
            "agent_server_url",
            "http://localhost:3000",
        )
        log_path = None
        try:
            lp = getattr(self.args, "_agent_service_log_path", None)
            if isinstance(lp, str) and lp.strip():
                log_path = Path(lp.strip())
        except Exception:
            log_path = None
        if log_path is None:
            log_path = get_agent_service_log_path(
                repo_root=repo_root,
                agent_server_url=str(agent_server_url),
            )

        tail = self._tail_text_file(log_path, max_lines=n)
        if not tail:
            return RouterResult(
                lines=[
                    f"agent-service log: {log_path}",
                    "⚠️ No log output yet (or file does not exist).",
                ],
            )
        return RouterResult(
            lines=[
                f"agent-service log: {log_path}",
                "",
                tail,
            ],
        )

    async def _handle_config_switch(
        self,
        *,
        prompt_text: Optional[PromptFn],
    ) -> RouterResult:
        if prompt_text is None:
            return RouterResult(
                lines=["⚠️ Config switching is only available in REPL mode."],
            )
        cfg_mgr = self.config_manager
        if cfg_mgr is None:
            return RouterResult(lines=["⚠️ Configuration manager is not initialized."])

        warn = "\n".join(
            [
                "⚠️  Switching configuration will:",
                "- Restart ConversationManager",
                "- Clear all conversation state (threads, notifications, in-flight actions)",
                "- Auto-snapshot the project before switching (rollback is possible)",
                "",
            ],
        )
        ans = (await prompt_text(warn + "Continue? (y/N): ")).strip().lower()
        if ans not in {"y", "yes"}:
            return RouterResult(lines=["(cancelled)"])

        # Snapshot first (best-effort, can take a moment).
        try:
            snap = await asyncio.to_thread(cfg_mgr.snapshot_state)
            setattr(self.args, "_last_config_snapshot", snap)
        except Exception as exc:
            return RouterResult(lines=[f"❌ Failed to snapshot project: {exc}"])

        # Prompt for config choice (reuse same menu as startup, but inline).
        last_used = cfg_mgr.load_config()

        def _prompt_choice() -> ActorConfig:
            print("Select Actor Configuration:")
            print(
                "1. SandboxSimulatedActor (simulated managers, no computer interface)",
            )
            print("2. CodeActActor + Simulated Managers (mock computer backend)")
            print("3. CodeActActor + Real Managers + Real Computer Interface")
            raw = input("Enter choice (1-3) or press Enter for last used: ").strip()
            if not raw:
                return last_used
            m = {"1": "simulated", "2": "codeact_simulated", "3": "codeact_real"}
            if raw in m:
                return ActorConfig(actor_type=m[raw])  # type: ignore[arg-type]
            return last_used

        new_cfg = await asyncio.to_thread(_prompt_choice)

        # Validate with retry/switch/exit (switch returns to prompt).
        while True:
            vr = await asyncio.to_thread(
                cfg_mgr.validate_config,
                new_cfg,
                agent_server_url=getattr(
                    self.args,
                    "agent_server_url",
                    "http://localhost:3000",
                ),
            )
            if vr.ok:
                break
            msg = "\n".join(
                [
                    "❌ Configuration Error",
                    "",
                    f"Failed to initialize: {vr.failed_component or 'Unknown'}",
                    f"Reason: {vr.error or 'Unknown'}",
                    "",
                    "Options:",
                    "1. Retry (after fixing infrastructure)",
                    "2. Switch to different configuration",
                    "3. Exit sandbox",
                    "",
                ],
            )
            choice = (await prompt_text(msg + "Enter choice (1-3): ")).strip()
            if choice == "1":
                continue
            if choice == "2":
                new_cfg = await asyncio.to_thread(_prompt_choice)
                continue
            return RouterResult(lines=["Exiting..."], should_exit=True)

        cfg_mgr.save_config(new_cfg)
        # Signal to outer sandbox loop that a restart is requested.
        setattr(self.args, "_restart_requested", True)
        setattr(self.args, "_restart_actor_config", new_cfg)
        return RouterResult(
            lines=["🔄 Restarting sandbox with selected configuration..."],
            should_exit=True,
        )

    async def _reset_best_effort(self) -> None:
        st = self.state

        # Clean up live voice session if active.
        if getattr(st, "live_voice_active", False):
            try:
                await self.publisher.end_live_call()
            except Exception:
                pass

        try:
            st.reset_ephemeral()
        except Exception:
            pass

        # Clear CM state best-effort.
        cm = self.cm
        try:
            cm.contact_index.clear_conversations()
        except Exception:
            pass
        try:
            cm.notifications_bar.notifications = []
        except Exception:
            pass
        try:
            cm.in_flight_actions.clear()
        except Exception:
            pass
        try:
            cm.chat_history.clear()
        except Exception:
            pass
        try:
            from unity.conversation_manager.types import Mode

            cm.mode = Mode.TEXT
        except Exception:
            pass
        try:
            cm.call_manager.call_contact = None
        except Exception:
            pass

    async def _handle_event(self, cmd: ParsedCommand) -> RouterResult:
        st = self.state
        st.last_event_published_at = asyncio.get_running_loop().time()

        # Queue events while paused.
        if getattr(st, "paused", False):
            try:
                st.queued_events.append(cmd)
            except Exception:
                pass
            return RouterResult(lines=[f"⏸️ Queued while paused: {cmd.name}"])

        if cmd.kind == "utterance":
            await self.publisher.publish_phone_utterance(cmd.args)
            return RouterResult(lines=[])

        if cmd.name == "sms":
            await self.publisher.publish_sms(cmd.args)
            return RouterResult(lines=[])
        if cmd.name == "email":
            working = cmd.args
            if "|" in working:
                subj, body = [p.strip() for p in working.split("|", 1)]
            else:
                parts = working.split(maxsplit=1)
                subj = parts[0] if parts else "No subject"
                body = parts[1] if len(parts) > 1 else ""
            await self.publisher.publish_email(subj, body)
            return RouterResult(lines=[])
        if cmd.name == "call":
            # Live voice mode: spawn real LiveKit voice agent
            if getattr(self.args, "live_voice", False):
                if getattr(st, "live_voice_active", False):
                    return RouterResult(
                        lines=["⚠️ A live voice call is already active."],
                    )
                try:
                    lines = await self.publisher.start_live_call()
                    return RouterResult(lines=lines)
                except Exception as exc:
                    st.in_call = False
                    return RouterResult(
                        lines=[f"❌ Failed to start live voice call: {exc}"],
                    )
            await self.publisher.publish_call_start()
            return RouterResult(
                lines=[
                    "📞 Call started.",
                    "Tip: use `say <text>` to speak, and `end_call` to finish.",
                ],
            )
        if cmd.name == "say":
            if not getattr(st, "in_call", False):
                return RouterResult(lines=["⚠️ No active call. Use `call` first."])
            if getattr(st, "live_voice_active", False):
                return RouterResult(
                    lines=[
                        "🎙️  Live voice is active — speak through your browser mic.",
                        "   The voice agent handles speech-to-text automatically.",
                    ],
                )
            await self.publisher.publish_phone_utterance(cmd.args)
            return RouterResult(lines=[])
        if cmd.name == "sayv":
            if not getattr(st, "in_call", False):
                return RouterResult(lines=["⚠️ No active call. Use `call` first."])
            if getattr(st, "live_voice_active", False):
                return RouterResult(
                    lines=[
                        "🎙️  Live voice is active — speak through your browser mic.",
                        "   The voice agent handles speech-to-text automatically.",
                    ],
                )
            if not self.allow_voice:
                return RouterResult(
                    lines=["⚠️ Voice input is not enabled in this mode."],
                )
            if not getattr(self.args, "voice", False):
                return RouterResult(
                    lines=["⚠️ Restart with `--voice` to enable recording."],
                )

            # Convenience: allow `sayv <text>` without recording.
            text = (cmd.args or "").strip()
            if not text:
                try:
                    from sandboxes.utils import (
                        record_for_seconds,
                        record_until_enter,
                        transcribe_deepgram,
                        transcribe_deepgram_no_input,
                    )
                except Exception as exc:
                    return RouterResult(lines=[f"⚠️ Voice mode unavailable ({exc})."])
                try:
                    # GUI callers pass prompt_text=None; avoid stdin-driven recording there.
                    if bool(getattr(self.args, "gui", False)):
                        audio = await asyncio.to_thread(record_for_seconds, 6.0)
                        text = (
                            await asyncio.to_thread(transcribe_deepgram_no_input, audio)
                            or ""
                        ).strip()
                    else:
                        audio = await asyncio.to_thread(record_until_enter)
                        text = (
                            await asyncio.to_thread(transcribe_deepgram, audio) or ""
                        ).strip()
                except Exception as exc:
                    return RouterResult(lines=[f"❌ Voice transcription failed: {exc}"])
                if not text:
                    return RouterResult(
                        lines=["⚠️ Transcription was empty. Please try again."],
                    )

            await self.publisher.publish_phone_utterance(text)
            return RouterResult(lines=[f"▶️ {text}"])
        if cmd.name == "end_call":
            # Live voice mode: clean up LiveKit room + subprocess
            if getattr(st, "live_voice_active", False):
                try:
                    lines = await self.publisher.end_live_call()
                    return RouterResult(lines=lines)
                except Exception as exc:
                    LG.error("end_live_call failed: %s", exc, exc_info=True)
                    st.in_call = False
                    st.live_voice_session = None
                    return RouterResult(
                        lines=[f"❌ Error ending live voice call: {exc}"],
                    )
            await self.publisher.publish_call_end()
            return RouterResult(lines=["📞 Call ended."])

        meet_event_cls = _MEET_INTERACTION_EVENTS.get(cmd.name)
        if meet_event_cls is not None:
            await self.publisher.publish_meet_interaction_event(
                meet_event_cls,
                cmd.args,
            )
            return RouterResult(lines=[])

        return RouterResult(lines=[f"⚠️ Unknown event command: {cmd.name}"])

    async def _handle_scenario(
        self,
        cmd: ParsedCommand,
        *,
        prompt_text: Optional[PromptFn],
    ) -> RouterResult:
        # Idle guard: this is also enforced in parse_command, but re-check for safety.
        if is_active(self.cm, self.state):
            return RouterResult(
                lines=[
                    "⚠️ Scenario seeding is disabled while a conversation/action is active.",
                    "   Use /stop or wait for completion.",
                ],
            )

        if cmd.kind == "scenario_seed_voice":
            if not self.allow_voice or not getattr(self.args, "voice", False):
                return RouterResult(
                    lines=[
                        "⚠️ Voice mode not enabled – restart with --voice or use 'us <description>' instead.",
                    ],
                )
            try:
                from sandboxes.utils import record_until_enter, transcribe_deepgram
            except Exception as exc:
                return RouterResult(
                    lines=[
                        f"⚠️ Voice mode unavailable ({exc}). Use 'us <description>' instead.",
                    ],
                )
            try:
                audio = record_until_enter()
                description = transcribe_deepgram(audio)
            except Exception as exc:
                return RouterResult(lines=[f"❌ Voice transcription failed: {exc}"])
            if not description or not description.strip():
                return RouterResult(
                    lines=["⚠️ Transcription was empty – please try again."],
                )
            desc = description.strip()
            lines = [f"▶️ {desc}"]
        else:
            desc = (cmd.args or "").strip()
            lines = []
            if not desc:
                if prompt_text is None:
                    return RouterResult(lines=["⚠️ Usage: us <description>"])
                desc = (await prompt_text("🧮 Describe scenario > ")).strip()
                if not desc:
                    return RouterResult(
                        lines=["⚠️ No description provided – cancelled."],
                    )

        gen = ScenarioGenerator(publisher=self.publisher, state=self.state)
        try:
            await gen.generate_and_publish(desc)
            return RouterResult(lines=lines)
        except Exception as exc:
            LG.error("Scenario generation failed: %s", exc, exc_info=True)
            return RouterResult(
                lines=lines + [f"❌ Failed to generate scenario: {exc}"],
            )


async def repl_prompt(prompt: str) -> str:
    """Default REPL prompt function (stdin-gated)."""
    return await asyncio.to_thread(gated_input, prompt)
