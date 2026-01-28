"""
Shared command execution for the ConversationManager sandbox (REPL + GUI).

The parser (`commands.parse_command`) is intentionally UI-agnostic; this module
implements the shared execution semantics so both the REPL and Textual GUI route
commands identically.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

import unify

from sandboxes.conversation_manager.commands import (
    HELP_TEXT,
    ParsedCommand,
    parse_command,
)
from sandboxes.conversation_manager.event_publisher import EventPublisher
from sandboxes.conversation_manager.io_gate import gated_input
from sandboxes.conversation_manager.scenario_generator import ScenarioGenerator
from sandboxes.conversation_manager.steering import SteeringController, is_active

LG = logging.getLogger("conversation_manager_sandbox")

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
            return RouterResult(lines=["Exiting…"], should_exit=True)
        if cmd.kind == "reset":
            await self._reset_best_effort()
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

    async def _reset_best_effort(self) -> None:
        st = self.state
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
            await self.publisher.publish_call_start()
            return RouterResult(lines=[])
        if cmd.name == "say":
            if not getattr(st, "in_call", False):
                return RouterResult(lines=["⚠️ No active call. Use `call` first."])
            await self.publisher.publish_phone_utterance(cmd.args)
            return RouterResult(lines=[])
        if cmd.name == "sayv":
            if not getattr(st, "in_call", False):
                return RouterResult(lines=["⚠️ No active call. Use `call` first."])
            if not self.allow_voice:
                return RouterResult(lines=["⚠️ Voice input is available in REPL mode."])
            if not getattr(self.args, "voice", False):
                return RouterResult(
                    lines=["⚠️ Restart with `--voice` to enable recording."],
                )

            # Convenience: allow `sayv <text>` without recording.
            text = (cmd.args or "").strip()
            if not text:
                try:
                    from sandboxes.utils import record_until_enter, transcribe_deepgram
                except Exception as exc:
                    return RouterResult(lines=[f"⚠️ Voice mode unavailable ({exc})."])
                try:
                    audio = record_until_enter()
                    text = (transcribe_deepgram(audio) or "").strip()
                except Exception as exc:
                    return RouterResult(lines=[f"❌ Voice transcription failed: {exc}"])
                if not text:
                    return RouterResult(
                        lines=["⚠️ Transcription was empty – please try again."],
                    )

            await self.publisher.publish_phone_utterance(text)
            return RouterResult(lines=[f"▶️ {text}"])
        if cmd.name == "end_call":
            await self.publisher.publish_call_end()
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
