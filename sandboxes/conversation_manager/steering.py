"""
Steering Controller + Brain Run Controller for ConversationManager sandbox.

Steering is dual-mode:
- If an active `SteerableToolHandle` exists, we use full handle steering.
- Otherwise, when a brain run is in-flight, we offer best-effort steering
  (queue-based pause/resume, interjection triggers an inbound event).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from unity.common.async_tool_loop import SteerableToolHandle

LG = logging.getLogger("conversation_manager_sandbox")


async def _maybe_await(v):
    """Await v if it is an awaitable/coroutine; otherwise return it."""
    try:
        if asyncio.iscoroutine(v):
            return await v
    except Exception:
        pass
    return v


def is_active(cm: Any, state: Any) -> bool:
    try:
        h = getattr(cm, "active_ask_handle", None)
        if h is not None and not h.done():
            return True
    except Exception:
        pass
    return bool(getattr(state, "brain_run_in_flight", False))


def get_active_handle(cm: Any) -> Optional[SteerableToolHandle]:
    try:
        h = getattr(cm, "active_ask_handle", None)
        if h is not None and not h.done():
            return h
    except Exception:
        return None
    return None


@dataclass
class BrainRunController:
    """Best-effort steering when CM brain is processing but no Actor handle exists."""

    publisher: Any
    state: Any

    async def pause(self) -> str:
        self.state.paused = True
        return "⏸️ Paused (queueing events until /resume)"

    async def resume(self) -> str:
        self.state.paused = False
        # Flush queued event commands in FIFO order
        queued = list(getattr(self.state, "queued_events", []) or [])
        try:
            self.state.queued_events.clear()
        except Exception:
            pass

        flushed = 0
        for cmd in queued:
            try:
                await _publish_from_parsed(self.publisher, cmd)
                flushed += 1
            except Exception as exc:
                LG.warning(
                    "Failed flushing queued event %s: %s",
                    getattr(cmd, "name", "?"),
                    exc,
                )
        if flushed:
            return f"▶️ Resumed (flushed {flushed} queued event(s))"
        return "▶️ Resumed"

    async def stop(self) -> str:
        # Best-effort: we can't cancel mid-generation, but we can return the UI to idle.
        self.state.brain_run_in_flight = False
        self.state.paused = False
        try:
            self.state.queued_events.clear()
        except Exception:
            pass
        return "🛑 Stopped (best-effort) — returning to idle"

    async def interject(self, message: str) -> str:
        # Best-effort: publish an inbound event that contains the interjection.
        # Use phone utterance when on call, otherwise SMS.
        if getattr(self.state, "in_call", False):
            await self.publisher.publish_phone_utterance(message)
        else:
            await self.publisher.publish_sms(message)
        return "✅ Interjection sent (triggered a fresh brain run)"

    async def ask(self, question: str) -> str:
        # Limited: no nested handle; just expose state snapshot.
        qn = question.strip()
        qn_display = qn[:120] + ("..." if len(qn) > 120 else "")
        return (
            "🧾 Brain-run status (best-effort)\n"
            f"- question: {qn_display}\n"
            f"- in_call: {bool(getattr(self.state, 'in_call', False))}\n"
            f"- paused: {bool(getattr(self.state, 'paused', False))}\n"
            f"- brain_run_in_flight: {bool(getattr(self.state, 'brain_run_in_flight', False))}\n"
            f"- queued_events: {len(getattr(self.state, 'queued_events', []) or [])}"
        )


async def _publish_from_parsed(publisher: Any, cmd: Any) -> None:
    """Helper used for flushing queued event commands in brain-run mode."""
    name = getattr(cmd, "name", "")
    args = getattr(cmd, "args", "") or ""
    kind = getattr(cmd, "kind", "")

    if kind == "utterance":
        await publisher.publish_phone_utterance(args)
        return
    if kind != "event":
        return

    if name == "sms":
        await publisher.publish_sms(args)
        return
    if name == "email":
        working = args
        if "|" in working:
            subj, body = [p.strip() for p in working.split("|", 1)]
        else:
            parts = working.split(maxsplit=1)
            subj = parts[0] if parts else "No subject"
            body = parts[1] if len(parts) > 1 else ""
        await publisher.publish_email(subj, body)
        return
    if name == "call":
        await publisher.publish_call_start()
        return
    if name == "say":
        await publisher.publish_phone_utterance(args)
        return
    if name == "end_call":
        await publisher.publish_call_end()
        return


@dataclass
class SteeringController:
    cm: Any
    state: Any
    publisher: Any
    chat_history: list[dict]
    args: Any

    def _parse(self, line: str) -> tuple[str, str]:
        """Return (cmd, arg_text). `line` should include leading '/'."""
        working = line.strip()
        if not working.startswith("/"):
            return "", ""
        body = working[1:].lstrip()
        if not body:
            return "", ""
        parts = body.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        return cmd, arg

    async def handle(self, line: str) -> str:
        """
        Handle a steering command line (starts with '/').
        Returns a user-visible message to print.
        """
        if not is_active(self.cm, self.state):
            return "(no active conversation) Steering commands only available during conversations."

        # Choose mode
        actor_handle = get_active_handle(self.cm)
        if actor_handle is not None:
            return await self._handle_actor(actor_handle, line)

        brain = BrainRunController(publisher=self.publisher, state=self.state)
        return await self._handle_brain(brain, line)

    async def _handle_actor(self, handle: SteerableToolHandle, line: str) -> str:
        cmd, arg = self._parse(line)
        if cmd in {"help"}:
            return "Controls: /i <msg>, /pause, /resume, /ask <q>, /stop [reason]"
        if cmd in {"pause", "p"}:
            await handle.pause()
            self.state.paused = True
            return "⏸️ Paused"
        if cmd in {"resume", "r"}:
            await handle.resume()
            self.state.paused = False
            return "▶️ Resumed"
        if cmd in {"i", "interject"}:
            if not arg.strip():
                return "⚠️ Usage: /i <message>"
            await _maybe_await(
                handle.interject(
                    arg,
                    parent_chat_context_cont=list(self.chat_history),
                ),
            )
            return "✅ Interjection sent"
        if cmd in {"ask", "?"}:
            if not arg.strip():
                return "⚠️ Usage: /ask <question>"
            helper = await handle.ask(
                arg,
                parent_chat_context_cont=list(self.chat_history),
            )
            try:
                answer = await helper.result()  # type: ignore[attr-defined]
            except Exception as exc:
                return f"❌ Ask failed: {exc}"
            # Some implementations return (answer, messages)
            if isinstance(answer, tuple) and answer:
                answer = answer[0]
            return f"🧾 {answer}"
        if cmd in {"stop", "cancel"}:
            reason = arg.strip() or None
            await _maybe_await(handle.stop(reason))
            self.state.paused = False
            self.state.brain_run_in_flight = False
            return "🛑 Stopped"
        if cmd == "freeform":
            # Minimal: treat freeform as interjection for now (better routing exists in await_with_interrupt).
            if not arg.strip():
                return "⚠️ Usage: /freeform <text>"
            await _maybe_await(
                handle.interject(arg, parent_chat_context_cont=list(self.chat_history)),
            )
            return "✅ Sent (freeform → interject)"
        return f"⚠️ Unknown steering command: /{cmd}. Try /help."

    async def _handle_brain(self, brain: BrainRunController, line: str) -> str:
        cmd, arg = self._parse(line)
        if cmd in {"help"}:
            return "Controls: /i <msg>, /pause, /resume, /ask <q>, /stop"
        if cmd in {"pause", "p"}:
            return await brain.pause()
        if cmd in {"resume", "r"}:
            return await brain.resume()
        if cmd in {"i", "interject"}:
            if not arg.strip():
                return "⚠️ Usage: /i <message>"
            return await brain.interject(arg)
        if cmd in {"ask", "?"}:
            if not arg.strip():
                return "⚠️ Usage: /ask <question>"
            return await brain.ask(arg)
        if cmd in {"stop", "cancel"}:
            return await brain.stop()
        if cmd == "freeform":
            if not arg.strip():
                return "⚠️ Usage: /freeform <text>"
            return await brain.interject(arg)
        return f"⚠️ Unknown steering command: /{cmd}. Try /help."
