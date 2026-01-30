"""
REPL loop for the ConversationManager sandbox.

The REPL is a thin UI layer: it renders prompts and delegates command execution
to `CommandRouter` so REPL and GUI share the same semantics.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from sandboxes.conversation_manager.command_router import CommandRouter, repl_prompt
from sandboxes.conversation_manager.commands import HELP_TEXT, ParsedCommand
from sandboxes.conversation_manager.event_publisher import EventPublisher
from sandboxes.conversation_manager.io_gate import gated_input
from sandboxes.utils import steering_controls_hint
from sandboxes.conversation_manager.steering import is_active

LG = logging.getLogger("conversation_manager_sandbox")


@dataclass
class SandboxState:
    chat_history: list[dict] = field(default_factory=list)
    in_call: bool = False
    brain_run_in_flight: bool = False
    paused: bool = False
    last_event_published_at: float = 0.0
    queued_events: list[ParsedCommand] = field(default_factory=list)
    _steering_hint_visible: bool = False

    def reset_ephemeral(self) -> None:
        """Reset sandbox-local state (CM state reset is handled separately)."""
        self.chat_history.clear()
        self.in_call = False
        self.brain_run_in_flight = False
        self.paused = False
        self.last_event_published_at = 0.0
        self.queued_events.clear()
        self._steering_hint_visible = False


def get_prompt(state: SandboxState) -> str:
    if state.paused:
        return "paused> "
    if state.in_call:
        return "call> "
    return "cm> "


def _print_welcome(*, args: Any) -> None:
    mode = "REAL-COMMS" if getattr(args, "real_comms", False) else "SIMULATED"
    gui = "on" if getattr(args, "gui", False) else "off"
    voice = "on" if getattr(args, "voice", False) else "off"
    cfg = getattr(args, "_actor_config", None)
    actor_type = getattr(cfg, "actor_type", None) if cfg is not None else None

    print("\n" + "═" * 72)
    print("ConversationManager Sandbox")
    print("═" * 72)
    print(f"Mode: {mode}")
    if actor_type:
        print(f"ActorConfig: {actor_type}")
    print(f"GUI:  {gui}")
    print(f"Voice:{voice}")
    print("\n" + HELP_TEXT + "\n")


async def run_repl(*, args: Any, state: SandboxState | None = None) -> None:
    """
    Run the sandbox REPL.
    """
    st = state or SandboxState()
    _print_welcome(args=args)

    # CM instance is injected by sandbox entrypoint.
    cm = getattr(args, "_cm", None)
    publisher: EventPublisher | None = None
    if cm is not None:
        publisher = EventPublisher(
            cm=cm,
            state=st,
        )

    router: CommandRouter | None = None
    if cm is not None and publisher is not None:
        router = CommandRouter(
            cm=cm,
            args=args,
            state=st,
            publisher=publisher,
            chat_history=st.chat_history,
            allow_voice=True,
            allow_save_project=True,
            config_manager=getattr(args, "_config_manager", None),
            trace_display=getattr(args, "_trace_display", None),
            event_tree_display=getattr(args, "_event_tree_display", None),
            log_aggregator=getattr(args, "_log_aggregator", None),
        )

    while True:
        try:
            raw = await asyncio.to_thread(gated_input, get_prompt(st))
            if router is None:
                print("❌ ConversationManager is not initialized (unexpected).")
                continue

            res = await router.execute_raw(
                raw,
                prompt_text=repl_prompt,
                in_call=st.in_call,
            )
            for ln in res.lines:
                print(ln)
            # Steering hint: print when we transition into an active state.
            try:
                active_now = is_active(cm, st) if cm is not None else False
                if active_now and not st._steering_hint_visible:
                    print(
                        steering_controls_hint(
                            voice_enabled=bool(getattr(args, "voice", False)),
                        ),
                    )
                    st._steering_hint_visible = True
                if (not active_now) and st._steering_hint_visible:
                    st._steering_hint_visible = False
            except Exception:
                pass
            if res.should_exit:
                return
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return
        except Exception as exc:
            LG.error("REPL error: %s", exc, exc_info=True)
            print(f"❌ Error: {exc}")
            continue
