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
from sandboxes.conversation_manager.commands import HELP_TEXT
from sandboxes.conversation_manager.event_publisher import EventPublisher
from sandboxes.conversation_manager.io_gate import gated_input

LG = logging.getLogger("conversation_manager_sandbox")


@dataclass
class SandboxState:
    chat_history: list[dict] = field(default_factory=list)
    in_call: bool = False
    in_meet: bool = False
    last_event_published_at: float = 0.0
    live_voice_session: object = None

    @property
    def live_voice_active(self) -> bool:
        return self.live_voice_session is not None

    @property
    def in_voice_session(self) -> bool:
        return self.in_call or self.in_meet

    def reset_ephemeral(self) -> None:
        """Reset sandbox-local state (CM state reset is handled separately)."""
        self.chat_history.clear()
        self.in_call = False
        self.in_meet = False
        self.last_event_published_at = 0.0
        self.live_voice_session = None


def get_prompt(state: SandboxState) -> str:
    return "> "


def _print_welcome(*, args: Any) -> None:
    mode = "REAL-COMMS" if getattr(args, "real_comms", False) else "SIMULATED"
    gui = "on" if getattr(args, "gui", False) else "off"
    voice = "on" if getattr(args, "voice", False) else "off"
    live_voice = "on" if getattr(args, "live_voice", False) else "off"
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
    if live_voice == "on":
        print(f"Live Voice: {live_voice} (calls use real LiveKit voice agent)")
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
            args=args,
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
        setattr(args, "_router", router)

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
            if res.should_exit:
                return
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return
        except Exception as exc:
            LG.error("REPL error: %s", exc, exc_info=True)
            print(f"❌ Error: {exc}")
            continue
