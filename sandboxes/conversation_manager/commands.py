"""
Command parsing for the ConversationManager sandbox REPL.

This module is UI-agnostic: it only classifies a single input line into a
`ParsedCommand`. Execution is implemented in `command_router.py` so both REPL
and GUI behave identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Parsed command model
# ──────────────────────────────────────────────────────────────────────────────

CommandKind = Literal[
    # Meta
    "help",
    "quit",
    "reset",
    "save_project",
    # Scenario seeding
    "scenario_seed",
    "scenario_seed_voice",
    # Steering
    "steering",
    # Event simulation (inbound)
    "event",
    # Freeform utterance (during a call)
    "utterance",
    # Unknown / invalid
    "unknown",
]


@dataclass(frozen=True)
class ParsedCommand:
    kind: CommandKind
    raw: str
    name: str
    args: str = ""
    error: Optional[str] = None


HELP_TEXT = """
ConversationManager Sandbox (REPL)
--------------------------------
Type a command at the prompt.

Meta commands:
  help | h            Show this help
  quit | exit         Exit the sandbox
  reset               Reset sandbox session state (best-effort)
  save_project | sp   Save a Unify project snapshot

Inbound event simulation:
  sms <msg>                     Simulate incoming SMS
  email <subject> | <body>      Simulate incoming email
  call                          Start simulated phone call
  say <text>                    Phone utterance (during a call)
  sayv                          Record voice, transcribe, and send as a phone utterance (requires --voice)
  sayv <text>                   Send <text> as a phone utterance (convenience; no recording)
  end_call                      End active phone call

Scenario seeding (idle-only):
  us <description>              Generate a synthetic scenario from text
  usv                           Generate a synthetic scenario from voice (requires --voice)

Steering (only while active):
  /pause, /resume, /i <msg>, /ask <q>, /stop [reason], /help
""".strip(
    "\n",
)


def parse_command(*, text: str, in_call: bool, active: bool) -> ParsedCommand:
    """
    Parse a user input line into a structured command.

    Precedence:
    1) Meta commands
    2) Scenario commands (only when idle)
    3) Steering commands (only when active)
    4) Event commands
    5) Freeform text: utterance if in_call else error
    """
    raw = (text or "").rstrip("\n")
    trimmed = raw.strip()
    lower = trimmed.lower()

    if not trimmed:
        return ParsedCommand(kind="unknown", raw=raw, name="empty", error="empty")

    # 1) Meta commands (highest priority)
    if lower in {"help", "h", "?"}:
        return ParsedCommand(kind="help", raw=raw, name="help")
    if lower in {"quit", "exit"}:
        return ParsedCommand(kind="quit", raw=raw, name="quit")
    if lower == "reset":
        return ParsedCommand(kind="reset", raw=raw, name="reset")
    if lower in {"save_project", "sp"}:
        return ParsedCommand(kind="save_project", raw=raw, name="save_project")

    # 2) Scenario commands — only when idle
    if lower == "usv":
        if active:
            return ParsedCommand(
                kind="unknown",
                raw=raw,
                name="usv",
                error="(busy) Scenario seeding is only available when idle.",
            )
        return ParsedCommand(kind="scenario_seed_voice", raw=raw, name="usv")
    if lower.startswith("us "):
        if active:
            return ParsedCommand(
                kind="unknown",
                raw=raw,
                name="us",
                error="(busy) Scenario seeding is only available when idle.",
            )
        return ParsedCommand(
            kind="scenario_seed",
            raw=raw,
            name="us",
            args=trimmed[3:].strip(),
        )

    # 3) Steering commands — only when active
    if trimmed.startswith("/"):
        if not active:
            return ParsedCommand(
                kind="unknown",
                raw=raw,
                name="steering",
                error="(no active conversation) Steering commands only available during conversations.",
            )
        # Keep the raw slash-command intact; the router will dispatch.
        return ParsedCommand(kind="steering", raw=raw, name="steering", args=trimmed)

    # 4) Event commands
    if lower.startswith("sms "):
        return ParsedCommand(
            kind="event",
            raw=raw,
            name="sms",
            args=trimmed[4:].strip(),
        )
    if lower.startswith("email "):
        return ParsedCommand(
            kind="event",
            raw=raw,
            name="email",
            args=trimmed[6:].strip(),
        )
    if lower == "call":
        return ParsedCommand(kind="event", raw=raw, name="call")
    if lower.startswith("say "):
        return ParsedCommand(
            kind="event",
            raw=raw,
            name="say",
            args=trimmed[4:].strip(),
        )
    if lower == "sayv":
        return ParsedCommand(kind="event", raw=raw, name="sayv", args="")
    if lower.startswith("sayv "):
        return ParsedCommand(
            kind="event",
            raw=raw,
            name="sayv",
            args=trimmed[5:].strip(),
        )
    if lower == "end_call":
        return ParsedCommand(kind="event", raw=raw, name="end_call")

    # 5) Freeform text
    if in_call:
        # During a call, any non-command text is treated as a phone utterance.
        return ParsedCommand(kind="utterance", raw=raw, name="utterance", args=trimmed)

    return ParsedCommand(
        kind="unknown",
        raw=raw,
        name="unknown",
        error=f"⚠️ Unknown command: {trimmed}. Type 'help' for available commands.",
    )
