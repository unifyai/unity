"""tasklist_sandbox.py
A tiny interactive CLI to *vibe-check* the TaskListManager.

• Seeds a fresh Unify project with a handful of meaningful tasks.
• Opens a REPL where you can issue natural-language *ask* or *update* commands.
  – Prefix with `ask:` or `update:` to force the method.
  – Without prefix we default to *ask*.
• Type `quit` or `exit` to leave.
"""

from __future__ import annotations

import readline  # noqa: F401 – enables command history & arrow keys
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple
import logging

# Add repo root to PYTHONPATH when run as standalone
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unify  # import after path tweak

from task_list_manager.task_list_manager import TaskListManager  # noqa: E402
from task_list_manager.types.priority import Priority  # noqa: E402
from task_list_manager.types.schedule import Schedule  # noqa: E402
from tests.test_task_list.test_update_text_complex import (
    _next_weekday,
)  # noqa: E402 – reuse helper


def _seed_scenario(tlm: TaskListManager) -> None:
    """Populate the project with a mini task list."""

    # Active
    tlm._create_task(
        name="Write quarterly report",
        description="Compile and draft the Q2 report for management.",
        status="active",
    )

    # Queued (two)
    tlm._create_task(
        name="Prepare slide deck",
        description="Create slides for the upcoming board meeting.",
        status="queued",
    )
    tlm._create_task(
        name="Client follow-up email",
        description="Send follow-up email about the proposal.",
        status="queued",
    )

    # Scheduled (next Monday)
    base = datetime.now(timezone.utc)
    next_mon = _next_weekday(base, 0).replace(hour=9, minute=0, second=0, microsecond=0)
    sched = Schedule(start_time=next_mon.isoformat(), prev_task=None, next_task=None)
    tlm._create_task(
        name="Send KPI report",
        description="Automated email of KPIs to leadership.",
        schedule=sched,
        priority=Priority.high,
    )

    # Paused
    tlm._create_task(
        name="Deploy new release",
        description="Roll out version 2.0 to production servers.",
        status="paused",
    )


def _dispatch(
    tlm: TaskListManager,
    raw: str,
    *,
    show_steps: bool,
) -> Tuple[str, str, list | None]:
    """Route user input; return (kind, answer, reasoning_steps)."""

    raw = raw.strip()
    if raw.lower().startswith("ask:"):
        ans, steps = tlm.ask(
            text=raw[4:].strip(),
            return_reasoning_steps=show_steps,
            log_tool_steps=show_steps,
        )
        return "ask", ans, steps
    if raw.lower().startswith("update:"):
        ans, steps = tlm.update(
            text=raw[7:].strip(),
            return_reasoning_steps=show_steps,
            log_tool_steps=show_steps,
        )
        return "update", ans, steps

    # Heuristic: treat questions (?) as ask
    if raw.endswith("?"):
        ans, steps = tlm.ask(
            text=raw,
            return_reasoning_steps=show_steps,
            log_tool_steps=show_steps,
        )
        return "ask", ans, steps

    ans, steps = tlm.update(
        text=raw,
        return_reasoning_steps=show_steps,
        log_tool_steps=show_steps,
    )
    return "update", ans, steps


def main() -> None:
    unify.activate("tasklist_sandbox", overwrite=True)

    tlm = TaskListManager()
    tlm.start()

    # Ensure the 'Tasks' context exists to avoid 404 errors on first queries
    unify.set_context("Tasks")

    _seed_scenario(tlm)

    # simple arg parsing
    silent = "--silent" in sys.argv or "-s" in sys.argv

    if not silent:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[logging.StreamHandler()],
        )

        # Ensure our library logger emits INFO
        from constants import LOGGER as _LG

        _LG.setLevel(logging.INFO)

        # Silence noisy logs from Unify internals
        for _name in ("unify", "unify.utils", "unify.logging"):
            logging.getLogger(_name).setLevel(logging.WARNING)

    print(
        "TaskListManager sandbox – type natural language. Prefix with 'ask:' or 'update:' to specify. 'quit' to exit.\n"
        "Verbose reasoning is {} by default (add --silent to disable).\n".format(
            "ON" if not silent else "OFF",
        ),
    )
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if line.lower() in {"quit", "exit"}:
            break
        if not line:
            continue

        kind, result, _ = _dispatch(tlm, line, show_steps=not silent)
        print(f"[{kind}] => {result}\n")


if __name__ == "__main__":
    main()
