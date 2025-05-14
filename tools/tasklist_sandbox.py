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


def _dispatch(tlm: TaskListManager, raw: str) -> Tuple[str, str]:
    """Route the user input to the appropriate method and return (kind, result)."""

    raw = raw.strip()
    if raw.lower().startswith("ask:"):
        res = tlm.ask(text=raw[4:].strip())
        return "ask", res
    if raw.lower().startswith("update:"):
        res = tlm.update(text=raw[7:].strip())
        return "update", res

    # Heuristic: treat questions (ending with ?) as ask; everything else update
    if raw.endswith("?"):
        res = tlm.ask(text=raw)
        return "ask", res

    res = tlm.update(text=raw)
    return "update", res


def main() -> None:
    if "tasklist_sandbox" in unify.list_projects():
        unify.delete_project("tasklist_sandbox")
    unify.activate("tasklist_sandbox")

    tlm = TaskListManager()
    tlm.start()

    # Ensure the 'Tasks' context exists to avoid 404 on first get_logs
    try:
        unify.create_context("Tasks", description="TaskListManager sandbox context")
    except Exception:
        # ignore if it already exists or API not available
        pass

    _seed_scenario(tlm)

    print(
        "TaskListManager sandbox – type natural language. Prefix with 'ask:' or 'update:' to specify. 'quit' to exit.\n",
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

        kind, result = _dispatch(tlm, line)
        print(f"[{kind}] => {result}\n")


if __name__ == "__main__":
    main()
