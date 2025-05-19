"""tasklist_sandbox.py  (voice mode, Deepgram SDK v4, sync)
===========================================================
Interactive sandbox for **TaskListManager** with optional hands‑free
voice input. All shared audio/STT/TTS helpers are imported from
`utils.py` to avoid duplication with other sandboxes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unify

from unity.constants import LOGGER as _LG  # type: ignore
from unity.task_list_manager.task_list_manager import TaskListManager  # type: ignore
from unity.task_list_manager.types.priority import Priority  # type: ignore
from unity.task_list_manager.types.schedule import Schedule  # type: ignore
from tests.test_task_list.test_update_text_complex import _next_weekday  # type: ignore
from sandboxes.utils import (
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
)  # type: ignore


# ---------------------------------------------------------------------------
# Scenario seeding helpers (fixed + LLM)
# ---------------------------------------------------------------------------


def _seed_fixed(tlm: TaskListManager) -> None:
    """Populate a small but varied set of starter tasks."""
    tlm._create_task(
        name="Write quarterly report",
        description="Compile and draft the Q2 report for management.",
        status="active",
    )
    tlm._create_task(
        name="Prepare slide deck",
        description="Create slides for the upcoming board meeting.",
        status="queued",
    )
    tlm._create_task(
        name="Client follow‑up email",
        description="Send follow‑up email about the proposal.",
        status="queued",
    )

    base = datetime.now(timezone.utc)
    next_mon = _next_weekday(base, 0).replace(hour=9, minute=0, second=0, microsecond=0)
    tlm._create_task(
        name="Send KPI report",
        description="Automated email of KPIs to leadership.",
        schedule=Schedule(
            start_time=next_mon.isoformat(),
            prev_task=None,
            next_task=None,
        ),
        priority=Priority.high,
    )
    tlm._create_task(
        name="Deploy new release",
        description="Roll out version 2.0 to production servers.",
        status="paused",
    )


def _seed_llm(tlm: TaskListManager) -> Optional[str]:
    """Generate a large realistic task backlog via LLM."""
    prompt = (
        "Generate a realistic task list for a small business. Pick a coherent theme. "
        "Create 110‑140 tasks across queues with positions, priorities & ISO start times. "
        "Return only JSON with top‑level 'tasks' and optional 'theme'."
    )
    client = unify.Unify("o4-mini@openai", cache=True)
    client.set_system_message(prompt)
    raw = client.generate("Produce scenario").strip()

    try:
        payload = json.loads(raw)
    except Exception:
        _LG.warning("LLM scenario failed – using fixed seed.")
        _seed_fixed(tlm)
        return None

    theme = payload.get("theme")
    tasks = payload["tasks"]

    # Group by queue_group, sort by queue_position for stable insertion order
    groups: Dict[str, List[dict]] = {}
    for t in tasks:
        groups.setdefault(t.get("queue_group", "default"), []).append(t)
    for g in groups.values():
        g.sort(key=lambda d: d.get("queue_position", 0))

    id_map: Dict[Tuple[str, int], int] = {}
    for g_name, g in groups.items():
        for idx, entry in enumerate(g):
            kwargs = {
                "name": entry["name"],
                "description": entry["description"],
                "status": entry.get("status", "queued"),
                "priority": entry.get("priority", Priority.normal),
            }
            if start := entry.get("start_time"):
                kwargs["schedule"] = Schedule(
                    start_time=start,
                    prev_task=None,
                    next_task=None,
                )
            task_id = tlm._create_task(**kwargs)
            id_map[(g_name, idx)] = task_id

    # Wire up prev/next links inside each queue group
    for g_name, g in groups.items():
        for idx, _ in enumerate(g):
            cur = id_map[(g_name, idx)]
            prev_ = id_map.get((g_name, idx - 1)) if idx > 0 else None
            next_ = id_map.get((g_name, idx + 1)) if idx < len(g) - 1 else None
            unify.update_logs(
                context="Tasks",
                logs=tlm._get_logs_by_task_ids(task_ids=cur),
                entries={"schedule": {"prev_task": prev_, "next_task": next_}},
                overwrite=True,
            )
    return theme


# ---------------------------------------------------------------------------
# Natural‑language dispatcher (ask vs update)
# ---------------------------------------------------------------------------


class _DispatchResp(BaseModel):
    require_update: bool = Field(...)
    fixed_text: str = Field(...)


def _dispatch(tlm: TaskListManager, raw: str, *, show_steps: bool):
    raw = raw.strip()

    llm = unify.Unify("gpt-4o@openai", response_format=_DispatchResp)
    llm.set_system_message(
        "There is a table containing a list of tasks, and all of their properties. "
        "The user has made a request via a speech‑to‑text process, which can introduce errors. "
        "Using the output schema provided, output a corrected transcript and decide whether the task table must be updated.",
    )
    resp = _DispatchResp.model_validate_json(llm.generate(raw))

    if resp.require_update:
        ans, steps = tlm.update(
            text=resp.fixed_text,
            return_reasoning_steps=show_steps,
            log_tool_steps=show_steps,
        )
        return "update", ans, steps

    ans, steps = tlm.ask(
        text=resp.fixed_text,
        return_reasoning_steps=show_steps,
        log_tool_steps=show_steps,
    )
    return "ask", ans, steps


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TaskListManager sandbox with minimalist voice mode (Deepgram v4, Cartesia)",
    )
    parser.add_argument(
        "--voice",
        "-v",
        action="store_true",
        help="enable voice capture/playback",
    )
    parser.add_argument("--scenario", choices=["fixed", "llm"], default="fixed")
    parser.add_argument("--new", "-n", action="store_true", help="wipe & reseed data")
    parser.add_argument(
        "--silent",
        "-s",
        action="store_true",
        help="suppress tool logs",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="verbose HTTP/LLM logging",
    )
    args = parser.parse_args()

    # Logging
    if not args.silent:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        _LG.setLevel(logging.INFO)
        if not args.debug:
            for noisy in ("unify", "unify.utils", "unify.logging", "requests", "httpx"):
                logging.getLogger(noisy).setLevel(logging.WARNING)

    # Unify context
    unify.activate("TaskListSandbox")
    fresh = "Tasks" not in unify.get_contexts() or args.new
    unify.set_context("Tasks", overwrite=fresh)

    # Manager
    tlm = TaskListManager()
    tlm.start()

    if fresh:
        if args.scenario == "llm":
            _seed_llm(tlm)
        else:
            _seed_fixed(tlm)

    print("TaskListManager sandbox – speak or type. 'quit' to exit.\n")

    # Interaction loop
    if args.voice:
        while True:
            audio_bytes = _record_until_enter()
            user_text = _transcribe_deepgram(audio_bytes).strip()
            if not user_text:
                continue
            print(f"▶️  {user_text}")
            if user_text.lower() in {"quit", "exit"}:
                break
            _speak("Working on this now…")
            kind, result, _ = _dispatch(tlm, user_text, show_steps=not args.silent)
            print(f"[{kind}] => {result}\n")
            _speak(result)
    else:
        try:
            while True:
                line = input("> ").strip()
                if line.lower() in {"quit", "exit"}:
                    break
                if not line:
                    continue
                kind, result, _ = _dispatch(tlm, line, show_steps=not args.silent)
                print(f"[{kind}] => {result}\n")
        except (EOFError, KeyboardInterrupt):
            print()


if __name__ == "__main__":
    main()
