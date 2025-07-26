"""
=====================================================================
Interactive sandbox for **TaskScheduler**.

It supports:
• Fixed or LLM‑generated seed data.
• Voice or plain‑text input (same helpers as the other sandboxes).
• Automatic dispatch to `ask`, `update` *or* `execute_task` depending on intent.
• Mid‑conversation interruption (pause / interject / cancel).
"""

from __future__ import annotations

# ─────────────────────────────── stdlib / vendored ──────────────────────────
import os
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import re
from datetime import datetime

import unify

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sandboxes.scenario_builder import ScenarioBuilder

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

# ────────────────────────────────  unity imports  ───────────────────────────
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.common.llm_helpers import SteerableToolHandle
from sandboxes.utils import (
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    get_custom_scenario,
    await_with_interrupt as _await_with_interrupt,
    build_cli_parser,
    activate_project,
)

LG = logging.getLogger("task_scheduler_sandbox")

# ═════════════════════════════════ seed helpers ═════════════════════════════


async def _build_scenario(custom: Optional[str] = None) -> Optional[str]:
    """
    Populate the task scheduler with sample data **through the official tools**
    using :class:`ScenarioBuilder`.  Falls back to the fixed seed on any error.
    """
    ts = TaskScheduler()

    description = (
        custom.strip()
        if custom
        else (
            "Generate a backlog of 12 realistic product‑development tasks split "
            "across 'Inbox', 'Next', 'Scheduled' and 'Waiting' queues.  Each task "
            "must have a short title, detailed description, due date and priority. "
            "Include dependencies between a few tasks so the schedule has depth."
        )
    )
    description += (
        "\nBatch actions: each `update` call can create or modify several tasks "
        "and `ask` can verify results to avoid duplications."
    )

    builder = ScenarioBuilder(
        description=description,
        tools={
            "update": ts.update,
            "ask": ts.ask,
        },
    )

    try:
        await builder.create()
    except Exception as exc:
        raise RuntimeError(f"LLM seeding via ScenarioBuilder failed. {exc}")

    return None


# ═════════════════════════════ intent dispatcher ════════════════════════════


class _Intent(BaseModel):
    action: str = Field(..., pattern="^(ask|update|start)$")
    cleaned_text: str


_INTENT_SYS_MSG = (
    "Decide whether the user input is a *query* about existing tasks (`ask`), "
    "a *mutation* that creates/updates/deletes tasks (`update`), or an "
    "instruction to begin working on a specific task (`start`). "
    "Return JSON {'action':'ask'|'update'|'start','cleaned_text':<fixed_input>}."
)


async def _dispatch_with_context(
    ts: TaskScheduler,
    raw: str,
    *,
    show_steps: bool,
    parent_chat_context: List[Dict[str, str]],
) -> Tuple[str, SteerableToolHandle]:
    """
    Decide whether to call `ask`, `update` or `execute_task`, forwarding
    *parent_chat_context* to the TaskScheduler methods.  `execute_task` requires
    a numeric *task_id* which is extracted from the user's text.
    """

    lowered = raw.lower()

    # ───── quick heuristics (fast‑path) ───────────────────────────────
    if lowered.startswith(
        (
            "add ",
            "create ",
            "update ",
            "change ",
            "delete ",
            "schedule ",
            "move ",
            "reschedule ",
        ),
    ):
        handle = await ts.update(
            raw,
            parent_chat_context=parent_chat_context,
            _return_reasoning_steps=show_steps,
        )
        return "update", handle

    # ───── everything else – ask an LLM judge ────────────────────────
    judge = unify.Unify("gpt-4o@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )

    if intent.action == "start":
        task_id = _extract_first_int(intent.cleaned_text)
        handle = await ts.execute_task(
            task_id,
            parent_chat_context=parent_chat_context,
        )
    elif intent.action == "update":
        handle = await ts.update(
            intent.cleaned_text,
            parent_chat_context=parent_chat_context,
            _return_reasoning_steps=show_steps,
        )
    else:  # ask
        handle = await ts.ask(
            intent.cleaned_text,
            parent_chat_context=parent_chat_context,
            _return_reasoning_steps=show_steps,
        )
    return intent.action, handle


def _extract_first_int(text: str) -> int:
    """Return the first integer found in *text* or raise ValueError."""
    m = re.search(r"\d+", text)
    if not m:
        raise ValueError("Could not find a task ID in the request.")
    return int(m.group())


# ══════════════════════════════════  CLI  ═══════════════════════════════════


async def _main_async() -> None:
    parser = build_cli_parser("TaskScheduler sandbox")
    args = parser.parse_args()

    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    activate_project(args.project_name, args.overwrite)
    base_ctx = unify.get_active_context().get("write")
    traces_ctx = f"{base_ctx}/Traces" if base_ctx else "Traces"
    unify.set_trace_context(traces_ctx)
    if args.overwrite:
        contexts = unify.get_contexts()
        if "Tasks" in contexts:
            unify.delete_context("Tasks")
        if traces_ctx in contexts:
            unify.delete_context(traces_ctx)
        unify.create_context(traces_ctx)

    # ─────────────────── project version handling ────────────────────
    if args.project_version != -1:
        commits = unify.get_project_commits(args.project_name)
        if commits:
            try:
                target = commits[args.project_version]
                unify.rollback_project(args.project_name, target["commit_hash"])
                LG.info("[version] Rolled back to commit %s", target["commit_hash"])
            except IndexError:
                LG.warning(
                    "[version] project_version index %s out of range, ignoring",
                    args.project_version,
                )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LG.setLevel(logging.INFO)

    ts = TaskScheduler()
    if args.traced:
        ts = unify.traced(ts)

    scenario_text: Optional[str] = get_custom_scenario(args)
    LG.info("[seed] building synthetic task list – this can take 20-40 s…")
    if args.voice:
        _speak("Sure thing, building your custom scenario now.")
    await _build_scenario(scenario_text)
    LG.info("[seed] done.")
    if args.voice:
        _speak("All done, your custom scenario is ready.")

    print("TaskScheduler sandbox – type or speak. 'quit' to exit.\n")

    _speak(
        "Press enter to record a question or request an update for the task schedule.",
    )

    # running memory of the dialogue
    chat_history: List[Dict[str, str]] = []

    # interaction loop
    while True:
        try:
            if args.voice:
                audio = _record_until_enter()
                raw = _transcribe_deepgram(audio).strip()
                if not raw:
                    continue
                print(f"▶️  {raw}")
            else:
                raw = input("> ").strip()

            if raw.lower() in {"quit", "exit"}:
                break
            if not raw:
                continue

            # ─────────────── save project snapshot ────────────────
            if raw.lower() in {"save_project", "sp"}:
                commit_hash = unify.commit_project(
                    args.project_name,
                    commit_message=f"Sandbox save {datetime.utcnow().isoformat()}",
                ).get("commit_hash")
                print(f"💾 Project saved at commit {commit_hash}")
                if args.voice:
                    _speak("Project saved")
                continue

            # ──────────────── remember the user's utterance ────────────────
            _kind, _handle = await _dispatch_with_context(
                ts,
                raw,
                show_steps=args.debug,
                parent_chat_context=list(chat_history),
            )
            chat_history.append({"role": "user", "content": raw})
            if args.voice:
                _speak("Let me take a look, give me a moment")

            answer = await _await_with_interrupt(_handle)
            if args.voice:
                _speak("Okay, that's all done")
            if isinstance(answer, tuple):  # reasoning steps requested
                answer, _steps = answer
            print(f"[{_kind}] → {answer}\n")

            # ──────────────── remember the assistant's reply ───────────────
            chat_history.append({"role": "assistant", "content": answer})
            if args.voice:
                _speak(f"{answer} Anything else I can help with?")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break
        except Exception as exc:
            LG.error("Error: %s", exc, exc_info=True)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
