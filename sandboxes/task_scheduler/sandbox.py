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
    await_with_interrupt as _await_with_interrupt,
    build_cli_parser,
    activate_project,
    _wait_for_tts_end as _wait_tts_end,
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

    # ───── immediate-execution heuristic (fast‑path) ─────────────────────
    # If the user clearly asks to perform the work now/immediately, prefer
    # starting the task over updating metadata.
    if any(
        token in lowered
        for token in (
            "right now",
            "immediately",
            "asap",
            "as soon as possible",
            "right away",
            "open a browser",
        )
    ):
        handle = await ts.execute_task(
            raw,
            parent_chat_context=parent_chat_context,
        )
        return "start", handle

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
    # Strengthen classification guidance: treat imperative, real-time
    # requests (e.g. "start", "begin", "open a browser", "do X now") as 'start'.
    start_bias_msg = (
        _INTENT_SYS_MSG
        + "\nRules:"
        + "\n- Classify as 'start' when the user asks you to begin doing the task now/immediately/ASAP (e.g. 'right now', 'as soon as possible', 'immediately'), or uses imperative phrasing to carry out the task (e.g. 'start', 'begin', 'open a browser', 'do five minutes of research')."
        + "\n- Use 'update' strictly for creating/updating/deleting tasks, schedules, priorities, or queue ordering."
        + "\n- Use 'ask' for information-only questions about the current task list."
        + "\nExamples:"
        + "\nInput: 'Start task 12' → {'action':'start','cleaned_text':'12'}"
        + "\nInput: 'Could you research ACME right now and tell me what you find?' → {'action':'start','cleaned_text':'research ACME right now and report back'}"
        + "\nInput: 'Move task 3 behind task 5' → {'action':'update','cleaned_text':'Move task 3 behind task 5'}"
        + "\nInput: 'What is due this week?' → {'action':'ask','cleaned_text':'What is due this week?'}"
    )
    intent = _Intent.model_validate_json(
        judge.set_system_message(start_bias_msg).generate(raw),
    )

    # For 'start', call execute_task so the scheduler resolves/creates the
    # relevant task and launches it.
    if intent.action in {"update"}:
        handle = await ts.update(
            intent.cleaned_text,
            parent_chat_context=parent_chat_context,
            _return_reasoning_steps=show_steps,
        )
    elif intent.action == "start":
        handle = await ts.execute_task(
            intent.cleaned_text,
            parent_chat_context=parent_chat_context,
        )
    else:  # ask
        handle = await ts.ask(
            intent.cleaned_text,
            parent_chat_context=parent_chat_context,
            _return_reasoning_steps=show_steps,
        )
    return intent.action, handle


# ══════════════════════════════════  CLI  ═══════════════════════════════════


async def _main_async() -> None:
    parser = build_cli_parser("TaskScheduler sandbox")

    # No automatic seeding – users can invoke 'us' / 'usv' commands to populate tasks when desired.

    args = parser.parse_args()

    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    # ─────────────────── Unify context ────────────────────
    activate_project(args.project_name, args.overwrite)
    base_ctx = unify.get_active_context().get("write")
    traces_ctx = f"{base_ctx}/Traces" if base_ctx else "Traces"
    unify.set_trace_context(traces_ctx)
    if args.overwrite:
        ctxs = unify.get_contexts()
        if "Tasks" in ctxs:
            unify.delete_context("Tasks")
        if traces_ctx in ctxs:
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

    # logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LG.setLevel(logging.INFO)

    ts = TaskScheduler()
    if args.traced:
        ts = unify.traced(ts)

    # ─────────────────── command helper output ────────────────────
    _COMMANDS_HELP = (
        "\nTaskScheduler sandbox – type commands below (press ↵ with an empty "
        "line to dictate via voice when --voice mode is active – type 'r' to record).  'quit' to exit.\n\n"
        "┌────────────────── accepted commands ─────────────────────┐\n"
        "│ us  {description}     – update_scenario (text)           │\n"
        "│ usv                   – update_scenario_vocally          │\n"
        "│ r / free text         – freeform ask / update / start    │\n"
        "│ save_project | sp     – save project snapshot            │\n"
        "│ help | h              – show this help                   │\n"
        "└──────────────────────────────────────────────────────────┘\n"
    )

    def _explain_commands() -> None:  # noqa: D401 – helper
        print(_COMMANDS_HELP)

    if args.voice:
        _speak(
            "Sandbox ready. You can type commands, or press enter on an empty line "
            "to record a voice query.  Use 'u-s-v' to build a new scenario vocally.",
        )
        _wait_tts_end()

    # running memory of the dialogue
    chat_history: List[Dict[str, str]] = []

    # interaction loop
    while True:
        # Keep command list visible similar to other sandboxes
        print()
        _explain_commands()
        print()

        try:
            if args.voice:
                # Ensure any ongoing TTS playback has finished before showing prompt
                _wait_tts_end()
            if args.voice:
                raw = input("command ('r' to record)> ").strip()
                if raw.lower() == "r":
                    audio = _record_until_enter()
                    raw = _transcribe_deepgram(audio).strip()
                    if not raw:
                        continue
                    print(f"▶️  {raw}")
            else:
                raw = input("command> ").strip()

            # Show help table
            if raw.lower() in {"help", "h", "?"}:
                _explain_commands()
                continue

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

            # ─────────────── scenario (re)seeding commands ────────────────
            parts = raw.split(maxsplit=1)
            cmd_lower = parts[0].lower()

            if cmd_lower in {"us", "update_scenario"}:
                description = parts[1].strip() if len(parts) > 1 else ""
                if not description:
                    description = input(
                        "🧮 Describe the task scenario you want to build > ",
                    ).strip()
                    if not description:
                        print("⚠️  No description provided – cancelled.")
                        continue

                print("[generate] Building synthetic tasks – this can take a moment…")
                if args.voice:
                    _speak("Sure thing, building your custom scenario now.")
                try:
                    await _build_scenario(description)
                    if args.voice:
                        _speak(
                            "All done, your custom scenario is built and ready to go.",
                        )
                except Exception as exc:
                    LG.error("Scenario generation failed: %s", exc, exc_info=True)
                    print(f"❌  Failed to generate scenario: {exc}")
                continue  # back to REPL

            if cmd_lower in {"usv", "update_scenario_vocally"}:
                if not args.voice:
                    print(
                        "⚠️  Voice mode not enabled – restart with --voice or use 'us' instead.",
                    )
                    continue

                audio = _record_until_enter()
                description = _transcribe_deepgram(audio).strip()
                if not description:
                    print("⚠️  Transcription was empty – please try again.")
                    continue
                print(f"▶️  {description}")

                print("[generate] Building synthetic tasks – this can take a moment…")
                try:
                    await _build_scenario(description)
                    if args.voice:
                        _speak(
                            "All done, your custom scenario is built and ready to go.",
                        )
                except Exception as exc:
                    LG.error("Scenario generation failed: %s", exc, exc_info=True)
                    print(f"❌  Failed to generate scenario: {exc}")
                continue  # back to REPL

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
    # end while


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
