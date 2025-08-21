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
from unity.planner.simulated import SimulatedPlanner
from sandboxes.utils import (
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    speak_and_wait as _speak_wait,
    await_with_interrupt as _await_with_interrupt,
    steering_controls_hint as _steer_hint,
    build_cli_parser,
    activate_project,
    _wait_for_tts_end as _wait_tts_end,
    configure_sandbox_logging,
)

LG = logging.getLogger("task_scheduler_sandbox")

# ─────────────────────── Simulation controls (defaults + parser) ───────────────────────

_DEFAULT_SIM_STEPS: int | None = None
_DEFAULT_SIM_TIMEOUT: float | None = None


class _SimConfig(BaseModel):
    core_text: str
    steps: int | None = None
    timeout_seconds: float | None = None


_SIM_PARSER_SYS = (
    "Extract optional simulation controls from the user's text for starting a task.\n"
    "Controls to detect (may appear anywhere, any order, any phrasing):\n"
    "- steps: an integer number of steps (e.g., 'in 5 steps', 'limit to 7 steps', 'steps=3').\n"
    "- timeout: a duration with units seconds or minutes (e.g., 'in 30 seconds', 'timeout 2 min', '90s', 'timeout=1.5 minutes').\n"
    "Return JSON with fields: core_text (original text with any control phrases removed),\n"
    "steps (int or null) and timeout_seconds (float seconds or null).\n"
    "Do not rewrite the rest of the text. Preserve meaning and wording. Remove only the control phrases and any glue words like 'with', 'for', 'in', 'limit to' that are only part of those control expressions.\n"
)


def _parse_simulation_config(text: str) -> _SimConfig:
    try:
        judge = unify.Unify("gpt-4o@openai", response_format=_SimConfig)
        return _SimConfig.model_validate_json(
            judge.set_system_message(_SIM_PARSER_SYS).generate(text),
        )
    except Exception:
        # Fallback – no parsing; keep text as-is
        return _SimConfig(core_text=text, steps=None, timeout_seconds=None)


# ═════════════════════════════════ seed helpers ═════════════════════════════


async def _build_scenario(custom: Optional[str] = None) -> Optional[str]:
    """
    Populate the task scheduler with sample data **through the official tools**
    using :class:`ScenarioBuilder`.  Falls back to the fixed seed on any error.
    """
    global _DEFAULT_SIM_STEPS, _DEFAULT_SIM_TIMEOUT
    # Apply current defaults to the scheduler's planner so later 'start' calls inherit them
    ts = TaskScheduler(
        planner=SimulatedPlanner(
            steps=_DEFAULT_SIM_STEPS,
            timeout=_DEFAULT_SIM_TIMEOUT,
        ),
    )

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

    # Allow scenario descriptions to include default simulation controls (steps/timeout)
    try:
        parsed = _parse_simulation_config(description)
        # Update global defaults if provided
        if parsed.steps is not None:
            _DEFAULT_SIM_STEPS = int(parsed.steps)
        if parsed.timeout_seconds is not None:
            _DEFAULT_SIM_TIMEOUT = float(parsed.timeout_seconds)
        description_core = parsed.core_text.strip() or description
    except Exception:
        description_core = description

    builder = ScenarioBuilder(
        description=description,
        tools={
            "update": ts.update,
            "ask": ts.ask,
        },
    )

    try:
        # Use the core description text without sim-control phrases
        builder._description = description_core  # type: ignore[attr-defined]
        await builder.create()
    except Exception as exc:
        raise RuntimeError(f"LLM seeding via ScenarioBuilder failed. {exc}")

    return None


# ═════════════════════════════ intent dispatcher ════════════════════════════


class _Intent(BaseModel):
    action: str = Field(..., pattern="^(ask|update|start)$")


_INTENT_SYS_MSG = (
    "You are an intent router for the TaskScheduler.\n"
    "Decide if the user's input is:\n"
    " - a read-only question about existing tasks ('ask'),\n"
    " - a write/mutation that creates/updates/deletes tasks or metadata ('update'), or\n"
    " - an instruction to begin working on a specific task ('start').\n"
    "Return ONLY JSON with this shape: {'action':'ask'|'update'|'start'}. Do not rewrite or summarize the user's input.\n"
    "Rules:\n"
    "- Classify as 'start' when the user asks to begin doing the task now/immediately/ASAP, or uses imperative phrasing to carry out the task (e.g., 'start', 'begin', 'execute', 'work on', 'open a browser').\n"
    "- Classify as 'update' for creating/updating/deleting tasks, schedules, priorities, queues, ordering, or status (pause/resume/cancel).\n"
    "- Classify as 'ask' for information-only queries about the current task list, schedules, priorities, or status.\n"
    "Examples:\n"
    "Input: 'Start task 12 right now' → {'action':'start'}\n"
    "Input: 'Could you research ACME ASAP and report back?' → {'action':'start'}\n"
    "Input: 'Move task 3 behind task 5' → {'action':'update'}\n"
    "Input: 'Pause the active task' → {'action':'update'}\n"
    "Input: 'What is due this week?' → {'action':'ask'}"
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
    *parent_chat_context* to the TaskScheduler methods. Always pass the original
    user input (*raw*) unchanged to the selected method.
    """

    # LLM-only routing
    judge = unify.Unify("gpt-4o@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )

    # For 'start' requests, peel off optional simulation controls and inject a per-call planner
    if intent.action == "update":
        handle = await ts.update(
            raw,
            parent_chat_context=parent_chat_context,
            _return_reasoning_steps=show_steps,
        )
    elif intent.action == "start":
        parsed = _parse_simulation_config(raw)
        # Mask only steps/timeout control phrases; pass the remaining text as-is
        core_text = parsed.core_text if parsed.core_text != "" else raw
        # Build a one-off planner using extracted controls falling back to defaults
        eff_steps = parsed.steps if parsed.steps is not None else _DEFAULT_SIM_STEPS
        eff_timeout = (
            parsed.timeout_seconds
            if parsed.timeout_seconds is not None
            else _DEFAULT_SIM_TIMEOUT
        )

        # Swap the planner on the instance for this call only
        original_planner = getattr(ts, "_planner", None)
        try:
            setattr(
                ts,
                "_planner",
                SimulatedPlanner(steps=eff_steps, timeout=eff_timeout),
            )
            handle = await ts.execute_task(
                core_text,
                parent_chat_context=parent_chat_context,
            )
        finally:
            if original_planner is not None:
                setattr(ts, "_planner", original_planner)
    else:  # ask
        handle = await ts.ask(
            raw,
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

    # logging via shared helper
    configure_sandbox_logging(
        args.log_in_terminal,
        None,
        args.log_tcp_port,
        http_tcp_port=args.http_log_tcp_port,
        unify_requests_log_file=".logs_unify_requests.txt",
    )
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
                raw = input("command ('r' to record)> ")
                if raw.strip().lower() == "r":
                    audio = _record_until_enter()
                    raw = _transcribe_deepgram(audio)
                    if not raw or raw.strip() == "":
                        continue
                    print(f"▶️  {raw}")
            else:
                raw = input("command> ")

            # Show help table
            if raw.strip().lower() in {"help", "h", "?"}:
                _explain_commands()
                continue

            if raw.strip().lower() in {"quit", "exit"}:
                break
            if raw.strip() == "":
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
            working = raw.strip()
            parts = working.split(maxsplit=1)
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

                if args.voice:
                    task = asyncio.create_task(_build_scenario(description))
                    _speak_wait("Got it, working on your custom scenario now.")
                    print(
                        "[generate] Building synthetic tasks – this can take a moment…",
                    )
                    try:
                        await task
                        _speak_wait(
                            "All done, your custom scenario is built and ready to go.",
                        )
                    except Exception as exc:
                        LG.error("Scenario generation failed: %s", exc, exc_info=True)
                        print(f"❌  Failed to generate scenario: {exc}")
                else:
                    print(
                        "[generate] Building synthetic tasks – this can take a moment…",
                    )
                    try:
                        await _build_scenario(description)
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
                description = _transcribe_deepgram(audio)
                if not description or description.strip() == "":
                    print("⚠️  Transcription was empty – please try again.")
                    continue
                print(f"▶️  {description}")

                task = asyncio.create_task(_build_scenario(description))
                _speak_wait("Got it, working on your custom scenario now.")
                print("[generate] Building synthetic tasks – this can take a moment…")
                try:
                    await task
                    _speak_wait(
                        "All done, your custom scenario is built and ready to go.",
                    )
                except Exception as exc:
                    LG.error("Scenario generation failed: %s", exc, exc_info=True)
                    print(f"❌  Failed to generate scenario: {exc}")
                continue  # back to REPL

            # Ignore steering commands when no request is running
            if raw.lstrip().startswith("/"):
                print(
                    "(no active request) Steering commands are only available while a call is running.",
                )
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
                _wait_tts_end()

            print(_steer_hint())
            answer = await _await_with_interrupt(
                _handle,
                enable_voice_steering=bool(args.voice),
            )
            if args.voice:
                _speak("Okay, that's all done")
                _wait_tts_end()
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
