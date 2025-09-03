"""
=====================================================================
Interactive sandbox for **TaskScheduler**.

It supports:
• Fixed or LLM‑generated seed data.
• Voice or plain‑text input (same helpers as the other sandboxes).
• Automatic dispatch to `ask`, `update` *or* `execute` depending on intent.
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

from dotenv import load_dotenv

load_dotenv()

import unify
from pydantic import BaseModel, Field
from sandboxes.scenario_builder import ScenarioBuilder

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ────────────────────────────────  unity imports  ───────────────────────────
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.common.llm_helpers import SteerableToolHandle
from unity.actor.simulated import SimulatedActor
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
    call_manager_with_optional_clarifications,
    pydantic_response_format,
    # Simulation planning (sandbox-only)
    SIMULATION_PLANS,
    SimulationParams,
    SimulationSelector,
    parse_simulation_params_kv,
    apply_per_task_simulation_patch,
)

LG = logging.getLogger("task_scheduler_sandbox")

# ─────────────────────── Simulation controls (defaults + parser) ───────────────────────

_DEFAULT_SIM_STEPS: int | None = None
_DEFAULT_SIM_TIMEOUT: float | None = None


class _SimConfig(BaseModel):
    core_text: str
    steps: int | None = None
    timeout_seconds: float | None = None
    # Extra, simulation-only knob
    simulation_guidance: str | None = None


_SIM_PARSER_SYS = (
    "Extract optional simulation controls from the user's text for starting a task.\n"
    "Controls to detect (may appear anywhere, any order, any phrasing):\n"
    "- steps: an integer number of steps (e.g., 'in 5 steps', 'limit to 7 steps', 'steps=3').\n"
    "- timeout: a duration with units seconds or minutes (e.g., 'in 30 seconds', 'timeout 2 min', '90s', 'timeout=1.5 minutes').\n"
    "- simulation_guidance: optional free-form guidance that should influence the simulated behaviour only (e.g., 'if asked about timing, say there is an issue and it's taking longer').\n"
    "Return JSON with fields: core_text (the user's original request with ANY full sentence or standalone clause that expresses simulation controls REMOVED),\n"
    "steps (int or null), timeout_seconds (float seconds or null), simulation_guidance (string or null).\n"
    "Rules for redaction when producing core_text:\n"
    "- Remove entire sentences that mention simulation parameters (timeout/steps) or are meta-instructions about the simulation.\n"
    "- If controls are embedded as a trailing or leading clause (e.g., '... and make it 60 seconds timeout'), remove the whole clause so the remaining sentence reads naturally.\n"
    "- Do not leave dangling words or punctuation (avoid artefacts like 'a.' or double spaces).\n"
    "- Preserve the user's task wording otherwise; do not paraphrase or summarize.\n"
    "Examples:\n"
    "1) Input: 'Could you actually start researching accounted limited, right now? As for the simulation, please make this have a timeout of sixty seconds. Thanks.'\n"
    "   core_text: 'Could you actually start researching accounted limited, right now? Thanks.' steps: null timeout_seconds: 60 simulation_guidance: null\n"
    "2) Input: 'Start task 12 and in 5 steps please'\n"
    "   core_text: 'Start task 12' steps: 5 timeout_seconds: null simulation_guidance: null\n"
    "3) Input: 'Begin now, with a timeout of 90s'\n"
    "   core_text: 'Begin now' steps: null timeout_seconds: 90 simulation_guidance: null\n"
    "4) Input: 'Start task 8. For the simulation, if you're asked how long it will take, say there was an issue and it's taking longer.'\n"
    "   core_text: 'Start task 8' steps: null timeout_seconds: null simulation_guidance: 'if you're asked how long it will take, say there was an issue and it's taking longer'\n"
    "5) Input: 'Start the task immediately, and whenever we ask for a progress update in the simulation, say it's taking longer than expected and should be done soon.'\n"
    "   core_text: 'Start the task immediately' steps: null timeout_seconds: null simulation_guidance: 'whenever we ask for a progress update in the simulation, say it's taking longer than expected and should be done soon'\n"
)


def _parse_simulation_config(text: str) -> _SimConfig:
    try:
        judge = unify.Unify(
            "gpt-5@openai",
            response_format=pydantic_response_format(_SimConfig),
        )
        parsed = _SimConfig.model_validate_json(
            judge.set_system_message(_SIM_PARSER_SYS).generate(text),
        )
        # Trust the LLM for sentence/phrase removal per system prompt guidance.
        core = (parsed.core_text or text).strip()
        try:
            LG.info(
                "[sim-config/parser] raw=%r -> core_text=%r, steps=%s, timeout_seconds=%s",
                text,
                core,
                parsed.steps,
                parsed.timeout_seconds,
            )
        except Exception:
            pass
        return _SimConfig(
            core_text=core,
            steps=parsed.steps,
            timeout_seconds=parsed.timeout_seconds,
            simulation_guidance=parsed.simulation_guidance,
        )
    except Exception:
        # Fallback – no parsing; keep text as-is
        try:
            LG.info(
                "[sim-config/parser] raw=%r -> PARSE FAILED; defaulting to steps=None, timeout_seconds=None",
                text,
            )
        except Exception:
            pass
        return _SimConfig(core_text=text, steps=None, timeout_seconds=None)


# ═════════════════════════════════ seed helpers ═════════════════════════════


async def _build_scenario(custom: Optional[str] = None) -> Optional[str]:
    """
    Populate the task scheduler with sample data **through the official tools**
    using :class:`ScenarioBuilder`.  Falls back to the fixed seed on any error.
    """
    global _DEFAULT_SIM_STEPS, _DEFAULT_SIM_TIMEOUT
    # Apply current defaults to the scheduler's actor so later 'start' calls inherit them
    ts = TaskScheduler(
        actor=SimulatedActor(
            steps=_DEFAULT_SIM_STEPS,
            duration=_DEFAULT_SIM_TIMEOUT,
            log_mode="print",
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
        try:
            LG.info(
                "[sim-config/seed] parsed steps=%s, timeout_seconds=%s; defaults now steps=%s, timeout_seconds=%s",
                parsed.steps,
                parsed.timeout_seconds,
                _DEFAULT_SIM_STEPS,
                _DEFAULT_SIM_TIMEOUT,
            )
        except Exception:
            pass
        # Also print immediately to the terminal so users see what was captured
        try:
            if parsed.steps is None and parsed.timeout_seconds is None:
                print(
                    "🧭 No scenario-level simulation guidance detected – defaults unchanged.",
                )
            else:
                print("🧭 Scenario-level simulation guidance detected:")
                if parsed.steps is not None:
                    print(f"   🔢 Steps: {parsed.steps}")
                if parsed.timeout_seconds is not None:
                    print(f"   ⏱️ Timeout: {parsed.timeout_seconds}s")
                if getattr(parsed, "simulation_guidance", None):
                    print(f"   🧭 Guidance: {parsed.simulation_guidance}")

            # Show non-None defaults only
            if _DEFAULT_SIM_STEPS is None and _DEFAULT_SIM_TIMEOUT is None:
                print("⚙️ Defaults now: none set (no step limit, no timeout)")
            else:
                print("⚙️ Defaults now:")
                if _DEFAULT_SIM_STEPS is not None:
                    print(f"   🔢 Steps: {_DEFAULT_SIM_STEPS}")
                if _DEFAULT_SIM_TIMEOUT is not None:
                    print(f"   ⏱️ Timeout: {_DEFAULT_SIM_TIMEOUT}s")
        except Exception:
            pass
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
    "Rules (disambiguation):\n"
    "- Choose 'start' when the user asks to begin doing a task now/immediately/ASAP, or uses imperative phrasing to carry out the task (e.g., 'start', 'begin', 'execute', 'work on', 'open a browser').\n"
    "- If the input combines an immediate start request with configuration or setup details (e.g., scheduling hints, limits, simulation controls, progress-response guidance, defaults), STILL choose 'start'. Treat those details as parameters for execution rather than a separate update.\n"
    "- Choose 'update' for requests whose primary goal is to create, modify, reorder, schedule, or delete tasks/metadata (including pause/resume/cancel of tasks) without instructing immediate execution.\n"
    "- Choose 'ask' for information-only queries such as progress/status checks ('how is it going', 'any update', 'have you scheduled X yet', 'is it done', 'ETA'), or questions about tasks, schedules, priorities, or status. Polite/indirect wording that requests information only is still 'ask'.\n"
    "- When a message mixes a question with an explicit directive to change data (create/update/delete/reorder/schedule), choose 'update'. When a message mixes a question with an explicit directive to begin doing the work now, choose 'start'.\n"
    "- Ignore meta-simulation instructions (e.g., 'make it 500 seconds', 'when asked, say it's taking longer') for classification purposes; they do not change 'start' vs 'update'.\n"
    "Examples:\n"
    "Input: 'Start task 12 right now' → {'action':'start'}\n"
    "Input: 'Could you research ACME ASAP and report back?' → {'action':'start'}\n"
    "Input: 'Start researching ACME now; set the simulation timeout to 500 seconds and, on progress requests, reply that it's taking longer' → {'action':'start'}\n"
    "Input: 'Create four tasks for next Monday in this order: …' → {'action':'update'}\n"
    "Input: 'Move task 3 behind task 5' → {'action':'update'}\n"
    "Input: 'Pause the active task' → {'action':'update'}\n"
    "Input: 'What is due this week?' → {'action':'ask'}\n"
    "Input: 'Could you let me know how that's coming along? Have you scheduled the tasks yet?' → {'action':'ask'}"
)


async def _dispatch_with_context(
    ts: TaskScheduler,
    raw: str,
    *,
    show_steps: bool,
    parent_chat_context: List[Dict[str, str]],
    clarifications_enabled: bool,
    enable_voice: bool,
) -> Tuple[
    str,
    SteerableToolHandle,
    Optional[asyncio.Queue[str]],
    Optional[asyncio.Queue[str]],
]:
    """
    Decide whether to call `ask`, `update` or `execute`, forwarding
    *parent_chat_context* to the TaskScheduler methods. Always pass the original
    user input (*raw*) unchanged to the selected method.
    """

    # LLM-only routing
    judge = unify.Unify("gpt-5@openai", response_format=_Intent)
    judge = unify.Unify(
        "gpt-5@openai",
        response_format=pydantic_response_format(_Intent),
    )
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )

    # Immediate terminal confirmation of selected method
    try:
        _selected = "execute" if intent.action == "start" else intent.action
        print(f"➡️  Selected: {_selected}")
    except Exception:
        pass

    # For 'start' requests, peel off optional simulation controls and inject a per-call actor
    # Prepare optional clarification channels
    clar_up_q: Optional[asyncio.Queue[str]] = None
    clar_down_q: Optional[asyncio.Queue[str]] = None
    if clarifications_enabled:
        clar_up_q = asyncio.Queue()
        clar_down_q = asyncio.Queue()

    if intent.action == "update":
        handle, clar_up_q, clar_down_q = (
            await call_manager_with_optional_clarifications(
                ts.update,
                raw,
                parent_chat_context=parent_chat_context,
                return_reasoning_steps=show_steps,
                clarifications_enabled=clarifications_enabled,
            )
        )
    elif intent.action == "start":
        parsed = _parse_simulation_config(raw)
        # Mask only steps/timeout control phrases; pass the remaining text as-is
        core_text = parsed.core_text if parsed.core_text != "" else raw
        # Build a one-off actor using extracted controls falling back to defaults,
        # and merge with any stored plan for a numeric task id if provided.
        eff_steps = parsed.steps if parsed.steps is not None else _DEFAULT_SIM_STEPS
        eff_timeout = (
            parsed.timeout_seconds
            if parsed.timeout_seconds is not None
            else _DEFAULT_SIM_TIMEOUT
        )
        eff_guidance = parsed.simulation_guidance

        # Default to a 20s duration when no explicit steps or timeout were provided
        if eff_steps is None and eff_timeout is None:
            eff_timeout = 20.0

        stripped_id = core_text.strip()
        if stripped_id.isdigit():
            try:
                tid_int = int(stripped_id)
                stored = SIMULATION_PLANS.resolve_for_task_id(tid_int)
                if stored is not None:
                    if eff_steps is None and stored.steps is not None:
                        eff_steps = stored.steps
                    if eff_timeout is None and stored.duration_seconds is not None:
                        eff_timeout = stored.duration_seconds
                    if not eff_guidance and stored.guidance:
                        eff_guidance = stored.guidance
            except Exception:
                pass

        try:
            LG.info(
                "[sim-config/start] parsed steps=%s, timeout_seconds=%s; effective steps=%s, timeout_seconds=%s; core_text=%r",
                parsed.steps,
                parsed.timeout_seconds,
                eff_steps,
                eff_timeout,
                core_text,
            )
        except Exception:
            pass
        # Print immediately so the user sees what was captured and what will be used
        try:
            print("🧭 Simulation:")
            # Always show the exact text that will be sent to execute
            print(f"   📝 Parsed text: {core_text}")
            # Only show values that will be used; annotate source
            if eff_steps is not None:
                origin = "parsed" if parsed.steps is not None else "default"
                print(f"   🔢 Steps ({origin}): {eff_steps}")
            if eff_timeout is not None:
                origin = "parsed" if parsed.timeout_seconds is not None else "default"
                print(f"   ⏱️ Timeout ({origin}): {eff_timeout}s")
            if eff_guidance:
                print(f"   🧠 Guidance: {eff_guidance}")
            if eff_steps is None and eff_timeout is None and not eff_guidance:
                print("   ℹ️ No step limit, no timeout, no guidance")
        except Exception:
            pass

        # Apply sandbox-only per-task simulation monkey-patch for the lifetime of this execute call
        per_call = SimulationParams(
            steps=eff_steps,
            duration_seconds=eff_timeout,
            guidance=eff_guidance,
            one_shot=False,
        )
        _restore_patch = apply_per_task_simulation_patch(
            per_call_overrides=per_call,
            log_mode="print",
        )

        handle = await ts.execute(
            core_text,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clar_up_q,
            clarification_down_q=clar_down_q,
        )

        async def _restore_actor_when_done():
            try:
                await handle.result()
            except Exception:
                # Restore even if the call was stopped/cancelled/errored
                pass
            try:
                _restore_patch()
            except Exception:
                pass

        # Fire-and-forget restoration
        asyncio.create_task(_restore_actor_when_done())

        # Consume one-shot rule for numeric id if applicable
        try:
            if stripped_id.isdigit():
                tid_int = int(stripped_id)
                params = SIMULATION_PLANS.resolve_for_task_id(tid_int)
                if params and params.one_shot:
                    SIMULATION_PLANS.consume_one_shot_for(
                        SimulationSelector(by_task_id=tid_int),
                        task_id=tid_int,
                    )
        except Exception:
            pass
    else:  # ask
        handle, clar_up_q, clar_down_q = (
            await call_manager_with_optional_clarifications(
                ts.ask,
                raw,
                parent_chat_context=parent_chat_context,
                return_reasoning_steps=show_steps,
                clarifications_enabled=clarifications_enabled,
            )
        )

    # Speak an acknowledgement if voice mode is on so users know work began
    if enable_voice:
        try:
            _speak("Working on it.")
        except Exception:
            pass

    return (
        "execute" if intent.action == "start" else intent.action,
        handle,
        clar_up_q,
        clar_down_q,
    )


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

    ts = TaskScheduler(actor=SimulatedActor(log_mode="print"))
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
            _kind, _handle, _clar_up, _clar_down = await _dispatch_with_context(
                ts,
                raw,
                show_steps=args.debug,
                parent_chat_context=list(chat_history),
                clarifications_enabled=not args.no_clarifications,
                enable_voice=bool(args.voice),
            )
            chat_history.append({"role": "user", "content": raw})
            if args.voice:
                _speak("Let me take a look, give me a moment")
                _wait_tts_end()

            print(
                _steer_hint(
                    voice_enabled=bool(args.voice),
                ),
            )
            answer = await _await_with_interrupt(
                _handle,
                enable_voice_steering=bool(args.voice),
                clarification_up_q=_clar_up,
                clarification_down_q=_clar_down,
                clarifications_enabled=not args.no_clarifications,
                chat_context=list(chat_history),
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
