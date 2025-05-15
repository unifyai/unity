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
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, List
import logging
import logging.config

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

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

handlers = {}
active = []

# Configure logging to only allow our LOGGER to output
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "default": {
                "format": "%(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "default",
            },
        },
        "root": {"level": "CRITICAL", "handlers": []},  # Block all root logging
        "loggers": {
            "unity": {  # Only allow our LOGGER to output
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            # Block all other loggers
            "": {  # Catch-all for unnamed loggers
                "level": "CRITICAL",
                "handlers": [],
                "propagate": False,
            },
        },
    },
)

from constants import LOGGER


# Initialize task list manager globally
TLM = TaskListManager()
TLM.start()


def _generate_project_name(scenario_type: str, theme: str = None) -> str:
    """Generate a unique project name based on scenario and current timestamp."""
    timestamp = datetime.now().strftime("%H-%M-%S_%d-%m-%Y")

    if scenario_type == "fixed":
        base_name = "SimpleTaskList"
    else:  # LLM scenario
        # Use theme from LLM if available, otherwise a generic name
        base_name = theme if theme else "LLMGeneratedTaskList"
        # Sanitize the base_name by removing special characters
        base_name = "".join(c for c in base_name if c.isalnum() or c in [" ", "-", "_"])
        base_name = base_name.replace(" ", "_")

    # ToDo: re-add nested timestamp once this task [{link}] is fixed
    return f"{base_name}"  # /{timestamp}"


def _seed_fixed(tlm: TaskListManager) -> None:
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


def _seed_llm(tlm: TaskListManager) -> str:
    """Ask an LLM to craft a *cohesive* scenario with >100 tasks.

    Expected JSON schema returned by the model:

        {
          "theme": "ACME car dealership – Q4 outreach",    # optional meta
          "tasks": [
            {
              "name": "Call lead – Ford Fiesta",
              "description": "Phone Marco Rossi to discuss financing options",
              "priority": "high",
              "queue_group": "sales",          # tasks with same group are chained
              "queue_position": 0,              # lower number == earlier in chain
              "status": "queued" | "active" | "paused" | "scheduled",
              "start_time": "2025-06-10T09:00:00Z"   # optional ISO (only for scheduled)
            },
            … (100-200 tasks) …
          ]
        }

    We interpret every *queue_group* separately, ordering by *queue_position*
    to create a linked list via the Schedule prev_task/next_task pointers.

    Returns the theme from the LLM response if available.
    """

    prompt = (
        "Generate a realistic task list for a small business. "
        "Pick a coherent *theme* (e.g. a car dealership following up leads, "
        "or a property agency, or a software release checklist). "
        "Create between 110 and 140 tasks that belong to **several logical "
        "queues** (e.g. 'sales-calls', 'paperwork', 'maintenance'). "
        "For each task output: name, description, status, optional priority, "
        "a queue_group label, queue_position (int, starting at 0 in each group) "
        "and, when status=='scheduled', a start_time in ISO-8601 UTC. "
        "Return JSON with a top-level key 'tasks' containing the list. Do NOT "
        "include any text outside the JSON literal."
    )

    client = unify.Unify("gpt-4o@openai", cache=True, traced=True)
    client.set_system_message(prompt)
    raw = client.generate("Produce scenario").strip()

    theme = None
    try:
        payload = json.loads(raw)
        tasks_data = payload["tasks"]
        theme = payload.get("theme")
    except Exception:
        print("LLM scenario generation failed – using fixed sample instead.")
        _seed_fixed(tlm)
        return theme

    # Sort tasks within each queue_group by queue_position to build linkage
    groups: Dict[str, List[dict]] = {}
    for t in tasks_data:
        groups.setdefault(t.get("queue_group", "default"), []).append(t)

    for grp in groups.values():
        grp.sort(key=lambda d: d.get("queue_position", 0))

    id_map: Dict[tuple[str, int], int] = {}

    # First pass – create tasks without schedule linkage
    for grp_name, grp in groups.items():
        for idx, entry in enumerate(grp):
            kwargs = {
                "name": entry["name"],
                "description": entry["description"],
                "status": entry.get("status", "queued"),
                "priority": entry.get("priority", Priority.normal),
            }

            start_time = entry.get("start_time")
            if start_time:
                kwargs["schedule"] = Schedule(
                    start_time=start_time,
                    prev_task=None,
                    next_task=None,
                )

            new_id = tlm._create_task(**kwargs)
            id_map[(grp_name, idx)] = new_id

    # Second pass – update schedule links within each group
    for grp_name, grp in groups.items():
        for idx, _ in enumerate(grp):
            cur_id = id_map[(grp_name, idx)]
            prev_id = id_map.get((grp_name, idx - 1)) if idx > 0 else None
            next_id = id_map.get((grp_name, idx + 1)) if idx < len(grp) - 1 else None

            sched_payload = {"prev_task": prev_id, "next_task": next_id}
            tlm._update_task_status(
                task_ids=cur_id,
                new_status=tlm._search(filter=f"task_id=={cur_id}")[0]["status"],
            )  # noop to ensure log exists
            unify.update_logs(
                context="Tasks",
                logs=tlm._get_logs_by_task_ids(task_ids=cur_id),
                entries={"schedule": sched_payload},
                overwrite=True,
            )

    return theme


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


# Voice Mode


class VoiceAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice AI assistant, responsible for listening to the user and then using the `ask` and `update` tools gain infomration about the status of the tasks in the backend, and to update the tasks in the backend. You do not have direct access to the tasks or the schema used to store them, and so the ask and request tools simply take english-language input, explaining your request as clearly and unambiguously as possible.",
        )

    @function_tool()
    async def ask(self, question: str) -> str:
        """Ask a question about the tasks."""
        return TLM.ask(question, log_tool_steps=True)

    @function_tool()
    async def update(self, request: str) -> str:
        """Update the tasks in some manner."""
        return TLM.update(request, log_tool_steps=True)


async def voice_entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=VoiceAssistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance.",
    )


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="TaskListManager interactive sandbox")
    parser.add_argument(
        "--silent",
        "-s",
        action="store_true",
        help="suppress tool logs",
    )
    parser.add_argument(
        "--scenario",
        choices=["fixed", "llm"],
        default="fixed",
        help="starting task set to load",
    )
    parser.add_argument(
        "--voice",
        "-v",
        action="store_true",
        help="use voice mode",
    )
    parser.add_argument(
        "--new",
        "-n",
        action="store_true",
        help="Create an new scenario, erasing the old one",
    )
    args = parser.parse_args()
    silent = args.silent
    scenario_type = args.scenario
    voice_mode = args.voice
    new_scenario = args.new

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

    # Generate initial project name
    project_name = _generate_project_name(scenario_type)

    # Activate the project with dynamic name
    unify.activate(project_name)

    # Create the Tasks context
    unify.set_context("Tasks", overwrite=new_scenario)

    # Seed with data
    theme = None
    if new_scenario:
        LOGGER.info(f"⏳ Adding data to task environment...")
        if scenario_type == "llm":
            theme = _seed_llm(TLM)
            if theme:
                # Update project name with the theme if it's available
                new_project_name = _generate_project_name(scenario_type, theme)
                # Only switch if names are different
                if new_project_name != project_name:
                    # Re-activate with the themed project name
                    unify.activate(new_project_name)
                    project_name = new_project_name
        else:
            _seed_fixed(TLM)
        LOGGER.info(f"✅ Data added")

    print(
        f"TaskListManager sandbox using project '{project_name}' – type natural language. "
        f"Prefix with 'ask:' or 'update:' to specify. 'quit' to exit.\n"
        f"Verbose reasoning is {('ON' if not silent else 'OFF')} by default (add --silent to disable).\n",
    )

    if voice_mode:
        # Save original args and clear sys.argv before calling agents CLI
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], "console"]
        agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=voice_entrypoint))
        # Restore original args if needed
        sys.argv = original_argv
    else:
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

            kind, result, _ = _dispatch(TLM, line, show_steps=not silent)
            print(f"[{kind}] => {result}\n")


if __name__ == "__main__":
    main()
