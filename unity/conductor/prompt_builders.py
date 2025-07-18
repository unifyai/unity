from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from typing import Dict, Callable

from ..task_scheduler.types.task import Task
from ..common.llm_helpers import SteerableToolHandle, class_api_overview
from ..memory_manager.rolling_activity import get_broader_context
from ..common.prompt_helpers import clarification_guidance

# ───────────────────────────────────── helpers ─────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return a *compact* mapping of {tool-name: '(<argspec>)'}."""
    return {n: str(inspect.signature(fn)) for n, fn in tools.items()}


def _now() -> str:
    """Current UTC timestamp in a friendly format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ─────────────────────────────────────────────────────────────────────────────
# Shared historic activity snippet
# ─────────────────────────────────────────────────────────────────────────────


def _rolling_activity_section() -> str:
    """Return a markdown summary of historic activity from cache."""

    try:
        overview = get_broader_context()
    except Exception:
        return ""

    if not overview:
        return ""

    return "\n".join(
        [
            "Historic Activity Overview",
            "---------------------------",
            "Below is a summary of the agent's historic activity (tasks, contacts, knowledge, transcripts, etc.).",
            "Some parts may be useful context for the current task while others might not – use your judgement.",
            "",
            overview,
            "",
        ],
    )


# ───────────────────────────────────── builders ─────────────────────────────────────


def build_ask_prompt(
    tools: Dict[str, Callable],
    *,
    include_activity: bool = True,
) -> str:
    """Dynamic **system** prompt for `Conductor.ask`."""
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    return "\n".join(
        [
            activity_block,
            "You are an assistant specialising in **read-only questions** about tasks,",
            "contacts, transcripts and the knowledge-base.  Interact with the tools",
            "below *step-by-step* until you can answer concisely.",
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            "Task schema (for filters):",
            json.dumps(Task.model_json_schema(), indent=4),
            "",
            "SteerableToolHandle class:",
            class_api_overview(SteerableToolHandle),
            "",
            f"Current UTC time is {_now()}.",
            clar_section,
            "",
        ],
    )


def build_request_prompt(
    tools: Dict[str, Callable],
    *,
    include_activity: bool = True,
) -> str:
    """Dynamic **system** prompt for `Conductor.request`."""
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    return "\n".join(
        [
            activity_block,
            "You have **full read-write control** over tasks, contacts, transcripts",
            "and the knowledge-base. Use *only* the tools supplied – never invent",
            "your own. Call them iteratively until the user's request is completely",
            "fulfilled, verifying state after each mutation." "",
            "If you are asked to perform a task, you should *always* proceed as follows:",
            "- Check if this task already exists via TaskScheduler.ask",
            "- Add a new task *if it doesn't already exist* in the task list via TaskScheduler.update",
            "- Start the task via TaskScheduler.execute_task if the user wants you to start now.",
            "  Otherwise set the scheduled start date/time when calling TaskScheduler.update above."
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            "Task schema:",
            json.dumps(Task.model_json_schema(), indent=4),
            "",
            "SteerableToolHandle class:",
            class_api_overview(SteerableToolHandle),
            "",
            f"Current UTC time is {_now()}.",
            clar_section,
            "",
        ],
    )
