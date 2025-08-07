from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from typing import Dict, Callable

from .types.task import Task
from ..memory_manager.broader_context import get_broader_context
from ..common.prompt_helpers import clarification_guidance

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {name: '(<argspec>)', …} for prettier JSON dumps."""
    return {n: str(inspect.signature(fn)) for n, fn in tools.items()}


def _now() -> str:
    """Current UTC timestamp in a compact, human-readable form."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ─────────────────────────────────────────────────────────────────────────────
# Shared historic activity snippet
# ─────────────────────────────────────────────────────────────────────────────


def _rolling_activity_section() -> str:
    """Return a markdown summary of the agent's historic activity from cache."""

    try:
        overview = get_broader_context()
    except Exception:  # pragma: no cover
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


# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def build_ask_prompt(
    tools: Dict[str, Callable],
    *,
    include_activity: bool = True,
) -> str:
    """
    Build the **system** prompt for the `ask` method.

    *Never* hard-codes the number, names or argument-specs of tools – those are
    injected live from the supplied *tools* dict.
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    return "\n".join(
        [
            activity_block,
            "You are an assistant specialising in **answering questions about the task list**.",
            "Interact with the read-only tools provided (see below) to gather whatever",
            "information you need, *step-by-step*.  When you have everything, respond",
            "with a concise, final answer.",
            "Please *always* mention the relevant task id(s) in your response.",
            "The user will almost certainly require the task ids in order to do anything meaningful with your answer.",
            "If the question refers to another person (such as communication oriented tasks), then we should call the `ContactManager.ask` tool first to ensure we have the full context on the person/people involved.",
            "Similarly, if a task refers to one or multiple 'contact_id' values (as part of the trigger for example), then we should also query 'ContactManager.ask' to learn more details about these contact(s)."
            "If the task is not specifically related to one or multiple people, then there is no need to query `ContactManager.ask`."
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            "Task schema:",
            json.dumps(Task.model_json_schema(), indent=4),
            "",
            f"Current UTC time is {_now()}.",
            clar_section,
            "",
        ],
    )


def build_update_prompt(
    tools: Dict[str, Callable],
    *,
    include_activity: bool = True,
) -> str:
    """
    Build the **system** prompt for the `update` method.
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    return "\n".join(
        [
            activity_block,
            "You are an assistant responsible for **creating and updating tasks**.",
            "Use the tools supplied *only* – never invent your own – until the task",
            "list fully reflects the user's intent.",
            "If a any tasks were created or updated in the process,",
            "then please *always* include these task id(s) in your final response.",
            "Whenever your update requires contact information (for example, building a trigger that should fire when specific contact(s) call), first call the `ContactManager.ask` tool to retrieve that contact id(s) and then insert into the trigger.",
            "",
            "If tasks are given in a *numbered order*, then please assume that these tasks "
            "should be *queued* in that *same order* unless explicitly stated otherwise.",
            "Having their `start_at` in ascending order is not enough, ",
            "tasks which are to be completed *sequentially* should also be *explicitly* queued."
            "This ensures smooth task progression, even if schedules overrun and `start_at` times"
            "are therefore not all adhered to."
            "",
            "ALWAYS check the existing tasks BEFORE creating new ones."
            "If you are asked to re-order or reschedule tasks, this is especially important. They likely already exist."
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            "Task schema:",
            json.dumps(Task.model_json_schema(), indent=4),
            "",
            f"Current UTC time is {_now()}.",
            clar_section,
            "",
        ],
    )


def build_execute_task_prompt(
    tools: Dict[str, Callable],
) -> str:
    """
    Build the **system** prompt for the `execute_task` method.
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    return "\n".join(
        [
            "You are an assistant that **starts tasks on demand**."
            "  The task referred to in the user's request may or may not already"
            "  exist in the task list.",
            "",
            "Use the tools below, step-by-step, following these rules:",
            "",
            "A. If the request contains a *numeric task_id*:",
            "   • **First** call `ask` (or another suitable read-only tool) to confirm the task exists.",
            "   • If exactly one matching task is found → call `execute_task_by_id`.",
            "   • If the id is **unknown** (zero results) → call `request_clarification` to ask the human whether to create a new task or provide a different reference.  Do **NOT** call `execute_task_by_id` when the task cannot be confirmed.",
            "",
            "B. If **no numeric id** is given:",
            "   1. Call `ask` with the free-form description to search for matching task(s).",
            "   2. Based on the result:",
            "      • **Exactly one** clear match → call `execute_task_by_id` with that id, do *not* bother the user with a `request_clarification` call.",
            "      • **Multiple / ambiguous** matches → call `request_clarification` so the user can disambiguate, only do so if it's *genuinely* unclear.",
            "      • **No match**:",
            "          – If it's ambiguous whether a task should be created/updated → `request_clarification`.",
            "          – If it is obvious we need to *create* a new task or *update* an existing one → call `update` to create/update the task, **then** call `execute_task_by_id` with the returned/newly discovered id.",
            "",
            "C. After creating a task with `update`, you may either read its id from the update response *or* call `ask` again to retrieve it before starting it.",
            "",
            "Respond *only* with tool calls until *after* `execute_task_by_id` returns.  You **must not** attempt `execute_task_by_id` until you are certain the referenced task exists. Once the task has started you may reply DONE.",
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulated helper
# ─────────────────────────────────────────────────────────────────────────────


def build_simulated_method_prompt(
    method: str,
    user_request: str,
    parent_chat_context: list[dict] | None = None,
) -> str:
    """Return instruction prompt for the *simulated* TaskScheduler."""
    import json

    preamble = f"On this turn you are simulating the '{method}' method."
    if method.lower() == "ask":
        behaviour = (
            "Please always *answer* the question with an imaginary but plausible response, "
            "mentioning the relevant task id(s). Do NOT ask for clarification or describe your process."
        )
    elif method.lower() == "update":
        behaviour = (
            "Please always act as though the task list has been updated **successfully**. "
            "Respond in past tense and include any created/updated task id(s) in your reply."
        )
    else:
        behaviour = "Provide a final response as though the requested operation has already completed (past tense)."

    parts: list[str] = [preamble, behaviour, "", f"The user input is:\n{user_request}"]
    if parent_chat_context:
        parts.append(
            f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}",
        )

    return "\n".join(parts)
