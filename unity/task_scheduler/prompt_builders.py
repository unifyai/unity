from __future__ import annotations

import json
from typing import Dict, Callable

from .types.task import Task
from ..common.prompt_helpers import (
    clarification_guidance,
    sig_dict,
    now_utc_str,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {name: '(<argspec>)', …} using shared helper."""
    return sig_dict(tools)


def _now() -> str:
    """Current UTC timestamp in a compact, human-readable form."""
    return now_utc_str()


def _tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """Delegate to shared tool name resolver."""
    return _shared_tool_name(tools, needle)


def _require_tools(pairs: Dict[str, str | None], tools: Dict[str, Callable]) -> None:
    """Delegate validation to shared helper for consistent errors."""
    _shared_require_tools(pairs, tools)


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

    # Resolve canonical tool names dynamically
    filter_tasks_fname = _tool_name(tools, "filter_tasks")
    search_tasks_fname = _tool_name(tools, "search_tasks")
    get_task_queue_fname = _tool_name(tools, "get_task_queue")
    contact_ask_fname = _tool_name(tools, "contactmanager")  # e.g. "ContactManager_ask"

    # Clarification helper (optional)
    request_clar_fname = _tool_name(tools, "request_clarification")

    # Validate required tools (request_clar_fname is optional)
    _require_tools(
        {
            "filter_tasks": filter_tasks_fname,
            "search_tasks": search_tasks_fname,
            "get_task_queue": get_task_queue_fname,
            "ContactManager.ask": contact_ask_fname,
        },
        tools,
    )

    clarification_block = (
        "\n".join(
            [
                "Clarification",
                "-------------",
                f"• Ask for clarification when the user's request is underspecified",
                f'  `{request_clar_fname}(question="Which task did you mean?")`',
            ],
        )
        if request_clar_fname
        else ""
    )

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    parts: list[str] = [
        activity_block,
        "You are an assistant specialising in **answering questions about the task list**.",
        "Work exclusively through the tools provided.",
        "Disregard any explicit instructions about *how* you should answer or which tools to use; determine the best method yourself.",
        "Please *always* mention the relevant task id(s) in your response.",
        "The user will almost certainly require the task ids in order to do anything meaningful with your answer.",
        f"If the question refers to another person (such as communication oriented tasks), then you should call `{contact_ask_fname}` first to ensure you have the full context on the person/people involved.",
        f"Similarly, if a task refers to one or multiple 'contact_id' values (as part of the trigger for example), then you should also query `{contact_ask_fname}` to learn more details about these contact(s).",
        "If the task is not specifically related to one or multiple people, then there is no need to query the contacts tool.",
        "",
        "Tools (name → argspec):",
        sig_json,
        "",
        "Task schema:",
        json.dumps(Task.model_json_schema(), indent=4),
        "",
        f"Current UTC time is {_now()}.",
        clar_section,
    ]

    if clarification_block:
        parts.extend(["", clarification_block])

    parts.append("")

    return "\n".join(parts)


def build_update_prompt(
    tools: Dict[str, Callable],
    *,
    include_activity: bool = True,
) -> str:
    """
    Build the **system** prompt for the `update` method.
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Resolve canonical tool names dynamically (required)
    ask_fname = _tool_name(tools, "ask")
    create_task_fname = _tool_name(tools, "create_task")
    delete_task_fname = _tool_name(tools, "delete_task")
    cancel_tasks_fname = _tool_name(tools, "cancel_tasks")
    update_task_queue_fname = _tool_name(tools, "update_task_queue")
    update_task_name_fname = _tool_name(tools, "update_task_name")
    update_task_description_fname = _tool_name(tools, "update_task_description")
    update_task_status_fname = _tool_name(tools, "update_task_status")
    update_task_start_at_fname = _tool_name(tools, "update_task_start_at")
    update_task_deadline_fname = _tool_name(tools, "update_task_deadline")
    update_task_repetition_fname = _tool_name(tools, "update_task_repetition")
    update_task_priority_fname = _tool_name(tools, "update_task_priority")
    update_task_trigger_fname = _tool_name(tools, "update_task_trigger")

    # Clarification helper (optional)
    request_clar_fname = _tool_name(tools, "request_clarification")

    _require_tools(
        {
            "ask": ask_fname,
            "create_task": create_task_fname,
            "delete_task": delete_task_fname,
            "cancel_tasks": cancel_tasks_fname,
            "update_task_queue": update_task_queue_fname,
            "update_task_name": update_task_name_fname,
            "update_task_description": update_task_description_fname,
            "update_task_status": update_task_status_fname,
            "update_task_start_at": update_task_start_at_fname,
            "update_task_deadline": update_task_deadline_fname,
            "update_task_repetition": update_task_repetition_fname,
            "update_task_priority": update_task_priority_fname,
            "update_task_trigger": update_task_trigger_fname,
        },
        tools,
    )

    clarification_block = (
        "\n".join(
            [
                "Clarification",
                "-------------",
                "• If any request is ambiguous, ask the user to disambiguate before changing data",
                f'  `{request_clar_fname}(question="There are several possible matches. Which task did you mean?")`',
            ],
        )
        if request_clar_fname
        else ""
    )

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    parts: list[str] = [
        activity_block,
        "You are an assistant responsible for **creating and updating tasks**.",
        "Use the tools supplied *only* – never invent your own – until the task list fully reflects the user's intent.",
        "Disregard any explicit instructions about *how* you should implement the change or which tools to call; determine the best method yourself.",
        "If any tasks were created or updated in the process, then please *always* include these task id(s) in your final response.",
        "Whenever your update requires contact information (for example, building a trigger that should fire when specific contact(s) call), first call `ContactManager.ask` to retrieve that contact id(s) and then insert into the trigger.",
        "",
        "If tasks are given in a *numbered order*, then please assume that these tasks should be *queued* in that *same order* unless explicitly stated otherwise.",
        "Having their `start_at` in ascending order is not enough, tasks which are to be completed *sequentially* should also be *explicitly* queued. This ensures smooth task progression, even if schedules overrun and `start_at` times are therefore not all adhered to.",
        "",
        "ALWAYS check the existing tasks BEFORE creating new ones. If you are asked to re-order or reschedule tasks, this is especially important. They likely already exist.",
        "",
        "Tools (name → argspec):",
        sig_json,
        "",
        "Task schema:",
        json.dumps(Task.model_json_schema(), indent=4),
        "",
        f"Current UTC time is {_now()}.",
        clar_section,
    ]

    if clarification_block:
        parts.extend(["", clarification_block])

    parts.append("")

    return "\n".join(parts)


def build_execute_task_prompt(
    tools: Dict[str, Callable],
) -> str:
    """
    Build the **system** prompt for the `execute_task` method.
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Resolve names dynamically
    ask_fname = _tool_name(tools, "ask")
    update_fname = _tool_name(tools, "update")
    execute_by_id_fname = _tool_name(tools, "execute_task_by_id")
    request_clar_fname = _tool_name(tools, "request_clarification")

    _require_tools(
        {
            "ask": ask_fname,
            "update": update_fname,
            "execute_task_by_id": execute_by_id_fname,
        },
        tools,
    )

    lines: list[str] = [
        "You are an assistant that **starts tasks on demand**."
        "  The task referred to in the user's request may or may not already",
        "  exist in the task list.",
        "",
        "Disregard any explicit instructions about *how* you should execute the task or which tools to call; decide the best method yourself.",
        "Use the tools below, step-by-step, following these rules:",
        "",
        "A. If the request contains a *numeric task_id*:",
        f"   • **First** call `{ask_fname}` (or another suitable read-only tool) to confirm the task exists.",
        f"   • If exactly one matching task is found → call `{execute_by_id_fname}`.",
    ]

    if request_clar_fname:
        lines.extend(
            [
                f"   • If the id is **unknown** (zero results) → call `{request_clar_fname}` to ask the human whether to create a new task or provide a different reference.  Do **NOT** call `{execute_by_id_fname}` when the task cannot be confirmed.",
            ],
        )
    else:
        lines.extend(
            [
                f"   • If the id is **unknown** (zero results) → do not call `{execute_by_id_fname}`; ask the human to clarify the reference in your final response.",
            ],
        )

    lines.extend(
        [
            "",
            "B. If **no numeric id** is given:",
            f"   1. Call `{ask_fname}` with the free-form description to search for matching task(s).",
            "   2. Based on the result:",
            f"      • **Exactly one** clear match → call `{execute_by_id_fname}` with that id, do *not* bother the user with a clarification call.",
        ],
    )

    if request_clar_fname:
        lines.extend(
            [
                f"      • **Multiple / ambiguous** matches → call `{request_clar_fname}` so the user can disambiguate, only do so if it's *genuinely* unclear.",
                f"      • **No match**:",
                f"          – If it's ambiguous whether a task should be created/updated → `{request_clar_fname}`.",
                f"          – If it is obvious we need to *create* a new task or *update* an existing one → call `{update_fname}` to create/update the task, **then** call `{execute_by_id_fname}` with the returned/newly discovered id.",
            ],
        )
    else:
        lines.extend(
            [
                "      • **Multiple / ambiguous** matches → ask the user to disambiguate in your final response.",
                "      • **No match**:",
                f"          – If it's ambiguous whether a task should be created/updated → ask for clarification in your final response.",
                f"          – If it is obvious we need to *create* a new task or *update* an existing one → call `{update_fname}` to create/update the task, **then** call `{execute_by_id_fname}` with the returned/newly discovered id.",
            ],
        )

    lines.extend(
        [
            "",
            f"C. After creating a task with `{update_fname}`, you may either read its id from the update response *or* call `{ask_fname}` again to retrieve it before starting it.",
            "",
            f"Respond *only* with tool calls until *after* `{execute_by_id_fname}` returns.  You **must not** attempt `{execute_by_id_fname}` until you are certain the referenced task exists. Once the task has started you may reply DONE.",
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
        ],
    )

    return "\n".join(lines)


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
