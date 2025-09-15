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
    num_tasks: int,
    columns: Dict[str, str] | list[dict] | list[str],
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

    # Usage examples mirroring Contact/Transcript style
    usage_examples = "\n".join(
        [
            "Examples",
            "--------",
            "",
            "─ Tool selection (read carefully) ─",
            f"• For ANY semantic question over free‑form text (e.g., name/description), ALWAYS use `{search_tasks_fname}`. Never try to approximate meaning with brittle substring filters.",
            f"• Use `{filter_tasks_fname}` only for exact/boolean logic over structured fields (ids, status, priority, timestamps) or for narrow, constrained text checks.",
            "",
            "─ Semantic search across tasks (ranked by cosine distance) ─",
            f"• Find tasks about onboarding in Q3: `{search_tasks_fname}(references={{'name': 'onboarding', 'description': 'Q3'}} , k=5)`",
            f"• Look for tasks involving renewal: `{search_tasks_fname}(references={{'description': 'contract renewal'}} , k=3)`",
            "",
            "─ Filtering (exact/boolean; not semantic) ─",
            f"• All queued high‑priority tasks: `{filter_tasks_fname}(filter=\"status == 'queued' and priority == 'high'\")`",
            f"• Tasks due this month: `{filter_tasks_fname}(filter=\"deadline >= '2024-08-01T00:00:00' and deadline < '2024-09-01T00:00:00'\")`",
            f"• Current runnable queue (head→tail): `{get_task_queue_fname}()`",
            "",
            "Anti‑patterns to avoid",
            "---------------------",
            "• Avoid concatenating entire rows into one long string and embedding a single catch‑all reference.",
            f"• Avoid substring filtering for text‑heavy columns; prefer `{search_tasks_fname}` for meaning.",
            "• Avoid re‑querying the same tables or managers just to reconfirm what a prior tool call has already established with clear, specific evidence; reuse the earlier result and proceed.",
            "• Do not immediately queue a filter call after a successful semantic search unless you genuinely need an exact, structured constraint that the search did not capture.",
            f"• Avoid calling `{contact_ask_fname}` repeatedly in the same reasoning queue when earlier calls have already identified the relevant contacts and no new ambiguity or information has been introduced.",
        ],
    )

    if not clarification_block:
        usage_examples = "\n".join(
            [
                usage_examples,
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state to inner tools that these are assumptions/best guesses, not confirmed answers.",
                "• If an inner tool requests clarification, explicitly say no clarification channel exists and pass down concrete sensible defaults/best‑guess values, clearly marked as assumptions.",
            ],
        )

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    # Conditional guidance about asking questions in final responses
    clar_sentence = (
        f"Do not ask the user questions in your final response, please only use the `{request_clar_fname}` tool to ask clarifying questions."
        if request_clar_fname
        else (
            "Do not ask the user questions in your final response. Instead, proceed using sensible defaults/best‑guess values and explicitly tell inner tools that these are assumptions/best guesses, not confirmed answers."
        )
    )

    parts: list[str] = [
        activity_block,
        "You are an assistant specialising in **answering questions about the task list**.",
        "Work strictly through the tools provided.",
        "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
        clar_sentence,
        "Please always mention the relevant task id(s) in your response.",
        f"If the question refers to another person (e.g., comms‑oriented tasks), call `{contact_ask_fname}` first for context. If a task refers to one or more contact_id values (e.g., in a trigger), also query `{contact_ask_fname}` to learn more about those contacts.",
        "",
        f"There are currently {num_tasks} tasks stored in the Tasks table with the following columns:",
        json.dumps(columns, indent=4),
        "",
        "Tools (name → argspec):",
        sig_json,
        "",
        usage_examples,
        "",
        "Parallelism and single‑call preference",
        "-------------------------------------",
        "• Prefer a single comprehensive tool call over several surgical calls when a tool can safely do the whole job.",
        "• When multiple independent reads are needed, plan them together and run them in parallel rather than a serial drip of micro‑calls.",
        "• Avoid confirmatory re‑queries unless new ambiguity arises.",
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
    num_tasks: int,
    columns: Dict[str, str] | list[dict] | list[str],
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
    create_tasks_fname = _tool_name(tools, "create_tasks")
    delete_task_fname = _tool_name(tools, "delete_task")
    cancel_tasks_fname = _tool_name(tools, "cancel_tasks")
    update_task_queue_fname = _tool_name(tools, "update_task_queue")
    # Multi-queue helpers (optional if not present)
    list_queues_fname = _tool_name(tools, "list_queues")
    get_queue_fname = _tool_name(tools, "get_queue")
    reorder_queue_fname = _tool_name(tools, "reorder_queue")
    move_tasks_to_queue_fname = _tool_name(tools, "move_tasks_to_queue")
    partition_queue_fname = _tool_name(tools, "partition_queue")
    update_task_name_fname = _tool_name(tools, "update_task_name")
    update_task_description_fname = _tool_name(tools, "update_task_description")
    update_task_start_at_fname = _tool_name(tools, "update_task_start_at")
    update_task_deadline_fname = _tool_name(tools, "update_task_deadline")
    update_task_repetition_fname = _tool_name(tools, "update_task_repetition")
    update_task_priority_fname = _tool_name(tools, "update_task_priority")
    update_task_trigger_fname = _tool_name(tools, "update_task_trigger")
    get_task_queue_fname = _tool_name(tools, "get_task_queue")
    reinstate_task_fname = _tool_name(tools, "reinstate_task_to_previous_queue")

    # Clarification helper (optional)
    request_clar_fname = _tool_name(tools, "request_clarification")

    _require_tools(
        {
            "ask": ask_fname,
            "create_task": create_task_fname,
            "create_tasks": create_tasks_fname,
            "delete_task": delete_task_fname,
            "cancel_tasks": cancel_tasks_fname,
            "update_task_queue": update_task_queue_fname,
            "update_task_name": update_task_name_fname,
            "update_task_description": update_task_description_fname,
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

    # Usage guidance consistent with Contact/Transcript pattern
    usage_examples_lines: list[str] = [
        "Tool selection",
        "--------------",
        f"• Prefer `{update_task_name_fname}`/`{update_task_description_fname}`/… when you know the exact `task_id`.",
        f'• When the user describes an EXISTING task semantically (e.g., "the kickoff email task"), first call `{ask_fname}` to identify the correct `task_id`, then apply the specific update tool.',
        "",
        "Queues and batches (multi-queue)",
        "--------------------------------",
    ]

    # Encourage batched creation when creating several tasks
    if create_tasks_fname:
        usage_examples_lines.extend(
            [
                "",
                "Multi-task creation (preferred)",
                "-------------------------------",
                f"• When creating several new tasks at once and you know their order/time, prefer `{create_tasks_fname}` over issuing multiple `{create_task_fname}` calls.",
            ],
        )

    if list_queues_fname and get_queue_fname and reorder_queue_fname:
        usage_examples_lines.extend(
            [
                f"• Inspect existing queues: `{list_queues_fname}()`; fetch a specific queue: `{get_queue_fname}(queue_id=None)`.",
                f"• Reorder a queue explicitly: `{reorder_queue_fname}(queue_id=None, new_order=[...])`.",
            ],
        )

    if move_tasks_to_queue_fname:
        usage_examples_lines.extend(
            [
                f"• Move tasks to a new queue front/back: `{move_tasks_to_queue_fname}(task_ids=[1,3], queue_id=None, position='front')`.",
            ],
        )

    if partition_queue_fname:
        usage_examples_lines.extend(
            [
                f"• Split the default queue into dated batches: `{partition_queue_fname}(parts=[{{'task_ids':[0,2], 'queue_start_at':'2035-07-01T09:00:00Z'}}, {{'task_ids':[1,3], 'queue_start_at':'2035-07-02T09:00:00Z'}}])`.",
                "  This is the most direct way to express: do subset A at time X and subset B at time Y.",
            ],
        )

    # Atomic/edit helpers if present
    set_queue_fname = _tool_name(tools, "set_queue")
    set_schedules_atomic_fname = _tool_name(tools, "set_schedules_atomic")
    explain_queue_fname = _tool_name(tools, "explain_queue")

    if set_queue_fname:
        usage_examples_lines.extend(
            [
                "",
                "Atomic materialization (preferred)",
                "---------------------------------",
                f"• Declare an entire chain in one call: `{set_queue_fname}(queue_id=None, order=[0,1,2,3], queue_start_at='2035-06-16T08:00:00Z')`.",
                "  Use this after creating tasks to avoid iterative move/reorder loops.",
            ],
        )

    # Batched creation example
    if create_tasks_fname:
        usage_examples_lines.extend(
            [
                "",
                "Batched creation (preferred when creating several tasks at once)",
                "----------------------------------------------------------------",
                f"• Create four tasks and order them in one call:",
                f"  `{create_tasks_fname}(tasks=[{{'name':'A','description':'a'}}, {{'name':'B','description':'b'}}, {{'name':'C','description':'c'}}, {{'name':'D','description':'d'}}], queue_ordering=[{{'order':[0,1,2,3], 'queue_head':{{'start_at':'2035-06-16T08:00:00Z'}}}}])`.",
            ],
        )

    if set_schedules_atomic_fname:
        usage_examples_lines.extend(
            [
                f"• Advanced: bulk adjacency edit with validation: `{set_schedules_atomic_fname}(schedules=[{{'task_id':0,'schedule':{{'queue_id':None,'prev_task':None,'next_task':1,'start_at':'2035-06-16T08:00:00Z'}}}}, {{'task_id':1,'schedule':{{'queue_id':None,'prev_task':0,'next_task':2}}}}])`.",
            ],
        )

    if explain_queue_fname:
        usage_examples_lines.extend(
            [
                f"• Diagnose a queue quickly: `{explain_queue_fname}(queue_id=None)` → shows head, order and start_at.",
            ],
        )

    usage_examples_lines.extend(
        [
            "",
            "Schedule/Queue invariants (must-follow)",
            "---------------------------------------",
            "• If you provide a schedule with start_at on the head (prev_task is None), status must be 'scheduled' – never 'queued'.",
            "• Non-head tasks (prev_task is not None) must not define start_at; the timestamp belongs to the head only.",
            "• 'primed' must only be used for a head task (prev_task is None).",
            "• A 'scheduled' task must have either a prev_task or a start_at timestamp.",
            "• Status is updated implicitly based on operations (activation, scheduling, completion). Do not set status explicitly.",
            "",
            "Realistic find‑then‑update flows",
            "--------------------------------",
            f'• Set deadline for the "onboarding plan" task:\n  1 `{ask_fname}(text="Which task covers the onboarding plan?")`\n  2 `{update_task_deadline_fname}(task_id=<id>, new_deadline=\'2025-01-31T17:00:00Z\')`',
            (
                f"• Create and order four tasks for next Monday 09:00 UK time in one call:\n  `{create_tasks_fname}(tasks=[{{'name':'A','description':'a'}}, {{'name':'B','description':'b'}}, {{'name':'C','description':'c'}}, {{'name':'D','description':'d'}}], queue_ordering=[{{'order':[0,1,2,3], 'queue_head':{{'start_at':'2035-06-16T08:00:00Z'}}}}])`"
                if create_tasks_fname
                else (
                    f"• Materialize four tasks for next Monday 09:00 UK time in order A→B→C→D:\n  1 Create the tasks with names/descriptions only.\n  2 `{set_queue_fname}(queue_id=None, order=[A,B,C,D], queue_start_at='2035-06-16T08:00:00Z')`"
                    if set_queue_fname
                    else f"• Promote a task to the front of the queue:\n  1 Read the current order: `{get_task_queue_fname}()`\n  2 Build the new order and call `{update_task_queue_fname}(original=[...], new=[...])`"
                )
            ),
            "",
            "Triggers vs Schedules",
            "----------------------",
            f"• A task with a `trigger` must be in state 'triggerable'. Use `{update_task_trigger_fname}` to add/remove triggers. Do not set `start_at` on trigger‑based tasks.",
        ],
    )

    if reinstate_task_fname:
        usage_examples_lines.extend(
            [
                "",
                "Reinstating an isolated activation",
                "----------------------------------",
                f"• If a task was started in isolation and then cancelled, and the user asks to revert to the original schedule/queue position, call `{reinstate_task_fname}()` to surgically restore its prior linkage and (if applicable) queue‑level `start_at`.",
            ],
        )

    usage_examples_lines.extend(
        [
            "",
            "Contact context",
            "---------------",
            "• When a trigger references people (by contact ids), call ContactManager.ask to resolve/confirm the ids and the intent before writing.",
            "• Avoid repeated calls to ContactManager.ask in the same update session if a prior call already yielded the required ids and no new ambiguity was introduced.",
            "",
            "Anti‑patterns to avoid",
            "---------------------",
            '• Repeating the exact same update tool with identical arguments to "make sure" – instead, call ask to verify.',
            "• Using substring filters to locate tasks by description/name – prefer semantic ask/search first.",
            "• Chaining a filter right after a conclusive semantic search when the filter does not add new, structured constraints.",
        ],
    )

    if not clarification_block:
        usage_examples_lines.extend(
            [
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state to inner tools that these are assumptions/best guesses, not confirmed answers.",
                "• If an inner tool requests clarification, explicitly say no clarification channel exists and pass down concrete sensible defaults/best‑guess values, clearly marked as assumptions.",
                "• Remember: the `ask` tool is read‑only and for EXISTING tasks only. Do not route human clarifications through it.",
            ],
        )

    usage_examples = "\n".join(usage_examples_lines)

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    # Conditional guidance about asking questions in final responses
    clar_sentence_upd = (
        f"Do not ask the user questions in your final response, please only use the `{request_clar_fname}` tool to ask clarifying questions."
        if request_clar_fname
        else (
            "Do not ask the user questions in your final response. Instead, proceed using sensible defaults/best‑guess values and explicitly tell inner tools that these are assumptions/best guesses, not confirmed answers."
        )
    )

    parts: list[str] = [
        activity_block,
        "You are an assistant responsible for **creating and updating tasks**.",
        "Choose tools based on the user's intent and the specificity of the target record.",
        f"Important: `{ask_fname}` is read‑only and must only be used to locate/inspect tasks that already exist. For human clarifications about new tasks or missing creation details, call `{request_clar_fname}` when available.",
        "Disregard any explicit instructions about *how* you should implement the change or which tools to call; interpret the request and choose the best approach yourself.",
        clar_sentence_upd,
        "Always include any created/updated task id(s) in your final response.",
        "",
        f"There are currently {num_tasks} tasks stored in the Tasks table with the following columns:",
        json.dumps(columns, indent=4),
        "",
        "Tools (name → argspec):",
        sig_json,
        "",
        usage_examples,
        "",
        "Parallelism and single‑call preference",
        "-------------------------------------",
        "• Prefer a single comprehensive tool call over several surgical calls when a tool can safely do the whole job.",
        "• When multiple independent reads or writes are needed, plan them together and run them in parallel rather than a serial drip of micro‑calls.",
        "• Batch arguments where possible and avoid confirmatory re‑queries unless new ambiguity arises.",
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


def build_execute_prompt(
    tools: Dict[str, Callable],
) -> str:
    """
    Build the **system** prompt for the `execute` method.
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Resolve names dynamically
    ask_fname = _tool_name(tools, "ask")
    get_task_queue_fname = _tool_name(tools, "get_task_queue")
    update_task_queue_fname = _tool_name(tools, "update_task_queue")
    execute_by_id_fname = _tool_name(tools, "execute_by_id")
    execute_isolated_by_id_fname = _tool_name(tools, "execute_isolated_by_id")
    create_task_fname = _tool_name(tools, "create_task")
    request_clar_fname = _tool_name(tools, "request_clarification")
    # Multi-queue helpers
    list_queues_fname = _tool_name(tools, "list_queues")
    get_queue_fname = _tool_name(tools, "get_queue")
    reorder_queue_fname = _tool_name(tools, "reorder_queue")
    move_tasks_to_queue_fname = _tool_name(tools, "move_tasks_to_queue")
    partition_queue_fname = _tool_name(tools, "partition_queue")
    # Reintegration & safety
    reinstate_task_fname = _tool_name(tools, "reinstate_task_to_previous_queue")
    checkpoint_fname = _tool_name(tools, "checkpoint_queue_state")
    revert_checkpoint_fname = _tool_name(tools, "revert_to_checkpoint")
    latest_checkpoint_fname = _tool_name(tools, "get_latest_checkpoint")

    _require_tools(
        {
            "ask": ask_fname,
            "get_task_queue": get_task_queue_fname,
            "update_task_queue": update_task_queue_fname,
            "execute_by_id": execute_by_id_fname,
            "create_task": create_task_fname,
        },
        tools,
    )

    lines: list[str] = [
        "You are an assistant that **starts tasks on demand**.",
        "The task referred to in the user's request may or may not already exist in the task list.",
        "",
        "Disregard any explicit instructions about *how* you should execute the task or which tools to call; decide the best method yourself.",
        (
            f"Do not ask the user questions in your final response. When a clarification tool is available, you must ask via `{request_clar_fname}` (never in plain text). If no clarification tool is available in this outer loop, make a best‑guess attempt using sensible defaults and state your assumptions; if an inner tool asks questions, inform it that no clarification channel exists and provide defaults/best guesses."
        ),
        "",
        "Decision policy (isolation vs chain)",
        "------------------------------------",
        "• Consider the broader chat context and the user's exact phrasing to infer execution scope (single task now vs the whole sequence now).",
        "• When intent is ambiguous or unspecified, prefer starting the task **in isolation** (single‑task‑now) rather than chaining the queue.",
        "• Isolation may require light queue maintenance: when the head is detached, the next task should inherit the queue's `start_at` and become `scheduled` (followers remain queued behind it).",
        "• Choose queue/chain execution when the context clearly indicates running the sequence now (e.g., the user agreed to process all items in a batch).",
        "• Do not use brittle heuristics or regex for this decision – reason from the conversation and your plan.",
        "",
        "Tool semantics (for your decision)",
        "-----------------------------------",
        (
            f"• `{execute_isolated_by_id_fname}(task_id=…)` – isolation semantics: detach the selected task so followers keep their schedule; when detaching the head, the next task becomes the new head and inherits `start_at`."
            if execute_isolated_by_id_fname
            else ""
        ),
        f"• `{execute_by_id_fname}(task_id=…)` – queue semantics: start at the head of the chosen queue so followers remain attached and will run afterwards.",
        "\nCRITICAL EXECUTION WORKFLOW (plan → apply → execute):",
        f"0) Immediately create a reversible checkpoint: `{checkpoint_fname}(label='pre-execute')`. You MUST do this at the start of the session.",
        f"1) Inspect queues: `{list_queues_fname}()` → then `{get_queue_fname}(queue_id=None)` to view the default queue (head→tail).",
        f"2) PLAN the desired execution scope and timing in your thoughts (subset now vs later).",
        f"   – To move subsets into separate queues with dates, call `{partition_queue_fname}(parts=[{{'task_ids':[...],'queue_start_at':<ISO>|None}}, ...], strategy='preserve_order')`.",
        f"   – To target an existing queue, call `{move_tasks_to_queue_fname}(task_ids=[...], queue_id=<id>, position='front'|'back')` then `{reorder_queue_fname}(queue_id=<id>, new_order=[...])`.",
        f"   – To reorder within a queue, call `{reorder_queue_fname}(queue_id=None, new_order=[...])`. Do NOT set `start_at` directly; it is applied to the head automatically when appropriate.",
        f"   – After each successful edit, immediately call `{checkpoint_fname}(label='post-edit')` to allow reverting if the user changes their mind. If the user requests a revert, call `{revert_checkpoint_fname}(checkpoint_id=<latest id>)` or `{reinstate_task_fname}(task_id=<id>, allow_active=false)` depending on context.",
        f"   – If you did not capture the last checkpoint id, call `{latest_checkpoint_fname}()` to retrieve it.",
        (
            f"3) EXECUTE by choosing `{execute_isolated_by_id_fname}` or `{execute_by_id_fname}` based on the decision policy above. "
            "Do NOT modify `start_at` timestamps to force execution."
            if execute_isolated_by_id_fname
            else f"3) EXECUTE by calling `{execute_by_id_fname}(task_id=<head of the 'now' queue>)`. Do NOT modify `start_at` timestamps to force execution."
        ),
        f"4) Do not write status fields directly; lifecycle is managed by the scheduler.",
        "",
        "Use the tools below, step-by-step, following these rules:",
        "",
        "GENERAL SAFETY RULE (state refresh)",
        "-----------------------------------",
        f"• After ANY mutating tool call (including `{execute_by_id_fname}`, `{execute_isolated_by_id_fname}`, `{reorder_queue_fname}`, `{move_tasks_to_queue_fname}`, `{partition_queue_fname}`), you MUST re-query the affected queues using `{list_queues_fname}()` and `{get_queue_fname}(queue_id=…)` before issuing further queue edits or building a new_order list.",
        f"• Never assume prior queue membership or order after detaching or moving tasks. Always refresh first.",
        "",
        "CLARIFICATION POLICY (always prefer tool over prose)",
        "----------------------------------------------------",
        (
            f"• Whenever you need information from the human (e.g., an unknown or ambiguous reference), and `{request_clar_fname}` is available, you must call `{request_clar_fname}` with a concise question. Do not propose options in a plain assistant message when this tool is available."
            if request_clar_fname
            else "• If no clarification tool is available, do not ask questions in your final response; proceed using sensible defaults/best‑guess values and state assumptions explicitly."
        ),
        "",
        "A. If the request contains a *numeric task_id*:",
        f"   • **First** call `{ask_fname}` (or `{get_task_queue_fname}`) to confirm the task exists and learn the current order.",
        (
            f"   • Decide isolation vs chain using the conversation context. If ambiguous, prefer isolation → call `{execute_isolated_by_id_fname}(task_id=<id>)` when available."
            if execute_isolated_by_id_fname
            else "   • Decide isolation vs chain using the conversation context."
        ),
        f"   • If you deliberately choose chain execution, reorder explicitly with `{update_task_queue_fname}` only when necessary, then call `{execute_by_id_fname}` on the intended head. Do not reorder purely to force execution when isolation suffices.",
        (
            f"   • Important: if you used `{execute_isolated_by_id_fname}`, the task is DETACHED and is no longer a member of its former queue. Do not include it in `{reorder_queue_fname}` new_order arrays for that queue. Refresh the queue first to see current membership."
            if execute_isolated_by_id_fname
            else ""
        ),
    ]

    if request_clar_fname:
        lines.extend(
            [
                f"   • STRICT RULE: If the id is **unknown** (zero results), you must immediately call `{request_clar_fname}` to ask whether to create a new task or provide a different reference. Do **NOT** produce a plain-text answer or propose options; use the tool and wait for the answer before proceeding. Do **NOT** call `{execute_by_id_fname}` when the task cannot be confirmed.",
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
            (
                f"      • **Exactly one** clear match → decide isolation vs chain using the conversation context. If ambiguous, prefer isolation (use `{execute_isolated_by_id_fname}` when available); otherwise, if a sequence is intended, reorder as needed and then `{execute_by_id_fname}`."
                if execute_isolated_by_id_fname
                else f"      • **Exactly one** clear match → decide isolation vs chain using the conversation context; if a sequence is intended, reorder as needed and then `{execute_by_id_fname}`."
            ),
            f"      • **Multiple tasks forming a sequence** and the user wants them in order → reorder explicitly (if needed) so the intended head is first; then `{execute_by_id_fname}(task_id=<head>)`.",
            (
                f"      • If you chose `{execute_isolated_by_id_fname}`, remember the started task is detached. Do NOT attempt to reorder its former queue including that id; refresh queues and operate only on current members."
                if execute_isolated_by_id_fname
                else ""
            ),
            f"      • **No match** and it is obvious we should create the task → call `{create_task_fname}(name=<short title>, description=<free‑form user request>)`, then call `{ask_fname}` again to retrieve the new id, optionally reorder, then `{execute_by_id_fname}`.",
            "",
            "   Naming guidance for creation:",
            "   • Derive a concise `name` by trimming punctuation and capitalising key words from the user's request.",
            "   • Use the full free‑form request as the `description` (possibly normalised by removing a trailing period).",
            "   • Do not specify status, schedule, start_at, prev_task/next_task, triggers, or deadlines here; the scheduler infers lifecycle and preserves invariants.",
        ],
    )

    if request_clar_fname:
        lines.extend(
            [
                f"      • **Multiple / ambiguous** matches → call `{request_clar_fname}` so the user can disambiguate, only do so if it's *genuinely* unclear.",
                f"      • **No match**:",
                f"          – If it's ambiguous whether a task should be created/updated → `{request_clar_fname}`.",
                f"          – If it is obvious we need to *create* a new task → the system will handle creation implicitly outside this tool list; once created, call `{execute_by_id_fname}` with its id.",
            ],
        )
    else:
        lines.extend(
            [
                "      • **Multiple / ambiguous** matches → do not ask questions in your final response; proceed with sensible defaults or best‑guess identification, and state your assumptions.",
                "      • **No match**:",
                f"          – If it's ambiguous whether a task should be created/updated → do not ask questions; make a best‑guess decision, state assumptions, and continue.",
                f"          – If it is obvious we need to *create* a new task → the system will handle creation implicitly outside this tool list; once created, call `{execute_by_id_fname}` with its id.",
            ],
        )

    lines.extend(
        [
            "",
            (
                f"C. The Tasks list is updated implicitly by the system. To control execution scope, use `{get_task_queue_fname}` and `{update_task_queue_fname}` explicitly. Do NOT write status fields or override `start_at` to force execution. If a new task is clearly required, use `{create_task_fname}` (name + description only), then call `{ask_fname}` to find its id and start using `{execute_isolated_by_id_fname}` or `{execute_by_id_fname}` per the decision policy above."
                if execute_isolated_by_id_fname
                else f"C. The Tasks list is updated implicitly by the system. To control execution scope, use `{get_task_queue_fname}` and `{update_task_queue_fname}` explicitly. Do NOT write status fields or override `start_at` to force execution. If a new task is clearly required, use `{create_task_fname}` (name + description only), then call `{ask_fname}` to find its id and `{execute_by_id_fname}` to start."
            ),
            "",
            "Stopping semantics (required):",
            "--------------------------------",
            "• When you need to stop an in-progress task, you must use the dynamic stop helper that requires `cancel: boolean`.",
            "  – Use `cancel=true` only when the user explicitly wants to abandon the task (e.g., 'cancel it', 'drop it').",
            "  – Use `cancel=false` when the user intends to defer or resume later (e.g., 'do it next week', 'as originally scheduled').",
            "• You may include a short `reason` string to aid logging.",
            "",
            (
                f"Respond *only* with tool calls until one of the following is true: (a) you have successfully started the task via `{execute_by_id_fname}` or `{execute_isolated_by_id_fname}` and can reply DONE, or (b) you have called `{request_clar_fname}`, received the answer, and it explicitly indicates to stop without starting. You **must not** attempt `{execute_by_id_fname}` until you are certain the referenced task exists."
                if request_clar_fname
                else f"Respond *only* with tool calls until *after* `{execute_by_id_fname}` returns.  You **must not** attempt `{execute_by_id_fname}` until you are certain the referenced task exists. Once the task has started you may reply DONE."
            ),
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
