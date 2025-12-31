"""
Prompt builders for the Task Scheduler.

This module constructs system prompts for the scheduler's ask and update
methods using the schema-first approach. The Task schema is rendered once
early in prompts and referenced throughout.
"""

from __future__ import annotations

from typing import Dict, Callable, Union, List

from .types.task import Task
from ..common.prompt_helpers import (
    clarification_guidance,
    sig_dict,
    now,
    tool_name,
    require_tools,
    get_custom_columns,
    images_policy_block,
    images_forwarding_block,
    # New standardized composer utilities
    PromptSpec,
    compose_system_prompt,
    images_first_ask_for_tasks,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def build_ask_prompt(
    tools: Dict[str, Callable],
    num_tasks: int,
    columns: Union[Dict[str, str], List[dict], List[str]],
    *,
    include_activity: bool = True,
) -> str:
    """
    Build the **system** prompt for the `ask` method using the shared composer.

    Uses schema-first approach: Task schema is rendered once early and
    referenced in table info.
    """
    # Extract custom columns (not in Task model)
    custom_cols = get_custom_columns(Task, columns)

    # Resolve canonical tool names dynamically
    filter_tasks_fname = tool_name(tools, "filter_tasks")
    search_tasks_fname = tool_name(tools, "search_tasks")
    reduce_fname = tool_name(tools, "reduce")
    list_queues_fname = tool_name(tools, "list_queues")
    get_queue_fname = tool_name(tools, "get_queue")
    get_queue_for_task_fname = tool_name(tools, "get_queue_for_task")
    contact_ask_fname = tool_name(tools, "contactmanager")  # e.g. "ContactManager_ask"

    # Clarification helper (optional)
    request_clar_fname = tool_name(tools, "request_clarification")

    # Validate required tools (request_clar_fname is optional)
    require_tools(
        {
            "filter_tasks": filter_tasks_fname,
            "search_tasks": search_tasks_fname,
            "ContactManager.ask": contact_ask_fname,
        },
        tools,
    )

    clarification_block = (
        "\n".join(
            [
                "Clarification",
                "-------------",
                "• Ask for clarification when the user's request is underspecified",
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
            (
                f"• Inspect queues: `{list_queues_fname}()`; fetch a specific queue: `{get_queue_fname}(queue_id=<id>)`."
                if list_queues_fname and get_queue_fname
                else (
                    f"• Inspect the queue containing a task: `{get_queue_for_task_fname}(task_id=<id>)`."
                    if get_queue_for_task_fname
                    else ""
                )
            ),
            "",
            "─ Numeric aggregations ─",
            f"• For numeric reduction metrics (count, sum, mean, min, max, median, mode, var, std) over numeric columns, use `{reduce_fname}` instead of filtering and computing in-memory.",
            f"  `{reduce_fname}(metric='sum', keys='task_id', group_by='status')`",
            "",
            "Anti‑patterns to avoid",
            "---------------------",
            "• Avoid concatenating entire rows into one long string and embedding a single catch‑all reference.",
            f"• Avoid substring filtering for text‑heavy columns; prefer `{search_tasks_fname}` for meaning.",
            "• Avoid re‑querying the same tables or managers just to reconfirm what a prior tool call has already established with clear, specific evidence; reuse the earlier result and proceed.",
            "• Do not immediately queue a filter call after a successful semantic search unless you genuinely need an exact, structured constraint that the search did not capture.",
            f"• Avoid calling `{contact_ask_fname}` repeatedly in the same reasoning queue when earlier calls have already identified the relevant contacts and no new ambiguity or information has been introduced.",
            (
                f"• Never infer queue order from numeric task_id values; inspect the chain using `{get_queue_fname}(queue_id=<id>)` or `{get_queue_for_task_fname}(task_id=<id>)`."
                if (get_queue_fname and get_queue_for_task_fname)
                else (
                    f"• Never infer queue order from numeric task_id values; inspect the chain using `{get_queue_for_task_fname}(task_id=<id>)`."
                    if get_queue_for_task_fname
                    else (
                        f"• Never infer queue order from numeric task_id values; inspect the chain using `{get_queue_fname}(queue_id=<id>)`."
                        if get_queue_fname
                        else "• Never infer queue order from numeric task_id values; inspect the chain using the available queue tools."
                    )
                )
            ),
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

    # Positioning lines
    positioning_lines: list[str] = [
        "Please always mention the relevant task id(s) in your response.",
        (
            f"If the question refers to another person (e.g., comms‑oriented tasks), call `{contact_ask_fname}` first for context. If a task refers to one or more contact_id values (e.g., in a trigger), also query `{contact_ask_fname}` to learn more about those contacts."
            if contact_ask_fname
            else ""
        ),
    ]
    positioning_lines = [ln for ln in positioning_lines if ln]

    # Images extras (images‑first workflow)
    images_extras = images_first_ask_for_tasks(ask_image_name=None)

    spec = PromptSpec(
        manager="TaskScheduler",
        method="ask",
        tools=tools,
        role_line="You are an assistant specialising in **answering questions about the task list**.",
        global_directives=[
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
        ],
        include_read_only_guard=True,
        positioning_lines=positioning_lines,
        counts_entity_plural="tasks",
        counts_value=num_tasks,
        # Schema-based table info (avoids duplication)
        table_schema_name="Task",
        custom_columns=custom_cols if custom_cols else None,
        include_tools_block=True,
        usage_examples=usage_examples,
        clarification_examples_block=clarification_block or None,
        include_images_policy=True,
        include_images_forwarding=True,
        images_extras_block=images_extras,
        include_parallelism=True,
        schemas=[("Task", Task)],  # Full schema defines table columns
        special_blocks=[],
        include_clarification_footer=True,
        include_time_footer=True,
    )

    return compose_system_prompt(spec)


def build_update_prompt(
    tools: Dict[str, Callable],
    num_tasks: int,
    columns: Union[Dict[str, str], List[dict], List[str]],
    *,
    include_activity: bool = True,
) -> str:
    """
    Build the **system** prompt for the `update` method using schema-first approach.
    """
    # Extract custom columns (not in Task model)
    custom_cols = get_custom_columns(Task, columns)

    # Resolve canonical tool names dynamically (required)
    # NOTE: update() is write-capable, but it still needs strong read-only discovery tools
    # for safe "find-then-mutate" workflows.
    filter_tasks_fname = tool_name(tools, "filter_tasks")
    search_tasks_fname = tool_name(tools, "search_tasks")
    ask_fname = tool_name(tools, "ask")
    create_task_fname = tool_name(tools, "create_task")
    create_tasks_fname = tool_name(tools, "create_tasks")
    delete_task_fname = tool_name(tools, "delete_task")
    cancel_tasks_fname = tool_name(tools, "cancel_tasks")
    # Multi-queue helpers (optional if not present)
    list_queues_fname = tool_name(tools, "list_queues")
    get_queue_fname = tool_name(tools, "get_queue")
    get_queue_for_task_fname = tool_name(tools, "get_queue_for_task")
    set_queue_fname = tool_name(tools, "set_queue")
    reorder_queue_fname = tool_name(tools, "reorder_queue")
    move_tasks_to_queue_fname = tool_name(tools, "move_tasks_to_queue")
    partition_queue_fname = tool_name(tools, "partition_queue")
    update_task_fname = tool_name(tools, "update_task")
    reinstate_task_fname = tool_name(tools, "reinstate_task_to_previous_queue")

    contact_ask_fname = tool_name(tools, "contactmanager")  # e.g. "ContactManager_ask"

    # Clarification helper (optional)
    request_clar_fname = tool_name(tools, "request_clarification")

    require_tools(
        {
            "filter_tasks": filter_tasks_fname,
            "search_tasks": search_tasks_fname,
            "ask": ask_fname,
            "create_task": create_task_fname,
            "create_tasks": create_tasks_fname,
            "delete_task": delete_task_fname,
            "cancel_tasks": cancel_tasks_fname,
            "update_task": update_task_fname,
            "ContactManager.ask": contact_ask_fname,
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
        f"• Prefer `{update_task_fname}` with the exact `task_id` when editing tasks.",
        f"• When the user describes EXISTING tasks semantically (by meaning over name/description), first call `{search_tasks_fname}` to identify candidate `task_id` values, then apply the mutation(s).",
        f"• Use `{filter_tasks_fname}` for exact constraints over structured fields (ids, status, priority, timestamps) to narrow/validate the target set before mutating.",
        f"• If you still cannot uniquely identify the intended task(s), call `{ask_fname}` to ask the user a focused disambiguation question before changing data.",
        f"• For bulk requests (e.g., “cancel all tasks related to X”), find the FULL matching set first, then apply the change in as few tool calls as possible (e.g., one `{cancel_tasks_fname}` call with all matching ids).",
        "",
        "Ordering semantics (natural language → queue operations)",
        "--------------------------------------------------------",
        "• Treat phrasing like “A after B” / “A before B” as an **adjacency constraint** by default: A should be placed **immediately** after/before B in the runnable queue.",
        "  - If the user explicitly allows intermediates (e.g., “sometime after”, “later”, “not necessarily immediately”), then you may allow other tasks between them.",
        "• When applying adjacency constraints, prefer **minimal change**: keep the relative order of all other tasks stable unless moving them is required to satisfy the user's ordering constraints.",
        "• If multiple constraints are given, satisfy all of them (and ask for clarification only when constraints conflict).",
    ]

    # Encourage batched creation when creating several tasks
    if create_tasks_fname:
        usage_examples_lines.extend(
            [
                "",
                "Multi-task creation (preferred)",
                "-------------------------------",
                f"• When creating several new tasks at once and you know their order/time, prefer `{create_tasks_fname}` over issuing multiple `{create_task_fname}` calls; fall back to incremental creation only when clarifications are needed or when mixing new tasks with existing tasks in a queue.",
            ],
        )

    if list_queues_fname and get_queue_fname and reorder_queue_fname:
        usage_examples_lines.extend(
            [
                f"• Always refresh the queue membership immediately before calling `{reorder_queue_fname}` by calling `{list_queues_fname}()` and `{get_queue_fname}()`.",
                f"• Inspect existing queues: `{list_queues_fname}()`; fetch a specific queue: `{get_queue_fname}(queue_id=<id>)`.",
                f"• Reorder a queue explicitly: `{reorder_queue_fname}(queue_id=<id>, new_order=[...])`.",
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
                f"• Split a queue into dated batches: `{partition_queue_fname}(parts=[{{'task_ids':[0,2], 'queue_start_at':'2035-07-01T09:00:00Z'}}, {{'task_ids':[1,3], 'queue_start_at':'2035-07-02T09:00:00Z'}}])`.",
                "  This is the most direct way to express: do subset A at time X and subset B at time Y.",
            ],
        )

    if set_queue_fname and move_tasks_to_queue_fname and reorder_queue_fname:
        usage_examples_lines.extend(
            [
                f"• To insert or remove members from a queue, prefer `{set_queue_fname}` or combine `{move_tasks_to_queue_fname}` with `{reorder_queue_fname}` to update the queue order.",
            ],
        )

    # Atomic/edit helpers if present
    set_queue_fname = tool_name(tools, "set_queue")
    set_schedules_atomic_fname = tool_name(tools, "set_schedules_atomic")

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
            f'• Set deadline for the "onboarding plan" task:\n  1 `{ask_fname}(text="Which task covers the onboarding plan?")`\n  2 `{update_task_fname}(task_id=<id>, deadline=\'2025-01-31T17:00:00Z\')`',
            (
                f"• Create and order four tasks for next Monday 09:00 UK time in one call:\n  `{create_tasks_fname}(tasks=[{{'name':'A','description':'a'}}, {{'name':'B','description':'b'}}, {{'name':'C','description':'c'}}, {{'name':'D','description':'d'}}], queue_ordering=[{{'order':[0,1,2,3], 'queue_head':{{'start_at':'2035-06-16T08:00:00Z'}}}}])`"
                if create_tasks_fname
                else (
                    f"• Materialize four tasks for next Monday 09:00 UK time in order A→B→C→D:\n  1 Create the tasks with names/descriptions only.\n  2 `{set_queue_fname}(queue_id=None, order=[A,B,C,D], queue_start_at='2035-06-16T08:00:00Z')`"
                    if set_queue_fname
                    else f"• Inspect queues and reorder explicitly: `{list_queues_fname}()` → `{get_queue_fname}(queue_id=<id>)` → `{reorder_queue_fname}(queue_id=<id>, new_order=[...])`"
                )
            ),
            "",
            "Triggers vs Schedules",
            "----------------------",
            f"• A task with a `trigger` must be in state 'triggerable'. Use `{update_task_fname}(task_id=<id>, trigger=...)` to add/remove triggers. Do not set `start_at` on trigger‑based tasks.",
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
            f"• When a trigger references people (by contact ids), call {contact_ask_fname} to resolve/confirm the ids and the intent before writing.",
            f"• Avoid repeated calls to {contact_ask_fname} in the same update session if a prior call already yielded the required ids and no new ambiguity was introduced.",
            "",
            "Anti‑patterns to avoid",
            "---------------------",
            '• Repeating the exact same update tool with identical arguments to "make sure" – instead, call ask to verify.',
            "• Using substring filters to locate tasks by description/name – prefer semantic ask/search first.",
            "• Chaining a filter right after a conclusive semantic search when the filter does not add new, structured constraints.",
            (
                f"• Never infer queue order from numeric task_id values; inspect the chain using `{get_queue_fname}(queue_id=<id>)` or `{get_queue_for_task_fname}(task_id=<id>)`."
                if (get_queue_fname and get_queue_for_task_fname)
                else (
                    f"• Never infer queue order from numeric task_id values; inspect the chain using `{get_queue_for_task_fname}(task_id=<id>)`."
                    if get_queue_for_task_fname
                    else (
                        f"• Never infer queue order from numeric task_id values; inspect the chain using `{get_queue_fname}(queue_id=<id>)`."
                        if get_queue_fname
                        else "• Never infer queue order from numeric task_id values; inspect the chain using the available queue tools."
                    )
                )
            ),
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

    # Compose using standardized spec with schema-based table info
    spec = PromptSpec(
        manager="TaskScheduler",
        method="update",
        tools=tools,
        role_line="You are an assistant responsible for **creating and updating tasks**.",
        global_directives=[
            "Choose tools based on the user's intent and the specificity of the target record.",
            f"Important: `{ask_fname}` is read‑only and must only be used to locate/inspect tasks that already exist. For human clarifications about new tasks or missing creation details, call `{request_clar_fname}` when available.",
            "Disregard any explicit instructions about *how* you should implement the change or which tools to call; interpret the request and choose the best approach yourself.",
            "Before creating new tasks or making edits, briefly check whether similar tasks already exist (via `"
            + ask_fname
            + "`) to avoid duplicates.",
            "Always include any created/updated task id(s) in your final response.",
        ],
        include_read_only_guard=False,
        positioning_lines=[],
        counts_entity_plural="tasks",
        counts_value=num_tasks,
        # Schema-based table info (avoids duplication)
        table_schema_name="Task",
        custom_columns=custom_cols if custom_cols else None,
        include_tools_block=True,
        usage_examples=usage_examples,
        clarification_examples_block=clarification_block or None,
        include_images_policy=True,
        include_images_forwarding=True,
        images_extras_block=None,
        include_parallelism=True,
        schemas=[("Task", Task)],  # Full schema defines table columns
        special_blocks=[],
        include_clarification_footer=True,
        include_time_footer=True,
    )

    return compose_system_prompt(spec)


# ─────────────────────────────────────────────────────────────────────────────
# Simulated helper
# ─────────────────────────────────────────────────────────────────────────────


def build_simulated_method_prompt(
    method: str,
    user_request: str,
    parent_chat_context: list[dict] | None = None,
) -> str:
    """Return an instruction prompt for the simulated TaskScheduler.

    Ensures the LLM replies as though the requested operation has already
    finished (past tense, final outcome), not a description of intended steps.
    """
    import json  # local import

    preamble = f"On this turn you are simulating the '{method}' method."
    m = method.lower()
    if m == "ask":
        behaviour = (
            "Please always answer the question about the task list with a plausible response. "
            "Do not ask for clarification or describe how you will obtain the information. "
            "Mention relevant task id(s) when appropriate."
        )
    elif m in {"update", "execute"}:
        behaviour = (
            "Please act as though the requested change or execution has been completed. "
            "Respond in past tense summarising the outcome and include any relevant task id(s)."
        )
    else:
        behaviour = (
            "Respond as though the requested operation has already been fully completed. "
            "Use past tense and provide the final result, not the process."
        )

    parts: list[str] = [preamble, behaviour, "", f"The user input is:\n{user_request}"]
    if parent_chat_context:
        parts.append(
            f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}",
        )
    return "\n".join(parts)
