from __future__ import annotations

import json
from typing import Dict, Callable

from ..task_scheduler.types.task import Task
from ..common.prompt_helpers import (
    clarification_guidance,
    sig_dict,
    now_utc_str,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
)

# ───────────────────────────────────── helpers ─────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} using shared helper."""
    return sig_dict(tools)


def _now() -> str:
    """Current UTC timestamp in a friendly format."""
    return now_utc_str()


def _tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """Delegate to shared tool name resolver (case-insensitive substring)."""
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
    """Dynamic system message for Conductor.ask (read-only across domains)."""
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Resolve canonical tool names (class-qualified; include_class_name=True)
    contact_ask_fname = _tool_name(tools, "contactmanager_ask")
    transcript_ask_fname = _tool_name(tools, "transcriptmanager_ask")
    knowledge_ask_fname = _tool_name(tools, "knowledgemanager_ask")
    task_ask_fname = _tool_name(tools, "taskscheduler_ask")
    web_ask_fname = _tool_name(tools, "websearcher_ask")

    # Clarification helper (optional)
    request_clar_fname = _tool_name(tools, "clarification")

    # Validate required tools (request_clarification and web-search are optional)
    _require_tools(
        {
            "ContactManager.ask": contact_ask_fname,
            "TranscriptManager.ask": transcript_ask_fname,
            "KnowledgeManager.ask": knowledge_ask_fname,
            "TaskScheduler.ask": task_ask_fname,
        },
        tools,
    )

    # Optional clarification usage block
    clarification_block = (
        "\n".join(
            [
                "Clarification",
                "-------------",
                "• Ask for clarification when the user's request is underspecified",
                f'  `{request_clar_fname}(question="Which domain or item are you referring to?")`',
            ],
        )
        if request_clar_fname
        else ""
    )

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    # High-level orchestration guidance (do not describe HOW, only orchestrate)
    guidance = [
        "You are an assistant that answers read-only questions by orchestrating high-level managers (Contacts, Transcripts, Knowledge, Tasks, WebSearch).",
        "Choose the most appropriate manager's `ask` tool per sub-question and compose the final answer.",
        "Do not explain HOW the question will be answered, which low-level tools will be used, or instruct the user how to phrase their question; that is handled entirely by the domain managers.",
        "Use the WebSearcher.ask tool for general knowledge, external information, industry concepts, best practices or anything that would reasonably be found on the web (and not in your internal managers).",
        "For live or time-sensitive facts (e.g., questions containing 'today', 'yesterday', 'this week', 'latest', 'current', 'now'), you must use WebSearcher.ask – do not rely on internal memory for these.",
        "Use Contact/Transcript/Knowledge/Task managers for internal state about people, messages, stored facts and tasks respectively.",
    ]

    web_example = (
        (
            f'\n• Web – explain the Eisenhower Matrix\n  `{web_ask_fname}(text="What is the Eisenhower Matrix and when should it be used?")`'
            f'\n• Web – live facts (weather today)\n  `{web_ask_fname}(text="What\'s the weather in Berlin today?")`'
            f'\n• Web – live facts (headlines this week)\n  `{web_ask_fname}(text="What are the major world news headlines this week?")`'
            f'\n• Web – live facts (yesterday\'s decision)\n  `{web_ask_fname}(text="Did the UN Security Council approve the resolution yesterday?")`'
        )
        if web_ask_fname
        else ""
    )

    usage_examples = "\n".join(
        [
            "Examples",
            "--------",
            f'• People – who is the Berlin-based product designer?\n  `{contact_ask_fname}(text="Who is the Berlin-based product designer?")`',
            f'• Messages – top-3 messages about budgeting and banking\n  `{transcript_ask_fname}(text="Show the latest 3 messages about banking and budgeting")`',
            f'• Knowledge – onboarding policy summary\n  `{knowledge_ask_fname}(text="Summarise the employee onboarding policy")`',
            f'• Tasks – list tasks due today\n  `{task_ask_fname}(text="Which tasks are due today?")`{web_example}',
        ],
    )

    return "\n".join(
        [
            activity_block,
            *guidance,
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            "Task schema (reference):",
            json.dumps(Task.model_json_schema(), indent=4),
            "",
            usage_examples,
            "",
            f"Current UTC time is {_now()}.",
            clar_section,
            "",
            clarification_block,
            "",
        ],
    )


def build_request_prompt(
    tools: Dict[str, Callable],
    *,
    include_activity: bool = True,
) -> str:
    """Dynamic system message for Conductor.request (read-write across domains)."""
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Resolve canonical tool names (class-qualified; include_class_name=True)
    contact_ask_fname = _tool_name(tools, "contactmanager_ask")
    contact_update_fname = _tool_name(tools, "contactmanager_update")
    transcript_ask_fname = _tool_name(tools, "transcriptmanager_ask")
    knowledge_ask_fname = _tool_name(tools, "knowledgemanager_ask")
    knowledge_update_fname = _tool_name(tools, "knowledgemanager_update")
    task_ask_fname = _tool_name(tools, "taskscheduler_ask")
    task_update_fname = _tool_name(tools, "taskscheduler_update")
    task_execute_fname = _tool_name(tools, "taskscheduler_execute")
    web_ask_fname = _tool_name(tools, "websearcher_ask")

    # Clarification helper (optional)
    request_clar_fname = _tool_name(tools, "clarification")

    # Validate required tools (web-search optional, but encouraged)
    _require_tools(
        {
            # Read-side helpers (should always be available)
            "ContactManager.ask": contact_ask_fname,
            "TranscriptManager.ask": transcript_ask_fname,
            "KnowledgeManager.ask": knowledge_ask_fname,
            "TaskScheduler.ask": task_ask_fname,
            # Write / action helpers
            "ContactManager.update": contact_update_fname,
            "KnowledgeManager.update": knowledge_update_fname,
            "TaskScheduler.update": task_update_fname,
            "TaskScheduler.execute": task_execute_fname,
        },
        tools,
    )

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    clarification_block = (
        "\n".join(
            [
                "Clarification",
                "-------------",
                "• If any request is ambiguous, ask the user to disambiguate before changing data",
                f'  `{request_clar_fname}(question="There are several possible matches. Which one did you mean?")`',
            ],
        )
        if request_clar_fname
        else ""
    )

    guidance_lines = [
        "You have read-write control over tasks, contacts, transcripts and the knowledge-base.",
        "Orchestrate by calling the appropriate managers' `ask` or `update` methods; do not describe or expose HOW the change will be implemented.",
        "Use WebSearcher.ask for external information, market practices, definitions, or anything you would reasonably look up online.",
        "For live or time-sensitive facts (e.g., 'today', 'yesterday', 'this week', 'latest', 'current', 'now'), you must call WebSearcher.ask rather than relying on internal memory.",
        "When the request involves tasks:",
        f"- Understand intent then check context via `{task_ask_fname}`",
        f"- Apply changes via `{task_update_fname}` if needed",
        f"- Start immediately via `{task_execute_fname}` when explicitly requested; otherwise schedule appropriately",
        "When tasks involve people (e.g. triggers referencing contacts), first resolve the relevant contact_id(s) via",
        f"`{contact_ask_fname}` and then proceed.",
    ]

    # Core philosophy for update tools: they are cautious, state-aware, and avoid duplication.
    update_philosophy_lines = [
        "",
        "Update tools – cautious, state-aware, and preferable to ask+update chains",
        "--------------------------------------------------------------------------------",
        "• All `update` methods (Contacts, Knowledge, Tasks) first inspect existing state and avoid duplications.",
        "• Prefer calling `update` directly with conditional logic instead of performing a preliminary `ask`.",
        "  - Example (Contacts):",
        f"    Prefer `{contact_update_fname}(text=\"Add David's number as {{number}} if it's not already stored\")`",
        "    over asking for David's current number first and then updating.",
        "  - Example (Knowledge):",
        f'    Prefer `{knowledge_update_fname}(text="Record that exchanges are allowed within 45 days if not already recorded")`.',
        "  - Example (Tasks):",
        f'    Prefer `{task_update_fname}(text="Create or update: Follow up with Contoso tomorrow at 09:00; if it exists, adjust start time")`.',
        "• Do not route update-related verification through `ask`; `update` handles conditional checks safely.",
        "• If there is an unrelated read-only question, you may run `ask` in parallel with an `update` to save time.",
    ]

    web_example = (
        f'\n• Research before update – look up a standard practice\n  `{web_ask_fname}(text="What\'s the typical definition of high priority in agile backlogs?")`'
        if web_ask_fname
        else ""
    )

    usage_examples = "\n".join(
        [
            "Examples",
            "--------",
            f'• Create a task and start it\n  1) `{task_update_fname}(text="Create a task: Call Alice about the Q3 budget")`\n  2) `{task_execute_fname}(text="Start the call task now")`',
            f'• Update knowledge and verify\n  1) `{knowledge_update_fname}(text="Store: Office hours are 9–5 PT")`\n  2) `{knowledge_ask_fname}(text="What are our office hours?")`',
            f'• Create or update a contact then confirm via read\n  1) `{contact_update_fname}(text="Create Jane Doe with email jane@example.com")`\n  2) `{contact_ask_fname}(text="Show Jane Doe\'s contact details")`{web_example}',
        ],
    )

    return "\n".join(
        [
            activity_block,
            *guidance_lines,
            *update_philosophy_lines,
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            "Task schema:",
            json.dumps(Task.model_json_schema(), indent=4),
            "",
            usage_examples,
            "",
            f"Current UTC time is {_now()}.",
            clar_section,
            "",
            clarification_block,
            "",
        ],
    )
