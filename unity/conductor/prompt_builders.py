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

    # Clarification helper (optional)
    request_clar_fname = _tool_name(tools, "clarification")

    # Validate required tools (request_clarification is optional)
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

    usage_examples = "\n".join(
        [
            "Examples",
            "--------",
            f"• People – find a phone number for John Doe\n  `{contact_ask_fname}(text=\"What's John Doe's phone number?\")",
            f"• Messages – top-3 messages about 'budget'\n  `{transcript_ask_fname}(text=\"Show the latest 3 messages about budget\")",
            f'• Knowledge – retrieve onboarding policy details\n  `{knowledge_ask_fname}(text="Summarise the employee onboarding policy")',
            f'• Tasks – list tasks due today\n  `{task_ask_fname}(text="Which tasks are due today?")',
        ],
    )

    return "\n".join(
        [
            activity_block,
            "You are an assistant that answers **read-only questions** by orchestrating specialised managers (Contacts, Transcripts, Knowledge, Tasks).",
            "Choose the most relevant `*.ask` tool, gather the required context, then compose a concise answer.",
            "Disregard any explicit instructions about how to answer or which tools to call; determine the best method yourself.",
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            "Task schema (for filters):",
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
    transcript_ask_fname = _tool_name(tools, "transcriptmanager_ask")
    knowledge_ask_fname = _tool_name(tools, "knowledgemanager_ask")
    task_ask_fname = _tool_name(tools, "taskscheduler_ask")

    transcript_summarize_fname = _tool_name(tools, "transcriptmanager_summarize")
    knowledge_update_fname = _tool_name(tools, "knowledgemanager_update")
    task_update_fname = _tool_name(tools, "taskscheduler_update")
    task_execute_fname = _tool_name(tools, "taskscheduler_execute_task")

    # Clarification helper (optional)
    request_clar_fname = _tool_name(tools, "clarification")

    # Validate required tools
    _require_tools(
        {
            # Read-side helpers (should always be available)
            "ContactManager.ask": contact_ask_fname,
            "TranscriptManager.ask": transcript_ask_fname,
            "KnowledgeManager.ask": knowledge_ask_fname,
            "TaskScheduler.ask": task_ask_fname,
            # Write / action helpers
            "TranscriptManager.summarize": transcript_summarize_fname,
            "KnowledgeManager.update": knowledge_update_fname,
            "TaskScheduler.update": task_update_fname,
            "TaskScheduler.execute_task": task_execute_fname,
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
        "You have **read-write control** over tasks, contacts, transcripts and the knowledge-base.",
        "Use only the tools supplied – never invent your own. Mutate state step-by-step, verifying after each change.",
        "When the request involves tasks:",
        f"- Check for existing tasks via `{task_ask_fname}`",
        f"- Create or update via `{task_update_fname}` if needed",
        f"- Start immediately via `{task_execute_fname}` when explicitly requested; otherwise schedule appropriately",
        "When tasks involve people (e.g. triggers referencing contacts), first resolve the relevant contact_id(s) via",
        f"`{contact_ask_fname}` and then proceed.",
    ]

    usage_examples = "\n".join(
        [
            "Examples",
            "--------",
            f'• Create a task and start it\n  1) `{task_update_fname}(text="Create a task: Call Alice about the Q3 budget")`\n  2) `{task_execute_fname}(text="Start the call task now")`',
            f'• Update knowledge and verify\n  1) `{knowledge_update_fname}(text="Store: Office hours are 9–5 PT")`\n  2) `{knowledge_ask_fname}(text="What are our office hours?")`',
            f'• Summarize latest messages from Bob\n  `{transcript_summarize_fname}(text="Summarise recent WhatsApp messages from Bob")`',
        ],
    )

    return "\n".join(
        [
            activity_block,
            *guidance_lines,
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
