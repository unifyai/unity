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
from ..common.read_only_ask_guard import read_only_ask_mutation_exit_block

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
    guidance_ask_fname = _tool_name(tools, "guidancemanager_ask")
    task_ask_fname = _tool_name(tools, "taskscheduler_ask")
    web_ask_fname = _tool_name(tools, "websearcher_ask")
    actor_act_fname = _tool_name(tools, "actor_act")

    # Clarification helper (optional)
    request_clar_fname = _tool_name(tools, "clarification")

    # Validate required tools (request_clarification and web-search are optional)
    _require_tools(
        {
            "ContactManager.ask": contact_ask_fname,
            "TranscriptManager.ask": transcript_ask_fname,
            "KnowledgeManager.ask": knowledge_ask_fname,
            "GuidanceManager.ask": guidance_ask_fname,
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
        "When choosing WebSearch: send exactly one high-level, natural-language question to WebSearcher.ask. Do NOT fan-out multiple WebSearcher.ask calls, do NOT include engine-specific operators (e.g., 'site:'), and do NOT hard-code provider choices. The WebSearcher internally selects sources, parallelizes searches, extracts, and composes references.",
        "Include any citation/link needs, time window, and scope in that single call. Do not immediately re-query just to 'confirm' the same thing.",
        "Issue a second WebSearcher.ask only if the first response clearly indicates missing coverage or ambiguity that requires a new targeted fetch.",
        "Use multiple WebSearcher.ask calls in parallel only when the user asks genuinely unrelated sub-questions; otherwise keep to one call and let WebSearcher fan-out internally.",
        "If refinement is needed, prefer a single follow-up via clarification rather than issuing multiple WebSearcher.ask calls in parallel.",
    ]

    # Mention Actor availability (read-only surface cannot invoke it)
    # Also steer routing intent: live ad-hoc walkthroughs are Actor territory
    if actor_act_fname:
        guidance.extend(
            [
                f"The Actor is an executor available on the write surface as `{actor_act_fname}`; it is not available here on ask.",
                "If the user's question implicitly asks for a live walkthrough (e.g., 'can you open a browser and show me?'), steer the conversation to the write surface and use the Actor there instead of tasks.",
            ],
        )
    else:
        guidance.append(
            "The Actor executor (Actor.act) is only available on the write surface (request). For live walkthroughs, defer answering here and suggest switching to request mode to use the Actor.",
        )

    # Single-session rule (informational)
    guidance.append(
        "Only one live session can run at a time – either a Task execution or an Actor session; while one is in-flight, the other surface is unavailable.",
    )

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
            f'• Guidance – find guidance about onboarding demos\n  `{guidance_ask_fname}(text="Find guidance about the onboarding demo")`',
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
            "",
            read_only_ask_mutation_exit_block(),
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
    guidance_ask_fname = _tool_name(tools, "guidancemanager_ask")
    guidance_update_fname = _tool_name(tools, "guidancemanager_update")
    task_ask_fname = _tool_name(tools, "taskscheduler_ask")
    task_update_fname = _tool_name(tools, "taskscheduler_update")
    task_execute_fname = _tool_name(tools, "taskscheduler_execute")
    web_ask_fname = _tool_name(tools, "websearcher_ask")
    actor_act_fname = _tool_name(tools, "actor_act")

    # Clarification helper (optional)
    request_clar_fname = _tool_name(tools, "clarification")

    # Validate required tools (web-search optional, but encouraged)
    _require_tools(
        {
            # Read-side helpers (should always be available)
            "ContactManager.ask": contact_ask_fname,
            "TranscriptManager.ask": transcript_ask_fname,
            "KnowledgeManager.ask": knowledge_ask_fname,
            "GuidanceManager.ask": guidance_ask_fname,
            "TaskScheduler.ask": task_ask_fname,
            # Write / action helpers
            "ContactManager.update": contact_update_fname,
            "KnowledgeManager.update": knowledge_update_fname,
            "GuidanceManager.update": guidance_update_fname,
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
        "When routing to WebSearch, send a single high-level natural-language question; do NOT issue multiple WebSearcher.ask calls with different sites or providers. The WebSearcher will fan-out, search, and aggregate internally.",
        "Include citation/link needs, time window, and scope in that single call. Do not immediately re-query just to 'confirm' the same thing.",
        "Issue a second WebSearcher.ask only if the first response clearly indicates missing coverage or ambiguity that requires a new targeted fetch.",
        "Use multiple WebSearcher.ask calls in parallel only for genuinely unrelated sub-questions; otherwise keep to one call and let WebSearcher fan-out internally.",
        "When the request involves tasks:",
        f"- Understand intent then check context via `{task_ask_fname}`",
        f"- Apply changes via `{task_update_fname}` if needed",
        f"- Start immediately via `{task_execute_fname}` when explicitly requested; otherwise schedule appropriately",
        "When tasks involve people (e.g. triggers referencing contacts), first resolve the relevant contact_id(s) via",
        f"`{contact_ask_fname}` and then proceed.",
        "Task execution policy — mandatory execute when asked to run/start:",
        f"- If the user says 'run', 'start', 'execute', 'begin', or 'launch' a task, you MUST call `{task_execute_fname}` exactly once.",
        f"- Do NOT use `{task_update_fname}` as a substitute for starting a task. Only use `{task_update_fname}` to create a missing task or to adjust fields prior to execution, then call `{task_execute_fname}`.",
        f"- If a start time is mentioned (e.g., 'today at 16:00'), still route through `{task_execute_fname}` and shape queues/order as needed before calling it; do not replace execution with an update-only flow.",
    ]

    if actor_act_fname:
        guidance_lines.extend(
            [
                "Execution entry-points:",
                f"- Use `{task_execute_fname}` when the activity is a clear, trackable Task (name/description/status).",
                f"- Use `{actor_act_fname}` for ad-hoc, conversational sandbox sessions (onboarding, live screen/browser guidance) that don't need task tracking.",
                "Routing rule (important): If the user requests a live walkthrough or immediate interactive guidance — phrases like 'open a browser', 'walk me through', 'let's set this up together', 'troubleshoot with me now' — call the Actor immediately. Do NOT create or update a task first.",
                "Only one can run at a time; while one is active, the other surface is hidden.",
            ],
        )

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
            f"• Run an existing task immediately\n  `{task_execute_fname}(text=\"Run the task named 'Email Contoso about invoices' now\")`",
            f"• Run a task at a specific time\n  `{task_execute_fname}(text=\"Start 'Prepare slides for kickoff' today at 16:00\")`",
            f'• Create or update guidance\n  `{guidance_update_fname}(text="Create guidance: Troubleshooting VPN issues")`',
        ],
    )

    if actor_act_fname:
        usage_examples += (
            f"\n• Execute a free-form activity (ad-hoc/sandbox; live now)\n  "
            f'`{actor_act_fname}(description="Open a browser window so we can walk through the setup together")`'
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
