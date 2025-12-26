from __future__ import annotations

from typing import Dict, Callable

from ..common.prompt_helpers import (
    clarification_guidance,
    sig_dict,
    now,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
    render_tools_block,
)

# ───────────────────────────────────── helpers ─────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} using shared helper."""
    return sig_dict(tools)


def _tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """Delegate to shared tool name resolver (case-insensitive substring)."""
    return _shared_tool_name(tools, needle)


def _require_tools(pairs: Dict[str, str | None], tools: Dict[str, Callable]) -> None:
    """Delegate validation to shared helper for consistent errors."""
    _shared_require_tools(pairs, tools)


# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def build_request_prompt(
    tools: Dict[str, Callable],
    *,
    include_activity: bool = True,
) -> str:
    """Dynamic system message for Conductor.request (read-write across domains)."""

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
    web_update_fname = _tool_name(tools, "websearcher_update")
    secret_ask_fname = _tool_name(tools, "secretmanager_ask")
    actor_act_fname = _tool_name(tools, "actor_act")
    fm_ask_fname = _tool_name(tools, "globalfilemanager_ask")
    fm_organize_fname = _tool_name(tools, "globalfilemanager_organize")
    cm_ask_fname = _tool_name(tools, "conversationmanagerhandle_ask")
    cm_interject_fname = _tool_name(tools, "conversationmanagerhandle_interject")
    cm_transcript_fname = _tool_name(
        tools,
        "conversationmanagerhandle_get_full_transcript",
    )

    # Clarification helper (optional)
    request_clar_fname = _tool_name(tools, "clarification")

    # Validate only foundational tools (optional managers may be disabled)
    _require_tools(
        {
            # Foundational read-side helpers
            "ContactManager.ask": contact_ask_fname,
            "TranscriptManager.ask": transcript_ask_fname,
            "TaskScheduler.ask": task_ask_fname,
            "ConversationManagerHandle.ask": cm_ask_fname,
            "ConversationManagerHandle.get_full_transcript": cm_transcript_fname,
            # Foundational write / action helpers
            "ContactManager.update": contact_update_fname,
            "TaskScheduler.update": task_update_fname,
            "TaskScheduler.execute": task_execute_fname,
            "ConversationManagerHandle.interject": cm_interject_fname,
        },
        tools,
    )

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
        "Note: Some managers mentioned in this guidance may not be available in your current toolset. This is expected – simply disregard any guidance that references tools you don't have access to.",
        "Use WebSearcher.ask for external information, market practices, definitions, or anything you would reasonably look up online.",
        "For live or time-sensitive facts (e.g., 'today', 'yesterday', 'this week', 'latest', 'current', 'now'), you must call WebSearcher.ask rather than relying on internal memory.",
        "When searching for data or research, always include 'with citations' or 'include source URLs' in your query so results are verifiable.",
        "When the request involves tasks:",
        f"- Understand intent then check context via `{task_ask_fname}`",
        f"- Apply changes via `{task_update_fname}` if needed",
        f"- Start immediately via `{task_execute_fname}` when explicitly requested; otherwise schedule appropriately",
        "When tasks involve people (e.g. triggers referencing contacts), first resolve the relevant contact_id(s) via",
        f"`{contact_ask_fname}` and then proceed.",
        "\nFiles (rename/move only)",
        "-----------------------",
        (
            f"- Discover targets via `{fm_ask_fname}` (cross-filesystem inventory and search)"
            if fm_ask_fname
            else "- Discover targets via GlobalFileManager.ask (cross-filesystem inventory and search)"
        ),
        (
            f"- Perform safe organization via `{fm_organize_fname}` (rename/move only; no create/delete)"
            if fm_organize_fname
            else "- Perform safe organization via GlobalFileManager.organize (rename/move only; no create/delete)"
        ),
        "- Use ask for discovery, then call organize to apply changes.",
        "\nWebsite Configuration (WebSearcher, not KnowledgeManager)",
        "------------------------------------------------------------",
        "- Saving/registering/configuring websites always goes to WebSearcher.update — NOT KnowledgeManager.",
        "- WebSearcher owns the Websites catalog. Do not store website info in the knowledge base.",
        f"- Flow: `{secret_ask_fname}` (find credentials) → `{web_update_fname}` (register site with credentials).",
        "- When searching gated websites, always request citations and source URLs in your query.",
        "- Results from gated sites should include: article titles, URLs, publication dates, and key data points.",
        "Task execution policy — mandatory execute when asked to run/start:",
        f"- If the user says 'run', 'start', 'execute', 'begin', or 'launch' a task, you MUST call `{task_execute_fname}` exactly once.",
        f"- Do NOT use `{task_update_fname}` as a substitute for starting a task. Only use `{task_update_fname}` to create a missing task or to adjust fields prior to execution, then call `{task_execute_fname}`.",
        f"- If a start time is mentioned (e.g., 'today at 16:00'), still route through `{task_execute_fname}` and shape queues/order as needed before calling it; do not replace execution with an update-only flow.",
        "\nSteering the Live Conversation",
        "----------------------------------------------",
        "In addition to your other tools, you have a direct connection to the live, ongoing conversation with the user, managed by a front-line assistant.",
        "Your role is to act as a 'bigger brain', ensuring the user gets accurate and efficient help by monitoring this conversation and steering it when necessary.",
        "**Your Decision Framework: When to Ask vs. When to Interject**",
        f"First, always use `{cm_transcript_fname}()` to get a snapshot of the conversation. After reviewing the transcript, choose your action:",
        f"**1. Use `{cm_ask_fname}` to Delegate a Question:**",
        "   - **When:** The conversation is stalled because the user's request is **ambiguous**, and the front-line assistant needs more information to proceed.",
        "   - **Your Action:** Formulate the *best possible clarifying question* and use `ask` to have the front-line assistant deliver it.",
        "   - **Example:** The transcript shows the user wants to 'schedule a service.' You should call:",
        f"     `{cm_ask_fname}(question='Of course. Are you trying to schedule a maintenance appointment, a delivery, or something else?')`",
        f"**2. Use `{cm_interject_fname}` to Provide a Correction:**",
        "   - **When:** The front-line assistant has provided **factually incorrect** information.",
        "   - **Your Action:** First, use your own powerful tools (`TaskScheduler_ask`, `ContactManager_ask`) to find the ground truth. Then, use `interject` to provide the correct answer directly to the user.",
        "   - **Example:** The assistant tells the user a task is 'not found', but you find it is 'in progress'. You should call:",
        f"     `{cm_interject_fname}(message='I found that task for you. The \\'Alpha Project kickoff\\' is currently in progress.')`",
        "**Core Principle:** Your intervention should be seamless. The user should perceive a single, helpful assistant.",
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

    # General decomposition and concurrency guidance (cross-domain, not task-specific)
    read_only_tools_line = (
        f"`{contact_ask_fname}`, `{transcript_ask_fname}`, `{knowledge_ask_fname}`, `{guidance_ask_fname}`, `{task_ask_fname}"
        + (f"`, `{web_ask_fname}`" if web_ask_fname else "")
        + (f"`, `{fm_ask_fname}`" if fm_ask_fname else "")
        + "`"
    )
    decomposition_concurrency_request_lines = [
        "",
        "Decompose and parallelize independent sub-requests",
        "--------------------------------------------------",
        "• Split the user's input into minimal sub-requests by intent and data dependency.",
        "• Execute independent sub-requests concurrently when runtime allows; otherwise, run them as separate tool calls in the same turn.",
        "• Serialize dependent sub-requests: resolve the required read(s) first, then apply the write(s).",
        "• Never satisfy a read-only sub-request using the narrative result of a write; always call the appropriate `ask` tool.",
        f"• Read-only tools include: {read_only_tools_line}.",
        f"• Write tools include: `{contact_update_fname}`, `{knowledge_update_fname}`, `{guidance_update_fname}`, `{task_update_fname}`"
        + (f", `{web_update_fname}`" if web_update_fname else "")
        + (f", `{fm_organize_fname}`" if fm_organize_fname else "")
        + ".",
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
        "• If there is an unrelated read-only sub-request, you should run the relevant `ask` concurrently with the `update` when runtime allows (or immediately before/after as a separate tool call). Do not rely on an `update` narrative to answer a read.",
    ]

    web_example = (
        f'\n• Research before update – look up a standard practice\n  `{web_ask_fname}(text="What\'s the typical definition of high priority in agile backlogs?")`'
        if web_ask_fname
        else ""
    )

    fm_usage = (
        f'\n• Rename/move/delete files across filesystems\n  `{fm_organize_fname}(text="Rename /docs/notes.txt to notes-2024.txt; move /invoices/jan.xlsx to /archive/; delete /tmp/old.log.")`'
        if fm_organize_fname
        else ""
    )

    usage_examples = "\n".join(
        [
            "Examples",
            "--------",
            f'• Create a task and start it\n  1) `{task_update_fname}(text="Create a task: Call Alice about the Q3 budget")`\n  2) `{task_execute_fname}(text="Start the call task now")`',
            f'• Update knowledge and verify\n  1) `{knowledge_update_fname}(text="Store: Office hours are 9–5 PT")`\n  2) `{knowledge_ask_fname}(text="What are our office hours?")`',
            f'• Create or update a contact then confirm via read\n  1) `{contact_update_fname}(text="Create Jane Doe with email jane@example.com")`\n  2) `{contact_ask_fname}(text="Show Jane Doe\'s contact details")`{web_example}{fm_usage}',
            f"• Run an existing task immediately\n  `{task_execute_fname}(text=\"Run the task named 'Email Contoso about invoices' now\")`",
            f"• Run a task at a specific time\n  `{task_execute_fname}(text=\"Start 'Prepare slides for kickoff' today at 16:00\")`",
            f'• Create or update guidance\n  `{guidance_update_fname}(text="Create guidance: Troubleshooting VPN issues")`',
        ],
    )

    # Gated website examples (WebSearcher owns website configuration)
    if secret_ask_fname and web_update_fname:
        usage_examples += (
            f"\n• Register gated websites (find credentials, then save to WebSearcher)\n"
            f'  1) `{secret_ask_fname}(text="Find credentials for medium.com and nytimes.com")`\n'
            f'  2) `{web_update_fname}(text="Register medium.com and nytimes.com as gated websites with the credentials found. Mark as subscribed.")`'
        )
    if web_ask_fname:
        usage_examples += (
            f"\n• Search a gated website (with citations)\n"
            f'  `{web_ask_fname}(text="Search HealthInvestor for recent elderly care acquisitions. Include source URLs, sale prices, buyers, and sellers.")`'
        )

    if actor_act_fname:
        usage_examples += (
            f"\n• Execute a free-form activity (ad-hoc/sandbox; live now)\n  "
            f'`{actor_act_fname}(description="Open a browser window so we can walk through the setup together")`'
        )

    return "\n".join(
        [
            *guidance_lines,
            *decomposition_concurrency_request_lines,
            *update_philosophy_lines,
            "",
            render_tools_block(tools),
            "",
            usage_examples,
            "",
            f"Current UTC time is {now()}.",
            clar_section,
            "",
            clarification_block,
            "",
        ],
    )
