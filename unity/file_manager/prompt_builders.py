from __future__ import annotations

import json
from typing import Dict, Callable

from ..common.prompt_helpers import (
    clarification_guidance,
    sig_dict,
    now,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
)
from ..common.read_only_ask_guard import read_only_ask_mutation_exit_block

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {name: '(<argspec>)', …} using shared helper."""
    return sig_dict(tools)


def _tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """Delegate to shared tool name resolver."""
    return _shared_tool_name(tools, needle)


def _require_tools(pairs: Dict[str, str | None], tools: Dict[str, Callable]) -> None:
    """Delegate validation to shared helper for consistent errors."""
    _shared_require_tools(pairs, tools)


# ─────────────────────────────────────────────────────────────────────────────
# FileManager prompt builders (single filesystem)
# ─────────────────────────────────────────────────────────────────────────────


def build_file_manager_ask_prompt(
    tools: Dict[str, Callable],
    *,
    num_files: int = 0,
    columns: Dict[str, str] | None = None,
    table_schemas_json: str | None = None,
    include_activity: bool = True,
) -> str:
    """
    Build the system prompt for AdapterFileManager.ask (filesystem-wide Q&A).
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    columns = columns or {}

    exists_fname = _tool_name(tools, "exists")
    parse_fname = _tool_name(tools, "parse")
    list_fname = _tool_name(tools, "list")
    filter_files_fname = _tool_name(tools, "_filter_files") or _tool_name(
        tools,
        "filter_files",
    )
    search_files_fname = _tool_name(tools, "_search_files") or _tool_name(
        tools,
        "search_files",
    )
    list_columns_fname = _tool_name(tools, "_list_columns") or _tool_name(
        tools,
        "list_columns",
    )
    filter_join_fname = _tool_name(tools, "_filter_join") or _tool_name(
        tools,
        "filter_join",
    )
    search_join_fname = _tool_name(tools, "_search_join") or _tool_name(
        tools,
        "search_join",
    )
    filter_mjoin_fname = _tool_name(tools, "_filter_multi_join") or _tool_name(
        tools,
        "filter_multi_join",
    )
    search_mjoin_fname = _tool_name(tools, "_search_multi_join") or _tool_name(
        tools,
        "search_multi_join",
    )
    request_clar_fname = _tool_name(tools, "request_clarification")

    # Require only the core read tools; search/filter are recommended when present
    _require_tools(
        {
            "exists": exists_fname,
            "parse": parse_fname,
            "list": list_fname,
        },
        tools,
    )

    clarification_block = (
        "\n".join(
            [
                "Clarification",
                "-------------",
                f"• Ask for clarification when the user's request is underspecified",
                f'  `{request_clar_fname}(question="Which file did you mean?")`',
            ],
        )
        if request_clar_fname
        else ""
    )

    usage_lines = [
        "Examples",
        "--------",
        "",
        "─ Columns ─",
        f"• Inspect schema{'' if not list_columns_fname else ''}",
    ]
    if list_columns_fname:
        usage_lines.append(f"  `{list_columns_fname}()`")
    # Inventory listing
    usage_lines += [
        "",
        "─ Inventory ─",
        "• List available files",
    ]
    if list_fname:
        usage_lines.append(f"  `{list_fname}()`")
    usage_lines += [
        "",
        "─ Tool selection (read carefully) ─",
        f"• Use `{exists_fname}` to check file availability before operations.",
    ]
    if search_files_fname:
        usage_lines.append(
            f"• Prefer `{search_files_fname}` for semantic/topic-oriented discovery when available.",
        )
    if filter_files_fname:
        usage_lines.append(
            f"• Use `{filter_files_fname}` for exact/boolean filtering over filenames/metadata.",
        )

    # Join usage guidance (global index ↔ per-file contexts ↔ extracted tables)
    usage_lines += [
        "",
        "─ Joining contexts (advanced but fast) ─",
        "• Use the provided join tools to combine the global index with a single file's context, and optionally with per-table contexts.",
        "• The concrete manager resolves context references for you; pass identifiers exactly as described by tool signatures.",
    ]
    if any([filter_join_fname, search_join_fname]):
        if filter_join_fname:
            usage_lines.append(
                f"• Filter a join result: `{filter_join_fname}(...)` (join global index to the specific file's context, then filter rows)",
            )
        if search_join_fname:
            usage_lines.append(
                f"• Semantic search over a join: `{search_join_fname}(...)` (rank rows after joining against a reference query)",
            )
    if any([filter_mjoin_fname, search_mjoin_fname]):
        usage_lines += [
            "• Chain multiple joins using the multi-join tools and reference the previous step with '$prev'.",
        ]
        if filter_mjoin_fname:
            usage_lines.append(
                f"• Multi-step filter: `{filter_mjoin_fname}(...)` (chain more than two contexts; use '$prev' to reference the previous result)",
            )
        if search_mjoin_fname:
            usage_lines.append(
                f"• Multi-step semantic search: `{search_mjoin_fname}(...)`",
            )

    # Add parse guidance (no direct byte opening)
    usage_lines += [
        "",
        f"─ Parsing Strategy (IMPORTANT: parsing is expensive) ─",
        f"• BEFORE calling `{parse_fname}`, check if parsed data already exists using retrieval tools:",
    ]
    if filter_files_fname:
        usage_lines.append(
            f"  - `{filter_files_fname}` to check for existing records",
        )
    if search_files_fname:
        usage_lines.append(
            f"  - `{search_files_fname}` to query existing parsed content",
        )
    usage_lines += [
        f"• ONLY call `{parse_fname}` if:",
        "  - No parsed data exists in the system, OR",
        "  - The user explicitly requests parsing",
        f"• When parsed data exists, prefer search/filter/join tools instead of `{parse_fname}`",
    ]

    if clarification_block:
        usage_lines += ["", clarification_block]
    else:
        usage_lines += [
            "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state assumptions.",
        ]

    clar_section = clarification_guidance(tools)
    clar_sentence = (
        f"Do not ask the user questions in your final response, please only use the `{request_clar_fname}` tool to ask clarifying questions."
        if request_clar_fname
        else (
            "Do not ask the user questions in your final response. Instead, proceed using sensible defaults/best‑guess values and explicitly tell inner tools that these are assumptions/best guesses, not confirmed answers."
        )
    )

    # Include tables overview if provided
    overview_block = table_schemas_json or "{}"

    return "\n".join(
        [
            "You are an assistant specializing in **retrieving file information and analyzing file contents**.",
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
            clar_sentence,
            "",
            read_only_ask_mutation_exit_block(),
            "",
            "You should attempt to answer *any* question as best you can, even if it seems out of scope.",
            "Use the tools provided to see if you can find any missing context *before* asking the user for clarifications.",
            "Please always mention the relevant file path(s) in your response.",
            "",
            "Context map",
            "-----------",
            "• Global index: a lightweight index of files",
            "• Per-file contexts: rows representing content and hierarchy for a single file",
            "• Per-table contexts: extracted tables (no predefined fields) for a single file",
            "",
            "Tables overview",
            "----------------",
            overview_block,
            "",
            f"There are currently {num_files} files stored in a table with the following columns:",
            json.dumps(columns, indent=4),
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            "\n".join(usage_lines),
            "",
            clar_section,
            "",
            f"Current UTC time is {_now()}.",
        ],
    )


def build_file_manager_ask_about_file_prompt(
    tools: Dict[str, Callable],
    *,
    table_schemas_json: str | None = None,
    include_activity: bool = True,
) -> str:
    """
    Build the focused system prompt for AdapterFileManager.ask_about_file.
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)
    request_clar_fname = _tool_name(tools, "request_clarification")

    clar_sentence = (
        f"Do not ask the user questions in your final response, please only use the `{request_clar_fname}` tool to ask clarifying questions."
        if request_clar_fname
        else (
            "Do not ask the user questions in your final response. Instead, proceed using sensible defaults/best‑guess values and explicitly tell inner tools that these are assumptions/best guesses, not confirmed answers."
        )
    )

    parse_fname = _tool_name(tools, "parse")

    parse_guidance_lines = []
    if parse_fname:
        parse_guidance_lines = [
            "",
            "Parsing Strategy (IMPORTANT: parsing is expensive)",
            "─────────────────────────────────────────────────",
            f"• Parsing with `{parse_fname}` is compute intensive",
            "• BEFORE parsing, check if you already have the data you need via retrieval tools:",
            "  - Use filter/search/join against existing contexts",
            f"• ONLY call `{parse_fname}` if:",
            "  - You need structured/extracted data that doesn't exist yet, OR",
            "  - The user explicitly requests parsing (e.g., 'parse', 'extract data')",
        ]

    # Add join/search/filter guidance for file-scoped questions
    filter_join_fname = _tool_name(tools, "_filter_join") or _tool_name(
        tools,
        "filter_join",
    )
    search_join_fname = _tool_name(tools, "_search_join") or _tool_name(
        tools,
        "search_join",
    )
    filter_mjoin_fname = _tool_name(tools, "_filter_multi_join") or _tool_name(
        tools,
        "filter_multi_join",
    )
    search_mjoin_fname = _tool_name(tools, "_search_multi_join") or _tool_name(
        tools,
        "search_multi_join",
    )

    join_guidance_lines = [
        "",
        "Context usage (file-scoped)",
        "---------------------------",
        "• Prefer joining the global index with this file's per-file context when you need content rows.",
        "• You may also join with per-table contexts (e.g., extracted sheets) for structured queries.",
        "• Use the provided join tools to combine these contexts; the manager resolves references for you.",
    ]
    if filter_join_fname or search_join_fname:
        join_guidance_lines.append(
            "• Filter or semantically search the joined result for precise answers.",
        )
    if filter_mjoin_fname or search_mjoin_fname:
        join_guidance_lines.append(
            "• Chain multiple joins when additional contexts are needed; use '$prev' to reference the previous result.",
        )

    overview_block = table_schemas_json or "{}"

    return "\n".join(
        [
            activity_block,
            "You are an assistant specializing in **analyzing the content of a specific file**.",
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
            clar_sentence,
            "You should attempt to answer *any* question as best you can, even if it seems out of scope.",
            "Use the tools provided to see if you can find any missing context *before* asking the user for clarifications.",
            "Please always mention the relevant file path in your response.",
            "",
            "Important: When calling tools, use the filename exactly as provided in the user message. Do not construct or modify file paths.",
            *parse_guidance_lines,
            "",
            "Tables overview",
            "----------------",
            overview_block,
            *join_guidance_lines,
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            clar_section,
            "",
            f"Current UTC time is {_now()}.",
        ],
    )


def build_file_manager_organize_prompt(
    tools: Dict[str, Callable],
    *,
    num_files: int = 0,
    columns: Dict[str, str] | None = None,
    table_schemas_json: str | None = None,
    include_activity: bool = True,
) -> str:
    """
    Build the system prompt for AdapterFileManager.organize (rename/move only).
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    columns = columns or {}

    ask_fname = _tool_name(tools, "ask")
    rename_file_fname = _tool_name(tools, "rename_file") or _tool_name(tools, "rename")
    move_file_fname = _tool_name(tools, "move_file") or _tool_name(tools, "move")
    request_clar_fname = _tool_name(tools, "request_clarification")

    # Core read tools are required; mutation tools are optional per backend capabilities
    # Only require ask for discovery; organize is mutation-focused
    _require_tools({"ask": ask_fname}, tools)

    clarification_block = (
        "\n".join(
            [
                "Clarification",
                "-------------",
                f"• Ask for clarification when the user's request is underspecified",
                f'  `{request_clar_fname}(question="What criteria should I use for grouping files?")`',
            ],
        )
        if request_clar_fname
        else ""
    )

    operation_tools = []
    if rename_file_fname:
        operation_tools.append(
            f"• Rename a file: `{rename_file_fname}(target_id_or_path='old.txt', new_name='new.txt')`",
        )
    if move_file_fname:
        operation_tools.append(
            f"• Move a file: `{move_file_fname}(target_id_or_path='file.txt', new_parent_path='/dest/')`",
        )
    if not operation_tools:
        operation_tools.append(
            "• No file modification tools are available for this filesystem.",
        )

    usage_lines = [
        "Examples",
        "--------",
        "",
        "─ Discovery via ask() (read-only) ─",
        f"• Delegate discovery to `{ask_fname}` to identify targets before mutating.",
        f"  `{ask_fname}(text='Which files under /docs mention quarterly reports?')`",
    ]

    usage_lines += [
        "",
        "─ File Organization Operations (if supported) ─",
        "\n".join(operation_tools),
        "",
        "Anti‑patterns to avoid",
        "---------------------",
        "• Do not create or delete files. Only rename or move existing ones.",
    ]

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)
    clar_sentence = (
        f"Do not ask the user questions in your final response, please only use the `{request_clar_fname}` tool to ask clarifying questions."
        if request_clar_fname
        else (
            "Do not ask the user questions in your final response. Instead, proceed using sensible defaults/best‑guess values and explicitly tell inner tools that these are assumptions/best guesses, not confirmed answers."
        )
    )

    overview_block = table_schemas_json or "{}"

    return "\n".join(
        [
            activity_block,
            "You are an assistant specializing in **organizing files and folders in the filesystem**.",
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the goal and choose the best approach yourself.",
            clar_sentence,
            "You should attempt to fulfill the organization goal as best you can, even if it seems out of scope.",
            "Use the ask() tool to discover targets before mutating; do not run read-only tools directly in organize.",
            "Please always mention the relevant filename(s) or folder(s) in your response.",
            "",
            f"There are currently {num_files} files indexed.",
            "",
            "Tables overview",
            "----------------",
            overview_block,
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            "\n".join(usage_lines),
            "",
            clar_section,
            "",
            f"Current UTC time is {_now()}.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# GlobalFileManager prompt builders (multi-filesystem)
# ─────────────────────────────────────────────────────────────────────────────


def build_global_file_manager_ask_prompt(
    tools: Dict[str, Callable],
    *,
    num_filesystems: int = 0,
    include_activity: bool = True,
) -> str:
    """
    Build the system prompt for the GlobalFileManager.ask method.
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    request_clar_fname = _tool_name(tools, "request_clarification")

    clarification_block = (
        "\n".join(
            [
                "Clarification",
                "-------------",
                f"• Ask for clarification when the user's request is underspecified",
                f'  `{request_clar_fname}(question="Which filesystem are you referring to?")`',
            ],
        )
        if request_clar_fname
        else ""
    )

    # Discover class-named per-manager tools dynamically for examples
    ask_tool_example = next(
        (
            n
            for n in tools
            if n.lower().endswith("_ask") and "globalfilemanager_" not in n.lower()
        ),
        None,
    )
    ask_about_file_example = next(
        (
            n
            for n in tools
            if n.lower().endswith("_ask_about_file")
            and "globalfilemanager_" not in n.lower()
        ),
        None,
    )
    usage_examples = "\n".join(
        [
            "Examples",
            "--------",
            "─ Filesystem Discovery ─",
            "• List available filesystems: `GlobalFileManager_list_filesystems()`",
            (
                f"• Ask a specific filesystem: `{ask_tool_example}(text='What are the largest files?')`"
                if ask_tool_example
                else "• Ask a specific filesystem: `<SomeFileManager>_ask(text='What are the largest files?')`"
            ),
            (
                f"• Ask about a file: `{ask_about_file_example}(filename='report.pdf', question='Summarize this.')`"
                if ask_about_file_example
                else "• Ask about a file: `<SomeFileManager>_ask_about_file(filename='report.pdf', question='Summarize this.')`"
            ),
        ],
    )

    if not clarification_block:
        usage_examples = "\n".join(
            [
                usage_examples,
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state assumptions.",
            ],
        )

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    clar_sentence = (
        f"Do not ask the user questions in your final response, please only use the `{request_clar_fname}` tool to ask clarifying questions."
        if request_clar_fname
        else (
            "Do not ask the user questions in your final response. Instead, proceed using sensible defaults/best‑guess values and explicitly tell inner tools that these are assumptions/best guesses, not confirmed answers."
        )
    )

    return "\n".join(
        [
            activity_block,
            "You are an assistant specializing in **managing and querying multiple filesystems**.",
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
            clar_sentence,
            "You should attempt to answer *any* question as best you can, even if it seems out of scope.",
            "Use the tools provided to see if you can find any missing context *before* asking the user for clarifications.",
            "",
            f"You have access to {num_filesystems} filesystems.",
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            usage_examples,
            "",
            clar_section,
            "",
            f"Current UTC time is {_now()}.",
        ],
    )


def build_global_file_manager_organize_prompt(
    tools: Dict[str, Callable],
    *,
    num_filesystems: int = 0,
    include_activity: bool = True,
) -> str:
    """
    Build the system prompt for the GlobalFileManager.organize method.
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    request_clar_fname = _tool_name(tools, "request_clarification")

    clarification_block = (
        "\n".join(
            [
                "Clarification",
                "-------------",
                f"• Ask for clarification when the user's request is underspecified",
                f'  `{request_clar_fname}(question="Which filesystem do you want to organize?")`',
            ],
        )
        if request_clar_fname
        else ""
    )

    # Discover class-named per-manager organize tool for examples
    organize_tool_example = next(
        (
            n
            for n in tools
            if n.lower().endswith("_organize") and "globalfilemanager_" not in n.lower()
        ),
        None,
    )
    usage_examples = "\n".join(
        [
            "Examples",
            "--------",
            "─ Filesystem Discovery via ask() (read‑only) ─",
            "• Delegate discovery to `GlobalFileManager_ask(text='Which files mention invoices?')`",
            (
                f"• Organize a specific filesystem: `{organize_tool_example}(text='group by year-month')`"
                if organize_tool_example
                else "• Organize a specific filesystem: `<SomeFileManager>_organize(text='group by year-month')`"
            ),
        ],
    )

    if not clarification_block:
        usage_examples = "\n".join(
            [
                usage_examples,
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state assumptions.",
            ],
        )

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    clar_sentence = (
        f"Do not ask the user questions in your final response, please only use the `{request_clar_fname}` tool to ask clarifying questions."
        if request_clar_fname
        else (
            "Do not ask the user questions in your final response. Instead, proceed using sensible defaults/best‑guess values and explicitly tell inner tools that these are assumptions/best guesses, not confirmed answers."
        )
    )

    return "\n".join(
        [
            activity_block,
            "You are an assistant specializing in **organizing files across multiple filesystems**.",
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the goal and choose the best approach yourself.",
            clar_sentence,
            "You should attempt to fulfill the organization goal as best you can, even if it seems out of scope.",
            "Use the tools provided to see if you can find any missing context *before* asking the user for clarifications.",
            "",
            f"You have access to {num_filesystems} filesystems.",
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            usage_examples,
            "",
            clar_section,
            "",
            f"Current UTC time is {now()}.",
        ],
    )


def build_simulated_method_prompt(
    method: str,
    user_request: str,
    parent_chat_context: list[dict] | None = None,
) -> str:
    """Return instruction prompt for the *simulated* FileManager/GlobalFileManager.

    This mirrors the guidance style used across other simulated managers.
    """
    import json

    m = (method or "").lower()

    preamble = f"On this turn you are simulating the '{method}' method."

    if m in {"ask", "ask_about_file", "global_ask"}:
        specifics = []
        if m == "ask_about_file":
            specifics.append(
                "Focus strictly on the specified file; mention the filename explicitly.",
            )
        if m == "global_ask":
            specifics.append(
                "Operate at the aggregated, cross-filesystem level. If you mention a filesystem, refer to it by manager class name (e.g. 'LocalFileManager'). Do not rely on alias-prefixed paths.",
            )
        behaviour = " ".join(
            [
                "Answer directly with an imaginary but plausible response about the file(s).",
                "Do NOT ask the human questions; simply produce a final answer.",
                *specifics,
            ],
        )
    elif m in {"organize", "global_organize"}:
        specifics = [
            "Only discuss rename or move operations; do not create or delete files.",
            "Write your response in past tense, summarising what was 'done' as a simulated plan.",
        ]
        if m == "global_organize":
            specifics.append(
                "Operate across manager class names (e.g. 'LocalFileManager') and keep re-organisation within a single root.",
            )
        behaviour = " ".join(
            [
                "Provide a concise organisation plan as if the steps had already been executed.",
                *specifics,
            ],
        )
    else:
        behaviour = (
            "Provide a succinct, plausible response suitable for this simulated method."
        )

    parts: list[str] = [preamble, behaviour, "", f"The user input is:\n{user_request}"]
    if parent_chat_context:
        parts.append(
            f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}",
        )

    return "\n".join(parts)
