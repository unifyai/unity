from __future__ import annotations
from typing import Dict, Callable, Optional

from ..common.prompt_helpers import (
    clarification_guidance,
    parallelism_guidance,
    sig_dict,
    now,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
    render_tools_block,
    clarification_top_sentence,
    PromptParts,
)
from ..common.read_only_ask_guard import read_only_ask_mutation_exit_block
from ..common.business_context import BusinessContextPayload

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


def build_cross_tool_orchestration(tools: Dict[str, Callable]) -> str:
    """
    Slim cross-tool orchestration guidance for FileManager system prompt.

    Tool-specific details (syntax, examples) now live in rich tool docstrings.
    This function provides only the high-level decision framework.
    """
    describe_fname = _tool_name(tools, "describe")
    list_columns_fname = _tool_name(tools, "list_columns")

    return "\n".join(
        [
            "Cross-tool orchestration",
            "------------------------",
            "",
            "Discovery phase (ALWAYS start here):",
            f"• `{describe_fname}` → get context paths and schemas (REQUIRED before filter/search/reduce)",
            f"• `{list_columns_fname}` → detailed column inspection for a specific context",
            "",
            "Retrieval phase (use context paths from describe):",
            "• Counts/sums/statistics → `reduce`",
            "• Semantic meaning/topics → `search_files`",
            "• Exact matches (ids, statuses) → `filter_files`",
            "• Cross-table correlation → `filter_join`, `search_join`",
            "• Visual charts/plots → `visualize`",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Generic block constants for slot-filling prompt composition
# ─────────────────────────────────────────────────────────────────────────────

# Read-only ask block - slim version (~150 tokens)
# Tool-specific details (syntax, examples, anti-patterns) are in tool docstrings
GENERIC_FILE_MANAGER_ASK_BLOCK = f"""\
You answer questions using the available data sources by calling tools.
You do not guess values that are not supported by the data.

{read_only_ask_mutation_exit_block()}

Context map
-----------
• Global index: FileRecords table (lightweight metadata for all files).
• Per-file Content: Retrieval surface for each file.
  - For PDFs/DOCX: hierarchical rows (document/section/paragraph/sentence).
  - For XLSX/CSV: `sheet` rows + `table` catalog rows (table profiles + summaries for discovery).
• Per-file Tables: `/Tables/<label>` contexts containing the actual table rows.

Answering
---------
• Discover schema before querying (use discovery tools).
• Minimize tool calls by retrieving exactly what you need.
• Do NOT mention tool names in final answer.
• If ambiguous, state your assumption explicitly.
• Always mention relevant file path(s) in your response.
"""

# ─────────────────────────────────────────────────────────────────────────────
# FileManager prompt builders (single filesystem)
# ─────────────────────────────────────────────────────────────────────────────


def build_file_manager_ask_about_file_prompt(
    tools: Dict[str, Callable],
    *,
    include_activity: bool = True,
    business_payload: Optional[BusinessContextPayload] = None,
) -> PromptParts:
    """
    Build the focused system prompt for AdapterFileManager.ask_about_file.

    Uses slot-filling pattern with file-scoped focus. Tool-specific details
    live in docstrings. Schema discovery via tools.
    """
    filter_files_fname = _tool_name(tools, "filter_files")
    search_files_fname = _tool_name(tools, "search_files")
    describe_fname = _tool_name(tools, "describe")

    # Require core ask_about_file tools
    _require_tools(
        {
            "filter_files": filter_files_fname,
            "search_files": search_files_fname,
            "describe": describe_fname,
        },
        tools,
    )

    # Assemble prompt using PromptParts for structured output
    parts = PromptParts()

    # LAYER 1: Business role FIRST (primacy effect)
    if business_payload and business_payload.role_description:
        parts.add(business_payload.role_description)
    else:
        parts.add(
            "You are an assistant specializing in analyzing the content of a specific file.",
        )
    parts.add(clarification_top_sentence(tools))

    # File-scoped focus guidance
    parts.add(
        "Important: When calling tools, use the filename exactly as provided in the user message. Do not construct or modify file paths.",
    )
    if describe_fname:
        parts.add(
            f"ALWAYS call `{describe_fname}` first to get exact context paths before using filter/search/reduce.",
            separator=False,
        )

    # LAYER 2: Generic capabilities block (slim)
    parts.add(GENERIC_FILE_MANAGER_ASK_BLOCK)

    # Structured extraction support
    structured_extraction = "\n".join(
        [
            "Structured extraction",
            "---------------------",
            "• When a response_format is provided, return strictly the requested JSON schema (no prose, no extra keys).",
            "• Use joins/search over content/tables contexts to collect only what is required to populate the schema.",
        ],
    )
    parts.add(structured_extraction)

    # LAYER 3: Cross-tool orchestration (slim)
    parts.add(build_cross_tool_orchestration(tools))

    parts.add(render_tools_block(tools))

    # LAYER 4: Domain context (business-only, from payload)
    if business_payload:
        if business_payload.domain_rules:
            domain_block = "\n".join(
                [
                    "Domain context & data rules",
                    "---------------------------",
                    business_payload.domain_rules,
                ],
            )
            parts.add(domain_block)

        data_overview = getattr(business_payload, "data_overview", None)
        if data_overview:
            parts.add(data_overview)

        if business_payload.retrieval_hints:
            parts.add(business_payload.retrieval_hints)

    # LAYER 5: Response guidelines (from payload or generic)
    if business_payload and business_payload.response_guidelines:
        response_block = "\n".join(
            [
                "Answering guidelines",
                "--------------------",
                business_payload.response_guidelines,
            ],
        )
        parts.add(response_block)

    # LAYER 6: Parallelism guidance
    parts.add(parallelism_guidance())

    # Runtime: Clarification guidance + timestamp (dynamic content)
    parts.add(clarification_guidance(tools), static=False)
    parts.add(f"Current UTC time is {now()}.", static=False)

    return parts


def build_simulated_method_prompt(
    method: str,
    user_request: str,
    parent_chat_context: list[dict] | None = None,
    business_payload: Optional[BusinessContextPayload] = None,
) -> str:
    """Return instruction prompt for the *simulated* FileManager.

    This mirrors the guidance style used across other simulated managers.
    Optionally accepts business_payload for consistency with production builders.
    """
    import json

    m = (method or "").lower()

    # Use business role if provided, otherwise use generic preamble
    if business_payload and business_payload.role_description:
        preamble = f"{business_payload.role_description}\n\nOn this turn you are simulating the '{method}' method."
    else:
        preamble = f"On this turn you are simulating the '{method}' method."

    if m == "ask_about_file":
        behaviour = " ".join(
            [
                "Answer directly with an imaginary but plausible response about the file.",
                "Do NOT ask the human questions; simply produce a final answer.",
                "Focus strictly on the specified file; mention the filename explicitly.",
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
