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
    tables_overview_fname = _tool_name(tools, "tables_overview")
    list_columns_fname = _tool_name(tools, "list_columns")
    file_info_fname = _tool_name(tools, "file_info")
    schema_explain_fname = _tool_name(tools, "schema_explain")

    return "\n".join(
        [
            "Cross-tool orchestration",
            "------------------------",
            "",
            "Discovery phase (understand what data exists):",
            f"• `{file_info_fname}` → file status, ingest mode, storage layout",
            f"• `{tables_overview_fname}` → available contexts (global index or per-file)",
            f"• `{list_columns_fname}` → column names for a specific table",
            f"• `{schema_explain_fname}` → natural-language schema explanation",
            "",
            "Retrieval phase (choose based on goal):",
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
) -> str:
    """
    Build the focused system prompt for AdapterFileManager.ask_about_file.

    Uses slot-filling pattern with file-scoped focus. Tool-specific details
    live in docstrings. Schema discovery via tools.
    """
    filter_files_fname = _tool_name(tools, "filter_files")
    search_files_fname = _tool_name(tools, "search_files")
    schema_explain_fname = _tool_name(tools, "schema_explain")
    tables_overview_fname = _tool_name(tools, "tables_overview")
    file_info_fname = _tool_name(tools, "file_info")

    # Require core ask_about_file tools
    _require_tools(
        {
            "filter_files": filter_files_fname,
            "search_files": search_files_fname,
            "schema_explain": schema_explain_fname,
            "tables_overview": tables_overview_fname,
        },
        tools,
    )

    # Assemble prompt using slot-filling pattern
    parts: list[str] = []

    # LAYER 1: Business role FIRST (primacy effect)
    if business_payload and business_payload.role_description:
        parts.append(business_payload.role_description)
    else:
        parts.append(
            "You are an assistant specializing in analyzing the content of a specific file.",
        )
    parts.append("")
    parts.append(clarification_top_sentence(tools))
    parts.append("")

    # File-scoped focus guidance
    parts.append(
        "Important: When calling tools, use the filename exactly as provided in the user message. Do not construct or modify file paths.",
    )
    if file_info_fname:
        parts.append(
            f"Use `{file_info_fname}` to check file status, ingest mode, and storage layout.",
        )
    parts.append("")

    # LAYER 2: Generic capabilities block (slim)
    parts.append(GENERIC_FILE_MANAGER_ASK_BLOCK)
    parts.append("")

    # Structured extraction support
    parts.append("Structured extraction")
    parts.append("---------------------")
    parts.append(
        "• When a response_format is provided, return strictly the requested JSON schema (no prose, no extra keys).",
    )
    parts.append(
        "• Use joins/search over content/tables contexts to collect only what is required to populate the schema.",
    )
    parts.append("")

    # LAYER 3: Cross-tool orchestration (slim)
    parts.append(build_cross_tool_orchestration(tools))
    parts.append("")

    parts.append(render_tools_block(tools))
    parts.append("")

    # LAYER 4: Domain context (business-only, from payload)
    if business_payload:
        if business_payload.domain_rules:
            parts.append("Domain context & data rules")
            parts.append("---------------------------")
            parts.append(business_payload.domain_rules)
            parts.append("")

        data_overview = getattr(business_payload, "data_overview", None)
        if data_overview:
            parts.append(data_overview)
            parts.append("")

        if business_payload.retrieval_hints:
            parts.append(business_payload.retrieval_hints)
            parts.append("")

    # LAYER 5: Response guidelines (from payload or generic)
    if business_payload and business_payload.response_guidelines:
        parts.append("Answering guidelines")
        parts.append("--------------------")
        parts.append(business_payload.response_guidelines)
        parts.append("")

    # LAYER 6: Parallelism guidance
    parts.append(parallelism_guidance())
    parts.append("")

    # Runtime: Clarification guidance + timestamp
    parts.append(clarification_guidance(tools))
    parts.append("")
    parts.append(f"Current UTC time is {now()}.")

    return "\n".join(parts)


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
