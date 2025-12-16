from __future__ import annotations

import json
from typing import Dict, Callable, Optional

from ..common.prompt_helpers import (
    clarification_guidance,
    sig_dict,
    now,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
    render_tools_block,
    render_counts_and_columns,
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


def build_shared_retrieval_usage(tools: Dict[str, Callable]) -> str:
    """Consolidated retrieval/joins guidance (single source of truth)."""
    list_columns_fname = _tool_name(tools, "list_columns")
    tables_overview_fname = _tool_name(tools, "tables_overview")
    filter_files_fname = _tool_name(tools, "filter_files")
    search_files_fname = _tool_name(tools, "search_files")
    reduce_fname = _tool_name(tools, "reduce")
    filter_join_fname = _tool_name(tools, "filter_join")
    search_join_fname = _tool_name(tools, "search_join")
    filter_mjoin_fname = _tool_name(tools, "filter_multi_join")
    search_mjoin_fname = _tool_name(tools, "search_multi_join")

    return "\n".join(
        [
            "Retrieval & Joins",
            "------------------",
            # Context discovery
            f"• Always call `{tables_overview_fname}()` for a global map, or `{tables_overview_fname}(file=...)` for a file-scoped view (ingest-aware).",
            "• Prefer path‑first references: use `<file_path>` for per‑file Content and `<file_path>.Tables.<label>` for per‑file tables.",
            f"• Introspect schemas via `{list_columns_fname}(table='<file_path>')` and `{list_columns_fname}(table='<file_path>.Tables.<label>')`.",
            f"• Return only column names (no types): `{list_columns_fname}(include_types=False)`.",
            "",
            # Path-first targeting
            "Path-first targeting",
            "--------------------",
            "• Do not pass the literal string '<file_path>' to tools.",
            "• Use the exact absolute `<file_path>` as stored in FileRecords and returned by stat/identity helpers.",
            "• Reference contexts directly using `<file_path>` and `<file_path>.Tables.<label>`.",
            "",
            "Examples",
            "--------",
            "• Per‑file mode:",
            "  - tables_overview(file='/abs/path.pdf') → {",
            "      'FileRecords': {...},",
            "      'abs_path_pdf': { 'Content': {...}, 'Tables': { 'Products': {...}, 'Prices': {...} } }",
            "    }",
            "  - Content filter: `tables=['/abs/path.pdf']`",
            "  - Table filter: `tables=['/abs/path.pdf.Tables.Products']`",
            "",
            "• Unified mode:",
            "  - tables_overview(file='/abs/path.pdf') → {",
            "      'FileRecords': {...},",
            "      'UnifiedDocs': { 'Content': {...} },",
            "      'abs_path_pdf': { 'Tables': { 'Products': {...} } }",
            "    }",
            "  - Content lives under the unified label; per‑file tables remain keyed by `<file_path>`.",
            "  - Content filter: `tables=['UnifiedDocs']`",
            "  - Table filter: `tables=['/abs/path.pdf.Tables.Products']`",
            "",
            # Index/multi-context retrieval
            "Index & multi-context retrieval",
            "-------------------------------",
            f"• Exact filtering over the index: `{filter_files_fname}(filter=\"status == 'success'\")`. Prefer `source_uri` when available.",
            f"• Semantic discovery over index fields: `{search_files_fname}(references={{'summary': 'ISO 27001'}}, k=10)`.",
            f"• Scan specific contexts: `{filter_files_fname}(filter=..., tables=['<file_path>', '<file_path>.Tables.<label>'])`.",
            f"• Paginate scans to control volume: `{filter_files_fname}(filter=..., offset=200, limit=100)`.",
            f"• Semantic search inside a specific context: `{search_files_fname}(references={{'content': 'paymentterms'}}, table='<file_path>', filter=\"file_format == 'pdf'\")`.",
            "",
            # Quantitative vs qualitative retrieval strategy
            "Quantitative vs qualitative retrieval strategy",
            "----------------------------------------------",
            "• QUANTITATIVE goals (counts, sums, averages, statistics):",
            f"  - ALWAYS prefer `{reduce_fname}` with appropriate metric(s), key(s), filter, and group_by.",
            f"  - `{reduce_fname}` is the fastest, most efficient path to numeric answers.",
            f"  - Combine multiple `{reduce_fname}` calls if needed (e.g., count + sum for different breakdowns).",
            "  - Do NOT retrieve raw rows just to count/sum them in-memory; let the backend compute it.",
            "• QUALITATIVE goals (inspect records, understand content, find examples):",
            "  - Start with conservative limits: limit ≤ 30 for exploratory queries.",
            "  - Gradually expand if needed: 30 → 50 → 75 → 100 (absolute max).",
            "  - NEVER exceed limit=100 under any circumstances.",
            f"  - Use `offset` for pagination: `{filter_files_fname}(filter=..., offset=0, limit=30)` then step offset by 30.",
            "",
            # Filter discipline & column selection
            "Filter discipline & column selection",
            "------------------------------------",
            "• Choose only the minimal, most relevant columns for filters; avoid broad multi‑column predicates unless necessary.",
            "• Do not OR across many similar fields (e.g., several date columns); pick the best single column based on schema/description and question intent.",
            "• Only combine multiple columns when unavoidable; even then keep limits low (<100) and validate with small pages first.",
            "",
            # Date/time filtering
            "Date/time filtering (temporal columns: date, time, datetime, timestamp – typed or inferred)",
            "-----------------------------------------------------------------------------------------",
            "• Use comparison operators (`>`, `<`, `>=`, `<=`) to define ranges when the exact value is unknown.",
            "• Examples:",
            "  - Range filter: `filter=\"created_at >= '2024-01-01' and created_at < '2024-02-01'\"`",
            "  - After a date: `filter=\"visit_date > '2024-06-15'\"`",
            "  - Before a date: `filter=\"completed_at <= '2024-12-31'\"`",
            "• ONLY use equality (`==`) or `in` on temporal columns when you have the EXACT complete value for that field.",
            "  - Partial or approximate values will NOT match and will return incorrect/empty results.",
            "  - When in doubt, use upper and/or lower bounds with comparison operators instead.",
            "",
            # Numeric aggregations (reduce is the primary tool for quantitative answers)
            "Numeric aggregations (CRITICAL for quantitative queries)",
            "--------------------------------------------------------",
            f"• `{reduce_fname}` is the PRIMARY tool for any quantitative question (count, sum, mean, min, max, median, mode, var, std).",
            f"• ALWAYS use `{reduce_fname}` instead of filtering rows and computing aggregates in-memory.",
            f"• Apply filters directly in `{reduce_fname}` to narrow the dataset before aggregation.",
            f"• Use `group_by` for breakdowns: `{reduce_fname}(table='...', metric='count', keys='id', group_by='category')`",
            "• Examples:",
            f"  - Total count: `{reduce_fname}(table='<file_path>.Tables.<label>', metric='count', keys='id')`",
            f"  - Sum with filter: `{reduce_fname}(table='<file_path>.Tables.<label>', metric='sum', keys='amount', filter=\"status == 'complete'\")`",
            f"  - Grouped average: `{reduce_fname}(table='<file_path>.Tables.<label>', metric='mean', keys='score', group_by='region')`",
            f"  - Multiple metrics: call `{reduce_fname}` multiple times with different metrics/keys as needed.",
            "• If the answer is a number or statistic, `reduce` is almost always the correct choice.",
            "",
            # Dict-based content_id usage
            "Dict-based hierarchical IDs (per-file Content)",
            "-----------------------------------------------",
            "• Per-file Content uses a dict `content_id` to encode hierarchy.",
            "  Examples:",
            "  - document row → {'document': 0}",
            "  - section row → {'document': 0, 'section': 2}",
            "  - paragraph row → {'document': 0, 'section': 2, 'paragraph': 1}",
            "  - sentence row → {'document': 0, 'section': 2, 'paragraph': 1, 'sentence': 3}",
            "  - table row → {'document': 0, 'section': 2, 'table': 0}",
            "  - image row → {'document': 0, 'section': 2, 'image': 0}",
            "• Filter using Pythonic dict access:",
            f"  `{filter_files_fname}(filter=\"content_type == 'table' and content_id.get('section') == 2\", tables=['<file_path>'])`",
            f"  `{filter_files_fname}(filter=\"content_id.get('document') == 0 and content_type in ('image','table')\", tables=['<file_path>'])`",
            "",
            # Joins (two tables)
            "Joins (two tables)",
            "-------------------",
            f"• Filter a join result: `{filter_join_fname}(tables=['<file_path>', '<file_path>.Tables.<label>'], join_expr=\"<file_path>.file_id == <file_path>.Tables.<label>.file_id\", select={{'<file_path>.file_id': 'fid', '<file_path>.Tables.<label>.col': 'val'}}, result_where=\"val == 'ok'\")`.",
            f"• Join modes and input filters: `{filter_join_fname}(tables=['A', 'B'], join_expr=\"A.id == B.a_id\", select={{'A.id': 'id', 'B.score': 'score'}}, mode='left', left_where=\"A.active\", right_where=\"B.score > 0\")`.",
            f"• Semantic join search: `{search_join_fname}(tables=['<file_path>', '<file_path>.Tables.<label>'], join_expr=..., select=..., references={{'val': 'target'}}, k=10)`.",
            "Notes: `left_where`/`right_where` filter inputs pre-join; `result_where` filters the projection and may only reference names from `select`.",
            "",
            # Multi-step joins
            "Multi-step joins",
            "-----------------",
            f"• Chain steps with `{filter_mjoin_fname}` / `{search_mjoin_fname}` and use '$prev' to reference the previous step.",
            f"  Example: `{filter_mjoin_fname}(joins=[{{'tables': ['<file_path>.Tables.A', '<file_path>.Tables.B'], 'join_expr': 'A.id == B.id', 'select': {{'A.id': 'id', 'B.score': 'score'}}}}, {{'tables': ['$prev', '<file_path>'], 'join_expr': \"prev.id == <file_path>.file_id\", 'select': {{'prev.score': 'score', '<file_path>.summary': 'summary'}}}}], result_where=\"score > 0\")`",
            f"• Semantic variant: `{search_mjoin_fname}(joins=[...], references={{'summary': 'budget update'}}, k=10)`.",
            "",
            # Patterns / anti-patterns
            "Patterns / Anti‑patterns",
            "-------------------------",
            f"• Pattern: Start with `{tables_overview_fname}` → `{list_columns_fname}` to decide which contexts and columns to use.",
            f"• Pattern: For quantitative answers (counts, sums, stats), use `{reduce_fname}` directly – do NOT fetch rows to count them.",
            "• Pattern: Prefer semantic search for long text; exact filters for ids/labels.",
            "• Pattern: For qualitative exploration, use conservative limits (≤30 initially, max 100) with offset-based pagination.",
            "• Pattern: Choose the minimal set of filter columns based on schema/column descriptions.",
            "• Anti‑pattern: Fetching rows with filter/search just to count or sum them – use `reduce` instead.",
            "• Anti‑pattern: Hand-constructing raw Unify contexts – always pass `<file_path>` or `<file_path>.Tables.<label>` instead.",
            "• Anti‑pattern: Using substring filters for meaning over long text (prefer semantic search).",
            "• Anti‑pattern: Referencing columns in `result_where` not present in `select`.",
            "• Anti‑pattern: Large limits (>100) that flood the context window; paginate instead.",
            "• Anti‑pattern: Using `==` or `in` on temporal columns without the exact complete value.",
            "• Anti‑pattern: Careless multi-column OR predicates across similar fields (e.g., several date columns); pick the single best column unless truly necessary.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Generic block constants for slot-filling prompt composition
# ─────────────────────────────────────────────────────────────────────────────

# Read-only ask block (~500 tokens) - stable, reusable across all clients
GENERIC_FILE_MANAGER_ASK_BLOCK = f"""\
You answer questions using the available data sources by calling tools.
You do not guess values that are not supported by the data.

{read_only_ask_mutation_exit_block()}

Context map
-----------
• Global index: Lightweight index of all files (FileRecords table).
• Per-file Content: Rows representing document hierarchy (document/section/paragraph/sentence/table/image).
• Per-file Tables: Extracted tables from documents (no predefined schema).

Path-first targeting (preferred)
--------------------------------
• Use the fully qualified absolute file path directly as the table reference:
  - Content: `table='<file_path>'` or `tables=['<file_path>']`
  - Per-file table: `table='<file_path>.Tables.<label>'`
• When needed, `tables_overview(file=<file_path>)` reveals the ingest layout.
• In unified mode, Content lives under the unified label (e.g., 'UnifiedDocs'); per-file tables remain under `<file_path>.Tables.<label>`.
• Do NOT pass the literal string '<file_path>' to tools; use the actual file path value.
• Accept absolute paths and provider URIs as-is; do not rewrite them.

Tools & capabilities
--------------------
• list_columns(table): Inspect columns in a table or file context.
• tables_overview(file=None): See what contexts exist for a file path.
• filter_files(filter, tables, offset, limit): Retrieve rows using boolean filters.
• search_files(references, table, filter, k): Semantic search over text fields.
• reduce(table, metric, keys, filter, group_by): Compute aggregates directly.
• filter_join/search_join/filter_multi_join/search_multi_join: Join tables.
• stat(path_or_uri): Check file existence and get canonical URI.
• exists(filename): Check filesystem availability.

Retrieval patterns (ALWAYS apply)
---------------------------------
• Numeric questions (counts, sums, averages): Use `reduce`, not row fetches.
• Text meaning (descriptions, topics): Use `search_files` with semantic references.
• Exact matches (ids, paths, statuses): Use `filter_files` with exact filters.
• Qualitative inspection: Small batches (~30 rows), paginate with `offset`.
• Joins: Single focused join; push filters into join (left_where/right_where).
• Temporal filters: Use comparison operators (>, <, >=, <=) for date ranges.

Parallelism & efficiency
------------------------
• Prefer one comprehensive join/search over many micro-calls when feasible.
• Plan independent checks together and run in parallel when possible.
• Avoid confirmatory re-queries unless new ambiguity arises.
• For large result sets: start with limit ≤ 30, expand gradually (max 100), use offset for pagination.
• NEVER exceed limit=100 under any circumstances.

Answering
---------
• Plan which tables/columns you need before calling tools.
• Minimize tool calls by retrieving exactly what you need.
• Do NOT mention tool names in final answer. Describe in human terms.
• If ambiguous, make reasonable assumption and state it explicitly.
• Always mention the relevant file path(s) or URI(s) in your response.
"""

# Organize block for mutation builders - stable, reusable
GENERIC_FILE_MANAGER_ORGANIZE_BLOCK = """\
You organize files by renaming, moving, or deleting them.
You do not create files or edit file contents.

Context map
-----------
• Global index: FileRecords table containing metadata for all files.
• Use ask() to discover files and their paths before mutating.

Tools & capabilities
--------------------
• ask(text): Discover files before mutating (read-only preflight).
• rename_file(file_id_or_path, new_name): Rename within same folder.
• move_file(file_id_or_path, new_parent_path): Move to different folder.
• delete_file(file_id_or_path): Delete a file (protected files may error).
• sync(file_path): Re-sync after external edits.

Patterns
--------
• Always preflight with ask() before mutations.
• Use fully-qualified absolute paths (starting with '/').
• Accept absolute paths and provider URIs as-is; do not rewrite them.

Parallelism & sequencing
------------------------
• Batch independent operations (e.g., renaming files in different folders).
• Sequence dependent operations carefully (e.g., move then rename).
• Avoid confirmatory re-queries after each mutation; verify end state succinctly.
• Always mention the relevant file path(s) in your response.
"""


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
    business_payload: Optional[BusinessContextPayload] = None,
) -> str:
    """
    Build the system prompt for AdapterFileManager.ask (filesystem-wide Q&A).

    Uses slot-filling pattern:
    1. Business role FIRST (from payload or generic fallback)
    2. Generic capabilities block
    3. Runtime data (tables overview, columns, tool signatures)
    4. Domain rules + retrieval hints (from payload)
    5. Response guidelines (from payload or generic)
    6. Clarification guidance + timestamp
    """
    columns = columns or {}

    exists_fname = _tool_name(tools, "exists")
    list_fname = _tool_name(tools, "list")

    # Require only the core read tools (no ingest_files - this is read-only)
    _require_tools(
        {
            "exists": exists_fname,
            "list": list_fname,
        },
        tools,
    )

    # Tables overview (runtime injection)
    overview_block = table_schemas_json or "{}"

    # Assemble prompt using slot-filling pattern
    parts: list[str] = []

    # LAYER 1: Business role FIRST (primacy effect)
    if business_payload and business_payload.role_description:
        parts.append(business_payload.role_description)
    else:
        parts.append(
            "You are an assistant specializing in retrieving file information and analyzing file contents.",
        )
    parts.append("")
    parts.append(clarification_top_sentence(tools))
    parts.append("")

    # LAYER 2: Generic capabilities block
    parts.append(GENERIC_FILE_MANAGER_ASK_BLOCK)
    parts.append("")

    # LAYER 3: Runtime data
    parts.append("Tables overview")
    parts.append("---------------")
    parts.append(overview_block)
    parts.append("")

    # Trimmed retrieval guidance (essential patterns only)
    parts.append(build_shared_retrieval_usage(tools))
    parts.append("")

    parts.append(
        render_counts_and_columns(
            entity_plural="files",
            count=num_files,
            columns_payload=columns,
        ),
    )
    parts.append("")

    parts.append(render_tools_block(tools))
    parts.append("")

    # LAYER 4: Domain rules + retrieval hints (from payload)
    if business_payload:
        if business_payload.domain_rules:
            parts.append("Domain context & data rules")
            parts.append("---------------------------")
            parts.append(business_payload.domain_rules)
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

    # Runtime: Clarification guidance + timestamp
    parts.append(clarification_guidance(tools))
    parts.append("")
    parts.append(f"Current UTC time is {now()}.")

    return "\n".join(parts)


def build_file_manager_ask_about_file_prompt(
    tools: Dict[str, Callable],
    *,
    table_schemas_json: str | None = None,
    include_activity: bool = True,
    business_payload: Optional[BusinessContextPayload] = None,
) -> str:
    """
    Build the focused system prompt for AdapterFileManager.ask_about_file.

    Uses slot-filling pattern with file-scoped focus. Retains structured
    extraction support (response_format handling).
    """
    stat_fname = _tool_name(tools, "stat")

    overview_block = table_schemas_json or "{}"

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
    if stat_fname:
        parts.append(
            f"Use `{stat_fname}` first to resolve the identifier to a canonical URI and check existence.",
        )
    parts.append("")

    # LAYER 2: Generic capabilities block
    parts.append(GENERIC_FILE_MANAGER_ASK_BLOCK)
    parts.append("")

    # Structured extraction support (RETAIN)
    parts.append("Structured extraction")
    parts.append("---------------------")
    parts.append(
        "• When a response_format is provided, return strictly the requested JSON schema (no prose, no extra keys).",
    )
    parts.append(
        "• Use joins/search over content/tables contexts to collect only what is required to populate the schema.",
    )
    parts.append("")

    # LAYER 3: Runtime data
    parts.append("Tables overview")
    parts.append("---------------")
    parts.append(overview_block)
    parts.append("")

    parts.append(build_shared_retrieval_usage(tools))
    parts.append("")

    parts.append(render_tools_block(tools))
    parts.append("")

    # LAYER 4: Domain rules + retrieval hints (from payload)
    if business_payload:
        if business_payload.domain_rules:
            parts.append("Domain context & data rules")
            parts.append("---------------------------")
            parts.append(business_payload.domain_rules)
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

    # Runtime: Clarification guidance + timestamp
    parts.append(clarification_guidance(tools))
    parts.append("")
    parts.append(f"Current UTC time is {now()}.")

    return "\n".join(parts)


def build_file_manager_organize_prompt(
    tools: Dict[str, Callable],
    *,
    num_files: int = 0,
    columns: Dict[str, str] | None = None,
    table_schemas_json: str | None = None,
    include_activity: bool = True,
    business_payload: Optional[BusinessContextPayload] = None,
) -> str:
    """
    Build the system prompt for AdapterFileManager.organize (rename/move/delete).

    Uses slot-filling pattern for mutation operations.
    """
    ask_fname = _tool_name(tools, "ask")

    # Only require ask for discovery; organize is mutation-focused
    _require_tools({"ask": ask_fname}, tools)

    overview_block = table_schemas_json or "{}"

    # Assemble prompt using slot-filling pattern
    parts: list[str] = []

    # LAYER 1: Business role FIRST (primacy effect)
    if business_payload and business_payload.role_description:
        parts.append(business_payload.role_description)
    else:
        parts.append(
            "You are an assistant specializing in organizing files and folders in the filesystem.",
        )
    parts.append("")
    parts.append(clarification_top_sentence(tools))
    parts.append("")

    # LAYER 2: Generic organize block
    parts.append(GENERIC_FILE_MANAGER_ORGANIZE_BLOCK)
    parts.append("")

    # LAYER 3: Runtime data
    parts.append(f"There are currently {num_files} files indexed.")
    parts.append("")

    parts.append("Tables overview")
    parts.append("---------------")
    parts.append(overview_block)
    parts.append("")

    parts.append(render_tools_block(tools))
    parts.append("")

    # LAYER 4: Domain rules (from payload)
    if business_payload and business_payload.domain_rules:
        parts.append("Domain context & data rules")
        parts.append("---------------------------")
        parts.append(business_payload.domain_rules)
        parts.append("")

    # Runtime: Clarification guidance + timestamp
    parts.append(clarification_guidance(tools))
    parts.append("")
    parts.append(f"Current UTC time is {now()}.")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# GlobalFileManager prompt builders (multi-filesystem)
# ─────────────────────────────────────────────────────────────────────────────


def build_global_file_manager_ask_prompt(
    tools: Dict[str, Callable],
    *,
    num_filesystems: int = 0,
    include_activity: bool = True,
    business_payload: Optional[BusinessContextPayload] = None,
) -> str:
    """
    Build the system prompt for the GlobalFileManager.ask method.

    Uses slot-filling pattern for multi-filesystem queries.
    """
    # Assemble prompt using slot-filling pattern
    parts: list[str] = []

    # LAYER 1: Business role FIRST
    if business_payload and business_payload.role_description:
        parts.append(business_payload.role_description)
    else:
        parts.append(
            "You are an assistant specializing in managing and querying multiple filesystems.",
        )
    parts.append("")
    parts.append(clarification_top_sentence(tools))
    parts.append("")

    # LAYER 2: Generic capabilities (trimmed for global context)
    parts.append(GENERIC_FILE_MANAGER_ASK_BLOCK)
    parts.append("")

    # Multi-filesystem specific guidance
    parts.append(f"You have access to {num_filesystems} filesystems.")
    parts.append(
        "Accept absolute paths and provider URIs; when delegating to a specific filesystem manager, pass identifiers as-is.",
    )
    parts.append("")

    parts.append(render_tools_block(tools))
    parts.append("")

    # LAYER 4: Domain rules (from payload)
    if business_payload and business_payload.domain_rules:
        parts.append("Domain context & data rules")
        parts.append("---------------------------")
        parts.append(business_payload.domain_rules)
        parts.append("")

    # Runtime: Clarification guidance + timestamp
    parts.append(clarification_guidance(tools))
    parts.append("")
    parts.append(f"Current UTC time is {now()}.")

    return "\n".join(parts)


def build_global_file_manager_organize_prompt(
    tools: Dict[str, Callable],
    *,
    num_filesystems: int = 0,
    include_activity: bool = True,
    business_payload: Optional[BusinessContextPayload] = None,
) -> str:
    """
    Build the system prompt for the GlobalFileManager.organize method.

    Uses slot-filling pattern for multi-filesystem mutations.
    """
    # Assemble prompt using slot-filling pattern
    parts: list[str] = []

    # LAYER 1: Business role FIRST
    if business_payload and business_payload.role_description:
        parts.append(business_payload.role_description)
    else:
        parts.append(
            "You are an assistant specializing in organizing files across multiple filesystems.",
        )
    parts.append("")
    parts.append(clarification_top_sentence(tools))
    parts.append("")

    # LAYER 2: Generic organize block
    parts.append(GENERIC_FILE_MANAGER_ORGANIZE_BLOCK)
    parts.append("")

    # Multi-filesystem specific guidance
    parts.append(f"You have access to {num_filesystems} filesystems.")
    parts.append(
        "Keep rename/move/delete within a single filesystem (cross-filesystem transfer is not implemented).",
    )
    parts.append("")

    parts.append(render_tools_block(tools))
    parts.append("")

    # LAYER 4: Domain rules (from payload)
    if business_payload and business_payload.domain_rules:
        parts.append("Domain context & data rules")
        parts.append("---------------------------")
        parts.append(business_payload.domain_rules)
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
    """Return instruction prompt for the *simulated* FileManager/GlobalFileManager.

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
            "Only perform and summarise rename, move, or delete operations; do not create files or folders.",
            "Write your response in past tense, summarising what was done.",
        ]
        if m == "global_organize":
            specifics.append(
                "Operate across manager class names (e.g. 'LocalFileManager') and keep re-organisation within a single root.",
            )
        behaviour = " ".join(
            [
                "Provide a concise summary of the mutations you executed.",
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
