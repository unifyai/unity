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
from .types.config import FilePipelineConfig as _FM_FilePipelineConfig

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
            # Result volume management & pagination
            "Result volume management & pagination",
            "-------------------------------------",
            "• Start with conservative limits (e.g., limit=50); increase to 100 only if strictly needed.",
            f"• Prefer paginating with `offset` instead of raising limits: `{filter_files_fname}(filter=..., offset=0, limit=50)` then step offset by 50.",
            "• Avoid very large limits (>100); they bloat payloads and the model context.",
            "",
            # Filter discipline & column selection
            "Filter discipline & column selection",
            "------------------------------------",
            "• Choose only the minimal, most relevant columns for filters; avoid broad multi‑column predicates unless necessary.",
            "• Do not OR across many similar fields (e.g., several date columns); pick the best single column based on schema/description and question intent.",
            "• Only combine multiple columns when unavoidable; even then keep limits low (<100) and validate with small pages first.",
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
            "• Pattern: Prefer semantic search for long text; exact filters for ids/labels.",
            "• Pattern: Use small limits (50→100 max) with offset-based pagination to manage volume.",
            "• Pattern: Choose the minimal set of filter columns based on schema/column descriptions.",
            "• Anti‑pattern: Hand-constructing raw Unify contexts – always pass `<file_path>` or `<file_path>.Tables.<label>` instead.",
            "• Anti‑pattern: Using substring filters for meaning over long text (prefer semantic search).",
            "• Anti‑pattern: Referencing columns in `result_where` not present in `select`.",
            "• Anti‑pattern: Large limits (>100) that flood the context window; paginate instead.",
            "• Anti‑pattern: Careless multi-column OR predicates across similar fields (e.g., several date columns); pick the single best column unless truly necessary.",
        ],
    )


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
    business_context: str | None = None,
) -> str:
    """
    Build the system prompt for AdapterFileManager.ask (filesystem-wide Q&A).
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    columns = columns or {}

    exists_fname = _tool_name(tools, "exists")
    stat_fname = _tool_name(tools, "stat")
    parse_fname = _tool_name(tools, "parse")
    list_fname = _tool_name(tools, "list")
    filter_files_fname = _tool_name(tools, "filter_files")
    search_files_fname = _tool_name(tools, "search_files")
    list_columns_fname = _tool_name(tools, "list_columns")
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
        "─ Identity & Paths ─",
        "• Absolute paths and provider URIs are accepted; do not rewrite user-provided identifiers.",
        "• The canonical identity is the provider URI (e.g., local:///abs/path, gdrive://<id>).",
        "",
        "─ Columns ─",
        f"• Inspect schema{'' if not list_columns_fname else ''}",
    ]
    if list_columns_fname:
        usage_lines.append(f"  `{list_columns_fname}()`")
        usage_lines.append(
            "  Note: per-file Content includes a dict `content_id` capturing document/section/paragraph/sentence/table/image indices.",
        )
        usage_lines.append(
            "  Access with Pythonic dict ops, e.g., content_id.get('section') == 2.",
        )
        usage_lines += [
            "  Path‑first targeting (preferred):",
            "  • Use the fully qualified absolute file path directly as the table reference:",
            "    - Content: `table='<file_path>'`",
            "    - Per‑file table: `table='<file_path>.Tables.<label>'`",
            "  • When needed, `tables_overview(file=<file_path>)` reveals the ingest layout.",
            "  • In unified mode, Content lives under the unified label; per‑file tables remain under `<file_path>.Tables.<label>`.",
        ]
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
        f"• Use `{exists_fname}` to check filesystem availability before operations.",
        "• Use `tables_overview()` for an index-wide map, or `tables_overview(file=...)` for a specific file.",
        "• If the user supplies an explicit path, prefer direct, path-first targeting instead of broad discovery.",
        "  - Reference Content with `table='<file_path>'` and per‑file tables with `table='<file_path>.Tables.<label>'`.",
        "  - Keep discovery lightweight (avoid full scans); use semantic search only when the path is unknown.",
    ]
    if stat_fname:
        usage_lines += [
            f"• Prefer `{stat_fname}` for a unified view (filesystem vs index existence) before parsing.",
            f"  - `{stat_fname}(path_or_uri)` → returns canonical_uri (provider URI) and both existence flags.",
            f"  - Use canonical_uri from `{stat_fname}` when filtering the index to avoid mismatches.",
        ]
    if search_files_fname:
        usage_lines.append(
            f"• Prefer `{search_files_fname}` for semantic/topic-oriented discovery when available.",
        )
    if filter_files_fname:
        usage_lines.append(
            f"• Use `{filter_files_fname}` for exact/boolean/id-based filtering over filenames/metadata/ids.",
        )

    # Tool usage summary (only include tools that are available)
    usage_lines += [
        "",
        "─ Tool usage summary ─",
    ]
    if exists_fname:
        usage_lines.append(
            f"• `{exists_fname}(filename)` → filesystem-only existence (no index).",
        )
    if stat_fname:
        usage_lines.append(
            f"• `{stat_fname}(path_or_uri)` → returns {{canonical_uri, filesystem_exists, indexed_exists, parsed_status}}.",
        )
    if list_fname:
        usage_lines.append(
            f"• `{list_fname}()` → inventory listing from the filesystem adapter (not the index).",
        )
    if filter_files_fname:
        usage_lines.append(
            f"• `{filter_files_fname}(filter=...)` → filter the parsed/indexed rows (e.g., status, file_format, source_uri, id).",
        )
    if search_files_fname:
        usage_lines.append(
            f"• `{search_files_fname}(references=..., k=10)` → semantic discovery over indexed fields (e.g., summary, key_topics).",
        )
    if parse_fname:
        usage_lines.append(
            f"• `{parse_fname}(filenames=[...])` → parse then ingest; local files in-place, remote files via temp export (identity unchanged).",
        )
    # Options schema dump (for parse config)
    try:
        _opts_schema_json = json.dumps(
            _FM_FilePipelineConfig.model_json_schema(),
            indent=2,
        )
    except Exception:
        _opts_schema_json = "{}"
    usage_lines += [
        "",
        "─ Parse Options Schema (config) ─",
        _opts_schema_json,
    ]
    # Structured extraction guidance
    ask_about_file_fname = _tool_name(tools, "ask_about_file")
    usage_lines += [
        "",
        "─ Structured extraction ─",
        "• For strict structured outputs, call the file-scoped tool with a response_format.",
        (
            f"  `{ask_about_file_fname}(file_path='/path/to/file.pdf', question='Extract KPIs', response_format=MyPydanticModel)`"
            if ask_about_file_fname
            else "  `<SomeFileManager>_ask_about_file(..., response_format=MyPydanticModel)`"
        ),
        "• When response_format is provided: output JSON only, no prose, no extra keys.",
    ]
    # Join/retrieval usage is centralized to avoid duplication

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
        "",
        "─ Filtering the index reliably ─",
        "• Prefer filtering by source_uri (canonical identity) when available.",
        "  - Example: filter_files(filter=\"source_uri == 'local:///abs/path/to/file.pdf'\")",
        "• If source_uri is not known, filter by file_path as a fallback.",
        "  - Example: filter_files(filter=\"file_path == '/docs/file.pdf'\")",
        "• Resolve canonical_uri first with stat when unsure, then filter by source_uri.",
        "  - Example: uri = stat('/abs/path/report.pdf').canonical_uri → filter_files(filter=f\"source_uri == '{uri}'\")",
        "",
        "─ Filtering per-file Content by hierarchy ─",
        "• Use dict access on `content_id`:",
        "  - Example: filter_files(filter=\"content_type == 'sentence' and content_id.get('paragraph') == 3\", tables=['<file_path>'])",
        "  - Example: filter_files(filter=\"content_id.get('section') == 1 and content_type in ('image','table')\", tables=['<file_path>'])",
        "",
        "─ Path-first targeting ─",
        "• Use the exact absolute `<file_path>` as stored in FileRecords and returned by stat/identity helpers.",
        "• Do not pass the literal string '<file_path>' to tools; use the actual file path value.",
        "• When needed, call `tables_overview(file=<file_path>)` to see the ingest layout.",
        "  - Per‑file mode: Content at `<file_path>`, Tables at `<file_path>.Tables.<label>`",
        "  - Unified mode: Content at unified label (e.g., 'UnifiedDocs'), Tables at `<file_path>.Tables.<label>`",
        "",
        "─ Parallelism and single‑call preference ─",
        "• Prefer one comprehensive join or search call over several micro-calls when feasible.",
        "• When multiple independent checks are needed (e.g., stat on multiple files), plan them and run together.",
        "• Avoid confirmatory re‑queries unless new ambiguity arises.",
        "• Control result volume: start with limit=50 and paginate with offset; only step to 100 if needed.",
        "• Avoid large limits (>100) and broad multi‑column filter predicates; choose the minimal relevant columns.",
        "",
        "Anti‑patterns to avoid",
        "---------------------",
        "• Rewriting or normalising user-provided identifiers (paths/URIs).",
        "• Calling parse before checking stat/indexed existence.",
        (
            f"• Using `{filter_files_fname}` for semantic meaning over large text – prefer `{search_files_fname}` when available."
            if search_files_fname
            else "• Using filter_files for semantic meaning over large text – prefer semantic search when available."
        ),
        "• Re-querying the same rows immediately to reconfirm facts without new constraints.",
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

    parts = [
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
            "Please always mention the relevant file path(s) or URI(s) in your response; never rewrite them.",
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
            build_shared_retrieval_usage(tools),
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
            f"Current UTC time is {now()}.",
        ],
    ]
    # Optional business context block
    if isinstance(business_context, str) and business_context.strip():
        parts.append(
            [
                "",
                "Business context",
                "----------------",
                business_context.strip(),
            ],
        )
    return "\n".join([ln for block in parts for ln in block])


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
    stat_fname = _tool_name(tools, "stat")

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
    filter_join_fname = _tool_name(tools, "filter_join")
    search_join_fname = _tool_name(tools, "search_join")
    filter_mjoin_fname = _tool_name(tools, "filter_multi_join")
    search_mjoin_fname = _tool_name(tools, "search_multi_join")

    overview_block = table_schemas_json or "{}"

    # Tool usage summary (only include tools that are available)
    tool_usage_lines = [
        "",
        "Tool usage summary",
        "-------------------",
    ]
    if stat_fname:
        tool_usage_lines.append(
            f"• `{stat_fname}(path_or_uri)` → returns canonical_uri and both existence flags.",
        )
    if parse_fname:
        tool_usage_lines.append(
            f"• `{parse_fname}(filenames=[filename])` → parse then ingest (local in-place; remote via temp export).",
        )
    if _tool_name(tools, "filter_files"):
        tool_usage_lines.append(
            "• `filter_files(filter=...)` → filter parsed/indexed rows (prefer source_uri; fallback to file_path).",
        )
    if _tool_name(tools, "search_files"):
        tool_usage_lines.append(
            "• `search_files(references=..., k=10)` → semantic discovery across indexed fields.",
        )
    # Dict-based content_id usage for file-scoped queries
    tool_usage_lines += [
        "• Per-file Content rows include a dict `content_id` encoding hierarchy.",
        "  - Filter examples:",
        "    `filter_files(filter=\"content_type == 'table' and content_id.get('section') == 2\", tables=['<file_path>'])`",
        "    `filter_files(filter=\"content_id.get('document') == 0 and content_type == 'image'\", tables=['<file_path>'])`",
        "• Path-first targeting: use the exact absolute `<file_path>` from the user or stat/identity helpers.",
        "  - Per‑file mode: Content: tables=['<file_path>']; table: tables=['<file_path>.Tables.<label>']",
        "  - Unified mode: Content lives under unified label (e.g., 'UnifiedDocs'); Tables: tables=['<file_path>.Tables.<label>']",
    ]
    # Argument combinations (file-scoped focus)
    if _tool_name(tools, "tables_overview"):
        tool_usage_lines.append(
            "• `tables_overview(file='<file_path>')` → file-scoped overview; use logical names from this map in joins.",
        )
    if _tool_name(tools, "list_columns"):
        tool_usage_lines.append(
            "• `list_columns(table='<file_path>.Tables.<label>')` and `list_columns(include_types=False)` for compact headers.",
        )
    if _tool_name(tools, "filter_files"):
        tool_usage_lines.append(
            "• `filter_files(filter=..., tables=['<file_path>', '<file_path>.Tables.<label>'], offset=0, limit=50)` to scan file Content/Tables.",
        )
    if _tool_name(tools, "search_files"):
        tool_usage_lines.append(
            "• `search_files(references={'content': 'revenue'}, table='<file_path>', filter=\"quarter == 'Q1'\", k=5)`.",
        )
    if any(
        [
            _tool_name(tools, "filter_join"),
            _tool_name(tools, "search_join"),
            _tool_name(tools, "filter_multi_join"),
            _tool_name(tools, "search_multi_join"),
        ],
    ):
        tool_usage_lines.append(
            "• Join tools operate over this file's per-file context and extracted tables (when present).",
        )
        tool_usage_lines.append(
            "• Example two-table join with filters: `filter_join(tables=['<file_path>.Tables.A','<file_path>.Tables.B'], join_expr='A.id == B.id', select={'A.id':'id','B.total':'total'}, left_where='A.ok', right_where='B.total > 0', result_where='total > 100')`.",
        )
        tool_usage_lines.append(
            "• Example chained join using $prev: `filter_multi_join(joins=[{'tables':['<file_path>.Tables.A','<file_path>.Tables.B'],'join_expr':'A.id == B.id','select':{'A.id':'id'}},{'tables':['$prev','<file_path>'],'join_expr':'id == <file_path>.file_id','select':{'id':'id','<file_path>.summary':'summary'}}])`.",
        )

    anti_patterns_lines = [
        "",
        "Anti‑patterns to avoid",
        "---------------------",
        "• Rewriting or normalising user-provided identifiers (paths/URIs).",
        "• Parsing without first checking stat/index existence when the data may already be present.",
        "• Using substring filters for rich text – prefer semantic search over indexed fields when available.",
    ]

    parallelism_lines = [
        "",
        "Parallelism and single‑call preference",
        "-------------------------------------",
        "• Prefer a single comprehensive join/search over many micro-calls when feasible.",
        "• Plan independent checks together (e.g., stat + file-scoped overview) rather than serial drip calls.",
        "• Avoid confirmatory re-queries unless new ambiguity arises.",
        "• Control result volume: start with limit=50 and paginate via offset; only step to 100 if needed.",
        "• Avoid large limits (>100) and careless multi‑column filters; pick the minimal relevant columns.",
    ]

    # Structured extraction block when response_format is present (descriptive guidance)
    structured_block = "\n".join(
        [
            "",
            "Structured extraction",
            "---------------------",
            "• When a response_format is provided, return strictly the requested JSON schema (no prose, no extra keys).",
            "• Use joins/search over content_ref/tables_ref contexts to collect only what is required to populate the schema.",
        ],
    )

    # Options schema dump
    try:
        _opts_schema_json = json.dumps(
            _FM_FilePipelineConfig.model_json_schema(),
            indent=2,
        )
    except Exception:
        _opts_schema_json = "{}"

    return "\n".join(
        [
            "You are an assistant specializing in **analyzing the content of a specific file**.",
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
            clar_sentence,
            "You should attempt to answer *any* question as best you can, even if it seems out of scope.",
            "Use the tools provided to see if you can find any missing context *before* asking the user for clarifications.",
            "Please always mention the relevant file path or URI in your response.",
            (
                f"Use `{stat_fname}` first to resolve the identifier to a canonical URI and to check existence in both the filesystem and the index."
                if stat_fname
                else ""
            ),
            (
                "Use the canonical URI returned by stat to filter the index when needed (e.g., \"source_uri == '<uri>'\")."
                if stat_fname
                else ""
            ),
            "",
            "Important: When calling tools, use the filename exactly as provided in the user message. Do not construct or modify file paths.",
            *parse_guidance_lines,
            "",
            "Tables overview",
            "----------------",
            overview_block,
            build_shared_retrieval_usage(tools),
            "",
            "Parse Options Schema (config)",
            "-----------------------------",
            _opts_schema_json,
            structured_block,
            *tool_usage_lines,
            *anti_patterns_lines,
            *parallelism_lines,
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            clar_section,
            "",
            f"Current UTC time is {now()}.",
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
    Build the system prompt for AdapterFileManager.organize (rename/move/delete).
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    columns = columns or {}

    ask_fname = _tool_name(tools, "ask")
    rename_file_fname = _tool_name(tools, "rename_file") or _tool_name(tools, "rename")
    move_file_fname = _tool_name(tools, "move_file") or _tool_name(tools, "move")
    delete_file_fname = _tool_name(tools, "delete_file") or _tool_name(tools, "delete")
    sync_fname = _tool_name(tools, "sync") or _tool_name(tools, "_sync")
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
            f"• Rename a file: `{rename_file_fname}(file_id_or_path='old.txt', new_name='new.txt')` or `{rename_file_fname}(file_id_or_path=123, new_name='new.txt')`",
        )
    if move_file_fname:
        operation_tools.append(
            f"• Move a file: `{move_file_fname}(file_id_or_path='file.txt', new_parent_path='/dest/')` or `{move_file_fname}(file_id_or_path=123, new_parent_path='/dest/')`",
        )
    if delete_file_fname:
        operation_tools.append(
            f"• Delete a file: `{delete_file_fname}(file_id_or_path='file.txt')` or `{delete_file_fname}(file_id_or_path=123)`",
        )
    if sync_fname:
        operation_tools.append(
            f"• Sync a file with filesystem changes: `{sync_fname}(file_path='/docs/report.pdf')`",
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
        f"• When you have explicit file paths from the user, pass them directly to `{ask_fname}`:",
        f"  `{ask_fname}(text='What is the status of /abs/path/to/file.pdf?')`",
        f"• Always use fully-qualified absolute paths when calling `{ask_fname}`; do not rewrite or normalize paths.",
    ]

    usage_lines += [
        "",
        "─ File Organization Operations (if supported) ─",
        "\n".join(operation_tools),
        "",
        "─ Argument combinations (mutations) ─",
        (
            f"• Rename within same folder: `{rename_file_fname}(file_id_or_path='a.txt', new_name='b.txt')` or `{rename_file_fname}(file_id_or_path=42, new_name='b.txt')`"
            if rename_file_fname
            else ""
        ),
        "  - new_name is a filename only (no directory components).",
        "  - file_id_or_path accepts either file_id (int) or fully-qualified file_path (str).",
        (
            f"• Rename with extension change: `{rename_file_fname}(file_id_or_path='data.csv', new_name='data_2024.csv')`"
            if rename_file_fname
            else ""
        ),
        (
            f"• Move into subfolder: `{move_file_fname}(file_id_or_path='a.txt', new_parent_path='/archive/')` or `{move_file_fname}(file_id_or_path=42, new_parent_path='/archive/')`"
            if move_file_fname
            else ""
        ),
        "  - new_parent_path must be a directory path; do not include the filename here.",
        (
            f"• Delete by file_path or file_id (protected files may raise PermissionError): `{delete_file_fname}(file_id_or_path='file.txt')` or `{delete_file_fname}(file_id_or_path=42)`"
            if delete_file_fname
            else ""
        ),
        (
            f"• Re-sync after external edits (purge+re-ingest): `{sync_fname}(file_path='/docs/report.pdf')`"
            if sync_fname
            else ""
        ),
        "",
        "─ Identity & Preflight ─",
        "• Accept absolute paths and provider URIs as-is; do not rewrite them.",
        "• CRITICAL: Always use fully-qualified absolute paths (starting with '/') when calling mutation tools.",
        "  - If the user provides a relative path, resolve it to absolute first using stat() or list().",
        "  - Example: stat('file.txt') → use the canonical_uri or resolve to absolute path before rename/move/delete.",
        f"• Use `{ask_fname}` to preflight with read-only checks (e.g., list, stat via ask-surface) before mutating.",
        f"• When calling `{ask_fname}`, always pass fully-qualified absolute paths in your query text if available.",
        "",
        "─ Parallelism & Sequencing ─",
        "• Group independent operations (e.g., renaming several files in different folders) and execute in a minimal number of calls.",
        "• Sequence dependent operations carefully (e.g., move then rename within the new directory).",
        "• Avoid confirmatory re-queries after each single mutation unless needed; verify the end state succinctly.",
        "",
        "Anti‑patterns to avoid",
        "---------------------",
        "• Do not create files or folders.",
        "• Do not attempt to edit file content here; only rename/move/delete.",
        "• Do not parse files inside organize; use ask for read-only and then mutate.",
    ]

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
            "You are an assistant specializing in **organizing files and folders in the filesystem**.",
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the goal and choose the best approach yourself.",
            clar_sentence,
            "You should attempt to fulfill the organization goal as best you can, even if it seems out of scope.",
            "Use the ask() tool to discover targets before mutating; do not run read-only tools directly in organize.",
            "Please always mention the relevant file path(s) or folder path(s) in your response.",
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
            f"Current UTC time is {now()}.",
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
                f"• Ask about a file: `{ask_about_file_example}(file_path='/path/to/report.pdf', question='Summarize this.')`"
                if ask_about_file_example
                else "• Ask about a file: `<SomeFileManager>_ask_about_file(file_path='/path/to/report.pdf', question='Summarize this.')`"
            ),
            "",
            "─ Identity & Delegation ─",
            "• Accept absolute paths and provider URIs; do not rewrite them.",
            "• When delegating, pass the identifier exactly as provided to the target FileManager.",
            "• Prefer resolving identity with the target filesystem's `stat` (via its ask surface).",
            "",
            "─ Path‑first targeting for file-scoped operations ─",
            "• Do not pass the literal '<file_path>' string to tools; use the actual file path value.",
            "• Prefer using the exact absolute `<file_path>` from the user or stat/identity helpers.",
            "  - Content: `tables=['<file_path>']`",
            "  - Table:   `tables=['<file_path>.Tables.<label>']`",
            "• In unified mode, Content lives under the unified label (e.g., 'UnifiedDocs'); per‑file tables remain keyed by `<file_path>`.",
            "",
            "Parallelism and single‑call preference",
            "-------------------------------------",
            "• Prefer one targeted ask to the correct filesystem over querying many managers indiscriminately.",
            "• Run independent per-filesystem queries in parallel when needed; avoid confirmatory re-queries.",
        ],
    )

    if not clarification_block:
        usage_examples = "\n".join(
            [
                usage_examples,
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state assumptions.",
            ],
        )

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
            "You are an assistant specializing in **managing and querying multiple filesystems**.",
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
            clar_sentence,
            "You should attempt to answer *any* question as best you can, even if it seems out of scope.",
            "Use the tools provided to see if you can find any missing context *before* asking the user for clarifications.",
            "Absolute paths and provider URIs are accepted; when delegating to a specific filesystem manager, pass identifiers as-is (do not rewrite).",
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
            "─ Mutations (rename/move/delete) ─",
            (
                f'• Rename: `{organize_tool_example}(text="Rename /docs/notes.txt to notes-2024.txt")`'
                if organize_tool_example
                else '• Rename: `<SomeFileManager>_organize(text="Rename /docs/notes.txt to notes-2024.txt")`'
            ),
            (
                f'• Move: `{organize_tool_example}(text="Move /reports/q1.pdf to /archive/")`'
                if organize_tool_example
                else '• Move: `<SomeFileManager>_organize(text="Move /reports/q1.pdf to /archive/")`'
            ),
            (
                f'• Delete: `{organize_tool_example}(text="Delete /tmp/old.log")`'
                if organize_tool_example
                else '• Delete: `<SomeFileManager>_organize(text="Delete /tmp/old.log")`'
            ),
            "",
            "─ Identity & Delegation ─",
            "• Accept absolute paths and provider URIs; do not rewrite them.",
            "• Preflight with the target filesystem's ask surface (use stat/list) before mutating.",
            "• Keep rename/move/delete within a single filesystem (cross-filesystem transfer is not implemented here).",
            "",
            "Parallelism & Sequencing",
            "------------------------",
            "• Batch independent mutations; sequence dependent operations (move, then rename).",
            "• Avoid confirmatory re-queries after each mutation; verify end state succinctly.",
        ],
    )

    if not clarification_block:
        usage_examples = "\n".join(
            [
                usage_examples,
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state assumptions.",
            ],
        )

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
