"""
Prompt builders for KnowledgeManager.

These builders use the centralized `PromptSpec` and `compose_system_prompt`
utilities where appropriate, while preserving domain-specific content for
the complex retrieval, update, and refactoring workflows.
"""

from __future__ import annotations

import json
import textwrap
from typing import Callable, Dict, List

from .types import column_type_schema
from ..memory_manager.broader_context import get_broader_context
from ..common.prompt_helpers import (
    sig_dict,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
    PromptSpec,
    compose_system_prompt,
)

# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} using shared helper."""
    return sig_dict(tools)


def _tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """Delegate to shared tool name resolver."""
    return _shared_tool_name(tools, needle)


def _require_tools(pairs: Dict[str, str | None], tools: Dict[str, Callable]) -> None:
    """Delegate validation to shared helper for consistent errors."""
    _shared_require_tools(pairs, tools)


# ────────────────────────────────────────────────────────────────────────────
# Shared historic activity snippet
# ────────────────────────────────────────────────────────────────────────────


def _rolling_activity_section() -> str:
    """Return a markdown summary of the agent's historic activity from cache."""

    try:
        overview = get_broader_context()
    except Exception:  # pragma: no cover
        return ""

    if not overview:
        return ""

    return "\n".join(
        [
            "Historic Activity Overview",
            "---------------------------",
            "Below is a summary of the agent's historic activity (tasks, contacts, knowledge, transcripts, etc.).",
            "Some parts may be useful context for the current task while others might not – use your judgement.",
            "",
            overview,
            "",
        ],
    )


# ────────────────────────────────────────────────────────────────────────────
# public builders
# ────────────────────────────────────────────────────────────────────────────


def build_refactor_prompt(
    tools: dict[str, callable],
    *,
    table_schemas_json: str,
    include_activity: bool = True,
    case_specific_instructions: str | None = None,
) -> str:
    """
    Construct the system-prompt for :pymeth:`KnowledgeManager.refactor`.

    The prompt makes three guarantees:
    1. *All* table/column-level tools are documented in explicit JSON schema.
    2. Clear, opinionated instructions describe **why** and **how** to
       normalise the schema (remove duplication, introduce surrogate keys,
       delete unused columns, etc.).
    3. Two worked examples illustrate the expected reasoning and tool use.
    """

    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Resolve a handful of canonical tool names dynamically for examples
    delete_column_fname = _tool_name(tools, "delete_column")
    create_empty_column_fname = _tool_name(tools, "create_empty_column")
    rename_column_fname = _tool_name(tools, "rename_column")
    request_clar_fname = _tool_name(tools, "request_clarification")

    _require_tools(
        {
            "delete_column": delete_column_fname,
            "create_empty_column": create_empty_column_fname,
            "rename_column": rename_column_fname,
        },
        tools,
    )

    examples = (
        textwrap.dedent(
            """
        ### EXAMPLE 1 — simple column move
        *Before*
        ┌─ Companies(name, revenue, opening_hours)
        └─ Contacts(first_name, surname, email_address, **opening_hours**)

        *Action*
        1. `{delete_column}(table="Contacts", column_name="opening_hours")`
        2. `{create_empty_column}(table="Companies", column_name="company_id", column_type="int")`
        3. `{rename_column}(table="Companies", old_name="name", new_name="company_name")`
        4. Update rows so every contact references `company_id`.

        ### EXAMPLE 2 — splitting a mixed-type column
        • Detect that `purchase_info` mixes JSON dicts and strings.
        • Create two new columns (`purchase_details` *dict*, `purchase_note` *str*)
        • Migrate the correct rows with tool calls (`create_derived_column`, `delete_column`, etc.).
        """,
        )
        .strip()
        .format(
            delete_column=delete_column_fname,
            create_empty_column=create_empty_column_fname,
            rename_column=rename_column_fname,
        )
    )

    # Additional tool-selection guidance & examples
    create_derived_column_fname = _tool_name(tools, "create_derived_column")
    transform_column_fname = _tool_name(tools, "transform_column")
    copy_column_fname = _tool_name(tools, "copy_column")
    move_column_fname = _tool_name(tools, "move_column")
    vectorize_column_fname = _tool_name(tools, "vectorize_column")
    update_rows_fname = _tool_name(tools, "update_rows")
    delete_rows_fname = _tool_name(tools, "delete_rows")
    ingest_documents_fname = _tool_name(tools, "ingest_documents")
    create_table_fname = _tool_name(tools, "create_table")
    rename_table_fname = _tool_name(tools, "rename_table")
    delete_tables_fname = _tool_name(tools, "delete_tables")

    usage_guidance = textwrap.dedent(
        f"""
Tool selection (read carefully)
--------------------------------
• Prefer **`{rename_column_fname}`** over delete+create when re-labelling an existing field.
• Use **`{create_empty_column_fname}`** for genuinely new attributes that don't yet exist.
• Use **`{create_derived_column_fname}`** when a column can be computed deterministically from others.
• Use **`{transform_column_fname}`** for in-place transformations (implemented as derive → swap).
• Use **`{copy_column_fname}`** to project a column into another table; **`{move_column_fname}`** to copy then remove from the source table.
• Use **`{vectorize_column_fname}`** only when you need a semantic vector for downstream search—don't vectorise everything.

Examples
--------
• Split a mixed-type column
  1. Detect that `purchase_info` mixes dicts/strings.
  2. Create `purchase_details: dict` and `purchase_note: str` with `{create_empty_column_fname}`.
  3. Move/transform values with `{transform_column_fname}` or a pair of derive+rename steps.
  4. Remove the legacy column with `{delete_column_fname}` once migration is complete.

• Normalise a denormalised reference
  1. Create a surrogate key on the parent table with `{create_empty_column_fname}` (e.g., `company_id: int`).
  2. `{rename_column_fname}` verbose labels to clearer names.
  3. `{copy_column_fname}` or `{move_column_fname}` columns into the correct tables.
  4. Update rows to reference the surrogate key instead of duplicating text blobs.

Tables (examples)
-----------------
• `{create_table_fname}(name="Benchmarks", columns={{"name":"str","domain":"str"}}, unique_key_name="benchmark_id")`
• `{rename_table_fname}(old_name="TmpResults", new_name="Results_v2")`
• `{delete_tables_fname}(tables=["Scratch"], startswith="tmp_")`

Rows (examples)
---------------
• `{update_rows_fname}(table="Models", updates={{3: {{"efficiency_km_per_kwh": 6.1}}}})`
• `{delete_rows_fname}(tables=["Sales"], filter="year < 2021", limit=1000)`

Files (example)
---------------
• `{ingest_documents_fname}(filenames=["catalog.pdf"], table="content", replace_existing=True)`

Anti-patterns to avoid
----------------------
• Avoid delete+create when a simple rename will do.
• Avoid duplicated denormalised strings across tables—introduce a key and normalise.
• Avoid mixed-type columns—split into well-typed columns.
• Don't vectorise columns unless semantic search requires it.
        """,
    ).strip()

    # Build clarification block if tool available
    clarification_block = (
        textwrap.dedent(
            f"""
Clarification
-------------
• Ask for clarification when the user's request is underspecified
  `{request_clar_fname}(question="Which specific items should be refactored?")`
            """,
        ).strip()
        if request_clar_fname
        else ""
    )

    # Build usage examples with clarification appended
    full_usage = f"{usage_guidance}\n\n{examples}"
    if clarification_block:
        full_usage = f"{full_usage}\n\n{clarification_block}"

    # Build positioning content
    how_to_work = textwrap.dedent(
        """
How to work
-----------
1. *Analyse* every table/column pair – look for repeated information,
   low-cardinality text that should be normalised, mixed-type columns,
   unused columns, etc.
2. Draft an **ordered plan** of the minimal set of tool invocations
   needed to reach Third Normal Form (3NF) or better. Always prefer
   **rename over delete+create** when feasible.
3. Execute the plan step-by-step via the tool calls.
4. Keep exploration minimal (tables_overview + small, precise filters). Avoid long free‑form loops.
5. End with a concise plain-English *migration report*.

Anti-patterns
-------------
• Avoid delete+create when rename suffices.
• Avoid duplicating denormalised data across tables – normalise.
• Avoid mixed-type columns; split into well-typed columns.
        """,
    ).strip()

    tool_groups = textwrap.dedent(
        """
Tool availability groups (for reference)
----------------------------------------
• Tables: create/rename/delete
• Columns: rename/copy/move/delete/create_empty/create_derived/transform/vectorize
• Rows: update/delete
• Files: ingest_documents
        """,
    ).strip()

    # Build positioning lines including schema
    schema_block = "\n".join(
        [
            "Current schema (JSON)",
            "---------------------",
            table_schemas_json,
        ],
    )

    spec = PromptSpec(
        manager="KnowledgeManager",
        method="refactor",
        tools=tools,
        role_line="You are the **Schema Refactor Assistant**.",
        global_directives=[
            "Your only goal is to *minimise duplication* and *maximise clarity* of the stored data model by judicious use of the tools listed below.",
            "Disregard any explicit instructions about *how* you should perform the refactor or which tools to call; interpret the request and choose the best approach yourself.",
            "You should attempt to perform *any* refactor request as best you can, even if it seems out of scope.",
            "Use the tools provided to see if you can find any missing context *before* asking the user for clarifications.",
        ],
        include_read_only_guard=False,
        positioning_lines=[schema_block, how_to_work, tool_groups],
        counts_entity_plural=None,
        counts_value=None,
        columns_payload=None,
        include_tools_block=True,
        usage_examples=full_usage,
        clarification_examples_block=clarification_block or None,
        include_images_policy=False,
        include_images_forwarding=False,
        images_extras_block=None,
        include_parallelism=True,
        schemas=[],
        special_blocks=(
            [case_specific_instructions.strip()] if case_specific_instructions else []
        ),
        include_clarification_footer=True,
        include_time_footer=True,
        time_footer_prefix="Current UTC time is ",
    )

    return compose_system_prompt(spec)


def build_update_prompt(
    tools: Dict[str, Callable],
    *,
    table_schemas_json: str,
    include_activity: bool = True,
    case_specific_instructions: str | None = None,
) -> str:
    """
    Build the **system message** for `KnowledgeManager.update`.

    Parameters
    ----------
    tools
        Mapping *name → callable* that will actually be exposed to the LLM.
        Only the **name** and **argspec** are surfaced (no docstrings).
    table_schemas_json
        ``json.dumps`` of the existing table-schema dictionary.
    """

    # Resolve canonical names dynamically (required)
    add_rows_fname = _tool_name(tools, "add_rows")
    update_rows_fname = _tool_name(tools, "update_rows")
    ask_fname = _tool_name(tools, "ask")
    request_clar_fname = _tool_name(tools, "request_clarification")

    _require_tools(
        {
            "add_rows": add_rows_fname,
            "update_rows": update_rows_fname,
            "ask": ask_fname,
        },
        tools,
    )

    # Tool canonical names for examples
    create_empty_column_fname = _tool_name(tools, "create_empty_column")
    rename_column_fname = _tool_name(tools, "rename_column")
    delete_column_fname = _tool_name(tools, "delete_column")
    delete_rows_fname = _tool_name(tools, "delete_rows")
    ingest_documents_fname = _tool_name(tools, "ingest_documents")
    create_table_fname = _tool_name(tools, "create_table")
    rename_table_fname = _tool_name(tools, "rename_table")
    delete_tables_fname = _tool_name(tools, "delete_tables")

    # Build clarification block if tool available
    clarification_block = (
        textwrap.dedent(
            f"""
Clarification
-------------
• If any request is ambiguous, ask the user to disambiguate before changing data
  `{request_clar_fname}(question="There are several possible matches. Which item did you mean?")`
            """,
        ).strip()
        if request_clar_fname
        else ""
    )

    usage_examples_base = f"""
Tool selection
--------------
• Prefer `{update_rows_fname}` when you know the exact row identifier(s) you want to mutate.
• When the user describes targets semantically (e.g., "all 2023 APAC sales for Model X"), first ask a freeform question with `{ask_fname}` to identify the correct row identifier(s), then call `{update_rows_fname}`.

Ask vs Clarification
--------------------
• `{ask_fname}` is ONLY for inspecting/locating records that already exist (e.g., to find row ids, verify fields).
• Do NOT use `{ask_fname}` to ask the human about NEW content to be created/changed in this update.
• For human clarifications about prospective/new knowledge (e.g., name spelling, missing numbers, preferred channel), call `{request_clar_fname}` when available.
• If the schema lacks a field the user wants to set, create it with `{create_empty_column_fname}` (typically `column_type='str'`) before updating.
• Use `{delete_column_fname}`, `{delete_rows_fname}`, `{delete_tables_fname}` only on explicit deletion requests

Schema evolution
----------------
• If the user wants to store a new attribute that does not map to existing columns, create it first:
  `{create_empty_column_fname}(table="Models", column_name="warranty_years", column_type="int")`
  Then update:
  `{update_rows_fname}(table="Models", updates={{42: {{"warranty_years": 5}}}})`
• Prefer **rename** over delete+create when re-labelling a column:
  `{rename_column_fname}(table="Models", old_name="capacity_kwh", new_name="battery_kwh")`

Tables
------
• Create: `{create_table_fname}(name="Markets", columns={{"name":"str","region":"str"}}, unique_key_name="market_id")`
• Rename: `{rename_table_fname}(old_name="TempModels", new_name="Models_v2")`
• Delete: `{delete_tables_fname}(tables=["Tmp1"], startswith="scratch_")`

Rows
----
• Create/Insert: `{add_rows_fname}(table="Sales", rows=[{{"model_id":1,"market_id":3,"year":2024,"units":12000}}])`
• Update/Modify: `{update_rows_fname}(table="Models", updates={{7: {{"warranty_years": 4}}}})`
• Delete: `{delete_rows_fname}(tables=["Sales"], filter="year < 2020", limit=500)`

Note: Whenever updating or adding rows, make sure the values are consistent with the column schema e.g. if the column
schema says the column is of type float and the value is 180, then the value should be 180.0.

Files
-----
• Ingest a batch of documents (with replacement):
  `{ingest_documents_fname}(filenames=["specs.pdf","pricing.docx"], table="content", replace_existing=True)`

Realistic find-then-update flows
--------------------------------
• Set a review_status for the document about renewable energy stored in the Berlin office
  1 Ask a freeform question (no instructions about how to answer):
    `{ask_fname}(text="Which document discusses renewable energy and is stored in the Berlin office?")`
  2 Update the returned id:
    `{update_rows_fname}(table="<table_name>", updates={{<id>: {{"review_status": "Requires monthly updates"}}}})`

• Mark priority=True for the research paper that covers machine learning and was recently cited in a conference
  1 Ask a freeform question (no instructions about how to answer):
    `{ask_fname}(text="Which research paper on machine learning was cited at a conference last week?")`
  2 Update the returned id:
    `{update_rows_fname}(table="<table_name>", updates={{<id>: {{"priority": True}}}})`

• Query may span multiple freeform fields (derived expression)
  1 Build a composite expression across `content`, `summary`, and a custom field like `topic`:
    `expr = "str({{content}}) + ' ' + str({{summary}}) + ' ' + str({{topic}})"`
  2 Ask a freeform question referring to the same clues:
    `{ask_fname}(text="What document discusses London-based renewable energy projects from 2023?")`
  3 Update the found record as requested using:
    `{update_rows_fname}(table="<table_name>", updates={{<id>: {{"field_name": "new_value"}}}})`

• Set access_level for the technical specification about quantum computing hardware
  1 Ask a freeform question (no instructions about how to answer):
    `{ask_fname}(text="Which technical specification covers quantum computing hardware components?")`
  2 Update the returned id:
    `{update_rows_fname}(table="<table_name>", updates={{<id>: {{"access_level": "restricted"}}}})`

• Mark needs_translation=True for the research document written in German about automotive engineering
  1 Ask a freeform question (no instructions about how to answer):
    `{ask_fname}(text="Which German-language document discusses automotive engineering innovations?")`
  2 Update the returned id:
    `{update_rows_fname}(table="<table_name>", updates={{<id>: {{"needs_translation": True}}}})`

• Populate missing battery_kwh for all models that mention "long-range" in description
  1 Find candidates:
     `{ask_fname}(text="List model ids where description semantically matches 'long-range' but battery_kwh is None")`
  2 Write updates in batch:
     `{update_rows_fname}(table="Models", updates={{1: {{"battery_kwh": 82}}, 7: {{"battery_kwh": 74}}}})`

Asking Questions
----------------
• If you're unsure about your write coverage, just `{ask_fname}` to verify progress:
  `{ask_fname}(text="Which Models rows still have warranty_years == None after my update?")`

Anti-patterns to avoid
----------------------
• Repeating the exact same tool call with the same arguments as a means to 'make sure it has completed', just call `{ask_fname}` to check the latest state of the knowledge base/tables
• Making assumptions about current table state, instead you should make liberal use of the `{ask_fname}` tool
• Joining a table to itself to locate targets when a direct filter on that table would suffice.

(When locating a record by semantics, always do a quick `{ask_fname}` step to resolve `row_id` before mutating. Prefer updating in place over recreating.)
    """
    usage_examples = textwrap.dedent(usage_examples_base).strip()
    if clarification_block:
        usage_examples = f"{usage_examples}\n\n{clarification_block}"

    # Build workflow instructions
    workflow = textwrap.dedent(
        f"""
Follow this workflow strictly:
1. Extract every fact (subject → attribute → value) from the message.
2. Use the `{ask_fname}` tool to search for any existing records that need to be updated.
3. Use this info to decide whether each fact updates an *existing* row / column / table or inserts a *new* row / column / table.
4. Add missing tables if necessary (should be quite rare)
5. Add missing columns with the correct data-type if necessary (should be quite rare)
5. Use `{add_rows_fname}` to insert and `{update_rows_fname}` to modify existing rows.
6. Use the `{ask_fname}` method again to verify everything was stored or updated correctly.
7. Reply with a short natural-language confirmation of what was stored or updated.

Do **not** hallucinate data.

Before adding new knowledge or making edits, briefly check whether similar records already exist (via `{ask_fname}`) to avoid duplicates.
When the user describes the update semantically, resolve the row identifier first by requesting the row identifier from the ask method, then perform the update via the row identifier.
Use the `{ask_fname}` method to see if you can find any missing context *before* you consider asking the user for clarifications.
            """,
    ).strip()

    # Build schema blocks
    column_schema_block = "\n".join(
        [
            "ColumnType Schema",
            "-----------------",
            json.dumps(column_type_schema, indent=4),
        ],
    )

    table_schema_block = "\n".join(
        [
            "Current table schemas",
            "---------------------",
            table_schemas_json,
        ],
    )

    spec = PromptSpec(
        manager="KnowledgeManager",
        method="update",
        tools=tools,
        role_line="Your task is to **store** new knowledge or **update** existing knowledge provided by the user.",
        global_directives=[
            "Keep the schema clean and future-proof – feel free to create tables / columns before inserting data.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the request and choose the best approach yourself.",
            "You should attempt to perform *any* storage request as best you can, even if it seems out of scope.",
            f"Important: `{ask_fname}` is read‑only and must only be used to locate/inspect knowledge that already exist.",
        ],
        include_read_only_guard=False,
        positioning_lines=[workflow],
        counts_entity_plural=None,
        counts_value=None,
        columns_payload=None,
        include_tools_block=True,
        usage_examples=usage_examples,
        clarification_examples_block=clarification_block or None,
        include_images_policy=False,
        include_images_forwarding=False,
        images_extras_block=None,
        include_parallelism=True,
        schemas=[],
        special_blocks=[column_schema_block, table_schema_block]
        + ([case_specific_instructions.strip()] if case_specific_instructions else []),
        include_clarification_footer=True,
        include_time_footer=True,
        time_footer_prefix="Current UTC time is ",
    )

    return compose_system_prompt(spec)


def build_ask_prompt(
    tools: Dict[str, Callable],
    *,
    table_schemas_json: str,
    include_activity: bool = True,
    case_specific_instructions: str | None = None,
    include_join_info: bool | None = None,
) -> str:
    """
    Build the **system message** for `KnowledgeManager.retrieve`.
    """

    # Resolve canonical tool names dynamically
    tables_overview_fname = _tool_name(tools, "tables_overview")
    filter_fname = _tool_name(tools, "filter")
    search_fname = _tool_name(
        tools,
        "search",
    )  # picks the basic semantic search, not joins
    reduce_fname = _tool_name(tools, "reduce")
    request_clar_fname = _tool_name(tools, "request_clarification")

    _require_tools(
        {
            "tables_overview": tables_overview_fname,
            "filter": filter_fname,
            "search": search_fname,
        },
        tools,
    )

    # Determine whether to include join-related guidance/examples.
    if include_join_info is None:
        join_tools_present = any(
            _tool_name(tools, name)
            for name in (
                "search_join",
                "filter_join",
                "search_multi_join",
                "filter_multi_join",
            )
        )
        include_join_info = bool(join_tools_present)
        if not include_join_info:
            try:
                schema_obj = (
                    json.loads(table_schemas_json) if table_schemas_json else {}
                )
                if isinstance(schema_obj, dict):
                    include_join_info = len(schema_obj.keys()) > 1
            except Exception:
                include_join_info = False

    join_hint = (
        """
   **Avoid joins on the same table** (including self-joins). When all required fields live in a single table,
   prefer using `{filter}` directly. Reserve join operations for combining **different** tables where a join
   is actually necessary.
        """.format(
            filter=filter_fname,
        )
        if include_join_info
        else ""
    )

    # Build mandatory steps
    mandatory_steps = textwrap.dedent(
        f"""
Mandatory steps:
1. List each distinct piece of information the question asks for.
2. Identify which tables / columns can hold that info.
3. Fetch *all* relevant rows (use `{search_fname}` for semantic search; use `{filter_fname}` for precise filters).{join_hint}
4. Aggregate results into a concise answer covering every fact.
5. Double-check nothing is missing; if so, repeat the search.

Do **not** hallucinate data.
        """,
    ).strip()

    # Build usage examples dynamically depending on whether we expose join tools
    selection_lines = [
        "─ Tool selection (read carefully) ─",
        f"• For ANY semantic question over free-form text, ALWAYS use `{search_fname}`. Never try to approximate meaning with a lot of brittle substring filters using the `{filter_fname}` tool.",
        f"• Use `{filter_fname}` only for exact/boolean logic over structured fields (ids, enums, ranges, null checks) or for narrow, constrained text.",
        "• Keep semantic search tight: prefer k ≤ 10 unless there is a clear reason to widen.",
    ]
    if include_join_info:
        selection_lines.append(
            f"• Reserve joins for combining **different** tables; if all fields live in a single table, prefer direct `{filter_fname}`. Avoid self-joins.",
        )
        selection_lines.append(
            "• **MANDATORY**: For questions requiring data from multiple tables, you MUST use the dedicated join tools (`filter_join`, `search_join`, `filter_multi_join`, `search_multi_join`). Do NOT simulate joins by filtering each table separately and combining results manually.",
        )

    parts_examples: List[str] = [
        "Examples",
        "--------",
        "",
        *selection_lines,
        "",
        "─ Semantic search (ranked by SUM of cosine distances across terms) ─",
        "• Single-table search with focused references:",
        f'  `{search_fname}(table="Products", references={{"description": "stainless steel water bottle"}}, k=5)`',
        "",
    ]

    if include_join_info:
        parts_examples.extend(
            [
                "─ Join retrieval (two tables) ─",
                "• Semantic search over a join:",
                '  `search_join(tables=["Orders","Customers"],',
                '              join_expr="Orders.customer_id == Customers.customer_id",',
                '              select={"Orders.notes":"notes","Customers.industry":"industry"},',
                '              references={"notes":"rush order","industry":"pharma"}, k=5)`',
                "• Exact filter over a join (result_where can only reference selected aliases):",
                '  `filter_join(tables=["Orders","Customers"],',
                '              join_expr="Orders.customer_id == Customers.customer_id",',
                '              select={"Orders.customer_id":"cid","Customers.segment":"segment"},',
                "              result_where=\"segment == 'enterprise' and cid > 1000\", result_limit=50)`",
                "  (Alias any source columns you need in `select` first; `result_where` may only use these aliases.)",
                "",
                "─ Multi-join retrieval (chained joins) ─",
                "• Chain joins with `$prev` to reference the previous step:",
                "  `search_multi_join(joins=[",
                '    {"tables":["Sales","Models"], "join_expr":"Sales.model_id == Models.model_id",',
                '      "select":{"Sales.units":"units","Models.name":"model"}},',
                '    {"tables":["$prev","Companies"], "join_expr":"_.company_id == Companies.company_id",',
                '      "select":{"$prev.units":"units","$prev.model":"model","Companies.name":"company"}}',
                '  ], references={"company":"Northwind","model":"Voyager"}, k=5)`',
                "• When filtering the final result of a multi-join, ensure the last step's `select` includes any columns you will reference in `result_where`.",
                "",
            ],
        )

    parts_examples.extend(
        [
            "─ Filtering (exact/boolean; not semantic) ─",
            f'• Equality: `{filter_fname}(tables="Products", filter="sku == \'ABC-123\'")`',
            f'• Range:    `{filter_fname}(tables="Sales", filter="year >= 2023 and units > 1000")`',
            "",
            "─ Numeric aggregations ─",
            f"• For numeric reduction metrics (count, sum, mean, min, max, median, mode, var, std) over numeric columns, use `{reduce_fname}` instead of filtering and computing in-memory.",
            f"  `{reduce_fname}(table=\"Sales\", metric='sum', keys='units', group_by='region')`",
            "",
            "Anti-patterns to avoid",
            "---------------------",
            "• Avoid concatenating a whole row into one giant reference string; pass multiple focused references keyed by their exact columns.",
            f"• Avoid substring filtering on large text columns; prefer `{search_fname}` for meaning. Never try to approximate meaning with a lot of brittle substring filters using the `{filter_fname}` tool.",
            f"• Do not re-query to reconfirm facts immediately after a conclusive search; only add `{filter_fname}` if you need **new** structured constraints.",
        ],
    )
    if include_join_info:
        parts_examples.append(
            "• Avoid joins on the same table; filter directly when possible.",
        )

    usage_examples = textwrap.dedent("\n".join(parts_examples)).strip()
    if request_clar_fname:
        clarification_usage = textwrap.dedent(
            f"""
─ Clarification ─
• If needed, ask the user to disambiguate:
  `{request_clar_fname}(question="There are several possible matches. Which one did you mean?")`
            """,
        ).strip()
        usage_examples = f"{usage_examples}\n\n{clarification_usage}"

    # Build schema blocks
    column_schema_block = "\n".join(
        [
            "ColumnType Schema",
            "-----------------",
            json.dumps(column_type_schema, indent=4),
        ],
    )

    table_schema_block = "\n".join(
        [
            "Current table schemas",
            "---------------------",
            table_schemas_json,
        ],
    )

    spec = PromptSpec(
        manager="KnowledgeManager",
        method="ask",
        tools=tools,
        role_line="Your task is to **retrieve** information requested by the user.",
        global_directives=[
            "Use the provided tools to search the schemas and tables so that every requested fact can be answered precisely.",
            "Work strictly through the tools provided.",
            "You should not attempt to refactor the schema, this is not your responsibility.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
            "You should attempt to perform *any* retrieval request as best you can, even if it seems out of scope.",
            "Use the tools provided to see if you can find any missing context *before* asking the user for clarifications.",
        ],
        include_read_only_guard=True,
        positioning_lines=[mandatory_steps],
        counts_entity_plural=None,
        counts_value=None,
        columns_payload=None,
        include_tools_block=True,
        usage_examples=usage_examples,
        clarification_examples_block=None,
        include_images_policy=False,
        include_images_forwarding=False,
        images_extras_block=None,
        include_parallelism=True,
        schemas=[],
        special_blocks=[column_schema_block, table_schema_block]
        + ([case_specific_instructions.strip()] if case_specific_instructions else []),
        include_clarification_footer=True,
        include_time_footer=True,
        time_footer_prefix="Current UTC time is ",
    )

    return compose_system_prompt(spec)


# ────────────────────────────────────────────────────────────────────────────
# Simulated helper
# ────────────────────────────────────────────────────────────────────────────


def build_simulated_method_prompt(
    method: str,
    user_request: str,
    parent_chat_context: list[dict] | None = None,
) -> str:
    """Return instruction prompt for *simulated* KnowledgeManager methods."""
    import json

    preamble = f"On this turn you are simulating the '{method}' method."
    if method.lower() in {"ask", "retrieve"}:
        behaviour = (
            "Please always return imaginary information answering the question. "
            "Do not ask for clarifications or describe how you will obtain the information."
        )
    elif method.lower() in {"update", "store"}:
        behaviour = (
            "Please always act as though the knowledge has been **stored or updated** successfully. "
            "Respond in past tense summarising what was stored."
        )
    elif method.lower() == "refactor":
        behaviour = (
            "Provide a short migration plan that would bring the schema to 3NF. "
            "Do not execute any tool calls – simply describe the completed refactor."
        )
    else:
        behaviour = "Respond as though the requested operation has already been fully completed."

    parts: list[str] = [preamble, behaviour, "", f"The user input is:\n{user_request}"]
    if parent_chat_context:
        parts.append(
            f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}",
        )
    return "\n".join(parts)
