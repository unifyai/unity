from __future__ import annotations

import json
import textwrap
from typing import Callable, Dict

from .types import column_type_schema
from ..memory_manager.broader_context import get_broader_context
from ..common.prompt_helpers import (
    clarification_guidance,
    sig_dict,
    now_utc_str,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
)

# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} using shared helper."""
    return sig_dict(tools)


def _now() -> str:  # UTC timestamp helper
    return now_utc_str()


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

    core_instructions = textwrap.dedent(
        f"""
        You are the **Schema Refactor Assistant**.
        Your only goal is to *minimise duplication* and *maximise clarity* of
        the stored data model by judicious use of the tools listed below.
        Disregard any explicit instructions about *how* you should perform the refactor or which tools to call; determine the best method and steps yourself.
        You should attempt to perform *any* refactor request as best you can, even if it seems out of scope.
        use the tools provided to see if you can find any missing context *before* asking the user for clarifications.

        --------------------------------------------------------------------
        ## Current schema (JSON)
        {table_schemas_json}

        --------------------------------------------------------------------
        ## Tools (name → argspec)
        {sig_json}

        Work strictly through the tools provided.
        Disregard any explicit instructions about *which* tools to call; determine the best approach yourself.

        Tool availability groups (for reference)
        ----------------------------------------
        • Tables: create/rename/delete
        • Columns: rename/copy/move/delete/create_empty/create_derived/transform/vectorize
        • Rows: update/delete
        • Files: ingest_documents

        --------------------------------------------------------------------
        ## How to work
        1. *Analyse* every table/column pair – look for repeated information,
           low-cardinality text that should be normalised, mixed-type columns,
           unused columns, etc.
        2. Draft an **ordered plan** of the minimal set of tool invocations
           needed to reach Third Normal Form (3NF) or better. Always prefer
           **rename over delete+create** when feasible.
        3. Execute the plan step-by-step via the tool calls.
        4. End with a concise plain-English *migration report*.

        Anti-patterns
        -------------
        • Avoid delete+create when rename suffices.
        • Avoid duplicating denormalised data across tables – normalise.
        • Avoid mixed-type columns; split into well-typed columns.

        --------------------------------------------------------------------
        ## Usage examples
        {examples}
        """,
    ).strip()

    # If provided, REPLACE the generic instructions with case-specific instructions
    instructions_block = core_instructions
    if case_specific_instructions:
        instructions_block += "\n\n" + case_specific_instructions.strip()

    # Additional tool-selection guidance & examples (mirrors CM/TM tone; refactor-specific)
    # Resolve more tool names dynamically if present
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

    def _nn(x: str | None) -> str:
        return x if x is not None else "<unavailable>"

    usage_guidance = textwrap.dedent(
        f"""
        Tool selection (read carefully)
        --------------------------------
        • Prefer **`{_nn(rename_column_fname)}`** over delete+create when re-labelling an existing field.
        • Use **`{_nn(create_empty_column_fname)}`** for genuinely new attributes that don’t yet exist.
        • Use **`{_nn(create_derived_column_fname)}`** when a column can be computed deterministically from others.
        • Use **`{_nn(transform_column_fname)}`** for in-place transformations (implemented as derive → swap).
        • Use **`{_nn(copy_column_fname)}`** to project a column into another table; **`{_nn(move_column_fname)}`** to copy then remove from the source table.
        • Use **`{_nn(vectorize_column_fname)}`** only when you need a semantic vector for downstream search—don’t vectorise everything.

        Examples
        --------
        • Split a mixed-type column
          1. Detect that `purchase_info` mixes dicts/strings.
          2. Create `purchase_details: dict` and `purchase_note: str` with `{_nn(create_empty_column_fname)}`.
          3. Move/transform values with `{_nn(transform_column_fname)}` or a pair of derive+rename steps.
          4. Remove the legacy column with `{_nn(delete_column_fname)}` once migration is complete.

        • Normalise a denormalised reference
          1. Create a surrogate key on the parent table with `{_nn(create_empty_column_fname)}` (e.g., `company_id: int`).
          2. `{_nn(rename_column_fname)}` verbose labels to clearer names.
          3. `{_nn(copy_column_fname)}` or `{_nn(move_column_fname)}` columns into the correct tables.
          4. Update rows to reference the surrogate key instead of duplicating text blobs.

        Tables (examples)
        -----------------
        • `{_nn(create_table_fname)}(name="Benchmarks", columns={{"name":"str","domain":"str"}}, unique_key_name="benchmark_id")`
        • `{_nn(rename_table_fname)}(old_name="TmpResults", new_name="Results_v2")`
        • `{_nn(delete_tables_fname)}(tables=["Scratch"], startswith="tmp_")`

        Rows (examples)
        ---------------
        • `{_nn(update_rows_fname)}(table="Models", updates={{3: {{"efficiency_km_per_kwh": 6.1}}}})`
        • `{_nn(delete_rows_fname)}(tables=["Sales"], filter="year < 2021", limit=1000)`

        Files (example)
        ---------------
        • `{_nn(ingest_documents_fname)}(filenames=["catalog.pdf"], table="content", replace_existing=True)`

        Anti-patterns to avoid
        ----------------------
        • Avoid delete+create when a simple rename will do.
        • Avoid duplicated denormalised strings across tables—introduce a key and normalise.
        • Avoid mixed-type columns—split into well-typed columns.
        • Don't vectorise columns unless semantic search requires it.
        """,
    ).strip()

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    # Conditional guidance about asking questions in final responses
    request_clar_fname = _tool_name(tools, "request_clarification")
    clar_sentence = (
        f"Do not ask the user questions in your final response; only use the `{request_clar_fname}` tool to ask clarifying questions."
        if request_clar_fname
        else (
            "Do not ask the user questions in your final response. Instead, proceed using sensible defaults/best‑guess values and explicitly tell inner tools that these are assumptions/best guesses, not confirmed answers."
        )
    )

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

    return "\n".join(
        [
            activity_block,
            instructions_block,
            "",
            usage_guidance,
            "",
            f"Current UTC time: {_now()}.",
            clar_sentence,
            clar_section,
            clarification_block,
            "",
        ],
    )


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

    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Resolve canonical names dynamically (required)
    add_rows_fname = _tool_name(tools, "add_rows")
    update_rows_fname = _tool_name(tools, "update_rows")

    # Optional clarification helper
    request_clar_fname = _tool_name(tools, "request_clarification")

    _require_tools(
        {
            "add_rows": add_rows_fname,
            "update_rows": update_rows_fname,
        },
        tools,
    )

    # Tool canonical names for examples (mirrors ContactManager structure; adapted to Knowledge)
    ask_fname = _tool_name(tools, "ask")
    create_empty_column_fname = _tool_name(tools, "create_empty_column")
    rename_column_fname = _tool_name(tools, "rename_column")
    delete_column_fname = _tool_name(tools, "delete_column")
    delete_rows_fname = _tool_name(tools, "delete_rows")
    ingest_documents_fname = _tool_name(tools, "ingest_documents")
    create_table_fname = _tool_name(tools, "create_table")
    rename_table_fname = _tool_name(tools, "rename_table")
    delete_tables_fname = _tool_name(tools, "delete_tables")

    def _nn(x: str | None) -> str:
        return x if x is not None else "<unavailable>"

    core_instructions = (
        textwrap.dedent(
            """
        Your task is to **store** new knowledge or **update** existing knowledge provided by the user.
        Keep the schema clean and future-proof – feel free to create,
        rename or delete tables / columns before inserting data.
        Disregard any explicit instructions about *how* you should store or update the knowledge or which tools to call; determine the best method yourself.
        You should attempt to perform *any* storage request as best you can, even if it seems out of scope.
        use the tools provided to see if you can find any missing context *before* asking the user for clarifications.

        If the user refers to creating or updating *tasks*, then you should **not** store any tasks.
        Tasks should exclusively be stored by a separate task manager, this is **not your responsibility**.
        Please explain this to the user in your response, if this is part of the their request.

        Follow this workflow strictly:
        1. Extract every fact (subject → attribute → value) from the message.
        2. Search the tables to see if there are any logs already associated with the data, which should be updated.
        3. Use this info to decide whether each fact updates an *existing* row / column or inserts a *new* row / column.
        4. Add missing columns with the correct data-type if necessary (should be quite rare)
        5. Use `{add_rows}` to insert and `{update_rows}` to modify existing rows.
        6. Search again to verify everything was stored or updated correctly.
        7. Reply with a short natural-language confirmation of what was stored.

        Do **not** hallucinate data.
        """,
        )
        .strip()
        .format(add_rows=add_rows_fname, update_rows=update_rows_fname)
    )

    # If provided, REPLACE the generic instructions with case-specific instructions
    instructions_block = core_instructions
    if case_specific_instructions:
        instructions_block += "\n\n" + case_specific_instructions.strip()

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

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    # Conditional guidance about asking questions in final responses (local to update prompt)
    clar_sentence_upd = (
        f"Do not ask the user questions in your final response, please only use the `{request_clar_fname}` tool to ask clarifying questions."
        if request_clar_fname
        else (
            "Do not ask the user questions in your final response. Instead, proceed using sensible defaults/best‑guess values and explicitly tell inner tools that these are assumptions/best guesses, not confirmed answers."
        )
    )

    usage_examples_base = f"""
Tool selection
--------------
• Prefer `{_nn(update_rows_fname)}` when you know the exact row identifier(s) you want to mutate.
• When the user describes targets semantically (e.g., "all 2023 APAC sales for Model X"), first call `{_nn(ask_fname)}` to identify the rows, then call `{_nn(update_rows_fname)}`.

Ask vs Clarification
--------------------
• `{_nn(ask_fname)}` is ONLY for inspecting/locating records that already exist (e.g., to find row ids, confirm columns).
• Do NOT use `{_nn(ask_fname)}` to ask the human about NEW content to be created/changed in this update; use `{_nn(request_clar_fname)}` if available for human clarifications.

Schema evolution
----------------
• If the user wants to store a new attribute that does not map to existing columns, create it first:
  `{_nn(create_empty_column_fname)}(table="Models", column_name="warranty_years", column_type="int")`
  Then update:
  `{_nn(update_rows_fname)}(table="Models", updates={{42: {{"warranty_years": 5}}}})`
• Prefer **rename** over delete+create when re-labelling a column:
  `{_nn(rename_column_fname)}(table="Models", old_name="capacity_kwh", new_name="battery_kwh")`

Tables
------
• Create: `{_nn(create_table_fname)}(name="Markets", columns={{"name":"str","region":"str"}}, unique_key_name="market_id")`
• Rename: `{_nn(rename_table_fname)}(old_name="TempModels", new_name="Models_v2")`
• Delete: `{_nn(delete_tables_fname)}(tables=["Tmp1"], startswith="scratch_")`

Rows
----
• Insert: `{_nn(add_rows_fname)}(table="Sales", rows=[{{"model_id":1,"market_id":3,"year":2024,"units":12000}}])`
• Update: `{_nn(update_rows_fname)}(table="Models", updates={{7: {{"warranty_years": 4}}}})`
• Delete: `{_nn(delete_rows_fname)}(tables=["Sales"], filter="year < 2020", limit=500)`

Files
-----
• Ingest a batch of documents (with replacement):
  `{_nn(ingest_documents_fname)}(filenames=["specs.pdf","pricing.docx"], table="content", replace_existing=True)`

Realistic find-then-update flows
--------------------------------
• Update a specific model’s warranty
  1 Locate the target row:
     `{_nn(ask_fname)}(text="Find the model 'Voyager S' in the Models table and give me its id")`
  2 Apply the mutation:
     `{_nn(update_rows_fname)}(table="Models", updates={{<id>: {{"warranty_years": 4}}}})`

• Populate missing battery_kwh for all models that mention "long-range" in description
  1 Find candidates:
     `{_nn(ask_fname)}(text="List model ids where description semantically matches 'long-range' but battery_kwh is None")`
  2 Write updates in batch:
     `{_nn(update_rows_fname)}(table="Models", updates={{1: {{"battery_kwh": 82}}, 7: {{"battery_kwh": 74}}}})`

Asking Questions
----------------
• If you’re unsure about your write coverage, just `ask` to verify progress:
  `{_nn(ask_fname)}(text="Which Models rows still have warranty_years == None after my update?")`

Anti-patterns to avoid
----------------------
• Repeating the same update call with identical arguments to 'make sure'—instead, call `{_nn(ask_fname)}` to verify.
• Making assumptions about current table state—always verify with `{_nn(ask_fname)}` and then update accordingly.
• Joining a table to itself to locate targets when a direct filter on that table would suffice.
"""
    usage_examples = textwrap.dedent(usage_examples_base).strip()
    if request_clar_fname:
        clarification_usage = textwrap.dedent(
            f"""
            Clarification
            -------------
            • If a write target is ambiguous, ask the human before mutating
              `{request_clar_fname}(question="There are several possible matches. Which item did you mean?")`
            """,
        ).strip()
        usage_examples = f"{usage_examples}\n\n{clarification_usage}"
    else:
        usage_examples = "\n".join(
            [
                usage_examples,
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best-guess values and explicitly state to inner tools that these are assumptions/best guesses, not confirmed answers.",
                "• If an inner tool requests clarification, explicitly say no clarification channel exists and pass down concrete sensible defaults/best-guess values, clearly marked as assumptions.",
            ],
        )

    parts: list[str] = [
        activity_block,
        instructions_block,
        clar_sentence_upd,
        clar_section,
        "",
        usage_examples,
        "",
        "Tools (name → argspec):",
        sig_json,
        "",
        parallelism_block,
        "",
        "ColumnType Schema",
        "-----------------",
        json.dumps(column_type_schema, indent=4),
        "",
        "Current table schemas",
        "---------------------",
        table_schemas_json,
        "",
        f"Current UTC time: {_now()}.",
    ]

    if clarification_block:
        parts.extend(["", clarification_block])
    else:
        parts.extend(
            [
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state to inner tools that these are assumptions/best guesses, not confirmed answers.",
                "• If an inner tool requests clarification, explicitly say no clarification channel exists and pass down concrete sensible defaults/best‑guess values, clearly marked as assumptions.",
            ],
        )

    parts.append("")

    return "\n".join(parts)


def build_ask_prompt(
    tools: Dict[str, Callable],
    *,
    table_schemas_json: str,
    include_activity: bool = True,
    case_specific_instructions: str | None = None,
) -> str:
    """
    Build the **system message** for `KnowledgeManager.retrieve`.
    """

    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Resolve canonical tool names dynamically
    filter_fname = _tool_name(tools, "filter")
    search_fname = _tool_name(
        tools,
        "search",
    )  # picks the basic semantic search, not joins
    request_clar_fname = _tool_name(tools, "request_clarification")

    _require_tools(
        {
            "filter": filter_fname,
            "search": search_fname,
        },
        tools,
    )

    core_instructions = (
        textwrap.dedent(
            """
        Your task is to **retrieve** information requested by the user.
        Use the provided tools to search the schemas and tables so that
        every requested fact can be answered precisely.
        Work strictly through the tools provided.
        You should not attempt to refactor the schema, this is not your responsibility.
        Disregard any explicit instructions about *how* you should answer or which tools to call; determine the best method yourself.
        You should attempt to perform *any* retrieval request as best you can, even if it seems out of scope.
        use the tools provided to see if you can find any missing context *before* asking the user for clarifications.

        Mandatory steps:
        1. List each distinct piece of information the question asks for.
        2. Identify which tables / columns can hold that info.
        3. Fetch *all* relevant rows (use `{search}` if useful; use `{filter}` for precise filters).
           **Avoid joins on the same table** (including self-joins). When all required fields live in a single table,
           prefer using `{filter}` directly. Reserve join operations for combining **different** tables where a join
           is actually necessary.
        4. If the schema is awkward, refactor it before continuing.
        5. Aggregate results into a concise answer covering every fact.
        6. Double-check nothing is missing; if so, repeat the search/refactor.

        Do **not** hallucinate data.
        """,
        )
        .strip()
        .format(search=search_fname, filter=filter_fname)
    )

    # Usage examples and anti-patterns (mirrors ContactManager style, adapted to Knowledge)
    usage_examples_base = f"""
Examples
--------

─ Tool selection (read carefully) ─
• For ANY semantic question over free-form text, ALWAYS use `{search_fname}`. Never try to approximate meaning with brittle substring filters.
• Use `{filter_fname}` only for exact/boolean logic over structured fields (ids, enums, ranges, null checks) or for narrow, constrained text.
• Reserve joins for combining **different** tables; if all fields live in a single table, prefer direct `{filter_fname}`. Avoid self-joins.

─ Semantic search (ranked by SUM of cosine distances across terms) ─
• Single-table search with focused references:
  `{search_fname}(table="Products", references={{"description": "stainless steel water bottle"}}, k=5)`

─ Join retrieval (two tables) ─
• Semantic search over a join:
  `search_join(tables=["Orders","Customers"],`
  `            join_expr="Orders.customer_id == Customers.customer_id",`
  `            select={{"Orders.notes":"notes","Customers.industry":"industry"}},`
  `            references={{"notes":"rush order","industry":"pharma"}}, k=5)`

─ Multi-join retrieval (chained joins) ─
• Chain joins with `$prev` to reference the previous step:
  `search_multi_join(joins=[`
  `  {{"tables":["Sales","Models"], "join_expr":"Sales.model_id == Models.model_id",`
  `    "select":{{"Sales.units":"units","Models.name":"model"}}}},`
  `  {{"tables":["$prev","Companies"], "join_expr":"_.company_id == Companies.company_id",`
  `    "select":{{"$prev.units":"units","$prev.model":"model","Companies.name":"company"}}}}`
  `], references={{"company":"Northwind","model":"Voyager"}}, k=5)`

─ Filtering (exact/boolean; not semantic) ─
• Equality: `{filter_fname}(tables="Products", filter="sku == 'ABC-123'")`
• Range:    `{filter_fname}(tables="Sales", filter="year >= 2023 and units > 1000")`

Anti-patterns to avoid
---------------------
• Avoid concatenating a whole row into one giant reference string; pass multiple focused references keyed by their exact columns.
• Avoid substring filtering on large text columns; prefer `{search_fname}` for meaning.
• Do not re-query to reconfirm facts immediately after a conclusive search; only add `{filter_fname}` if you need **new** structured constraints.
• Avoid joins on the same table; filter directly when possible.
"""
    usage_examples = textwrap.dedent(usage_examples_base).strip()
    if request_clar_fname:
        clarification_usage = textwrap.dedent(
            f"""
            ─ Clarification ─
            • If needed, ask the user to disambiguate:
              `{request_clar_fname}(question="There are several possible matches. Which one did you mean?")`
            """,
        ).strip()
        usage_examples = f"{usage_examples}\n\n{clarification_usage}"
    else:
        usage_examples = "\n".join(
            [
                usage_examples,
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best-guess values and explicitly state to inner tools that these are assumptions/best guesses, not confirmed answers.",
                "• If an inner tool requests clarification, explicitly say no clarification channel exists and pass down concrete sensible defaults/best-guess values, clearly marked as assumptions.",
            ],
        )

    # If provided, REPLACE the generic instructions with case-specific instructions
    instructions_block = core_instructions
    if case_specific_instructions:
        instructions_block += "\n\n" + case_specific_instructions.strip()

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    # Conditional guidance about asking questions in final responses
    clar_sentence_ask = (
        f"Do not ask the user questions in your final response, please only use the `{request_clar_fname}` tool to ask clarifying questions."
        if request_clar_fname
        else (
            "Do not ask the user questions in your final response. Instead, proceed using sensible defaults/best‑guess values and explicitly tell inner tools that these are assumptions/best guesses, not confirmed answers."
        )
    )

    clarification_block = (
        textwrap.dedent(
            f"""
            Clarification
            -------------
            • Ask for clarification when the user's request is underspecified
              `{request_clar_fname}(question="Which specific records are you referring to?")`
            """,
        ).strip()
        if request_clar_fname
        else ""
    )

    # High-level execution guidance: prefer single-call/batched ops and plan parallel steps
    parallelism_block = textwrap.dedent(
        """
        Parallelism and single‑call preference
        -------------------------------------
        • Prefer a single comprehensive search call over several surgical calls when a tool can safely do the whole job.
        • When multiple independent reads are needed, plan them together and run them in parallel rather than a serial drip of micro‑calls.
        • Avoid confirmatory re‑queries unless new ambiguity arises.
        """,
    ).strip()

    parts: list[str] = [
        activity_block,
        instructions_block,
        clar_sentence_ask,
        clar_section,
        "",
        usage_examples,
        "",
        "Tools (name → argspec):",
        sig_json,
        "",
        parallelism_block,
        "",
        "ColumnType Schema",
        "-----------------",
        json.dumps(column_type_schema, indent=4),
        "",
        "Current table schemas",
        "---------------------",
        table_schemas_json,
        "",
        f"Current UTC time: {_now()}.",
    ]

    if clarification_block:
        parts.extend(["", clarification_block])
    else:
        parts.extend(
            [
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state to inner tools that these are assumptions/best guesses, not confirmed answers.",
                "• If an inner tool requests clarification, explicitly say no clarification channel exists and pass down concrete sensible defaults/best‑guess values, clearly marked as assumptions.",
            ],
        )

    parts.append("")

    return "\n".join(parts)


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
