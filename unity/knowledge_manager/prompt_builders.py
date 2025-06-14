from __future__ import annotations

import inspect
import json
import textwrap
from datetime import datetime, timezone
from typing import Callable, Dict

from .types import column_type_schema

# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} for pretty JSON dumps."""
    return {n: str(inspect.signature(fn)) for n, fn in tools.items()}


def _now() -> str:  # UTC timestamp helper
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ────────────────────────────────────────────────────────────────────────────
# public builders
# ────────────────────────────────────────────────────────────────────────────


def build_refactor_prompt(
    tools: dict[str, callable],
    *,
    table_schemas_json: str,
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

    tools_section = "\n".join(
        f"- **{name}**{sig}"
        for name, sig in ((t.__name__, str(t.__annotations__)) for t in tools.values())
    )

    examples = textwrap.dedent(
        """
        ### EXAMPLE 1 — simple column move
        *Before*
        ┌─ Companies(name, revenue, opening_hours)
        └─ Contacts(first_name, surname, email_address, **opening_hours**)

        *Action*
        1. `delete_column(table="Contacts", column_name="opening_hours")`
        2. `create_empty_column(table="Companies", column_name="company_id", column_type="int")`
        3. `rename_column(table="Companies", old_name="name", new_name="company_name")`
        4. Update rows so every contact references `company_id`.

        ### EXAMPLE 2 — splitting a mixed-type column
        • Detect that `purchase_info` mixes JSON dicts and strings.
        • Create two new columns (`purchase_details` *dict*, `purchase_note` *str*)
        • Migrate the correct rows with tool calls (`create_derived_column`, `delete_column`, etc.).
        """,
    ).strip()

    return textwrap.dedent(
        f"""
        You are the **Schema Refactor Assistant**.
        Your only goal is to *minimise duplication* and *maximise clarity* of
        the stored data model by judicious use of the tools listed below.
        You should attempt to perform *any* refactor request as best you can, even if it seems out of scope.
        use the tools provided to see if you can find any missing context *before* asking the user for clarifications.

        --------------------------------------------------------------------
        ## Current schema (JSON)
        {table_schemas_json}

        --------------------------------------------------------------------
        ## Available tools
        {tools_section}

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

        --------------------------------------------------------------------
        ## Usage examples
        {examples}
        """,
    ).strip()


def build_store_prompt(
    tools: Dict[str, Callable],
    *,
    table_schemas_json: str,
) -> str:
    """
    Build the **system message** for `KnowledgeManager.store`.

    Parameters
    ----------
    tools
        Mapping *name → callable* that will actually be exposed to the LLM.
        Only the **name** and **argspec** are surfaced (no docstrings).
    table_schemas_json
        ``json.dumps`` of the existing table-schema dictionary.
    """

    sig_json = json.dumps(_sig_dict(tools), indent=4)

    core_instructions = textwrap.dedent(
        """
        Your task is to **store** new knowledge provided by the user.
        Keep the schema clean and future-proof – feel free to create,
        rename or delete tables / columns before inserting data.
        You should attempt to perform *any* storage request as best you can, even if it seems out of scope.
        use the tools provided to see if you can find any missing context *before* asking the user for clarifications.

        If the user refers to creating or updating *tasks*, then you should **not** store any tasks.
        Tasks should exclusively be stored by a separate task manager, this is **not your responsibility**.
        Please explain this to the user in your response, if this is part of the their request.

        Follow this workflow strictly:
        1. Extract every fact (subject → attribute → value) from the message.
        2. Decide whether each fact updates an existing row or inserts a new one.
        3. Add missing columns with the correct data-type where necessary.
        4. Use `_add_rows` to insert and `_update_rows` to modify existing ones.
        5. Search again to verify everything stored correctly.
        6. Reply with a short natural-language confirmation of what was stored.

        If anything is ambiguous, call `request_clarification` **before** writing.
        Do **not** hallucinate data.
        """,
    ).strip()

    return "\n".join(
        [
            core_instructions,
            "",
            "Tools (name → argspec)",
            "---------------------",
            sig_json,
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
        ],
    )


def build_retrieve_prompt(
    tools: Dict[str, Callable],
    *,
    table_schemas_json: str,
) -> str:
    """
    Build the **system message** for `KnowledgeManager.retrieve`.
    """

    sig_json = json.dumps(_sig_dict(tools), indent=4)

    core_instructions = textwrap.dedent(
        """
        Your task is to **retrieve** information requested by the user.
        Use the provided tools to search, transform or even refactor the
        schema so that every requested fact can be answered precisely.
        You should attempt to perform *any* retrieval request as best you can, even if it seems out of scope.
        use the tools provided to see if you can find any missing context *before* asking the user for clarifications.

        Mandatory steps:
        1. List each distinct piece of information the question asks for.
        2. Identify which tables / columns can hold that info.
        3. Fetch *all* relevant rows (use `_nearest` if useful).
        4. If the schema is awkward, refactor it before continuing.
        5. Aggregate results into a concise answer covering every fact.
        6. Double-check nothing is missing; if so, repeat the search/refactor.

        Call `request_clarification` whenever uncertain.
        """,
    ).strip()

    return "\n".join(
        [
            core_instructions,
            "",
            "Tools (name → argspec)",
            "---------------------",
            sig_json,
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
        ],
    )
