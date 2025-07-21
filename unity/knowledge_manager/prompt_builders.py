from __future__ import annotations

import inspect
import json
import textwrap
from datetime import datetime, timezone
from typing import Callable, Dict

from .types import column_type_schema
from ..memory_manager.broader_context import get_broader_context
from ..common.prompt_helpers import clarification_guidance

# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} for pretty JSON dumps."""
    return {n: str(inspect.signature(fn)) for n, fn in tools.items()}


def _now() -> str:  # UTC timestamp helper
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


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

    base_prompt = textwrap.dedent(
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

    activity_block = "{broader_context}" if include_activity else ""
    return activity_block + "\n\n" + base_prompt


def build_update_prompt(
    tools: Dict[str, Callable],
    *,
    table_schemas_json: str,
    include_activity: bool = True,
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
        Your task is to **store** new knowledge or **update** existing knowledge provided by the user.
        Keep the schema clean and future-proof – feel free to create,
        rename or delete tables / columns before inserting data.
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
        5. Use `_add_rows` to insert and `_update_rows` to modify existing rows.
        6. Search again to verify everything was stored or updated correctly.
        7. Reply with a short natural-language confirmation of what was stored.

        Do **not** hallucinate data.
        """,
    ).strip()

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    return "\n".join(
        [
            activity_block,
            core_instructions,
            clar_section,
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


def build_ask_prompt(
    tools: Dict[str, Callable],
    *,
    table_schemas_json: str,
    include_activity: bool = True,
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

        Do **not** hallucinate data.
        """,
    ).strip()

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    return "\n".join(
        [
            activity_block,
            core_instructions,
            clar_section,
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
