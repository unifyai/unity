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

    usage_examples_placeholder = textwrap.dedent(
        """
        Examples
        --------
        <add store-usage examples here>
        """,
    ).strip()

    core_instructions = textwrap.dedent(
        """
        Your task is to **store** new knowledge provided by the user.
        Keep the schema clean and future-proof – feel free to create,
        rename or delete tables / columns before inserting data.

        If the user refers to creating *tasks*, then you should **not** store any tasks.
        Tasks should exclusively be sotred by a separate task manager, this is **not your responsibility**.
        Please explain this to the user in your response, if this is part of the their request.

        Follow this workflow strictly:
        1. Extract every fact (subject → attribute → value) from the message.
        2. Decide whether each fact updates an existing row or inserts a new one.
        3. Add missing columns with the correct data-type where necessary.
        4. Use `_add_data` to write the changes.
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
            usage_examples_placeholder,
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

    usage_examples_placeholder = textwrap.dedent(
        """
        Examples
        --------
        <add retrieve-usage examples here>
        """,
    ).strip()

    core_instructions = textwrap.dedent(
        """
        Your task is to **retrieve** information requested by the user.
        Use the provided tools to search, transform or even refactor the
        schema so that every requested fact can be answered precisely.

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
            usage_examples_placeholder,
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
