from __future__ import annotations

import inspect
import json
import textwrap
from datetime import datetime, timezone
from typing import Dict, Callable, List

from .types.contact import Contact
from ..knowledge_manager.types import column_type_schema


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} for pretty JSON dumps."""
    return {n: str(inspect.signature(fn)) for n, fn in tools.items()}


def _now() -> str:  # UTC timestamp helper
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def _permanent_columns() -> str:
    """
    Return a comma-separated string of the *built-in* column names taken
    directly from the `Contact` Pydantic model (works on v1 & v2).  Any
    extra/custom fields are ignored because they are not part of the schema
    at import time.
    """
    return ", ".join(sorted(Contact.model_fields.keys()))


def _tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """
    Best-effort lookup utility: find the *first* tool whose name contains
    the given *needle* (case-insensitive).  Returns ``None`` if not found.
    """
    needle = needle.lower()
    return next((n for n in tools if needle in n.lower()), None)


def build_ask_prompt(
    tools: Dict[str, Callable],
    num_contacts: int,
    columns: List[Dict[str, str]],
) -> str:
    """Return the system-prompt used by *ask*."""
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    # Assume there is exactly *one* search-tool in the dict:
    search_name = next(iter(tools))

    # ------------------------------------------------------------------ #
    #  Dynamic helpers for custom-column tools
    # ------------------------------------------------------------------ #
    create_custom = _tool_name(tools, "create_custom_column")
    delete_custom = _tool_name(tools, "delete_custom_column")
    list_columns = _tool_name(tools, "list_columns")
    nearest_search = _tool_name(tools, "nearest_column")

    # ------------------------------------------------------------------ #
    #  Usage snippets (standard search + custom-column examples)
    # ------------------------------------------------------------------ #
    usage_examples = textwrap.dedent(
        f"""
        Examples
        --------
        • Find contacts with first name **John**
          `{search_name}(filter="first_name == 'John'")`
        • Find surname **Doe**
          `{search_name}(filter="surname == 'Doe'")`
        • Specific email **john.doe@example.com**
          `{search_name}(filter="email_address == 'john.doe@example.com'")`
        • Phone containing **555**
          `{search_name}(filter="'555' in phone_number")`
        • Exact phone **+14445556666**
          `{search_name}(filter="phone_number == '+14445556666'")`
        • Name **Alice Smith**
          `{search_name}(filter="surname == 'Smith' and first_name == 'Alice'")`
        • Email **a@b.com** *or* phone **123-456-7890**
          `{search_name}(filter="email_address == 'a@b.com' or phone_number == '123-456-7890'")`
        • Missing phone number
          `{search_name}(filter="phone_number is None")`
        • Has any email (not None)
          `{search_name}(filter="email_address is not None")`

        ─ Semantic search ─
        • Find contacts *similar* to "machine-learning expert" in the *description* field
          `{nearest_search}(source='description', text='machine-learning expert')`

        ─ Custom columns ─
        • Inspect schema
          `{list_columns}()`
        • Add a “linkedin” field
          `{create_custom}(column_name='linkedin', column_type='str')`
        • Delete it again
          `{delete_custom}(column_name='linkedin')`
    """,
    ).strip()

    if num_contacts < 50:
        guidance = f"given that the number of contacts is so small, you should simply use {search_name} with *no filter arguments* for now, so you can unpack the *full* contact list and answer the question directly."
    else:
        guidance = (
            "If the question is open-ended or doesn't clearly match any of the column names,",
            f"then try {nearest_search} on the most relevant column(s) and see if you can find any semantic match.",
        )

    return "\n".join(
        [
            "You are an assistant specializing in **retrieving contact information**.",
            "Work strictly through the tools provided.",
            "You should attempt to answer *all* questions as best you can, even if they seem out of scope."
            "",
            f"There are currently {num_contacts} contacts are stored in a table with the following colums:",
            json.dumps(columns, indent=4),
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            usage_examples if num_contacts >= 50 else "",
            "",
            guidance,
            "",
            f"Current UTC time is {_now()}.",
        ],
    )


def build_update_prompt(tools: Dict[str, Callable]) -> str:
    """Return the system-prompt used by *update*."""
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Pick out canonical names heuristically (all dynamic)
    create_name = _tool_name(tools, "create_contact")
    update_name = _tool_name(tools, "update_contact")
    search_name = _tool_name(tools, "search_contacts")

    # Custom-column helpers (dynamic)
    create_custom = _tool_name(tools, "create_custom_column")
    delete_custom = _tool_name(tools, "delete_custom_column")
    list_columns = _tool_name(tools, "list_columns")
    nearest_search = _tool_name(tools, "nearest_column")

    usage_examples = textwrap.dedent(
        f"""
        Examples
        --------
        • **Create** a new contact
          `{create_name}(first_name='Jane', surname='Roe', email_address='jane.roe@example.com')`

        • **Update** John Doe's phone '+1 55512-345-67' when you already know the ID is *42*
          `{update_name}(contact_id=42, phone_number='+15551234567')` (note spaces and dashes removed)

        • **Update** a contact referred to only by name
          1 Find ID → `{search_name}(filter="first_name == 'John' and surname == 'Doe'")`
          2 Then update → `{update_name}(contact_id=<returned_id>, email_address='john.new@example.com')`

        • **Parse** a full name on create
          `"Frank P. Castle"` → `{create_name}(first_name='Frank P.', surname='Castle')`

        ─ Custom columns ─
        • New column *“department”*
          `{create_custom}(column_name='department', column_type='str')`
        • Update a contact's department
          `{update_name}(contact_id=42, department='Engineering')`
        • Remove the column later
          `{delete_custom}(column_name='department')`

        ─ Semantic search example ─
        • Retrieve top 3 contacts whose *department* is semantically close to "data science"
          `{nearest_search}(source='department', text='data science', k=3)`
    """,
    ).strip()

    return "\n".join(
        [
            "You are an assistant in charge of **creating or editing contacts**.",
            "Use the tools provided to create new entries or update existing ones.",
            "",
            "Custom columns:",
            "---------------",
            f"• Required columns ({_permanent_columns()}) **cannot** be deleted.",
            f"• Add a new column with `{create_custom}(…)`, remove with `{delete_custom}(…)`,",
            f"  and list columns with `{list_columns}()`.",
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            usage_examples,
            "",
            "Contact schema:",
            json.dumps(Contact.model_json_schema(), indent=4),
            "",
            "ColumnType schema (for custom columns):",
            json.dumps(column_type_schema, indent=4),
            "",
            f"Current UTC time is {_now()}.",
        ],
    )
