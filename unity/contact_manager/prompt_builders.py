from __future__ import annotations

import json
import textwrap
from typing import Dict, Callable, List

from .types.contact import Contact
from ..knowledge_manager.types import column_type_schema
from ..common.prompt_helpers import (
    clarification_guidance,
    sig_dict,
    now_utc_str,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} using shared helper."""
    return sig_dict(tools)


def _now() -> str:  # UTC timestamp helper
    return now_utc_str()


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
    """Delegates to shared tool name resolver."""
    return _shared_tool_name(tools, needle)


def _require_tools(pairs: Dict[str, str | None], tools: Dict[str, Callable]) -> None:
    """Delegate validation to the shared helper for consistent errors."""
    _shared_require_tools(pairs, tools)


# Replace build_ask_prompt with a space-indented version including improved guidance


def build_ask_prompt(
    tools: Dict[str, Callable],
    num_contacts: int,
    columns: List[Dict[str, str]],
    *,
    include_activity: bool = True,
) -> str:
    """Return the system-prompt used by *ask*."""
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # ------------------------------------------------------------------ #
    #  Dynamic helpers for custom-column tools
    # ------------------------------------------------------------------ #
    filter_contacts_fname = _tool_name(tools, "filter_contacts")
    search_contacts_fname = _tool_name(tools, "search_contacts")
    list_columns_fname = _tool_name(tools, "list_columns")

    # Clarification helper (only present when the caller provided queues)
    request_clar_fname = _tool_name(tools, "request_clarification")

    # Validate required tools (request_clar_fname is optional)
    _require_tools(
        {
            "filter_contacts": filter_contacts_fname,
            "search_contacts": search_contacts_fname,
            "list_columns": list_columns_fname,
        },
        tools,
    )

    # ------------------------------------------------------------------ #
    #  Usage snippets (standard search + custom-column examples)
    # ------------------------------------------------------------------ #
    clarification_block = (
        textwrap.dedent(
            f"""
            ─ Clarification ─
            • Ambiguous request for "Alice" when multiple Alices exist – ask the user which one they mean
              `{request_clar_fname}(question="There are several contacts named Alice. Which one did you mean?")`
            """,
        ).strip()
        if request_clar_fname
        else ""
    )

    # Strongly emphasize correct tool selection and realistic semantic-search usage
    usage_examples_base = f"""
Examples
--------

─ Columns ─
• Inspect schema
  `{list_columns_fname}()`

─ Tool selection ─
• If the question asks about a semantic property inside any text field (e.g., where someone lives, what they do, age, or recent activity), USE `{search_contacts_fname}`.
• Use `{filter_contacts_fname}` only for exact/boolean logic over structured fields (e.g., exact email match, phone equals a number, missing/nonnull checks) or when you must enumerate precise rows.

─ Semantic search (realistic scenarios) ─
• Find the contact who lives in Berlin and works as a product designer (signal usually lives in free‑text `bio`)
  `{search_contacts_fname}(references={{'bio': 'lives in Berlin product designer'}}, k=3)`

• Identify who recently moved to New York and is training for a marathon (biographical signal in `bio`)
  `{search_contacts_fname}(references={{'bio': 'moved to New York training for a marathon'}}, k=2)`

• Combine multiple signals from different columns – rank by the SUM of cosine distances
  (e.g., someone whose `rolling_summary` mentions "wrapped up a kickoff call last week" and whose occupation mentions "footballer")
  `{search_contacts_fname}(references={{'rolling_summary': 'wrapped up a kickoff call last week', 'occupation': 'footballer'}}, k=3)`

• Search across freeform fields with a single derived expression when the clue may appear across `bio`, `rolling_summary`, or a custom field like `occupation`
  First build an expression, then pass it as the reference key:
  `expr = "str({{bio}}) + ' ' + str({{rolling_summary}}) + ' ' + str({{occupation}})"`
  For a query like "based in London, 28 years old, software engineer" this will surface matching bios, recent activity, and occupations:
  `{search_contacts_fname}(references={{expr: 'London 28 software engineer'}}, k=2)`

• Prefer multiple smaller, targeted references over one generic catch‑all when you know where the signal likely lives
  (e.g., split across `bio` and `rolling_summary` rather than concatenating everything) – this often improves ranking.

─ Filtering (exact matching, not semantic) ─
• First name is exactly John
  `{filter_contacts_fname}(filter="first_name == 'John'")`
• Surname is Doe
  `{filter_contacts_fname}(filter="surname == 'Doe'")`
• Specific email is john.doe@example.com
  `{filter_contacts_fname}(filter="email_address == 'john.doe@example.com'")`
• Phone contains 555 (substring logic; not semantic similarity)
  `{filter_contacts_fname}(filter="'555' in phone_number")`
• Exact phone equals +14445556666
  `{filter_contacts_fname}(filter="phone_number == '+14445556666'")`
• Missing phone number
  `{filter_contacts_fname}(filter="phone_number is None")`
• Has any email (not None)
  `{filter_contacts_fname}(filter="email_address is not None")`
    """
    usage_examples = textwrap.dedent(usage_examples_base).strip()
    if clarification_block:
        usage_examples = f"{usage_examples}\n{clarification_block}"

    # Decision guidance – emphasize semantic search even on small tables
    if num_contacts < 50:
        guidance = "\n".join(
            [
                "The table is small, but still choose tools by intent:",
                f"• If the user asks anything semantic about text (habits, preferences, summaries), call {search_contacts_fname}.",
                f"• Only fetch all contacts via {filter_contacts_fname}(filter=None) if you truly need to list or scan everything explicitly.",
            ],
        )
    else:
        guidance = "\n".join(
            [
                "When the question is open‑ended or refers to meaning rather than exact values,",
                f"use {search_contacts_fname} on the most relevant text columns.",
                "Split the query across multiple columns when signals live in different places; the ranking minimizes the sum of cosine distances.",
            ],
        )

    # ─ Clarification guidance ─
    clar_section = clarification_guidance(tools)

    activity_block = "{broader_context}" if include_activity else ""

    return "\n".join(
        [
            activity_block,
            "You are an assistant specializing in **retrieving contact information**.",
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best method yourself.",
            "You should attempt to answer *any* question as best you can, even if it seems out of scope.",
            "use the tools provided to see if you can find any missing context *before* asking the user for clarifications.",
            "",
            f"There are currently {num_contacts} contacts are stored in a table with the following colums:",
            json.dumps(columns, indent=4),
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            usage_examples,
            "",
            guidance,
            clar_section,
            "",
            f"Current UTC time is {_now()}.",
        ],
    )


def build_update_prompt(
    tools: Dict[str, Callable],
    *,
    include_activity: bool = True,
) -> str:
    """Return the system-prompt used by *update*."""
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Pick out canonical names heuristically (all dynamic)
    create_fname = _tool_name(tools, "create_contact")
    update_fname = _tool_name(tools, "update_contact")
    delete_fname = _tool_name(tools, "delete_contact")
    merge_fname = _tool_name(tools, "merge_contacts")
    ask_fname = _tool_name(tools, "ask")

    # Custom-column helpers (dynamic)
    create_custom_fname = _tool_name(tools, "create_custom_column")
    delete_custom_fname = _tool_name(tools, "delete_custom_column")

    # Clarification helper (optional)
    request_clar_fname = _tool_name(tools, "request_clarification")

    # Validate required tools (request_clar_fname is optional)
    _require_tools(
        {
            "create_contact": create_fname,
            "update_contact": update_fname,
            "delete_contact": delete_fname,
            "merge_contacts": merge_fname,
            "create_custom_column": create_custom_fname,
            "delete_custom_column": delete_custom_fname,
            "ask": ask_fname,
        },
        tools,
    )

    clarification_block = (
        textwrap.dedent(
            f"""
Clarification
-------------
• If any request is ambiguous, ask the user to disambiguate before changing data
  `{request_clar_fname}(question="There are several possible matches. Which contact did you mean?")`
            """,
        ).strip()
        if request_clar_fname
        else ""
    )

    usage_examples_base = f"""
Tool selection
--------------
• Prefer `{update_fname}` when you know the exact `contact_id` for a mutation.
• When the user refers to a contact semantically (e.g., "the footballer who wrapped up a kickoff call last week"), first ask a freeform question with `{ask_fname}` to identify the correct `contact_id`, then call `{update_fname}`.
• If the schema lacks a field the user wants to set, create it with `{create_custom_fname}` (typically `column_type='str'`) before updating.
• Use `{merge_fname}` only when the user explicitly asks to combine two known contacts or when duplicates are clearly identified.
• Use `{delete_fname}` only on explicit deletion requests. Never delete system contacts with id 0 or 1.

Realistic find-then-update flows
--------------------------------
• Set a policy for the contact living in Berlin working as a product designer
  1 Ask a freeform question (no instructions about how to answer):
    `{ask_fname}(text="Who is the contact living in Berlin working as a product designer?")`
  2 Update the returned id:
    `{update_fname}(contact_id=<id>, response_policy="Share design updates weekly")`

• Mark respond_to=True for the contact who is a footballer and recently wrapped up a kickoff call
  1 Ask a freeform question (no instructions about how to answer):
    `{ask_fname}(text="Which footballer wrapped up a kickoff call last week?")`
  2 Update the returned id:
    `{update_fname}(contact_id=<id>, respond_to=True)`

• Query may span multiple freeform fields (derived expression)
  1 Build a composite expression across `bio`, `rolling_summary`, and a custom field like `occupation`:
    `expr = "str({{bio}}) + ' ' + str({{rolling_summary}}) + ' ' + str({{occupation}})"`
  2 Ask a freeform question referring to the same clues:
    `{ask_fname}(text="Who is the London-based 28-year-old software engineer?")`
  3 Update the found record as requested.

Schema evolution and custom columns
----------------------------------
• If the user asks to store a new attribute that does not map to built-ins, create a custom column first:
  `{create_custom_fname}(column_name='occupation', column_type='str')`
  Then apply the update:
  `{update_fname}(contact_id=42, occupation='Designer')`
• Required columns ({_permanent_columns()}) cannot be deleted. Remove optional columns with `{delete_custom_fname}(column_name=...)` only when explicitly asked.

Merge and delete
----------------
• Merge two contacts when instructed. Decide field winners via the `overrides` map and protect ids 0 and 1 from deletion:
  `{merge_fname}(contact_id_1=12, contact_id_2=34, overrides={{'contact_id': 12, 'email_address': 2}})`
• Delete a contact only when clearly requested (never ids 0 or 1):
  `{delete_fname}(contact_id=77)`

Basic create/update
-------------------
• Create a new contact
  `{create_fname}(first_name='Jane', surname='Roe', email_address='jane.roe@example.com')`
• Update a known contact id
  `{update_fname}(contact_id=42, phone_number='+15551234567')`

(When locating a record by semantics, always do a quick `{ask_fname}` step to resolve `contact_id` before mutating. Prefer updating in place over recreating.)
    """
    usage_examples = textwrap.dedent(usage_examples_base).strip()
    if clarification_block:
        usage_examples = f"{usage_examples}\n{clarification_block}"

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    return "\n".join(
        [
            activity_block,
            "You are an assistant in charge of **creating or editing contacts**.",
            "Choose tools based on the user's intent and the specificity of the target record.",
            "Prefer minimal, precise mutations to existing records identified by contact_id.",
            "When the user describes a contact semantically, resolve the id first by calling the ask method and using semantic search, then perform the update.",
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
            clar_section,
            "",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulated helper
# ─────────────────────────────────────────────────────────────────────────────


def build_simulated_method_prompt(
    method: str,
    user_request: str,
    parent_chat_context: list[dict] | None = None,
) -> str:
    """Return an *instruction* prompt for the simulated ContactManager.

    This helper is used *only* by the **simulated** implementation to give the
    LLM very explicit guidance so that it *pretends* the method call has
    *already* finished.  It avoids responses such as "I'll process that now …"
    and instead instructs the model to respond in **past tense** – as if the
    requested action has been *completed*.

    The wording mirrors the style already used in :class:`SimulatedContactManager`.
    """
    import json  # local import to avoid polluting module namespace

    preamble = f"On this turn you are simulating the '{method}' method."
    if method.lower() == "ask":
        behaviour = (
            "Please always *answer* the question (invent a plausible response). "
            "Do *not* ask the user for clarification and do *not* describe how "
            "you will find the answer – simply provide the final, imaginary answer."
        )
    else:  # update / store / etc.
        behaviour = (
            "Please always act as though the request has been **completed**. "
            "Respond in past tense, e.g. 'Completed the requested update – here are the details: …'. "
            "Do *not* say things like 'I'll process this now'."
        )

    parts: list[str] = [preamble, behaviour, "", f"The user input is:\n{user_request}"]
    if parent_chat_context:
        parts.append(
            f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}",
        )

    return "\n".join(parts)
