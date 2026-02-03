from __future__ import annotations

import textwrap
from typing import Dict, Callable, List, Union

from .types.contact import Contact
from ..knowledge_manager.types import column_type_schema
from ..common.prompt_helpers import (
    sig_dict,
    now,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
    get_custom_columns,
    # New standardized composer utilities
    PromptSpec,
    PromptParts,
    compose_system_prompt,
    special_contacts_block as _special_contacts_block,
)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} using shared helper."""
    return sig_dict(tools)


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
    columns: Union[List[Dict[str, str]], Dict[str, str]],
    *,
    include_activity: bool = True,
) -> PromptParts:
    """Return the system-prompt used by *ask* using the shared composer."""
    # Extract custom columns (not in Contact model)
    custom_cols = get_custom_columns(Contact, columns)

    # ------------------------------------------------------------------ #
    #  Dynamic helpers for custom-column tools
    # ------------------------------------------------------------------ #
    filter_contacts_fname = _tool_name(tools, "filter_contacts")
    search_contacts_fname = _tool_name(tools, "search_contacts")
    list_columns_fname = _tool_name(tools, "list_columns")
    reduce_fname = _tool_name(tools, "reduce")

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

─ Tool selection (read carefully) ─
• For ANY semantic question over free‑form text (bio, rolling_summary, custom text columns), ALWAYS use `{search_contacts_fname}`. Never try to approximate meaning with brittle substring filters.
• Use `{filter_contacts_fname}` only for exact/boolean logic over structured fields (emails, phone numbers, null checks) or for narrow, constrained text where substring checks make sense (e.g., case‑insensitive contains on first_name).

─ Semantic search: targeted references across columns ─
• When the clue could appear across several free‑form fields, provide separate, surgical references instead of one catch‑all. This yields stronger ranking than concatenating everything into one giant string. For example, find the San Francisco software engineer we worked on onboarding with last week:
  `{search_contacts_fname}(references={{'bio': 'San Francisco software engineer', 'rolling_summary': 'worked on onboarding last week'}}, k=2)`

• Find someone based in Berlin working as a product designer (signal lives in `bio`)
  `{search_contacts_fname}(references={{'bio': 'based in Berlin product designer'}}, k=3)`

• Find the accountant who we had a call with last week
  `{search_contacts_fname}(references={{'occupation': 'accountant', 'rolling_summary': 'had a call last week'}}, k=3)`

─ Derived expression (fallback, when you truly cannot target columns) ─
• Build one composite expression spanning likely fields, then search it. Prefer multi‑column references when you know where the signal lives.
  `expr = "str({{skills}}) + ' ' + str({{occupation}}) "`
  `{search_contacts_fname}(references={{expr: 'Software engineering'}}, k=2)`

─ Filtering (exact/boolean or constrained text only; not semantic) ─
• Exact email match
  `{filter_contacts_fname}(filter="email_address == 'jane.roe@example.com'")`
• Has no phone number
  `{filter_contacts_fname}(filter="phone_number is None")`
• Case‑insensitive first‑name contains (acceptable because the field is short and constrained)
  `{filter_contacts_fname}(filter="first_name is not None and 'dan' in first_name.lower()")`

─ Numeric aggregations ─
• For numeric reduction metrics (count, sum, mean, min, max, median, mode, var, std) over numeric columns, use `{reduce_fname}` instead of filtering and computing in-memory.
  `{reduce_fname}(metric='mean', keys='age', group_by='city')`

Anti‑patterns to avoid
---------------------
• Avoid the default search behaviour of concatenating every column into one long string and comparing a single embedding of the whole question. Instead, pass multiple, focused reference texts keyed by their specific columns. The ranking minimizes the sum of cosine distances and is more accurate and robust.
• Avoid filtering for text-heavy columns, sub-string matching is *very* brittle
• Avoid making another search/filter call to reconfirm information that a previous step already established clearly (e.g., the same contact_id and fields); proceed with the established result unless new ambiguity arises.
• Do not automatically chain a filter immediately after a successful semantic search unless you need an exact, structured constraint that the search result does not provide.
    """
    usage_examples = textwrap.dedent(usage_examples_base).strip()
    if clarification_block:
        usage_examples = f"{usage_examples}\n{clarification_block}"
    else:
        # No clarification tool – append conditional anti‑pattern bullets (no extra heading)
        usage_examples = "\n".join(
            [
                usage_examples,
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state to inner tools that these are assumptions/best guesses, not confirmed answers.",
                "• If an inner tool requests clarification, explicitly say no clarification channel exists and pass down concrete sensible defaults/best‑guess values, clearly marked as assumptions.",
                "• Avoid repeating `ask` within the same reasoning chain when earlier calls already surfaced the required contact(s) and no new ambiguity has been introduced.",
            ],
        )

    # Special contacts via shared helper
    special_block = _special_contacts_block()

    # Early exit policy for mutation-intent requests reaching ask() is handled by composer toggle

    # Build spec using standardized composer with schema-based table info
    spec = PromptSpec(
        manager="ContactManager",
        method="ask",
        tools=tools,
        role_line="You are an assistant specializing in **retrieving contact information**.",
        global_directives=[
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
            "You should attempt to answer *any* question as best you can, even if it seems out of scope.",
            "use the tools provided to see if you can find any missing context *before* asking the user for clarifications.",
        ],
        include_read_only_guard=True,
        positioning_lines=[],
        counts_entity_plural="contacts",
        counts_value=num_contacts,
        # Schema-based table info (avoids duplication)
        table_schema_name="Contact",
        custom_columns=custom_cols if custom_cols else None,
        include_tools_block=True,
        usage_examples=usage_examples,
        clarification_examples_block=clarification_block or None,
        include_images_policy=True,
        include_images_forwarding=True,
        images_extras_block=None,
        include_parallelism=True,
        schemas=[("Contact", Contact)],
        special_blocks=[special_block],
        include_clarification_footer=True,
        include_time_footer=True,
    )

    return compose_system_prompt(spec)


def build_update_prompt(
    tools: Dict[str, Callable],
    num_contacts: int,
    columns: Union[List[Dict[str, str]], Dict[str, str]],
    *,
    include_activity: bool = True,
) -> PromptParts:
    """Return the system-prompt used by *update* using the shared composer."""
    # Extract custom columns (not in Contact model)
    custom_cols = get_custom_columns(Contact, columns)

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

Ask vs Clarification
----------------------
• `{ask_fname}` is ONLY for inspecting/locating contacts that ALREADY EXIST (e.g., to find `contact_id`, verify fields).
• Do NOT use `{ask_fname}` to ask the human for details about NEW contacts being created/changed in this update request.
• For human clarifications about prospective/new contacts (e.g., name spelling, missing numbers, preferred channel), call `{request_clar_fname}` when available.
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

• Mark should_respond=True for the contact who is a footballer and recently wrapped up a kickoff call
  1 Ask a freeform question (no instructions about how to answer):
    `{ask_fname}(text="Which footballer wrapped up a kickoff call last week?")`
  2 Update the returned id:
    `{update_fname}(contact_id=<id>, should_respond=True)`

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
• Merge two contacts when instructed. Use the `overrides` map to choose winners; include "contact_id" set to 1 or 2 to select the surviving id. For each field, use 1 or 2 to select from the corresponding source (never literal ids). Protect ids 0 and 1 from deletion:
  `{merge_fname}(contact_id_1=12, contact_id_2=34, overrides={{'contact_id': 1, 'email_address': 2}})`
• Delete a contact only when clearly requested (never ids 0 or 1):
  `{delete_fname}(contact_id=77)`

Basic create/update
-------------------
• Create a new contact
  `{create_fname}(first_name='Jane', surname='Roe', email_address='jane.roe@example.com')`
• Update a known contact id
  `{update_fname}(contact_id=42, phone_number='+15551234567')`

Asking Questions
----------------
• It can often be very difficult to keep track of your own 'update' progress purely via tool call histories. If you're unsure, always just `ask`!
  `{ask_fname}(text="I think I've now updated all of the contact 'occupation' columns to engineer, but I might have missed some. Could you list all engineers in the contact list so I can check my progress?")`

Anti‑patterns to avoid
---------------------
• Repeating the exact same tool call with the same arguments as a means to 'make sure it has completed', just call `ask` to check the latest state of the contacts list
• Making *any* assumptions about the current state of the contacts list, instead you should make liberal use of the `ask` tool

(When locating a record by semantics, always do a quick `{ask_fname}` step to resolve `contact_id` before mutating. Prefer updating in place over recreating.)
    """
    usage_examples = textwrap.dedent(usage_examples_base).strip()
    if clarification_block:
        usage_examples = f"{usage_examples}\n{clarification_block}"
    else:
        usage_examples = "\n".join(
            [
                usage_examples,
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state to inner tools that these are assumptions/best guesses, not confirmed answers.",
                "• If an inner tool requests clarification, explicitly say no clarification channel exists and pass down concrete sensible defaults/best‑guess values, clearly marked as assumptions.",
            ],
        )

    # Compose using standardized composer with schema-based table info
    special_block = _special_contacts_block()

    # Schemas: Contact for table columns, ColumnType for custom column creation
    schemas = [
        ("Contact", Contact),  # Full schema defines table columns
        ("ColumnType (for custom columns)", column_type_schema),
    ]

    spec = PromptSpec(
        manager="ContactManager",
        method="update",
        tools=tools,
        role_line="You are an assistant in charge of **creating or editing contacts**.",
        global_directives=[
            "Choose tools based on the user's intent and the specificity of the target record.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the request and choose the best approach yourself.",
            f"Important: `{ask_fname}` is read‑only and must only be used to locate/inspect contacts that already exist.",
            "Before creating new contacts or making edits, briefly check whether similar contacts already exist (via `"
            + ask_fname
            + "`) to avoid duplicates.",
            "Prefer minimal, precise mutations to existing records identified by contact_id.",
            "When the user describes a contact semantically, resolve the id first by requesting the contact_id from the ask method, then perform the update via the contact_id.",
            "use the `ask` method to see if you can find any missing context *before* you consider asking the user for clarifications.",
            "If the `ask` method is the only available tool, then ask a *read-only question*, mutation-capable tools will be exposed in subsequent turns.",
        ],
        include_read_only_guard=False,
        positioning_lines=[],
        counts_entity_plural="contacts",
        counts_value=num_contacts,
        # Schema-based table info (avoids duplication)
        table_schema_name="Contact",
        custom_columns=custom_cols if custom_cols else None,
        include_tools_block=True,
        usage_examples=usage_examples,
        clarification_examples_block=clarification_block or None,
        include_images_policy=True,
        include_images_forwarding=True,
        images_extras_block=None,
        include_parallelism=True,
        schemas=schemas,
        special_blocks=[
            special_block,
            "Do not create new columns if an alias already exists.",
        ],
        include_clarification_footer=True,
        include_time_footer=True,
    )

    return compose_system_prompt(spec)


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
