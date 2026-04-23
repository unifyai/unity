"""
Prompt builders for SecretManager.

These builders parallel *contact_manager/prompt_builders.py*: they receive
a **live** ``tools``-dict and construct the corresponding **system** messages
*without ever hard-coding* tool counts, names or arg-signatures.
"""

from __future__ import annotations

import textwrap
from typing import Dict, Callable

from ..common.prompt_helpers import (
    clarification_guidance,
    sig_dict,
    now,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
    # Standardized composer utilities
    PromptSpec,
    PromptParts,
    compose_system_prompt,
)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} using shared helper."""
    return sig_dict(tools)


def _tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """Delegate to shared tool name resolver."""
    return _shared_tool_name(tools, needle)


def _require_tools(pairs: Dict[str, str | None], tools: Dict[str, Callable]) -> None:
    """Delegate validation to shared helper for consistent errors."""
    _shared_require_tools(pairs, tools)


# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def build_ask_prompt(*, tools: Dict[str, Callable]) -> PromptParts:
    """Return the system prompt used by SecretManager.ask using the shared composer.

    Emphasises: never reveal raw secret values; reference via ${name};
    use provided tools to list/search/filter. All storage is in Unify.
    """
    # Resolve canonical tool names dynamically
    list_columns_fname = _tool_name(tools, "list_columns")
    filter_secrets_fname = _tool_name(tools, "filter_secrets")
    search_secrets_fname = _tool_name(tools, "search_secrets")
    list_secret_keys_fname = _tool_name(tools, "list_secret_keys")
    request_clar_fname = _tool_name(tools, "request_clarification")

    # Validate required tools
    _require_tools(
        {
            "filter_secrets": filter_secrets_fname,
            "search_secrets": search_secrets_fname,
            "list_secret_keys": list_secret_keys_fname,
        },
        tools,
    )

    # Build clarification block
    clarification_block = (
        textwrap.dedent(
            f"""
            ─ Clarification ─
            • Ambiguity about which secret you meant – ask the user to specify
              `{request_clar_fname}(question="There are several matching secrets. Which one do you mean?")`
            """,
        ).strip()
        if request_clar_fname
        else ""
    )

    # Usage examples
    usage_examples_base = f"""
Examples
--------

─ Tool selection (read carefully) ─
• For ANY semantic question over secret descriptions, use `{search_secrets_fname}`. Never try to approximate meaning with brittle substring filters.
• Use `{filter_secrets_fname}` only for exact/boolean logic over structured fields (name, secret_id).
• Use `{list_secret_keys_fname}` to enumerate all stored secret names.

─ Discovering secrets ─
• List all secret keys
  `{list_secret_keys_fname}()`

• Search by description (semantic)
  `{search_secrets_fname}(references={{'description': 'API key for authentication'}}, k=5)`

─ Filtering (exact/boolean; not semantic) ─
• Exact name match
  `{filter_secrets_fname}(filter="name == 'openai_api_key'")`

• Filter by secret_id
  `{filter_secrets_fname}(filter="secret_id == 42")`

Anti‑patterns to avoid
---------------------
• NEVER reveal raw secret values in your response. Always reference secrets via ${{name}}.
• You MAY surface non-sensitive metadata such as the secret name and secret_id.
• When enumerating stored secrets, ALWAYS call `{list_secret_keys_fname}` (do not rely on memory).
• Avoid re-querying the same tools merely to reconfirm facts that a prior tool call has already established with clear, specific evidence; reuse earlier results and proceed.
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

    # Special security block
    security_block = "\n".join(
        [
            "Security (CRITICAL)",
            "-------------------",
            "• You MUST NEVER reveal raw secret values. Always reference secrets via ${name}.",
            "• You MAY surface non-sensitive metadata such as the secret name and secret_id.",
            "• When enumerating stored secrets, ALWAYS call the list tool (do not rely on memory).",
            "• All writes must keep raw values out of messages – only tool I/O may carry them internally.",
        ],
    )

    # Build using standardized composer
    spec = PromptSpec(
        manager="SecretManager",
        method="ask",
        tools=tools,
        role_line="You are an assistant specialising in **looking up secrets by name or description**.",
        global_directives=[
            "Work strictly through the tools provided.",
            "You MUST NEVER reveal raw secret values. Always reference secrets via ${name}.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
        ],
        include_read_only_guard=True,
        positioning_lines=[],
        counts_entity_plural=None,
        counts_value=None,
        columns_payload=None,
        columns_heading="columns",
        include_tools_block=True,
        usage_examples=usage_examples,
        clarification_examples_block=clarification_block or None,
        include_images_policy=False,  # SecretManager doesn't handle images
        include_images_forwarding=False,
        images_extras_block=None,
        include_parallelism=True,
        schemas=[],
        special_blocks=[security_block],
        include_clarification_footer=True,
        include_time_footer=True,
    )

    return compose_system_prompt(spec)


def build_update_prompt(*, tools: Dict[str, Callable]) -> PromptParts:
    """Return the system prompt used by SecretManager.update using the shared composer.

    Emphasises mutation rules and strict non-disclosure of raw values.
    """
    # Resolve canonical tool names dynamically
    ask_fname = _tool_name(tools, "ask")
    create_secret_fname = _tool_name(tools, "create_secret")
    update_secret_fname = _tool_name(tools, "update_secret")
    delete_secret_fname = _tool_name(tools, "delete_secret")
    bind_secret_fname = _tool_name(tools, "bind_secret")
    unbind_secret_fname = _tool_name(tools, "unbind_secret")
    set_hive_default_fname = _tool_name(tools, "set_hive_default")
    request_clar_fname = _tool_name(tools, "request_clarification")

    # Validate required tools
    _require_tools(
        {
            "ask": ask_fname,
            "create_secret": create_secret_fname,
            "update_secret": update_secret_fname,
            "delete_secret": delete_secret_fname,
        },
        tools,
    )

    # Build clarification block
    clarification_block = (
        textwrap.dedent(
            f"""
Clarification
-------------
• If any request is ambiguous, ask the user to disambiguate before changing data
  `{request_clar_fname}(question="There are several possible matches. Which secret did you mean?")`
            """,
        ).strip()
        if request_clar_fname
        else ""
    )

    # Usage examples
    usage_examples_base = f"""
Tool selection
--------------
• Use `{ask_fname}` strictly for read-only inspection of existing secrets (e.g., to check if a name exists).
• Use `{create_secret_fname}` to add a new secret; use `{update_secret_fname}` to modify existing; use `{delete_secret_fname}` to remove.

Ask vs Clarification
--------------------
• `{ask_fname}` is ONLY for inspecting/locating secrets that ALREADY EXIST (e.g., to find secret_id, verify names).
• Do NOT use `{ask_fname}` to ask the human for details about NEW secrets being created/changed in this update request.
• For human clarifications about prospective/new secrets (e.g., name spelling, missing values), call `{request_clar_fname}` when available.

Create / Update / Delete
------------------------
• Create a new secret
  `{create_secret_fname}(name='openai_api_key', value='sk-...', description='API key for OpenAI')`
• Update an existing secret's value
  `{update_secret_fname}(name='openai_api_key', value='sk-new-...', description='Updated API key')`
• Delete a secret
  `{delete_secret_fname}(name='old_api_key')`

Integration bindings (credential selection)
-------------------------------------------
• A **binding** tells this body which vault entry to use for a given integration
  (e.g. `"salesforce"` → the `"Salesforce Admin"` vault row). Without a binding,
  Hive bodies fall back to the Hive-wide default, and solo bodies fall back to
  a vault row whose `name` equals the integration.
• Create or replace this body's binding
  `{bind_secret_fname}(integration='salesforce', secret_name='Salesforce Admin')`
• Remove this body's binding
  `{unbind_secret_fname}(integration='salesforce')`
• Install a **Hive-wide default** (Hive members only — solo bodies cannot
  install defaults and should not attempt to)
  `{set_hive_default_fname}(integration='salesforce', secret_name='Salesforce Admin')`
• Prefer binding over creating duplicate vault rows: if two bodies need
  different credentials for the same integration, each body calls
  `{bind_secret_fname}` with the vault row it owns.

Naming When User Omits Key
--------------------------
• If the user does not give a secret name, derive a concise snake_case name from the request (e.g., 'mac_desktop_password').
• Canonical form: lowercase letters, digits and underscores only; must start with a letter; keep under 64 chars.
• Prefer nouns and context (platform, scope, purpose); avoid PII and user-identifying data.
• Check for collisions using `{ask_fname}`. If taken, append a short qualifier (e.g., '_staging', '_prod', or a minimal version suffix like '_v2').
• If an existing key appears to already represent the same concept (by name or search), REQUEST CLARIFICATION whether to update that existing key or create a new one.

Batching and Verification
-------------------------
• Handle requests that include MULTIPLE secrets comprehensively: create/update ALL specified secrets in this turn.
• After performing creations/updates/deletions, VERIFY results using `{ask_fname}` and reflect the confirmed outcomes in your message.
• Avoid claiming success unless verification tools confirm the new/updated keys exist (or were removed).

Anti‑patterns to avoid
---------------------
• NEVER echo raw secret values in responses. Only reference secrets via ${{name}}.
• Do not reference external stores like .env – Unify is the single source of truth.
• In messages, always reference secrets via ${{name}}. You MAY include non-sensitive metadata like secret_id; NEVER include raw values.
• Repeating the exact same tool call with the same arguments as a means to 'make sure it has completed' – just call `{ask_fname}` to verify.
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

    # Special security block
    security_block = "\n".join(
        [
            "Security (CRITICAL)",
            "-------------------",
            "• NEVER echo raw secret values in responses. Always reference via ${name}.",
            "• You MAY include non-sensitive metadata such as secret_id.",
            "• When a user provides a value, write it to Unify storage via the appropriate tool.",
            "• Do not reference external stores like .env – Unify is the single source of truth.",
        ],
    )

    # Compose using standardized composer
    spec = PromptSpec(
        manager="SecretManager",
        method="update",
        tools=tools,
        role_line="You are an assistant in charge of **creating, updating, or deleting secrets**.",
        global_directives=[
            "Choose tools based on the user's intent and the specificity of the target record.",
            "You MUST NEVER echo raw secret values in responses. Always reference via ${name}.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the request and choose the best approach yourself.",
            f"Important: `{ask_fname}` is read‑only and must only be used to locate/inspect secrets that already exist.",
        ],
        include_read_only_guard=False,
        positioning_lines=[],
        counts_entity_plural=None,
        counts_value=None,
        columns_payload=None,
        columns_heading="columns",
        include_tools_block=True,
        usage_examples=usage_examples,
        clarification_examples_block=clarification_block or None,
        include_images_policy=False,  # SecretManager doesn't handle images
        include_images_forwarding=False,
        images_extras_block=None,
        include_parallelism=True,
        schemas=[],
        special_blocks=[security_block],
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
    """Return an instruction prompt for the simulated SecretManager.

    Ensures the LLM replies **as if** the requested operation has already
    finished, avoiding responses like "I'll process that now".
    """
    import json  # local import
    from unity.common.context_dump import make_messages_safe_for_context_dump

    preamble = f"On this turn you are simulating the '{method}' method."
    if method.lower() == "ask":
        behaviour = (
            "Please answer the question about secrets with a plausible response. "
            "NEVER reveal raw secret values – always reference via ${name}. "
            "Do not ask for clarification or describe your process."
        )
    else:
        behaviour = (
            "Please act as though the secret operation has been fully completed. "
            "Respond in past tense summarising what was done. "
            "NEVER echo raw secret values – always reference via ${name}."
        )

    parts: list[str] = [preamble, behaviour, "", f"The user input is:\n{user_request}"]
    if parent_chat_context:
        parts.append(
            f"\nCalling chat context:\n{json.dumps(make_messages_safe_for_context_dump(parent_chat_context), indent=4)}",
        )

    return "\n".join(parts)
