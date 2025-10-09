from __future__ import annotations

import json
from typing import Dict, Callable

from ..common.prompt_helpers import (
    sig_dict,
    clarification_guidance,
    now_utc_str,
)


def _now() -> str:
    return now_utc_str()


def build_ask_prompt(*, tools: Dict[str, Callable]) -> str:
    """Return the system prompt used by SecretManager.ask.

    Emphasises: never reveal raw secret values; reference via ${name};
    use provided tools to list/search/filter and, when requested, perform
    secret-related actions (create/update/delete). All storage is in Unify.
    """
    sig_json = json.dumps(sig_dict(tools), indent=4)

    lines: list[str] = []
    lines += [
        "Purpose",
        "-------",
        "- You are a SecretManager.ask tool.",
        "- You can look up secrets by name or description using the provided read-only tools.",
        "- You MUST NEVER reveal raw secret values. Always reference secrets via ${name}.",
        "",
        "Tools (name → argspec):",
        sig_json,
        "",
        "Answer Requirements",
        "-------------------",
        "- Provide concise answers. Never echo raw values.",
        "- When referring to a secret, use its placeholder, e.g. ${NAME}.",
        "- When enumerating stored secrets, ALWAYS call `_list_secret_keys` (do not rely on memory).",
        "- If you just observed new creations in this conversation, prefer listing via `_list_secret_keys` and, if needed, confirm specific names with `_filter_secrets`.",
        "- All writes must keep raw values out of messages – only tool I/O may carry them internally.",
    ]

    # Clarification guidance (only shown when request_clarification is present)
    lines += ["", clarification_guidance(tools)]

    lines += ["", f"Current UTC time is {_now()}."]
    return "\n".join(lines)


def build_update_prompt(*, tools: Dict[str, Callable]) -> str:
    """Return the system prompt used by SecretManager.update.

    Emphasises mutation rules and strict non-disclosure of raw values.
    """
    sig_json = json.dumps(sig_dict(tools), indent=4)

    lines: list[str] = []
    lines += [
        "Purpose",
        "-------",
        "- You are a SecretManager.update tool.",
        "- You can create, update, or delete secrets, and you MAY call read-only helpers (list/search/filter/columns) as needed.",
        "- NEVER echo raw secret values in responses. Always reference via ${name}.",
        "",
        "General Rules",
        "-------------",
        "- When a user provides a value, write it to Unify storage via the appropriate tool.",
        "- Handle requests that include MULTIPLE secrets comprehensively: create/update ALL specified secrets in this turn.",
        "- After performing creations/updates/deletions, VERIFY results using `_list_secret_keys` and/or `_filter_secrets` and reflect the confirmed outcomes in your message.",
        "- Avoid claiming success unless verification tools confirm the new/updated keys exist (or were removed).",
        "- Do not reference external stores like .env – Unify is the single source of truth.",
        "- In messages, always reference secrets via ${name}.",
        "",
        "Naming When User Omits Key",
        "--------------------------",
        "- If the user does not give a secret name, derive a concise snake_case name from the request (e.g., 'mac_desktop_password').",
        "- Canonical form: lowercase letters, digits and underscores only; must start with a letter; keep under 64 chars.",
        "- Prefer nouns and context (platform, scope, purpose); avoid PII and user-identifying data.",
        "- Check for collisions using ask tools. If taken, append a short qualifier (e.g., '_staging', '_prod', or a minimal version suffix like '_v2').",
        "- If an existing key appears to already represent the same concept (by name or search), REQUEST CLARIFICATION whether to update that existing key or create a new one (e.g., with a qualifier/suffix).",
        "- If genuinely ambiguous, call request_clarification with 2–3 suggested names and proceed with the user's choice.",
        "- Once a name is decided, call _create_secret(name=..., value=..., description=?) with concise summary as description. Do not echo the raw value.",
        "- When the user chooses to update an existing key instead, call _update_secret(name=..., value=..., description=?), never echo the raw value.",
        "",
        "Batching and Tool Usage",
        "------------------------",
        "- You may call several tools in sequence to fulfil the user's request (e.g., multiple `_create_secret` calls).",
        "- Prefer a small number of purposeful calls over one per token; combine checks (list/search) thoughtfully to avoid omissions.",
        "",
        "Tools (name → argspec):",
        sig_json,
    ]

    # Clarification guidance (only shown when request_clarification is present)
    lines += ["", clarification_guidance(tools)]

    lines += ["", f"Current UTC time is {_now()}."]
    return "\n".join(lines)
