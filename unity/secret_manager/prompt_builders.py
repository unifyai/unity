from __future__ import annotations

import json
from typing import Dict, Callable

from ..common.prompt_helpers import (
    sig_dict,
    clarification_guidance,
)


def build_ask_prompt(*, tools: Dict[str, Callable]) -> str:
    """Return the system prompt used by SecretManager.ask.

    Emphasises: never reveal raw secret values; reference via ${name};
    use provided tools to list/search/filter.
    """
    sig_json = json.dumps(sig_dict(tools), indent=4)

    lines: list[str] = []
    lines += [
        "Purpose",
        "-------",
        "- You are a SecretManager.ask tool.",
        "- You can look up secrets by name or description using the provided tools.",
        "- You MUST NEVER reveal raw secret values. Always reference secrets via ${name}.",
        "",
        "Tools (name → argspec):",
        sig_json,
        "",
        "Answer Requirements",
        "-------------------",
        "- Provide concise answers. Never echo raw values.",
        "- When referring to a secret, use its placeholder, e.g. ${NAME}.",
    ]

    # Clarification guidance (only shown when request_clarification is present)
    lines += ["", clarification_guidance(tools)]

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
        "- You can create, update, or delete secrets using the provided tools.",
        "- NEVER echo raw secret values in responses. Always reference via ${name}.",
        "",
        "General Rules",
        "-------------",
        "- When a user provides a value, write it to storage.",
        "- Persist to .env with KEY derived from the name (upper snake).",
        "- In messages, always reference secrets via ${name}.",
        "",
        "Tools (name → argspec):",
        sig_json,
    ]

    # Clarification guidance (only shown when request_clarification is present)
    lines += ["", clarification_guidance(tools)]

    return "\n".join(lines)
