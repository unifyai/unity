"""Prompt builders for TranscriptManager.

These builders parallel *contact_manager/prompt_builders.py*:
they receive a **live** ``tools``-dict and construct the
corresponding **system** messages *without ever hard-coding* tool
counts, names or arg-signatures.  Each prompt also contains an
explicit "Examples" placeholder to make it easy to append
illustrative calls at runtime if desired.
"""

from __future__ import annotations

import json
import textwrap
from typing import Callable, Dict

# Schemas used in the prompt -------------------------------------------------
from ..contact_manager.types.contact import Contact
from .types.message import Message
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


def _now() -> str:
    """UTC timestamp helper for prompt reproducibility."""
    return now_utc_str()


def _tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """Delegate to shared tool name resolver."""
    return _shared_tool_name(tools, needle)


def _require_tools(pairs: Dict[str, str | None], tools: Dict[str, Callable]) -> None:
    """Delegate validation to shared helper for consistent errors."""
    _shared_require_tools(pairs, tools)


# ─────────────────────────────────────────────────────────────────────────────
# Shared historic activity snippet
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def build_ask_prompt(
    tools: Dict[str, Callable],
    *,
    include_activity: bool = True,
) -> str:  # noqa: C901 – long, but flat
    """
    Build the system-prompt for :pyfunc:`TranscriptManager.ask`.

    The generated prompt:
      • lists the *actual* tools and their arg-specs,
      • embeds the three Pydantic schemas the model needs,
      • shows a handful of **dynamic** usage examples whose function
        names always reflect the *current* toolkit,
      • contains a placeholder block ready for additional examples.
    """

    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Resolve canonical tool names dynamically
    filter_messages_fname = _tool_name(tools, "filter_messages")
    search_messages_fname = _tool_name(tools, "search_messages")
    request_clar_fname = _tool_name(tools, "request_clarification")

    # Validate required tools (clarification is optional)
    _require_tools(
        {
            "filter_messages": filter_messages_fname,
            "search_messages": search_messages_fname,
        },
        tools,
    )

    clarification_block = (
        textwrap.dedent(
            f"""
            • Ask for clarification when the user's request is underspecified
              `{request_clar_fname}(question="Which conversation are you referring to?")`
            """,
        ).strip()
        if request_clar_fname
        else ""
    )

    usage_examples_base = f"""
        Examples
        --------
        • **Semantic search** – top-3 messages about *banking and budgeting*
          `{search_messages_fname}(text="banking and budgeting", k=3)`

        • **Filter search** – most recent WhatsApp from *contact 7*
          `{filter_messages_fname}(filter="contact_id == 7 and medium == 'whatsapp_message'", limit=1, offset=0)`

        Important: if the question refers to message *content* (topic etc.) rather than meta-data (datetime, medium etc.) then you should almost always use {search_messages_fname} before trying exact string matching via {filter_messages_fname}. You're much more likely to get a match on your first attempt.
    """
    usage_examples = textwrap.dedent(usage_examples_base).strip()
    if clarification_block:
        usage_examples = f"{usage_examples}\n{clarification_block}"

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    return "\n".join(
        [
            activity_block,
            "You are an assistant specialised in **querying and analysing communication transcripts**.",
            "Work **exclusively** through the tools listed below to gather data",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best method yourself.",
            "before composing your final answer.",
            "",
            "Tools (name → argspec)",
            "----------------------",
            sig_json,
            "",
            usage_examples,
            "",
            "Schemas",
            "-------",
            f"Contact  = {json.dumps(Contact.model_json_schema(), indent=4)}",
            "",
            f"Message  = {json.dumps(Message.model_json_schema(), indent=4)}",
            "",
            f"Current UTC time: {_now()}.",
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
    """Return an instruction prompt for the *simulated* TranscriptManager.

    Ensures the LLM replies **as if** the requested operation has already
    finished, avoiding responses like "I'll process that now".
    """
    import json  # local import

    preamble = f"On this turn you are simulating the '{method}' method."
    if method.lower() == "ask":
        behaviour = (
            "Please always *answer* the question (inventing a plausible response) – "
            "do **not** ask for clarification or explain your steps."
        )
    elif method.lower() == "summarize":
        behaviour = (
            "Please always provide an **imaginary summary** that looks realistic. "
            "Do not answer in future tense and do not describe how you will summarise."
        )
    else:
        behaviour = (
            "Please act as though the request has been fully satisfied. "
            "Respond in past tense with the final outcome, not the process."
        )

    parts: list[str] = [preamble, behaviour, "", f"The user input is:\n{user_request}"]
    if parent_chat_context:
        parts.append(
            f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}",
        )

    return "\n".join(parts)
