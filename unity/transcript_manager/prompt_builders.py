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
    num_messages: int,
    transcript_columns: Dict[str, str] | list[dict] | list[str],
    contact_columns: Dict[str, str] | list[dict] | list[str],
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

    # Strongly emphasize correct tool selection in a format consistent with ContactManager
    usage_examples_base = f"""
 Examples
 --------

 ─ Tool selection (read carefully) ─
 • For ANY semantic question over free‑form text (message content, free‑text custom columns), ALWAYS use `{search_messages_fname}`. Never try to approximate meaning with brittle substring filters.
 • Use `{filter_messages_fname}` only for exact/boolean logic over structured message fields (ids, mediums, equality checks) or for narrow, constrained text where substring checks make sense. Contact fields (sender profile) are NOT available in `{filter_messages_fname}`.

 ─ Semantic search: targeted references across columns (ranked by SUM of cosine distances) ─
 • Find top‑3 messages about budgeting and banking (signal in `content`)
   `{search_messages_fname}(references={{'content': 'banking and budgeting'}}, k=3)`

 • Combine message content with sender profile (contact‑side signal)
   `{search_messages_fname}(references={{'content': 'contract renewal', 'bio': 'procurement manager'}}, k=5)`

 • Use a derived expression for content when you need normalisation
   `expr = "str({{content}}).lower()"`
   `{search_messages_fname}(references={{expr: 'kickoff call summary'}}, k=5)`

 ─ Filtering (exact/boolean; not semantic) ─
 • Most recent WhatsApp from contact 7
   `{filter_messages_fname}(filter="sender_id == 7 and medium == 'whatsapp_message'", limit=1, offset=0)`
 • Last month’s emails (if datetime comparisons are supported by your backend)
   `{filter_messages_fname}(filter="medium == 'email' and timestamp >= '2024-01-01T00:00:00' and timestamp < '2024-02-01T00:00:00'", limit=100)`

 Anti‑patterns to avoid
 ---------------------
 • Avoid the default search behaviour of concatenating every column into one long string and comparing a single embedding of the whole question. Instead, pass multiple, focused reference texts keyed by their specific columns. The ranking minimises the sum of cosine distances and is more robust.
 • Avoid filtering for text‑heavy columns; substring matching is brittle. Prefer `{search_messages_fname}` for content‑based queries.
 • Do not attempt to reference contact fields (e.g., `bio`, `occupation`) inside `{filter_messages_fname}`; those fields live on the Contacts table. Use `{search_messages_fname}` to leverage sender contact fields.
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
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
            "Use the tools to gather missing context before asking the user for clarifications.",
            "",
            f"There are currently {num_messages} messages stored in the Transcripts table.",
            "Transcript columns:",
            json.dumps(transcript_columns, indent=4),
            "",
            "Sender contact columns (fields available on the Contacts table for the message sender):",
            json.dumps(contact_columns, indent=4),
            "",
            "Two-table reasoning:",
            "- Use semantic `{search_messages_fname}` when you need message content and/or sender contact attributes (e.g., `bio`, `first_name`). The tool will internally ensure embeddings and, when needed, join Transcripts with Contacts on `sender_id == contact_id` to rank results by the sum of per-term similarities.",
            "- Use exact `{filter_messages_fname}` only over transcript fields (ids, mediums, timestamps, content equality/contains). Contact fields are not in scope for filtering.",
            "",
            "Tools (name → argspec):",
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
