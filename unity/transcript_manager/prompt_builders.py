"""Prompt builders for TranscriptManager.

These builders parallel *contact_manager/prompt_builders.py*:
they receive a **live** ``tools``-dict and construct the
corresponding **system** messages *without ever hard-coding* tool
counts, names or arg-signatures.  Each prompt also contains an
explicit "Examples" placeholder to make it easy to append
illustrative calls at runtime if desired.
"""

from __future__ import annotations

import inspect
import json
import textwrap
from datetime import datetime, timezone
from typing import Callable, Dict

# Schemas used in the prompt -------------------------------------------------
from ..contact_manager.types.contact import Contact
from .types.message import Message
from ..memory_manager.broader_context import get_broader_context
from ..common.prompt_helpers import clarification_guidance

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} for the *Tools* section."""

    return {n: str(inspect.signature(fn)) for n, fn in tools.items()}


def _now() -> str:
    """UTC timestamp helper for prompt reproducibility."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ─────────────────────────────────────────────────────────────────────────────
# Shared historic activity snippet
# ─────────────────────────────────────────────────────────────────────────────


def _rolling_activity_section() -> str:
    """Return a human-readable summary of historic agent activity using cache."""

    try:
        overview = get_broader_context()
    except Exception:  # pragma: no cover – safe fallback
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

    # Heuristically infer canonical names — fall back to placeholders if absent
    summarise_name = next(
        (n for n in tools if "summarize" in n.lower()),
        "summarize",
    )
    search_contacts_name = next(
        (n for n in tools if "search" in n.lower() and "contact" in n.lower()),
        "search_contacts",
    )
    search_messages_name = next(
        (n for n in tools if "search" in n.lower() and "message" in n.lower()),
        "search_messages",
    )
    search_summaries_name = next(
        (n for n in tools if "search" in n.lower() and "summary" in n.lower()),
        "search_summaries",
    )
    nearest_messages_name = next(
        (n for n in tools if "nearest" in n.lower()),
        "nearest_messages",
    )
    clar_name = next(
        (n for n in tools if "clarification" in n.lower()),
        "request_clarification",
    )

    usage_examples = textwrap.dedent(
        f"""
        Examples
        --------
        • **Semantic search** – top-3 messages about *banking and budgeting*
          `{nearest_messages_name}(text="banking and budgeting", k=3)`

        • **Ask for clarification** when the user's request is underspecified
          `{clar_name}(question="Which conversation are you referring to?")`

        • **Filter search** – most recent WhatsApp from *contact 7*
          `{search_messages_name}(filter="contact_id == 7 and medium == 'whatsapp_message'", limit=1, offset=0)`

        Important: if the question, refers to message *content* (topic etc.) rather than meta-data (datetime, medium etc.) then you should *almost always* use {nearest_messages_name} before trying exact string matching via {search_messages_name}. You're much more likely to get a match on your first attempt.
    """,
    ).strip()

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    return "\n".join(
        [
            activity_block,
            "You are an assistant specialised in **querying and analysing communication transcripts**.",
            "Work **exclusively** through the tools listed below to gather data",
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
