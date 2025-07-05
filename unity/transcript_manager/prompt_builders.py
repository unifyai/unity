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
from typing import Callable, Dict, Optional

# Schemas used in the prompt -------------------------------------------------
from ..contact_manager.types.contact import Contact
from .types.message import Message
from .types.message_exchange_summary import MessageExchangeSummary

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
    """Return a human-readable summary of historic agent activity."""

    try:
        # Local import to avoid circular deps at import time
        from ..memory_manager.memory_manager import MemoryManager  # noqa: WPS433

        overview = MemoryManager().get_rolling_activity()
    except Exception:  # pragma: no cover – keep prompts robust
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


def build_ask_prompt(tools: Dict[str, Callable]) -> str:  # noqa: C901 – long, but flat
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

        • **Summarise** two exchanges (23 & 24) before answering
          `{summarise_name}(from_exchanges=[23, 24])`

        Important: if the question, refers to message *content* (topic etc.) rather than meta-data (datetime, medium etc.) then you should *almost always* use {nearest_messages_name} before trying exact string matching via {search_messages_name}. You're much more likely to get a match on your first attempt.
    """,
    ).strip()

    return "\n".join(
        [
            _rolling_activity_section(),
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
            f"Summary  = {json.dumps(MessageExchangeSummary.model_json_schema(), indent=4)}",
            "",
            f"Current UTC time: {_now()}.",
        ],
    )


def build_summarize_prompt(guidance: Optional[str] = None) -> str:
    """
    Build the system-prompt for :pyfunc:`TranscriptManager.summarize`.

    No tool names are hard-coded; the prompt simply instructs the model
    to produce a cross-exchange summary and reminds it to ask for
    clarification if needed.  A *guidance* string can be injected.
    """

    guidance_block = (
        f"\n\nAdditional guidance provided by the caller:\n{guidance}"
        if guidance
        else ""
    )

    return "\n".join(
        [
            _rolling_activity_section(),
            "You will receive one or more message exchanges.",
            "Craft a concise summary that captures the most important points",
            "**across** all exchanges. If anything is unclear in the guidance provided,",
            "then use the `request_clarification` tool – do **not** hallucinate.",
            "",
            "However, if the guidance is already clear, then please try to 'read between the lines'",
            "and do not use `request_clarification` unless something is genuinely contradictory,",
            "unclear, or there is missing information.",
            "",
            f"Current UTC time: {_now()}.",
            guidance_block,
        ],
    )
