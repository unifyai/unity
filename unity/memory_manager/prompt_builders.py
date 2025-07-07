# memory_manager/prompt_builders.py
from __future__ import annotations

import json
import inspect
from datetime import datetime, timezone
from typing import Callable, Dict, Optional


# ── utils ───────────────────────────────────────────────────────────────
def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    return {n: str(inspect.signature(fn)) for n, fn in tools.items()}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ── three tiny builders (one per public method) ─────────────────────────
def _with_guidance(lines: list[str], guidance: Optional[str]) -> str:
    """
    Helper: append caller-supplied guidance, if any, to the block.
    """
    if guidance:
        lines.extend(
            [
                "",
                "🔖 **Caller guidance – prioritise:**",
                guidance,
            ],
        )
    return "\n".join(lines)


def build_contact_update_prompt(
    tools: Dict[str, Callable],
    guidance: Optional[str] = None,
) -> str:
    lines = [
        _rolling_activity_section(),
        "You are the **offline MemoryManager** tasked with *creating or amending*",
        "contact records — names, phone numbers, emails, etc. — after reading",
        "a 50-message transcript chunk.",
        "",
        "Work **only** via the tools below.  First figure out what changed,",
        "then call the appropriate update tool(s).  Finally return a short",
        "human-readable summary of what you did.",
        "",
        "Tools (name → argspec):",
        json.dumps(_sig_dict(tools), indent=4),
        "",
        "Current UTC time: " + _now(),
    ]
    return _with_guidance(lines, guidance)


def build_bio_prompt(
    tools: Dict[str, Callable],
    guidance: Optional[str] = None,
) -> str:
    lines = [
        _rolling_activity_section(),
        "You are the **MemoryManager** updating the *bio* column for ONE contact.",
        "Input is the last 50 messages plus the *existing* bio (if any).",
        "",
        "1️⃣ Decide whether the bio should change.",
        "2️⃣ If yes, call the specialised `set_bio` tool.",
        "3️⃣ Return only the *new* bio text (or the unchanged one).",
        "",
        "Tools (name → argspec):",
        json.dumps(_sig_dict(tools), indent=4),
        "",
        "Current UTC time: " + _now(),
    ]
    return _with_guidance(lines, guidance)


def build_rolling_prompt(
    tools: Dict[str, Callable],
    guidance: Optional[str] = None,
) -> str:
    lines = [
        _rolling_activity_section(),
        "You are the **MemoryManager** refreshing the 50-message *rolling summary*",
        "for ONE contact.  Start from the previous rolling summary (if supplied).",
        "",
        "Produce a concise, up-to-date summary **<= 120 words** capturing:",
        "• main conversation theme(s)",
        "• immediate goals / outstanding tasks",
        "• tone or sentiment shifts if relevant",
        "",
        "You may call the specialised `set_rolling_summary` tool exactly once.",
        "Finally, return the summary text you stored.",
        "",
        "Tools (name → argspec):",
        json.dumps(_sig_dict(tools), indent=4),
        "",
        "Current UTC time: " + _now(),
    ]
    return _with_guidance(lines, guidance)


def build_knowledge_prompt(
    tools: Dict[str, Callable],
    guidance: Optional[str] = None,
) -> str:
    lines = [
        _rolling_activity_section(),
        "You are the **MemoryManager** tasked with mining *long-term*",
        "knowledge from the latest 50-message transcript chunk.",
        "",
        "• Identify *reusable* facts about people, projects, company",
        "  processes, requirements, or domain knowledge.",
        "• Before writing, call `KnowledgeManager.refactor` to check if an",
        "  existing card should be merged / extended instead of creating",
        "  duplication.",
        "• Finally, persist using `KnowledgeManager.update` (or skip if",
        "  nothing valuable was found).",
        "",
        "Return a short, human-readable summary of what you stored; if",
        "nothing was stored, just return 'no-op'.",
        "",
        "Tools (name → argspec):",
        json.dumps(_sig_dict(tools), indent=4),
        "",
        "Current UTC time: " + _now(),
    ]
    return _with_guidance(lines, guidance)


# ─────────────────────────────────────────────────────────────────────────────
# Shared historic activity snippet (uses *lazy* import to avoid cycles)
# ─────────────────────────────────────────────────────────────────────────────


def _rolling_activity_section() -> str:
    """Return a markdown summary of the agent's historic activity."""

    try:
        # Lazy import to avoid circular dependency issues
        from .memory_manager import MemoryManager  # noqa: WPS433

        overview = MemoryManager().get_rolling_activity()
    except Exception:  # pragma: no cover
        return ""

    # Skip the entire section when there's nothing meaningful to report.
    if not overview:
        return ""

    # Compose the section with a *closing* dashed line to clearly separate it
    # from any subsequent system-message content.
    return "\n".join(
        [
            "Historic Activity Overview",
            "---------------------------",
            "Below is a summary of the agent's historic activity (tasks, contacts, knowledge, transcripts, etc.).",
            "Some parts may be useful context for the current task while others might not – use your judgement.",
            "",
            overview,
            "---------------------------",
            "",
        ],
    )
