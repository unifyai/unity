# memory_manager/prompt_builders.py
from __future__ import annotations

import json
import inspect
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from .rolling_activity import get_rolling_activity


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
        "Your task is to create or amend contact records — names, phone numbers, emails, bios, etc. — whenever the **current transcript chunk reveals new or changed facts**.",
        "",
        'The transcript will rarely contain an explicit instruction such as *"please update the address book"*.  Instead you must listen for *any* statement that implies new contact information.  Examples include:',
        "• A participant casually mentioning a new phone number or email address.",
        "• Someone referring to a person we have never seen before, even without any contact details.",
        "",
        "When you detect such information, you should:",
        "1️⃣ Create a **new** contact entry if it does not yet exist, even if all you have is a first name plus a short descriptive bio.",
        "2️⃣ Amend the **existing** contact if we already have a record but the information has changed or been extended.",
        "",
        "Work **only** via the tools given.  First figure out what changed (if anything), then call the appropriate update tool(s).",
        "Finally return a short human-readable summary of what you did.",
        "Please do *not* perform the same action more than once. "
        "If you have updated/added a contact already via the `ContactManager` update method, "
        "then you do not need to do this again!"
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
        "Input is a chunk of the most recent messages plus the *existing* bio (if any).",
        "",
        "1️⃣ Decide whether the bio should change.",
        "2️⃣ If yes, call the specialised `set_bio` tool.",
        "3️⃣ Return only the *new* bio text (or the unchanged one).",
        "",
        "Please do *not* perform the same action more than once. "
        "If you have already updated the bio via the `set_bio` tool, "
        "then you do not need to do this again!"
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
        "You are the **MemoryManager** refreshing the *rolling summary*",
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
        "Please do *not* perform the same action more than once. "
        "If you have already updated the rolling summary via the `set_rolling_summary` tool, "
        "then you do not need to do this again!"
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
        "knowledge from the latest transcript chunk.",
        "",
        "• Identify *reusable* facts about people, projects, company",
        "  processes, requirements, or domain knowledge.",
        "• Before writing, call `KnowledgeManager.refactor` to check if an",
        "  existing card should be merged / extended instead of creating",
        "  duplication.",
        "• Finally, persist using `KnowledgeManager.update` (or skip if",
        "  nothing valuable was found).",
        "",
        "Please do *not* perform the same action more than once. "
        "If you have already persisted this knowledge via the `KnowledgeManager.update` method "
        "(or refactored via `KnowledgeManager.refactor`), then you do not need to do this again!"
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


def build_task_prompt(
    tools: Dict[str, Callable],
    guidance: Optional[str] = None,
) -> str:
    lines = [
        _rolling_activity_section(),
        "You are the **MemoryManager** tasked with updating the task list based on the latest transcript chunk.",
        "",
        "• Identify tasks that should be created, modified, cancelled or reordered.",
        "• Always begin by calling `TaskScheduler.ask` to inspect the current list.",
        "• Apply the required changes using `TaskScheduler.update`.",
        "",
        "Please do *not* perform the same action more than once. If you have already manipulated the task list via the `TaskScheduler.update` method, then you do not need to do this again!",
        "",
        "Return a short, human-readable summary of what you stored; if nothing was stored, just return 'no-op'.",
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
    """Return a markdown summary of the agent's historic activity.

    Reads the **process-wide** cached snapshot instead of querying the backend
    via `MemoryManager().get_rolling_activity()` on every call.  Callers can
    still rely on the helper to return an *empty string* when nothing useful
    has been recorded yet.
    """

    try:
        overview = get_rolling_activity()
    except Exception:  # pragma: no cover – defensive guard
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
            "---------------------------",
            "",
        ],
    )
