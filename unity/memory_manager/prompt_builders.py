# memory_manager/prompt_builders.py
from __future__ import annotations

import json
import inspect
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from .broader_context import get_broader_context


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
        get_broader_context(),
        "",
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
        "🔒  If the transcript chunk contains a `manager_method` event from the ConversationManager indicating this operation is already in progress or completed, treat it as handled and **do not** perform it again.",
        "",
        "Tools (name → argspec):",
        json.dumps(_sig_dict(tools), indent=4),
        "",
        "Read through the broader context of your role and recent activity for orientation, especially in cases where you're not sure whether a new person should actually be treated as a contact.",
        "",
        "Current UTC time: " + _now(),
    ]
    return _with_guidance(lines, guidance)


def build_bio_prompt(
    tools: Dict[str, Callable],
    guidance: Optional[str] = None,
) -> str:
    lines = [
        get_broader_context(),
        "",
        "You are responsible for the *bio* column for ONE contact.",
        "Input: the latest transcript chunk *plus* the current bio (if any).",
        "",
        "The bio is **concise freeform text (≤ 500 words)** describing relatively *time-invariant* information about the person: background, role, expertise, personality traits, important history, etc.",
        "Do **NOT** include fleeting topics, moment-to-moment tasks, or random facts that will quickly become irrelevant.",
        "",
        "Update logic:",
        "1️⃣ Read the transcript chunk and decide whether it contains new information that *belongs* in the bio.",
        "2️⃣ If the answer is yes, weave the new detail into the existing text, striving for a holistic overview that evolves gracefully over time (small, precise edits rather than wholesale rewrites).",
        "3️⃣ Use the specialised `set_bio` tool exactly once to persist the updated text.",
        "4️⃣ Return **only** the text that was stored (or the unchanged one).",
        "",
        "Please do *not* perform the same action more than once. "
        "If you have already updated the bio via the `set_bio` tool, "
        "then you do not need to do this again!"
        "🔒  If the transcript chunk contains a `manager_method` event from the ConversationManager indicating this operation is already in progress or completed, treat it as handled and **do not** perform it again.",
        "",
        "Tools (name → argspec):",
        json.dumps(_sig_dict(tools), indent=4),
        "",
        "Read through the broader context of your role and recent activity for orientation, especially in cases where you're not sure what should be updated in the bio (if anything).",
        "",
        "Current UTC time: " + _now(),
    ]
    return _with_guidance(lines, guidance)


def build_rolling_prompt(
    tools: Dict[str, Callable],
    guidance: Optional[str] = None,
) -> str:
    lines = [
        get_broader_context(),
        "",
        "You are refreshing the *rolling summary*",
        "for ONE contact.  Start from the previous rolling summary (if supplied).",
        "",
        "Produce **concise holistic freeform text (≤ 500 words)** that weaves recent information into the existing summary instead of tacking items on as a list.",
        "The summary should capture:",
        "• main conversation theme(s)",
        "• immediate goals / outstanding tasks",
        "• tone or sentiment shifts if relevant",
        "",
        "Balance *recency* with *importance*: trivial chit-chat from moments ago should not eclipse significant developments from earlier in the conversation (e.g. a job change announced yesterday).  Use judgement to keep the most relevant and durable points visible while still reflecting genuinely new events.",
        "",
        "Update logic:",
        "1️⃣ Decide whether the transcript chunk introduces information that deserves to replace or adjust part of the existing summary.",
        "2️⃣ If yes, edit the text to integrate the change smoothly, preserving valuable prior context.",
        "3️⃣ Use `set_rolling_summary` exactly once to persist the new text.",
        "4️⃣ Finally, return the text you stored.",
        "",
        "Please do *not* perform the same action more than once. "
        "If you have already updated the rolling summary via the `set_rolling_summary` tool, "
        "then you do not need to do this again!"
        "🔒  If the transcript chunk contains a `manager_method` event from the ConversationManager indicating this operation is already in progress or completed, treat it as handled and **do not** perform it again.",
        "",
        "Tools (name → argspec):",
        json.dumps(_sig_dict(tools), indent=4),
        "",
        "Read through the broader context of your role and recent activity for orientation, especially in cases where you're not sure what should be updated in the summary (if anything).",
        "",
        "Current UTC time: " + _now(),
    ]
    return _with_guidance(lines, guidance)


def build_knowledge_prompt(
    tools: Dict[str, Callable],
    guidance: Optional[str] = None,
) -> str:
    lines = [
        get_broader_context(),
        "",
        "You are tasked with mining *long-term* knowledge from the latest transcript chunk.",
        "",
        "🧭 **General process:**",
        "1️⃣ Reflect on the broader context of your role and recent activity above and decide which kinds of facts would be *truly valuable* to retain long-term.",
        "2️⃣ Read the transcript chunk and pick out any pieces of information that fit those criteria.  It is acceptable if **none** are found.",
        "3️⃣ For *each* candidate fact:",
        "   • Call `KnowledgeManager.ask` to check whether this fact (or an equivalent) already exists in the knowledge store.",
        "   • If it **does exist**, skip to the next fact.",
        "   • If storing the new fact would be awkward or duplicative because of the current table/column layout, call `KnowledgeManager.refactor` **once** with clear instructions for restructuring to achieve cleaner, less-redundant storage for the *pre-existing data*.",
        "   • Finally, add the new fact with `KnowledgeManager.update`.",
        "",
        "🚫 **Avoid redundant actions:** If you have already asked, refactored, or updated during this turn you do **NOT** need to repeat the same tool call.",
        "🔒  If the transcript chunk contains a `manager_method` event from the ConversationManager indicating this operation is already in progress or completed, treat it as handled and **do not** perform it again.",
        "",
        "Return a short, human-readable summary of what you stored; if nothing was stored, just return 'no-op'.",
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
        get_broader_context(),
        "",
        "You are responsible for maintaining the *task schedule* in light of the **latest transcript chunk**.",
        "",
        "🧭 **General process:**",
        "1️⃣ Reflect on the broader context of your role and recent activity above and decide whether the conversation requests or implies new tasks or any changes to the existing tasks.",
        "2️⃣ Always begin by calling `TaskScheduler.ask` to retrieve the **current** task list.",
        "3️⃣ For each required change:",
        "   • Create a **new** task if it does not yet exist.",
        "   • Update the **existing** task if details (status, priority, due date, etc.) have changed.",
        "   • Cancel a task that is no longer relevant.",
        "   • Re-order tasks or adjust priorities where it improves clarity.",
        "   • Perform these adjustments via **a single** `TaskScheduler.update` call whenever possible.",
        "",
        "🚫 **Avoid redundant actions:** If you have already inspected or updated the task list during this turn you do **NOT** need to repeat the same tool call.",
        "🔒  If the transcript chunk contains a `manager_method` event from the ConversationManager indicating this operation is already in progress or completed, treat it as handled and **do not** perform it again.",
        "",
        "Return a short, human-readable summary of what you changed; if nothing required updating, just return 'no-op'.",
        "",
        "Tools (name → argspec):",
        json.dumps(_sig_dict(tools), indent=4),
        "",
        "Current UTC time: " + _now(),
    ]
    return _with_guidance(lines, guidance)


def build_activity_events_summary_prompt(
    guidance: Optional[str] = None,
) -> str:
    """Return a detailed system prompt for summarising a JSON array of events.

    The prompt is used by MemoryManager when it needs to collapse many raw
    ManagerMethod events or lower-level summaries into **one concise English
    paragraph**.  The summary should capture the essence of the activity while
    staying below ~50 words so it can be embedded in larger prompts without
    noise.
    """

    lines: list[str] = [
        get_broader_context(),
        "",
        "You will receive **only** a JSON array – each element representing an "
        "event or pre-computed summary of recent manager activity.",
        "",
        "Your goal is to distil this array into **one plain-English paragraph** "
        "(≈ 50 words, never exceeding 60) that communicates:",
        "• What happened (high-level actions, state changes, notable decisions)",
        "• Who/what was involved (manager names, key entities) – keep it brief",
        "• Any important outcomes or follow-ups",
        "",
        "⚠️  Do **not** quote the JSON or enumerate every single entry.  Merge "
        "related events, remove noise, and write in the third person.  Omit "
        "trivia.  Return an **empty string** if the input conveys nothing of "
        "lasting importance.",
        "",
        "Format requirements:",
        "• Single paragraph, no bullet points, no markdown headers",
        "• Do not mention these instructions or the word *JSON*",
        "",
        "Read through the broader context of your role and recent activity for orientation, especially in cases where you're not sure which aspects should be emphasized in the summary.",
        "",
        "Current UTC time: " + _now(),
    ]

    return _with_guidance(lines, guidance)
