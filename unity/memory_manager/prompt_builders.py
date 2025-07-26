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


# ---------------------------------------------------------------------------
# Helper to retrieve the assistant's full name (contact_id == 0) so prompts
# can mention it explicitly when instructing the model to switch to
# second-person language.  Implemented lazily to avoid heavy imports / cycles.
# ---------------------------------------------------------------------------


def _assistant_name() -> str:
    # 1) Prefer the globally initialised assistant record populated by
    #    `unity.init` (avoids unnecessary database look-ups and possible
    #    circular imports).
    try:
        from unity import ASSISTANT  # noqa: WPS433 – local import

        if ASSISTANT is not None:
            first = ASSISTANT.get("first_name") or ""
            last = ASSISTANT.get("surname") or ASSISTANT.get("last_name") or ""
            name = f"{first} {last}".strip()
            if name:
                return name
    except Exception:
        # Silent fall-through to backup strategy
        pass

    # 2) Fallback: query ContactManager (may hit a stub in offline tests)
    try:
        from unity.contact_manager.contact_manager import ContactManager  # noqa: WPS433

        cm = ContactManager()
        assist = cm._search_contacts(filter="contact_id == 0", limit=1)
        if assist:
            a = assist[0]
            name = " ".join(p for p in [a.first_name, a.surname] if p).strip()
            if name:
                return name
    except Exception:
        pass

    # 3) Last resort generic label
    return "the assistant"


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
        "3️⃣ **Merge duplicate contacts** when two records actually refer to the same person.  This often happens because we *automatically* create a new contact whenever a message arrives from an *unseen* phone number / email address etc.",
        '   • Example: You had a call with **David Smith** (contact_id 7).  A few minutes later an email comes from a new address and the system creates contact_id 12 with just that email column filled.  The email content says *"Hey, it\'s David – just got a new email address!"*.',
        "     → Detect that ids 7 and 12 are the same person and call `merge_contacts` like:",
        "       `merge_contacts(contact_id_1=7, contact_id_2=12, overrides={ 'email_address': 2, 'contact_id': 1 })`.",
        "     Remember every column needs a decision: supply an `overrides` map choosing whether the surviving value comes from contact **1** or contact **2**; any column omitted keeps the first non-None value when scanning 1 → 2.",
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
    contact_name: str,
    tools: Dict[str, Callable],
    *,
    guidance: Optional[str] = None,
) -> str:
    lines = [
        get_broader_context(),
        "",
    ]

    # Provide the model with an unambiguous identifier so it knows *who* it is updating

    # ── assistant reference rule ───────────────────────────────────────────
    # The assistant (contact_id == 0, typically named <first last>) must ALWAYS
    # be referred to in **second person** so that subsequent prompts shown to
    # the assistant remain unambiguous.
    assistant_full = _assistant_name()
    lines.append(
        f"You are updating the *bio* for contact **{contact_name}**.",
    )
    lines.append(
        f"This bio is being written exclusively for {assistant_full}! Therefore always address them using second-person pronouns – e.g. 'you', 'your' – rather than their name {assistant_full} or third-person forms.",
    )
    lines.append(
        f"In contrast, {contact_name} *must* always be referred to in the third-party ('{contact_name} works here' etc.), such that {assistant_full} clearly knows who the bio is about when reading it.",
    )

    lines += [
        "Input: the latest transcript chunk *plus* the current bio (if any).",
        "",
        "The bio is **concise freeform text (≤ 500 words)** describing relatively *time-invariant* information about the person: background, role, expertise, personality traits, important history, etc.",
        "Do **NOT** include fleeting topics, moment-to-moment tasks, or random facts that will quickly become irrelevant.",
        "🚫 **ABSOLUTELY NO HALLUCINATIONS:** Include **only** information that is explicitly stated in the provided transcript chunk. If a detail is not clearly present, you must *not* invent, infer, or elaborate on it.",
        "✅ The bio can be **very short** (even a single sentence) if limited information is available – there is no minimum length requirement, only the upper limit mentioned above.",
        "",
        "Update logic:",
        "1️⃣ First read the existing bio (if any) to understand what we already know about this contact.",
        "2️⃣ Read the transcript chunk and decide whether it contains new information that *belongs* in the bio.",
        "3️⃣ If the answer is yes, weave the new detail into the existing text, striving for a holistic overview that evolves gracefully over time (small, precise edits rather than wholesale rewrites).",
        "4️⃣ Use the specialised `set_bio` tool exactly once to persist the updated text.",
        "5️⃣ Finally, once you've (maybe) called the relevant update tool, then respond with your full **rationale** for the updates you did (or did not) make .",
        "",
        "Please do *not* perform the same action more than once. "
        "If you have already updated the bio via the `set_bio` tool, and it didn't result in any errors, "
        "then you do not need to do this again!"
        "🔒  If the transcript chunk contains a `manager_method` event from the ConversationManager indicating this exact operation is already in progress or completed, treat it as handled and **do not** perform it again.",
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
    contact_name: str,
    tools: Dict[str, Callable],
    *,
    guidance: Optional[str] = None,
) -> str:
    lines = [
        get_broader_context(),
        "",
    ]

    # Provide the model with an unambiguous identifier so it knows *who* it is updating

    # ── assistant reference rule ───────────────────────────────────────────
    # The assistant (contact_id == 0, typically named <first last>) must ALWAYS
    # be referred to in **second person** so that subsequent prompts shown to
    # the assistant remain unambiguous.
    assistant_full = _assistant_name()
    lines.append("")
    lines.append(
        f"You are updating the *rolling summary* for contact **{contact_name}**.",
    )
    lines.append(
        f"This summary is being written exclusively for {assistant_full}! Therefore always address them using second-person pronouns – e.g. 'you', 'your' – rather than their name {assistant_full} or third-person forms.",
    )
    lines.append(
        f"In contrast, {contact_name} *must* always be referred to in the third-party ('{contact_name} did this' etc.), such that {assistant_full} clearly knows who the summary is about when reading it.",
    )

    lines += [
        "Produce **concise holistic freeform text (≤ 500 words)** that weaves recent information into the existing summary instead of tacking items on as a list.",
        f"The summary must remain **highly specific to {contact_name}** (the person whose summary you are updating).",
        "🚫 **ABSOLUTELY NO HALLUCINATIONS:** Include **only** information that is explicitly stated in the provided transcript chunk. If a detail is not clearly present, you must *not* invent, infer, or elaborate on it.",
        "✅ The rolling summary can be **very short** (even a single sentence) if limited information is available – there is no minimum length requirement, only the upper limit mentioned above.",
        "• Mention other people or company-wide context ONLY when it directly affects or involves the target contact.",
        "• If the latest transcript chunk contains little or no new information about the target contact, keep the update extremely brief or leave the summary unchanged.",
        "• Do NOT pad the summary with generic project context, team member lists, or unrelated details.",
        "The summary should capture (when relevant):",
        "• main conversation theme(s) *as they relate to the target contact*",
        "• immediate goals / outstanding tasks for the target contact",
        "• tone or sentiment shifts involving the target contact",
        "• any direct interactions the target contact had or was mentioned in",
        "",
        "Balance *recency* with *importance*: trivial chit-chat from moments ago should not eclipse significant developments from earlier in the conversation (e.g. a job change announced yesterday).  Use judgement to keep the most relevant and durable points visible while still reflecting genuinely new events.",
        "",
        "Update logic:",
        "1️⃣ First read the existing rolling summary (if any) to understand what we already know about this contact's recent activity.",
        "1️⃣ Decide whether the transcript chunk introduces information that deserves to replace or adjust part of the existing summary.",
        "2️⃣ If yes, edit the text to integrate the change smoothly, preserving valuable prior context.",
        "3️⃣ Use `set_rolling_summary` exactly once to persist the new text.",
        "4️⃣ Finally, once you've (maybe) called the relevant update tool, then respond with your full **rationale** for the updates you did (or did not) make .",
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
        "🔬 **Scope of the knowledge store:**",
        "• ❌ Do **NOT** include any contact-specific or biographic details such as where someone works, their role or title, or what they have recently been doing – these belong in contact records, bios, or rolling summaries and can be assumed handled elsewhere.",
        "• ✅ Store only general, non-public facts that are relevant to the assistant's role and responsibilities as outlined in the broader context above and that may be useful in future.",
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
        "Return a short, human-readable summary of what you stored; if nothing was stored, then please breifly explain why.",
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
        "⚠️  **Important:** General descriptions of responsibilities or examples of typical duties (e.g. 'You'll be responsible for X') are **NOT** explicit task requests. Only create, update, or cancel tasks when the transcript includes a clear utterance whereby a concrete action should be taken, and sufficient details are given to act upon (such as 'Please do X when you get the chance' or 'Y was already completed by Z, so don't worry about it'). When in doubt, make no changes.",
        "🧠 **Not tasks:** Requests such as 'remember that…', 'store this information', or sharing credentials like passwords are **knowledge storage operations**, not actionable tasks. These must *never* generate or modify tasks in the schedule.",
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
        "Return a short, human-readable summary of what you changed; if nothing required updating, then please briefly explain why.",
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
