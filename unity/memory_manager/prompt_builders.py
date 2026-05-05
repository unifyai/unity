# memory_manager/prompt_builders.py
from __future__ import annotations

import json
import inspect
from typing import Callable, Dict, Optional

from .broader_context import get_broader_context
from ..common.prompt_helpers import now


# -- utils --------------------------------------------------------------------
def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    return {n: str(inspect.signature(fn)) for n, fn in tools.items()}


def _assistant_name() -> str:
    from unity.session_details import SESSION_DETAILS  # noqa: WPS433

    if SESSION_DETAILS.assistant.name:
        return SESSION_DETAILS.assistant.name

    try:
        from unity.manager_registry import ManagerRegistry  # noqa: WPS433

        cm = ManagerRegistry.get_contact_manager()
        assist = cm.filter_contacts(
            filter=f"contact_id == {SESSION_DETAILS.self_contact_id}",
            limit=1,
        )
        if assist:
            a = assist[0]
            name = " ".join(p for p in [a.first_name, a.surname] if p).strip()
            if name:
                return name
    except Exception:
        pass

    return "the assistant"


def _with_guidance(lines: list[str], guidance: Optional[str]) -> str:
    if guidance:
        lines.extend(
            [
                "",
                "\U0001f516 **Caller guidance \u2013 prioritise:**",
                guidance,
            ],
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual builders (used by the standalone public methods)
# ---------------------------------------------------------------------------


def build_contact_update_prompt(
    tools: Dict[str, Callable],
    guidance: Optional[str] = None,
) -> str:
    lines = [
        get_broader_context(),
        "",
        "Your task is to create or amend contact records \u2014 names, phone numbers, emails, bios, rolling summaries, response policies, etc. \u2014 whenever the **current transcript chunk reveals new or changed facts**.",
        "",
        'The transcript will rarely contain an explicit instruction such as *"please update the address book"*.  Instead you must listen for *any* statement that implies new contact information.  Examples include:',
        "\u2022 A participant casually mentioning a new phone number or email address.",
        "\u2022 Someone referring to a person we have never seen before, even without any contact details.",
        "",
        "When you detect such information, you should:",
        "1\ufe0f\u20e3 Create a **new** contact entry if it does not yet exist, even if all you have is a first name plus a short descriptive bio.",
        "2\ufe0f\u20e3 Amend the **existing** contact if we already have a record but the information has changed or been extended.",
        "3\ufe0f\u20e3 **Merge duplicate contacts** when two records actually refer to the same person.",
        "",
        "\u26a0\ufe0f  **Nameless contacts \u2014 organisation / service contacts:**",
        "Some contacts have no `first_name` or `surname` because they represent an organisation or service rather than a specific person.  The `bio` will describe the entity.  When you encounter such a contact:",
        "\u2022 Do **not** populate `first_name` / `surname` with the name of whoever happens to answer or sign off \u2014 they are a transient representative, not the contact\u2019s identity.",
        "\u2022 You may still update other fields (email, phone, bio details) if the transcript reveals useful information about the organisation or service.",
        "Conversely, if a contact has no name simply because the person\u2019s name is not yet known, **do** populate the name as soon as the transcript reveals it.",
        "",
        "Work **only** via the tools given.  First figure out what changed (if anything), then call the appropriate update tool(s).",
        "Finally return a short human-readable summary of what you did.",
        "Please do *not* perform the same action more than once. "
        "\U0001f512  If the transcript chunk contains a `manager_method` event indicating this operation is already in progress or completed, treat it as handled and **do not** perform it again.",
        "",
        "Tools (name \u2192 argspec):",
        json.dumps(_sig_dict(tools), indent=4),
        "",
        "Read through the broader context of your role and recent activity for orientation.",
        "",
        "Current UTC time: " + now(),
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
        "\U0001f52c **Scope of the knowledge store:**",
        "\u2022 \u274c Do **NOT** include any contact-specific or biographic details such as where someone works, their role or title, or what they have recently been doing \u2013 these belong in contact records, bios, or rolling summaries and can be assumed handled elsewhere.",
        "\u2022 \u2705 Store only general, non-public facts that are relevant to the assistant\u2019s role and responsibilities as outlined in the broader context above and that may be useful in future.",
        "",
        "\U0001f9ed **General process:**",
        "1\ufe0f\u20e3 Reflect on the broader context of your role and recent activity above and decide which kinds of facts would be *truly valuable* to retain long-term.",
        "2\ufe0f\u20e3 Read the transcript chunk and pick out any pieces of information that fit those criteria.  It is acceptable if **none** are found.",
        "3\ufe0f\u20e3 For *each* candidate fact:",
        "   \u2022 Call `KnowledgeManager.ask` to check whether this fact (or an equivalent) already exists in the knowledge store.",
        "   \u2022 If it **does exist**, skip to the next fact.",
        "   \u2022 If storing the new fact would be awkward or duplicative because of the current table/column layout, call `KnowledgeManager.refactor` **once** with clear instructions for restructuring.",
        "   \u2022 Finally, add the new fact with `KnowledgeManager.update`.",
        "",
        "\U0001f6ab **Avoid redundant actions:** If you have already asked, refactored, or updated during this turn you do **NOT** need to repeat the same tool call.",
        "\U0001f512  If the transcript chunk contains a `manager_method` event indicating this operation is already in progress or completed, treat it as handled and **do not** perform it again.",
        "",
        "Return a short, human-readable summary of what you stored; if nothing was stored, then please briefly explain why.",
        "",
        "Tools (name \u2192 argspec):",
        json.dumps(_sig_dict(tools), indent=4),
        "",
        "Current UTC time: " + now(),
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
        "\U0001f9ed **General process:**",
        "\u26a0\ufe0f  **Important:** General descriptions of responsibilities or examples of typical duties (e.g. \u2018You\u2019ll be responsible for X\u2019) are **NOT** explicit task requests. Only create, update, or cancel tasks when the transcript includes a clear utterance whereby a concrete action should be taken, and sufficient details are given to act upon (such as \u2018Please do X when you get the chance\u2019 or \u2018Y was already completed by Z, so don\u2019t worry about it\u2019). When in doubt, make no changes.",
        "\U0001f9e0 **Not tasks:** Requests such as \u2018remember that\u2026\u2019, \u2018store this information\u2019, or sharing credentials like passwords are **knowledge storage operations**, not actionable tasks. These must *never* generate or modify tasks in the schedule.",
        "1\ufe0f\u20e3 Reflect on the broader context of your role and recent activity above and decide whether the conversation requests or implies new tasks or any changes to the existing tasks.",
        "2\ufe0f\u20e3 Always begin by calling `TaskScheduler.ask` to retrieve the **current** task list.",
        "3\ufe0f\u20e3 For each required change:",
        "   \u2022 Create a **new** task if it does not yet exist.",
        "   \u2022 Update the **existing** task if details (status, priority, due date, etc.) have changed.",
        "   \u2022 Cancel a task that is no longer relevant.",
        "   \u2022 Re-order tasks or adjust priorities where it improves clarity.",
        "   \u2022 Perform these adjustments via **a single** `TaskScheduler.update` call whenever possible.",
        "",
        "\U0001f6ab **Avoid redundant actions:** If you have already inspected or updated the task list during this turn you do **NOT** need to repeat the same tool call.",
        "\U0001f512  If the transcript chunk contains a `manager_method` event indicating this operation is already in progress or completed, treat it as handled and **do not** perform it again.",
        "",
        "Return a short, human-readable summary of what you changed; if nothing required updating, then please briefly explain why.",
        "",
        "Tools (name \u2192 argspec):",
        json.dumps(_sig_dict(tools), indent=4),
        "",
        "Current UTC time: " + now(),
    ]
    return _with_guidance(lines, guidance)


# ---------------------------------------------------------------------------
# Unified prompt (used by process_chunk for passive 50-message trigger)
# ---------------------------------------------------------------------------


def build_unified_prompt(
    tools: Dict[str, Callable],
    *,
    contacts: bool = True,
    bios: bool = True,
    rolling_summaries: bool = True,
    response_policies: bool = True,
    knowledge: bool = True,
    tasks: bool = True,
    guidance: Optional[str] = None,
) -> str:
    """Build a single system prompt covering all enabled memory capabilities."""

    assistant_full = _assistant_name()

    lines = [
        get_broader_context(),
        "",
        "You are the **offline memory maintenance agent**.  You have been given the latest transcript chunk and your job is to extract and persist any valuable information from it.",
        "",
        "\U0001f6ab **ABSOLUTELY NO HALLUCINATIONS:** Include **only** information that is explicitly stated in the provided transcript chunk. If a detail is not clearly present, you must *not* invent, infer, or elaborate on it.",
        "\U0001f512  If the transcript chunk contains a `manager_method` event indicating an operation is already in progress or completed, treat it as handled and **do not** perform it again.",
        "Please do *not* perform the same action more than once.",
        f"All text you write about contacts is being written for {assistant_full}. Always address {assistant_full} using second-person pronouns (\u2018you\u2019, \u2018your\u2019) rather than their name or third-person forms. Refer to other contacts in the third person so {assistant_full} clearly knows who the text is about.",
        "",
    ]

    # ---- Contact section ----
    if contacts:
        lines.extend(
            [
                "## Contacts",
                "",
                "Create or amend contact records \u2014 names, phone numbers, emails, etc. \u2014 whenever the transcript reveals new or changed facts.",
                "\u2022 Create a **new** contact if it does not yet exist (even if all you have is a first name plus a short bio).",
                "\u2022 Amend an **existing** contact if information has changed or been extended.",
                "\u2022 **Merge duplicate contacts** when two records refer to the same person (use `merge_contacts`).",
                "\u2022 **Nameless contacts** (organisations/services): do **not** populate `first_name`/`surname` with transient representatives\u2019 names.",
                "",
            ],
        )

    if bios:
        lines.extend(
            [
                "## Bios",
                "",
                "For each contact that appears in the transcript, check whether the transcript reveals new *time-invariant* information about them (background, role, expertise, personality, etc.).",
                "If so, update the contact\u2019s `bio` field via `update_contact`.  Bios should be concise freeform text (\u2264 500 words).  Weave new details into the existing bio rather than rewriting wholesale.",
                "Do **not** include fleeting topics or moment-to-moment tasks in bios.",
                "",
            ],
        )

    if rolling_summaries:
        lines.extend(
            [
                "## Rolling Summaries",
                "",
                "For each contact that appears in the transcript, check whether the transcript introduces information worth capturing in their `rolling_summary`.",
                "The rolling summary is concise freeform text (\u2264 500 words) capturing recent activity: conversation themes, outstanding tasks, tone/sentiment, direct interactions.",
                "Weave new information into the existing summary smoothly.  Balance recency with importance \u2014 trivial chit-chat should not eclipse significant developments.",
                "If the transcript adds little about a contact, leave their rolling summary unchanged.",
                "",
            ],
        )

    if response_policies:
        lines.extend(
            [
                "## Response Policies",
                "",
                "When the transcript contains directives about how you (the assistant) should communicate with a specific contact, update that contact\u2019s `response_policy` field.",
                "Response policies cover: tone/formality, response-time expectations, topics to prioritise or avoid, preferred channels, escalation rules.",
                "Only update when a clear directive exists in the transcript.  If no relevant directive is present, leave the policy unchanged.",
                "",
            ],
        )

    if knowledge:
        lines.extend(
            [
                "## Knowledge",
                "",
                "Mine *long-term*, non-contact-specific facts from the transcript and persist them to the knowledge base.",
                "\u2022 Do **NOT** include contact-specific details (those belong in bios/rolling summaries).",
                "\u2022 Store only general facts relevant to the assistant\u2019s role that may be useful in future.",
                "\u2022 Call `KnowledgeManager.ask` first to check for duplicates before storing.",
                "\u2022 Use `KnowledgeManager.refactor` if the current schema needs restructuring.",
                "",
            ],
        )

    if tasks:
        lines.extend(
            [
                "## Tasks",
                "",
                "Maintain the task schedule based on the transcript.",
                "\u2022 Only create/update/cancel tasks when the transcript includes a clear, concrete action request with sufficient detail.",
                "\u2022 General descriptions of responsibilities are **NOT** task requests.",
                "\u2022 \u2018Remember that\u2026\u2019 or credential-sharing are knowledge operations, not tasks.",
                "\u2022 Always call `TaskScheduler.ask` first to check the current task list.",
                "",
            ],
        )

    lines.extend(
        [
            "## Process",
            "",
            "Read the transcript chunk carefully, then use the available tools to persist any relevant information.  Work through each enabled category above.  When finished, return a short summary of what you did (or \u2018no-op\u2019 if nothing needed updating).",
            "",
            "Tools (name \u2192 argspec):",
            json.dumps(_sig_dict(tools), indent=4),
            "",
            "Current UTC time: " + now(),
        ],
    )

    return _with_guidance(lines, guidance)
