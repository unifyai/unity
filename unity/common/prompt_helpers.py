from typing import Callable, Dict, Any, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
import json

__all__ = [
    "clarification_guidance",
    "sig_dict",
    "now_utc_str",
    "tool_name",
    "require_tools",
    "parallelism_guidance",
    "images_policy_block",
    "images_forwarding_block",
    # New standardized prompt composition utilities
    "PromptSpec",
    "compose_system_prompt",
    "render_tools_block",
    "render_counts_and_columns",
    "render_schemas",
    "clarification_top_sentence",
    "clarification_else_policy",
    "special_contacts_block",
    "two_table_reasoning_block",
    "task_queue_invariants_block",
    "task_execute_decision_policy_block",
    "images_extras_for_transcripts",
    "images_first_ask_for_tasks",
]


def clarification_guidance(tools: Dict[str, Callable]) -> str:
    """Return a *single* guidance sentence on how to use the clarification tool.

    The helper looks up the first tool whose name contains the substring
    ``"clarification"`` (case-insensitive).  If such a tool is present the
    returned sentence instructs the model to call it whenever further
    information is required.  Otherwise an **empty string** is returned so
    callers can simply concatenate the result without extra conditionals.
    """
    clar_tool = next((n for n in tools if "clarification" in n.lower()), None)
    if not clar_tool:
        return ""

    return (
        f"If anything is unclear or ambiguous, you must always call the `{clar_tool}` *tool* to "
        "ask the user for clarification before proceeding. Do *not* request clarifications with your final response."
    )


# ---------------------------------------------------------------------------
# Shared utilities for prompt builders
# ---------------------------------------------------------------------------


def sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} for pretty JSON dumps.

    Centralises the inspect.signature → string conversion so all prompts render
    a consistent tool signature block.
    """
    import inspect

    return {name: str(inspect.signature(fn)) for name, fn in tools.items()}


def now_utc_str(time_only: bool = False) -> str:
    """Return current UTC timestamp as a compact human-readable string.

    Parameters
    ----------
    time_only : bool, default False
        When True, return only the time component (HH:MM:SS UTC); otherwise return
        full date and time (YYYY-MM-DD HH:MM:SS UTC).
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return (
        now.strftime("%H:%M:%S UTC")
        if time_only
        else now.strftime("%Y-%m-%d %H:%M:%S UTC")
    )


def tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """Best-effort lookup: find the first tool whose name contains ``needle``.

    Comparison is case-insensitive. Returns ``None`` if not found.
    """
    lowered = needle.lower()
    return next((name for name in tools if lowered in name.lower()), None)


def require_tools(pairs: Dict[str, str | None], tools: Dict[str, Callable]) -> None:
    """Validate dynamic tool resolution and raise a clear error if any are missing.

    Parameters
    ----------
    pairs: mapping of a human-friendly expected substring → resolved tool name (or None)
    tools: the full tool mapping; used only to produce a helpful error message
    """
    missing = [substr for substr, resolved in pairs.items() if resolved is None]
    if not missing:
        return

    available = ", ".join(sorted(tools.keys())) or "<none>"
    expected = ", ".join(missing)
    raise ValueError(
        f"Missing required tools: expected to find tool names containing: {expected}. "
        f"Available tools: {available}.",
    )


def parallelism_guidance() -> str:
    """Return a shared block encouraging batching/parallel tool use."""
    return (
        "Parallelism and single\u2011call preference\n"
        "-------------------------------------\n"
        "\u2022 Prefer a single comprehensive tool call over several surgical calls when a tool can safely do the whole job.\n"
        "\u2022 When several reads or writes are independent, plan them together and run them in parallel rather than a serial drip of micro\u2011calls.\n"
        "\u2022 Batch arguments where possible and avoid confirmatory re\u2011queries unless new ambiguity arises."
    )


def images_policy_block() -> str:
    """Return a generic images policy block suitable for inclusion in system prompts.

    This block is intentionally phrased to apply only when images are present,
    so it can be safely included unconditionally by managers.
    """
    return (
        "Images policy (when images are present)\n"
        "--------------------------------------\n"
        "- Treat images as freeform user-provided visuals (screenshots, photos, UI, attachments).\n"
        "- Do not assume system-specific identifiers or structured record fields (e.g., ids, names, statuses, queue/thread references,\n"
        "  timestamps, due/deadline dates) are visible unless they are clearly shown. This applies across managers (e.g., tasks,\n"
        "  contacts, transcripts).\n"
        "- Default-first question: if the caption is vague or absent, start with a very simple descriptive question such as\n"
        "  'What is shown in this image? What activity appears to be in progress? Which app/page is visible?' Extract salient,\n"
        "  observable elements (apps, UI sections, headings, steps, key text snippets) — not database fields.\n"
        "- If the caption already clearly describes the scene and intent, you may skip the broad question and proceed directly\n"
        "  to a targeted question about a specific on-screen detail.\n"
        "- When information is needed from a single image, prefer `ask_image` with a narrowly scoped question to extract concrete,\n"
        "  observable details — never invent system-specific fields that may not be present on-screen.\n"
        "- Use any extracted cues (e.g., what is in the image, what appears to be done if this is a screen-share) to guide downstream\n"
        "  tool choices (e.g., semantic searches guided by inferred activity or content).\n"
        "- Forwarding rule: when delegating to another tool that declares an `images` parameter, forward the relevant images and\n"
        "  rewrite/augment their annotations so they align with the delegated question or action (not the original user phrasing).\n"
        "  Prefer AnnotatedImageRefs; include a curated subset and preserve user-referenced ordering when it matters.\n"
        "- Anti-patterns to avoid:\n"
        "  • Asking for system-specific identifiers or structured record fields in the first question unless those are clearly visible.\n"
        "  • Assuming the screenshot is a structured record view from a specific manager.\n"
        "  • Re-asking a broad description when the caption already provides that description.\n"
        "- Attach images (`attach_image_raw`) when persistent visual context is helpful for follow-up turns; otherwise prefer targeted `ask_image` calls."
    )


def images_forwarding_block() -> str:
    """General guidance for forwarding images into nested tools.

    Manager‑agnostic: safe to include in any prompt where nested tool calls may occur.
    """
    return (
        "Images forwarding to nested tools\n"
        "----------------------------------\n"
        "• When delegating to another tool that declares an `images` parameter, forward the relevant images.\n"
        "• Rewrite or augment image annotations so they align with the delegated question/action (not the original phrasing).\n"
        "• Prefer AnnotatedImageRefs; include a curated subset and preserve user‑referenced ordering when it matters.\n"
        "• If no images are relevant, omit them rather than attaching unrelated visuals."
    )


# ---------------------------------------------------------------------------
# Standardized prompt composition (step 1 of migration)
# ---------------------------------------------------------------------------


@dataclass
class PromptSpec:
    """Specification for assembling a standardized system prompt.

    Fields are intentionally generic to allow manager/method specific content
    without changing the common ordering. Absent/None fields are skipped.
    """

    manager: str
    method: str
    tools: Dict[str, Callable]

    # Core header
    role_line: str
    global_directives: List[str] = field(default_factory=list)

    # Method-policy toggles
    include_read_only_guard: bool = False  # ask only
    include_execute_policy: bool = (
        False  # execute only (builders usually pass this via positioning_lines)
    )

    # Positioning/cross-manager lines (before counts/tools)
    positioning_lines: List[str] = field(default_factory=list)

    # Counts/columns
    counts_entity_plural: Optional[str] = None
    counts_value: Optional[int] = None
    columns_payload: Optional[Any] = None  # dict | list
    columns_heading: str = "columns"

    # Tools block
    include_tools_block: bool = True

    # Examples
    usage_examples: Optional[str] = None
    clarification_examples_block: Optional[str] = None

    # Images
    include_images_policy: bool = True
    include_images_forwarding: bool = True
    images_extras_block: Optional[str] = None

    # Parallelism
    include_parallelism: bool = True

    # Schemas
    schemas: List[Tuple[str, dict]] = field(default_factory=list)

    # Special blocks near the end
    special_blocks: List[str] = field(default_factory=list)

    # Clarification footer (single-sourced helper sentence)
    include_clarification_footer: bool = True

    # Footer
    include_time_footer: bool = True
    # Optional override for the time footer prefix (allows ':' vs 'is')
    time_footer_prefix: str = "Current UTC time is "


def render_tools_block(tools: Dict[str, Callable]) -> str:
    """Render a labeled tools block with arg-specs as JSON."""
    sig_json = json.dumps(sig_dict(tools), indent=4)
    return "\n".join(["Tools (name → argspec):", sig_json])


def render_counts_and_columns(
    *,
    entity_plural: str,
    count: int,
    columns_payload: Any,
    columns_heading: str = "columns",
) -> str:
    """Render a standard counts + columns section."""
    return "\n".join(
        [
            f"There are currently {count} {entity_plural} stored in a table with the following {columns_heading}:",
            json.dumps(columns_payload, indent=4),
        ],
    )


def render_schemas(schemas: Sequence[Tuple[str, dict]]) -> str:
    """Render multiple schemas under a single heading."""
    if not schemas:
        return ""
    lines: List[str] = ["Schemas", "-------"]
    for name, schema in schemas:
        lines.append(f"{name} = {json.dumps(schema, indent=4)}")
        lines.append("")
    # Drop trailing blank line for neatness
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def clarification_top_sentence(tools: Dict[str, Callable]) -> str:
    """Return a single sentence establishing clarification policy at the top.

    Uses the first tool whose name contains 'request_clarification'.
    """
    name = tool_name(tools, "request_clarification")
    if name:
        return f"Do not ask the user questions in your final response, please only use the `{name}` tool to ask clarifying questions."
    return clarification_else_policy()


def clarification_else_policy() -> str:
    """Else-case when no clarification tool is available."""
    return "Do not ask the user questions in your final response. Instead, proceed using sensible defaults/best‑guess values and explicitly tell inner tools that these are assumptions/best guesses, not confirmed answers."


def special_contacts_block() -> str:
    """Standard block describing special contact ids 0 and 1."""
    return "\n".join(
        [
            "Special contacts",
            "----------------",
            "• contact_id==0 is the assistant (this agent). Do not include the assistant in suggestions, rankings, or comparisons unless it makes sense from the broader context.",
            "• contact_id==1 is the central user (the assistant's supervisor). Many requests originate from this user; do not propose the central user as a candidate unless it makes sense from the broader context.",
        ],
    )


def two_table_reasoning_block(
    *,
    filter_fname: Optional[str],
    search_fname: Optional[str],
) -> str:
    """Guidance for TranscriptManager two-table (Messages + Contacts) reasoning."""
    if not (filter_fname or search_fname):
        return ""
    parts: List[str] = ["Two-table reasoning:"]
    if search_fname:
        parts.append(
            f"- Use semantic `{search_fname}` when you need message content and/or sender contact attributes (e.g., `bio`, `first_name`). The tool will internally ensure embeddings and, when needed, join Transcripts with Contacts on `sender_id == contact_id` to rank results by the sum of per-term similarities.",
        )
    if filter_fname:
        parts.append(
            f"- Use exact `{filter_fname}` only over transcript fields (ids, mediums, timestamps, content equality/contains). Contact fields are not in scope for filtering.",
        )
    return "\n".join(parts)


def task_queue_invariants_block() -> str:
    """Shared queue/schedule invariants used in TaskScheduler.update."""
    return "\n".join(
        [
            "Schedule/Queue invariants (must-follow)",
            "---------------------------------------",
            "• If you provide a schedule with start_at on the head (prev_task is None), status must be 'scheduled' – never 'queued'.",
            "• Non-head tasks (prev_task is not None) must not define start_at; the timestamp belongs to the head only.",
            "• 'primed' must only be used for a head task (prev_task is None).",
            "• A 'scheduled' task must have either a prev_task or a start_at timestamp.",
            "• Status is updated implicitly based on operations (activation, scheduling, completion). Do not set status explicitly.",
        ],
    )


def task_execute_decision_policy_block(
    *,
    execute_by_id_fname: Optional[str],
    execute_isolated_by_id_fname: Optional[str],
    list_queues_fname: Optional[str],
    get_queue_fname: Optional[str],
) -> str:
    """Decision policy and workflow for TaskScheduler.execute."""
    lines: List[str] = [
        "Decision policy (isolation vs chain)",
        "------------------------------------",
        "• Consider the broader chat context and the user's exact phrasing to infer execution scope (single task now vs the whole sequence now).",
        "• Choose isolation for “start X now” requests. Choose queue/chained execution only when the user clearly requests running the whole sequence now.",
        "• Do not attempt to modify queue order or dates during execute; execute does not have queue editing tools.",
        "",
        "Tool semantics (for your decision)",
        "-----------------------------------",
    ]
    if execute_isolated_by_id_fname:
        lines.append(
            f"• `{execute_isolated_by_id_fname}(task_id=…)` – isolation: detach the selected task and start only that task.",
        )
    if execute_by_id_fname:
        lines.append(
            f"• `{execute_by_id_fname}(task_id=…)` – queue mode: start the selected task within its queue so followers remain attached.",
        )
    lines.extend(
        [
            "",
            "EXECUTION WORKFLOW (no queue mutation):",
        ],
    )
    if list_queues_fname and get_queue_fname:
        lines.append(
            f"1) Optionally inspect queues using `{list_queues_fname}()` and `{get_queue_fname}(queue_id=…)` to confirm context.",
        )
    else:
        lines.append(
            "1) Optionally inspect the queue context using the available queue tools.",
        )
    if execute_isolated_by_id_fname and execute_by_id_fname:
        lines.append(
            f"2) Execute by choosing `{execute_isolated_by_id_fname}` (preferred for single‑task‑now) or `{execute_by_id_fname}` (for explicit chain‑now).",
        )
    elif execute_by_id_fname:
        lines.append(f"2) Execute by calling `{execute_by_id_fname}(task_id=<id>)`.")
    lines.append(
        "3) Do not write status fields directly; lifecycle is managed by the scheduler.",
    )
    return "\n".join(lines)


def images_extras_for_transcripts(
    *,
    get_imgs_msg_fname: Optional[str],
    ask_image_fname: Optional[str],
    attach_image_fname: Optional[str],
    attach_msg_imgs_fname: Optional[str],
) -> str:
    """Tool-aware images extras block for TranscriptManager.ask."""
    lines: List[str] = []
    any_tools = any(
        [
            get_imgs_msg_fname,
            ask_image_fname,
            attach_image_fname,
            attach_msg_imgs_fname,
        ],
    )
    if not any_tools:
        return ""
    lines.extend(
        [
            "Images (vision)",
            "---------------",
        ],
    )
    if get_imgs_msg_fname:
        lines.append(
            f"• List images referenced by a specific message (metadata only; no base64). Each item includes any provided freeform annotation explaining how the image relates to the text.\n  `{get_imgs_msg_fname}(message_id=123)`",
        )
    if ask_image_fname:
        lines.append(
            f'• Ask a one‑off question about an image (text answer only; DOES NOT persist visual context)\n  `{ask_image_fname}(image_id=45, question="What color is dominant?")`',
        )
    if attach_image_fname:
        lines.append(
            f'• Attach a specific image for persistent visual reasoning in this loop\n  `{attach_image_fname}(image_id=45, note="Need to inspect the layout")`',
        )
    if attach_msg_imgs_fname:
        lines.append(
            f"• Attach multiple images linked from a message (limit to first 2). For each attached image, the meta includes any provided annotation that aligns the image to the text.\n  `{attach_msg_imgs_fname}(message_id=123, limit=2)`",
        )
    lines.extend(
        [
            "",
            "Guidance on when to use which image tool",
            "---------------------------------------",
            "• Prefer `ask_image` when you need a quick textual observation about a single image, without changing the current loop context.",
            "• Use `attach_image_to_context` or message‑images attachment when follow‑up turns should continue to see the image(s) as visual context in this loop.",
            "• For multi‑image reasoning, side‑by‑side comparisons, or multi‑attribute judgments, attach the relevant image(s) so they are visible within the same loop before answering.",
            "• When images are already linked to a message, prefer attaching them with an appropriate limit in one step; otherwise attach specific image ids individually.",
            "• Avoid issuing several independent one‑off image questions when the answer depends on considering multiple images together; attach once, then reason over the attached visual context.",
        ],
    )
    return "\n".join(lines)


def images_first_ask_for_tasks(*, ask_image_name: Optional[str]) -> str:
    """Images‑first workflow guidance for TaskScheduler.ask."""
    lines: List[str] = [
        "Images-first workflow for ask()",
        "--------------------------------",
        "• When images are present, first interpret the visuals before mapping them to tasks.",
    ]
    if ask_image_name:
        lines.append(
            f"• If captions are vague, call `{ask_image_name}` with a broad, descriptive question (e.g., 'What is shown in this image? What activity appears to be in progress? Which app/page is visible?').",
        )
    else:
        lines.append(
            "• If captions are vague, ask a broad, descriptive question to interpret the screenshot before mapping it to tasks.",
        )
    lines.extend(
        [
            "• If captions already describe the scene and intent clearly, you may skip the broad question and either ask a targeted image question or proceed to a semantic tasks lookup guided by the inferred activity.",
            "• Only ask the image for structured values when they are visibly present on-screen; never assume task metadata (task_id, queue_id, due dates) is visible in generic screenshots.",
        ],
    )
    return "\n".join(lines)


def compose_system_prompt(spec: PromptSpec) -> str:
    """Compose a standardized system prompt based on the provided spec.

    The block order is fixed; absent/None parts are skipped. Builders are
    expected to prepare method/manager‑specific text (e.g., examples) and pass
    them via the `PromptSpec` while relying on this function to normalize
    structure and shared wording.
    """

    from .read_only_ask_guard import read_only_ask_mutation_exit_block

    def _nonempty(s: Optional[str]) -> bool:
        return bool(s and s.strip())

    parts: List[str] = []

    # 1) Role and global directives
    parts.append(spec.role_line)
    if spec.global_directives:
        parts.extend(spec.global_directives)

    # 2) Method‑specific policy guard
    if spec.include_read_only_guard:
        parts.append("")
        parts.append(read_only_ask_mutation_exit_block())

    # 3) Clarification – top sentence
    parts.append("")
    parts.append(clarification_top_sentence(spec.tools))

    # 4) Positioning lines
    if spec.positioning_lines:
        parts.append("")
        for idx, block in enumerate(spec.positioning_lines):
            if idx > 0:
                parts.append("")
            parts.append(block)

    # 5) Counts and columns
    if (
        spec.counts_entity_plural is not None
        and spec.counts_value is not None
        and spec.columns_payload is not None
    ):
        parts.append("")
        parts.append(
            render_counts_and_columns(
                entity_plural=spec.counts_entity_plural,
                count=spec.counts_value,
                columns_payload=spec.columns_payload,
                columns_heading=spec.columns_heading,
            ),
        )

    # 6) Tools block
    if spec.include_tools_block:
        parts.append("")
        parts.append(render_tools_block(spec.tools))

    # 7) Usage examples (+ optional clarification examples)
    if _nonempty(spec.usage_examples):
        parts.append("")
        parts.append(spec.usage_examples or "")
    if _nonempty(spec.clarification_examples_block):
        parts.append(spec.clarification_examples_block or "")

    # 8) Images policy/forwarding/extras
    if spec.include_images_policy:
        parts.append("")
        parts.append(images_policy_block())
    if spec.include_images_forwarding:
        parts.append("")
        parts.append(images_forwarding_block())
    if _nonempty(spec.images_extras_block):
        parts.append("")
        parts.append(spec.images_extras_block or "")

    # 9) Parallelism guidance
    if spec.include_parallelism:
        parts.append("")
        parts.append(parallelism_guidance())

    # 10) Schemas
    if spec.schemas:
        rendered = render_schemas(spec.schemas)
        if _nonempty(rendered):
            parts.append("")
            parts.append(rendered)

    # 11) Special blocks
    for block in spec.special_blocks:
        if _nonempty(block):
            parts.append("")
            parts.append(block)

    # 12) Current time footer
    if spec.include_time_footer:
        parts.append("")
        parts.append(f"{spec.time_footer_prefix}{now_utc_str()}.")

    # 13) Clarification footer (single-sourced guidance sentence)
    if spec.include_clarification_footer:
        parts.append("")
        parts.append(clarification_guidance(spec.tools))

    # Clean leading/trailing empties and join
    normalized: List[str] = []
    for p in parts:
        if p == "" and (not normalized or normalized[-1] == ""):
            continue
        normalized.append(p)
    if normalized and normalized[-1] == "":
        normalized.pop()
    return "\n".join(normalized)
