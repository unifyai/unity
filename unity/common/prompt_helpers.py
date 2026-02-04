from typing import Callable, Dict, Any, List, Optional, Sequence, Tuple, Type, Union
from dataclasses import dataclass, field
import json

from pydantic import BaseModel

__all__ = [
    "clarification_guidance",
    "sig_dict",
    "now",
    "tool_name",
    "require_tools",
    "parallelism_guidance",
    "tool_availability_guidance",
    "images_policy_block",
    "images_forwarding_block",
    # New standardized prompt composition utilities
    "PromptSpec",
    "PromptParts",
    "compose_system_prompt",
    "render_tools_block",
    "render_counts_and_columns",
    "render_schemas",
    "render_table_info",
    "get_custom_columns",
    "clarification_top_sentence",
    "clarification_else_policy",
    "special_contacts_block",
    "two_table_reasoning_block",
    "task_queue_invariants_block",
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
    import inspect, re

    def _stable(sig_str: str) -> str:
        # Normalize process-specific object addresses printed by Python for
        # sentinel defaults (e.g., "<object object at 0x...>") to a stable marker.
        return re.sub(r"<object object at 0x[0-9a-fA-F]+>", "<UNSET>", sig_str)

    return {name: _stable(str(inspect.signature(fn))) for name, fn in tools.items()}


def now(time_only: bool = False, as_string: bool = True) -> "str | datetime":
    """Return the current timestamp in the assistant's timezone.

    The assistant is the system contact with ``contact_id == 0`` in the
    Contacts table. We read its ``timezone`` field (an IANA timezone
    identifier like "America/New_York") and convert UTC to local time.

    Args:
        time_only: If True and as_string=True, return only the time portion.
        as_string: If True, return formatted string. If False, return datetime object.

    Returns:
        If as_string=True: "Thursday, January 15, 2026 at 02:09 PM UTC" (or time only)
        If as_string=False: datetime object

    In tests, this function is monkeypatched by tests/conftest.py to return
    fixed or incrementing datetimes for cache consistency.
    """
    from datetime import datetime, timezone as dt_timezone
    from zoneinfo import ZoneInfo
    import unify as _unify

    # Resolve Contacts context for the active project
    try:
        _ctxs = _unify.get_active_context()
        _read_ctx = _ctxs.get("read")
    except Exception:
        _read_ctx = None
    _contacts_ctx = f"{_read_ctx}/Contacts" if _read_ctx else "Contacts"

    # Default to UTC if assistant row/field is unavailable
    tz_name: str = "UTC"
    try:
        rows = _unify.get_logs(
            context=_contacts_ctx,
            filter="contact_id == 0",
            limit=1,
            from_fields=["timezone"],
        )
        if rows:
            val = rows[0].entries.get("timezone")
            if isinstance(val, str) and val.strip():
                tz_name = val.strip()
    except Exception:
        # Best-effort only; fall back to UTC
        tz_name = "UTC"

    # Convert UTC now to the target timezone
    utc_now = datetime.now(dt_timezone.utc)
    try:
        tz_info = ZoneInfo(tz_name)
        local_dt = utc_now.astimezone(tz_info)
        label = tz_name
    except Exception:
        # Invalid timezone identifier; fall back to UTC
        local_dt = utc_now
        label = "UTC"

    if not as_string:
        return local_dt

    if time_only:
        return local_dt.strftime("%I:%M %p ") + label

    return local_dt.strftime("%A, %B %d, %Y at %I:%M %p ") + label


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


def tool_availability_guidance() -> str:
    """Guidance about per-turn tool availability and common first-turn patterns.

    This is automatically included in all state manager prompts to explain why
    tools may be restricted to a subset on certain turns.
    """
    return (
        "Tool availability\n"
        "-----------------\n"
        "On some turns, only a subset of the tools listed above may be callable. "
        "This is intentional—common patterns include: semantic search before lexical "
        "filtering, and read-only lookup before mutation. "
        "Use the available tool(s) to gather context; others unlock on subsequent turns."
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
        "• If the user uses ordered deictic references (e.g., 'this', 'then this', 'finally this') or otherwise indicates a specific\n"
        "  ordering over images, treat the provided image list order as authoritative. Do NOT reorder images based on your own guess of\n"
        "  the user's intent or by re-sorting based on image content.\n"
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

    Schema-based Table Handling
    ---------------------------
    To avoid duplicating column definitions, use `table_schema_name` to reference
    a schema already defined in `schemas`. The table info will then say
    "Columns are defined in the X schema above" instead of listing them again.

    - `schemas`: List of (name, model_or_dict) pairs rendered early in the prompt
    - `table_schema_name`: Name of the schema that defines the table columns
    - `custom_columns`: Only non-schema columns (dynamic/custom fields)
    - `columns_payload`: Legacy field; use only when NOT using schema-based approach
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

    # Counts/columns (legacy approach - duplicates schema if both used)
    counts_entity_plural: Optional[str] = None
    counts_value: Optional[int] = None
    columns_payload: Optional[Any] = None  # dict | list (legacy, prefer schema-based)
    columns_heading: str = "columns"

    # Schema-based table info (preferred - avoids duplication)
    table_schema_name: Optional[str] = None  # e.g., "Contact" - references schema
    custom_columns: Optional[Dict[str, str]] = None  # Only non-schema columns

    # Tools block
    include_tools_block: bool = True
    include_tool_availability_guidance: bool = True  # explain per-turn tool masking

    # Examples
    usage_examples: Optional[str] = None
    clarification_examples_block: Optional[str] = None

    # Images
    include_images_policy: bool = True
    include_images_forwarding: bool = True
    images_extras_block: Optional[str] = None

    # Parallelism
    include_parallelism: bool = True

    # Schemas - rendered early; accepts model classes or dicts
    schemas: List[Tuple[str, Any]] = field(default_factory=list)  # (name, Model | dict)

    # Special blocks near the end
    special_blocks: List[str] = field(default_factory=list)

    # Clarification footer (single-sourced helper sentence)
    include_clarification_footer: bool = True

    # Footer
    include_time_footer: bool = True
    # Optional override for the time footer prefix (allows ':' vs 'is')
    time_footer_prefix: str = "Current UTC time is "


@dataclass
class PromptParts:
    """Structured prompt builder with List[Dict] internal representation.

    This class replaces the raw `List[str]` accumulator in `compose_system_prompt`
    with a structured representation where each part is stored as
    `{"type": "text", "text": "...", "_static": True/False}`.

    The `add` method handles separator insertion (blank lines between blocks),
    and `flatten` performs normalization (collapsing consecutive blanks) before
    joining into the final prompt string.
    """

    _parts: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, part: str, separator: bool = True, static: bool = True) -> None:
        """Add a part, optionally with a preceding blank line separator.

        Consecutive parts with the same `static` value are merged into a single
        content block. A new entry is created only when the `static` value
        differs from the previous content part. Empty parts are skipped.

        Parameters
        ----------
        part : str
            The content to add. Empty strings are ignored.
        separator : bool
            If True (default), adds ``\\n\\n`` before the part.
            If False, only a single newline is added.
        static : bool
            If True (default), the part is marked as static content.
            Set to False for dynamic content that may change between runs.
        """
        # Skip empty parts
        if not part:
            return

        if not self._parts:
            # First item - add directly without separator
            self._parts.append({"type": "text", "text": part, "_static": static})
        elif self._parts[-1]["_static"] == static:
            # Same static - merge with previous content
            joiner = "\n\n" if separator else "\n"
            self._parts[-1]["text"] += joiner + part
        else:
            # Different static - add new block
            content = ("\n\n" + part) if separator else "\n" + part
            self._parts.append({"type": "text", "text": content, "_static": static})

    def to_list(self) -> List[Dict[str, Any]]:
        """Return the internal structured parts."""
        return list(self._parts)

    def flatten(self) -> str:
        """Return the full prompt string by concatenating all parts."""
        return "".join(p["text"] for p in self._parts)

    def __str__(self) -> str:
        return self.flatten()


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


def get_custom_columns(
    model: Type[BaseModel],
    columns: Union[Dict[str, str], List[Dict[str, str]], List[str]],
) -> Dict[str, str]:
    """Extract columns that are NOT defined in the Pydantic model (i.e., custom/dynamic columns).

    Parameters
    ----------
    model : Type[BaseModel]
        The Pydantic model class (e.g., Contact, Task).
    columns : Union[Dict[str, str], List[Dict[str, str]], List[str]]
        The runtime columns, either as a dict {name: type} or list of dicts/strings.

    Returns
    -------
    Dict[str, str]
        Only the columns not present in model.model_fields.
    """
    builtin_fields = set(model.model_fields.keys())

    # Normalize columns to dict
    if isinstance(columns, dict):
        cols_dict = columns
    elif isinstance(columns, list):
        if not columns:
            return {}
        if isinstance(columns[0], dict):
            # List of single-key dicts: [{"field1": "type1"}, ...]
            cols_dict = {}
            for item in columns:
                cols_dict.update(item)
        else:
            # List of strings: ["field1", "field2", ...]
            cols_dict = {c: "unknown" for c in columns}
    else:
        return {}

    # Return only non-builtin columns (excluding embedding columns)
    return {
        name: typ
        for name, typ in cols_dict.items()
        if name not in builtin_fields
        and not str(name).startswith("_")
        and not str(name).endswith("_emb")
    }


def render_table_info(
    *,
    entity_plural: str,
    count: int,
    schema_name: Optional[str] = None,
    custom_columns: Optional[Dict[str, str]] = None,
) -> str:
    """Render table info that references a schema instead of duplicating column definitions.

    Parameters
    ----------
    entity_plural : str
        Plural name of the entity (e.g., "contacts", "tasks").
    count : int
        Current number of entities in the table.
    schema_name : Optional[str]
        Name of the schema to reference (e.g., "Contact"). If provided, columns
        are referenced rather than listed.
    custom_columns : Optional[Dict[str, str]]
        Additional custom/dynamic columns not in the schema.

    Returns
    -------
    str
        Formatted table info block.
    """
    parts = [f"There are currently {count} {entity_plural}."]

    if schema_name:
        parts.append(f"Columns are defined in the {schema_name} schema above.")

    if custom_columns:
        parts.append(f"Additional custom columns: {json.dumps(custom_columns)}")

    return "\n".join(parts)


# Type alias for schema entries: (name, model_class_or_dict)
SchemaEntry = Tuple[str, Union[Type[BaseModel], dict]]


def render_schemas(schemas: Sequence[SchemaEntry]) -> str:
    """Render multiple schemas under a single heading.

    Accepts either Pydantic model classes or pre-computed dicts. When a model
    class is provided, its full JSON schema is extracted automatically.

    Parameters
    ----------
    schemas : Sequence[Tuple[str, Union[Type[BaseModel], dict]]]
        List of (name, schema_or_model) pairs.

    Returns
    -------
    str
        Formatted schemas block.
    """
    if not schemas:
        return ""
    lines: List[str] = ["Schemas", "-------"]
    for name, schema_or_model in schemas:
        if isinstance(schema_or_model, type) and issubclass(schema_or_model, BaseModel):
            schema = schema_or_model.model_json_schema()
        else:
            schema = schema_or_model
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


def compose_system_prompt(spec: PromptSpec) -> PromptParts:
    """Compose a standardized system prompt based on the provided spec.

    The block order is fixed; absent/None parts are skipped. Builders are
    expected to prepare method/manager‑specific text (e.g., examples) and pass
    them via the `PromptSpec` while relying on this function to normalize
    structure and shared wording.

    Schema-Based Table Handling
    ---------------------------
    When `table_schema_name` is set, schemas are rendered early and the table
    info references the schema instead of duplicating column definitions.
    This avoids the duplication that occurs when both `columns_payload` and
    `schemas` contain the same field information.
    """

    from .read_only_ask_guard import read_only_ask_mutation_exit_block

    def _nonempty(s: Optional[str]) -> bool:
        return bool(s and s.strip())

    # Determine if using schema-based table info (preferred) vs legacy columns
    use_schema_table_info = spec.table_schema_name is not None

    parts = PromptParts()

    # 1) Role and global directives
    parts.add(spec.role_line, separator=False)
    for directive in spec.global_directives:
        parts.add(directive, separator=False)

    # 2) Method‑specific policy guard
    if spec.include_read_only_guard:
        parts.add(read_only_ask_mutation_exit_block())

    # 3) Clarification – top sentence
    parts.add(clarification_top_sentence(spec.tools))

    # 4) Positioning lines
    if spec.positioning_lines:
        for idx, block in enumerate(spec.positioning_lines):
            # First positioning line gets a separator, subsequent ones also get separators
            parts.add(block)

    # 5) Schemas - render EARLY when using schema-based table info
    #    This ensures schemas appear before they are referenced in table info
    if use_schema_table_info and spec.schemas:
        rendered = render_schemas(spec.schemas)
        if _nonempty(rendered):
            parts.add(rendered)

    # 6) Counts and table info
    if spec.counts_entity_plural is not None and spec.counts_value is not None:
        if use_schema_table_info:
            # Schema-based: reference schema instead of duplicating columns
            parts.add(
                render_table_info(
                    entity_plural=spec.counts_entity_plural,
                    count=spec.counts_value,
                    schema_name=spec.table_schema_name,
                    custom_columns=spec.custom_columns,
                ),
            )
        elif spec.columns_payload is not None:
            # Legacy: include full columns (may duplicate schema)
            parts.add(
                render_counts_and_columns(
                    entity_plural=spec.counts_entity_plural,
                    count=spec.counts_value,
                    columns_payload=spec.columns_payload,
                    columns_heading=spec.columns_heading,
                ),
            )

    # 7) Tools block
    if spec.include_tools_block:
        parts.add(render_tools_block(spec.tools))

    # 7b) Tool availability guidance (per-turn masking explanation)
    if spec.include_tool_availability_guidance:
        parts.add(tool_availability_guidance())

    # 8) Usage examples (+ optional clarification examples)
    if _nonempty(spec.usage_examples):
        parts.add(spec.usage_examples or "")
    if _nonempty(spec.clarification_examples_block):
        parts.add(spec.clarification_examples_block or "", separator=False)

    # 9) Images policy/forwarding/extras
    if spec.include_images_policy:
        parts.add(images_policy_block())
    if spec.include_images_forwarding:
        parts.add(images_forwarding_block())
    if _nonempty(spec.images_extras_block):
        parts.add(spec.images_extras_block or "")

    # 10) Parallelism guidance
    if spec.include_parallelism:
        parts.add(parallelism_guidance())

    # 11) Schemas - render late if NOT using schema-based table info (legacy)
    if not use_schema_table_info and spec.schemas:
        rendered = render_schemas(spec.schemas)
        if _nonempty(rendered):
            parts.add(rendered)

    # 12) Special blocks
    for block in spec.special_blocks:
        if _nonempty(block):
            parts.add(block)

    # 13) Current time footer
    if spec.include_time_footer:
        parts.add(f"{spec.time_footer_prefix}{now()}.", static=False)

    # 14) Clarification footer (single-sourced guidance sentence)
    if spec.include_clarification_footer:
        parts.add(clarification_guidance(spec.tools), static=False)

    return parts
