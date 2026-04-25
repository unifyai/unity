"""Prompt builders for TranscriptManager.

These builders use the centralized `PromptSpec` and `compose_system_prompt`
utilities from common/prompt_helpers.py to ensure consistent prompt structure.

The TranscriptManager queries two tables:
1. Transcripts table (messages) - defined by the Message schema
2. Contacts table (sender info) - defined by the Contact schema

Schemas are rendered once early in the prompt and referenced throughout.
"""

from __future__ import annotations

import json
import textwrap
from typing import Callable, Dict, Union, List, Optional

# Schemas used in the prompt -------------------------------------------------
from ..contact_manager.types.contact import Contact
from .types.message import Message
from unity.conversation_manager.cm_types import Medium
from unity.session_details import SESSION_DETAILS
from ..common.prompt_helpers import (
    clarification_guidance,
    sig_dict,
    now,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
    get_custom_columns,
    # New standardized composer utilities
    PromptSpec,
    PromptParts,
    compose_system_prompt,
    two_table_reasoning_block as _two_table_reasoning_block,
    images_extras_for_transcripts as _images_extras_for_transcripts,
)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} using shared helper."""
    return sig_dict(tools)


def _tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """Delegate to shared tool name resolver."""
    return _shared_tool_name(tools, needle)


def _require_tools(pairs: Dict[str, str | None], tools: Dict[str, Callable]) -> None:
    """Delegate validation to shared helper for consistent errors."""
    _shared_require_tools(pairs, tools)


# ─────────────────────────────────────────────────────────────────────────────
# Two-table info helper
# ─────────────────────────────────────────────────────────────────────────────


def _render_two_table_info(
    num_messages: int,
    transcript_custom_columns: Optional[Dict[str, str]] = None,
) -> str:
    """Render table info for TranscriptManager's two-table architecture.

    The Transcripts table stores messages; the Contacts table provides sender info.
    Both schemas are rendered early in the prompt; this block references them.
    """
    lines = [
        f"There are currently {num_messages} messages.",
        "",
        "Data architecture:",
        "- **Transcripts table**: Stores messages. Columns defined in the Message schema above.",
        "- **Contacts table**: Stores sender information. Columns defined in the Contact schema above.",
        "- Semantic search (`search_messages`) can query across both tables via sender_id → contact_id joins.",
        "- Exact filtering (`filter_messages`) is limited to Transcript columns only.",
    ]

    if transcript_custom_columns:
        lines.append(
            f"- Additional custom columns on Transcripts: {json.dumps(transcript_custom_columns)}",
        )

    return "\n".join(lines)


def _team_history_block() -> str | None:
    """Return Hive transcript authorship guidance for the active body."""

    current_assistant_id = SESSION_DETAILS.assistant.agent_id
    if SESSION_DETAILS.hive_id is None or current_assistant_id is None:
        return None

    return textwrap.dedent(
        f"""
        Shared transcript authorship:
        - This body is assistant id {current_assistant_id}.
        - When a retrieved message has `auth_aid` / `authoring_assistant_id` different from {current_assistant_id}, treat that row as team history.
          Phrase it as "my colleague ..." or "another body on my team ..." rather than as something you personally remember saying.
        - When `auth_aid` matches {current_assistant_id} or is absent, keep the normal first-person framing.
        """,
    ).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def build_ask_prompt(
    tools: Dict[str, Callable],
    num_messages: int,
    transcript_columns: Union[Dict[str, str], List[dict], List[str]],
    contact_columns: Union[Dict[str, str], List[dict], List[str]],
    *,
    include_activity: bool = True,
) -> PromptParts:  # noqa: C901 – long, but flat
    """
    Build the system-prompt for :pyfunc:`TranscriptManager.ask`.

    Uses schema-first approach: Message and Contact schemas are rendered once
    early in the prompt. Table info references these schemas instead of
    duplicating column definitions.
    """
    # Extract custom columns for transcripts (not in Message model)
    transcript_custom_cols = get_custom_columns(Message, transcript_columns)

    # Resolve canonical tool names dynamically
    filter_messages_fname = _tool_name(tools, "filter_messages")
    search_messages_fname = _tool_name(tools, "search_messages")
    reduce_fname = _tool_name(tools, "reduce")
    request_clar_fname = _tool_name(tools, "request_clarification")
    # Image-aware helpers (may be absent; document if present)
    get_imgs_msg_fname = _tool_name(tools, "get_images_for_message")
    ask_image_fname = _tool_name(tools, "ask_image")
    attach_image_fname = _tool_name(tools, "attach_image_to_context")
    attach_msg_imgs_fname = _tool_name(tools, "attach_message_images_to_context")

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
            Ask vs Clarification
            --------------------
            • `{search_messages_fname}` / `{filter_messages_fname}` are for querying **existing** transcripts only.
            • Do NOT use `ask` to ask the human questions. For human clarifications about which conversation/date/person, call:
              `{request_clar_fname}(question=\"Which conversation are you referring to?\")`
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
 • Most recent SMS from contact 7
   `{filter_messages_fname}(filter="sender_id == 7 and medium == 'sms_message'", limit=1, offset=0)`
 • Last month's emails (if datetime comparisons are supported by your backend)
   `{filter_messages_fname}(filter="medium == 'email' and timestamp >= '2024-01-01T00:00:00' and timestamp < '2024-02-01T00:00:00'", limit=100)`

 ─ Numeric aggregations ─
 • For numeric reduction metrics (count, sum, mean, min, max, median, mode, var, std) over numeric columns, use `{reduce_fname}` instead of filtering and computing in-memory.
   `{reduce_fname}(metric='sum', keys='message_id', group_by='medium')`

 ─ Images (vision) ─
 • List images referenced by a specific message (metadata only; no base64). Each item includes any provided freeform annotation explaining how the image relates to the text.
   `{get_imgs_msg_fname}(message_id=123)`
 • Ask a one‑off question about an image (text answer only; DOES NOT persist visual context)
   `{ask_image_fname}(image_id=45, question="What color is dominant?")`
 • Attach a specific image for persistent visual reasoning in this loop
   `{attach_image_fname}(image_id=45, note="Need to inspect the layout")`
 • Attach multiple images linked from a message (limit to first 2). For each attached image, the meta includes any provided `annotation` that aligns the image to the text.
   `{attach_msg_imgs_fname}(message_id=123, limit=2)`

 Guidance on when to use which image tool
 ---------------------------------------
 • Prefer `{ask_image_fname}` when you need a quick textual observation about a single image, without changing the current loop context.
  • Use `{attach_image_fname}` or `{attach_msg_imgs_fname}` when follow‑up turns should continue to see the image(s) as visual context in this loop.
  • For multi‑image reasoning, side‑by‑side comparisons, or multi‑attribute judgments (e.g., relative brightness, visual complexity, or counts of UI elements), attach the relevant image(s) so they are visible within the same loop before answering.
  • When images are already linked to a message, prefer `{attach_msg_imgs_fname}` with an appropriate `limit` to attach them in one step; otherwise attach specific image ids individually using `{attach_image_fname}`.
  • Avoid issuing several independent one‑off image questions when the answer depends on considering multiple images together; attach once, then reason over the attached visual context.

 Anti‑patterns to avoid
 ---------------------
 • Avoid the default search behaviour of concatenating every column into one long string and comparing a single embedding of the whole question. Instead, pass multiple, focused reference texts keyed by their specific columns. The ranking minimises the sum of cosine distances and is more robust.
 • Avoid filtering for text‑heavy columns; substring matching is brittle. Prefer `{search_messages_fname}` for content‑based queries.
 • Do not attempt to reference contact fields (e.g., `bio`, `occupation`) inside `{filter_messages_fname}`; those fields live on the Contacts table. Use `{search_messages_fname}` to leverage sender contact fields.
 • Avoid re‑querying the same tables or managers merely to reconfirm facts that a prior tool call has already established with clear, specific evidence; reuse earlier results and proceed.
 • Do not automatically chain a `{filter_messages_fname}` call immediately after a successful `{search_messages_fname}` result unless you genuinely need an exact, structured constraint that the semantic search did not provide.
 • If you call ContactManager tools during transcript analysis, avoid repeating those calls in the same reasoning chain when earlier results already identified the necessary contacts and no new ambiguity has arisen.
     """
    # Hide image sections gracefully when tools are absent
    examples_text = textwrap.dedent(usage_examples_base)
    if not get_imgs_msg_fname or not ask_image_fname:
        examples_text = examples_text.replace("\n ─ Images (vision) ─\n", "\n")
    if not get_imgs_msg_fname:
        examples_text = examples_text.replace(
            f"\n • List images referenced by a specific message (metadata only; no base64)\n   `{{get_imgs}}(message_id=123)`".replace(
                "{get_imgs}",
                str(get_imgs_msg_fname),
            ),
            "",
        )
    if not ask_image_fname:
        examples_text = examples_text.replace(
            f'\n • Ask a one‑off question about an image (text answer only; DOES NOT persist visual context)\n   `{{ask_img}}(image_id=45, question="What color is dominant?")`'.replace(
                "{ask_img}",
                str(ask_image_fname),
            ),
            "",
        )
    if not attach_image_fname:
        examples_text = examples_text.replace(
            f'\n • Attach a specific image for persistent visual reasoning in this loop\n   `{{attach_img}}(image_id=45, note="Need to inspect the layout")`'.replace(
                "{attach_img}",
                str(attach_image_fname),
            ),
            "",
        )
    if not attach_msg_imgs_fname:
        examples_text = examples_text.replace(
            f"\n • Attach multiple images linked from a message (limit to first 2)\n   `{{attach_msg}}(message_id=123, limit=2)`".replace(
                "{attach_msg}",
                str(attach_msg_imgs_fname),
            ),
            "",
        )

    usage_examples = textwrap.dedent(examples_text).strip()
    if clarification_block:
        usage_examples = f"{usage_examples}\n{clarification_block}"
    else:
        usage_examples = "\n".join(
            [
                usage_examples,
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state to inner tools that these are assumptions/best guesses, not confirmed answers.",
                "• If an inner tool requests clarification, explicitly say no clarification channel exists and pass down concrete sensible defaults/best‑guess values, clearly marked as assumptions.",
                "• Remember: `ask` is read‑only and for EXISTING transcripts only. Do not route human clarifications through it.",
            ],
        )

    clar_section = clarification_guidance(tools)

    # Build using standardized composer with schema-based table info
    two_table_block = _two_table_reasoning_block(
        filter_fname=filter_messages_fname,
        search_fname=search_messages_fname,
    )

    # Two-table info: explains both tables and references schemas
    two_table_info = _render_two_table_info(
        num_messages=num_messages,
        transcript_custom_columns=(
            transcript_custom_cols if transcript_custom_cols else None
        ),
    )

    positioning_lines = [
        "Please mention relevant `message_id` and/or `exchange_id` values in your response when possible.",
        "Use the tools to gather missing context before asking the user for clarifications.",
        two_table_block,
    ]
    positioning_lines = [ln for ln in positioning_lines if ln]

    images_extras = _images_extras_for_transcripts(
        get_imgs_msg_fname=get_imgs_msg_fname,
        ask_image_fname=ask_image_fname,
        attach_image_fname=attach_image_fname,
        attach_msg_imgs_fname=attach_msg_imgs_fname,
    )

    # Build schemas: use model classes for automatic JSON schema extraction
    medium_descriptions = {m.value: m.description for m in Medium}

    schemas = [
        ("Contact", Contact),  # Full schema for sender info
        ("Message", Message),  # Full schema for transcript messages
        ("Communication Channels (Mediums)", medium_descriptions),
        ("Message field shorthand (full → shorthand)", Message.shorthand_map()),
        ("Message field shorthand (shorthand → full)", Message.shorthand_inverse_map()),
    ]

    special_blocks = [two_table_info]
    team_history = _team_history_block()
    if team_history:
        special_blocks.append(team_history)

    spec = PromptSpec(
        manager="TranscriptManager",
        method="ask",
        tools=tools,
        role_line="You are an assistant specialised in **querying and analysing communication transcripts**.",
        global_directives=[
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
        ],
        include_read_only_guard=True,
        positioning_lines=positioning_lines,
        # Schema-based table info: schemas define columns, custom block explains two-table architecture
        table_schema_name="Message",  # Primary table schema
        counts_entity_plural=None,  # Handled by two_table_info special block
        counts_value=None,
        include_tools_block=True,
        usage_examples=usage_examples,
        clarification_examples_block=clarification_block or None,
        include_images_policy=True,
        include_images_forwarding=True,
        images_extras_block=images_extras or None,
        include_parallelism=True,
        schemas=schemas,
        special_blocks=special_blocks,  # Custom transcript context blocks
        include_clarification_footer=True,
        include_time_footer=True,
    )

    return compose_system_prompt(spec)


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
    from unity.common.context_dump import make_messages_safe_for_context_dump

    preamble = f"On this turn you are simulating the '{method}' method."
    if method.lower() == "ask":
        behaviour = (
            "Please always *answer* the question (inventing a plausible yet self‑consistent response) – "
            "do **not** ask for clarification or explain your steps.\n\n"
            "Output must contain the actual transcript message(s), not just a summary. "
            "By default, include the full content and key metadata so that no follow‑up call is needed.\n\n"
            "When the user asks for a single or the 'most recent' item, return exactly one message. "
            "When they ask to list or show multiple, return the requested number (or a small number if unspecified).\n\n"
            "For each message, include at minimum:\n"
            "- Timestamp (UTC ISO)\n"
            "- Channel (e.g., Email, SMS, Call)\n"
            "- Subject (if applicable)\n"
            "- Sender (name and role if known)\n"
            "- Recipients (list or 'N/A')\n"
            "- Message ID (integer)\n"
            "- Exchange ID (integer)\n"
            "- Content (the body text, quoted verbatim)\n\n"
            "Format as concise bullet points or compact JSON‑like blocks. "
            "Avoid hedging language and avoid meta‑commentary about the process."
        )
    else:
        behaviour = (
            "Please act as though the request has been fully satisfied. "
            "Respond in past tense with the final outcome, not the process."
        )

    parts: list[str] = [preamble, behaviour, "", f"The user input is:\n{user_request}"]
    if parent_chat_context:
        parts.append(
            f"\nCalling chat context:\n{json.dumps(make_messages_safe_for_context_dump(parent_chat_context), indent=4)}",
        )

    return "\n".join(parts)
