"""
Prompt builders for GuidanceManager.

Uses schema-first approach: Guidance schema is rendered once early
and referenced in table info to avoid duplication.
"""

from __future__ import annotations

import textwrap
from typing import Callable, Dict, List, Optional, Union

from .types.guidance import Guidance
from ..common.prompt_helpers import (
    clarification_guidance,
    sig_dict,
    now,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
    get_custom_columns,
    # Standardized composer utilities
    PromptSpec,
    compose_system_prompt,
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
# Shared image/function extras builder
# ─────────────────────────────────────────────────────────────────────────────


def _images_extras_for_guidance(
    *,
    get_imgs_fname: Optional[str],
    ask_image_fname: Optional[str],
    attach_image_fname: Optional[str],
    attach_guid_imgs_fname: Optional[str],
    get_funcs_fname: Optional[str],
    attach_funcs_fname: Optional[str],
) -> str:
    """Build images/functions extras block for GuidanceManager.ask."""
    lines: List[str] = []
    any_image_tools = any(
        [get_imgs_fname, ask_image_fname, attach_image_fname, attach_guid_imgs_fname],
    )
    if any_image_tools:
        lines.extend(
            [
                "Images (vision)",
                "---------------",
            ],
        )
        if get_imgs_fname:
            lines.append(
                f"• List images referenced by a guidance item (metadata only; no base64). Each item includes any freeform annotation explaining the relationship to the text.\n  `{get_imgs_fname}(guidance_id=42)`",
            )
        if ask_image_fname:
            lines.append(
                f'• Ask a one‑off question about an image (text answer only; DOES NOT persist visual context)\n  `{ask_image_fname}(image_id=12, question="What text is visible?")`',
            )
        if attach_image_fname:
            lines.append(
                f'• Attach a specific image for persistent visual reasoning in this loop\n  `{attach_image_fname}(image_id=12, note="Need to see the layout")`',
            )
        if attach_guid_imgs_fname:
            lines.append(
                f"• Attach multiple images linked from a guidance item (limit to first 2). For each attached image, the meta includes any collected annotations describing its relevance.\n  `{attach_guid_imgs_fname}(guidance_id=42, limit=2)`",
            )
        lines.extend(
            [
                "",
                "Guidance on when to use which image tool",
                "---------------------------------------",
                f"• Prefer `{ask_image_fname or 'ask_image'}` when you need a quick textual observation about a single image, without changing the current loop context.",
                f"• Use `{attach_image_fname or 'attach_image_to_context'}` or `{attach_guid_imgs_fname or 'attach_guidance_images_to_context'}` when follow‑up turns should continue to see the image(s) as visual context in this loop.",
                "• For multi‑image reasoning, side‑by‑side comparisons, or multi‑attribute judgments (e.g., relative brightness, counts of UI elements), attach the relevant image(s) so they are visible within the same loop before answering.",
                f"• When images are already linked to a guidance row, prefer `{attach_guid_imgs_fname or 'attach_guidance_images_to_context'}` with an appropriate limit to attach them in one step; otherwise attach specific image ids individually using `{attach_image_fname or 'attach_image_to_context'}`.",
                "• Avoid issuing several independent one‑off image questions when the answer depends on considering multiple images together; attach once, then reason over the persistent visual context.",
            ],
        )

    # Functions section (if tools present)
    if get_funcs_fname or attach_funcs_fname:
        if lines:
            lines.append("")
        lines.extend(
            [
                "Functions",
                "---------",
            ],
        )
        if get_funcs_fname:
            lines.append(
                f"• List functions relevant to a guidance item (by ids stored on the row)\n  `{get_funcs_fname}(guidance_id=42, include_implementations=False)`",
            )
        if attach_funcs_fname:
            lines.append(
                f"• Attach related functions into this loop's context for direct reasoning\n  `{attach_funcs_fname}(guidance_id=42, include_implementations=False, limit=3)`",
            )

    return "\n".join(lines) if lines else ""


# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def build_ask_prompt(
    tools: Dict[str, Callable],
    num_items: int,
    columns: Union[List[Dict[str, str]], List[str], Dict[str, str]],
    *,
    include_activity: bool = True,
) -> str:
    """Return the system-prompt used by *ask* using the shared composer."""
    # Extract custom columns (not in Guidance model)
    custom_cols = get_custom_columns(Guidance, columns)

    # Resolve canonical tool names dynamically
    filter_fname = _tool_name(tools, "filter")
    search_fname = _tool_name(tools, "search")
    list_columns_fname = _tool_name(tools, "list_columns")
    reduce_fname = _tool_name(tools, "reduce")
    request_clar_fname = _tool_name(tools, "request_clarification")

    # Image-aware helpers (may be absent; document if present)
    get_imgs_fname = _tool_name(tools, "get_images_for_guidance")
    ask_image_fname = _tool_name(tools, "ask_image")
    attach_image_fname = _tool_name(tools, "attach_image_to_context")
    attach_guid_imgs_fname = _tool_name(tools, "attach_guidance_images_to_context")

    # Function-aware helpers
    get_funcs_fname = _tool_name(tools, "get_functions_for_guidance")
    attach_funcs_fname = _tool_name(tools, "attach_functions_for_guidance_to_context")

    # Validate required tools (request_clar_fname is optional)
    _require_tools(
        {
            "filter": filter_fname,
            "search": search_fname,
            "list_columns": list_columns_fname,
        },
        tools,
    )

    # Build clarification block
    clarification_block = (
        textwrap.dedent(
            f"""
            ─ Clarification ─
            • Ambiguity about which guidance you meant – ask the user to specify
              `{request_clar_fname}(question="There are several relevant guidance entries. Which one do you mean?")`
            """,
        ).strip()
        if request_clar_fname
        else ""
    )

    # Usage examples
    usage_examples_base = f"""
Examples
--------

─ Columns ─
• Inspect schema
  `{list_columns_fname}()`

─ Tool selection (read carefully) ─
• For ANY semantic question over free‑form text (title, content), ALWAYS use `{search_fname}`. Never try to approximate meaning with brittle substring filters.
• Use `{filter_fname}` only for exact/boolean logic over structured fields (ids, equality checks) or for narrow, constrained text where substring checks make sense.

─ Semantic search: targeted references across columns ─
• Find onboarding demo guidance
  `{search_fname}(references={{'title': 'onboarding demo', 'content': 'how to onboard users'}}, k=3)`

• Find guidance about troubleshooting VPN issues
  `{search_fname}(references={{'title': 'VPN', 'content': 'troubleshooting connection issues'}}, k=5)`

─ Filtering (exact/boolean; not semantic) ─
• Exact id match
  `{filter_fname}(filter="guidance_id == 42", limit=1)`

• Filter by id range
  `{filter_fname}(filter="guidance_id >= 10 and guidance_id <= 20")`

Anti‑patterns to avoid
---------------------
• Avoid concatenating entire rows into one long string and comparing a single embedding of the whole question. Instead, pass multiple, focused reference texts keyed by their specific columns.
• Avoid filtering for text‑heavy columns; substring matching is brittle. Prefer `{search_fname}` for content-based queries.
• Avoid re-querying the same tables merely to reconfirm facts that a prior tool call has already established with clear, specific evidence; reuse earlier results and proceed.
    """
    usage_examples = textwrap.dedent(usage_examples_base).strip()
    if clarification_block:
        usage_examples = f"{usage_examples}\n{clarification_block}"
    else:
        usage_examples = "\n".join(
            [
                usage_examples,
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state to inner tools that these are assumptions/best guesses, not confirmed answers.",
                "• If an inner tool requests clarification, explicitly say no clarification channel exists and pass down concrete sensible defaults/best‑guess values, clearly marked as assumptions.",
            ],
        )

    # Build positioning lines
    positioning_lines: List[str] = []
    if ask_image_fname or attach_image_fname or attach_guid_imgs_fname:
        positioning_lines.append(
            f"For images: prefer `{ask_image_fname or 'ask_image'}` for targeted, single‑image Q&A; use `{attach_image_fname or 'attach_image_to_context'}` / `{attach_guid_imgs_fname or 'attach_guidance_images_to_context'}` when you need persistent, multi‑step or multi‑image visual context in this loop.",
        )

    # Build images/functions extras
    images_extras = _images_extras_for_guidance(
        get_imgs_fname=get_imgs_fname,
        ask_image_fname=ask_image_fname,
        attach_image_fname=attach_image_fname,
        attach_guid_imgs_fname=attach_guid_imgs_fname,
        get_funcs_fname=get_funcs_fname,
        attach_funcs_fname=attach_funcs_fname,
    )

    # Build using standardized composer with schema-based table info
    spec = PromptSpec(
        manager="GuidanceManager",
        method="ask",
        tools=tools,
        role_line="You are an assistant specialising in **retrieving distilled guidance items**.",
        global_directives=[
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
        ],
        include_read_only_guard=True,
        positioning_lines=positioning_lines,
        counts_entity_plural="guidance entries",
        counts_value=num_items,
        # Schema-based table info (avoids duplication)
        table_schema_name="Guidance",
        custom_columns=custom_cols if custom_cols else None,
        include_tools_block=True,
        usage_examples=usage_examples,
        clarification_examples_block=clarification_block or None,
        include_images_policy=True,
        include_images_forwarding=True,
        images_extras_block=images_extras or None,
        include_parallelism=True,
        schemas=[("Guidance", Guidance)],  # Full schema defines table columns
        special_blocks=[],
        include_clarification_footer=True,
        include_time_footer=True,
    )

    return compose_system_prompt(spec)


def build_update_prompt(
    tools: Dict[str, Callable],
    num_items: int,
    columns: Union[List[Dict[str, str]], List[str], Dict[str, str]],
    *,
    include_activity: bool = True,
) -> str:
    """Return the system-prompt used by *update* using schema-first approach."""
    # Extract custom columns (not in Guidance model)
    custom_cols = get_custom_columns(Guidance, columns)

    # Resolve canonical tool names dynamically
    add_fname = _tool_name(tools, "add_guidance")
    upd_fname = _tool_name(tools, "update_guidance")
    del_fname = _tool_name(tools, "delete_guidance")
    ask_fname = _tool_name(tools, "ask")
    create_custom_fname = _tool_name(tools, "create_custom_column")
    delete_custom_fname = _tool_name(tools, "delete_custom_column")
    request_clar_fname = _tool_name(tools, "request_clarification")

    # Validate required tools
    _require_tools(
        {
            "add_guidance": add_fname,
            "update_guidance": upd_fname,
            "delete_guidance": del_fname,
            "create_custom_column": create_custom_fname,
            "delete_custom_column": delete_custom_fname,
            "ask": ask_fname,
        },
        tools,
    )

    # Build clarification block
    clarification_block = (
        textwrap.dedent(
            f"""
Clarification
-------------
• If any request is ambiguous, ask the user to disambiguate before changing data
  `{request_clar_fname}(question="There are several possible matches. Which guidance did you mean?")`
            """,
        ).strip()
        if request_clar_fname
        else ""
    )

    # Usage examples
    usage_examples_base = f"""
Tool selection
--------------
• Prefer `{upd_fname}` when you know the exact `guidance_id` for a mutation.
• When the user refers to a guidance item semantically (e.g., "the onboarding demo guidance"), first call `{ask_fname}` to identify the correct `guidance_id`, then call `{upd_fname}`.

Ask vs Clarification
--------------------
• `{ask_fname}` is ONLY for inspecting/locating guidance that ALREADY EXISTS (e.g., to find `guidance_id`, verify fields).
• Do NOT use `{ask_fname}` to ask the human for details about NEW guidance being created/changed in this update request.
• For human clarifications about prospective/new guidance (e.g., title spelling, missing content details), call `{request_clar_fname}` when available.

Create / Update / Delete
------------------------
• Create a new guidance entry
  `{add_fname}(title='Setup demo', content='How to set up the product demo...', images=[{{"raw_image_ref": {{"image_id": 12}}, "annotation": "overview"}}], function_ids=[1, 2])`
• Update a known guidance id
  `{upd_fname}(guidance_id=42, content='Updated narrative...', images=[{{"raw_image_ref": {{"image_id": 15}}, "annotation": "overview"}}], function_ids=[2, 3])`
• Delete a guidance entry
  `{del_fname}(guidance_id=77)`

Schema evolution and custom columns
----------------------------------
• If the user asks to store a new attribute that does not map to built-ins, create a custom column first:
  `{create_custom_fname}(column_name='priority', column_type='str')`
  Then apply the update:
  `{upd_fname}(guidance_id=42, priority='high')`
• Remove optional custom columns with `{delete_custom_fname}(column_name=...)` only when explicitly asked.

Realistic find‑then‑update flows
--------------------------------
• Update the content of the onboarding guidance
  1 Ask a freeform question (no instructions about how to answer):
    `{ask_fname}(text="Which guidance covers onboarding?")`
  2 Update the returned id:
    `{upd_fname}(guidance_id=<id>, content='Updated onboarding steps...')`

Anti‑patterns to avoid
---------------------
• Repeating the exact same tool call with the same arguments as a means to 'make sure it has completed', just call `{ask_fname}` to check the latest state of the guidance
• Making *any* assumptions about the current state of the guidance list, instead you should make liberal use of the `{ask_fname}` tool
    """
    usage_examples = textwrap.dedent(usage_examples_base).strip()
    if clarification_block:
        usage_examples = f"{usage_examples}\n{clarification_block}"
    else:
        usage_examples = "\n".join(
            [
                usage_examples,
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state to inner tools that these are assumptions/best guesses, not confirmed answers.",
                "• If an inner tool requests clarification, explicitly say no clarification channel exists and pass down concrete sensible defaults/best‑guess values, clearly marked as assumptions.",
            ],
        )

    # Compose using standardized composer with schema-based table info
    spec = PromptSpec(
        manager="GuidanceManager",
        method="update",
        tools=tools,
        role_line="You are an assistant in charge of **creating or editing guidance entries**.",
        global_directives=[
            "Choose tools based on the user's intent and the specificity of the target record.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the request and choose the best approach yourself.",
            f"Important: `{ask_fname}` is read‑only and must only be used to locate/inspect guidance that already exists.",
            f"Before creating new guidance or making edits, briefly check whether similar guidance already exists (via `{ask_fname}`) to avoid duplicates.",
        ],
        include_read_only_guard=False,
        positioning_lines=[],
        counts_entity_plural="guidance entries",
        counts_value=num_items,
        # Schema-based table info (avoids duplication)
        table_schema_name="Guidance",
        custom_columns=custom_cols if custom_cols else None,
        include_tools_block=True,
        usage_examples=usage_examples,
        clarification_examples_block=clarification_block or None,
        include_images_policy=True,
        include_images_forwarding=True,
        images_extras_block=None,
        include_parallelism=True,
        schemas=[("Guidance", Guidance)],  # Full schema defines table columns
        special_blocks=[],
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
    """Return an instruction prompt for the simulated GuidanceManager.

    Ensures the LLM replies **as if** the requested operation has already
    finished, avoiding responses like "I'll process that now".
    """
    import json  # local import

    preamble = f"On this turn you are simulating the '{method}' method."
    if method.lower() == "ask":
        behaviour = (
            "Please always *answer* the question (inventing a plausible yet self‑consistent response) – "
            "do **not** ask for clarification or explain your steps.\n\n"
            "Output must contain the actual guidance item(s), not just a summary. "
            "By default, include the full content and key metadata so that no follow‑up call is needed."
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
