"""
Prompt builders for WebSearcher.

These builders use the centralized `PromptSpec` and `compose_system_prompt`
utilities from common/prompt_helpers.py to ensure consistent prompt structure
across all state managers.
"""

from __future__ import annotations

import textwrap
from typing import Dict, Callable, List

from ..common.prompt_helpers import (
    sig_dict,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
    PromptSpec,
    PromptParts,
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
# Dynamic tool documentation builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_ask_tools_documentation(tools: Dict[str, Callable]) -> str:
    """Build dynamic tools documentation section for ask prompt."""
    have_search = "search" in tools
    have_extract = "extract" in tools
    have_crawl = "crawl" in tools
    have_map = "map" in tools

    lines: List[str] = [
        "Tools Available",
        "---------------",
    ]
    if have_search:
        lines += [
            "- search: find relevant sources and provide a concise summary in the 'answer' key.",
            "  • Parameters: max_results, start_date, end_date, include_images",
            "  • Examples:",
            '    - search(query="latest vector database trends", max_results=5)',
            '    - search(query="Q1 updates", start_date="2025-01-01", end_date="2025-03-31")',
            '    - search(query="product logos", include_images=True)',
        ]
    if have_extract:
        lines += [
            "- extract: read page content for specific URLs (batch when possible).",
            "  • Parameters: urls, include_images",
            "  • Examples:",
            '    - extract(urls=["https://site/a", "https://site/b"]) ',
            '    - extract(urls="https://site/a", include_images=True)',
        ]
    if have_crawl:
        lines += [
            "- crawl: explore a site with guidance.",
            "  • Parameters: start_url, instructions, max_depth, max_breadth, limit, include_images",
            "  • Example:",
            '    - crawl(start_url="https://docs.example.com", instructions="Find SDK pages", max_depth=1, max_breadth=3, limit=20)',
        ]
    if have_map:
        lines += [
            "- map: create a structured overview of a topic.",
            "  • Parameters: query, instructions, max_depth, max_breadth, limit, include_images",
            "  • Example:",
            '    - map(query="AI evaluation frameworks", instructions="Group by approach", max_depth=1, max_breadth=3, limit=30)',
        ]

    return "\n".join(lines)


def _build_ask_guidance_sections() -> str:
    """Build static guidance sections for ask prompt."""
    return "\n".join(
        [
            "General Rules and Guidance",
            "--------------------------",
            "- Keep queries concise; if complex, split into smaller, focused searches.",
            "- Prefer a small, high-quality set of sources; cite them in the answer.",
            "- Only fetch page content when you need details beyond snippets.",
            "",
            "Decision Policy and When to Stop",
            "---------------------------------",
            "1. Run a targeted search and read the snippets.",
            "2. If snippets suffice, STOP and write the answer (no more tools).",
            "3. Otherwise, extract at most one highly relevant URL.",
            "4. If still insufficient, do one more targeted step (search OR extract), then STOP and answer.",
            "5. Do not loop through many tools or repeat equivalent steps.",
            "",
            "Answer Requirements",
            "-------------------",
            "- Be precise and concise; cite sources inline (title or URL).",
            "- If evidence is insufficient, do one targeted step; otherwise answer with best-supported facts.",
            "- After you write the final answer, do not call further tools.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def build_ask_prompt(*, tools: Dict[str, Callable]) -> PromptParts:
    """Return the system prompt used by WebSearcher.ask using the shared composer."""
    request_clar_fname = _tool_name(tools, "request_clarification")

    # Build clarification block
    clarification_block = (
        textwrap.dedent(
            f"""
            ─ Clarification ─
            • If the query is ambiguous, ask the user to specify
              `{request_clar_fname}(question="Could you clarify what you mean?")`
            """,
        ).strip()
        if request_clar_fname
        else ""
    )

    # Build dynamic tools documentation
    tools_doc = _build_ask_tools_documentation(tools)

    # Build usage examples
    usage_examples = textwrap.dedent(
        """
Examples
--------
- General web query:
  1) `search(query="what are the major headlines today?", max_results=5)`
- Time-bounded query:
  1) `search(query="AI news", start_date="2025-01-01", end_date="2025-03-31", max_results=3)`
- Deep-dive on a specific URL:
  1) `search(query="Python 3.13 release notes", max_results=3)` → find relevant URL
  2) `extract(urls="https://docs.python.org/release/3.13.0/whatsnew/")` → get full content
- Explore a documentation site:
  1) `crawl(start_url="https://docs.example.com", instructions="Find SDK pages", max_depth=1, max_breadth=3, limit=20)`

Anti-patterns to avoid
---------------------
• Do not loop through many tools or repeat equivalent steps.
• Do not call extract on many URLs; pick the most relevant one or two.
• Do not re-query just to confirm a previous result.
    """,
    ).strip()

    if clarification_block:
        usage_examples = f"{usage_examples}\n{clarification_block}"

    # Build guidance sections
    guidance_block = _build_ask_guidance_sections()

    # Compose using standardized composer
    spec = PromptSpec(
        manager="WebSearcher",
        method="ask",
        tools=tools,
        role_line="You are a **web research assistant**.",
        global_directives=[
            "Use the available tools to answer the user's question.",
            "Produce concise, factual answers with optional inline citations (title or URL).",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
        ],
        include_read_only_guard=True,
        positioning_lines=[tools_doc, guidance_block],
        counts_entity_plural=None,
        counts_value=None,
        columns_payload=None,
        columns_heading="columns",
        include_tools_block=True,
        usage_examples=usage_examples,
        clarification_examples_block=clarification_block or None,
        include_images_policy=False,
        include_images_forwarding=False,
        images_extras_block=None,
        include_parallelism=True,
        schemas=[],
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
    """Return instruction prompt for the simulated WebSearcher.

    Ensures the LLM replies **as if** the requested operation has already
    finished, avoiding responses like "I'll process that now".
    """
    import json  # local import
    from unity.common.context_dump import make_messages_safe_for_context_dump

    preamble = f"On this turn you are simulating the '{method}' method."
    behaviour = (
        "Please always answer the question with an imaginary but plausible "
        "response about the web research findings. Do NOT ask for "
        "clarification or describe your process. Provide a concise answer "
        "with brief source-like references (titles or URLs) as if you had searched."
    )

    parts: list[str] = [preamble, behaviour, "", f"The user input is:\n{user_request}"]
    if parent_chat_context:
        parts.append(
            f"\nCalling chat context:\n{json.dumps(make_messages_safe_for_context_dump(parent_chat_context), indent=4)}",
        )

    return "\n".join(parts)
