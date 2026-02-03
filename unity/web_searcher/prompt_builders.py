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
    # Tool names are stripped of leading underscores by methods_to_tool_dict
    have_filter_websites = "filter_websites" in tools
    have_search_websites = "search_websites" in tools
    have_search_gated = "gated_website_search" in tools

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
    if have_filter_websites:
        lines += [
            "- filter_websites: list websites matching a boolean filter over columns.",
            "  • Parameters: filter, offset, limit",
            "  • Examples:",
            '    - filter_websites(filter="gated == True")',
            "    - filter_websites(filter=\"host == 'medium.com'\", limit=1)",
        ]
    if have_search_websites:
        lines += [
            "- search_websites: semantic search over the Websites catalog using notes similarity.",
            "  • Parameters: notes, k",
            "  • Example:",
            '    - search_websites(notes="subscription sources for ML news", k=5)',
        ]
    if have_search_gated:
        lines += [
            "- gated_website_search: search a specific website via the Actor (handles login if gated).",
            "  • Parameters: queries (str or list[str]), website",
            "  • **IMPORTANT**: Spawns an expensive web session. Call exactly ONCE per site — never retry the same site.",
            "  • **Multi-query support**: Pass multiple queries for DIFFERENT purposes (not variations of the same search).",
            "    Good: queries=['latest news', 'recent deals'] — different content types.",
            "    Bad: queries=['AI trends', 'AI news', 'AI updates'] — redundant variations of same topic.",
            "    Bad: queries=['Bushey WD23', 'WD23 2NN', 'Bushey care home'] — redundant location variations.",
            "    For location searches, use ONE clear query with the town name (e.g., 'care homes in Bushey').",
            "  • **Multi-site queries**: For different gated sites, call this tool exactly once per site.",
            "    Example: 'Search Medium and TDS for AI and LLM' → 1 call for Medium + 1 call for TDS = 2 total calls.",
            "  • **Query tips**: For location searches, use ONE query with the town/city name — do NOT create multiple variations.",
            "    Good: queries='care homes in Bushey' — single, clear location query.",
            "    Bad: queries=['Bushey WD23', 'WD23 2NN', 'Hertfordshire care home WD23'] — redundant variations.",
            "  • **Returns raw content**: The tool returns ALL raw page content found (not pre-summarized).",
            "    After receiving ALL results, synthesize and summarize into a coherent answer with inline citations.",
            "  • Examples:",
            '    - gated_website_search(queries="latest AI trends", website={"host": "medium.com"})',
            '    - gated_website_search(queries=["AI trends", "LLM fine-tuning"], website={"host": "medium.com"})',
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
            "- Do not claim inability to log into personal accounts. When a Website entry exists and credentials are available, the Actor can attempt sign-in securely. If credentials are missing or login fails, proceed with public content and clearly state assumptions.",
            "- If the request mentions a specific website (host like 'medium.com' or a human-friendly name like 'Medium'), first consult the Websites catalog:",
            "  • Use `filter_websites` for exact host/name filters; use `search_websites` when only thematic notes are given.",
            "  • If a row exists and `gated=True`, use `gated_website_search(queries=..., website=...)` to browse with login.",
            "  • Otherwise, use general tools (`search`, `extract`, `crawl`, `map`).",
            "",
            "Website-aware Routing",
            "----------------------",
            "- Use `search_websites` to find relevant Website entries by notes similarity (catalog lookup only; does not browse).",
            "- Use `filter_websites` for exact/boolean matches over columns (including host like 'medium.com' or name like 'Medium').",
            "- When answering a question that targets a specific site:",
            "  1) Look up the site using `filter_websites` or `search_websites`.",
            "  2) If the site exists and `gated=True`, use `gated_website_search(queries=..., website=...)` to login with saved credentials and browse.",
            "  3) If not gated or no matching Website entry exists, use general tools (`search`, then optionally `extract`/`crawl`/`map`).",
            "- Do NOT use `search_websites` to read web content; it only searches the Websites catalog.",
            "",
            "Decision Policy and When to Stop",
            "---------------------------------",
            "1. Run a targeted search and read the snippets.",
            "2. If snippets suffice, STOP and write the answer (no more tools).",
            "3. Otherwise, extract at most one highly relevant URL.",
            "4. If still insufficient, do one more targeted step (search OR extract), then STOP and answer.",
            "5. Do not loop through many tools or repeat equivalent steps.",
            "6. **Gated websites**: Call `gated_website_search` ONCE per site. Do NOT retry the same site.",
            "   Pass multiple queries as a list to search different topics on the same site in one call.",
            "   For multi-site queries, call consecutively for each site, then synthesize all results together.",
            "7. **After gated search, STOP**: Once you have called `gated_website_search` ONCE per site for all requested sites,",
            "   do NOT call `search`, `extract`, `crawl`, or `map` for additional content. Synthesize what you have and answer.",
            "",
            "Answer Requirements",
            "-------------------",
            "- Be precise and concise; cite sources inline (title or URL).",
            "- If evidence is insufficient, do one targeted step; otherwise answer with best-supported facts.",
            "- **For gated website results**: Synthesize the raw content into a coherent summary.",
            "  If multiple sites were searched, combine findings and note which source each fact came from.",
            "  Include inline citations (e.g., [Source Title](URL) or 'according to <title>') for each key fact.",
            "- After you write the final answer, do not call further tools.",
        ],
    )


def _build_update_tools_documentation(tools: Dict[str, Callable]) -> str:
    """Build dynamic tools documentation section for update prompt."""
    # Tool names are stripped of leading underscores by methods_to_tool_dict
    have_create = "create_website" in tools
    have_update = "update_website" in tools
    have_delete = "delete_website" in tools
    have_ask = "ask" in tools

    lines: List[str] = [
        "Tools Available",
        "---------------",
    ]
    if have_create:
        lines += [
            "- create_website: create a new Website row (unique by host).",
            "  • Parameters: name, host, gated, subscribed, credentials, actor_entrypoint, notes",
            "  • Examples:",
            "    - create_website(name='Medium', host='medium.com', gated=True, subscribed=True, credentials=[101, 102], notes='Tech journalism and tutorials')",
            "    - create_website(name='arXiv', host='arxiv.org', gated=False, subscribed=False, notes='Academic preprints')",
        ]
    if have_update:
        lines += [
            "- update_website: update fields of an existing Website.",
            "  • Identify by one of: website_id, match_host, match_name",
            "  • Updatable fields: name, host, gated, subscribed, credentials, actor_entrypoint, notes",
            "  • Examples:",
            "    - update_website(match_host='medium.com', subscribed=False)",
            "    - update_website(website_id=3, name='NYTimes', host='nytimes.com')",
        ]
    if have_delete:
        lines += [
            "- delete_website: delete a Website row by host or website_id (exact match).",
            "  • Parameters: name, host, website_id",
            "  • Examples:",
            "    - delete_website(name='Financial Times')",
            "    - delete_website(host='example.com')",
            "    - delete_website(website_id=42)",
        ]
    if have_ask:
        lines += [
            "- ask: read-only inspection helper (calls catalog tools like filter_websites/search_websites).",
            "  • Parameters: text",
            "  • Examples:",
            "    - ask(text='List gated websites')  → should call filter_websites(filter=\"gated == True\")",
            "    - ask(text='Which websites match ML news subscriptions?')  → should call search_websites(notes='ML news subscription')",
        ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def build_ask_prompt(*, tools: Dict[str, Callable]) -> str:
    """Return the system prompt used by WebSearcher.ask using the shared composer."""
    request_clar_fname = _tool_name(tools, "request_clarification")

    # Build clarification block
    clarification_block = (
        textwrap.dedent(
            f"""
            ─ Clarification ─
            • If the query is ambiguous, ask the user to specify
              `{request_clar_fname}(question="Which website or topic did you mean?")`
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
- Login to my GitHub and summarize my profile:
  1) `filter_websites(filter="host == 'github.com' or name == 'GitHub'", limit=1)`
  2) If found and gated=True: `gated_website_search(queries='summarize my GitHub profile', website=<row>)`
  3) Else: use `crawl`/`extract` as appropriate.
- Access my Towards Data Science subscription article and summarize:
  1) `filter_websites(filter="host == 'towardsdatascience.com' or name == 'Towards Data Science'", limit=1)`
  2) If found and gated=True: `gated_website_search(queries='summarize the latest paywalled article on my reading list', website=<row>)`
- Search multiple gated sites for the same topic (call `gated_website_search` ONCE per site):
  1) `filter_websites(filter="host == 'medium.com' or host == 'towardsdatascience.com'")` → returns rows for both
  2) `gated_website_search(queries='LLM fine-tuning techniques', website=<medium_row>)`
  3) `gated_website_search(queries='LLM fine-tuning techniques', website=<tds_row>)`
  4) Synthesize results from both sites into a unified answer with citations from each source.
- Search one site for multiple topics (pass multiple queries in ONE call):
  1) `filter_websites(filter="host == 'medium.com'", limit=1)`
  2) `gated_website_search(queries=['AI trends', 'LLM fine-tuning', 'vector databases'], website=<row>)`
  3) Synthesize results for all topics into a unified answer.
- Summarize updates on docs.example.com:
  1) `filter_websites(filter="host == 'docs.example.com'")`
  2) If gated=False or absent: `crawl(start_url='https://docs.example.com', instructions='Find recent updates')`
- General web query (non-site specific):
  1) `search(query="how is the uk temperature in london tomorrow?", max_results=3)`

Anti‑patterns to avoid
---------------------
• Do not loop through many tools or repeat equivalent steps.
• Do not retry `gated_website_search` on the same site – call ONCE per site.
• After gated search completes for all requested sites, do NOT call additional search/extract/crawl/map.
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
        include_images_policy=False,  # WebSearcher has its own image handling
        include_images_forwarding=False,
        images_extras_block=None,
        include_parallelism=True,
        schemas=[],
        special_blocks=[],
        include_clarification_footer=True,
        include_time_footer=True,
    )

    return compose_system_prompt(spec)


def build_update_prompt(*, tools: Dict[str, Callable]) -> str:
    """Return the system prompt used by WebSearcher.update using the shared composer."""
    ask_fname = _tool_name(tools, "ask")
    request_clar_fname = _tool_name(tools, "request_clarification")

    # Build clarification block
    clarification_block = (
        textwrap.dedent(
            f"""
Clarification
-------------
• If any request is ambiguous, ask the user to disambiguate before changing data
  `{request_clar_fname}(question="There are several possible matches. Which website did you mean?")`
            """,
        ).strip()
        if request_clar_fname
        else ""
    )

    # Build dynamic tools documentation
    tools_doc = _build_update_tools_documentation(tools)

    # Build usage examples
    usage_examples_base = f"""
Tool selection
--------------
• Use `{ask_fname or 'ask'}` strictly for read-only inspection of the Websites table (e.g., to check if a host exists).
• Use `create_website` to add a new entry; use `update_website` to modify; use `delete_website` to remove.
• Do not try to browse the web from `update`; web research belongs in `ask`.
• When the user describes target sites semantically (e.g., 'ML news subscriptions'), first call `{ask_fname or 'ask'}` to identify candidates.
• When the user specifies exact columns (e.g., host or gated), first call `{ask_fname or 'ask'}` with `filter_websites(filter=...)` to confirm matches before mutating.

General Rules
-------------
• Treat `host` as the natural unique key for a website entry.
• **Naming**: Use a human-friendly `name` (e.g., 'Medium', 'GitHub', 'Financial Times'), NOT the host address.
  Good: name='HealthInvestor', host='healthinvestor.co.uk'
  Bad: name='healthinvestor.co.uk', host='healthinvestor.co.uk'
• After any mutation (create/update/delete), verify results using `{ask_fname or 'ask'}`.
• Prefer minimal, targeted tool calls; handle multiple entries comprehensively when requested.

Security & Data Hygiene
------------------------
• Never include raw credential values in messages. Only reference `credentials` by their integer `secret_id`s.
• When creating a website entry, pass `credentials=[int, ...]` only; do not attempt to resolve secret values.
• Prefer `actor_entrypoint` ids when bespoke behaviour is available; otherwise the system default will be used at runtime.

Examples
--------
• Create a gated site with credentials and verify:
  1) `create_website(name='Medium', host='medium.com', gated=True, subscribed=True, credentials=[101, 102], notes='Tech journalism and tutorials')`
  2) `{ask_fname or 'ask'}(text='List gated websites')` → should call `filter_websites(filter="gated == True")`
• Find relevant sites by notes then delete one:
  1) `{ask_fname or 'ask'}(text='Which websites are for ML news subscriptions?')` → should call `search_websites(notes='ML news subscription')`
  2) `delete_website(host='example.com')`
• Bulk creation from a list in one turn (handle ALL entries):
  - `create_website(name='arXiv', host='arxiv.org', gated=False, subscribed=False, notes='Academic preprints')`
  - `create_website(name='Financial Times', host='ft.com', gated=True, subscribed=True, credentials=[205, 206], notes='Finance and markets')`
  Then verify via `{ask_fname or 'ask'}` using `filter_websites(filter="gated == True")`.

Anti‑patterns to avoid
---------------------
• Never call `gated_website_search` from `update` (that is a browsing action in `ask`).
• Do not call `search`/`extract`/`crawl`/`map` from `update`.
• Repeating the exact same tool call with the same arguments as a means to 'make sure it has completed' – just call `{ask_fname or 'ask'}` to verify.
    """
    usage_examples = textwrap.dedent(usage_examples_base).strip()
    if clarification_block:
        usage_examples = f"{usage_examples}\n{clarification_block}"

    # Compose using standardized composer
    spec = PromptSpec(
        manager="WebSearcher",
        method="update",
        tools=tools,
        role_line="You are an assistant that **manages the WebSearcher configuration** (Websites catalog).",
        global_directives=[
            "Create, update, and delete entries in the Websites table, and use `ask` to inspect/verify.",
            "Do not answer general web research questions here; use `ask` for read-only inspection when needed.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the request and choose the best approach yourself.",
            f"Important: `{ask_fname or 'ask'}` is read‑only and must only be used to locate/inspect websites that already exist.",
        ],
        include_read_only_guard=False,
        positioning_lines=[tools_doc],
        counts_entity_plural=None,
        counts_value=None,
        columns_payload=None,
        columns_heading="columns",
        include_tools_block=True,
        usage_examples=usage_examples,
        clarification_examples_block=clarification_block or None,
        include_images_policy=False,  # WebSearcher doesn't use images policy
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

    preamble = f"On this turn you are simulating the '{method}' method."
    if method.lower() == "ask":
        behaviour = (
            "Please always answer the question with an imaginary but plausible "
            "response about the web research findings. Do NOT ask for "
            "clarification or describe your process. Provide a concise answer "
            "with brief source-like references (titles or URLs) as if you had searched."
        )
    else:
        behaviour = (
            "Provide a final response as though the requested operation has "
            "already completed (past tense)."
        )

    parts: list[str] = [preamble, behaviour, "", f"The user input is:\n{user_request}"]
    if parent_chat_context:
        parts.append(
            f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}",
        )

    return "\n".join(parts)
