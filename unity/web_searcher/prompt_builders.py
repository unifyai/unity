from typing import Dict, Callable
from ..common.prompt_helpers import clarification_guidance, now
from ..common.read_only_ask_guard import read_only_ask_mutation_exit_block


def build_ask_prompt(*, tools: Dict[str, Callable]) -> str:
    """Return the system prompt used by WebSearcher.ask formatted as sections."""
    have_search = "search" in tools
    have_extract = "extract" in tools
    have_crawl = "crawl" in tools
    have_map = "map" in tools
    have_filter_websites = "_filter_websites" in tools
    have_search_websites = "_search_websites" in tools
    have_search_gated = "_search_gated_website" in tools

    lines: list[str] = []
    # Purpose
    lines += [
        "Purpose",
        "-------",
        "- You are a web research assistant.",
        "- Use the available tools to answer the user's question.",
        "- Produce concise, factual answers with optional inline citations (title or URL).",
    ]

    # Tools available
    lines += [
        "",
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
            "- _filter_websites: list websites matching a boolean filter over columns.",
            "  • Parameters: filter, offset, limit",
            "  • Examples:",
            '    - _filter_websites(filter="gated == True")',
            "    - _filter_websites(filter=\"host == 'medium.com'\", limit=1)",
        ]
    if have_search_websites:
        lines += [
            "- _search_websites: semantic search over the Websites catalog using notes similarity.",
            "  • Parameters: notes, k",
            "  • Example:",
            '    - _search_websites(notes="subscription sources for ML news", k=5)',
        ]
    if have_search_gated:
        lines += [
            "- _search_gated_website: search a specific website via the Actor (handles login if gated).",
            "  • Parameters: query, website",
            "  • Examples:",
            '    - _search_gated_website(query="latest AI trends", website={"host": "medium.com"})',
        ]

    # General rules and guidance
    lines += [
        "",
        "General Rules and Guidance",
        "--------------------------",
        "- Keep queries concise; if complex, split into smaller, focused searches.",
        "- Prefer a small, high-quality set of sources; cite them in the answer.",
        "- Only fetch page content when you need details beyond snippets.",
        "- Do not claim inability to log into personal accounts. When a Website entry exists and credentials are available, the Actor can attempt sign-in securely. If credentials are missing or login fails, proceed with public content and clearly state assumptions.",
        "- If the request mentions a specific website (host like 'medium.com' or a human-friendly name like 'Medium'), first consult the Websites catalog:",
        "  • Use `_filter_websites` for exact host/name filters; use `_search_websites` when only thematic notes are given.",
        "  • If a row exists and `gated=True`, use `_search_gated_website(query=..., website=...)` to browse with login.",
        "  • Otherwise, use general tools (`search`, `extract`, `crawl`, `map`).",
    ]

    # Website-aware routing guidance
    lines += [
        "",
        "Website-aware Routing",
        "----------------------",
        "- Use `_search_websites` to find relevant Website entries by notes similarity (catalog lookup only; does not browse).",
        "- Use `_filter_websites` for exact/boolean matches over columns (including host like 'medium.com' or name like 'Medium').",
        "- When answering a question that targets a specific site:",
        "  1) Look up the site using `_filter_websites` or `_search_websites`.",
        "  2) If the site exists and `gated=True`, use `_search_gated_website(query=..., website=...)` to login with saved credentials and browse.",
        "  3) If not gated or no matching Website entry exists, use general tools (`search`, then optionally `extract`/`crawl`/`map`).",
        "- Do NOT use `_search_websites` to read web content; it only searches the Websites catalog.",
    ]

    # Concrete examples for routing
    lines += [
        "",
        "Examples",
        "--------",
        "- Login to my GitHub and summarize my profile:",
        "  1) `_filter_websites(filter=\"host == 'github.com' or name == 'GitHub'\", limit=1)`",
        "  2) If found and gated=True: `_search_gated_website(query='summarize my GitHub profile', website=<row>)`",
        "  3) Else: use `crawl`/`extract` as appropriate.",
        "- Access my Towards Data Science subscription article and summarize:",
        "  1) `_filter_websites(filter=\"host == 'towardsdatascience.com' or name == 'Towards Data Science'\", limit=1)`",
        "  2) If found and gated=True: `_search_gated_website(query='summarize the latest paywalled article on my reading list', website=<row>)`",
        "- Summarize updates on docs.example.com:",
        "  1) `_filter_websites(filter=\"host == 'docs.example.com'\")`",
        "  2) If gated=False or absent: `crawl(start_url='https://docs.example.com', instructions='Find recent updates')`",
        "- General web query (non-site specific):",
        '  1) `search(query="how is the uk temperature in london tomorrow?", max_results=3)`',
    ]

    # Decision policy and when to stop
    lines += [
        "",
        "Decision Policy and When to Stop",
        "---------------------------------",
        "1. Run a targeted search and read the snippets.",
        "2. If snippets suffice, STOP and write the answer (no more tools).",
        "3. Otherwise, extract at most one highly relevant URL.",
        "4. If still insufficient, do one more targeted step (search OR extract), then STOP and answer.",
        "5. Do not loop through many tools or repeat equivalent steps.",
    ]

    lines += [
        "",
        "Answer Requirements",
        "-------------------",
        "- Be precise and concise; cite sources inline (title or URL).",
        "- If evidence is insufficient, do one targeted step; otherwise answer with best-supported facts.",
        "- After you write the final answer, do not call further tools.",
    ]

    # Clarification guidance (conditionally references request_clarification when available)
    lines += ["", clarification_guidance(tools)]
    # Early exit policy for mutation-intent requests reaching ask()
    lines += ["", read_only_ask_mutation_exit_block()]
    # Current time (for reproducibility and deterministic caching in tests)
    lines += ["", f"Current UTC time is {now()}."]

    return "\n".join(lines)


def build_update_prompt(*, tools: Dict[str, Callable]) -> str:
    """Return the system prompt used by WebSearcher.update formatted as sections."""
    have_create = "_create_website" in tools
    have_update = "_update_website" in tools
    have_delete = "_delete_website" in tools
    have_ask = "ask" in tools

    lines: list[str] = []
    # Purpose
    lines += [
        "Purpose",
        "-------",
        "- You manage mutations to the WebSearcher configuration.",
        "- Specifically, you create and delete entries in the Websites table, and use `ask` to inspect/verify.",
        "- Do not answer general web research questions here; use `ask` for read-only inspection when needed.",
    ]

    # Tools available (dynamic, like ask)
    lines += [
        "",
        "Tools Available",
        "---------------",
    ]
    if have_create:
        lines += [
            "- _create_website: create a new Website row (unique by host).",
            "  • Parameters: name, host, gated, subscribed, credentials, actor_entrypoint, notes",
            "  • Examples:",
            "    - _create_website(name='Medium', host='medium.com', gated=True, subscribed=True, credentials=[101, 102], notes='Tech journalism and tutorials')",
            "    - _create_website(name='arXiv', host='arxiv.org', gated=False, subscribed=False, notes='Academic preprints')",
        ]
    if have_update:
        lines += [
            "- _update_website: update fields of an existing Website.",
            "  • Identify by one of: website_id, match_host, match_name",
            "  • Updatable fields: name, host, gated, subscribed, credentials, actor_entrypoint, notes",
            "  • Examples:",
            "    - _update_website(match_host='medium.com', subscribed=False)",
            "    - _update_website(website_id=3, name='NYTimes', host='nytimes.com')",
        ]
    if have_delete:
        lines += [
            "- _delete_website: delete a Website row by host or website_id (exact match).",
            "  • Parameters: name, host, website_id",
            "  • Examples:",
            "    - _delete_website(name='Financial Times')",
            "    - _delete_website(host='example.com')",
            "    - _delete_website(website_id=42)",
        ]
    if have_ask:
        lines += [
            "- ask: read-only inspection helper (calls catalog tools like _filter_websites/_search_websites).",
            "  • Parameters: text",
            "  • Examples:",
            "    - ask(text='List gated websites')  → should call _filter_websites(filter=\"gated == True\")",
            "    - ask(text='Which websites match ML news subscriptions?')  → should call _search_websites(notes='ML news subscription')",
        ]

    # General rules
    lines += [
        "",
        "General Rules",
        "-------------",
        "- Treat `host` as the natural unique key for a website entry.",
        "- After any mutation (create/delete), verify results using `ask` (e.g., `_filter_websites` or `_search_websites`).",
        "- Prefer minimal, targeted tool calls; handle multiple entries comprehensively when requested.",
    ]

    # Ask vs Mutation guidance
    lines += [
        "",
        "Ask vs Mutations",
        "-----------------",
        "- Use `ask` strictly for read-only inspection of the Websites table (e.g., to check if a host exists).",
        "- Use `_create_website` to add a new entry; use `_delete_website` to remove an entry.",
        "- Do not try to browse the web from `update`; web research belongs in `ask`.",
    ]

    # Tool selection (aligned with ask routing, but for mutations)
    lines += [
        "",
        "Tool selection (read carefully)",
        "--------------------------------",
        "- When the user describes target sites semantically (e.g., 'ML news subscriptions'), first call `ask` to identify candidates using `_search_websites(notes=...)`.",
        "- When the user specifies exact columns (e.g., host or gated), first call `ask` with `_filter_websites(filter=...)` to confirm matches before mutating.",
        "- Never call `_search_gated_website` from `update` (that is a browsing action in `ask`).",
        "- Do not call `search`/`extract`/`crawl`/`map` from `update`.",
    ]

    # Security and data hygiene
    lines += [
        "",
        "Security & Data Hygiene",
        "------------------------",
        "- Never include raw credential values in messages. Only reference `credentials` by their integer `secret_id`s.",
        "- When creating a website entry, pass `credentials=[int, ...]` only; do not attempt to resolve secret values.",
        "- Prefer `actor_entrypoint` ids when bespoke behaviour is available; otherwise the system default will be used at runtime.",
    ]

    # Examples
    lines += [
        "",
        "Examples",
        "--------",
        "- Create a gated site with credentials and verify:",
        "  1) _create_website(host='medium.com', gated=True, subscribed=True, credentials=[101, 102], notes='Tech journalism and tutorials')",
        "  2) ask(text='List gated websites')  → should call _filter_websites(filter=\"gated == True\")",
        "- Find relevant sites by notes then delete one:",
        "  1) ask(text='Which websites are for ML news subscriptions?') → should call _search_websites(notes='ML news subscription')",
        "  2) _delete_website(host='example.com')",
        "- Bulk creation from a list in one turn (handle ALL entries):",
        "  • _create_website(host='arxiv.org', gated=False, subscribed=False, notes='Academic preprints')",
        "  • _create_website(host='ft.com', gated=True, subscribed=True, credentials=[205, 206], notes='Finance and markets')",
        '  Then verify via ask using _filter_websites(filter="gated == True").',
    ]

    # Clarification guidance (conditionally references request_clarification when available)
    lines += ["", clarification_guidance(tools)]

    # Time for deterministic caching in tests
    lines += ["", f"Current UTC time is {now()}."]

    return "\n".join(lines)


def build_simulated_method_prompt(
    method: str,
    user_request: str,
    parent_chat_context: list[dict] | None = None,
) -> str:
    """Return instruction prompt for the simulated WebSearcher."""
    import json

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
