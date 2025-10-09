from typing import Dict, Callable
from ..common.prompt_helpers import clarification_guidance, now_utc_str


def _now() -> str:
    return now_utc_str()


def build_ask_prompt(*, tools: Dict[str, Callable]) -> str:
    """Return the system prompt used by WebSearcher.ask formatted as sections."""
    have_search = "search" in tools
    have_extract = "extract" in tools
    have_crawl = "crawl" in tools
    have_map = "map" in tools

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

    # General rules and guidance
    lines += [
        "",
        "General Rules and Guidance",
        "--------------------------",
        "- Keep queries concise; if complex, split into smaller, focused searches.",
        "- Prefer a small, high-quality set of sources; cite them in the answer.",
        "- Only fetch page content when you need details beyond snippets.",
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
    # Current time (for reproducibility and deterministic caching in tests)
    lines += ["", f"Current UTC time is {_now()}."]

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
