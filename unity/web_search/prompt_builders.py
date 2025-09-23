from typing import Dict, Callable


def build_ask_prompt(*, tools: Dict[str, Callable]) -> str:
    """Return the system prompt used by WebSearch.ask.

    Includes concise, actionable guidance for using enabled tools and
    parameters. The tool-use loop handles orchestration; keep outputs
    factual, concise, and cite sources inline (title or URL).
    """
    have_search = "search" in tools
    have_extract = "extract" in tools
    have_crawl = "crawl" in tools
    have_map = "map" in tools

    lines: list[str] = []
    lines.append(
        "You are a web research assistant. Use the tools to answer the user's question.",
    )
    lines.append("")
    lines.append("General guidance")
    lines.append(
        "- Keep queries concise; if complex, split into smaller, focused searches.",
    )
    lines.append(
        "- Prefer a small, high-quality set of sources; cite them in the answer.",
    )
    lines.append("- Only fetch page content when you need details beyond snippets.")

    if have_search:
        lines.append("")
        lines.append("Search (find relevant sources)")
        lines.append("- max_results: limit to the top N results (default 5) for focus.")
        lines.append(
            "- start_date/end_date: filter by published date when recency matters (YYYY-MM-DD).",
        )
        lines.append("- include_images: set only when image evidence is needed.")
        lines.append("Examples:")
        lines.append('  search(query="latest vector database trends", max_results=5)')
        lines.append(
            '  search(query="Q1 updates", start_date="2025-01-01", end_date="2025-03-31")',
        )
        lines.append('  search(query="product logos", include_images=True)')

    if have_extract:
        lines.append("")
        lines.append("Extract (read page content)")
        lines.append(
            "- Use after search to fetch content for specific URLs (batch when possible).",
        )
        lines.append("- include_images: request images only when necessary.")
        lines.append("Examples:")
        lines.append('  extract(urls=["https://site/a", "https://site/b"]) ')
        lines.append('  extract(urls="https://site/a", include_images=True)')

    if have_crawl:
        lines.append("")
        lines.append("Crawl (site exploration)")
        lines.append("- Provide clear instructions to target what to collect.")
        lines.append(
            "- max_depth/breadth: keep small to avoid noise; limit caps total pages.",
        )
        lines.append("- include_images: enable only if images are required.")
        lines.append("Example:")
        lines.append(
            '  crawl(start_url="https://docs.example.com", instructions="Find SDK pages", max_depth=1, max_breadth=3, limit=20)',
        )

    if have_map:
        lines.append("")
        lines.append("Map (structured overview)")
        lines.append("- Use for high-level mapping; guide with instructions.")
        lines.append(
            "- max_depth/breadth and limit control scope; include_images only if needed.",
        )
        lines.append("Example:")
        lines.append(
            '  map(query="AI evaluation frameworks", instructions="Group by approach", max_depth=1, max_breadth=3, limit=30)',
        )

    lines.append("")
    lines.append("Answer requirements")
    lines.append("- Be precise and concise; cite sources inline (title or URL).")
    lines.append(
        "- If evidence is insufficient, run another targeted search or extract.",
    )

    return "\n".join(lines)
