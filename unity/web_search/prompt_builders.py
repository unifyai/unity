from typing import Dict, Callable


def build_ask_prompt(*, tools: Dict[str, Callable]) -> str:
    """Return the system prompt for WebSearch.ask using provided tools.

    Keeps guidance minimal; tool loop handles orchestration.
    """
    tool_names = ", ".join(sorted(tools.keys()))
    return (
        "You are a helpful research assistant for web research.\n\n"
        "Available tools: "
        f"{tool_names}.\n"
        "- Use `search` to find sources; optionally read pages with `extract`.\n"
        "- Use `crawl` for site traversal and `map` for structured mapping.\n"
        "Answer concisely, cite sources when appropriate, and avoid speculation."
    )
