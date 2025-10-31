"""
Shared assertion helpers for tests
==================================

Contains common functions for formatting assertion error messages
with detailed context including reasoning steps from LLM tool usage,
as well as reusable prompt formatting assertions for system messages.
"""

import json
import re
from typing import Any, List, Dict, Optional


def format_reasoning_steps(reasoning: List[Dict[str, Any]]) -> str:
    """Format reasoning steps from LLM tool use loops for better readability."""
    if not reasoning:
        return "No reasoning steps available"

    # Pretty print the reasoning steps, handling nested content fields
    def format_json_content(msg):
        if "content" in msg and msg["content"]:
            try:
                msg["content"] = json.loads(msg["content"])
            except (json.JSONDecodeError, TypeError):
                pass
        return msg

    formatted_reasoning = [format_json_content(msg) for msg in reasoning]
    formatted_reasoning = json.dumps(formatted_reasoning, indent=4)
    formatted_reasoning = formatted_reasoning.replace("\\n", "\n")

    return formatted_reasoning


def assertion_failed(
    expected: Any,
    actual: Any,
    reasoning: List[Dict[str, Any]],
    description: str = "",
    context_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a detailed error message for assertion failures with LLM reasoning.

    Args:
        expected: The expected value
        actual: The actual value received
        reasoning: List of reasoning steps from LLM tool use
        description: Optional description of the assertion
        context_data: Optional additional context data to include (e.g., tasks, messages)

    Returns:
        Formatted error message string
    """
    context_str = ""
    if context_data:
        for label, data in context_data.items():
            context_str += f"\n{label}:\n{json.dumps(data, indent=4)}\n"

    formatted_reasoning = format_reasoning_steps(reasoning)

    return (
        f"\n{description}\n"
        f"Expected:\n{expected}\n"
        f"Got:\n{actual}\n"
        f"{context_str}"
        f"Reasoning:\n{formatted_reasoning}\n"
    )


# ---------------------------------------------------------------------------
# System prompt assertion helpers (shared across system message tests)
# ---------------------------------------------------------------------------


def extract_tools_dict(prompt: str) -> dict:
    """Parse the tools JSON block from a standardized system prompt.

    Looks for the "Tools (name → argspec):" header, then captures the first
    well-formed JSON object that follows it by matching braces depth.
    """
    idx = prompt.find("Tools (name")
    assert idx != -1, "Missing tools block header"
    json_start = prompt.find("{", idx)
    assert json_start != -1, "Missing tools JSON start"
    depth = 0
    end = None
    for i in range(json_start, len(prompt)):
        c = prompt[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end is not None, "Unclosed tools JSON block"
    return json.loads(prompt[json_start:end])


def assert_in_order(prompt: str, markers: List[str]) -> None:
    """Assert each marker appears in `prompt` in the given sequence.

    Raises an AssertionError if any marker is missing or out of order.
    """
    pos = -1
    missing: list[str] = []
    for m in markers:
        i = prompt.find(m)
        if i == -1:
            missing.append(m)
            continue
        assert i > pos, f"Marker out of order: {m!r}\n\nFull system prompt:\n\n{prompt}"
        pos = i
    assert (
        not missing
    ), f"Missing markers (order check): {missing}\n\nFull system prompt:\n\n{prompt}"


def assert_section_spacing(prompt: str) -> None:
    """Assert that dashed underline headers are preceded by a blank line."""
    lines = prompt.splitlines()
    errors: list[str] = []
    for idx in range(len(lines) - 1):
        line = lines[idx]
        next_line = lines[idx + 1]
        if re.fullmatch(r"-+", next_line.strip()):
            if idx == 0 or lines[idx - 1].strip() != "":
                errors.append(f"Missing blank line before section header: '{line}'")
    assert not errors, "\n".join(errors) + f"\n\nFull system prompt:\n\n{prompt}"


def assert_selected_headers_have_blank_line(prompt: str, titles: List[str]) -> None:
    """Assert that provided section titles are preceded by a blank line."""
    lines = prompt.splitlines()
    missing: list[str] = []
    for i, line in enumerate(lines):
        title = line.strip()
        if title in titles:
            if i == 0 or lines[i - 1].strip() != "":
                missing.append(title)
    assert (
        not missing
    ), f"Missing blank line before: {missing}\n\nFull system prompt:\n\n{prompt}"


def assert_time_footer(prompt: str, prefix: str) -> None:
    """Assert the final non-empty line matches the standardized time footer.

    `prefix` is the literal prefix used by the builder (e.g.,
    "Current UTC time is " or "Current UTC time: ").
    """
    non_empty_lines = [ln for ln in prompt.splitlines() if ln.strip()]
    assert non_empty_lines, (
        "Prompt should not be empty\n\nFull system prompt:\n\n" + prompt
    )
    last = non_empty_lines[-1]
    pattern = re.compile(
        rf"{re.escape(prefix)}\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}} UTC\.",
    )
    assert pattern.fullmatch(
        last,
    ), f"Unexpected last line: {last!r}\n\nFull system prompt:\n\n{prompt}"
